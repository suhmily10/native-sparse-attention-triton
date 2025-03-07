# Copyright 2025 Xunhao Lai & Jianqiao Lu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import triton
import math
import time
from flash_attn import flash_attn_varlen_func
from native_sparse_attention.ops.triton.flash_attention import flash_attention_varlen
from native_sparse_attention.ops.torch.compress_key_value import avgpool_compress
from native_sparse_attention.module.native_sparse_attention import NativeSparseAttentionNoRoPE, NativeSparseAttentionQKV
from native_sparse_attention.module.rope import RopeConfig

def bench(func, warmup_steps=3, test_steps=10):
    for i in range(warmup_steps):
        func()
    torch.cuda.synchronize()
    st = time.time()
    for i in range(test_steps):
        func()
    torch.cuda.synchronize()
    ed = time.time()
    torch.cuda.empty_cache()
    return (ed - st) / test_steps * 1000  # Convert to ms

def benchmark_forward_backward():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 6)],
            line_arg="provider",
            line_vals=[
                "flash-forward",
                "triton-flash-forward",
                "nsa-forward",
                "flash-backward",
                "triton-flash-backward",
                "nsa-backward",
            ],
            line_names=[
                "Flash Forward",
                "Triton-Flash Forward",
                "NSA Forward",
                "Flash Backward",
                "Triton-Flash Backward",
                "NSA Backward",
            ],
            styles=[
                ("green", "-"), ("green", "--"), ("blue", "-"),
                ("red", "-"), ("red", "--"), ("purple", "-")
            ],
            ylabel="ms",
            plot_name="** forward/backward speed comparison **",
            args={"H": 32, "D": 128},
        )
    )
    def benchmark(N, H, D, provider):
        torch.manual_seed(42)
        hidden_size = H * D
        num_kv_heads = H // 4
        
        # Clear CUDA cache before creating new tensors
        torch.cuda.empty_cache()
        
        # Create inputs
        x = torch.randn((N, hidden_size), device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        sm_scale = 1 / math.sqrt(D)
        
        # For vanilla flash attention
        q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((N, num_kv_heads, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((N, num_kv_heads, D), device="cuda", dtype=torch.bfloat16)
        
        # For NSA
        kernel_size = 32
        kernel_stride = 16
        block_size = 64
        topk = 16
        init_blocks = 8
        local_blocks = 4
        window_size = 256

        # Initialize NSA model (QKV version for direct comparison)
        nsa_qkv = NativeSparseAttentionQKV(
            hidden_size=hidden_size,
            num_q_heads=H,
            num_kv_heads=num_kv_heads,
            head_dim=D,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            window_size=window_size,
        ).cuda().to(torch.bfloat16)
        
        # Also keep the original NSA for complete comparison
        nsa = NativeSparseAttentionNoRoPE(
            hidden_size=hidden_size,
            num_q_heads=H,
            num_kv_heads=num_kv_heads,
            head_dim=D,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            window_size=window_size,
        ).cuda().to(torch.bfloat16)
        
        # Pre-computation for NSA
        x_clone = x.detach().clone().requires_grad_(True)
        
        # For backward pass, create gradients
        grad_output = torch.randn_like(x)
        grad_output_attn = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        
        quantiles = [0.5, 0.2, 0.8]
        
        try:
            if provider == "flash-forward":
                ms = bench(
                    lambda: flash_attn_varlen_func(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        cu_seqlens,
                        cu_seqlens,
                        N,
                        N,
                        dropout_p=0.0,
                        causal=True,
                        softmax_scale=sm_scale,
                    )
                )
                min_ms = ms
                max_ms = ms
                
            elif provider == "triton-flash-forward":
                ms = bench(
                    lambda: flash_attention_varlen(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        cu_seqlens,
                        cu_seqlens,
                        N,
                        N,
                        True,
                        sm_scale
                    )
                )
                min_ms = ms
                max_ms = ms
                
            elif provider == "nsa-forward":
                # Use the QKV version for direct comparison with flash attention
                ms = bench(
                    lambda: nsa_qkv(q.clone(), k.clone(), v.clone(), cu_seqlens)
                )
                min_ms = ms
                max_ms = ms
                
            elif provider == "flash-backward":
                # Prepare tensors outside the timing loop
                q_back = q.clone().requires_grad_(True)
                k_back = k.clone().requires_grad_(True)
                v_back = v.clone().requires_grad_(True)
                
                # First run to warm up and create graph
                o = flash_attn_varlen_func(
                    q_back, k_back, v_back, cu_seqlens, cu_seqlens, N, N,
                    dropout_p=0.0, causal=True, softmax_scale=sm_scale
                )
                loss = (o * grad_output_attn).sum()
                
                # Only time the backward pass
                ms = bench(
                    lambda: loss.backward(retain_graph=True)
                )
                min_ms = ms
                max_ms = ms
                
            elif provider == "triton-flash-backward":
                # Prepare tensors outside the timing loop
                q_back = q.clone().requires_grad_(True)
                k_back = k.clone().requires_grad_(True)
                v_back = v.clone().requires_grad_(True)
                
                # First run to warm up and create graph
                o = flash_attention_varlen(
                    q_back, k_back, v_back, cu_seqlens, cu_seqlens, N, N, True, sm_scale
                )
                loss = (o * grad_output_attn).sum()
                
                # Only time the backward pass
                ms = bench(
                    lambda: loss.backward(retain_graph=True)
                )
                min_ms = ms
                max_ms = ms
                
            elif provider == "nsa-backward":
                # Use the QKV version for direct comparison with flash attention
                q_back = q.clone().requires_grad_(True)
                k_back = k.clone().requires_grad_(True)
                v_back = v.clone().requires_grad_(True)
                
                # First run to warm up and create graph
                o = nsa_qkv(q_back, k_back, v_back, cu_seqlens)
                loss = (o * grad_output).sum()
                
                # Only time the backward pass
                ms = bench(
                    lambda: loss.backward(retain_graph=True)
                )
                min_ms = ms
                max_ms = ms
        finally:
            # Clean up all tensors
            del x, cu_seqlens, q, k, v, x_clone, grad_output, grad_output_attn, nsa, nsa_qkv
            # Force garbage collection before emptying cache
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)

def benchmark_batch_sizes():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["B"],  # Test different batch sizes
            x_vals=[1, 2, 4, 8, 16],
            line_arg="provider",
            line_vals=[
                "flash-forward",
                "triton-flash-forward",
                "nsa-forward",
                "flash-backward",
                "triton-flash-backward",
                "nsa-backward",
            ],
            line_names=[
                "Flash Forward",
                "Triton-Flash Forward", 
                "NSA Forward",
                "Flash Backward",
                "Triton-Flash Backward",
                "NSA Backward",
            ],
            styles=[
                ("green", "-"), ("green", "--"), ("blue", "-"),
                ("red", "-"), ("red", "--"), ("purple", "-")
            ],
            ylabel="ms",
            plot_name="** batch size performance comparison 16384 **",
            args={"N": 16384, "H": 32, "D": 128},  # Fixed sequence length
        )
    )
    def benchmark(B, N, H, D, provider):
        torch.manual_seed(42)
        hidden_size = H * D
        num_kv_heads = H // 4
        
        # Clear CUDA cache before creating new tensors
        torch.cuda.empty_cache()
        print("start",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
        # Create batched inputs
        x = torch.randn((B, N, hidden_size), device="cuda", dtype=torch.bfloat16)
        
        # Create cumulative sequence lengths for batched input
        cu_seqlens = torch.zeros(B+1, device="cuda", dtype=torch.int32)
        for i in range(B):
            cu_seqlens[i+1] = cu_seqlens[i] + N
        
        total_tokens = B * N
        sm_scale = 1 / math.sqrt(D)
        
        print("before QKV",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
        
        # For vanilla flash attention
        q = torch.randn((total_tokens, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((total_tokens, num_kv_heads, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((total_tokens, num_kv_heads, D), device="cuda", dtype=torch.bfloat16)
        
        print("after QKV", torch.cuda.memory_reserved()//1024**3, "GB")
        
        # For NSA
        kernel_size = 32
        kernel_stride = 16
        block_size = 64
        topk = 16
        init_blocks = 8
        local_blocks = 4
        window_size = 512

        # Initialize NSA model (QKV version for direct comparison)
        nsa_qkv = NativeSparseAttentionQKV(
            hidden_size=hidden_size,
            num_q_heads=H,
            num_kv_heads=num_kv_heads,
            head_dim=D,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            window_size=window_size,
        ).cuda().to(torch.bfloat16)
        
        # Also keep the original NSA
        nsa = NativeSparseAttentionNoRoPE(
            hidden_size=hidden_size,
            num_q_heads=H,
            num_kv_heads=num_kv_heads,
            head_dim=D,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            window_size=window_size,
        ).cuda().to(torch.bfloat16)
        
        # Pre-computation for NSA
        x_flat = x.reshape(-1, hidden_size)
        x_clone = x_flat.detach().clone().requires_grad_(True)
        
        # For backward pass, create gradients
        grad_output = torch.randn_like(x_flat)
        grad_output_attn = torch.randn((total_tokens, H, D), device="cuda", dtype=torch.bfloat16)
        
        print("before test",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
        
        quantiles = [0.5, 0.2, 0.8]
        
        try:
            if provider == "flash-forward":
                flash_attn_varlen_func(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    cu_seqlens,
                    cu_seqlens,
                    total_tokens,
                    total_tokens,
                    dropout_p=0.0,
                    causal=True,
                    softmax_scale=sm_scale,
                )
                print("after flash-forward",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
                ms = bench(
                    lambda: flash_attn_varlen_func(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        cu_seqlens,
                        cu_seqlens,
                        total_tokens,
                        total_tokens,
                        dropout_p=0.0,
                        causal=True,
                        softmax_scale=sm_scale,
                    )
                )
                min_ms = ms
                max_ms = ms
                print("after benchmark flash-forward",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
                # Clean up immediately after benchmark
                torch.cuda.empty_cache()
                
            elif provider == "triton-flash-forward":
                ms = bench(
                    lambda: flash_attention_varlen(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        cu_seqlens,
                        cu_seqlens,
                        total_tokens,
                        total_tokens,
                        True,
                        sm_scale
                    )
                )
                min_ms = ms
                max_ms = ms
                print("after benchmark triton-flash-forward",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
                # Clean up immediately after benchmark
                torch.cuda.empty_cache()
                
            elif provider == "nsa-forward":
                # Use the QKV version for direct comparison with flash attention
                ms = bench(
                    lambda: nsa_qkv(q.clone(), k.clone(), v.clone(), cu_seqlens)
                )
                min_ms = ms
                max_ms = ms
                # Clean up immediately after benchmark
                torch.cuda.empty_cache()
                
            elif provider == "flash-backward":
                # Prepare tensors outside the timing loop
                q_back = q.clone().requires_grad_(True)
                k_back = k.clone().requires_grad_(True)
                v_back = v.clone().requires_grad_(True)
                
                # First run to warm up and create graph
                o = flash_attn_varlen_func(
                    q_back, k_back, v_back, cu_seqlens, cu_seqlens, total_tokens, total_tokens,
                    dropout_p=0.0, causal=True, softmax_scale=sm_scale
                )
                loss = (o * grad_output_attn).sum()
                
                # Only time the backward pass
                ms = bench(
                    lambda: loss.backward(retain_graph=True)
                )
                min_ms = ms
                max_ms = ms
                print("after benchmark flash-backward",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
                # Clean up outside the timing loop
                del q_back, k_back, v_back, o, loss
                torch.cuda.empty_cache()
                
            elif provider == "triton-flash-backward":
                # Prepare tensors outside the timing loop
                q_back = q.clone().requires_grad_(True)
                k_back = k.clone().requires_grad_(True)
                v_back = v.clone().requires_grad_(True)
                
                # First run to warm up and create graph
                o = flash_attention_varlen(
                    q_back, k_back, v_back, cu_seqlens, cu_seqlens, total_tokens, total_tokens, True, sm_scale
                )
                loss = (o * grad_output_attn).sum()
                
                # Only time the backward pass
                ms = bench(
                    lambda: loss.backward(retain_graph=True)
                )
                min_ms = ms
                max_ms = ms
                print("after benchmark triton-flash-backward",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
                # Clean up outside the timing loop
                del q_back, k_back, v_back, o, loss
                torch.cuda.empty_cache()
                
            elif provider == "nsa-backward":
                # Use the QKV version for direct comparison with flash attention
                q_back = q.clone().requires_grad_(True)
                k_back = k.clone().requires_grad_(True)
                v_back = v.clone().requires_grad_(True)
                
                # First run to warm up and create graph
                o = nsa_qkv(q_back, k_back, v_back, cu_seqlens)
                loss = (o * grad_output).sum()
                
                # Only time the backward pass
                ms = bench(
                    lambda: loss.backward(retain_graph=True)
                )
                min_ms = ms
                max_ms = ms
                print("after benchmark nsa-backward",B, provider, torch.cuda.memory_reserved()//1024**3, "GB")
                # Clean up outside the timing loop
                del q_back, k_back, v_back, o, loss
                torch.cuda.empty_cache()
        finally:
            # Clean up all tensors
            del x, cu_seqlens, q, k, v, x_clone, grad_output, grad_output_attn, nsa, nsa_qkv, x_flat
            # Force garbage collection before emptying cache
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        return ms, min_ms, max_ms

    benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    benchmark_forward_backward()
    # benchmark_batch_sizes() 