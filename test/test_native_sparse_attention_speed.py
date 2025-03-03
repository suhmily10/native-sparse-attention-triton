# Copyright 2025 Xunhao Lai.
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
from native_sparse_attention.module.native_sparse_attention import NativeSparseAttention, NativeSparseAttentionNoRoPE
from native_sparse_attention.module.rope import RopeConfig
from native_sparse_attention.ops.triton.flash_attention import (
    _flash_attention_fwd,
    _flash_attention_bwd
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward
)


def generate_topk_idx_example(
    seqlens: torch.Tensor, block_size: int, topk: int, num_heads: int
) -> torch.Tensor:
    """Generate topk idx example for test.

    Args:
        seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        block_size (int): key value block size
        topk (int): selected topk
        num_heads (int): number of key value heads

    Returns:
        torch.Tensor: shape [num_heads, total_seqlen, topk], topk key value block idx for each query. -1 means padding.
    """
    batch_size = seqlens.shape[0]
    num_blocks = torch.ceil(seqlens / block_size).to(torch.int32)
    topk_idx_all_heads = []
    for _ in range(num_heads):
        topk_idx = [
            torch.randn(seqlens[i], num_blocks[i], device="cuda")
            .topk(min(topk, num_blocks[i]), dim=-1)
            .indices.to(torch.int32)
            for i in range(batch_size)
        ]
        topk_idx = [
            torch.nn.functional.pad(
                topk_idx[i], (0, topk - topk_idx[i].shape[-1]), value=topk
            )
            for i in range(batch_size)
        ]
        topk_idx = torch.cat(topk_idx, dim=0)
        topk_idx = torch.sort(topk_idx, dim=1).values
        topk_idx[:, 0] = 0
        q_idx = torch.cat(
            [torch.arange(seqlens[i], device="cuda") for i in range(batch_size)], dim=0
        )
        topk_idx[topk_idx > (q_idx // block_size)[:, None]] = -1  # -1 means padding
        topk_idx_all_heads.append(topk_idx)
    topk_idx = torch.stack(topk_idx_all_heads, dim=0)
    return topk_idx


def setup_native_sparse_attention(
    hidden_size, num_q_heads, num_kv_heads, head_dim, 
    kernel_size, kernel_stride, block_size, topk, init_blocks, 
    local_blocks, window_size, use_rope=False
):
    if use_rope:
        rope_config = RopeConfig(
            dim=head_dim,
            base=10000.0,
            scale=1.0,
            factor=1.0,
            scaling_factor=None
        )
        return NativeSparseAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            window_size=window_size,
            rope_config=rope_config
        )
    else:
        return NativeSparseAttentionNoRoPE(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            window_size=window_size
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Benchmark parameters with varying batch sizes
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 6)],
            line_arg="provider",
            line_vals=["flash-b1", "flash-b2",
                       "triton-flash-b1", "triton-flash-b2",
                       "native-sparse-b1", "native-sparse-b2"],
            line_names=[
                "Flash-batch1", "Flash-batch2",
                "Triton-Flash-batch1", "Triton-Flash-batch2",
                "Native-Sparse-batch1", "Native-Sparse-batch2",
            ],
            styles=[("green", "-"), ("green", "--"),
                    ("red", "-"), ("red", "--"),
                    ("blue", "-"), ("blue", "--")],
            ylabel="ms",
            plot_name="** forward pass comparison **",
            args={"H": 32, "D": 128},
        )
    )
    def benchmark_forward(N, H, D, provider):
        # Parse provider to get method and batch size
        parts = provider.split('-')
        batch_info = parts[-1]  # Get the last part (b1, b2)
        method = '-'.join(parts[:-1])  # Join all parts except the last one
        batch_size = int(batch_info[1:])  # Extract number after 'b'
        
        # Common parameters
        hidden_size = H * D
        num_q_heads = H
        num_kv_heads = H // 8
        head_dim = D
        
        # 修正：总序列长度 = N * batch_size
        total_seqlen = N * batch_size  # 新增总序列长度计算
        
        # 修正：每个样本的序列长度固定为N
        cu_seqlens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        for i in range(1, batch_size + 1):
            cu_seqlens[i] = cu_seqlens[i-1] + N  # 每个样本固定长度N
        
        sm_scale = 1 / math.sqrt(D)
        
        # Additional parameters for native-sparse
        kernel_size = 16
        kernel_stride = 4
        block_size = 64
        topk = 8
        init_blocks = 1
        local_blocks = 2
        window_size = 256
        
        quantiles = [0.5, 0.2, 0.8]
        
        if method == "flash":
            # 修正输入形状：总token数 = batch_size * N
            q = torch.randn((batch_size * N, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            k = torch.randn((batch_size * N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            v = torch.randn((batch_size * N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attn_varlen_forward(
                    q, k, v, cu_seqlens, cu_seqlens, 
                    N,  # max_seqlen_q 修正为N
                    N,  # max_seqlen_k 修正为N
                    dropout_p=0.0,
                    causal=True,
                    softmax_scale=sm_scale,
                ),
                quantiles=quantiles,
            )
            
        elif method == "triton-flash":
            # 同样修正输入形状
            q = torch.randn((batch_size * N, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            k = torch.randn((batch_size * N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            v = torch.randn((batch_size * N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attention_fwd(
                    q, k, v, cu_seqlens, cu_seqlens, N, N, True, sm_scale  # 修正max_seqlen参数
                ),
                quantiles=quantiles,
            )
            
        elif method == "native-sparse":
            # 修正输入形状
            x = torch.randn((batch_size * N, hidden_size), device="cuda", dtype=torch.bfloat16)
            
            # Setup native sparse attention model
            model = setup_native_sparse_attention(
                hidden_size=hidden_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kernel_size=kernel_size,
                kernel_stride=kernel_stride,
                block_size=block_size,
                topk=topk,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                window_size=window_size
            ).cuda().to(torch.bfloat16)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: model(x, cu_seqlens),
                quantiles=quantiles,
            )
            
        return ms, min_ms, max_ms

    benchmark_forward.run(show_plots=True, print_data=True)
    
    # Benchmark backward pass with different batch sizes
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 6)],
            line_arg="provider",
            line_vals=["flash-b1", "flash-b2",
                       "triton-flash-b1", "triton-flash-b2",
                       "native-sparse-b1", "native-sparse-b2"],
            line_names=[
                "Flash-batch1", "Flash-batch2",
                "Triton-Flash-batch1", "Triton-Flash-batch2",
                "Native-Sparse-batch1", "Native-Sparse-batch2",
            ],
            styles=[("green", "-"), ("green", "--"),
                    ("red", "-"), ("red", "--"),
                    ("blue", "-"), ("blue", "--")],
            ylabel="ms",
            plot_name="** backward pass comparison **",
            args={"H": 32, "D": 128},
        )
    )
    def benchmark_backward(N, H, D, provider):
        # Parse provider to get method and batch size
        parts = provider.split('-')
        batch_info = parts[-1]  # Get the last part (b1, b2)
        method = '-'.join(parts[:-1])  # Join all parts except the last one
        batch_size = int(batch_info[1:])  # Extract number after 'b'
        
        # Common parameters
        hidden_size = H * D
        num_q_heads = H
        num_kv_heads = H // 8
        head_dim = D
        
        # Additional parameters for native-sparse
        kernel_size = 16
        kernel_stride = 4
        block_size = 64
        topk = 8
        init_blocks = 1
        local_blocks = 2
        window_size = 256
        
        # Setup cu_seqlens for different batch sizes
        seqlen_per_batch = N // batch_size
        cu_seqlens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        for i in range(1, batch_size + 1):
            cu_seqlens[i] = cu_seqlens[i-1] + seqlen_per_batch
        
        sm_scale = 1 / math.sqrt(D)
        
        quantiles = [0.5, 0.2, 0.8]
        
        if method == "flash":
            # Create inputs with proper shapes for flash attention
            q = torch.randn((batch_size * seqlen_per_batch, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn((batch_size * seqlen_per_batch, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn((batch_size * seqlen_per_batch, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            
            # Forward pass to get outputs needed for backward
            o = _flash_attn_varlen_forward(
                q, k, v, cu_seqlens, cu_seqlens, seqlen_per_batch, seqlen_per_batch, dropout_p=0.0, causal=True, softmax_scale=sm_scale
            )[0]
            
            do = torch.randn_like(o)
            lse = torch.randn((num_q_heads, batch_size * seqlen_per_batch), device="cuda", dtype=torch.float32)
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attn_varlen_backward(
                    do, q, k, v, o, lse, dq, dk, dv, cu_seqlens, cu_seqlens, 
                    seqlen_per_batch, seqlen_per_batch, dropout_p=0.0, causal=True, softmax_scale=sm_scale,
                    window_size_left=-1, window_size_right=-1, softcap=0.0, 
                    alibi_slopes=None, deterministic=False, zero_tensors=False
                ),
                quantiles=quantiles,
            )
            
        elif method == "triton-flash":
            # Create inputs with proper shapes for triton-flash
            q = torch.randn((batch_size * seqlen_per_batch, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn((batch_size * seqlen_per_batch, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn((batch_size * seqlen_per_batch, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            
            # Forward pass to get outputs needed for backward
            o, lse = _flash_attention_fwd(q, k, v, cu_seqlens, cu_seqlens, seqlen_per_batch, seqlen_per_batch, True, sm_scale)
            do = torch.randn_like(o)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attention_bwd(o, do, lse, q, k, v, cu_seqlens, cu_seqlens, seqlen_per_batch, seqlen_per_batch, True, sm_scale),
                quantiles=quantiles,
            )
            
        elif method == "native-sparse":
            # Create input for native sparse attention
            x = torch.randn((batch_size * seqlen_per_batch, hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            
            # Gradients for backward
            grad_out = torch.randn((batch_size * seqlen_per_batch, hidden_size), device="cuda", dtype=torch.bfloat16)
            
            # Setup native sparse attention model
            model = setup_native_sparse_attention(
                hidden_size=hidden_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kernel_size=kernel_size,
                kernel_stride=kernel_stride,
                block_size=block_size,
                topk=topk,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                window_size=window_size
            ).cuda().to(torch.bfloat16)
            
            # Forward pass
            output = model(x, cu_seqlens)
            
            # Benchmark backward pass
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.autograd.backward(
                    output, grad_out, retain_graph=True
                ),
                quantiles=quantiles,
            )
            
        return ms, min_ms, max_ms

    benchmark_backward.run(show_plots=True, print_data=True)