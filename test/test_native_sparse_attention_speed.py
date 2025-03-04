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
    local_blocks, window_size, use_rope=False,
    use_compressed_attn=True, use_topk_sparse_attn=True, use_sliding_window_attn=True
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
            rope_config=rope_config,
            use_compressed_attn=use_compressed_attn,
            use_topk_sparse_attn=use_topk_sparse_attn,
            use_sliding_window_attn=use_sliding_window_attn
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
            window_size=window_size,
            use_compressed_attn=use_compressed_attn,
            use_topk_sparse_attn=use_topk_sparse_attn,
            use_sliding_window_attn=use_sliding_window_attn
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
                       "native-sparse-b1", "native-sparse-b2",
                       "nsa-compressed-b1", "nsa-compressed-b2",
                       "nsa-topk-b1", "nsa-topk-b2",
                       "nsa-sliding-b1", "nsa-sliding-b2"],
            line_names=[
                "Flash-b1", "Flash-b2",
                "TritonFlash-b1", "TritonFlash-b2",
                "NSA-All-b1", "NSA-All-b2",
                "NSA-Comp-b1", "NSA-Comp-b2",
                "NSA-Topk-b1", "NSA-Topk-b2",
                "NSA-Slide-b1", "NSA-Slide-b2",
            ],
            styles=[("green", "-"), ("green", "--"),
                    ("red", "-"), ("red", "--"),
                    ("blue", "-"), ("blue", "--"),
                    ("purple", "-"), ("purple", "--"),
                    ("orange", "-"), ("orange", "--"),
                    ("brown", "-"), ("brown", "--")],
            ylabel="ms",
            plot_name="** forward pass comparison **",
            args={"H": 64, "D": 192},
        )
    )
    def benchmark_forward(N, H, D, provider):
        # Parse provider to get method and batch size
        parts = provider.split('-')
        batch_info = parts[-1]  # Get the last part (b1, b2)
        method = '-'.join(parts[:-1])  # Join all parts except the last one
        batch_size = int(batch_info[1:])  # Extract number after 'b'
        
        # Common parameters based on paper's GQA setup
        head_dim = D
        num_q_heads = H
        num_kv_heads = 4  # Number of groups
        hidden_size = num_q_heads * head_dim  # Total hidden dimension
        
        # Total sequence length = N * batch_size
        total_seqlen = N * batch_size
        
        # Each sample has fixed length N
        cu_seqlens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        for i in range(1, batch_size + 1):
            cu_seqlens[i] = cu_seqlens[i-1] + N
        
        sm_scale = 1 / math.sqrt(D)
        
        # Additional parameters for native-sparse
        kernel_size = 32       # 压缩块大小 l=32
        kernel_stride = 16     # 滑动步长 d=16
        block_size = 64        # 选择块大小 l'=64
        topk = 16              # 选择块数 n=16
        init_blocks = 1        # 固定初始块数
        local_blocks = 2       # 固定局部块数
        window_size = 512      # 滑动窗口大小 w=512
        
        quantiles = [0.5, 0.2, 0.8]
        
        if method == "flash":
            # Input shape: total token count = batch_size * N
            q = torch.randn((batch_size * N, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            k = torch.randn((batch_size * N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            v = torch.randn((batch_size * N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attn_varlen_forward(
                    q, k, v, cu_seqlens, cu_seqlens, 
                    N,  # max_seqlen_q
                    N,  # max_seqlen_k
                    dropout_p=0.0,
                    causal=True,
                    softmax_scale=sm_scale,
                ),
                quantiles=quantiles,
            )
            
        elif method == "triton-flash":
            # Same input shape
            q = torch.randn((batch_size * N, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            k = torch.randn((batch_size * N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            v = torch.randn((batch_size * N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            
            # 压缩K/V使用公共参数
            compressed_k = []
            compressed_v = []
            new_cu_seqlens_k = [0]
            for i in range(batch_size):
                seq_start = i * N
                seq_k = k[seq_start:seq_start+N]
                compressed_len = (N - kernel_size) // kernel_stride + 1
                compressed_k.append(torch.nn.functional.avg_pool1d(
                    seq_k.permute(1,2,0), kernel_size=kernel_size, stride=kernel_stride
                ).permute(2,0,1))
                compressed_v.append(torch.nn.functional.avg_pool1d(
                    v[seq_start:seq_start+N].permute(1,2,0), kernel_size=kernel_size, stride=kernel_stride
                ).permute(2,0,1))
                new_cu_seqlens_k.append(new_cu_seqlens_k[-1] + compressed_len)
            
            k = torch.cat(compressed_k, dim=0)
            v = torch.cat(compressed_v, dim=0)
            cu_seqlens_k = torch.tensor(new_cu_seqlens_k, device="cuda", dtype=torch.int32)
            max_seqlen_k = (N - kernel_size) // kernel_stride + 1
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attention_fwd(
                    q, k, v, cu_seqlens, cu_seqlens_k, N, max_seqlen_k, True, sm_scale
                ),
                quantiles=quantiles,
            )
            
        elif method == "native-sparse":
            # All mechanisms enabled
            x = torch.randn((batch_size * N, hidden_size), device="cuda", dtype=torch.bfloat16)
            
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
            
        elif method == "nsa-compressed":
            # Only compressed attention enabled
            x = torch.randn((batch_size * N, hidden_size), device="cuda", dtype=torch.bfloat16)
            
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
                window_size=window_size,
                use_compressed_attn=True,
                use_topk_sparse_attn=False,
                use_sliding_window_attn=False
            ).cuda().to(torch.bfloat16)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: model(x, cu_seqlens),
                quantiles=quantiles,
            )
            
        elif method == "nsa-topk":
            # Only topk sparse attention enabled
            # Note: topk depends on compressed attention for topk_idx
            x = torch.randn((batch_size * N, hidden_size), device="cuda", dtype=torch.bfloat16)
            
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
                window_size=window_size,
                use_compressed_attn=True,  # Needed for topk_idx
                use_topk_sparse_attn=True,
                use_sliding_window_attn=False
            ).cuda().to(torch.bfloat16)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: model(x, cu_seqlens),
                quantiles=quantiles,
            )
            
        elif method == "nsa-sliding":
            # Only sliding window attention enabled
            x = torch.randn((batch_size * N, hidden_size), device="cuda", dtype=torch.bfloat16)
            
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
                window_size=window_size,
                use_compressed_attn=False,
                use_topk_sparse_attn=False,
                use_sliding_window_attn=True
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
                       "native-sparse-b1", "native-sparse-b2",
                       "nsa-compressed-b1", "nsa-compressed-b2",
                       "nsa-topk-b1", "nsa-topk-b2",
                       "nsa-sliding-b1", "nsa-sliding-b2"],
            line_names=[
                "Flash-b1", "Flash-b2",
                "TritonFlash-b1", "TritonFlash-b2",
                "NSA-All-b1", "NSA-All-b2",
                "NSA-Comp-b1", "NSA-Comp-b2",
                "NSA-Topk-b1", "NSA-Topk-b2",
                "NSA-Slide-b1", "NSA-Slide-b2",
            ],
            styles=[("green", "-"), ("green", "--"),
                    ("red", "-"), ("red", "--"),
                    ("blue", "-"), ("blue", "--"),
                    ("purple", "-"), ("purple", "--"),
                    ("orange", "-"), ("orange", "--"),
                    ("brown", "-"), ("brown", "--")],
            ylabel="ms",
            plot_name="** backward pass comparison **",
            args={"H": 64, "D": 192},
        )
    )
    def benchmark_backward(N, H, D, provider):
        # Parse provider to get method and batch size
        parts = provider.split('-')
        batch_info = parts[-1]  # Get the last part (b1, b2)
        method = '-'.join(parts[:-1])  # Join all parts except the last one
        batch_size = int(batch_info[1:])  # Extract number after 'b'
        
        # Common parameters based on paper's GQA setup
        head_dim = D
        num_q_heads = H
        num_kv_heads = 4  # Number of groups
        hidden_size = num_q_heads * head_dim  # Total hidden dimension
        
        # Additional parameters for native-sparse
        kernel_size = 32       # 统一kernel参数
        kernel_stride = 16     # 统一stride参数
        block_size = 64        # 统一块大小
        topk = 16              # 统一topk数
        init_blocks = 1        # 统一初始块数
        local_blocks = 2       # 统一局部块数
        window_size = 512      # 统一窗口大小
        
        # Setup cu_seqlens for different batch sizes - keep sequence length fixed at N
        cu_seqlens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        for i in range(1, batch_size + 1):
            cu_seqlens[i] = cu_seqlens[i-1] + N  # Each sample has fixed length N
        
        # Total sequence length across all batches
        total_seqlen = N * batch_size
        
        sm_scale = 1 / math.sqrt(D)
        
        quantiles = [0.5, 0.2, 0.8]
        
        if method == "flash":
            # Create inputs with proper shapes for flash attention
            q = torch.randn((total_seqlen, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn((total_seqlen, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn((total_seqlen, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            
            # Forward pass to get outputs needed for backward
            o = _flash_attn_varlen_forward(
                q, k, v, cu_seqlens, cu_seqlens, N, N, dropout_p=0.0, causal=True, softmax_scale=sm_scale
            )[0]
            
            do = torch.randn_like(o)
            lse = torch.randn((num_q_heads, total_seqlen), device="cuda", dtype=torch.float32)
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attn_varlen_backward(
                    do, q, k, v, o, lse, dq, dk, dv, cu_seqlens, cu_seqlens, 
                    N, N, dropout_p=0.0, causal=True, softmax_scale=sm_scale,
                    window_size_left=-1, window_size_right=-1, softcap=0.0, 
                    alibi_slopes=None, deterministic=False, zero_tensors=False
                ),
                quantiles=quantiles,
            )
            
        elif method == "triton-flash":
            # 使用公共参数压缩K/V
            q = torch.randn((total_seqlen, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn((total_seqlen, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn((total_seqlen, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            
            # 压缩K/V
            compressed_k = []
            compressed_v = []
            new_cu_seqlens_k = [0]
            for i in range(batch_size):
                seq_start = i * N
                seq_k = k[seq_start:seq_start+N]
                compressed_len = (N - kernel_size) // kernel_stride + 1
                compressed_k.append(torch.nn.functional.avg_pool1d(
                    seq_k.permute(1,2,0), kernel_size=kernel_size, stride=kernel_stride
                ).permute(2,0,1))
                compressed_v.append(torch.nn.functional.avg_pool1d(
                    v[seq_start:seq_start+N].permute(1,2,0), kernel_size=kernel_size, stride=kernel_stride
                ).permute(2,0,1))
                new_cu_seqlens_k.append(new_cu_seqlens_k[-1] + compressed_len)
            
            k = torch.cat(compressed_k, dim=0)
            v = torch.cat(compressed_v, dim=0)
            cu_seqlens_k = torch.tensor(new_cu_seqlens_k, device="cuda", dtype=torch.int32)
            max_seqlen_k = (N - kernel_size) // kernel_stride + 1
            
            # Forward pass
            o, lse = _flash_attention_fwd(q, k, v, cu_seqlens, cu_seqlens_k, N, max_seqlen_k, True, sm_scale)
            do = torch.randn_like(o)
            
            # Backward pass
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attention_bwd(o, do, lse, q, k, v, cu_seqlens, cu_seqlens_k, N, max_seqlen_k, True, sm_scale),
                quantiles=quantiles,
            )
            
        elif method == "native-sparse":
            # All attention mechanisms enabled
            x = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            grad_out = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16)
            model = setup_native_sparse_attention(
                hidden_size=hidden_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kernel_size=kernel_size,  # 使用公共参数
                kernel_stride=kernel_stride,  # 使用公共参数
                block_size=block_size,
                topk=topk,
                init_blocks=init_blocks,
                local_blocks=local_blocks,
                window_size=window_size
            ).cuda().to(torch.bfloat16)
            
            output = model(x, cu_seqlens)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.autograd.backward(
                    output, grad_out, retain_graph=True
                ),
                quantiles=quantiles,
            )
            
        elif method == "nsa-compressed":
            # Only compressed attention enabled
            x = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            grad_out = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16)
            
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
                window_size=window_size,
                use_compressed_attn=True,
                use_topk_sparse_attn=False,
                use_sliding_window_attn=False
            ).cuda().to(torch.bfloat16)
            
            output = model(x, cu_seqlens)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.autograd.backward(
                    output, grad_out, retain_graph=True
                ),
                quantiles=quantiles,
            )
            
        elif method == "nsa-topk":
            # Only topk sparse attention enabled (with compressed for topk_idx)
            x = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            grad_out = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16)
            
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
                window_size=window_size,
                use_compressed_attn=True,  # Needed for topk_idx
                use_topk_sparse_attn=True,
                use_sliding_window_attn=False
            ).cuda().to(torch.bfloat16)
            
            output = model(x, cu_seqlens)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.autograd.backward(
                    output, grad_out, retain_graph=True
                ),
                quantiles=quantiles,
            )
            
        elif method == "nsa-sliding":
            # Only sliding window attention enabled
            x = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            grad_out = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16)
            
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
                window_size=window_size,
                use_compressed_attn=False,
                use_topk_sparse_attn=False,
                use_sliding_window_attn=True
            ).cuda().to(torch.bfloat16)
            
            output = model(x, cu_seqlens)
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.autograd.backward(
                    output, grad_out, retain_graph=True
                ),
                quantiles=quantiles,
            )
            
        return ms, min_ms, max_ms

    benchmark_backward.run(show_plots=True, print_data=True)