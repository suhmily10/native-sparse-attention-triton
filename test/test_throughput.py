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
    
    # Benchmark prefilling performance - processing entire context at once
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],  # sequence length
            x_vals=[1024 * 2**i for i in range(0, 6)],  # from 1K to 32K
            line_arg="provider",
            line_vals=["flash-b1", "flash-b2", 
                      "triton-flash-b1", "triton-flash-b2",
                      "native-sparse-b1", "native-sparse-b2"],
            line_names=[
                "Flash-b1", "Flash-b2",
                "TritonFlash-b1", "TritonFlash-b2", 
                "NativeSparse-b1", "NativeSparse-b2"
            ],
            styles=[("green", "-"), ("green", "--"),
                   ("red", "-"), ("red", "--"),
                   ("blue", "-"), ("blue", "--")],
            ylabel="Tokens/Second",
            plot_name="Prefilling Throughput",
            args={"H": 32, "D": 128},
        )
    )
    def benchmark_prefilling(N, H, D, provider):
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
        
        # Total sequence length = N * batch_size
        total_seqlen = N * batch_size
        
        # Each sample has fixed length N
        cu_seqlens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        for i in range(1, batch_size + 1):
            cu_seqlens[i] = cu_seqlens[i-1] + N
        
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
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attention_fwd(
                    q, k, v, cu_seqlens, cu_seqlens, N, N, True, sm_scale
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
        
        # Calculate tokens per second
        tokens_per_second = (total_seqlen / (ms / 1000))
        tokens_per_second_min = (total_seqlen / (max_ms / 1000))  # max time = min throughput
        tokens_per_second_max = (total_seqlen / (min_ms / 1000))  # min time = max throughput
        
        return tokens_per_second, tokens_per_second_min, tokens_per_second_max

    benchmark_prefilling.run(show_plots=True, print_data=True)
    
    # Benchmark decoding performance - simulate token-by-token generation
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],  # context length
            x_vals=[1024 * 2**i for i in range(0, 6)],  # from 1K to 32K
            line_arg="provider",
            line_vals=["flash-b1", "flash-b2", 
                      "triton-flash-b1", "triton-flash-b2",
                      "native-sparse-b1", "native-sparse-b2"],
            line_names=[
                "Flash-b1", "Flash-b2",
                "TritonFlash-b1", "TritonFlash-b2", 
                "NativeSparse-b1", "NativeSparse-b2"
            ],
            styles=[("green", "-"), ("green", "--"),
                   ("red", "-"), ("red", "--"),
                   ("blue", "-"), ("blue", "--")],
            ylabel="Tokens/Second",
            plot_name="Decoding Throughput",
            args={"H": 32, "D": 128},
        )
    )
    def benchmark_decoding(N, H, D, provider):
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
        
        # For decoding, we only compute attention for the last token
        # N = context length, context + 1 new token
        
        # Setup cu_seqlens
        cu_seqlens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
        for i in range(1, batch_size + 1):
            cu_seqlens[i] = cu_seqlens[i-1] + (N + 1)  # Each sequence has (N+1) tokens
        
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
            # Total sequence length includes context and the new token
            total_seqlen = (N + 1) * batch_size
            
            q = torch.randn((total_seqlen, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            k = torch.randn((total_seqlen, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            v = torch.randn((total_seqlen, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            
            # For proper decoding simulation, we should only run attention for the last token
            # Extract just the last query token for each sequence in the batch
            q_last = torch.zeros((batch_size, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            for i in range(batch_size):
                q_last[i] = q[cu_seqlens[i+1]-1]
            
            # Create cu_seqlens_q for just the query tokens (one per sequence)
            cu_seqlens_q = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
            for i in range(1, batch_size + 1):
                cu_seqlens_q[i] = i  # One token per sequence
            
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attn_varlen_forward(
                    q_last.reshape(-1, num_q_heads, head_dim),  # Reshape to [batch_size, H, D]
                    k, v, cu_seqlens_q, cu_seqlens, 
                    1,      # max_seqlen_q (just 1 token per sequence)
                    N + 1,  # max_seqlen_k (full context length)
                    dropout_p=0.0,
                    causal=True,
                    softmax_scale=sm_scale,
                ),
                quantiles=quantiles,
            )
            
        elif method == "triton-flash":
            # Total sequence length includes context and the new token
            total_seqlen = (N + 1) * batch_size
            
            # For triton flash attention, we need to create full sequences
            # but we'll simulate decoding by only measuring the last token
            q = torch.randn((total_seqlen, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            k = torch.randn((total_seqlen, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            v = torch.randn((total_seqlen, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
            
            # Triton flash attention requires q_len == k_len == v_len
            # so we need to process the whole sequence and can't just use the last token
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attention_fwd(
                    q, k, v, cu_seqlens, cu_seqlens, N + 1, N + 1, True, sm_scale
                ),
                quantiles=quantiles,
            )
            
        elif method == "native-sparse":
            # For decoding, we need the full sequence to handle convolution properly
            total_seqlen = (N + 1) * batch_size
            x = torch.randn((total_seqlen, hidden_size), device="cuda", dtype=torch.bfloat16)
            
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
            
            # For decoding, we need to process the whole sequence
            # We'll time the entire operation and then estimate the per-token cost
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: model(x, cu_seqlens),
                quantiles=quantiles,
            )
        
        # For decoding, we measure tokens per second based on batch_size
        # since we're only generating one new token per sequence
        tokens_per_second = (batch_size / (ms / 1000))
        tokens_per_second_min = (batch_size / (max_ms / 1000))
        tokens_per_second_max = (batch_size / (min_ms / 1000))
        
        return tokens_per_second, tokens_per_second_min, tokens_per_second_max

    benchmark_decoding.run(show_plots=True, print_data=True)
    
    # Benchmark single attention operation throughput with varying sequence lengths
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],  # sequence length
            x_vals=[512 * 2**i for i in range(0, 6)],  # from 512 to 16K
            line_arg="provider",
            line_vals=["flash-prefill", "triton-flash-prefill", "native-sparse-prefill",
                       "flash-decode", "triton-flash-decode", "native-sparse-decode"],
            line_names=[
                "Flash-Prefill", "TritonFlash-Prefill", "NativeSparse-Prefill",
                "Flash-Decode", "TritonFlash-Decode", "NativeSparse-Decode"
            ],
            styles=[("green", "-"), ("red", "-"), ("blue", "-"),
                   ("green", "--"), ("red", "--"), ("blue", "--")],
            ylabel="Tokens/Second",
            plot_name="Single Attention Throughput",
            args={"H": 32, "D": 128},
        )
    )
    def benchmark_single_attention(N, H, D, provider):
        # Parse provider to get method and operation type
        parts = provider.split('-')
        operation = parts[-1]  # Get "prefill" or "decode"
        method = '-'.join(parts[:-1])  # Join all parts except the last one
        
        # Common parameters
        hidden_size = H * D
        num_q_heads = H
        num_kv_heads = H // 8
        head_dim = D
        batch_size = 1  # Single attention operation
        
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
        
        if operation == "prefill":
            # Setup cu_seqlens for prefilling (entire sequence)
            cu_seqlens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
            cu_seqlens[1] = N  # Single sequence of length N
            
            if method == "flash":
                q = torch.randn((N, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                k = torch.randn((N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: _flash_attn_varlen_forward(
                        q, k, v, cu_seqlens, cu_seqlens, 
                        N, N, dropout_p=0.0, causal=True, softmax_scale=sm_scale,
                    ),
                    quantiles=quantiles,
                )
                
            elif method == "triton-flash":
                q = torch.randn((N, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                k = torch.randn((N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((N, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: _flash_attention_fwd(
                        q, k, v, cu_seqlens, cu_seqlens, N, N, True, sm_scale
                    ),
                    quantiles=quantiles,
                )
                
            elif method == "native-sparse":
                x = torch.randn((N, hidden_size), device="cuda", dtype=torch.bfloat16)
                
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
            
            # Calculate tokens per second for prefilling (all tokens in sequence)
            tokens_per_second = (N / (ms / 1000))
            tokens_per_second_min = (N / (max_ms / 1000))
            tokens_per_second_max = (N / (min_ms / 1000))
        
        else:  # operation == "decode"
            # Setup for decoding - we have N context tokens plus 1 new token
            context_len = N
            total_len = N + 1
            
            # Setup cu_seqlens
            cu_seqlens = torch.zeros(batch_size + 1, device="cuda", dtype=torch.int32)
            cu_seqlens[1] = total_len  # Single sequence of length N+1
            
            if method == "flash":
                q = torch.randn((total_len, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                k = torch.randn((total_len, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((total_len, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                
                # Extract only the last token for decoding
                q_last = q[-1:].clone()
                
                # Create cu_seqlens_q for the single query token
                cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
                
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: _flash_attn_varlen_forward(
                        q_last, k, v, cu_seqlens_q, cu_seqlens, 
                        1, total_len, dropout_p=0.0, causal=True, softmax_scale=sm_scale,
                    ),
                    quantiles=quantiles,
                )
                
            elif method == "triton-flash":
                # For Triton Flash, we need the same sequence length for q, k, v
                # So we run the full attention and time it
                q = torch.randn((total_len, num_q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                k = torch.randn((total_len, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((total_len, num_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
                
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: _flash_attention_fwd(
                        q, k, v, cu_seqlens, cu_seqlens, total_len, total_len, True, sm_scale
                    ),
                    quantiles=quantiles,
                )
                
                # Since we had to process the whole sequence, adjust the time to estimate
                # just the last token by scaling proportionally to sequence position
                # This is an approximation as the last token costs more with causal attention
                ms = ms * (1 / total_len)
                min_ms = min_ms * (1 / total_len)
                max_ms = max_ms * (1 / total_len)
                
            elif method == "native-sparse":
                # For decoding, we need the full sequence to handle convolution properly
                x = torch.randn((total_len, hidden_size), device="cuda", dtype=torch.bfloat16)
                
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
                
                # For decoding, we need to process the whole sequence
                # We'll time the entire operation and then estimate the per-token cost
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: model(x, cu_seqlens),
                    quantiles=quantiles,
                )
            
            # For decoding, we're generating 1 token per operation
            tokens_per_second = (1 / (ms / 1000))
            tokens_per_second_min = (1 / (max_ms / 1000))
            tokens_per_second_max = (1 / (min_ms / 1000))
        
        return tokens_per_second, tokens_per_second_min, tokens_per_second_max

    benchmark_single_attention.run(show_plots=True, print_data=True)
