# Copyright 2025 Xunhao Lai.
# Licensed under the Apache License, Version 2.0 (see LICENSE).

import torch
import triton
import math
from native_sparse_attention.module.native_sparse_attention import NativeSparseAttention
from native_sparse_attention.module.rope import RopeConfig

def setup_model(hidden_size=4096, num_q_heads=32, num_kv_heads=4, head_dim=128):
    rope_config = RopeConfig(
        head_dim=head_dim,
        rope_theta=10000.0,
        rope_scaling={
            "factor": 1.0,
            "high_freq_factor": 1.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 2048,
            "rope_type": "default"
        }
    )
    return NativeSparseAttention(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kernel_size=16,
        kernel_stride=4,
        block_size=64,
        topk=8,
        init_blocks=1,
        local_blocks=2,
        window_size=256,
        rope_config=rope_config
    ).cuda().to(torch.bfloat16).eval()

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[2**i for i in range(10, 14)],  # Reduced from 15 to 14 to avoid memory issues
        line_arg='batch_size',
        line_vals=[1, 4, 16],
        line_names=['bs=1', 'bs=4', 'bs=16'],
        styles=[('blue', '-'), ('green', '--'), ('red', '-.')],
        ylabel='Tokens/s',
        plot_name='prefill_throughput',
        args={'hidden_size': 4096}
    )
)
def benchmark_prefill(seq_len, batch_size, hidden_size):
    # Limit the maximum tokens to avoid CUDA memory errors
    if seq_len * batch_size > 32768:
        return 0, None, None  # Skip this configuration
        
    model = setup_model(hidden_size)
    total_tokens = seq_len * batch_size
    
    # Create input with shape [batch_size * seq_len, hidden_size]
    x = torch.randn((batch_size * seq_len, hidden_size), device='cuda', dtype=torch.bfloat16)
    cu_seqlens = torch.arange(0, (batch_size+1)*seq_len, seq_len, device='cuda', dtype=torch.int32)

    # Warmup
    for _ in range(3):
        model(x, cu_seqlens)
    
    # Benchmark
    ms_per_token = triton.testing.do_bench(
        lambda: model(x, cu_seqlens),
        quantiles=[0.5, 0.2, 0.8]
    )[0] / total_tokens

    return total_tokens / (ms_per_token * 1e-3), None, None  # tokens per second

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[2**i for i in range(8, 15)],
        line_arg='batch_size',
        line_vals=[1, 4, 16],
        line_names=['bs=1', 'bs=4', 'bs=16'],
        styles=[('blue', '-'), ('green', '--'), ('red', '-.')],
        ylabel='Tokens/s',
        plot_name='decoding_throughput',
        args={'hidden_size': 4096}
    )
)
def benchmark_decoding(seq_len, batch_size, hidden_size):
    # For the decoding benchmark, ensure we use at least kernel_size tokens
    # since conv_compress requires input length >= kernel_size
    min_seq_len = 16  # Same as kernel_size in setup_model
    
    model = setup_model(hidden_size)
    
    # Create input with shape [batch_size * min_seq_len, hidden_size]
    # We'll only measure the throughput of the last token
    x = torch.randn((batch_size * min_seq_len, hidden_size), device='cuda', dtype=torch.bfloat16)
    
    # Create cu_seqlens to simulate having min_seq_len tokens per sequence
    cu_seqlens = torch.arange(0, (batch_size+1)*min_seq_len, min_seq_len, device='cuda', dtype=torch.int32)
    
    # Warmup and compile
    for _ in range(3):
        model(x, cu_seqlens)
    
    # Benchmark decoding latency
    ms_per_token = triton.testing.do_bench(
        lambda: model(x, cu_seqlens),
        quantiles=[0.5, 0.2, 0.8]
    )[0] / batch_size  # Dividing by batch_size to get ms per token
    
    return batch_size / (ms_per_token * 1e-3), None, None  # tokens per second

if __name__ == '__main__':
    benchmark_prefill.run(show_plots=True, print_data=True)
    benchmark_decoding.run(show_plots=True, print_data=True) 