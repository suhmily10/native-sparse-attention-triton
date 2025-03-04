# test_compression_methods.py
import torch
import triton
import math
from native_sparse_attention.module.native_sparse_attention import NativeSparseAttention, NativeSparseAttentionNoRoPE
from native_sparse_attention.module.rope import RopeConfig

def setup_native_sparse_attention(
    hidden_size=4096,
    num_q_heads=32,
    num_kv_heads=8,
    head_dim=128,
    kernel_size=32,
    kernel_stride=16,
    block_size=128,
    topk=8,
    init_blocks=1,
    local_blocks=2,
    window_size=1024,
    use_compressed_attn=True,
    use_topk_sparse_attn=True,
    use_sliding_window_attn=True,
    compression_method="linear",
    use_rope=True,
):
    if use_rope:
        model = NativeSparseAttention(
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
            use_sliding_window_attn=use_sliding_window_attn,
            rope_config=RopeConfig(head_dim=head_dim)
        )
    else:
        model = NativeSparseAttentionNoRoPE(
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
            use_sliding_window_attn=use_sliding_window_attn,
        )
    
    # Set the compression method
    if hasattr(model, 'compression_method'):
        model.compression_method = compression_method
    
    return model

def benchmark_compression_methods():
    # Benchmark configuration
    batch_size = 2
    seq_length = 4096
    hidden_size = 4096
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128
    
    # NSA configuration
    kernel_size = 32
    kernel_stride = 16
    block_size = 128
    topk = 8
    init_blocks = 1
    local_blocks = 2
    window_size = 1024
    
    # Setup device and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available. Exiting.")
        return
    
    # Create input data
    total_seqlen = batch_size * seq_length
    x = torch.randn((total_seqlen, hidden_size), device=device, dtype=torch.bfloat16)
    grad_out = torch.randn((total_seqlen, hidden_size), device=device, dtype=torch.bfloat16)
    
    # Create sequence lengths
    seqlens = torch.full((batch_size,), seq_length, device=device, dtype=torch.int32)
    cu_seqlens = torch.zeros((batch_size + 1,), device=device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    
    # Compression methods to benchmark
    compression_methods = ["conv", "linear", "avgpool", "weighted"]
    
    # Benchmark parameters
    quantiles = [0.5, 0.2, 0.8]
    
    # Print header
    print(f"\n{'=' * 80}")
    print(f"Benchmarking different compression methods with sequence length: {seq_length}, batch size: {batch_size}")
    print(f"{'=' * 80}")
    
    # Forward pass benchmarks
    print("\nFORWARD PASS BENCHMARKS")
    print(f"{'Method':<12} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print(f"{'-' * 50}")
    
    for method in compression_methods:
        # Create model with specific compression method
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
            use_topk_sparse_attn=True,
            use_sliding_window_attn=False,
            compression_method=method
        ).to(device).to(torch.bfloat16)
        
        # Benchmark forward pass
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: model(x, cu_seqlens),
            quantiles=quantiles,
        )
        
        print(f"{method:<12} {ms:<12.4f} {min_ms:<12.4f} {max_ms:<12.4f}")
    
    # Backward pass benchmarks
    print("\nBACKWARD PASS BENCHMARKS")
    print(f"{'Method':<12} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print(f"{'-' * 50}")
    
    for method in compression_methods:
        # Create model with specific compression method and requires_grad
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
            use_topk_sparse_attn=True,
            use_sliding_window_attn=False,
            compression_method=method
        ).to(device).to(torch.bfloat16)
        
        # Forward pass to get output for backward
        x.requires_grad_(True)
        output = model(x, cu_seqlens)
        
        # Benchmark backward pass
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.autograd.backward(
                output, grad_out, retain_graph=True
            ),
            quantiles=quantiles,
        )
        
        # Reset requires_grad for next iteration
        x.requires_grad_(False)
        
        print(f"{method:<12} {ms:<12.4f} {min_ms:<12.4f} {max_ms:<12.4f}")
    
    print(f"\n{'=' * 80}")

if __name__ == "__main__":
    benchmark_compression_methods()