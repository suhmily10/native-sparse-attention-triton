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
# See the License for the specific
import torch
import triton
import math
import logging  # Add logging module
import time     # Add time module for timestamps
import gc  # Add garbage collection module
from native_sparse_attention.ops.torch.topk_sparse_attention import (
    topk_sparse_attention_torch,
)
from native_sparse_attention.ops.triton.topk_sparse_attention import (
    topk_sparse_attention,
    _topk_sparse_attention_fwd,
    _topk_sparse_attention_bwd,
)
from native_sparse_attention.ops.triton.flash_attention import (
    _flash_attention_fwd,
    _flash_attention_bwd,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from native_sparse_attention.ops.torch.topk_sparse_attention_flash import (
    topk_sparse_attention_flash,
)

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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
                topk_idx[i], (0, topk - topk_idx[i].shape[-1]), value=-1
            )
            for i in range(batch_size)
        ]
        topk_idx = torch.cat(topk_idx, dim=0)
        topk_idx = torch.sort(topk_idx, dim=1).values
        topk_idx[:, 0] = 0
        q_idx = torch.cat(
            [torch.arange(seqlens[i], device="cuda") for i in range(batch_size)], dim=0
        )
        
        topk_idx_all_heads.append(topk_idx)
    topk_idx = torch.stack(topk_idx_all_heads, dim=0)
    return topk_idx

# Define bench function similar to test_attention_speed.py
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

if __name__ == "__main__":
    logger.debug("Starting test script execution")
    torch.manual_seed(42)
    batch_size = 3
    block_size = 256
    topk = 4
    
    logger.debug("Preparing test data and parameters")
    # Ensure all sequence lengths are at least blocksize*topk
    min_seqlen = block_size * topk
    # Ensure all sequence lengths are multiples of block_size and greater than min_seqlen
    seqlens = torch.LongTensor([1024, 2048, 4096]).int().cuda()  # All divisible by 64 and > min_seqlen
    
    # Verify that all sequences can select topk blocks
    for seq_len in seqlens:
        num_blocks = math.ceil(seq_len.item() / block_size)
        assert num_blocks >= topk, f"Sequence length {seq_len} has only {num_blocks} blocks, need at least {topk}"
    
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    max_seqlen = seqlens.max().item()
    q = (
        torch.empty(cu_seqlens[-1], 8, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    k = (
        torch.empty(cu_seqlens[-1], 4, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    v = (
        torch.empty(cu_seqlens[-1], 4, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    topk_idx = generate_topk_idx_example(seqlens, block_size, topk, 4)

    logger.debug("Running test implementation: topk_sparse_attention_flash")
    torch.manual_seed(42)
    q1 = q.clone().detach().requires_grad_()
    k1 = k.clone().detach().requires_grad_()
    v1 = v.clone().detach().requires_grad_()
    topk_idx1 = topk_idx.clone().detach()
    cu_seqlens1 = cu_seqlens.clone().detach()

    o1 = topk_sparse_attention_flash(q1, k1, v1, topk_idx1, block_size, cu_seqlens1)

    randn = torch.randn_like(o1)
    loss1 = (o1 * randn).sum()
    loss1.backward()
    logger.debug("Completed test implementation backward pass")

    logger.debug("Running reference implementation: topk_sparse_attention_torch")
    q2 = q.clone().detach().requires_grad_()
    k2 = k.clone().detach().requires_grad_()
    v2 = v.clone().detach().requires_grad_()
    topk_idx2 = topk_idx.clone().detach()
    
    o2 = topk_sparse_attention_torch(q2, k2, v2, topk_idx2, block_size, cu_seqlens)
    
    randn2 = randn.clone().detach()
    loss2 = (o2 * randn2).sum()
    loss2.backward()
    logger.debug("Completed reference implementation backward pass")

    logger.debug("Comparing results between implementations")
    print("Same Output:", torch.allclose(o1, o2, atol=0.01, rtol=0.01))
    print("Max Error:", (o1 - o2).abs().max().item())
    print()
    print("Same Query Gradient:", torch.allclose(q1.grad, q2.grad, atol=0.01, rtol=0.01))
    print("Max Query Gradient Error:", (q1.grad - q2.grad).abs().max().item())
    print()
    print("Same Key Gradient:", torch.allclose(k1.grad, k2.grad, atol=0.01, rtol=0.01))
    print("Max Key Gradient Error:", (k1.grad - k2.grad).abs().max().item())
    print()
    print("Same Value Gradient:", torch.allclose(v1.grad, v2.grad, atol=0.01, rtol=0.01))
    print("Max Value Gradient Error:", (v1.grad - v2.grad).abs().max().item())
    print()
    logger.debug("Comparison completed")

    # benchmark forward pass
    logger.debug("Setting up forward pass benchmark")
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 6)],
            line_arg="provider",
            line_vals=[
                "flash", 
                "topk-flash"
            ],
            line_names=[
                "Flash",
                "TopK-Flash",
            ],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name="** forward with block size 256 **",
            args={"H": 8, "D": 96},
        )
    )
    def benchmark_forward(N, H, D, provider):
        logger.debug(f"Forward benchmark: N={N}, H={H}, D={D}, provider={provider}")
        q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((N, H // 2, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((N, H // 2, D), device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        sm_scale = 1 / math.sqrt(D)

        # Generate topk indices for sparse attention
        topk = 4
        top_idx = generate_topk_idx_example(cu_seqlens[1:], 256, topk, H // 2)

        try:
            if provider == "flash":
                logger.debug(f"Running flash-attention forward benchmark with N={N}")
                start_time = time.time()
                ms = bench(
                    lambda: _flash_attn_varlen_forward(
                        q,
                        k,
                        v,
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
                logger.debug(f"Completed flash-attention forward benchmark in {time.time() - start_time:.2f}s")
            if provider == "topk-flash":
                logger.debug(f"Running topk-flash-attention forward benchmark with N={N}")
                start_time = time.time()
                ms = bench(
                    lambda: topk_sparse_attention_flash(
                        q, k, v, top_idx, 256, cu_seqlens, sm_scale
                    )
                )
                min_ms = ms
                max_ms = ms
                logger.debug(f"Completed topk-flash-attention forward benchmark in {time.time() - start_time:.2f}s")
        finally:
            # Clean up all tensors
            del q, k, v, cu_seqlens, top_idx
            # Force garbage collection before emptying cache
            gc.collect()
            torch.cuda.empty_cache()
        
        return ms, min_ms, max_ms

    logger.debug("Starting forward benchmark runs")
    benchmark_forward.run(show_plots=True, print_data=True)
    logger.debug("Completed forward benchmark runs")

    # benchmark backward pass
    logger.debug("Setting up backward pass benchmark")
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[1024 * 2**i for i in range(1, 6)],
            line_arg="provider",
            line_vals=["flash", "topk-flash"],
            line_names=[
                "Flash",
                "TopK-Flash",
            ],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name="** backward with block size 256 **",
            args={"H": 8, "D": 96},
        )
    )
    def benchmark_backward(N, H, D, provider):
        logger.debug(f"Backward benchmark: N={N}, H={H}, D={D}, provider={provider}")
        q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((N, H // 2, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((N, H // 2, D), device="cuda", dtype=torch.bfloat16)
        o = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        do = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
        lse = torch.randn((N, H), device="cuda", dtype=torch.bfloat16)
        sm_scale = 1 / math.sqrt(D)
        cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        # Generate topk indices for sparse attention
        topk = 4
        top_idx = generate_topk_idx_example(cu_seqlens[1:], 256, topk, H // 2)

        try:
            if provider == "flash":
                logger.debug(f"Running flash-attention backward benchmark with N={N}")
                start_time = time.time()
                ms = bench(
                    lambda: _flash_attn_varlen_backward(
                        do,
                        q,
                        k,
                        v,
                        o,
                        lse.transpose(0, 1),
                        dq,
                        dk,
                        dv,
                        cu_seqlens,
                        cu_seqlens,
                        N,
                        N,
                        dropout_p=0.0,
                        causal=True,
                        softmax_scale=sm_scale,
                        window_size_left=-1,
                        window_size_right=-1,
                        softcap=0.0,
                        alibi_slopes=None,
                        deterministic=False,
                    )
                )
                min_ms = ms
                max_ms = ms
                logger.debug(f"Completed flash-attention backward benchmark in {time.time() - start_time:.2f}s")
            elif provider == "topk-flash":
                logger.debug(f"Running topk-flash-attention backward benchmark with N={N}")
                start_time = time.time()
                
                # For backward benchmarking, we need to run forward first with grad enabled
                q_bench = q.clone().detach().requires_grad_()
                k_bench = k.clone().detach().requires_grad_()
                v_bench = v.clone().detach().requires_grad_()
                
                def run_forward_backward():
                    # Forward pass
                    out = topk_sparse_attention_flash(
                        q_bench, k_bench, v_bench, top_idx, 256, cu_seqlens, sm_scale
                    )
                    # Backward pass
                    out.backward(do, retain_graph=True)
                    
                ms = bench(run_forward_backward)
                min_ms = ms
                max_ms = ms
                logger.debug(f"Completed topk-flash-attention backward benchmark in {time.time() - start_time:.2f}s")
        finally:
            # Clean up all tensors
            del q, k, v, o, do, lse, cu_seqlens, dq, dk, dv, top_idx
            if provider == "topk-flash":
                del q_bench, k_bench, v_bench
            # Force garbage collection before emptying cache
            gc.collect()
            torch.cuda.empty_cache()
        
        return ms, min_ms, max_ms

    logger.debug("Starting backward benchmark runs")
    benchmark_backward.run(show_plots=True, print_data=True)
    logger.debug("Completed backward benchmark runs and script execution")

    # benchmark batch sizes with fixed sequence length
    logger.debug("Setting up batch size benchmark")
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["B"],  # Test different batch sizes
            x_vals=[1,2,4,8,16],
            line_arg="provider",
            line_vals=[
                "flash", 
                "topk-flash"
            ],
            line_names=[
                "Flash",
                "TopK-Flash",
            ],
            styles=[("green", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name="** batch size performance comparison 8192 **",
            args={"N": 4096, "H": 8, "D": 96},  # Fixed sequence length
        )
    )
    def benchmark_batch_sizes(B, N, H, D, provider):
        logger.debug(f"Batch size benchmark: B={B}, N={N}, H={H}, D={D}, provider={provider}")
        
        # Clear CUDA cache before creating new tensors
        torch.cuda.empty_cache()
        logger.debug(f"Starting benchmark B={B}, provider={provider}, memory: {torch.cuda.memory_reserved()//1024**3} GB")
        
        # Total number of tokens across all batches
        total_tokens = B * N
        
        # Create cumulative sequence lengths for batched input
        cu_seqlens = torch.zeros(B+1, device="cuda", dtype=torch.int32)
        for i in range(B):
            cu_seqlens[i+1] = cu_seqlens[i] + N
        
        sm_scale = 1 / math.sqrt(D)
        
        # Create input tensors
        q = torch.randn((total_tokens, H, D), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((total_tokens, H // 2, D), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((total_tokens, H // 2, D), device="cuda", dtype=torch.bfloat16)
        
        # Parameters for topk sparse attention
        block_size = 256
        topk = 4
        
        # Generate topk indices for sparse attention
        top_idx = generate_topk_idx_example(torch.ones(B, device="cuda", dtype=torch.int32) * N, 
                                            block_size, topk, H // 2)
        
        try:
            if provider == "flash":
                logger.debug(f"Running flash-attention benchmark with B={B}, N={N}")
                start_time = time.time()
                ms = bench(
                    lambda: _flash_attn_varlen_forward(
                        q,
                        k,
                        v,
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
                logger.debug(f"Completed flash-attention benchmark in {time.time() - start_time:.2f}s")
                
            elif provider == "topk-flash":
                logger.debug(f"Running topk-flash-attention benchmark with B={B}, N={N}")
                start_time = time.time()
                ms = bench(
                    lambda: topk_sparse_attention_flash(
                        q, k, v, top_idx, block_size, cu_seqlens, sm_scale
                    )
                )
                min_ms = ms
                max_ms = ms
                logger.debug(f"Completed topk-flash-attention benchmark in {time.time() - start_time:.2f}s")
        finally:
            # Clean up all tensors
            del q, k, v, cu_seqlens, top_idx
            # Force garbage collection before emptying cache
            gc.collect()
            torch.cuda.empty_cache()
            
        return ms, min_ms, max_ms

    logger.debug("Starting batch size benchmark runs")
    benchmark_batch_sizes.run(show_plots=True, print_data=True)
    logger.debug("Completed batch size benchmark runs")

 