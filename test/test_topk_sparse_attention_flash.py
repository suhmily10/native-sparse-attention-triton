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
    level=logging.INFO,
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
        torch.Tensor: shape [num_heads, total_seqlen, topk], topk key value block idx for each query.
    """
    batch_size = seqlens.shape[0]
    
    # 计算每个序列的块数
    num_blocks = torch.ceil(seqlens / block_size).to(torch.int32)
    
    # 计算每个序列的起始块索引（绝对块偏移）
    block_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.cumsum(num_blocks, dim=0)
    ])
    
    # 计算每个序列的起始token索引（绝对token偏移）
    token_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.cumsum(seqlens, dim=0)
    ])
    
    topk_idx_all_heads = []
    for _ in range(num_heads):
        topk_idx = []
        for i in range(batch_size):
            seq_blocks = num_blocks[i]
            block_offset = block_offsets[i]  # 当前序列的块偏移
            seq_len = seqlens[i]
            token_offset = token_offsets[i]  # 当前序列的token偏移
            
            # 确保每个序列都有足够的块可选
            assert seq_blocks >= topk, f"Sequence {i} has only {seq_blocks} blocks, fewer than topk={topk}"
            
            # 为每个token生成相对块索引并转换为绝对块索引
            relative_idx = torch.randn(seq_len, seq_blocks, device="cuda") \
                             .topk(topk, dim=-1).indices
            absolute_idx = relative_idx + block_offset  # 转换为绝对块索引
            
            topk_idx.append(absolute_idx.to(torch.int32))
            
        topk_idx = torch.cat(topk_idx, dim=0)
        topk_idx = torch.sort(topk_idx, dim=1).values
        
        # 每个查询至少需要一个有效块
        # 确保第一个索引总是有效的（设为对应的块索引）
        flat_q_idx = torch.cat(
            [torch.arange(seqlens[i], device="cuda") + token_offsets[i] for i in range(batch_size)]
        )
        first_block_idx = torch.div(flat_q_idx, block_size, rounding_mode='floor')
        topk_idx[:, 0] = first_block_idx
        
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
    block_size = 64
    topk = 16
    
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
        torch.empty(cu_seqlens[-1], 32, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    k = (
        torch.empty(cu_seqlens[-1], 8, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    v = (
        torch.empty(cu_seqlens[-1], 8, 128, device="cuda")
        .uniform_(-1, 1)
        .to(torch.bfloat16)
    )
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    # import pdb; pdb.set_trace()
    topk_idx = generate_topk_idx_example(seqlens, block_size, topk, 8)

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
    
    gc.collect()
    torch.cuda.empty_cache()
    

    # benchmark forward pass
    logger.debug("Setting up forward pass benchmark")
    def benchmark_forward():
        N_vals = [1024 * 2**i for i in range(1, 6)]
        H = 32
        D = 128
        providers = ["flash", "topk-flash"]
        
        print("\n** Forward benchmark with block size 64 **")
        print(f"{'N':<10} {'Flash (ms)':<15} {'TopK-Flash (ms)':<15}")
        print("-" * 40)
        
        for N in N_vals:
            results = {}
            
            for provider in providers:
                # Log memory usage before benchmark
                memory_before = torch.cuda.memory_allocated() / (1024**3)
                logger.debug(f"Before {provider} forward benchmark (N={N}): {memory_before:.2f} GB allocated")
                
                logger.debug(f"Forward benchmark: N={N}, H={H}, D={D}, provider={provider}")
                q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
                k = torch.randn((N, H // 4, D), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((N, H // 4, D), device="cuda", dtype=torch.bfloat16)
                cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
                sm_scale = 1 / math.sqrt(D)

                # Generate topk indices for sparse attention
                topk = 16
                top_idx = generate_topk_idx_example(cu_seqlens[1:], 64, topk, H // 4)

                try:
                    if provider == "flash":
                        logger.debug(f"Running flash-attention forward benchmark with N={N}")
                        start_time = time.time()
                        ms = bench(
                            lambda: _flash_attn_varlen_forward(
                                q, k, v, cu_seqlens, cu_seqlens, N, N,
                                dropout_p=0.0, causal=True, softmax_scale=sm_scale,
                            )
                        )
                        logger.debug(f"Completed flash-attention forward benchmark in {time.time() - start_time:.2f}s")
                    elif provider == "topk-flash":
                        logger.debug(f"Running topk-flash-attention forward benchmark with N={N}")
                        start_time = time.time()
                        ms = bench(
                            lambda: topk_sparse_attention_flash(
                                q, k, v, top_idx, 64, cu_seqlens, sm_scale
                            )
                        )
                        logger.debug(f"Completed topk-flash-attention forward benchmark in {time.time() - start_time:.2f}s")
                    
                    results[provider] = ms
                finally:
                    # Clean up all tensors
                    del q, k, v, cu_seqlens, top_idx
                    # Force garbage collection before emptying cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Log memory usage after cleanup
                    memory_after = torch.cuda.memory_allocated() / (1024**3)
                    logger.debug(f"After {provider} forward benchmark (N={N}): {memory_after:.2f} GB allocated")
                    logger.debug(f"Memory cleaned up: {memory_before - memory_after:.2f} GB")
            
            print(f"{N:<10} {results.get('flash', 'N/A'):<15.2f} {results.get('topk-flash', 'N/A'):<15.2f}")

    logger.debug("Starting forward benchmark runs")
    benchmark_forward()
    logger.debug("Completed forward benchmark runs")

    # benchmark backward pass
    logger.debug("Setting up backward pass benchmark")
    def benchmark_backward():
        N_vals = [1024 * 2**i for i in range(1, 6)]
        H = 32
        D = 128
        providers = ["flash", "topk-flash"]
        
        print("\n** Backward benchmark with block size 64 **")
        print(f"{'N':<10} {'Flash (ms)':<15} {'TopK-Flash (ms)':<15}")
        print("-" * 40)
        
        for N in N_vals:
            results = {}
            
            for provider in providers:
                # Log memory usage before benchmark
                memory_before = torch.cuda.memory_allocated() / (1024**3)
                logger.debug(f"Before {provider} backward benchmark (N={N}): {memory_before:.2f} GB allocated")
                
                logger.debug(f"Backward benchmark: N={N}, H={H}, D={D}, provider={provider}")
                q = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
                k = torch.randn((N, H // 4, D), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((N, H // 4, D), device="cuda", dtype=torch.bfloat16)
                o = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
                do = torch.randn((N, H, D), device="cuda", dtype=torch.bfloat16)
                lse = torch.randn((N, H), device="cuda", dtype=torch.bfloat16)
                sm_scale = 1 / math.sqrt(D)
                cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
                dq = torch.zeros_like(q)
                dk = torch.zeros_like(k)
                dv = torch.zeros_like(v)
                
                # Generate topk indices for sparse attention
                topk = 16
                top_idx = generate_topk_idx_example(cu_seqlens[1:], 64, topk, H // 4)

                try:
                    if provider == "flash":
                        logger.debug(f"Running flash-attention backward benchmark with N={N}")
                        start_time = time.time()
                        ms = bench(
                            lambda: _flash_attn_varlen_backward(
                                do, q, k, v, o, lse.transpose(0, 1), dq, dk, dv,
                                cu_seqlens, cu_seqlens, N, N, dropout_p=0.0, causal=True,
                                softmax_scale=sm_scale, window_size_left=-1, window_size_right=-1,
                                softcap=0.0, alibi_slopes=None, deterministic=False,
                            )
                        )
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
                                q_bench, k_bench, v_bench, top_idx, 64, cu_seqlens, sm_scale
                            )
                            # Backward pass
                            out.backward(do, retain_graph=True)
                            
                        ms = bench(run_forward_backward)
                        logger.debug(f"Completed topk-flash-attention backward benchmark in {time.time() - start_time:.2f}s")
                    
                    results[provider] = ms
                finally:
                    # Clean up all tensors
                    del q, k, v, o, do, lse, cu_seqlens, dq, dk, dv, top_idx
                    if provider == "topk-flash":
                        try:
                            del q_bench, k_bench, v_bench
                        except:
                            pass
                    # Force garbage collection before emptying cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Log memory usage after cleanup
                    memory_after = torch.cuda.memory_allocated() / (1024**3)
                    logger.debug(f"After {provider} backward benchmark (N={N}): {memory_after:.2f} GB allocated")
                    logger.debug(f"Memory cleaned up: {memory_before - memory_after:.2f} GB")
            
            print(f"{N:<10} {results.get('flash', 'N/A'):<15.2f} {results.get('topk-flash', 'N/A'):<15.2f}")

    logger.debug("Starting backward benchmark runs")
    benchmark_backward()
    logger.debug("Completed backward benchmark runs")

    # benchmark batch sizes with fixed sequence length
    logger.debug("Setting up batch size benchmark")
    def benchmark_batch_sizes():
        B_vals = [1, 2, 4, 8, 16]
        N = 8192
        H = 32
        D = 128
        providers = ["flash", "topk-flash"]
        
        print(f"\n** Batch size performance comparison with seq length {N} **")
        print(f"{'B':<10} {'Flash (ms)':<15} {'TopK-Flash (ms)':<15}")
        print("-" * 40)
        
        for B in B_vals:
            results = {}
            # Log memory before batch test
            memory_before_batch = torch.cuda.memory_allocated() / (1024**3)
            logger.debug(f"Before batch B={B} benchmark: {memory_before_batch:.2f} GB allocated")
            
            # Clear CUDA cache before creating new tensors
            torch.cuda.empty_cache()
            
            for provider in providers:
                # Log memory before provider test
                memory_before = torch.cuda.memory_allocated() / (1024**3)
                logger.debug(f"Before {provider} batch benchmark (B={B}): {memory_before:.2f} GB allocated")
                
                # Total number of tokens across all batches
                total_tokens = B * N
                
                # Create cumulative sequence lengths for batched input
                cu_seqlens = torch.zeros(B+1, device="cuda", dtype=torch.int32)
                for i in range(B):
                    cu_seqlens[i+1] = cu_seqlens[i] + N
                
                sm_scale = 1 / math.sqrt(D)
                
                # Create input tensors
                q = torch.randn((total_tokens, H, D), device="cuda", dtype=torch.bfloat16)
                k = torch.randn((total_tokens, H // 4, D), device="cuda", dtype=torch.bfloat16)
                v = torch.randn((total_tokens, H // 4, D), device="cuda", dtype=torch.bfloat16)
                
                # Parameters for topk sparse attention
                block_size = 64
                topk = 16
                
                # Generate topk indices for sparse attention
                top_idx = generate_topk_idx_example(torch.ones(B, device="cuda", dtype=torch.int32) * N, 
                                                   block_size, topk, H // 4)
                
                try:
                    if provider == "flash":
                        logger.debug(f"Running flash-attention benchmark with B={B}, N={N}")
                        start_time = time.time()
                        ms = bench(
                            lambda: _flash_attn_varlen_forward(
                                q, k, v, cu_seqlens, cu_seqlens, total_tokens, total_tokens,
                                dropout_p=0.0, causal=True, softmax_scale=sm_scale,
                            )
                        )
                        logger.debug(f"Completed flash-attention benchmark in {time.time() - start_time:.2f}s")
                    elif provider == "topk-flash":
                        logger.debug(f"Running topk-flash-attention benchmark with B={B}, N={N}")
                        start_time = time.time()
                        ms = bench(
                            lambda: topk_sparse_attention_flash(
                                q, k, v, top_idx, block_size, cu_seqlens, sm_scale
                            )
                        )
                        logger.debug(f"Completed topk-flash-attention benchmark in {time.time() - start_time:.2f}s")
                    
                    results[provider] = ms
                finally:
                    # Clean up all tensors
                    del q, k, v, cu_seqlens, top_idx
                    # Force garbage collection before emptying cache
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Log memory usage after cleanup
                    memory_after = torch.cuda.memory_allocated() / (1024**3)
                    logger.debug(f"After {provider} batch benchmark (B={B}): {memory_after:.2f} GB allocated")
                    logger.debug(f"Memory cleaned up: {memory_before - memory_after:.2f} GB")
            
            print(f"{B:<10} {results.get('flash', 'N/A'):<15.2f} {results.get('topk-flash', 'N/A'):<15.2f}")
            
            # Log memory after batch test
            memory_after_batch = torch.cuda.memory_allocated() / (1024**3)
            logger.debug(f"After batch B={B} benchmark: {memory_after_batch:.2f} GB allocated")
            logger.debug(f"Total memory cleaned up for batch: {memory_before_batch - memory_after_batch:.2f} GB")
            logger.debug("-" * 30)

    logger.debug("Starting batch size benchmark runs")
    benchmark_batch_sizes()
    logger.debug("Completed batch size benchmark runs and script execution")

 