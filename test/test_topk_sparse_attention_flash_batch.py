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
import logging
import time
import gc
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
    
    # benchmark batch sizes with fixed sequence length
    logger.debug("Setting up batch size benchmark")
    def benchmark_batch_sizes():
        B_vals = [1, 2]
        N = 16384
        H = 8
        D = 96
        providers = ["flash", "topk-flash"]
        
        print("\n** Batch size performance comparison with seq length 4096 **")
        print(f"{'B':<10} {'Flash (ms)':<15} {'TopK-Flash (ms)':<15}")
        print("-" * 40)
        
        for B in B_vals:
            # Add gc at the beginning of each batch size iteration
            gc.collect()
            torch.cuda.empty_cache()
            
            results = {}
            logger.debug(f"Batch size benchmark: B={B}, N={N}, H={H}, D={D}")
            
            # Clear CUDA cache before creating new tensors
            torch.cuda.empty_cache()
            logger.debug(f"Starting benchmark B={B}, memory: {torch.cuda.memory_reserved()//1024**3} GB")
            
            for provider in providers:
                # Add gc at the beginning of each provider benchmark
                gc.collect()
                torch.cuda.empty_cache()
                
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
                
                # Add additional gc after each provider benchmark is complete
                gc.collect()
                torch.cuda.empty_cache()
            
            print(f"{B:<10} {results.get('flash', 'N/A'):<15.2f} {results.get('topk-flash', 'N/A'):<15.2f}")
            # Add final gc after each batch size iteration is complete
            gc.collect()
            torch.cuda.empty_cache()

    logger.debug("Starting batch size benchmark runs")
    # Add gc before starting benchmarks
    gc.collect()
    torch.cuda.empty_cache()
    benchmark_batch_sizes()
    # Add gc after all benchmarks complete
    gc.collect()
    torch.cuda.empty_cache()
    logger.debug("Completed batch size benchmark runs and script execution")

 