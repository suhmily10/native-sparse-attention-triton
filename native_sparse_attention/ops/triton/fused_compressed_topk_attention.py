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
import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .conv_compress import conv_compress
from .topk_sparse_attention import topk_sparse_attention
from .compressed_attention import CompressedAttention, transform_score, _get_attention_score


def fused_compressed_topk_attention(
    q: torch.Tensor,  # [total_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_kv_heads, head_dim]
    compress_key: torch.Tensor,  # Compression parameters for key
    compress_value: torch.Tensor,  # Compression parameters for value
    intra_block_pe: torch.Tensor,  # Positional encoding
    kernel_size: int,
    kernel_stride: int, 
    block_size: int,
    topk: int,
    cu_seqlens: torch.Tensor,
    max_seqlen: Optional[int] = None,
    sm_scale: Optional[float] = None,
    init_blocks: int = 1,
    local_blocks: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combined function that performs both compressed attention and topk sparse attention.
    
    This function first applies compression to the key and value tensors, then computes
    the topk indices, and finally performs sparse attention using these indices.
    
    Args:
        q: Query tensor [total_len, num_q_heads, head_dim]
        k: Key tensor [total_len, num_kv_heads, head_dim]
        v: Value tensor [total_len, num_kv_heads, head_dim]
        compress_key: Key compression weights
        compress_value: Value compression weights
        intra_block_pe: Positional encoding for intra-block
        kernel_size: Kernel size for compression
        kernel_stride: Stride for compression
        block_size: Block size for sparse attention
        topk: Number of top key blocks to attend to
        cu_seqlens: Cumulative sequence lengths
        max_seqlen: Maximum sequence length (optional)
        sm_scale: Softmax scale (optional)
        init_blocks: Number of initial blocks to include
        local_blocks: Number of local blocks to include
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Output tensor after fused attention
            - Topk indices for potential reuse
    """
    # Calculate max_seqlen if not provided
    if max_seqlen is None:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    
    # 1. Compress key and value tensors
    compressed_k, compressed_cu_seqlens = conv_compress(
        k,
        compress_key,
        cu_seqlens,
        kernel_size,
        kernel_stride,
        intra_block_pe,
    )
    compressed_v, _ = conv_compress(
        v,
        compress_value,
        cu_seqlens,
        kernel_size,
        kernel_stride,
        None,
    )
    
    # Get dimensions for compressed sequences
    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    max_compressed_seqlen = compressed_seqlens.max().item()
    
    # 2. Compute compressed attention and get topk indices
    compressed_attn_output, lse = CompressedAttention.apply(
        q,
        compressed_k,
        compressed_v,
        kernel_size,
        kernel_stride,
        cu_seqlens,
        compressed_cu_seqlens,
        max_seqlen,
        max_compressed_seqlen,
        sm_scale,
    )
    
    # Generate topk indices
    with torch.no_grad():
        # Compute attention scores for topk selection
        score = _get_attention_score(
            q,
            compressed_k,
            lse,
            kernel_size,
            kernel_stride,
            cu_seqlens,
            compressed_cu_seqlens,
            max_seqlen,
            max_compressed_seqlen,
            sm_scale,
        )
        
        # Transform score to block-wise score
        score = transform_score(
            score,
            kernel_size,
            kernel_stride,
            block_size,
            cu_seqlens,
            compressed_cu_seqlens,
            max_seqlen,
            max_compressed_seqlen,
            init_blocks,
            local_blocks,
        )
        
        # Get topk indices based on score
        batch_size = cu_seqlens.shape[0] - 1
        q_idx = torch.cat(
            [
                torch.arange(cu_seqlens[i + 1] - cu_seqlens[i], device=q.device)
                for i in range(batch_size)
            ],
            dim=0,
        )
        q_idx = q_idx // block_size
        topk = min(topk, score.shape[-1])
        topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1  # Apply causal masking
        topk_idx = topk_idx.to(torch.int32)
    
    # 3. Apply the sparse attention using topk indices
    sparse_attn_output = topk_sparse_attention(
        q, k, v, topk_idx, block_size, cu_seqlens, sm_scale
    )
    
    return sparse_attn_output, topk_idx 