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
import math
from typing import Optional


def topk_sparse_attention_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Simple topk sparse attention varlen version implemented in torch. Extremly slow, only for debugging.

    Args:
        q (torch.Tensor): shape [total_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        topk_idx (torch.Tensor): topk block idx for each query, shape [num_kv_heads, total_len, topk]. -1 means padding.
        block_size (int): key value block size.
        cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        softmax_scale (Optional[float], optional): Defaults to None, means 1/sqrt(head_dim).

    Returns:
        torch.Tensor: attention output, shape [total_len, num_q_heads, head_dim]
    """
    total_seqlen, num_q_heads, head_dim = q.shape
    _, num_kv_heads, _ = k.shape
    num_share_q_heads = num_q_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    topk = topk_idx.shape[-1]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Apply causal masking to the topk indices
    q_idx = torch.cat(
        [torch.arange(seqlens[i], device=q.device) for i in range(batch_size)], dim=0
    )
    topk_idx = topk_idx.clone()
    topk_idx[topk_idx > (q_idx // block_size)[None, :, None]] = -1
    
    # Initialize the output tensor
    o = torch.zeros_like(q)
    
    # Process all heads and queries in parallel
    for h_kv in range(num_kv_heads):
        # Get all valid blocks for this head
        head_topk_idx = topk_idx[h_kv]  # [total_len, topk]
        
        # Find all valid queries and their corresponding blocks
        # A query is valid if it has at least one valid block to attend to
        valid_q_mask = (head_topk_idx != -1).any(dim=1)  # [total_len]
        if not valid_q_mask.any():
            continue
        
        valid_q_indices = torch.nonzero(valid_q_mask, as_tuple=True)[0]  # [num_valid_queries]
        
        # For each valid query, gather its corresponding keys and values
        for batch_idx in range(batch_size):
            batch_start = cu_seqlens[batch_idx].item()
            batch_end = cu_seqlens[batch_idx + 1].item()
            
            # Find valid queries in this batch
            batch_mask = (valid_q_indices >= batch_start) & (valid_q_indices < batch_end)
            if not batch_mask.any():
                continue
            
            batch_q_indices = valid_q_indices[batch_mask]  # [num_valid_batch_queries]
            batch_q = q[batch_q_indices, h_kv*num_share_q_heads:(h_kv+1)*num_share_q_heads]  # [num_valid_batch_queries, num_share_q_heads, head_dim]
            
            # For each query in batch, collect all keys it should attend to
            for q_idx, global_q_idx in enumerate(batch_q_indices):
                local_q_idx = global_q_idx - batch_start
                
                # Get blocks for this query
                q_blocks = head_topk_idx[global_q_idx]
                valid_blocks = q_blocks[q_blocks != -1]
                
                if valid_blocks.numel() == 0:
                    continue
                
                # Create key mask for all valid blocks
                key_mask = torch.zeros(batch_end - batch_start, dtype=torch.bool, device=q.device)
                
                # Fill in mask for each valid block
                for block_idx in valid_blocks:
                    block_start = block_idx * block_size
                    block_end = min((block_idx + 1) * block_size, batch_end - batch_start)
                    if block_start < batch_end - batch_start:
                        key_mask[block_start:block_end] = True
                
                # Get actual key indices
                key_indices = batch_start + torch.nonzero(key_mask, as_tuple=True)[0]
                
                # Collect keys and values
                k_selected = k[key_indices, h_kv]  # [num_keys, head_dim]
                v_selected = v[key_indices, h_kv]  # [num_keys, head_dim]
                
                # Reshape for batch matrix multiplication
                q_current = batch_q[q_idx]  # [num_share_q_heads, head_dim]
                
                # Calculate attention scores - more efficient batched version
                attn_scores = torch.matmul(q_current, k_selected.transpose(0, 1)) * softmax_scale  # [num_share_q_heads, num_keys]
                
                # Apply softmax and compute weighted sum
                attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v_selected)  # [num_share_q_heads, head_dim]
                
                # Store the result
                o[global_q_idx, h_kv*num_share_q_heads:(h_kv+1)*num_share_q_heads] = attn_output
    
    return o
