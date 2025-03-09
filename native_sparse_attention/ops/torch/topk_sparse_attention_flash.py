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
    """Simple topk sparse attention varlen version implemented in torch. Optimized version.

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

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    

    
    # Initialize the output tensor
    o = torch.zeros_like(q)
    
    # Process each KV head
    for h_kv in range(num_kv_heads):
        # Get corresponding q heads for this kv head
        q_heads_slice = slice(h_kv * num_share_q_heads, (h_kv + 1) * num_share_q_heads)
        q_heads = q[:, q_heads_slice]  # [total_len, num_share_q_heads, head_dim]
        
        # Get all blocks for this head
        head_topk_idx = topk_idx[h_kv]  # [total_len, topk]
        
        # Create a mask of valid blocks (non-padding)
        valid_block_mask = head_topk_idx != -1  # [total_len, topk]
        
        # Skip if no valid blocks for this head
        if not valid_block_mask.any():
            continue
        
        # Find which batch each query belongs to
        batch_indices = torch.zeros(total_seqlen, dtype=torch.long, device=q.device)
        for b in range(batch_size):
            batch_indices[cu_seqlens[b]:cu_seqlens[b+1]] = b
        
        # Gather all valid q positions and their corresponding block indices
        q_positions, block_positions = torch.nonzero(valid_block_mask, as_tuple=True)
        
        # Unique query positions (each query may attend to multiple blocks)
        unique_q_positions, q_counts = torch.unique(q_positions, return_counts=True)
        
        # Get the valid block indices
        valid_block_indices = head_topk_idx[q_positions, block_positions]
        
        # Process q positions in batches
        start_idx = 0
        for q_pos in unique_q_positions:
            # Get batch info for this query
            batch_idx = batch_indices[q_pos]
            batch_start = cu_seqlens[batch_idx].item()
            
            # Get all block indices for this query position
            count = q_counts[unique_q_positions == q_pos].item()
            q_block_indices = valid_block_indices[start_idx:start_idx+count]
            start_idx += count
            
            # Calculate key indices for all blocks
            key_indices = []
            for block_idx in q_block_indices:
                block_start = batch_start + block_idx * block_size
                block_end = min(batch_start + (block_idx + 1) * block_size, cu_seqlens[batch_idx + 1])
                key_indices.extend(range(block_start, block_end))
            
            if not key_indices:
                continue
                
            key_indices = torch.tensor(key_indices, device=q.device)
            
            # Extract keys and values
            k_selected = k[key_indices, h_kv]  # [num_keys, head_dim]
            v_selected = v[key_indices, h_kv]  # [num_keys, head_dim]
            
            # Calculate attention scores
            q_current = q_heads[q_pos]  # [num_share_q_heads, head_dim]
            attn_scores = torch.matmul(q_current, k_selected.transpose(0, 1)) * softmax_scale
            
            # Apply softmax and compute weighted sum
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v_selected)
            
            # Store the result
            o[q_pos, q_heads_slice] = attn_output
    
    return o
