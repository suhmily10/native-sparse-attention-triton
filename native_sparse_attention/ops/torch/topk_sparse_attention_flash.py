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
    """Simple topk sparse attention varlen version implemented in torch. Extremely slow, only for debugging.

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
    
    # Process all KV heads
    for h_kv in range(num_kv_heads):
        # Get all blocks for this head
        head_topk_idx = topk_idx[h_kv]  # [total_len, topk]
        
        # Get corresponding q heads for this kv head
        q_heads_slice = slice(h_kv * num_share_q_heads, (h_kv + 1) * num_share_q_heads)
        q_heads = q[:, q_heads_slice]  # [total_len, num_share_q_heads, head_dim]
        
        # Find which batch each query belongs to
        batch_indices = torch.zeros(total_seqlen, dtype=torch.long, device=q.device)
        for b in range(batch_size):
            batch_indices[cu_seqlens[b]:cu_seqlens[b+1]] = b
        
        # Process each query position
        for q_pos in range(total_seqlen):
            q_blocks = head_topk_idx[q_pos]  # [topk]
            valid_blocks = q_blocks[q_blocks != -1]  # [num_valid_blocks]
            
            if valid_blocks.numel() == 0:
                continue
            
            # Get batch information for this query
            batch_idx = batch_indices[q_pos]
            batch_start = cu_seqlens[batch_idx].item()
            batch_end = cu_seqlens[batch_idx + 1].item()
            
            # Calculate key indices for all valid blocks
            block_starts = valid_blocks * block_size
            block_ends = (valid_blocks + 1) * block_size
            
            # Create a mask for keys in the current batch
            key_mask = torch.zeros(batch_end - batch_start, dtype=torch.bool, device=q.device)
            
            # Mark all keys from valid blocks
            for b_start, b_end in zip(block_starts, block_ends):
                b_start = max(0, b_start.item())
                b_end = min(batch_end - batch_start, b_end.item())
                if b_start < b_end:
                    key_mask[b_start:b_end] = True
            
            # Get actual key indices
            key_indices = batch_start + torch.nonzero(key_mask, as_tuple=True)[0]
            
            if key_indices.numel() == 0:
                continue
                
            # Extract keys and values for this query
            k_selected = k[key_indices, h_kv]  # [num_keys, head_dim]
            v_selected = v[key_indices, h_kv]  # [num_keys, head_dim]
            
            # Calculate attention scores
            q_current = q_heads[q_pos]  # [num_share_q_heads, head_dim]
            attn_scores = torch.matmul(q_current, k_selected.transpose(0, 1)) * softmax_scale  # [num_share_q_heads, num_keys]
            
            # Apply softmax and compute weighted sum
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v_selected)  # [num_share_q_heads, head_dim]
            
            # Store the result
            o[q_pos, q_heads_slice] = attn_output
    
    return o
