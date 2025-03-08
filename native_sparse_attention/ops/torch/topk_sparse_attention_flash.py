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
    
    # Find which batch each query belongs to
    batch_indices = torch.zeros(total_seqlen, dtype=torch.long, device=q.device)
    for b in range(batch_size):
        batch_indices[cu_seqlens[b]:cu_seqlens[b+1]] = b
    
    # Process all queries and all KV heads in parallel
    h_kv_indices = torch.arange(num_kv_heads, device=q.device)
    q_pos_indices = torch.arange(total_seqlen, device=q.device)
    
    # Create a query position to batch mapping
    q_to_batch = batch_indices
    batch_starts = cu_seqlens[q_to_batch]
    batch_ends = cu_seqlens[q_to_batch + 1]
    
    # Create masks for sparse attention (total_len, num_kv_heads, total_len)
    attn_mask = torch.zeros(total_seqlen, num_kv_heads, total_seqlen, 
                            dtype=torch.bool, device=q.device)
    
    # For each query position and kv head, create a mask for the keys to attend to
    for h_kv in range(num_kv_heads):
        head_topk_idx = topk_idx[h_kv]  # [total_len, topk]
        
        for q_pos in range(total_seqlen):
            q_blocks = head_topk_idx[q_pos]  # [topk]
            valid_blocks = q_blocks[q_blocks != -1]  # [num_valid_blocks]
            
            if valid_blocks.numel() == 0:
                continue
                
            # Get batch information for this query
            batch_idx = q_to_batch[q_pos]
            batch_start = batch_starts[q_pos].item()
            batch_end = batch_ends[q_pos].item()
            
            # Calculate key indices for all valid blocks
            block_starts = valid_blocks * block_size
            block_ends = torch.minimum((valid_blocks + 1) * block_size, 
                                       torch.tensor(batch_end - batch_start, device=q.device))
            
            # Mark keys that should be attended to
            for b_start, b_end in zip(block_starts, block_ends):
                b_start = max(0, b_start.item())
                b_end = b_end.item()
                if b_start < b_end:
                    attn_mask[q_pos, h_kv, batch_start + b_start:batch_start + b_end] = True
    
    # Compute attention using the masks
    o_new = torch.zeros_like(q)
    
    for q_pos in range(total_seqlen):
        for h_kv in range(num_kv_heads):
            # Get corresponding q heads for this kv head
            q_heads_slice = slice(h_kv * num_share_q_heads, (h_kv + 1) * num_share_q_heads)
            q_current = q[q_pos, q_heads_slice]  # [num_share_q_heads, head_dim]
            
            # Get the mask for this query and kv head
            key_mask = attn_mask[q_pos, h_kv]
            key_indices = torch.nonzero(key_mask, as_tuple=True)[0]
            
            if key_indices.numel() == 0:
                continue
                
            # Extract keys and values
            k_selected = k[key_indices, h_kv]  # [num_keys, head_dim]
            v_selected = v[key_indices, h_kv]  # [num_keys, head_dim]
            
            # Calculate attention scores
            attn_scores = torch.matmul(q_current, k_selected.transpose(0, 1)) * softmax_scale
            
            # Apply softmax and compute weighted sum
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v_selected)  # [num_share_q_heads, head_dim]
            
            # Store the result
            o_new[q_pos, q_heads_slice] = attn_output
    
    return o_new
