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
    total_seqlen, num_kv_heads, head_dim = k.shape
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
    
    # Process each batch and head separately
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        batch_len = end - start
        
        for h_kv in range(num_kv_heads):
            # Gather the relevant keys and values for this batch and head
            batch_topk_idx = topk_idx[h_kv, start:end]  # [batch_len, topk]
            
            # For each query position, compute attention with its selected key blocks
            for j in range(batch_len):
                valid_indices = batch_topk_idx[j] != -1
                if not valid_indices.any():
                    continue
                
                valid_block_indices = batch_topk_idx[j, valid_indices]
                
                # Create a mask for all keys in this sequence
                key_mask = torch.zeros(batch_len, dtype=torch.bool, device=q.device)
                
                # Fill in the mask for each valid block
                for block_idx in valid_block_indices:
                    block_start = block_idx * block_size
                    block_end = min(block_start + block_size, batch_len)
                    if block_start < batch_len:
                        key_mask[block_start:block_end] = True
                
                if not key_mask.any():
                    continue
                
                # Get indices of keys to attend to
                key_indices = start + torch.nonzero(key_mask, as_tuple=True)[0]
                
                # Get the keys and values for these indices
                k_selected = k[key_indices, h_kv].unsqueeze(0)  # [1, num_keys, head_dim]
                v_selected = v[key_indices, h_kv].unsqueeze(0)  # [1, num_keys, head_dim]
                
                # Compute attention for this query across all heads that share this kv head
                q_j = q[start + j, h_kv*num_share_q_heads:(h_kv+1)*num_share_q_heads]  # [num_share_q_heads, head_dim]
                
                # Calculate attention scores
                attn_scores = torch.matmul(q_j, k_selected.transpose(-1, -2)) * softmax_scale  # [num_share_q_heads, num_keys]
                
                # Apply softmax in float32 for numerical stability, then convert back
                attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
                
                # Apply attention weights to values
                attn_output = torch.matmul(attn_weights, v_selected)  # [num_share_q_heads, head_dim]
                
                # Store the output
                o[start + j, h_kv*num_share_q_heads:(h_kv+1)*num_share_q_heads] = attn_output
    
    return o
