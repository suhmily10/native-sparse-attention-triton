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
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)


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
    topk = topk_idx.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    

    
    # Initialize the output tensor
    o = torch.zeros_like(q)
    
    # 生成块掩码并重组张量
    batch_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=q.device),
        cu_seqlens[1:] - cu_seqlens[:-1]
    )
    
    # 生成块偏移量 [batch_size, max_blocks+1]
    max_blocks = (cu_seqlens[1:] - cu_seqlens[:-1] + block_size - 1) // block_size
    block_offsets = torch.cat([torch.zeros(1, device=q.device)] + [
        torch.arange(0, seqlen, block_size, device=q.device) 
        for seqlen in (cu_seqlens[1:] - cu_seqlens[:-1])
    ])
    
    # 重组KV数据 [num_blocks, num_kv_heads, block_size, head_dim]
    k_blocks = k.unfold(0, block_size, block_size)
    v_blocks = v.unfold(0, block_size, block_size)
    
    # 根据topk_idx选择块 [num_kv_heads, total_len, topk] -> [num_kv_heads, total_len*topk]
    valid_mask = topk_idx != -1
    selected_blocks = topk_idx[valid_mask]
    
    # 修正后的批次索引扩展方式
    batch_expanded = batch_indices.unsqueeze(0).unsqueeze(-1).expand_as(topk_idx)[valid_mask]
    
    # 计算实际块索引
    block_starts = block_offsets[batch_expanded] + selected_blocks * block_size
    block_indices = (block_starts.unsqueeze(-1) + torch.arange(block_size, device=q.device)).long()
    
    # 收集选中的KV块 [num_selected_blocks, num_kv_heads, block_size, head_dim]
    k_selected = k[block_indices.clamp_max(k.size(0)-1)]
    v_selected = v[block_indices.clamp_max(v.size(0)-1)]
    
    # 重组为批量计算形状 [num_kv_heads, total_len, topk, block_size, head_dim]
    k_selected = k_selected.view(num_kv_heads, total_seqlen, topk, block_size, head_dim)
    v_selected = v_selected.view(num_kv_heads, total_seqlen, topk, block_size, head_dim)
    
    # 扩展Q张量用于批量计算 [num_kv_heads, total_len, num_share_q_heads, head_dim]
    q_expanded = q.view(total_seqlen, num_kv_heads, num_share_q_heads, head_dim)
    
    # Reshape q_expanded to align with k_selected
    # [total_len, num_kv_heads, num_share_q_heads, head_dim] -> [1, total_len, num_kv_heads, num_share_q_heads, head_dim]
    q_expanded = q_expanded.unsqueeze(0)
    # import pdb; pdb.set_trace()
    # Compute attention scores
    # Note: We compute dot product between query vectors and key vectors along the head_dim dimension
    attn_scores = torch.einsum('blkhd,blnsd->blknhs', q_expanded, k_selected) * softmax_scale
    
    # 计算注意力权重并加权求和
    attn_weights = torch.softmax(attn_scores, dim=-1)
    o = torch.einsum('blknhs,blnsd->blkhd', attn_weights, v_selected)
    
    # 重组输出张量 [total_len, num_q_heads, head_dim]
    return o.view(total_seqlen, num_q_heads, head_dim)
