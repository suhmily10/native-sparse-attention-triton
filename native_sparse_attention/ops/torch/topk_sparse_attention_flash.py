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
import logging  # Add logging
import time     # Add time for timestamps
from typing import Optional
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from einops import rearrange
from functools import lru_cache

# Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%H:%M:%S'
# )
logger = logging.getLogger(__name__)

@lru_cache(maxsize=16)
def calc_topk_chunks(topk_idx: torch.Tensor, block_size: int):
    """Calculate and cache chunk information for topk sparse attention"""
    unique_chunks = torch.unique(topk_idx[topk_idx >= 0])
    num_chunk = unique_chunks.size(0)
    
    # Calculate chunk indices for KV selection
    chunk_indices = (unique_chunks[:, None] * block_size + 
                    torch.arange(0, block_size, device=topk_idx.device)).flatten()
    
    # Calculate cumulative sequence lengths for chunks
    cu_seqlen = torch.arange(0, (num_chunk + 1), device=topk_idx.device, dtype=torch.int32) * block_size
    
    return chunk_indices, cu_seqlen

def topk_sparse_attention_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,  # 形状 [num_kv_heads, total_len, topk]
    block_size: int,
    cu_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    profile_profiling: bool = False,  # 新增性能分析开关
) -> torch.Tensor:
    """使用预计算的topk索引实现的稀疏注意力计算
    
    Args:
        q (torch.Tensor): [total_len, num_q_heads, head_dim]
        k (torch.Tensor): [total_len, num_kv_heads, head_dim]
        v (torch.Tensor): [total_len, num_kv_heads, head_dim]
        topk_idx (torch.Tensor): 每个查询的topk块索引，形状 [num_kv_heads, total_len, topk]，-1表示填充
        block_size (int): key-value块大小
        cu_seqlens (torch.Tensor): 形状 [batch_size + 1]，与flash_attn中的cu_seqlens相似
        softmax_scale (Optional[float]): 默认为None，表示1/sqrt(head_dim)
        profile_profiling (bool): 是否进行性能分析，默认为False
        
    Returns:
        torch.Tensor: 注意力输出，形状 [total_len, num_q_heads, head_dim]
    """

    # 合并变量定义和基本设置
    total_len, num_q_head, head_dim = q.shape
    num_kv_head = topk_idx.shape[0]
    softmax_scale = softmax_scale or head_dim ** (-0.5)
    
    # Use cached chunk calculations
    chunk_indices, moba_cu_seqlen = calc_topk_chunks(topk_idx, block_size)
    
    # Efficient KV selection using pre-calculated indices
    filtered_kv = torch.stack((k, v), dim=1)[chunk_indices]
    
    # Optimize query and KV matrix transformations
    moba_q = rearrange(q, 't h d -> (h t) 1 d')
    moba_kv = rearrange(filtered_kv, '(c b) pair h d -> (c h) b pair d', b=block_size)
    moba_kv = rearrange(moba_kv, 'n b pair d -> (n b) pair 1 d')
    
    # 处理不同数量的q和kv heads (修改后)
    # Flash Attention 2+ 原生支持GQA，直接传递原始head数量即可
    if num_q_head != num_kv_head:
        assert num_q_head % num_kv_head == 0, "q_heads must be multiple of kv_heads for GQA"
        # 不再需要重复interleave操作
    
    # 使用flash attention计算输出

    # 使用异步执行和混合精度
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        out = flash_attn_varlen_func(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen,
            cu_seqlens_k=moba_cu_seqlen,
            max_seqlen_q=total_len,
            max_seqlen_k=block_size,
            causal=False,
            dropout_p=0.0,
        )

    
    # 优化返回结果转换
    return rearrange(out, '(h t) 1 d -> t h d', h=num_q_head).to(q.dtype)  # 合并reshape和transpose