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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)



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
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Starting topk_sparse_attention_flash with shapes: q={q.shape}, k={k.shape}, v={v.shape}, topk_idx={topk_idx.shape}")
        logger.debug(f"block_size={block_size}, cu_seqlens={cu_seqlens}")
    
    # 合并变量定义和基本设置
    total_len, num_q_head, head_dim = q.shape
    num_kv_head = topk_idx.shape[0]
    softmax_scale = softmax_scale or head_dim ** (-0.5)
    
    # 高效地提取唯一块索引，减少内存占用
    mask = topk_idx >= 0
    unique_chunks = torch.unique(torch.masked_select(topk_idx, mask))
    num_chunk = unique_chunks.size(0)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Found {num_chunk} unique chunk indices")
    
    # 创建KV索引并选择相关KV (合并操作减少中间变量)
    filtered_kv = torch.stack((k, v), dim=1).index_select(
        0, 
        (unique_chunks[:, None] * block_size + torch.arange(0, block_size, device=q.device)).flatten()
    )
    
    # 创建cu_seqlen和准备查询矩阵 (减少中间变量)
    moba_cu_seqlen = torch.arange(0, (num_chunk * num_q_head + 1) * block_size, block_size, 
                                 device=q.device, dtype=torch.int32)
    
    # 准备查询和KV矩阵 (合并操作)
    moba_q = q.transpose(0, 1).reshape(-1, head_dim).unsqueeze(1)  # [num_q_head*total_len, 1, head_dim]
    
    # 重组KV矩阵 (简化转换步骤)
    moba_kv = filtered_kv.reshape(-1, 2, num_kv_head, head_dim)
    moba_kv = moba_kv.transpose(1, 2).reshape(-1, block_size, 2, head_dim)
    
    # 处理不同数量的q和kv heads (修改后)
    # Flash Attention 2+ 原生支持GQA，直接传递原始head数量即可
    if num_q_head != num_kv_head:
        assert num_q_head % num_kv_head == 0, "q_heads must be multiple of kv_heads for GQA"
        # 不再需要重复interleave操作
    
    # 准备最终KV格式 (保持原有形状)
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)  # [num_chunks*block_size, 2, 1, head_dim]
    
    # 使用flash attention计算输出
    try:
        if profile_profiling:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                with_stack=False,
                record_shapes=False,
                profile_memory=True,
                with_flops=True,
                use_cuda=True
            ) as prof:
                torch.cuda.synchronize()
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                
                start_event.record()
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
                end_event.record()
                torch.cuda.synchronize()
                logger.info(f"GPU Time: {start_event.elapsed_time(end_event):.2f} ms")
                logger.info(f"Profiling summary:\n{prof.key_averages().table(sort_by='cuda_time_total', row_limit=10)}")
        else:
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
    except Exception as e:
        logger.error(f"Error in flash_attn_varlen_func: {str(e)}")
        logger.error(f"Shapes: q={moba_q.shape}, k={moba_kv[:, 0].shape}, v={moba_kv[:, 1].shape}")
        raise
    
    # 直接重塑输出并返回 (简化类型转换)
    return out.reshape(num_q_head, total_len, head_dim).transpose(0, 1).to(q.dtype)