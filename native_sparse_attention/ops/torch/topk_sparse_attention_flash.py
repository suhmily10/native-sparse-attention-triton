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

@lru_cache(maxsize=128)
def calc_chunks(cu_seqlen, moba_chunk_size, topk_idx=None):
    """calc chunks for moba attention with option to filter by topk indices
    
    Args:
        cu_seqlen: Cumulative sequence lengths
        moba_chunk_size: Size of each chunk
        topk_idx: Optional tensor of shape [num_kv_heads, total_len, topk] containing topk block indices
                 If provided, only return chunks selected by these indices
    """
    logger.debug(f"calc_chunks: cu_seqlen shape={cu_seqlen.shape}, moba_chunk_size={moba_chunk_size}")

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    # batch_num_chunk[batch_idx] = how many chunk in batch idx
    batch_num_chunk = (batch_sizes + (moba_chunk_size - 1)) // moba_chunk_size
    # cu_num_chunk[batch_idx] = first chunk id of this batch
    cu_num_chunk = torch.ones(
        batch_num_chunk.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_chunk.dtype,
    )
    cu_num_chunk[1:] = batch_num_chunk.cumsum(dim=0)
    # total chunk ( for all batch )
    num_chunk = cu_num_chunk[-1]
    # chunk_sizes[chunk_idx] = chunk_size of chunk idx
    chunk_sizes = torch.full(
        (num_chunk + 1,), moba_chunk_size, dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_sizes[0] = 0  # for calc cu chunk
    batch_last_chunk_size = batch_sizes - (batch_num_chunk - 1) * moba_chunk_size
    chunk_sizes[cu_num_chunk[1:]] = batch_last_chunk_size
    # cu_chunk[chunk_idx] = the start chunk offset of chunk idx
    cu_chunk = chunk_sizes.cumsum(dim=-1, dtype=torch.int32)
    # chunk_to_batch[chunk_idx] = batch idx of the chunk idx
    chunk_to_batch = torch.zeros(
        (num_chunk,), dtype=torch.int32, device=cu_seqlen.device
    )
    chunk_to_batch[cu_num_chunk[1:-1]] = 1
    chunk_to_batch = chunk_to_batch.cumsum(dim=0, dtype=torch.int32)

    # Default: use all chunks
    all_chunk_indices = torch.arange(num_chunk, device=cu_seqlen.device)
    
    # If topk_idx is provided, filter chunks to only include those in topk_idx
    if topk_idx is not None:
        logger.debug(f"Filtering chunks using topk_idx with shape {topk_idx.shape}")
        # Extract unique chunk indices from topk_idx
        # Reshape to flatten all dimensions and remove padding (-1 values)
        flat_topk = topk_idx.reshape(-1)
        selected_chunks = flat_topk[flat_topk >= 0].unique()
        
        # Ensure selected_chunks are within valid range
        valid_mask = (selected_chunks < num_chunk)
        if not valid_mask.all():
            logger.warning(f"Found {(~valid_mask).sum()} invalid chunk indices in topk_idx")
            selected_chunks = selected_chunks[valid_mask]
        
        # Use selected chunks instead of all chunks
        all_chunk_indices = selected_chunks
        logger.debug(f"Selected {len(selected_chunks)} unique chunks from topk_idx")

    logger.debug(f"calc_chunks: returning {len(all_chunk_indices)} chunks")
    return (
        cu_chunk,
        all_chunk_indices,
        num_chunk,
        all_chunk_indices,
        chunk_to_batch,
    )


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
    
    # 直接从topk_idx提取唯一块索引 (减少中间变量)
    unique_chunks = topk_idx.reshape(-1)[topk_idx.reshape(-1) >= 0].unique()
    num_chunk = len(unique_chunks)
    
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
    
    # 处理不同数量的q和kv heads
    if num_q_head > num_kv_head:
        moba_kv = torch.repeat_interleave(moba_kv, num_q_head // num_kv_head, dim=0)
    
    # 准备最终KV格式
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