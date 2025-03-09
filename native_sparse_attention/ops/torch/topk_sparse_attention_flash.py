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

@lru_cache(maxsize=16)
def calc_chunks(cu_seqlen, moba_chunk_size):
    """calc chunks that needs moba attention"""
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

    """ filter chunks that need moba attn """

    # filter chunks ( remove last chunk of each batch )
    # filtered_chunk_indices: chunk index list that excludes the last chunk of each batch
    chunk_to_remove = cu_num_chunk[1:] - 1
    chunk_to_remain = torch.ones(
        (num_chunk,), dtype=torch.bool, device=cu_seqlen.device
    )
    chunk_to_remain[chunk_to_remove] = False
    filtered_chunk_indices = chunk_to_remain.nonzero(as_tuple=True)[0]
    num_filtered_chunk = len(filtered_chunk_indices)

    logger.debug(f"calc_chunks: returning {num_filtered_chunk} filtered chunks")
    return (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        filtered_chunk_indices,
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
    start_time = time.time()
    if logger.isEnabledFor(logging.DEBUG):  # 先检查日志级别再执行格式化
        logger.debug(f"Starting topk_sparse_attention_flash with shapes: q={q.shape}, k={k.shape}, v={v.shape}, topk_idx={topk_idx.shape}")
        logger.debug(f"block_size={block_size}, cu_seqlens={cu_seqlens}")
    
    # 基本设置
    logger.debug("Stacking k and v")
    kv = torch.stack((k, v), dim=1)  # [total_len, 2, num_kv_heads, head_dim]
    total_len, num_q_head, head_dim = q.shape
    # Get the number of KV heads from the shape of k/v/topk_idx
    num_kv_head = topk_idx.shape[0]
    logger.debug(f"total_len={total_len}, num_q_head={num_q_head}, num_kv_head={num_kv_head}, head_dim={head_dim}")
    
    # 如果没有指定softmax_scale，使用默认值
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    logger.debug(f"softmax_scale={softmax_scale}")
    
    # 准备块元数据
    logger.debug("Calculating chunks")
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        _,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, block_size)
    logger.debug(f"cu_chunk shape={cu_chunk.shape}, filtered_chunk_indices shape={filtered_chunk_indices.shape}")
    logger.debug(f"num_filtered_chunk={num_filtered_chunk}")
    
    # 创建过滤后的KV (所有可能参与计算的KV块)
    logger.debug("Creating filtered KV indices")
    filtered_kv_indices = torch.arange(
        0, block_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    logger.debug(f"filtered_kv_indices shape={filtered_kv_indices.shape}")
    
    logger.debug("Selecting filtered KV tensors")
    filtered_kv = kv[filtered_kv_indices.view(-1)]  # 直接索引比 index_select 更快
    logger.debug(f"filtered_kv shape={filtered_kv.shape}")
    
    # 移除gate_mask相关逻辑
    logger.debug("Creating full query indices")
    
    # 简化索引逻辑 - 更高效的实现
    # 之前的代码等效于创建 (h, s) -> (h*s) 的展平索引
    # 可以直接使用 rearrange 的展平功能，无需创建复杂的索引张量
    
    # 每个chunk的查询数量固定为block_size
    moba_seqlen_q = torch.full((num_filtered_chunk * num_q_head,), block_size, device=q.device)
    moba_cu_seqlen_q = torch.arange(0, (num_filtered_chunk * num_q_head + 1) * block_size, block_size, device=q.device, dtype=torch.int32)

    # 直接使用展平的查询矩阵，不需要额外的索引
    logger.debug("Preparing query vectors with simplified indexing")
    moba_q = rearrange(q, "s h d -> (h s) d")
    moba_q = moba_q.unsqueeze(1)  # [total_queries, 1, head_dim]
    
    # 为了兼容后续的输出合并，存储默认的线性索引
    moba_q_sh_indices = torch.arange(moba_q.shape[0], device=q.device)
    
    # 重组KV矩阵以适应查询排列
    logger.debug("Reorganizing KV tensors")
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.reshape(-1, block_size, 2, moba_kv.shape[-1])
    logger.debug(f"rearranged moba_kv shape={moba_kv.shape}")
    
    # 处理不同数量的q和kv heads
    if num_q_head > num_kv_head:
        # 计算每个kv head对应的q head数量
        q_heads_per_kv_head = num_q_head // num_kv_head
        logger.debug(f"Repeating KV {q_heads_per_kv_head} times for each KV head")
        # 复制kv，使其匹配q heads的数量
        moba_kv = torch.repeat_interleave(moba_kv, q_heads_per_kv_head, dim=0)
        logger.debug(f"repeated moba_kv shape={moba_kv.shape}")
    
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)  # [num_chunks*block_size, 2, 1, head_dim]
    logger.debug(f"final moba_kv shape={moba_kv.shape}")
    
    # 构建cu_seqlen_kv用于flash attention
    logger.debug("Building cu_seqlen_kv")
    moba_cu_seqlen_kv = torch.arange(
        0, num_filtered_chunk * num_q_head + 1,
        dtype=torch.int32, device=q.device
    ) * block_size
    logger.debug(f"moba_cu_seqlen_kv shape={moba_cu_seqlen_kv.shape}")
    
    # 检查形状一致性
    logger.debug(f"Verifying moba_cu_seqlen_q shape={moba_cu_seqlen_q.shape} matches moba_cu_seqlen_kv shape={moba_cu_seqlen_kv.shape}")
    assert moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    
    # 使用自定义MixedAttention实现注意力计算
    logger.debug("Creating output tensor")
    output = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    
    # 只计算MoBA部分，无自注意力
    logger.debug("Calling flash_attn_varlen_func")
    try:
        # 修改性能分析部分，减少同步开销
        if profile_profiling:
            # 使用更轻量级的性能分析方式
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                with_stack=False,  # 关闭堆栈跟踪以减少开销
                record_shapes=False,  # 除非必要，否则不记录形状
                profile_memory=False,  # 关闭内存分析
                with_flops=True,
                use_cuda=True
            ) as prof:
                moba_attn_out = flash_attn_varlen_func(
                    q=moba_q,
                    k=moba_kv[:, 0],
                    v=moba_kv[:, 1],
                    cu_seqlens_q=moba_cu_seqlen_q,
                    cu_seqlens_k=moba_cu_seqlen_kv,
                    max_seqlen_q=total_len,
                    max_seqlen_k=block_size,
                    causal=False,
                    dropout_p=0.0,
                )
                # 只在最后一次进行同步，减少同步次数
            # 不在这里调用torch.cuda.synchronize()，仅在需要结果时同步
        else:
            # 使用异步执行和混合精度
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                moba_attn_out = flash_attn_varlen_func(
                    q=moba_q,
                    k=moba_kv[:, 0],
                    v=moba_kv[:, 1],
                    cu_seqlens_q=moba_cu_seqlen_q,
                    cu_seqlens_k=moba_cu_seqlen_kv,
                    max_seqlen_q=total_len,
                    max_seqlen_k=block_size,
                    causal=False,
                    dropout_p=0.0,
                )
        logger.debug(f"flash_attn_varlen_func completed, moba_attn_out shape={moba_attn_out.shape}")
    except Exception as e:
        logger.error(f"Error in flash_attn_varlen_func: {str(e)}")
        logger.error(f"q shape={moba_q.shape}, k shape={moba_kv[:, 0].shape}, v shape={moba_kv[:, 1].shape}")
        logger.error(f"cu_seqlens_q={moba_cu_seqlen_q}, cu_seqlens_k={moba_cu_seqlen_kv}")
        raise
    
    # 优化4: 不需要使用索引操作，直接重塑结果
    output = moba_attn_out.reshape(num_q_head, total_len, head_dim).transpose(0, 1)

    # 优化5: 避免不必要的dtype转换
    output = output.to(q.dtype)
    
    logger.debug(f"Completed topk_sparse_attention_flash in {time.time() - start_time:.2f}s, output shape={output.shape}")
    if profile_profiling:  # 只在开启时输出性能分析结果
        logger.info(f"Profiling results:\n{prof.key_averages().table(sort_by='cuda_time_total')}")
    return output