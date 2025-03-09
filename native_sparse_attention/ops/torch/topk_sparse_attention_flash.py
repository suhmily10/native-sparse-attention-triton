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
    logger.info(f"calc_chunks: cu_seqlen shape={cu_seqlen.shape}, moba_chunk_size={moba_chunk_size}")

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

    logger.info(f"calc_chunks: returning {num_filtered_chunk} filtered chunks")
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
        
    Returns:
        torch.Tensor: 注意力输出，形状 [total_len, num_q_heads, head_dim]
    """
    start_time = time.time()
    logger.info(f"Starting topk_sparse_attention_flash with shapes: q={q.shape}, k={k.shape}, v={v.shape}, topk_idx={topk_idx.shape}")
    logger.info(f"block_size={block_size}, cu_seqlens={cu_seqlens}")
    
    # 基本设置
    logger.info("Stacking k and v")
    kv = torch.stack((k, v), dim=1)  # [total_len, 2, num_kv_heads, head_dim]
    total_len, num_q_head, head_dim = q.shape
    # Get the number of KV heads from the shape of k/v/topk_idx
    num_kv_head = topk_idx.shape[0]
    logger.info(f"total_len={total_len}, num_q_head={num_q_head}, num_kv_head={num_kv_head}, head_dim={head_dim}")
    
    # 如果没有指定softmax_scale，使用默认值
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    logger.info(f"softmax_scale={softmax_scale}")
    
    # 准备块元数据
    logger.info("Calculating chunks")
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        _,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, block_size)
    logger.info(f"cu_chunk shape={cu_chunk.shape}, filtered_chunk_indices shape={filtered_chunk_indices.shape}")
    logger.info(f"num_filtered_chunk={num_filtered_chunk}")
    
    # 创建过滤后的KV (所有可能参与计算的KV块)
    logger.info("Creating filtered KV indices")
    filtered_kv_indices = torch.arange(
        0, block_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    logger.info(f"filtered_kv_indices shape={filtered_kv_indices.shape}")
    
    logger.info("Selecting filtered KV tensors")
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))  # [num_filtered_chunk * block_size, 2, num_kv_heads, head_dim]
    logger.info(f"filtered_kv shape={filtered_kv.shape}")
    
    # 处理topk_idx，创建注意力mask
    logger.info("Creating gate mask")
    gate_mask = torch.zeros(
        (num_filtered_chunk, num_q_head, total_len), 
        dtype=torch.bool, 
        device=q.device
    )
    logger.info(f"gate_mask shape={gate_mask.shape}")
    
    # 对每个位置，将对应的topk块在gate_mask中标记为True
    logger.info("Updating gate mask based on topk blocks")
    # Vectorized implementation with bounds checking
    h_kv = 0  # Use first KV head for indices
    valid_mask = topk_idx[h_kv] != -1  # [total_len, topk]
    valid_s, valid_k = torch.where(valid_mask)
    valid_block_indices = topk_idx[h_kv, valid_s, valid_k]  # [num_valid]
    
    # Create sorted version of filtered chunks for search
    sorted_filtered, _ = torch.sort(filtered_chunk_indices)
    # Find positions where valid blocks exist in filtered chunks
    pos = torch.searchsorted(sorted_filtered, valid_block_indices)
    # Create mask for valid positions
    valid_pos_mask = (pos < len(sorted_filtered)) & (sorted_filtered[pos] == valid_block_indices)
    
    # Get final valid indices
    filtered_idx = pos[valid_pos_mask]
    valid_s_filtered = valid_s[valid_pos_mask]
    
    # Update gate mask safely
    if filtered_idx.numel() > 0:
        gate_mask[filtered_idx, :, valid_s_filtered] = True
    
    logger.info("Finding queries that need attention")
    # 组合所有需要注意力的查询索引
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[-1]  # (head * seq) indices
    logger.info(f"moba_q_indices shape={moba_q_indices.shape}")
    
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()  # 每个(chunk,head)对应的查询数量
    logger.info(f"moba_seqlen_q shape={moba_seqlen_q.shape}, sum={moba_seqlen_q.sum().item()}")
    
    # 选择所有需要注意力的查询向量
    logger.info("Selecting query vectors")
    moba_q = rearrange(q, "s h d -> (h s) d").index_select(0, moba_q_indices)  # [selected_queries, head_dim]
    moba_q = moba_q.unsqueeze(1)  # [selected_queries, 1, head_dim]
    logger.info(f"moba_q shape={moba_q.shape}")
    
    # 记录这些查询在原始张量中的位置
    moba_q_sh_indices = moba_q_indices % total_len * num_q_head + moba_q_indices // total_len
    logger.info(f"moba_q_sh_indices shape={moba_q_sh_indices.shape}")
    
    # 过滤掉没有查询的块
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    logger.info(f"zero_expert_count={zero_expert_count}, valid_experts={valid_expert_mask.sum().item()}")
    
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
        logger.info(f"filtered moba_seqlen_q shape={moba_seqlen_q.shape}")
    
    # 构建cu_seqlen_q用于flash attention
    logger.info("Building cu_seqlen_q")
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0
    ).to(torch.int32)
    logger.info(f"moba_cu_seqlen_q shape={moba_cu_seqlen_q.shape}")
    
    # 重组KV矩阵以适应查询排列
    logger.info("Reorganizing KV tensors")
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(block_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)  # [num_filtered_chunk * num_kv_heads, block_size, 2, head_dim]
    logger.info(f"rearranged moba_kv shape={moba_kv.shape}")
    
    # 处理不同数量的q和kv heads
    if num_q_head > num_kv_head:
        # 计算每个kv head对应的q head数量
        q_heads_per_kv_head = num_q_head // num_kv_head
        logger.info(f"Repeating KV {q_heads_per_kv_head} times for each KV head")
        # 复制kv，使其匹配q heads的数量
        moba_kv = torch.repeat_interleave(moba_kv, q_heads_per_kv_head, dim=0)
        logger.info(f"repeated moba_kv shape={moba_kv.shape}")
    
    if zero_expert_count > 0:
        logger.info("Filtering out zero experts in KV")
        moba_kv = moba_kv[valid_expert_mask]
        logger.info(f"filtered moba_kv shape={moba_kv.shape}")
    
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)  # [num_chunks*block_size, 2, 1, head_dim]
    logger.info(f"final moba_kv shape={moba_kv.shape}")
    
    # 构建cu_seqlen_kv用于flash attention
    logger.info("Building cu_seqlen_kv")
    moba_cu_seqlen_kv = torch.arange(
        0, num_filtered_chunk * num_q_head + 1 - zero_expert_count,
        dtype=torch.int32, device=q.device
    ) * block_size
    logger.info(f"moba_cu_seqlen_kv shape={moba_cu_seqlen_kv.shape}")
    
    # 检查形状一致性
    logger.info(f"Verifying moba_cu_seqlen_q shape={moba_cu_seqlen_q.shape} matches moba_cu_seqlen_kv shape={moba_cu_seqlen_kv.shape}")
    assert moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    
    # 使用自定义MixedAttention实现注意力计算
    logger.info("Creating output tensor")
    output = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )
    
    # 只计算MoBA部分，无自注意力
    logger.info("Calling flash_attn_varlen_func")
    try:
        moba_attn_out = flash_attn_varlen_func(
            q=moba_q.to(torch.bfloat16),
            k=moba_kv[:, 0].to(torch.bfloat16),
            v=moba_kv[:, 1].to(torch.bfloat16),
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=total_len,
            max_seqlen_k=block_size,
            causal=False,
            dropout_p=0.0,
        )
        logger.info(f"flash_attn_varlen_func completed, moba_attn_out shape={moba_attn_out.shape}")
    except Exception as e:
        logger.error(f"Error in flash_attn_varlen_func: {str(e)}")
        logger.error(f"q shape={moba_q.shape}, k shape={moba_kv[:, 0].shape}, v shape={moba_kv[:, 1].shape}")
        logger.error(f"cu_seqlens_q={moba_cu_seqlen_q}, cu_seqlens_k={moba_cu_seqlen_kv}")
        raise
    
    # 将结果重新分配到输出张量
    logger.info("Distributing attention results to output tensor")
    output_2d = output.view(-1, q.shape[2])
    raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
    raw_attn_out = raw_attn_out.to(output_2d.dtype)
    output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
    output = output.to(q.dtype)
    
    logger.info(f"Completed topk_sparse_attention_flash in {time.time() - start_time:.2f}s, output shape={output.shape}")
    return output