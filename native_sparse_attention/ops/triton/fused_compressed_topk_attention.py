import torch
from typing import Tuple, Optional, Literal
from native_sparse_attention.ops.triton.compressed_attention import compressed_attention
from native_sparse_attention.ops.triton.topk_sparse_attention import topk_sparse_attention
from native_sparse_attention.ops.torch.compress_key_value import (
    conv_compress,
    linear_compress,
    avgpool_compress,
    weightedpool_compress,
)

def fused_compressed_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compress_key: torch.Tensor,
    compress_value: torch.Tensor,
    intra_block_pe: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens: torch.Tensor,
    max_seqlen: Optional[int] = None,
    sm_scale: Optional[float] = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    compression_method: Literal["conv", "linear", "avgpool", "weighted"] = "linear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused operation that combines compressed attention and topk sparse attention.
    
    This function first compresses keys and values, then generates topk indices and 
    performs sparse attention in one logically combined operation for efficiency.

    Args:
        q (torch.Tensor): shape [total_q_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        compress_key (torch.Tensor): Key compression weights
        compress_value (torch.Tensor): Value compression weights
        intra_block_pe (torch.Tensor): Intra-block positional encoding
        kernel_size (int): Kernel size for compression
        kernel_stride (int): Stride for compression
        block_size (int): Block size for topk attention
        topk (int): Number of top blocks to attend to
        cu_seqlens (torch.Tensor): Cumulative sequence lengths
        max_seqlen (Optional[int]): Maximum sequence length
        sm_scale (Optional[float]): Softmax scale, defaults to 1/sqrt(head_dim)
        init_blocks (int): Number of initial blocks to always include
        local_blocks (int): Number of local blocks to always include
        compression_method (str): Compression method, one of ["conv", "linear", "avgpool", "weighted"]
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Attention output and topk indices
    """
    # First compress keys and values
    if compression_method == "conv":
        compressed_k, compressed_cu_seqlens = conv_compress(
            k,
            compress_key,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )
        compressed_v, _ = conv_compress(
            v,
            compress_value,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            None,
        )
    elif compression_method == "linear":
        # Reshape to match expected shape for linear_compress
        total_len, num_heads, head_dim = k.shape
        reshaped_key_weights = compress_key.view(num_heads, kernel_size * head_dim, head_dim)
        reshaped_value_weights = compress_value.view(num_heads, kernel_size * head_dim, head_dim)
        
        compressed_k, compressed_cu_seqlens = linear_compress(
            k,
            reshaped_key_weights,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )
        compressed_v, _ = linear_compress(
            v,
            reshaped_value_weights,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            None,
        )
    elif compression_method == "avgpool":
        compressed_k, compressed_cu_seqlens = avgpool_compress(
            k,
            None,  # avgpool doesn't need weights
            cu_seqlens,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )
        compressed_v, _ = avgpool_compress(
            v,
            None,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            None,
        )
    elif compression_method == "weighted":
        # Reshape to match expected shape for weightedpool_compress
        total_len, num_heads, head_dim = k.shape
        reshaped_key_weights = compress_key.view(num_heads, kernel_size)
        reshaped_value_weights = compress_value.view(num_heads, kernel_size)
        
        compressed_k, compressed_cu_seqlens = weightedpool_compress(
            k,
            reshaped_key_weights,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            intra_block_pe,
        )
        compressed_v, _ = weightedpool_compress(
            v,
            reshaped_value_weights,
            cu_seqlens,
            kernel_size,
            kernel_stride,
            None,
        )
    else:
        raise ValueError(f"Unsupported compression method: {compression_method}")
    
    # Calculate required metrics
    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    if max_seqlen is None:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    max_compressed_seqlen = compressed_seqlens.max().item()
    
    # Get attention output and topk indices from compressed attention
    _, topk_idx = compressed_attention(
        q,
        compressed_k,
        compressed_v,
        kernel_size,
        kernel_stride,
        block_size,
        topk,
        cu_seqlens,
        compressed_cu_seqlens,
        max_seqlen,
        max_compressed_seqlen,
        sm_scale,
        init_blocks,
        local_blocks,
    )
    
    # Use topk indices for sparse attention with original (non-compressed) k/v
    attn_output = topk_sparse_attention(
        q, k, v, topk_idx, block_size, cu_seqlens, sm_scale
    )
    
    return attn_output, topk_idx
