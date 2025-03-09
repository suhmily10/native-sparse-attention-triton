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
# See the License for the specific
import torch
import triton
import math
from native_sparse_attention.ops.torch.topk_sparse_attention import (
    topk_sparse_attention_torch,
)
from native_sparse_attention.ops.triton.topk_sparse_attention import (
    topk_sparse_attention,
    _topk_sparse_attention_fwd,
    _topk_sparse_attention_bwd,
)
from native_sparse_attention.ops.triton.flash_attention import (
    _flash_attention_fwd,
    _flash_attention_bwd,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from native_sparse_attention.ops.torch.topk_sparse_attention_flash import (
    topk_sparse_attention_flash,
)


def generate_topk_idx_example(
    seqlens: torch.Tensor, block_size: int, topk: int, num_heads: int
) -> torch.Tensor:
    """Generate topk idx example for test.

    Args:
        seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        block_size (int): key value block size
        topk (int): selected topk
        num_heads (int): number of key value heads

    Returns:
        torch.Tensor: shape [num_heads, total_seqlen, topk], topk key value block idx for each query. -1 means padding.
    """
    batch_size = seqlens.shape[0]
    num_blocks = torch.ceil(seqlens / block_size).to(torch.int32)
    topk_idx_all_heads = []
    for _ in range(num_heads):
        topk_idx = [
            torch.randn(seqlens[i], num_blocks[i], device="cuda")
            .topk(min(topk, num_blocks[i]), dim=-1)
            .indices.to(torch.int32)
            for i in range(batch_size)
        ]
        topk_idx = [
            torch.nn.functional.pad(
                topk_idx[i], (0, topk - topk_idx[i].shape[-1]), value=topk
            )
            for i in range(batch_size)
        ]
        topk_idx = torch.cat(topk_idx, dim=0)
        topk_idx = torch.sort(topk_idx, dim=1).values
        topk_idx[:, 0] = 0
        q_idx = torch.cat(
            [torch.arange(seqlens[i], device="cuda") for i in range(batch_size)], dim=0
        )
        
        topk_idx_all_heads.append(topk_idx)
    topk_idx = torch.stack(topk_idx_all_heads, dim=0)
    return topk_idx


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 3
    block_size = 64
    topk = 5
    
    # Ensure all sequence lengths are at least blocksize*topk
    min_seqlen = block_size * topk
    # Ensure all sequence lengths are multiples of block_size and greater than min_seqlen
    seqlens = torch.LongTensor([960, 1984, 4096]).int().cuda()  # All divisible by 64 and > min_seqlen
    
    # Verify that all sequences can select topk blocks
    for seq_len in seqlens:
        num_blocks = math.ceil(seq_len.item() / block_size)
        assert num_blocks >= topk, f"Sequence length {seq_len} has only {num_blocks} blocks, need at least {topk}"
    
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seqlens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    max_seqlen = seqlens.max().item()
    q = (
        torch.empty(cu_seqlens[-1], 1, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float)
    )
    k = (
        torch.empty(cu_seqlens[-1], 1, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float32)
    )
    v = (
        torch.empty(cu_seqlens[-1], 1, 96, device="cuda")
        .uniform_(-1, 1)
        .to(torch.float32)
    )
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    topk_idx = generate_topk_idx_example(seqlens, block_size, topk, 1)

    o = topk_sparse_attention_torch(q, k, v, topk_idx, block_size, cu_seqlens)

    randn = torch.randn_like(o)
    loss = (o * randn).sum()
    loss.backward()

    torch.manual_seed(42)
    q1 = q.clone().detach().requires_grad_()
    k1 = k.clone().detach().requires_grad_()
    v1 = v.clone().detach().requires_grad_()
    topk_idx1 = topk_idx.clone().detach()
    cu_seqlens1 = cu_seqlens.clone().detach()

    o1 = topk_sparse_attention_flash(q1, k1, v1, topk_idx1, block_size, cu_seqlens1)

    randn2 = randn.clone().detach()
    loss2 = (o1 * randn2).sum()
    loss2.backward()

    print("Same Output:", torch.allclose(o, o1, atol=0.01, rtol=0.01))
    print("Max Error:", (o - o1).abs().max().item())
    print()
    print("Same Query Gradient:", torch.allclose(q.grad, q1.grad, atol=0.01, rtol=0.01))
    print("Max Query Gradient Error:", (q.grad - q1.grad).abs().max().item())
    print()
    print("Same Key Gradient:", torch.allclose(k.grad, k1.grad, atol=0.01, rtol=0.01))
    print("Max Key Gradient Error:", (k.grad - k1.grad).abs().max().item())
    print()
    print("Same Value Gradient:", torch.allclose(v.grad, v1.grad, atol=0.01, rtol=0.01))
    print("Max Value Gradient Error:", (v.grad - v1.grad).abs().max().item())
    print()

 