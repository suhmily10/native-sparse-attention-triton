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
from flash_attn import flash_attn_varlen_func
from native_sparse_attention.ops import (
    compressed_attention,
    topk_sparse_attention,
    conv_compress,
    get_compressed_attention_topk,
)
from einops import rearrange
from native_sparse_attention.module.rope import RopeConfig, RotaryEmbedding



class NativeSparseAttentionQKV(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        window_size: int,
        # Keeping unused parameters for compatibility
        kernel_size: int = None,
        kernel_stride: int = None,
        block_size: int = None,
        topk: int = None,
        init_blocks: int = None,
        local_blocks: int = None,
    ):
        super().__init__()
        # configs
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size

        # Only output projection (no qkv projections)
        self.proj_o = torch.nn.Linear(
            self.num_q_heads * self.head_dim, self.hidden_size, bias=False
        )

        # init parameters
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(
        self,
        q: torch.Tensor,  # shape: [total_len, num_q_heads, head_dim]
        k: torch.Tensor,  # shape: [total_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # shape: [total_len, num_kv_heads, head_dim]
        cu_seqlens: torch.Tensor,  # shape: [batch_size + 1]
    ):
        
        cu_seqlens = cu_seqlens.to(torch.int32)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        # sliding window attention only
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            seqlens.max().item(),
            seqlens.max().item(),
            causal=True,
            window_size=(self.window_size, -1),
        )

        # output proj
        # attn_output = attn_output.reshape(-1, self.num_q_heads * self.head_dim)
        # attn_output = self.proj_o(attn_output)

        return attn_output
