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
from einops import rearrange
from native_sparse_attention.ops.triton.flash_attention import flash_attention_varlen
from native_sparse_attention.ops import conv_compress
from native_sparse_attention.module.rope import RopeConfig, RotaryEmbedding


class GQATritonFlashAttention(torch.nn.Module):
    """GQA Attention implementation using Triton's Flash Attention kernel.
    
    This class supports both regular flash attention and can operate with
    compressed keys/values for comparing with Native Sparse Attention.
    """
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_rope: bool = False,
        rope_config: RopeConfig = None,
        use_compression: bool = False,
        kernel_size: int = 32,
        kernel_stride: int = 16,
    ):
        super().__init__()
        # configs
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_compression = use_compression
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        
        # projections
        self.proj_q = torch.nn.Linear(
            self.hidden_size, self.num_q_heads * self.head_dim, bias=False
        )
        self.proj_k = torch.nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.proj_v = torch.nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.proj_o = torch.nn.Linear(
            self.num_q_heads * self.head_dim, self.hidden_size, bias=False
        )
        
        # parameters for compression if enabled
        if self.use_compression:
            self.compress_key = torch.nn.Parameter(
                torch.zeros(
                    self.num_kv_heads * self.head_dim, self.head_dim, self.kernel_size
                )
            )
            self.compress_value = torch.nn.Parameter(
                torch.zeros(
                    self.num_kv_heads * self.head_dim, self.head_dim, self.kernel_size
                )
            )
            self.intra_block_pe = torch.nn.Parameter(
                torch.zeros(self.num_kv_heads, self.kernel_size, self.head_dim)
            )
        
        # rope for positional embedding if enabled
        self.use_rope = use_rope
        if self.use_rope:
            assert rope_config is not None, "rope_config must be provided when use_rope is True"
            self.rope_config = rope_config
            self.rope = RotaryEmbedding(self.rope_config)
        
        # init parameters
        self.init_params()
    
    def init_params(self):
        for p in self.parameters():
            torch.nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,  # shape: [total_len, hidden_size]
        cu_seqlens: torch.Tensor,  # shape: [batch_size + 1]
    ):
        # dtype and shape check
        assert x.dtype == torch.bfloat16 or x.dtype == torch.float16
        assert x.shape[-1] == self.hidden_size
        cu_seqlens = cu_seqlens.to(torch.int32)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        
        # qkv proj
        q = self.proj_q(x).view(-1, self.num_q_heads, self.head_dim)
        k = self.proj_k(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.proj_v(x).view(-1, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q = self.rope(q, cu_seqlens)
            k = self.rope(k, cu_seqlens)
        
        # Handle compression if enabled
        if self.use_compression:
            # Compress key and value tensors
            compressed_k, compressed_cu_seqlens = conv_compress(
                k,
                self.compress_key,
                cu_seqlens,
                self.kernel_size,
                self.kernel_stride,
                self.intra_block_pe,
            )
            compressed_v, _ = conv_compress(
                v,
                self.compress_value,
                cu_seqlens,
                self.kernel_size,
                self.kernel_stride,
                None,
            )
            
            # Apply RoPE to compressed key if needed
            if self.use_rope:
                compressed_k = self.rope(
                    compressed_k, compressed_cu_seqlens, start=0, stride=self.kernel_stride
                )
            
            # Get max sequence lengths
            compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
            max_seqlen_q = seqlens.max().item()
            max_seqlen_k = compressed_seqlens.max().item()
            
            # Use the compressed k/v with Triton flash attention
            attn_output = flash_attention_varlen(
                q, 
                compressed_k,
                compressed_v,
                cu_seqlens,
                compressed_cu_seqlens,
                max_seqlen_q,
                max_seqlen_k,
                causal=True,  # Assuming causal attention for LLM
                sm_scale=1.0 / math.sqrt(self.head_dim),
                gqa_interleave=False,  # Use Llama-style GQA
            )
        else:
            # Regular flash attention with original k/v
            max_seqlen = seqlens.max().item()
            attn_output = flash_attention_varlen(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                causal=True,  # Assuming causal attention for LLM
                sm_scale=1.0 / math.sqrt(self.head_dim),
                gqa_interleave=False,  # Use Llama-style GQA
            )
        
        # rearrange and output proj
        attn_output = rearrange(attn_output, "n h d -> n (h d)")
        attn_output = self.proj_o(attn_output)
        
        return attn_output


class GQATritonFlashAttentionNoRoPE(GQATritonFlashAttention):
    """GQA Attention without RoPE for easier comparison."""
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_compression: bool = False,
        kernel_size: int = 32,
        kernel_stride: int = 16,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_rope=False,
            rope_config=None,
            use_compression=use_compression,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
        )


class GQATritonFlashAttentionWithRoPE(GQATritonFlashAttention):
    """GQA Attention with RoPE for easier comparison."""
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_config: RopeConfig,
        use_compression: bool = False,
        kernel_size: int = 32,
        kernel_stride: int = 16,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_rope=True,
            rope_config=rope_config,
            use_compression=use_compression,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
        ) 