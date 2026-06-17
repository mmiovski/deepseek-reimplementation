"""Multi-head Latent Attention layer."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from deepseek_reimpl.layers.rope import RotaryEmbedding
from deepseek_reimpl.model.config import GPTConfig


class MLAAttention(nn.Module):
    """DeepSeek-inspired Multi-head Latent Attention analogue.

    The implemented geometry follows the DeepSeek MLA pattern where query/key
    dimensions are decoupled from d_model / n_heads:

    - qk_nope_head_dim: non-rotary query/key head dimension
    - qk_rope_head_dim: rotary query/key head dimension
    - v_head_dim: value head dimension
    - kv_latent_dim: compressed key/value latent rank

    This is still a local single-GPU analogue, not FlashMLA or a full
    DeepSeek-V2/V3 production implementation.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        if config.attention_type != "mla":
            msg = "MLAAttention requires attention_type='mla'"
            raise ValueError(msg)

        if config.mla_kv_latent_dim is None or config.mla_q_rope_dim is None:
            msg = "MLAAttention requires MLA latent and RoPE dimensions"
            raise ValueError(msg)

        self.n_heads = config.n_heads
        self.kv_latent_dim = config.mla_kv_latent_dim
        self.qk_nope_head_dim = config.mla_qk_nope_dim
        self.qk_rope_head_dim = config.mla_q_rope_dim
        self.qk_head_dim = config.mla_qk_head_dim
        self.v_head_dim = config.mla_v_dim

        self.q_proj = nn.Linear(
            config.d_model,
            self.n_heads * self.qk_head_dim,
            bias=False,
        )
        self.kv_down_proj = nn.Linear(
            config.d_model,
            self.kv_latent_dim + self.qk_rope_head_dim,
            bias=False,
        )
        self.k_nope_up_proj = nn.Linear(
            self.kv_latent_dim,
            self.n_heads * self.qk_nope_head_dim,
            bias=False,
        )
        self.v_up_proj = nn.Linear(
            self.kv_latent_dim,
            self.n_heads * self.v_head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            self.n_heads * self.v_head_dim,
            config.d_model,
            bias=False,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.qk_rope_head_dim)

    def _split_heads(
        self,
        x: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x), self.qk_head_dim)
        q_nope, q_rope = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv_and_rope = self.kv_down_proj(x)
        kv_latent, k_rope = torch.split(
            compressed_kv_and_rope,
            [self.kv_latent_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        k_nope = self._split_heads(
            self.k_nope_up_proj(kv_latent),
            self.qk_nope_head_dim,
        )
        value = self._split_heads(
            self.v_up_proj(kv_latent),
            self.v_head_dim,
        )

        k_rope = k_rope.view(
            batch_size,
            seq_len,
            1,
            self.qk_rope_head_dim,
        ).expand(-1, -1, self.n_heads, -1)

        q_rope, k_rope = self.rope.apply(q_rope, k_rope)

        query = torch.cat((q_nope, q_rope), dim=-1)
        key = torch.cat((k_nope, k_rope), dim=-1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.qk_head_dim)

        causal_mask = torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=x.device,
        ).triu(1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.n_heads * self.v_head_dim)

        output = self.out_proj(context)
        return cast(torch.Tensor, self.resid_dropout(output))
