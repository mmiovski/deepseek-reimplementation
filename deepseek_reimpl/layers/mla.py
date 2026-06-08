"""Small-scale MLA-style causal attention.

This module is a faithful small-scale simplification of DeepSeek-style
Multi-head Latent Attention. It captures the architecture-level idea of joint
low-rank key/value compression plus decoupled rotary positional key handling,
but it is not FlashMLA, not production KV-cache engineering, and not a custom
CUDA implementation.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from deepseek_reimpl.layers.rope import RotaryEmbedding
from deepseek_reimpl.model.config import GPTConfig


class MLAAttention(nn.Module):
    """MLA-style multi-head causal attention with compressed KV state.

    The module preserves the same external contract as dense attention:

    input:  (batch, sequence, d_model)
    output: (batch, sequence, d_model)

    Internal structure:
    - q is projected directly from hidden states.
    - key/value content is jointly compressed to a latent KV state.
    - no-position keys and values are reconstructed from the latent state.
    - a separate rotary key path carries positional information.
    - RoPE is applied only to the query/key rotary subspace.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        if config.attention_type != "mla":
            msg = "MLAAttention requires attention_type='mla'"
            raise ValueError(msg)

        if config.mla_kv_latent_dim is None or config.mla_q_rope_dim is None:
            msg = "MLAAttention requires MLA dimensions in GPTConfig"
            raise ValueError(msg)

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.block_size = config.block_size
        self.kv_latent_dim = config.mla_kv_latent_dim
        self.q_rope_dim = config.mla_q_rope_dim
        self.q_nope_dim = self.head_dim - self.q_rope_dim

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.kv_down_proj = nn.Linear(config.d_model, self.kv_latent_dim, bias=False)
        self.k_nope_up_proj = nn.Linear(
            self.kv_latent_dim,
            config.n_heads * self.q_nope_dim,
            bias=False,
        )
        self.v_up_proj = nn.Linear(self.kv_latent_dim, config.d_model, bias=False)

        self.k_rope_proj = nn.Linear(
            config.d_model,
            config.n_heads * self.q_rope_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.q_rope_dim)

        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_full_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _split_nope_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.q_nope_dim).transpose(1, 2)

    def _split_rope_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.q_rope_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLA-style causal self-attention."""
        _, seq_len, _ = x.shape

        if seq_len > self.block_size:
            msg = "sequence length exceeds configured block_size"
            raise ValueError(msg)

        q = self._split_full_heads(self.q_proj(x))
        q_nope = q[..., : self.q_nope_dim]
        q_rope = q[..., self.q_nope_dim :]

        kv_latent = self.kv_down_proj(x)
        k_nope = self._split_nope_heads(self.k_nope_up_proj(kv_latent))
        v = self._split_full_heads(self.v_up_proj(kv_latent))

        k_rope = self._split_rope_heads(self.k_rope_proj(x))
        q_rope, k_rope = self.rope.apply(q_rope, k_rope)

        q_full = torch.cat((q_nope, q_rope), dim=-1)
        k_full = torch.cat((k_nope, k_rope), dim=-1)

        attn_scores = q_full @ k_full.transpose(-2, -1)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        causal_mask = cast(torch.Tensor, self.causal_mask)
        mask = causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ v
        output: torch.Tensor = self.out_proj(self._merge_heads(context))
        return output
