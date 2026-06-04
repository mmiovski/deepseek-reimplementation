"""Dense causal self-attention layers."""

from __future__ import annotations

import math

import torch
from torch import nn

from deepseek_reimpl.layers.rope import RotaryEmbedding
from deepseek_reimpl.model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Standard dense multi-head causal self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.block_size = config.block_size
        self.use_rope = config.positional_encoding == "rope"

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(self.head_dim) if self.use_rope else None

        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention to hidden states."""
        _, seq_len, _ = x.shape

        if seq_len > self.block_size:
            msg = "sequence length exceeds configured block_size"
            raise ValueError(msg)

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            q, k = self.rope.apply(q, k)

        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ v
        output: torch.Tensor = self.out_proj(self._merge_heads(context))
        return output
