"""Decoder block for GPT-style models."""

from __future__ import annotations

import torch
from torch import nn

from deepseek_reimpl.layers.attention import build_attention
from deepseek_reimpl.layers.ffn import build_ffn
from deepseek_reimpl.layers.rmsnorm import RMSNorm
from deepseek_reimpl.model.config import GPTConfig


def build_norm(config: GPTConfig) -> nn.Module:
    """Build the configured normalization layer."""
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model)

    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.d_model)

    msg = f"Unsupported norm_type: {config.norm_type}"
    raise ValueError(msg)


class DecoderBlock(nn.Module):
    """Pre-norm Transformer decoder block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn_norm = build_norm(config)
        self.attn = build_attention(config)
        self.ffn_norm = build_norm(config)
        self.ffn = build_ffn(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and feedforward residual updates."""
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
