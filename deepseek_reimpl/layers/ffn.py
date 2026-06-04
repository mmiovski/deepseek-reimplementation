"""Dense feedforward block factory."""

from __future__ import annotations

import torch
from torch import nn

from deepseek_reimpl.layers.swiglu import SwiGLU
from deepseek_reimpl.model.config import GPTConfig


class GELUMLP(nn.Module):
    """Standard GELU MLP feedforward block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP block."""
        output: torch.Tensor = self.net(x)
        return output


def build_ffn(config: GPTConfig) -> nn.Module:
    """Build the configured dense feedforward module."""
    if config.ffn_type == "swiglu":
        return SwiGLU(config.d_model, config.d_ff, config.dropout)

    if config.ffn_type == "gelu_mlp":
        return GELUMLP(config.d_model, config.d_ff, config.dropout)

    msg = f"Unsupported ffn_type: {config.ffn_type}"
    raise ValueError(msg)
