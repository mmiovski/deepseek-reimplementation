"""SwiGLU feedforward layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """SwiGLU feedforward projection block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            msg = "d_model must be positive"
            raise ValueError(msg)
        if d_ff <= 0:
            msg = "d_ff must be positive"
            raise ValueError(msg)

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation."""
        gated: torch.Tensor = F.silu(self.gate_proj(x)) * self.up_proj(x)
        hidden = self.dropout(gated)
        output: torch.Tensor = self.down_proj(hidden)
        return output
