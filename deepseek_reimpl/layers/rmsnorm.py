"""Root mean square normalization."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm layer for Transformer blocks."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0:
            msg = "d_model must be positive"
            raise ValueError(msg)

        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization over the final dimension."""
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
