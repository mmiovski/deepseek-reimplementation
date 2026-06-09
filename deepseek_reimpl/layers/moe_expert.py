"""Expert feedforward modules for small-scale DeepSeekMoE-style layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MoEExpert(nn.Module):
    """Independently parameterized SwiGLU-style routed expert.

    The expert preserves hidden dimension:

    input:  (tokens, d_model)
    output: (tokens, d_model)
    """

    def __init__(self, *, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()

        if d_model <= 0:
            msg = "d_model must be positive"
            raise ValueError(msg)
        if d_ff <= 0:
            msg = "d_ff must be positive"
            raise ValueError(msg)
        if not 0.0 <= dropout < 1.0:
            msg = "dropout must be in the interval [0, 1)"
            raise ValueError(msg)

        self.d_model = d_model
        self.d_ff = d_ff

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the expert FFN transformation."""
        if hidden_states.ndim != 2:
            msg = (
                "MoEExpert expects flattened hidden states with shape "
                f"(tokens, d_model), got rank {hidden_states.ndim}"
            )
            raise ValueError(msg)

        if hidden_states.shape[-1] != self.d_model:
            msg = (
                "hidden_states final dimension must equal d_model; "
                f"got {hidden_states.shape[-1]} and expected {self.d_model}"
            )
            raise ValueError(msg)

        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden = self.dropout(gated)
        output: torch.Tensor = self.down_proj(hidden)
        return output
