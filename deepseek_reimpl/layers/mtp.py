"""Multi-token prediction layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MTPOutput:
    """Output container for multi-token prediction.

    Attributes:
        next_token_logits: Standard next-token logits with shape
            (batch, sequence, vocab_size).
        future_token_logits: Auxiliary future-token logits with shape
            (num_future_tokens, batch, sequence, vocab_size).
    """

    next_token_logits: torch.Tensor
    future_token_logits: torch.Tensor


class MultiTokenPredictionHead(nn.Module):
    """Small-scale auxiliary heads for future-token prediction."""

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        num_future_tokens: int,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if num_future_tokens <= 0:
            raise ValueError(f"num_future_tokens must be positive, got {num_future_tokens}")

        self.num_future_tokens = num_future_tokens
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, bias=False) for _ in range(num_future_tokens)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return stacked future-token logits.

        Args:
            hidden_states: Tensor with shape (batch, sequence, d_model).

        Returns:
            Tensor with shape (num_future_tokens, batch, sequence, vocab_size).
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape (batch, sequence, d_model), "
                f"got rank {hidden_states.ndim}"
            )

        logits = [head(hidden_states) for head in self.heads]
        return torch.stack(logits, dim=0)
