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
            (num_auxiliary_horizons, batch, sequence, vocab_size).
        horizons: Token offsets corresponding to the first dimension of
            ``future_token_logits``. Horizon 1 is intentionally excluded because
            it is already predicted by ``next_token_logits``.
    """

    next_token_logits: torch.Tensor
    future_token_logits: torch.Tensor
    horizons: tuple[int, ...]


class MultiTokenPredictionHead(nn.Module):
    """Small-scale auxiliary heads for future-token prediction."""

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        horizons: tuple[int, ...],
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if not horizons:
            raise ValueError("horizons must not be empty")
        if any(horizon <= 1 for horizon in horizons):
            raise ValueError("auxiliary horizons must all be greater than 1")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("horizons must be unique and strictly increasing")

        self.horizons = horizons
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in horizons])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return stacked future-token logits.

        Args:
            hidden_states: Tensor with shape (batch, sequence, d_model).

        Returns:
            Tensor with shape (num_auxiliary_horizons, batch, sequence, vocab_size).
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape (batch, sequence, d_model), "
                f"got rank {hidden_states.ndim}"
            )

        logits = [head(hidden_states) for head in self.heads]
        return torch.stack(logits, dim=0)
