"""Training loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def next_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute next-token cross-entropy from raw logits and target token IDs.

    Args:
        logits: Tensor with shape (batch, sequence, vocab_size).
        targets: Tensor with shape (batch, sequence).

    Returns:
        Scalar cross-entropy loss tensor.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (batch, sequence, vocab), got rank {logits.ndim}")

    if targets.ndim != 2:
        raise ValueError(f"targets must have shape (batch, sequence), got rank {targets.ndim}")

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "logits batch/sequence dimensions must match targets; "
            f"got logits {tuple(logits.shape)} and targets {tuple(targets.shape)}"
        )

    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
