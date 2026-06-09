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


def multi_token_cross_entropy(
    future_token_logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Compute auxiliary future-token prediction loss.

    Args:
        future_token_logits: Tensor with shape
            (num_future_tokens, batch, sequence, vocab_size). Horizon index 0
            predicts one token ahead, horizon index 1 predicts two tokens ahead,
            and so on.
        token_ids: Original unshifted input token IDs with shape
            (batch, sequence).

    Returns:
        A tuple of:
            - scalar mean auxiliary loss across horizons,
            - per-horizon detached loss values as Python floats.
    """
    if future_token_logits.ndim != 4:
        raise ValueError(
            "future_token_logits must have shape "
            "(num_future_tokens, batch, sequence, vocab), "
            f"got rank {future_token_logits.ndim}"
        )

    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must have shape (batch, sequence), got rank {token_ids.ndim}")

    num_future_tokens, batch_size, sequence_length, vocab_size = future_token_logits.shape

    if token_ids.shape != (batch_size, sequence_length):
        raise ValueError(
            "future_token_logits batch/sequence dimensions must match token_ids; "
            f"got logits {tuple(future_token_logits.shape)} and token_ids {tuple(token_ids.shape)}"
        )

    if num_future_tokens >= sequence_length:
        raise ValueError("num_future_tokens must be smaller than sequence length")

    horizon_losses: list[torch.Tensor] = []

    for horizon_index in range(num_future_tokens):
        shift = horizon_index + 1
        valid_logits = future_token_logits[horizon_index, :, :-shift, :]
        valid_targets = token_ids[:, shift:]

        horizon_loss = F.cross_entropy(
            valid_logits.reshape(-1, vocab_size),
            valid_targets.reshape(-1),
        )
        horizon_losses.append(horizon_loss)

    mtp_loss = torch.stack(horizon_losses).mean()
    per_horizon_losses = tuple(float(loss.detach().item()) for loss in horizon_losses)
    return mtp_loss, per_horizon_losses
