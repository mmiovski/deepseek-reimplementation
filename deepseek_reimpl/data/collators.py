"""Batch collation utilities for language modeling."""

from __future__ import annotations

import torch
from torch import Tensor


def causal_lm_collate(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Stack language-modeling examples into batch tensors."""
    if not batch:
        raise ValueError("batch must contain at least one example.")

    inputs, targets = zip(*batch, strict=True)

    return torch.stack(list(inputs), dim=0), torch.stack(list(targets), dim=0)
