"""Training utility helpers."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve a configured device string into a torch.device."""
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device("cuda")

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raise ValueError(f"device must be one of 'cpu', 'cuda', or 'auto', got {device_name!r}")


def unpack_lm_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack a language-modeling batch into input and target tensors."""
    if isinstance(batch, Mapping):
        input_ids = batch.get("input_ids")
        targets = batch.get("target_ids", batch.get("labels"))

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("batch['input_ids'] must be a torch.Tensor")
        if not isinstance(targets, torch.Tensor):
            raise TypeError("batch must contain tensor targets under 'target_ids' or 'labels'")

        return input_ids, targets

    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        input_ids, targets = batch
        if not isinstance(input_ids, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("tuple/list batch must contain two torch.Tensor objects")
        return input_ids, targets

    raise TypeError("batch must be a mapping or a two-item tuple/list")


def move_batch_to_device(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a language-modeling batch to a target device."""
    input_ids, targets = unpack_lm_batch(batch)
    return input_ids.to(device), targets.to(device)


def count_batch_tokens(batch: Any) -> int:
    """Count input tokens in a language-modeling batch."""
    input_ids, _ = unpack_lm_batch(batch)
    return int(input_ids.numel())
