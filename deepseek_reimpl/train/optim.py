"""Optimizer construction utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def _validate_betas(betas: tuple[float, float]) -> None:
    if len(betas) != 2:
        raise ValueError(f"betas must contain exactly two values, got {len(betas)}")

    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"beta2 must be in [0, 1), got {beta2}")


def build_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer for all model parameters."""
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    if weight_decay < 0:
        raise ValueError(f"weight_decay must be nonnegative, got {weight_decay}")

    _validate_betas(betas)

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
    )


def build_optimizer(model: nn.Module, train_config: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Build the default Phase 3 optimizer from a train config mapping."""
    betas_value = train_config["betas"]
    betas = (float(betas_value[0]), float(betas_value[1]))

    return build_adamw(
        model,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=betas,
    )
