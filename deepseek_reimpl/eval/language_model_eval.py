"""Language-model evaluation utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from deepseek_reimpl.train.losses import next_token_cross_entropy


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate language-model evaluation metrics."""

    loss: float
    perplexity: float
    num_batches: int
    num_tokens: int


def _perplexity_from_loss(loss: float) -> float:
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


def _unpack_lm_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
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


def evaluate_language_model(
    model: nn.Module,
    dataloader: Iterable[Any],
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> EvaluationMetrics:
    """Evaluate a causal language model with loss and perplexity."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches must be positive or None, got {max_batches}")

    was_training = model.training
    model.eval()

    total_loss_times_tokens = 0.0
    total_tokens = 0
    num_batches = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                if max_batches is not None and num_batches >= max_batches:
                    break

                input_ids, targets = _unpack_lm_batch(batch)
                input_ids = input_ids.to(device)
                targets = targets.to(device)

                logits = model(input_ids)
                loss = next_token_cross_entropy(logits, targets)

                batch_tokens = int(targets.numel())
                total_loss_times_tokens += float(loss.item()) * batch_tokens
                total_tokens += batch_tokens
                num_batches += 1
    finally:
        if was_training:
            model.train()

    if num_batches == 0 or total_tokens == 0:
        raise ValueError("evaluation dataloader produced no batches")

    average_loss = total_loss_times_tokens / total_tokens

    return EvaluationMetrics(
        loss=average_loss,
        perplexity=_perplexity_from_loss(average_loss),
        num_batches=num_batches,
        num_tokens=total_tokens,
    )
