"""Reusable training loop utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import cycle
from typing import Any

import torch
import torch.nn as nn

from deepseek_reimpl.eval.language_model_eval import EvaluationMetrics, evaluate_language_model
from deepseek_reimpl.instrumentation.memory import get_peak_memory_bytes, reset_peak_memory
from deepseek_reimpl.instrumentation.throughput import ThroughputMeter
from deepseek_reimpl.train.losses import next_token_cross_entropy
from deepseek_reimpl.train.train_utils import count_batch_tokens, move_batch_to_device


@dataclass(frozen=True)
class TrainStepMetrics:
    """Metrics returned from one optimizer step."""

    loss: float
    num_tokens: int
    grad_norm: float | None


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Configuration for a minimal fixed-budget training loop."""

    max_steps: int | None
    max_tokens: int | None
    eval_interval: int | None
    log_interval: int
    eval_batches: int
    grad_clip: float | None = None


@dataclass(frozen=True)
class TrainingSummary:
    """Summary returned by the training loop."""

    steps: int
    train_tokens: int
    final_train_loss: float
    train_tokens_per_second: float
    peak_memory_bytes: int | None
    validation_loss: float | None
    validation_perplexity: float | None
    test_loss: float | None
    test_perplexity: float | None


def train_step(
    model: nn.Module,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: float | None = None,
) -> TrainStepMetrics:
    """Run one next-token-prediction optimization step."""
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError(f"grad_clip must be positive or None, got {grad_clip}")

    model.train()
    input_ids, targets = move_batch_to_device(batch, device)
    num_tokens = count_batch_tokens((input_ids, targets))

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = next_token_cross_entropy(logits, targets)
    loss.backward()

    grad_norm: float | None = None
    if grad_clip is not None:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm = float(grad_norm_tensor.item())

    optimizer.step()

    return TrainStepMetrics(
        loss=float(loss.item()),
        num_tokens=num_tokens,
        grad_norm=grad_norm,
    )


def _validate_training_loop_config(config: TrainingLoopConfig) -> None:
    if config.max_steps is None and config.max_tokens is None:
        raise ValueError("at least one of max_steps or max_tokens must be set")

    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError(f"max_steps must be positive or None, got {config.max_steps}")

    if config.max_tokens is not None and config.max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive or None, got {config.max_tokens}")

    if config.eval_interval is not None and config.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive or None, got {config.eval_interval}")

    if config.log_interval <= 0:
        raise ValueError(f"log_interval must be positive, got {config.log_interval}")

    if config.eval_batches <= 0:
        raise ValueError(f"eval_batches must be positive, got {config.eval_batches}")


def train_loop(
    model: nn.Module,
    train_dataloader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    config: TrainingLoopConfig,
    validation_dataloader: Iterable[Any] | None = None,
    test_dataloader: Iterable[Any] | None = None,
    log_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingSummary:
    """Train a model for a small fixed step/token budget."""
    _validate_training_loop_config(config)

    model.to(device)
    reset_peak_memory(device)
    throughput = ThroughputMeter()

    steps = 0
    train_tokens = 0
    final_train_loss = float("nan")
    validation_metrics: EvaluationMetrics | None = None
    test_metrics: EvaluationMetrics | None = None

    for batch in cycle(train_dataloader):
        if config.max_steps is not None and steps >= config.max_steps:
            break
        if config.max_tokens is not None and train_tokens >= config.max_tokens:
            break

        step_metrics = train_step(
            model,
            batch,
            optimizer,
            device=device,
            grad_clip=config.grad_clip,
        )

        steps += 1
        train_tokens += step_metrics.num_tokens
        final_train_loss = step_metrics.loss
        throughput.update(step_metrics.num_tokens)

        if log_callback is not None and steps % config.log_interval == 0:
            snapshot = throughput.snapshot()
            log_callback(
                {
                    "step": steps,
                    "train_loss": step_metrics.loss,
                    "tokens": train_tokens,
                    "tokens_per_second": snapshot.tokens_per_second,
                    "grad_norm": step_metrics.grad_norm,
                }
            )

        if (
            validation_dataloader is not None
            and config.eval_interval is not None
            and steps % config.eval_interval == 0
        ):
            validation_metrics = evaluate_language_model(
                model,
                validation_dataloader,
                device=device,
                max_batches=config.eval_batches,
            )

    if steps == 0:
        raise ValueError("training loop completed zero steps")

    if validation_dataloader is not None:
        validation_metrics = evaluate_language_model(
            model,
            validation_dataloader,
            device=device,
            max_batches=config.eval_batches,
        )

    if test_dataloader is not None:
        test_metrics = evaluate_language_model(
            model,
            test_dataloader,
            device=device,
            max_batches=config.eval_batches,
        )

    snapshot = throughput.snapshot()

    return TrainingSummary(
        steps=steps,
        train_tokens=train_tokens,
        final_train_loss=final_train_loss,
        train_tokens_per_second=snapshot.tokens_per_second,
        peak_memory_bytes=get_peak_memory_bytes(device),
        validation_loss=None if validation_metrics is None else validation_metrics.loss,
        validation_perplexity=None if validation_metrics is None else validation_metrics.perplexity,
        test_loss=None if test_metrics is None else test_metrics.loss,
        test_perplexity=None if test_metrics is None else test_metrics.perplexity,
    )
