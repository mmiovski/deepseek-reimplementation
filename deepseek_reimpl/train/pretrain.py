"""Pretraining orchestration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from deepseek_reimpl.data.collators import causal_lm_collate
from deepseek_reimpl.data.datasets import LanguageModelingDataset
from deepseek_reimpl.data.tokenization import encode_text_file
from deepseek_reimpl.instrumentation.activated_params import summarize_activated_parameters
from deepseek_reimpl.instrumentation.logging_utils import append_jsonl, write_json
from deepseek_reimpl.instrumentation.parameters import count_parameters, count_trainable_parameters
from deepseek_reimpl.instrumentation.routing_stats import summarize_routing_stats
from deepseek_reimpl.model.model_factory import build_model_from_config
from deepseek_reimpl.tokenizer.load_tokenizer import load_tokenizer
from deepseek_reimpl.train.optim import build_optimizer
from deepseek_reimpl.train.train_utils import resolve_device, set_seed
from deepseek_reimpl.train.trainer import TrainingLoopConfig, TrainingSummary, train_loop
from deepseek_reimpl.utils.config import load_yaml_config
from deepseek_reimpl.utils.paths import project_path


def _require_file(path: str | Path, *, purpose: str, remediation: str) -> Path:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = project_path(resolved_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {resolved_path}. {remediation}")

    return resolved_path


def _build_lm_dataloader(
    *,
    text_path: str | Path,
    tokenizer_path: str | Path,
    block_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    tokenizer_file = _require_file(
        tokenizer_path,
        purpose="tokenizer artifact",
        remediation="Run scripts/tokenizer/train_tokenizer.py first.",
    )
    text_file = _require_file(
        text_path,
        purpose="processed text artifact",
        remediation="Run scripts/data/prepare_tinystories.py first.",
    )

    tokenizer = load_tokenizer(tokenizer_file)
    token_ids = encode_text_file(text_file, tokenizer)

    dataset = LanguageModelingDataset(token_ids, block_size=block_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=causal_lm_collate,
    )


def _training_loop_config_from_mapping(train_config: Mapping[str, Any]) -> TrainingLoopConfig:
    return TrainingLoopConfig(
        max_steps=train_config["max_steps"],
        max_tokens=train_config["max_tokens"],
        eval_interval=train_config["eval_interval"],
        log_interval=train_config["log_interval"],
        eval_batches=train_config["eval_batches"],
        grad_clip=train_config["grad_clip"],
    )


def _summary_to_dict(summary: TrainingSummary) -> dict[str, Any]:
    return asdict(summary)


def _activated_parameter_summary_to_dict(model: torch.nn.Module) -> dict[str, Any]:
    summary = summarize_activated_parameters(model)
    return asdict(summary)


def _routing_stats_summary_to_dict(model: torch.nn.Module) -> dict[str, Any] | None:
    summary = summarize_routing_stats(model)
    if summary is None:
        return None
    return asdict(summary)


def run_pretraining_from_experiment_config(experiment_config_path: str | Path) -> dict[str, Any]:
    """Run a configured baseline pretraining smoke/control job."""
    experiment_wrapper = load_yaml_config(experiment_config_path)
    experiment_config = experiment_wrapper["experiment"]

    model_config = load_yaml_config(experiment_config["model_config"])
    data_config = load_yaml_config(experiment_config["data_config"])
    tokenizer_config = load_yaml_config(experiment_config["tokenizer_config"])
    train_wrapper = load_yaml_config(experiment_config["train_config"])
    train_config = train_wrapper["train"]

    set_seed(int(train_config["seed"]))
    device = resolve_device(str(train_config["device"]))

    tokenizer_path = tokenizer_config["artifacts"]["tokenizer_json"]
    data_artifacts = data_config["artifacts"]

    train_dataloader = _build_lm_dataloader(
        text_path=data_artifacts["train_text"],
        tokenizer_path=tokenizer_path,
        block_size=int(train_config["block_size"]),
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        shuffle=True,
    )
    validation_dataloader = _build_lm_dataloader(
        text_path=data_artifacts["validation_text"],
        tokenizer_path=tokenizer_path,
        block_size=int(train_config["block_size"]),
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        shuffle=False,
    )
    test_dataloader = _build_lm_dataloader(
        text_path=data_artifacts["test_text"],
        tokenizer_path=tokenizer_path,
        block_size=int(train_config["block_size"]),
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        shuffle=False,
    )

    model = build_model_from_config(model_config)
    optimizer = build_optimizer(model, train_config)

    output_dir = project_path(experiment_config["output_dir"])
    metrics_dir = project_path(experiment_config["metrics_dir"])
    train_log_path = output_dir / "train_log.jsonl"
    summary_path = metrics_dir / "summary.json"

    if train_log_path.exists():
        train_log_path.unlink()

    def log_record(record: dict[str, Any]) -> None:
        append_jsonl(train_log_path, record)

    summary = train_loop(
        model,
        train_dataloader,
        optimizer,
        device=device,
        config=_training_loop_config_from_mapping(train_config),
        validation_dataloader=validation_dataloader,
        test_dataloader=test_dataloader,
        log_callback=log_record,
    )

    activated_parameter_summary = _activated_parameter_summary_to_dict(model)
    routing_stats_summary = _routing_stats_summary_to_dict(model)

    summary_payload: dict[str, Any] = {
        "experiment_name": experiment_config["name"],
        "model_name": model_config["model"]["name"],
        "device": str(device),
        "seed": int(train_config["seed"]),
        "batch_size": int(train_config["batch_size"]),
        "block_size": int(train_config["block_size"]),
        "max_steps": train_config["max_steps"],
        "max_tokens": train_config["max_tokens"],
        "precision": train_config["precision"],
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "activated_parameters": activated_parameter_summary,
        "routing_stats": routing_stats_summary,
        **_summary_to_dict(summary),
    }

    write_json(summary_path, summary_payload)
    return summary_payload
