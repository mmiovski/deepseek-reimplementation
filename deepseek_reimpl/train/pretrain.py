"""Pretraining orchestration helpers."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from deepseek_reimpl.data.collators import causal_lm_collate
from deepseek_reimpl.data.datasets import (
    LanguageModelingDataset,
    MemmapLanguageModelingDataset,
    RandomMemmapLanguageModelingDataset,
)
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


def _resolve_project_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = project_path(resolved_path)
    return resolved_path


def _load_tokenized_metadata(data_artifacts: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata_path_value = data_artifacts.get("tokenized_metadata")
    if metadata_path_value is None:
        return None

    metadata_path = _resolve_project_path(str(metadata_path_value))
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Tokenized metadata must be a JSON object: {metadata_path}")
    return cast(dict[str, Any], metadata)


def _tokenized_split_info(
    data_artifacts: Mapping[str, Any],
    tokenized_metadata: Mapping[str, Any] | None,
    *,
    split: str,
) -> tuple[str | None, int | None]:
    token_key_by_split = {
        "train": "train_token_ids",
        "validation": "validation_token_ids",
        "test": "test_token_ids",
    }
    token_key = token_key_by_split[split]
    token_ids_path = data_artifacts.get(token_key)

    if token_ids_path is None and tokenized_metadata is None:
        return None, None

    if token_ids_path is None or tokenized_metadata is None:
        raise ValueError(
            "Partial tokenized-data configuration detected. "
            "Run scripts/data/tokenize_lm_corpus.py or remove tokenized artifact keys."
        )

    split_metadata = tokenized_metadata.get("splits", {}).get(split)
    if not isinstance(split_metadata, Mapping):
        raise ValueError(f"Missing tokenized metadata for split: {split}")

    num_tokens = split_metadata.get("num_tokens")
    if not isinstance(num_tokens, int):
        raise ValueError(f"Tokenized metadata for split {split} must include integer num_tokens.")

    return str(token_ids_path), num_tokens


def _build_lm_dataloader(
    *,
    text_path: str | Path,
    tokenizer_path: str | Path,
    block_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    split_name: str = "train",
    token_ids_path: str | Path | None = None,
    token_count: int | None = None,
    seed: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    if token_ids_path is not None:
        if token_count is None:
            raise ValueError(f"token_count is required for tokenized split {split_name}")

        token_file = _require_file(
            token_ids_path,
            purpose=f"tokenized {split_name} token-ID artifact",
            remediation="Run scripts/data/tokenize_lm_corpus.py first.",
        )

        dataset: Any

        if shuffle:
            dataset = RandomMemmapLanguageModelingDataset(
                token_file,
                num_tokens=token_count,
                block_size=block_size,
                seed=seed,
            )
            dataloader_shuffle = False
        else:
            dataset = MemmapLanguageModelingDataset(
                token_file,
                num_tokens=token_count,
                block_size=block_size,
            )
            dataloader_shuffle = False
    else:
        tokenizer_file = _require_file(
            tokenizer_path,
            purpose="tokenizer artifact",
            remediation="Run scripts/tokenizer/train_tokenizer.py first.",
        )
        text_file = _require_file(
            text_path,
            purpose="processed text artifact",
            remediation=(
                "Run scripts/data/prepare_tinystories.py or "
                "scripts/data/prepare_wikitext.py first."
            ),
        )

        tokenizer = load_tokenizer(tokenizer_file)
        token_ids = encode_text_file(text_file, tokenizer)

        dataset = LanguageModelingDataset(token_ids, block_size=block_size)
        dataloader_shuffle = shuffle

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
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


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _token_parameter_accounting(
    *,
    train_corpus_tokens: int | None,
    requested_train_tokens: int,
    observed_train_tokens: int,
    total_parameters: int,
    trainable_parameters: int,
    activated_parameters_per_token: int,
) -> dict[str, float | int | None]:
    return {
        "train_corpus_tokens": train_corpus_tokens,
        "epoch_equivalent": _safe_ratio(observed_train_tokens, train_corpus_tokens),
        "requested_epoch_equivalent": _safe_ratio(requested_train_tokens, train_corpus_tokens),
        "tokens_per_total_parameter": _safe_ratio(observed_train_tokens, total_parameters),
        "tokens_per_trainable_parameter": _safe_ratio(observed_train_tokens, trainable_parameters),
        "tokens_per_activated_parameter": _safe_ratio(
            observed_train_tokens,
            activated_parameters_per_token,
        ),
        "requested_tokens_per_total_parameter": _safe_ratio(
            requested_train_tokens,
            total_parameters,
        ),
        "requested_tokens_per_trainable_parameter": _safe_ratio(
            requested_train_tokens,
            trainable_parameters,
        ),
        "requested_tokens_per_activated_parameter": _safe_ratio(
            requested_train_tokens,
            activated_parameters_per_token,
        ),
    }


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    """Return JSON-serializable runtime metadata for reproducibility."""
    cuda_device_name: str | None = None
    cuda_device_index: int | None = None

    if device.type == "cuda" and torch.cuda.is_available():
        cuda_device_index = device.index
        if cuda_device_index is None:
            cuda_device_index = torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(cuda_device_index)

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_index": cuda_device_index,
        "cuda_device_name": cuda_device_name,
    }


def _mtp_summary_metadata(
    model_config: Mapping[str, Any],
) -> dict[str, bool | int | float]:
    """Return normalized MTP fields for a pretraining summary."""
    return {
        "mtp_enabled": bool(model_config.get("mtp_enabled", False)),
        "mtp_num_future_tokens": int(model_config.get("mtp_num_future_tokens", 0)),
        "mtp_loss_weight": float(model_config.get("mtp_loss_weight", 0.0)),
        "mtp_share_lm_head": bool(model_config.get("mtp_share_lm_head", False)),
    }


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
    tokenized_metadata = _load_tokenized_metadata(data_artifacts)

    train_token_ids_path, train_token_count = _tokenized_split_info(
        data_artifacts,
        tokenized_metadata,
        split="train",
    )
    validation_token_ids_path, validation_token_count = _tokenized_split_info(
        data_artifacts,
        tokenized_metadata,
        split="validation",
    )
    test_token_ids_path, test_token_count = _tokenized_split_info(
        data_artifacts,
        tokenized_metadata,
        split="test",
    )

    seed = int(train_config["seed"])

    train_dataloader = _build_lm_dataloader(
        split_name="train",
        text_path=data_artifacts["train_text"],
        tokenizer_path=tokenizer_path,
        token_ids_path=train_token_ids_path,
        token_count=train_token_count,
        block_size=int(train_config["block_size"]),
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        shuffle=True,
        seed=seed,
    )
    validation_dataloader = _build_lm_dataloader(
        split_name="validation",
        text_path=data_artifacts["validation_text"],
        tokenizer_path=tokenizer_path,
        token_ids_path=validation_token_ids_path,
        token_count=validation_token_count,
        block_size=int(train_config["block_size"]),
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        shuffle=False,
        seed=seed,
    )
    test_dataloader = _build_lm_dataloader(
        split_name="test",
        text_path=data_artifacts["test_text"],
        tokenizer_path=tokenizer_path,
        token_ids_path=test_token_ids_path,
        token_count=test_token_count,
        block_size=int(train_config["block_size"]),
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        shuffle=False,
        seed=seed,
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

    total_parameters = count_parameters(model)
    trainable_parameters = count_trainable_parameters(model)
    activated_parameter_summary = _activated_parameter_summary_to_dict(model)
    routing_stats_summary = _routing_stats_summary_to_dict(model)
    token_parameter_accounting = _token_parameter_accounting(
        train_corpus_tokens=train_token_count,
        requested_train_tokens=int(train_config["max_tokens"]),
        observed_train_tokens=int(summary.train_tokens),
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        activated_parameters_per_token=int(
            activated_parameter_summary["activated_parameters_per_token"]
        ),
    )

    summary_payload: dict[str, Any] = {
        "experiment_name": experiment_config["name"],
        "model_name": model_config["model"]["name"],
        "experiment_config_path": str(experiment_config_path),
        "config_paths": {
            "model_config": experiment_config["model_config"],
            "data_config": experiment_config["data_config"],
            "tokenizer_config": experiment_config["tokenizer_config"],
            "train_config": experiment_config["train_config"],
        },
        "model_config": model_config["model"],
        "train_config": train_config,
        "data_config": {
            "dataset": data_config.get("dataset"),
            "splits": data_config.get("splits"),
            "artifacts": data_config.get("artifacts"),
            "tokenized_metadata": tokenized_metadata,
        },
        "tokenizer_config": {
            "tokenizer": tokenizer_config.get("tokenizer"),
            "artifacts": tokenizer_config.get("artifacts"),
        },
        "tokenizer_artifact": tokenizer_path,
        "runtime": _runtime_metadata(device),
        "device": str(device),
        "seed": seed,
        "batch_size": int(train_config["batch_size"]),
        "block_size": int(train_config["block_size"]),
        "max_steps": train_config["max_steps"],
        "max_tokens": train_config["max_tokens"],
        "precision": train_config["precision"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        **_mtp_summary_metadata(model_config["model"]),
        "activated_parameters": activated_parameter_summary,
        **token_parameter_accounting,
        "routing_stats": routing_stats_summary,
        **_summary_to_dict(summary),
    }

    write_json(summary_path, summary_payload)
    return summary_payload
