"""Pretraining orchestration helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from deepseek_reimpl.data.collators import causal_lm_collate
from deepseek_reimpl.data.datasets import (
    FixedEvaluationWindowDataset,
    LanguageModelingDataset,
    MemmapLanguageModelingDataset,
    RandomMemmapLanguageModelingDataset,
)
from deepseek_reimpl.data.tokenization import encode_text_file
from deepseek_reimpl.instrumentation.activated_params import summarize_activated_parameters
from deepseek_reimpl.instrumentation.logging_utils import (
    append_jsonl,
    truncate_jsonl_to_step,
    write_json,
)
from deepseek_reimpl.instrumentation.parameters import count_parameters, count_trainable_parameters
from deepseek_reimpl.instrumentation.routing_stats import summarize_routing_stats
from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.model.model_factory import build_model_from_config
from deepseek_reimpl.tokenizer.load_tokenizer import load_tokenizer
from deepseek_reimpl.train.checkpointing import (
    atomic_save_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
)
from deepseek_reimpl.train.optim import build_optimizer
from deepseek_reimpl.train.train_utils import configure_determinism, resolve_device, set_seed
from deepseek_reimpl.train.trainer import TrainingLoopConfig, TrainingSummary, train_loop
from deepseek_reimpl.utils.artifacts import atomic_write_json, sha256_file, sha256_json
from deepseek_reimpl.utils.config import load_yaml_config
from deepseek_reimpl.utils.paths import project_path


def _require_file(path: str | Path, *, purpose: str, remediation: str) -> Path:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = project_path(resolved_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {resolved_path}. {remediation}")

    return resolved_path


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    observed = set(mapping)
    if observed != expected:
        raise ValueError(
            f"{label} keys do not match the protocol schema; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )


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
        raise FileNotFoundError(f"Missing tokenized-data metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Tokenized metadata must be a JSON object: {metadata_path}")
    if metadata.get("artifact_type") != "tokenized_corpus_metadata":
        raise ValueError("Unexpected tokenized-data metadata artifact type")
    if metadata.get("schema_version") != 2:
        raise ValueError("Tokenized-data metadata schema_version must be 2")
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

    if split_metadata.get("dtype") != "int32":
        raise ValueError(f"Tokenized metadata for split {split} must use int32")

    configured_path = _resolve_project_path(str(token_ids_path)).resolve()
    metadata_token_path = split_metadata.get("token_ids_path")
    if not isinstance(metadata_token_path, str):
        raise ValueError(f"Missing token_ids_path in metadata for split {split}")
    if configured_path != _resolve_project_path(metadata_token_path).resolve():
        raise ValueError(f"Token artifact path mismatch for split {split}")
    if not configured_path.exists():
        raise FileNotFoundError(f"Missing token artifact for split {split}: {configured_path}")

    expected_bytes = num_tokens * 4
    if split_metadata.get("token_bytes") != expected_bytes:
        raise ValueError(f"Token byte-count metadata mismatch for split {split}")
    if configured_path.stat().st_size != expected_bytes:
        raise ValueError(f"Token artifact size mismatch for split {split}")
    expected_sha256 = split_metadata.get("token_ids_sha256")
    if not isinstance(expected_sha256, str) or sha256_file(configured_path) != expected_sha256:
        raise ValueError(f"Token artifact SHA-256 mismatch for split {split}")

    special_counts = split_metadata.get("special_token_counts")
    if not isinstance(special_counts, Mapping) or any(
        not isinstance(count, int) or count != 0 for count in special_counts.values()
    ):
        raise ValueError(f"Unexpected special-token IDs in split {split}")

    text_path = _resolve_project_path(str(data_artifacts[f"{split}_text"])).resolve()
    if text_path != _resolve_project_path(str(split_metadata.get("text_path"))).resolve():
        raise ValueError(f"Source text path mismatch for split {split}")
    if sha256_file(text_path) != split_metadata.get("text_sha256"):
        raise ValueError(f"Source text SHA-256 mismatch for split {split}")
    record_path = _resolve_project_path(str(data_artifacts[f"{split}_records"])).resolve()
    if (
        record_path
        != _resolve_project_path(str(split_metadata.get("record_manifest_path"))).resolve()
    ):
        raise ValueError(f"Record-manifest path mismatch for split {split}")
    if sha256_file(record_path) != split_metadata.get("record_manifest_sha256"):
        raise ValueError(f"Record-manifest SHA-256 mismatch for split {split}")

    return str(token_ids_path), num_tokens


def _record_manifest_fingerprint(data_artifacts: Mapping[str, Any]) -> dict[str, Any]:
    split_indices: dict[str, list[int]] = {}
    all_indices: set[int] = set()
    for split in ("train", "validation", "test"):
        path = _resolve_project_path(str(data_artifacts[f"{split}_records"]))
        indices: list[int] = []
        for expected_ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            if record.get("ordinal") != expected_ordinal:
                raise ValueError(f"Non-contiguous record ordinals in split {split}")
            source_index = record.get("source_stream_index")
            if not isinstance(source_index, int):
                raise ValueError(f"Invalid source_stream_index in split {split}")
            if source_index in all_indices:
                raise ValueError("Prepared train/validation/test source records overlap")
            all_indices.add(source_index)
            indices.append(source_index)
        if not indices:
            raise ValueError(f"Empty record manifest for split {split}")
        split_indices[split] = indices
    return {
        split: {
            "records": len(indices),
            "source_stream_indices_sha256": sha256_json(indices),
        }
        for split, indices in split_indices.items()
    }


def _evaluation_window_metadata(
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    dataset = dataloader.dataset
    if not isinstance(dataset, FixedEvaluationWindowDataset):
        raise TypeError("Primary evaluation requires a fixed-window dataset")
    starts = list(dataset.start_indices)
    if len(starts) != len(set(starts)):
        raise ValueError("Evaluation window starts must be unique")
    if any(
        right - left < dataset.block_size for left, right in zip(starts, starts[1:], strict=False)
    ):
        raise ValueError("Evaluation windows must not overlap")
    return {
        "num_samples": len(starts),
        "block_size": dataset.block_size,
        "num_tokens": len(starts) * dataset.block_size,
        "start_indices": starts,
        "start_indices_sha256": sha256_json(starts),
    }


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
    fixed_eval_samples: int | None = None,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    if shuffle and fixed_eval_samples is not None:
        raise ValueError("fixed_eval_samples cannot be used with shuffle=True")

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

    if fixed_eval_samples is not None:
        dataset = FixedEvaluationWindowDataset(
            dataset,
            block_size=block_size,
            num_samples=fixed_eval_samples,
        )

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        num_workers=num_workers,
        collate_fn=causal_lm_collate,
        # DataLoader iterator construction otherwise consumes the global Torch
        # RNG, which would break exact dropout state across checkpoint resume.
        generator=dataloader_generator,
    )


def _training_loop_config_from_mapping(train_config: Mapping[str, Any]) -> TrainingLoopConfig:
    return TrainingLoopConfig(
        max_steps=train_config["max_steps"],
        max_tokens=train_config["max_tokens"],
        eval_interval=train_config["eval_interval"],
        log_interval=train_config["log_interval"],
        eval_batches=train_config["eval_batches"],
        grad_clip=train_config["grad_clip"],
        checkpoint_interval=train_config.get("checkpoint_interval"),
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
    cuda_driver_version: str | None = None

    if device.type == "cuda" and torch.cuda.is_available():
        cuda_device_index = device.index
        if cuda_device_index is None:
            cuda_device_index = torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(cuda_device_index)
        driver_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cuda_driver_version = driver_query.stdout.splitlines()[cuda_device_index].strip()

    dependency_names = ("numpy", "PyYAML", "matplotlib", "datasets", "tokenizers", "torch")
    dependencies = {name: importlib_metadata.version(name) for name in dependency_names}
    lock_hashes = {
        name: sha256_file(project_path(name))
        for name in ("requirements.txt", "requirements-cuda.txt")
        if project_path(name).is_file()
    }
    runtime = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_index": cuda_device_index,
        "cuda_device_name": cuda_device_name,
        "cuda_driver_version": cuda_driver_version,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "dependencies": dependencies,
        "dependency_lock_sha256": lock_hashes,
    }
    runtime["environment_sha256"] = sha256_json(runtime)
    return runtime


def _code_fingerprint() -> dict[str, Any]:
    """Hash every executable/configuration input used by the study."""
    root = project_path()
    if (root / ".git").exists():
        tracked_or_untracked = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    else:
        tracked_or_untracked = [
            str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
        ]
    included_roots = ("deepseek_reimpl/", "scripts/", "configs/")
    included_names = {
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-cuda.txt",
    }

    def is_source_path(path_value: str) -> bool:
        normalized = path_value.replace("\\", "/")
        return normalized.startswith(included_roots) or normalized in included_names

    records = []
    for relative in sorted(set(tracked_or_untracked)):
        normalized = relative.replace("\\", "/")
        if not is_source_path(normalized):
            continue
        path = root / relative
        if path.is_file():
            records.append({"path": normalized, "sha256": sha256_file(path)})

    commit: str | None = None
    status = ""
    if (root / ".git").exists():
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    return {
        "git_commit": commit,
        "git_dirty": any(
            is_source_path(line[3:].split(" -> ")[-1])
            for line in status.splitlines()
            if len(line) >= 4
        ),
        "source_file_count": len(records),
        "source_tree_sha256": sha256_json(records),
    }


def _validate_precision(
    train_config: Mapping[str, Any],
) -> str:
    """Validate and return the implemented training precision."""
    precision = str(train_config["precision"])

    if precision != "fp32":
        raise ValueError(
            "Only precision='fp32' is implemented; " f"received precision={precision!r}."
        )

    return precision


def _mtp_summary_metadata(
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return normalized MTP fields for a pretraining summary."""
    return {
        "mtp_enabled": bool(model_config.get("mtp_enabled", False)),
        "mtp_horizons": list(model_config.get("mtp_horizons", [])),
        "mtp_auxiliary_head_count": len(model_config.get("mtp_horizons", [])),
        "mtp_loss_weight": float(model_config.get("mtp_loss_weight", 0.0)),
        "mtp_share_lm_head": bool(model_config.get("mtp_share_lm_head", False)),
    }


def _archive_incomplete_run_artifacts(
    *,
    experiment_name: str,
    output_dir: Path,
    metrics_dir: Path,
    checkpoint_dir: Path,
) -> Path:
    """Atomically preserve an unresumable primary run before a clean restart."""
    resolved_dirs = [output_dir.resolve(), metrics_dir.resolve(), checkpoint_dir.resolve()]
    run_root = resolved_dirs[0].parent
    if any(path.parent != run_root for path in resolved_dirs):
        raise RuntimeError("Incomplete-run recovery requires sibling artifact directories")

    allowed_runs_root = project_path("results", "runs").resolve()
    if run_root.parent != allowed_runs_root or run_root.name != experiment_name:
        raise RuntimeError(
            f"Refusing to recover artifacts outside the primary run namespace: {run_root}"
        )
    if not run_root.is_dir():
        raise FileNotFoundError(f"Incomplete run root is missing: {run_root}")

    archive_parent = project_path("tmp", "interrupted_runs").resolve()
    archive_parent.mkdir(parents=True, exist_ok=True)
    archive_path = archive_parent / f"{experiment_name}-{time.time_ns()}-{os.getpid()}"
    if archive_path.exists():
        raise FileExistsError(f"Incomplete-run archive target already exists: {archive_path}")

    os.rename(run_root, archive_path)
    atomic_write_json(
        archive_path / "recovery.json",
        {
            "artifact_type": "incomplete_training_recovery",
            "schema_version": 1,
            "experiment_name": experiment_name,
            "reason": "run artifacts existed without a completed summary or resumable checkpoint",
            "source_run_root": run_root.relative_to(project_path().resolve()).as_posix(),
        },
    )
    return archive_path


def run_pretraining_from_experiment_config(
    experiment_config_path: str | Path,
    *,
    restart_incomplete: bool = False,
) -> dict[str, Any]:
    """Run a configured baseline pretraining smoke/control job."""
    experiment_wrapper = load_yaml_config(experiment_config_path)
    experiment_config = experiment_wrapper["experiment"]

    model_config = load_yaml_config(experiment_config["model_config"])
    data_config = load_yaml_config(experiment_config["data_config"])
    tokenizer_config = load_yaml_config(experiment_config["tokenizer_config"])
    train_wrapper = load_yaml_config(experiment_config["train_config"])
    train_config = train_wrapper["train"]
    precision = _validate_precision(train_config)
    GPTConfig.from_dict(model_config)

    protocol_id = str(experiment_config.get("protocol_id", "unversioned"))
    if protocol_id == "primary_matrix_2026":
        _require_exact_keys(
            experiment_config,
            {
                "name",
                "protocol_id",
                "variant",
                "budget",
                "seed",
                "requested_train_tokens",
                "output_dir",
                "metrics_dir",
                "checkpoint_dir",
                "model_config",
                "data_config",
                "tokenizer_config",
                "train_config",
            },
            label="experiment config",
        )
        _require_exact_keys(
            train_config,
            {
                "seed",
                "device",
                "batch_size",
                "block_size",
                "max_steps",
                "max_tokens",
                "eval_interval",
                "eval_batches",
                "learning_rate",
                "weight_decay",
                "betas",
                "grad_clip",
                "num_workers",
                "checkpoint_interval",
                "log_interval",
                "precision",
                "deterministic",
            },
            label="training config",
        )
        _require_exact_keys(
            data_config,
            {
                "dataset",
                "splits",
                "paths",
                "caps",
                "streaming",
                "preprocessing",
                "artifacts",
            },
            label="data config",
        )
        _require_exact_keys(
            tokenizer_config,
            {"tokenizer", "special_tokens", "training", "artifacts"},
            label="tokenizer config",
        )
        expected_identity = {
            "seed": int(train_config["seed"]),
            "requested_train_tokens": int(train_config["max_tokens"]),
        }
        for field, expected in expected_identity.items():
            if experiment_config.get(field) != expected:
                raise ValueError(f"Experiment identity field {field!r} does not match config")
        expected_variant = Path(str(experiment_config["model_config"])).stem
        if experiment_config.get("variant") != expected_variant:
            raise ValueError("Experiment variant does not match model configuration")
        budget_tokens = {"10m": 10_000_000, "25m": 25_000_000, "50m": 50_000_000}
        if budget_tokens.get(str(experiment_config.get("budget"))) != int(
            train_config["max_tokens"]
        ):
            raise ValueError("Experiment budget label does not match requested token budget")
        expected_run_root = f"results/runs/{experiment_config['name']}"
        expected_outputs = {
            "output_dir": f"{expected_run_root}/logs",
            "metrics_dir": f"{expected_run_root}/metrics",
            "checkpoint_dir": f"{expected_run_root}/checkpoints",
        }
        for field, expected_output in expected_outputs.items():
            if str(experiment_config.get(field)).replace("\\", "/") != expected_output:
                raise ValueError(f"Primary output field {field!r} is outside its run namespace")

    output_dir = project_path(experiment_config["output_dir"])
    metrics_dir = project_path(experiment_config["metrics_dir"])
    train_log_path = output_dir / "train_log.jsonl"
    summary_path = metrics_dir / "summary.json"
    checkpoint_dir = project_path(experiment_config.get("checkpoint_dir", output_dir))
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {summary_path}")

    if protocol_id == "primary_matrix_2026":
        run_root = output_dir.resolve().parent
        has_incomplete_artifacts = run_root.is_dir() and any(run_root.iterdir())
    else:
        run_root = output_dir.resolve()
        has_incomplete_artifacts = train_log_path.exists()
    if not checkpoint_path.exists() and has_incomplete_artifacts:
        if not restart_incomplete:
            raise FileExistsError(
                "Run artifacts exist without a resumable checkpoint: "
                f"{run_root}. Pass restart_incomplete=True only after confirming "
                "the run is incomplete."
            )
        archive_path = _archive_incomplete_run_artifacts(
            experiment_name=str(experiment_config["name"]),
            output_dir=output_dir,
            metrics_dir=metrics_dir,
            checkpoint_dir=checkpoint_dir,
        )
        print(f"Archived incomplete run artifacts before restart: {archive_path}")

    configure_determinism(enabled=bool(train_config.get("deterministic", True)))
    set_seed(int(train_config["seed"]))
    device = resolve_device(str(train_config["device"]))

    tokenizer_path = tokenizer_config["artifacts"]["tokenizer_json"]
    data_artifacts = data_config["artifacts"]
    tokenized_metadata = _load_tokenized_metadata(data_artifacts)

    model_block_size = int(model_config["model"]["block_size"])
    train_block_size = int(train_config["block_size"])
    if model_block_size != train_block_size:
        raise ValueError("Model and training block_size values must match")
    if int(train_config["num_workers"]) != 0:
        raise ValueError("Exact primary-run resume requires num_workers=0")

    tokenizer_file = _require_file(
        tokenizer_path,
        purpose="tokenizer artifact",
        remediation="Run scripts/tokenizer/train_tokenizer.py first.",
    )
    tokenizer = load_tokenizer(tokenizer_file)
    actual_vocab_size = tokenizer.get_vocab_size()
    if int(model_config["model"]["vocab_size"]) != actual_vocab_size:
        raise ValueError(
            "Model/tokenizer vocabulary mismatch: "
            f"model={model_config['model']['vocab_size']}, tokenizer={actual_vocab_size}"
        )

    if tokenized_metadata is not None:
        encoding_metadata = tokenized_metadata.get("encoding")
        if (
            not isinstance(encoding_metadata, Mapping)
            or encoding_metadata.get("add_special_tokens") is not False
        ):
            raise ValueError("Tokenized corpus must disable special-token injection")
        if tokenized_metadata.get("tokenizer_sha256") != sha256_file(tokenizer_file):
            raise ValueError("Tokenizer hash does not match tokenized corpus metadata")
        if tokenized_metadata.get("data_config_sha256") != sha256_file(
            _resolve_project_path(experiment_config["data_config"])
        ):
            raise ValueError("Data config hash does not match tokenized corpus metadata")
        if tokenized_metadata.get("tokenizer_config_sha256") != sha256_file(
            _resolve_project_path(experiment_config["tokenizer_config"])
        ):
            raise ValueError("Tokenizer config hash does not match tokenized corpus metadata")

    tokenizer_metadata_path = tokenizer_config["artifacts"].get("metadata_json")
    if tokenizer_metadata_path is not None:
        tokenizer_metadata_file = _require_file(
            tokenizer_metadata_path,
            purpose="tokenizer metadata",
            remediation="Regenerate the tokenizer artifact set.",
        )
        tokenizer_metadata = json.loads(tokenizer_metadata_file.read_text(encoding="utf-8"))
        if (
            tokenizer_metadata.get("artifact_type") != "tokenizer_training_metadata"
            or tokenizer_metadata.get("schema_version") != 2
        ):
            raise ValueError("Unsupported tokenizer metadata schema")
        if tokenizer_metadata.get("config_sha256") != sha256_json(tokenizer_config):
            raise ValueError("Tokenizer config does not match tokenizer metadata")
        if tokenizer_metadata.get("tokenizer_json", {}).get("sha256") != sha256_file(
            tokenizer_file
        ):
            raise ValueError("Tokenizer JSON does not match tokenizer metadata")

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
    record_manifest_fingerprint = (
        None if tokenized_metadata is None else _record_manifest_fingerprint(data_artifacts)
    )

    seed = int(train_config["seed"])
    fixed_eval_samples = int(train_config["eval_batches"]) * int(train_config["batch_size"])

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
        fixed_eval_samples=fixed_eval_samples,
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
        fixed_eval_samples=fixed_eval_samples,
    )
    validation_windows = _evaluation_window_metadata(validation_dataloader)
    test_windows = _evaluation_window_metadata(test_dataloader)

    model = build_model_from_config(model_config)
    model.to(device)
    optimizer = build_optimizer(model, train_config)

    code_fingerprint = _code_fingerprint()
    runtime_metadata = _runtime_metadata(device)
    artifact_fingerprint = {
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "tokenized_metadata_sha256": (
            None
            if data_artifacts.get("tokenized_metadata") is None
            else sha256_file(_resolve_project_path(data_artifacts["tokenized_metadata"]))
        ),
        "validation_window_sha256": validation_windows["start_indices_sha256"],
        "test_window_sha256": test_windows["start_indices_sha256"],
        "record_manifests": record_manifest_fingerprint,
    }
    run_identity = {
        "protocol_id": protocol_id,
        "variant": experiment_config.get("variant"),
        "budget": experiment_config.get("budget"),
        "experiment_name": experiment_config["name"],
        "experiment": experiment_config,
        "model_config": model_config,
        "data_config": data_config,
        "tokenizer_config": tokenizer_config,
        "train_config": train_wrapper,
        "code": code_fingerprint,
        "environment": runtime_metadata,
        "artifacts": artifact_fingerprint,
    }
    run_identity_sha256 = sha256_json(run_identity)
    initial_state: dict[str, Any] | None = None
    train_dataset = train_dataloader.dataset

    if checkpoint_path.exists():
        # Load through host memory so a resumed large model does not retain a
        # duplicate GPU-resident model state alongside the live parameters.
        checkpoint = load_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
        if checkpoint.get("run_identity_sha256") != run_identity_sha256:
            raise ValueError("Checkpoint run identity does not match the requested experiment")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if not isinstance(train_dataset, RandomMemmapLanguageModelingDataset):
            raise TypeError("Resumable primary training requires random memmap dataset")
        train_dataset.load_state_dict(checkpoint["dataset_state"])
        initial_state = checkpoint["training_state"]
        restore_rng_state(checkpoint["rng_state"])
        truncate_jsonl_to_step(train_log_path, max_step=int(initial_state["steps"]))
        del checkpoint
    elif train_log_path.exists():
        raise RuntimeError("Incomplete-run recovery failed to clear the prior training log")

    def log_record(record: dict[str, Any]) -> None:
        append_jsonl(train_log_path, record)

    def save_training_checkpoint(training_state: dict[str, Any]) -> None:
        if not isinstance(train_dataset, RandomMemmapLanguageModelingDataset):
            raise TypeError("Checkpointing requires random memmap training dataset")
        atomic_save_checkpoint(
            checkpoint_path,
            {
                "schema_version": 1,
                "run_identity_sha256": run_identity_sha256,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "dataset_state": train_dataset.state_dict(),
                "rng_state": capture_rng_state(),
                "training_state": training_state,
            },
        )

    summary = train_loop(
        model,
        train_dataloader,
        optimizer,
        device=device,
        config=_training_loop_config_from_mapping(train_config),
        validation_dataloader=validation_dataloader,
        test_dataloader=test_dataloader,
        log_callback=log_record,
        initial_state=initial_state,
        checkpoint_callback=save_training_checkpoint,
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
    training_window_sampler: dict[str, Any] | None = None
    if isinstance(train_dataset, RandomMemmapLanguageModelingDataset):
        final_dataset_state = train_dataset.state_dict()
        generator_state = final_dataset_state.get("generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("Completed random-window sampler has no generator state")
        samples_yielded = int(final_dataset_state["samples_yielded"])
        expected_samples = int(summary.train_tokens) // int(train_config["block_size"])
        if samples_yielded != expected_samples:
            raise RuntimeError("Training sampler count does not match consumed token count")
        training_window_sampler = {
            "seed": int(final_dataset_state["seed"]),
            "samples_yielded": samples_yielded,
            "generator_state_sha256": hashlib.sha256(
                generator_state.cpu().numpy().tobytes()
            ).hexdigest(),
        }

    summary_payload: dict[str, Any] = {
        "experiment_name": experiment_config["name"],
        "schema_version": 3,
        "protocol_id": protocol_id,
        "variant": experiment_config.get("variant"),
        "budget": experiment_config.get("budget"),
        "run_completed": True,
        "run_identity_sha256": run_identity_sha256,
        "run_identity": run_identity,
        "code_fingerprint": code_fingerprint,
        "artifact_fingerprint": artifact_fingerprint,
        "model_name": model_config["model"]["name"],
        "experiment_config_path": str(experiment_config_path),
        "config_paths": {
            "model_config": experiment_config["model_config"],
            "data_config": experiment_config["data_config"],
            "tokenizer_config": experiment_config["tokenizer_config"],
            "train_config": experiment_config["train_config"],
        },
        "config_sha256": {
            "experiment": sha256_file(_resolve_project_path(experiment_config_path)),
            "model": sha256_file(_resolve_project_path(experiment_config["model_config"])),
            "data": sha256_file(_resolve_project_path(experiment_config["data_config"])),
            "tokenizer": sha256_file(_resolve_project_path(experiment_config["tokenizer_config"])),
            "train": sha256_file(_resolve_project_path(experiment_config["train_config"])),
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
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "evaluation_windows": {
            "validation": validation_windows,
            "test": test_windows,
        },
        "training_window_sampler": training_window_sampler,
        "runtime": runtime_metadata,
        "device": str(device),
        "seed": seed,
        "batch_size": int(train_config["batch_size"]),
        "block_size": int(train_config["block_size"]),
        "max_steps": train_config["max_steps"],
        "max_tokens": train_config["max_tokens"],
        "precision": precision,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        **_mtp_summary_metadata(model_config["model"]),
        "activated_parameters": activated_parameter_summary,
        **token_parameter_accounting,
        "routing_stats": summary.test_routing_stats,
        "routing_controller_state": routing_stats_summary,
        **_summary_to_dict(summary),
    }

    summary_payload["train_log_sha256"] = sha256_file(train_log_path)
    write_json(summary_path, summary_payload)
    checkpoint_path.unlink(missing_ok=True)
    return summary_payload
