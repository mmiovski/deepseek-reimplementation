"""Export curated main-large batch-size sweep summaries.

This script converts per-run summary.json files from the main-large batch sweep
into compact JSON/CSV artifacts. The sweep is a feasibility/control artifact,
not main experimental evidence. It selects the largest common batch size that
successfully runs all six large architecture variants.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BATCH_SIZES = [2, 4]
ARCHITECTURE_SUFFIXES = [
    "00_dense_121m",
    "01_mla_121m",
    "02_mtp_121m",
    "03_moe_220m",
    "04_mla_moe_220m",
    "05_v3_routing_220m",
]

EXPECTED_MODELS = {
    "baseline_gpt",
    "mla_gpt",
    "mtp_gpt",
    "moe_gpt",
    "mla_moe_gpt",
    "v3_routing_gpt",
}

CSV_COLUMNS = [
    "experiment_name",
    "model_name",
    "model_config",
    "batch_size",
    "block_size",
    "max_tokens",
    "train_tokens",
    "steps",
    "train_tokens_per_second",
    "peak_memory_bytes",
    "peak_memory_gb",
    "final_train_loss",
    "validation_loss",
    "test_loss",
    "total_parameters",
    "trainable_parameters",
    "activated_parameters_per_token",
    "activated_to_total_ratio",
    "tokens_per_total_parameter",
    "tokens_per_trainable_parameter",
    "tokens_per_activated_parameter",
    "routing_stats_present",
    "mtp_enabled",
]


def _required(data: dict[str, Any], key: str, *, source: Path) -> Any:
    if key not in data:
        raise KeyError(f"Missing required key {key!r} in {source}")
    return data[key]


def _load_run(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    activated = _required(data, "activated_parameters", source=summary_path)
    if not isinstance(activated, dict):
        raise TypeError(f"activated_parameters must be a dict in {summary_path}")

    config_paths = _required(data, "config_paths", source=summary_path)
    if not isinstance(config_paths, dict):
        raise TypeError(f"config_paths must be a dict in {summary_path}")

    peak_memory_bytes = _required(data, "peak_memory_bytes", source=summary_path)
    if peak_memory_bytes is None:
        raise ValueError(f"peak_memory_bytes is null in {summary_path}")

    return {
        "experiment_name": _required(data, "experiment_name", source=summary_path),
        "metrics_file": str(summary_path).replace("\\", "/"),
        "model_name": _required(data, "model_name", source=summary_path),
        "model_config": config_paths["model_config"],
        "train_config": config_paths["train_config"],
        "batch_size": _required(data, "batch_size", source=summary_path),
        "block_size": _required(data, "block_size", source=summary_path),
        "max_tokens": _required(data, "max_tokens", source=summary_path),
        "train_tokens": _required(data, "train_tokens", source=summary_path),
        "steps": _required(data, "steps", source=summary_path),
        "train_tokens_per_second": _required(
            data,
            "train_tokens_per_second",
            source=summary_path,
        ),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gb": float(peak_memory_bytes) / 1_000_000_000.0,
        "final_train_loss": _required(data, "final_train_loss", source=summary_path),
        "validation_loss": _required(data, "validation_loss", source=summary_path),
        "test_loss": _required(data, "test_loss", source=summary_path),
        "total_parameters": _required(data, "total_parameters", source=summary_path),
        "trainable_parameters": _required(data, "trainable_parameters", source=summary_path),
        "activated_parameters_per_token": activated["activated_parameters_per_token"],
        "activated_to_total_ratio": activated["activated_to_total_ratio"],
        "tokens_per_total_parameter": _required(
            data,
            "tokens_per_total_parameter",
            source=summary_path,
        ),
        "tokens_per_trainable_parameter": _required(
            data,
            "tokens_per_trainable_parameter",
            source=summary_path,
        ),
        "tokens_per_activated_parameter": _required(
            data,
            "tokens_per_activated_parameter",
            source=summary_path,
        ),
        "routing_stats_present": data.get("routing_stats") is not None,
        "mtp_enabled": _required(data, "mtp_enabled", source=summary_path),
    }


def export_batch_sweep_summary(
    *,
    metrics_root: Path,
    output_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for batch_size in BATCH_SIZES:
        for suffix in ARCHITECTURE_SUFFIXES:
            name = f"main_large_batch_sweep_b{batch_size}_{suffix}"
            summary_path = metrics_root / name / "summary.json"
            if not summary_path.exists():
                missing.append(str(summary_path))
                continue
            rows.append(_load_run(summary_path))

    if missing:
        raise FileNotFoundError("Missing batch sweep summaries:\n" + "\n".join(missing))

    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_batch.setdefault(int(row["batch_size"]), []).append(row)

    for batch_size in BATCH_SIZES:
        batch_rows = by_batch.get(batch_size, [])
        observed_models = {row["model_name"] for row in batch_rows}
        if observed_models != EXPECTED_MODELS:
            raise ValueError(
                f"Batch {batch_size} did not cover all six models. "
                f"Expected {EXPECTED_MODELS}, got {observed_models}"
            )

    decision = {
        "selected_batch_size": 4,
        "selection_rule": (
            "Use the largest common batch size that successfully completed all six "
            "large variants without CUDA OOM."
        ),
        "rationale": (
            "Batch size 4 completed all six variants, had max peak memory below 5 GB "
            "in the sweep, and improved slowest observed throughput over batch size 2."
        ),
    }

    summary_by_batch = {}
    for batch_size, batch_rows in sorted(by_batch.items()):
        summary_by_batch[str(batch_size)] = {
            "runs": len(batch_rows),
            "max_peak_memory_gb": max(float(row["peak_memory_gb"]) for row in batch_rows),
            "min_train_tokens_per_second": min(
                float(row["train_tokens_per_second"]) for row in batch_rows
            ),
            "max_train_tokens_per_second": max(
                float(row["train_tokens_per_second"]) for row in batch_rows
            ),
        }

    payload = {
        "scope": "main_large_batch_size_sweep",
        "description": (
            "Batch-size feasibility sweep for the six large primary architecture "
            "variants. This sweep selects the standardized batch size for the main "
            "10M/25M/50M fixed-token matrix."
        ),
        "num_runs": len(rows),
        "batch_sizes": BATCH_SIZES,
        "architecture_suffixes": ARCHITECTURE_SUFFIXES,
        "decision": decision,
        "summary_by_batch": summary_by_batch,
        "runs": rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("results/metrics"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/analysis/main_large_batch_sweep_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/analysis/main_large_batch_sweep_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_batch_sweep_summary(
        metrics_root=args.metrics_root,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
