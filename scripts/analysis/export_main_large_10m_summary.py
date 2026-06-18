"""Export curated main-large 10M fixed-budget summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BUDGET = "10m"
EXPECTED_EXPERIMENTS = [
    "main_large_10m_00_dense_121m",
    "main_large_10m_01_mla_121m",
    "main_large_10m_02_mtp_121m",
    "main_large_10m_03_moe_220m",
    "main_large_10m_04_mla_moe_220m",
    "main_large_10m_05_v3_routing_220m",
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
    "train_config",
    "requested_train_tokens",
    "train_tokens",
    "train_token_overshoot",
    "train_token_overshoot_ratio",
    "steps",
    "batch_size",
    "block_size",
    "train_tokens_per_second",
    "peak_memory_bytes",
    "peak_memory_gb",
    "final_train_loss",
    "final_lm_loss",
    "final_mtp_loss",
    "validation_loss",
    "validation_perplexity",
    "test_loss",
    "test_perplexity",
    "total_parameters",
    "trainable_parameters",
    "activated_parameters_per_token",
    "activated_to_total_ratio",
    "tokens_per_total_parameter",
    "tokens_per_trainable_parameter",
    "tokens_per_activated_parameter",
    "requested_tokens_per_total_parameter",
    "requested_tokens_per_trainable_parameter",
    "requested_tokens_per_activated_parameter",
    "epoch_equivalent",
    "requested_epoch_equivalent",
    "mtp_enabled",
    "mtp_num_future_tokens",
    "mtp_loss_weight",
    "routing_stats_present",
    "mean_aux_loss",
    "mean_routing_entropy",
    "mean_expert_load_variance",
]


def _required(data: dict[str, Any], key: str, *, source: Path) -> Any:
    if key not in data:
        raise KeyError(f"Missing required key {key!r} in {source}")
    return data[key]


def _routing_value(data: dict[str, Any], key: str) -> Any:
    routing_stats = data.get("routing_stats")
    if routing_stats is None:
        return None
    if not isinstance(routing_stats, dict):
        raise TypeError("routing_stats must be a dict or None")
    return routing_stats.get(key)


def _load_run(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    activated = _required(data, "activated_parameters", source=summary_path)
    if not isinstance(activated, dict):
        raise TypeError(f"activated_parameters must be a dict in {summary_path}")

    config_paths = _required(data, "config_paths", source=summary_path)
    if not isinstance(config_paths, dict):
        raise TypeError(f"config_paths must be a dict in {summary_path}")

    peak_memory_bytes = _required(data, "peak_memory_bytes", source=summary_path)
    train_tokens = _required(data, "train_tokens", source=summary_path)
    requested_train_tokens = _required(data, "requested_train_tokens", source=summary_path)

    if train_tokens < requested_train_tokens:
        raise ValueError(
            f"{summary_path} trained fewer tokens than requested: "
            f"{train_tokens} < {requested_train_tokens}"
        )

    return {
        "experiment_name": _required(data, "experiment_name", source=summary_path),
        "metrics_file": str(summary_path).replace("\\", "/"),
        "model_name": _required(data, "model_name", source=summary_path),
        "model_config": config_paths["model_config"],
        "train_config": config_paths["train_config"],
        "requested_train_tokens": requested_train_tokens,
        "train_tokens": train_tokens,
        "train_token_overshoot": _required(data, "train_token_overshoot", source=summary_path),
        "train_token_overshoot_ratio": _required(
            data,
            "train_token_overshoot_ratio",
            source=summary_path,
        ),
        "steps": _required(data, "steps", source=summary_path),
        "batch_size": _required(data, "batch_size", source=summary_path),
        "block_size": _required(data, "block_size", source=summary_path),
        "train_tokens_per_second": _required(
            data,
            "train_tokens_per_second",
            source=summary_path,
        ),
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gb": float(peak_memory_bytes) / 1_000_000_000.0,
        "final_train_loss": _required(data, "final_train_loss", source=summary_path),
        "final_lm_loss": _required(data, "final_lm_loss", source=summary_path),
        "final_mtp_loss": data.get("final_mtp_loss"),
        "final_mtp_per_horizon_losses": data.get("final_mtp_per_horizon_losses"),
        "validation_loss": _required(data, "validation_loss", source=summary_path),
        "validation_perplexity": _required(data, "validation_perplexity", source=summary_path),
        "test_loss": _required(data, "test_loss", source=summary_path),
        "test_perplexity": _required(data, "test_perplexity", source=summary_path),
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
        "requested_tokens_per_total_parameter": _required(
            data,
            "requested_tokens_per_total_parameter",
            source=summary_path,
        ),
        "requested_tokens_per_trainable_parameter": _required(
            data,
            "requested_tokens_per_trainable_parameter",
            source=summary_path,
        ),
        "requested_tokens_per_activated_parameter": _required(
            data,
            "requested_tokens_per_activated_parameter",
            source=summary_path,
        ),
        "epoch_equivalent": _required(data, "epoch_equivalent", source=summary_path),
        "requested_epoch_equivalent": _required(
            data,
            "requested_epoch_equivalent",
            source=summary_path,
        ),
        "mtp_enabled": _required(data, "mtp_enabled", source=summary_path),
        "mtp_num_future_tokens": _required(data, "mtp_num_future_tokens", source=summary_path),
        "mtp_loss_weight": _required(data, "mtp_loss_weight", source=summary_path),
        "routing_stats_present": data.get("routing_stats") is not None,
        "mean_aux_loss": _routing_value(data, "mean_aux_loss"),
        "mean_routing_entropy": _routing_value(data, "mean_routing_entropy"),
        "mean_expert_load_variance": _routing_value(data, "mean_expert_load_variance"),
    }


def export_main_large_10m_summary(
    *,
    metrics_root: Path,
    output_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for name in EXPECTED_EXPERIMENTS:
        summary_path = metrics_root / name / "summary.json"
        if not summary_path.exists():
            missing.append(str(summary_path))
            continue
        rows.append(_load_run(summary_path))

    if missing:
        raise FileNotFoundError("Missing 10M summaries:\n" + "\n".join(missing))

    observed_models = {row["model_name"] for row in rows}
    if observed_models != EXPECTED_MODELS:
        raise ValueError(f"Expected models {EXPECTED_MODELS}, got {observed_models}")

    payload = {
        "scope": "main_large_10m_fixed_budget_results",
        "description": (
            "Curated results for the 10M-token subset of the main large fixed-budget "
            "matrix. These are main experimental results."
        ),
        "budget": BUDGET,
        "requested_train_tokens": 10_000_000,
        "num_runs": len(rows),
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
    parser.add_argument("--metrics-root", type=Path, default=Path("results/metrics"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/analysis/main_large_10m_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/analysis/main_large_10m_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_main_large_10m_summary(
        metrics_root=args.metrics_root,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
