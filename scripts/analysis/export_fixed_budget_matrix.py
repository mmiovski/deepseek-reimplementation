from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

EXPECTED_BUDGETS = {"10m", "25m", "50m"}
EXPECTED_SUFFIXES = {
    "00_baseline",
    "01_mla",
    "02_moe",
    "03_mla_moe",
    "04_v3_routing",
    "05_mtp",
}


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected summary JSON object in {path}, got {type(data).__name__}")
    return data


def _budget_from_name(experiment_name: str) -> str:
    parts = experiment_name.split("_")
    if len(parts) < 4 or parts[0] != "local":
        raise ValueError(f"Unexpected experiment name: {experiment_name}")
    return parts[1]


def _suffix_from_name(experiment_name: str) -> str:
    parts = experiment_name.split("_")
    if len(parts) < 4 or parts[0] != "local":
        raise ValueError(f"Unexpected experiment name: {experiment_name}")
    return "_".join(parts[2:])


def _row(summary: dict[str, Any]) -> dict[str, Any]:
    active_params = summary["activated_parameters"]["activated_parameters_per_token"]
    return {
        "budget": _budget_from_name(summary["experiment_name"]),
        "experiment_name": summary["experiment_name"],
        "model_name": summary["model_name"],
        "total_parameters": summary["total_parameters"],
        "trainable_parameters": summary["trainable_parameters"],
        "activated_parameters_per_token": active_params,
        "activated_to_total_ratio": summary["activated_parameters"]["activated_to_total_ratio"],
        "train_corpus_tokens": summary["train_corpus_tokens"],
        "requested_train_tokens": summary["requested_train_tokens"],
        "train_tokens": summary["train_tokens"],
        "train_token_overshoot": summary["train_token_overshoot"],
        "epoch_equivalent": summary["epoch_equivalent"],
        "tokens_per_total_parameter": summary["tokens_per_total_parameter"],
        "tokens_per_trainable_parameter": summary["tokens_per_trainable_parameter"],
        "tokens_per_activated_parameter": summary["tokens_per_activated_parameter"],
        "steps": summary["steps"],
        "validation_loss": summary["validation_loss"],
        "test_loss": summary["test_loss"],
        "validation_perplexity": summary["validation_perplexity"],
        "test_perplexity": summary["test_perplexity"],
        "train_tokens_per_second": summary["train_tokens_per_second"],
        "peak_memory_bytes": summary["peak_memory_bytes"],
        "peak_memory_mb": summary["peak_memory_bytes"] / 1_000_000,
        "mtp_enabled": summary["mtp_enabled"],
        "routing_stats_present": summary["routing_stats"] is not None,
        "seed": summary["seed"],
        "batch_size": summary["batch_size"],
        "block_size": summary["block_size"],
        "precision": summary["precision"],
        "model_config": summary["config_paths"]["model_config"],
        "data_config": summary["config_paths"]["data_config"],
        "tokenizer_config": summary["config_paths"]["tokenizer_config"],
        "train_config": summary["config_paths"]["train_config"],
    }


def main() -> None:
    summary_paths = sorted(Path("results/metrics").glob("local_*m_*/summary.json"))
    rows = [_row(_load_summary(path)) for path in summary_paths]

    seen = {(row["budget"], _suffix_from_name(row["experiment_name"])) for row in rows}
    expected = {(budget, suffix) for budget in EXPECTED_BUDGETS for suffix in EXPECTED_SUFFIXES}
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)

    shared_fields = {
        "train_corpus_tokens": sorted({row["train_corpus_tokens"] for row in rows}),
        "seed": sorted({row["seed"] for row in rows}),
        "batch_size": sorted({row["batch_size"] for row in rows}),
        "block_size": sorted({row["block_size"] for row in rows}),
        "precision": sorted({row["precision"] for row in rows}),
        "data_config": sorted({row["data_config"] for row in rows}),
        "tokenizer_config": sorted({row["tokenizer_config"] for row in rows}),
    }

    qc = {
        "num_rows": len(rows),
        "expected_rows": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "shared_fields": shared_fields,
        "valid": len(rows) == len(expected) and not missing and not unexpected,
    }

    output_dir = Path("results/tables/fixed_budget_matrix")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "fixed_budget_matrix.csv"
    json_path = output_dir / "fixed_budget_matrix.json"
    qc_path = output_dir / "fixed_budget_matrix_qc.json"

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(qc, indent=2, sort_keys=True))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {qc_path}")

    if not qc["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
