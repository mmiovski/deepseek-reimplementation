from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("results/analysis/balanced_10seed_matrix_manifest.csv")

OUT_FLAT_CSV = Path("results/analysis/balanced_10seed_matrix_summary_flat.csv")
OUT_FLAT_JSON = Path("results/analysis/balanced_10seed_matrix_summary_flat.json")
OUT_SCHEMA_JSON = Path("results/analysis/balanced_10seed_matrix_summary_schema.json")
OUT_AUDIT_JSON = Path("results/analysis/balanced_10seed_matrix_extraction_audit.json")

EXPECTED_MODELS = [
    "dense_121m",
    "mla_121m",
    "mtp_121m",
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
]

EXPECTED_BUDGETS = ["10m", "25m", "50m"]

EXPECTED_BUDGET_TOKENS = {
    "10m": 10_000_000,
    "25m": 25_000_000,
    "50m": 50_000_000,
}

EXPECTED_SEEDS = [
    "1337",
    "2027",
    "31415",
    "4441",
    "5501",
    "6173",
    "8191",
    "10007",
    "11213",
    "12721",
]

EXPECTED_ROW_COUNT = 180

REQUIRED_CANONICAL_METRICS: dict[str, list[str]] = {
    "validation_loss": [
        "validation_loss",
        "final_validation_loss",
        "eval_validation_loss",
    ],
    "test_loss": [
        "test_loss",
        "final_test_loss",
        "eval_test_loss",
    ],
    "validation_perplexity": [
        "validation_perplexity",
        "final_validation_perplexity",
        "eval_validation_perplexity",
    ],
    "test_perplexity": [
        "test_perplexity",
        "final_test_perplexity",
        "eval_test_perplexity",
    ],
    "train_tokens_per_second": [
        "train_tokens_per_second",
        "tokens_per_second",
        "train_tokens_per_sec",
    ],
    "peak_memory_bytes": [
        "peak_memory_bytes",
        "peak_gpu_memory_bytes",
        "max_memory_bytes",
    ],
    "total_parameters": [
        "total_parameters",
        "parameter_count",
        "num_parameters",
    ],
    "trainable_parameters": [
        "trainable_parameters",
        "trainable_parameter_count",
        "num_trainable_parameters",
    ],
    "activated_parameters_per_token": [
        "activated_parameters_per_token",
        "activated_parameters",
    ],
    "tokens_per_total_parameter": [
        "tokens_per_total_parameter",
    ],
    "tokens_per_trainable_parameter": [
        "tokens_per_trainable_parameter",
    ],
    "tokens_per_activated_parameter": [
        "tokens_per_activated_parameter",
    ],
    "requested_tokens_per_total_parameter": [
        "requested_tokens_per_total_parameter",
    ],
    "requested_tokens_per_trainable_parameter": [
        "requested_tokens_per_trainable_parameter",
    ],
    "requested_tokens_per_activated_parameter": [
        "requested_tokens_per_activated_parameter",
    ],
}

OPTIONAL_CANONICAL_METRICS: dict[str, list[str]] = {
    "train_loss": [
        "train_loss",
        "final_train_loss",
        "final_total_train_loss",
    ],
    "lm_loss": [
        "lm_loss",
        "final_lm_loss",
    ],
    "mean_aux_loss": [
        "mean_aux_loss",
        "aux_loss",
        "final_aux_loss",
    ],
    "mean_expert_load_variance": [
        "mean_expert_load_variance",
        "expert_load_variance",
    ],
    "mean_routing_entropy": [
        "mean_routing_entropy",
        "routing_entropy",
    ],
    "mean_router_probability": [
        "mean_router_probability",
        "router_probability",
    ],
    "expert_bias_mean": [
        "expert_bias_mean",
    ],
    "expert_bias_std": [
        "expert_bias_std",
        "expert_bias_standard_deviation",
    ],
    "mtp_loss": [
        "mtp_loss",
        "final_mtp_loss",
    ],
    "mtp_loss_weight": [
        "mtp_loss_weight",
    ],
    "mtp_num_future_tokens": [
        "mtp_num_future_tokens",
    ],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            clean_key = str(key)
            next_prefix = f"{prefix}.{clean_key}" if prefix else clean_key
            flat.update(flatten_json(value, next_prefix))
        return flat

    if isinstance(obj, list):
        for index, value in enumerate(obj):
            next_prefix = f"{prefix}[{index}]"
            flat.update(flatten_json(value, next_prefix))
        return flat

    flat[prefix] = obj
    return flat


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def scalarize(value: Any) -> Any:
    if is_scalar(value):
        return value
    return json.dumps(value, sort_keys=True)


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required_columns = {
        "budget",
        "seed",
        "model",
        "experiment_name",
        "summary_path",
    }
    missing_columns = required_columns - set(rows[0]) if rows else required_columns
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_columns)}")

    return rows


def lower_key_map(flat: dict[str, Any]) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {}
    for key in flat:
        mapped.setdefault(key.lower(), []).append(key)
    return mapped


def candidate_key_matches(
    flat: dict[str, Any],
    candidate_names: list[str],
) -> list[str]:
    mapped = lower_key_map(flat)
    matches: list[str] = []

    for candidate in candidate_names:
        candidate_lower = candidate.lower()

        for key_lower, original_keys in mapped.items():
            if key_lower == candidate_lower:
                matches.extend(original_keys)

        if matches:
            continue

        suffix = f".{candidate_lower}"
        for key_lower, original_keys in mapped.items():
            if key_lower.endswith(suffix):
                matches.extend(original_keys)

        if matches:
            continue

    return sorted(set(matches))


def select_metric_key(
    flat: dict[str, Any],
    metric_name: str,
    candidate_names: list[str],
    *,
    required: bool,
) -> tuple[str | None, list[str]]:
    matches = candidate_key_matches(flat, candidate_names)

    numeric_matches = [
        key for key in matches if flat.get(key) is None or isinstance(flat.get(key), int | float)
    ]

    if len(numeric_matches) == 1:
        return numeric_matches[0], []

    if len(numeric_matches) > 1:
        return None, numeric_matches

    if required:
        return None, []

    return None, []


def validate_manifest_scope(rows: list[dict[str, str]]) -> dict[str, Any]:
    row_count = len(rows)
    models = sorted({row["model"] for row in rows})
    budgets = sorted({row["budget"] for row in rows})
    seeds = sorted({str(row["seed"]) for row in rows})

    matrix_counts = Counter((row["model"], row["budget"], str(row["seed"])) for row in rows)
    model_budget_counts = Counter((row["model"], row["budget"]) for row in rows)

    duplicate_matrix_cells = {
        "|".join(key): count for key, count in matrix_counts.items() if count != 1
    }

    bad_model_budget_counts = {
        f"{model}|{budget}": {
            "expected": len(EXPECTED_SEEDS),
            "actual": model_budget_counts[(model, budget)],
        }
        for model in EXPECTED_MODELS
        for budget in EXPECTED_BUDGETS
        if model_budget_counts[(model, budget)] != len(EXPECTED_SEEDS)
    }

    invalid_models = sorted(set(models) - set(EXPECTED_MODELS))
    invalid_budgets = sorted(set(budgets) - set(EXPECTED_BUDGETS))
    invalid_seeds = sorted(set(seeds) - set(EXPECTED_SEEDS))

    missing_models = sorted(set(EXPECTED_MODELS) - set(models))
    missing_budgets = sorted(set(EXPECTED_BUDGETS) - set(budgets))
    missing_seeds = sorted(set(EXPECTED_SEEDS) - set(seeds))

    scope_passed = (
        row_count == EXPECTED_ROW_COUNT
        and not invalid_models
        and not invalid_budgets
        and not invalid_seeds
        and not missing_models
        and not missing_budgets
        and not missing_seeds
        and not duplicate_matrix_cells
        and not bad_model_budget_counts
    )

    return {
        "row_count": row_count,
        "expected_row_count": EXPECTED_ROW_COUNT,
        "models": models,
        "budgets": budgets,
        "seeds": seeds,
        "invalid_models": invalid_models,
        "invalid_budgets": invalid_budgets,
        "invalid_seeds": invalid_seeds,
        "missing_models": missing_models,
        "missing_budgets": missing_budgets,
        "missing_seeds": missing_seeds,
        "duplicate_matrix_cell_count": len(duplicate_matrix_cells),
        "duplicate_matrix_cells": duplicate_matrix_cells,
        "bad_model_budget_count_count": len(bad_model_budget_counts),
        "bad_model_budget_counts": bad_model_budget_counts,
        "scope_passed": scope_passed,
    }


def main() -> None:
    manifest_rows = read_manifest()
    scope_audit = validate_manifest_scope(manifest_rows)

    if not scope_audit["scope_passed"]:
        write_json(OUT_AUDIT_JSON, {"scope_audit": scope_audit})
        raise ValueError("Manifest failed balanced 10-seed scope validation.")

    output_rows: list[dict[str, Any]] = []
    raw_field_names: set[str] = set()
    selected_required_metric_keys: dict[str, Counter[str]] = {
        metric: Counter() for metric in REQUIRED_CANONICAL_METRICS
    }
    selected_optional_metric_keys: dict[str, Counter[str]] = {
        metric: Counter() for metric in OPTIONAL_CANONICAL_METRICS
    }
    missing_required_metrics: dict[str, list[str]] = {
        metric: [] for metric in REQUIRED_CANONICAL_METRICS
    }
    ambiguous_required_metrics: dict[str, dict[str, list[str]]] = {
        metric: {} for metric in REQUIRED_CANONICAL_METRICS
    }
    ambiguous_optional_metrics: dict[str, dict[str, list[str]]] = {
        metric: {} for metric in OPTIONAL_CANONICAL_METRICS
    }
    summary_parse_errors: list[dict[str, str]] = []

    for run_index, manifest_row in enumerate(manifest_rows, start=1):
        summary_path = Path(manifest_row["summary_path"])
        experiment_name = manifest_row["experiment_name"]

        if not summary_path.exists():
            summary_parse_errors.append(
                {
                    "experiment_name": experiment_name,
                    "summary_path": str(summary_path),
                    "error": "summary_path_missing",
                }
            )
            continue

        try:
            summary = load_json(summary_path)
        except Exception as exc:
            summary_parse_errors.append(
                {
                    "experiment_name": experiment_name,
                    "summary_path": str(summary_path),
                    "error": repr(exc),
                }
            )
            continue

        flat = flatten_json(summary)
        raw_field_names.update(flat)

        budget = manifest_row["budget"]
        output_row: dict[str, Any] = {
            "run_index": run_index,
            "model": manifest_row["model"],
            "budget": budget,
            "budget_tokens": EXPECTED_BUDGET_TOKENS[budget],
            "seed": str(manifest_row["seed"]),
            "experiment_name": experiment_name,
            "summary_path": str(summary_path),
        }

        for metric_name, candidates in REQUIRED_CANONICAL_METRICS.items():
            selected_key, ambiguous_keys = select_metric_key(
                flat,
                metric_name,
                candidates,
                required=True,
            )
            output_key = f"metric__{metric_name}"

            if selected_key is None:
                output_row[output_key] = None
                if ambiguous_keys:
                    ambiguous_required_metrics[metric_name][experiment_name] = ambiguous_keys
                else:
                    missing_required_metrics[metric_name].append(experiment_name)
            else:
                output_row[output_key] = flat[selected_key]
                selected_required_metric_keys[metric_name][selected_key] += 1

        for metric_name, candidates in OPTIONAL_CANONICAL_METRICS.items():
            selected_key, ambiguous_keys = select_metric_key(
                flat,
                metric_name,
                candidates,
                required=False,
            )
            output_key = f"metric__{metric_name}"

            if selected_key is None:
                output_row[output_key] = None
                if ambiguous_keys:
                    ambiguous_optional_metrics[metric_name][experiment_name] = ambiguous_keys
            else:
                output_row[output_key] = flat[selected_key]
                selected_optional_metric_keys[metric_name][selected_key] += 1

        for key, value in flat.items():
            output_row[f"raw__{key}"] = scalarize(value)

        output_rows.append(output_row)

    required_missing_counts = {
        metric: len(experiments)
        for metric, experiments in missing_required_metrics.items()
        if experiments
    }
    required_ambiguous_counts = {
        metric: len(experiments)
        for metric, experiments in ambiguous_required_metrics.items()
        if experiments
    }
    optional_ambiguous_counts = {
        metric: len(experiments)
        for metric, experiments in ambiguous_optional_metrics.items()
        if experiments
    }

    extraction_passed = (
        len(output_rows) == EXPECTED_ROW_COUNT
        and not summary_parse_errors
        and not required_missing_counts
        and not required_ambiguous_counts
    )

    canonical_fields = [
        "run_index",
        "model",
        "budget",
        "budget_tokens",
        "seed",
        "experiment_name",
        "summary_path",
    ]
    canonical_fields.extend(f"metric__{metric}" for metric in REQUIRED_CANONICAL_METRICS)
    canonical_fields.extend(f"metric__{metric}" for metric in OPTIONAL_CANONICAL_METRICS)

    raw_fields = [f"raw__{field}" for field in sorted(raw_field_names)]
    fieldnames = canonical_fields + [field for field in raw_fields if field not in canonical_fields]

    OUT_FLAT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FLAT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    write_json(OUT_FLAT_JSON, output_rows)

    schema = {
        "source_manifest": str(MANIFEST_PATH),
        "row_count": len(output_rows),
        "field_count": len(fieldnames),
        "canonical_fields": canonical_fields,
        "raw_field_count": len(raw_fields),
        "raw_fields": raw_fields,
        "required_canonical_metrics": REQUIRED_CANONICAL_METRICS,
        "optional_canonical_metrics": OPTIONAL_CANONICAL_METRICS,
        "selected_required_metric_keys": {
            metric: dict(counter) for metric, counter in selected_required_metric_keys.items()
        },
        "selected_optional_metric_keys": {
            metric: dict(counter) for metric, counter in selected_optional_metric_keys.items()
        },
    }
    write_json(OUT_SCHEMA_JSON, schema)

    audit = {
        "artifact_type": "balanced_10seed_matrix_extraction_audit",
        "v1_controlled_design_guardrail": (
            "Extraction reads only balanced_10seed_matrix_manifest.csv and does "
            "not scan results/metrics broadly."
        ),
        "scope_audit": scope_audit,
        "summary_parse_error_count": len(summary_parse_errors),
        "summary_parse_errors": summary_parse_errors,
        "required_missing_counts": required_missing_counts,
        "required_missing_metrics": {
            metric: experiments[:20]
            for metric, experiments in missing_required_metrics.items()
            if experiments
        },
        "required_ambiguous_counts": required_ambiguous_counts,
        "required_ambiguous_metrics": {
            metric: dict(list(experiments.items())[:5])
            for metric, experiments in ambiguous_required_metrics.items()
            if experiments
        },
        "optional_ambiguous_counts": optional_ambiguous_counts,
        "optional_ambiguous_metrics": {
            metric: dict(list(experiments.items())[:5])
            for metric, experiments in ambiguous_optional_metrics.items()
            if experiments
        },
        "output_rows": len(output_rows),
        "expected_output_rows": EXPECTED_ROW_COUNT,
        "flat_csv": str(OUT_FLAT_CSV),
        "flat_json": str(OUT_FLAT_JSON),
        "schema_json": str(OUT_SCHEMA_JSON),
        "audit_json": str(OUT_AUDIT_JSON),
        "extraction_passed": extraction_passed,
    }
    write_json(OUT_AUDIT_JSON, audit)

    print(json.dumps(audit, indent=2, sort_keys=True))

    if not extraction_passed:
        raise ValueError("Balanced 10-seed extraction failed. See audit JSON.")


if __name__ == "__main__":
    main()
