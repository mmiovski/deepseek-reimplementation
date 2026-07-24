from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

INPUT_FLAT_CSV = Path("results/analysis/balanced_10seed_matrix_summary_flat.csv")

OUT_DESCRIPTIVES_CSV = Path(
    "results/analysis/balanced_10seed_matrix_descriptives_full_precision.csv"
)
OUT_DESCRIPTIVES_JSON = Path(
    "results/analysis/balanced_10seed_matrix_descriptives_full_precision.json"
)
OUT_METRIC_AVAILABILITY_CSV = Path(
    "results/analysis/balanced_10seed_matrix_metric_availability.csv"
)
OUT_METRIC_AVAILABILITY_JSON = Path(
    "results/analysis/balanced_10seed_matrix_metric_availability.json"
)
OUT_AUDIT_JSON = Path("results/analysis/balanced_10seed_matrix_descriptives_audit.json")

EXPECTED_MODELS = [
    "dense_121m",
    "mla_121m",
    "mtp_121m",
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
]

EXPECTED_BUDGETS = ["10m", "25m", "50m"]

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
EXPECTED_GROUP_SEED_COUNT = 10

# Exact 97.5th percentile for Student t distribution with df=9.
# Used because every model x budget group is scope-validated to n=10.
T_CRITICAL_975_DF9 = 2.2621571627409915

METRIC_GROUPS: dict[str, list[str]] = {
    "quality": [
        "validation_loss",
        "test_loss",
        "validation_perplexity",
        "test_perplexity",
    ],
    "optimization": [
        "train_loss",
        "lm_loss",
    ],
    "efficiency": [
        "train_tokens_per_second",
        "peak_memory_bytes",
        "total_parameters",
        "trainable_parameters",
        "activated_parameters_per_token",
        "tokens_per_total_parameter",
        "tokens_per_trainable_parameter",
        "tokens_per_activated_parameter",
        "requested_tokens_per_total_parameter",
        "requested_tokens_per_trainable_parameter",
        "requested_tokens_per_activated_parameter",
    ],
    "routing": [
        "mean_aux_loss",
        "mean_expert_load_variance",
        "mean_routing_entropy",
        "mean_router_probability",
        "expert_bias_mean",
        "expert_bias_std",
    ],
    "mtp": [
        "mtp_loss",
        "mtp_loss_weight",
        "mtp_num_future_tokens",
    ],
}

PRIMARY_METRICS = [
    "validation_loss",
    "test_loss",
    "validation_perplexity",
    "test_perplexity",
    "train_tokens_per_second",
    "peak_memory_bytes",
    "total_parameters",
    "trainable_parameters",
    "activated_parameters_per_token",
    "tokens_per_total_parameter",
    "tokens_per_trainable_parameter",
    "tokens_per_activated_parameter",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input artifact: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    clean = str(value).strip()
    if clean == "":
        return None

    lowered = clean.lower()
    if lowered in {"none", "null", "nan"}:
        return None

    parsed = float(clean)
    if math.isnan(parsed):
        return None
    return parsed


def validate_scope(rows: list[dict[str, str]]) -> dict[str, Any]:
    models = sorted({row["model"] for row in rows})
    budgets = sorted({row["budget"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})

    matrix_counts = Counter((row["model"], row["budget"], row["seed"]) for row in rows)
    model_budget_counts = Counter((row["model"], row["budget"]) for row in rows)

    duplicate_cells = {"|".join(key): count for key, count in matrix_counts.items() if count != 1}

    bad_model_budget_counts = {
        f"{model}|{budget}": {
            "expected": EXPECTED_GROUP_SEED_COUNT,
            "actual": model_budget_counts[(model, budget)],
        }
        for model in EXPECTED_MODELS
        for budget in EXPECTED_BUDGETS
        if model_budget_counts[(model, budget)] != EXPECTED_GROUP_SEED_COUNT
    }

    scope_passed = (
        len(rows) == EXPECTED_ROW_COUNT
        and models == sorted(EXPECTED_MODELS)
        and budgets == sorted(EXPECTED_BUDGETS)
        and seeds == sorted(EXPECTED_SEEDS)
        and not duplicate_cells
        and not bad_model_budget_counts
    )

    return {
        "row_count": len(rows),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "models": models,
        "expected_models": sorted(EXPECTED_MODELS),
        "budgets": budgets,
        "expected_budgets": sorted(EXPECTED_BUDGETS),
        "seeds": seeds,
        "expected_seeds": sorted(EXPECTED_SEEDS),
        "duplicate_cell_count": len(duplicate_cells),
        "duplicate_cells": duplicate_cells,
        "bad_model_budget_count_count": len(bad_model_budget_counts),
        "bad_model_budget_counts": bad_model_budget_counts,
        "scope_passed": scope_passed,
    }


def metric_group_for(metric: str) -> str:
    for group_name, metrics in METRIC_GROUPS.items():
        if metric in metrics:
            return group_name
    return "other"


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    seed_count = len(values)
    if seed_count == 0:
        return {
            "seed_count": 0,
            "mean": None,
            "std": None,
            "standard_error": None,
            "ci95_low": None,
            "ci95_high": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "iqr": None,
        }

    mean_value = statistics.fmean(values)
    median_value = statistics.median(values)
    min_value = min(values)
    max_value = max(values)

    if seed_count == 1:
        std_value = None
        standard_error = None
        ci95_low = None
        ci95_high = None
    else:
        std_value = statistics.stdev(values)
        standard_error = std_value / math.sqrt(seed_count)
        ci_margin = T_CRITICAL_975_DF9 * standard_error
        ci95_low = mean_value - ci_margin
        ci95_high = mean_value + ci_margin

    if seed_count >= 2:
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        q25 = quartiles[0]
        q75 = quartiles[2]
        iqr = q75 - q25
    else:
        q25 = None
        q75 = None
        iqr = None

    return {
        "seed_count": seed_count,
        "mean": mean_value,
        "std": std_value,
        "standard_error": standard_error,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "median": median_value,
        "min": min_value,
        "max": max_value,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
    }


def metric_fields(rows: list[dict[str, str]]) -> list[str]:
    all_fields = set(rows[0]) if rows else set()
    fields = sorted(field for field in all_fields if field.startswith("metric__"))
    return fields


def main() -> None:
    rows = read_rows(INPUT_FLAT_CSV)
    scope_audit = validate_scope(rows)

    if not scope_audit["scope_passed"]:
        write_json(OUT_AUDIT_JSON, {"scope_audit": scope_audit})
        raise ValueError("Input flat artifact failed balanced 10-seed scope validation.")

    metrics = [field.removeprefix("metric__") for field in metric_fields(rows)]

    descriptives: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []

    for metric in metrics:
        field = f"metric__{metric}"
        metric_group = metric_group_for(metric)

        non_missing_total = sum(1 for row in rows if parse_float(row.get(field)) is not None)
        availability_rows.append(
            {
                "metric": metric,
                "metric_group": metric_group,
                "non_missing_count": non_missing_total,
                "missing_count": len(rows) - non_missing_total,
                "total_rows": len(rows),
                "available_for_all_rows": non_missing_total == len(rows),
                "is_primary_metric": metric in PRIMARY_METRICS,
            }
        )

        for budget in EXPECTED_BUDGETS:
            for model in EXPECTED_MODELS:
                group_rows = [
                    row for row in rows if row["budget"] == budget and row["model"] == model
                ]

                values = [
                    parsed
                    for parsed in (parse_float(row.get(field)) for row in group_rows)
                    if parsed is not None
                ]

                summary = summarize_values(values)
                group_seed_count = len({row["seed"] for row in group_rows})
                non_missing_seed_count = len(values)

                descriptives.append(
                    {
                        "metric": metric,
                        "metric_group": metric_group,
                        "budget": budget,
                        "budget_tokens": int(group_rows[0]["budget_tokens"]),
                        "model": model,
                        "group_seed_count": group_seed_count,
                        "non_missing_seed_count": non_missing_seed_count,
                        "missing_seed_count": group_seed_count - non_missing_seed_count,
                        "is_primary_metric": metric in PRIMARY_METRICS,
                        **{key: value for key, value in summary.items() if key != "seed_count"},
                    }
                )

    missing_primary_groups = [
        {
            "metric": row["metric"],
            "budget": row["budget"],
            "model": row["model"],
            "non_missing_seed_count": row["non_missing_seed_count"],
        }
        for row in descriptives
        if row["is_primary_metric"] and row["non_missing_seed_count"] != EXPECTED_GROUP_SEED_COUNT
    ]

    bad_group_seed_counts = [
        {
            "metric": row["metric"],
            "budget": row["budget"],
            "model": row["model"],
            "group_seed_count": row["group_seed_count"],
        }
        for row in descriptives
        if row["group_seed_count"] != EXPECTED_GROUP_SEED_COUNT
    ]

    descriptives_passed = not missing_primary_groups and not bad_group_seed_counts

    OUT_DESCRIPTIVES_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_DESCRIPTIVES_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "metric",
            "metric_group",
            "budget",
            "budget_tokens",
            "model",
            "group_seed_count",
            "non_missing_seed_count",
            "missing_seed_count",
            "is_primary_metric",
            "mean",
            "std",
            "standard_error",
            "ci95_low",
            "ci95_high",
            "median",
            "min",
            "max",
            "q25",
            "q75",
            "iqr",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(descriptives)

    with OUT_METRIC_AVAILABILITY_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "metric",
            "metric_group",
            "non_missing_count",
            "missing_count",
            "total_rows",
            "available_for_all_rows",
            "is_primary_metric",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(availability_rows)

    write_json(OUT_DESCRIPTIVES_JSON, descriptives)
    write_json(OUT_METRIC_AVAILABILITY_JSON, availability_rows)

    audit = {
        "artifact_type": "balanced_10seed_matrix_descriptives_audit",
        "input_flat_csv": str(INPUT_FLAT_CSV),
        "v1_controlled_design_guardrail": (
            "Descriptives are generated only from the balanced 10-seed flat "
            "artifact, which itself reads only the audited manifest."
        ),
        "scope_audit": scope_audit,
        "metric_count": len(metrics),
        "metrics": metrics,
        "primary_metrics": PRIMARY_METRICS,
        "descriptive_row_count": len(descriptives),
        "metric_availability_row_count": len(availability_rows),
        "missing_primary_group_count": len(missing_primary_groups),
        "missing_primary_groups": missing_primary_groups,
        "bad_group_seed_count_count": len(bad_group_seed_counts),
        "bad_group_seed_counts": bad_group_seed_counts,
        "descriptives_csv": str(OUT_DESCRIPTIVES_CSV),
        "descriptives_json": str(OUT_DESCRIPTIVES_JSON),
        "metric_availability_csv": str(OUT_METRIC_AVAILABILITY_CSV),
        "metric_availability_json": str(OUT_METRIC_AVAILABILITY_JSON),
        "audit_json": str(OUT_AUDIT_JSON),
        "descriptives_passed": descriptives_passed,
    }
    write_json(OUT_AUDIT_JSON, audit)

    print(json.dumps(audit, indent=2, sort_keys=True))

    if not descriptives_passed:
        raise ValueError("Descriptive artifact validation failed. See audit JSON.")


if __name__ == "__main__":
    main()
