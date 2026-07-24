from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

INPUT_FLAT_CSV = Path("results/analysis/balanced_10seed_matrix_summary_flat.csv")

OUT_TRENDS_CSV = Path("results/analysis/balanced_10seed_matrix_budget_trends_full_precision.csv")
OUT_TRENDS_JSON = Path("results/analysis/balanced_10seed_matrix_budget_trends_full_precision.json")
OUT_PAIR_DELTAS_CSV = Path(
    "results/analysis/balanced_10seed_matrix_budget_pair_deltas_full_precision.csv"
)
OUT_PAIR_DELTAS_JSON = Path(
    "results/analysis/balanced_10seed_matrix_budget_pair_deltas_full_precision.json"
)
OUT_SEED_TRENDS_CSV = Path("results/analysis/balanced_10seed_matrix_budget_trend_seed_records.csv")
OUT_SEED_TRENDS_JSON = Path(
    "results/analysis/balanced_10seed_matrix_budget_trend_seed_records.json"
)
OUT_AUDIT_JSON = Path("results/analysis/balanced_10seed_matrix_budget_trends_audit.json")

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
EXPECTED_GROUP_SEED_COUNT = 10

# Exact 97.5th percentile for Student t distribution with df=9.
T_CRITICAL_975_DF9 = 2.2621571627409915

METRIC_GROUPS = {
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

METRIC_DIRECTION = {
    "validation_loss": "lower_is_better",
    "test_loss": "lower_is_better",
    "validation_perplexity": "lower_is_better",
    "test_perplexity": "lower_is_better",
    "train_loss": "lower_is_better",
    "lm_loss": "lower_is_better",
    "train_tokens_per_second": "higher_is_better",
    "peak_memory_bytes": "lower_is_better",
    "mean_aux_loss": "lower_is_better",
    "mean_expert_load_variance": "lower_is_better",
    "expert_bias_std": "lower_is_better",
}

PRIMARY_TREND_METRICS = [
    "validation_loss",
    "test_loss",
    "validation_perplexity",
    "test_perplexity",
    "train_tokens_per_second",
    "peak_memory_bytes",
    "tokens_per_total_parameter",
    "tokens_per_trainable_parameter",
    "tokens_per_activated_parameter",
]

BUDGET_PAIRS = [
    ("10m", "25m"),
    ("25m", "50m"),
    ("10m", "50m"),
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
            "expected": len(EXPECTED_SEEDS),
            "actual": model_budget_counts[(model, budget)],
        }
        for model in EXPECTED_MODELS
        for budget in EXPECTED_BUDGETS
        if model_budget_counts[(model, budget)] != len(EXPECTED_SEEDS)
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


def metric_fields(rows: list[dict[str, str]]) -> list[str]:
    fields = sorted(field for field in rows[0] if field.startswith("metric__"))
    return fields


def metric_group_for(metric: str) -> str:
    for group_name, metric_names in METRIC_GROUPS.items():
        if metric in metric_names:
            return group_name
    return "other"


def metric_direction_for(metric: str) -> str:
    return METRIC_DIRECTION.get(metric, "descriptive_only")


def exact_sign_flip_p_value(values: list[float]) -> float | None:
    nonzero_abs_values = [abs(value) for value in values if value != 0.0]

    if not nonzero_abs_values:
        return 1.0

    if len(nonzero_abs_values) > 20:
        return None

    observed_abs_mean = abs(statistics.fmean(values))
    assignment_count = 0
    extreme_count = 0

    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero_abs_values)):
        signed_values = [
            sign * value for sign, value in zip(signs, nonzero_abs_values, strict=True)
        ]
        assigned_abs_mean = abs(statistics.fmean(signed_values))
        assignment_count += 1

        if assigned_abs_mean >= observed_abs_mean - 1e-15:
            extreme_count += 1

    return extreme_count / assignment_count


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    count = len(values)

    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "standard_error": None,
            "ci95_low": None,
            "ci95_high": None,
            "median": None,
            "min": None,
            "max": None,
        }

    mean_value = statistics.fmean(values)
    median_value = statistics.median(values)
    min_value = min(values)
    max_value = max(values)

    if count == 1:
        std_value = None
        standard_error = None
        ci95_low = None
        ci95_high = None
    else:
        std_value = statistics.stdev(values)
        standard_error = std_value / math.sqrt(count)
        if count == EXPECTED_GROUP_SEED_COUNT:
            ci_margin = T_CRITICAL_975_DF9 * standard_error
            ci95_low = mean_value - ci_margin
            ci95_high = mean_value + ci_margin
        else:
            ci95_low = None
            ci95_high = None

    return {
        "count": count,
        "mean": mean_value,
        "std": std_value,
        "standard_error": standard_error,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "median": median_value,
        "min": min_value,
        "max": max_value,
    }


def simple_slope(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    denominator = sum((x - x_mean) ** 2 for x in xs)

    if denominator == 0.0:
        raise ValueError("Cannot compute slope with zero x variance.")

    return numerator / denominator


def directional_improvement_count(values: list[float], metric_direction: str) -> int | None:
    if metric_direction == "lower_is_better":
        return sum(1 for value in values if value < 0.0)
    if metric_direction == "higher_is_better":
        return sum(1 for value in values if value > 0.0)
    return None


def directional_worsening_count(values: list[float], metric_direction: str) -> int | None:
    if metric_direction == "lower_is_better":
        return sum(1 for value in values if value > 0.0)
    if metric_direction == "higher_is_better":
        return sum(1 for value in values if value < 0.0)
    return None


def main() -> None:
    rows = read_rows(INPUT_FLAT_CSV)
    scope_audit = validate_scope(rows)

    if not scope_audit["scope_passed"]:
        write_json(OUT_AUDIT_JSON, {"scope_audit": scope_audit})
        raise ValueError("Input flat artifact failed balanced 10-seed scope validation.")

    metrics = [field.removeprefix("metric__") for field in metric_fields(rows)]
    row_by_key = {(row["model"], row["budget"], row["seed"]): row for row in rows}

    seed_trend_rows: list[dict[str, Any]] = []
    pair_delta_rows: list[dict[str, Any]] = []
    trend_summary_rows: list[dict[str, Any]] = []
    missing_seed_trends: list[dict[str, str]] = []

    for metric in metrics:
        metric_field = f"metric__{metric}"
        metric_group = metric_group_for(metric)
        metric_direction = metric_direction_for(metric)

        for model in EXPECTED_MODELS:
            seed_slopes = []
            seed_delta_25m_minus_10m = []
            seed_delta_50m_minus_25m = []
            seed_delta_50m_minus_10m = []
            values_by_budget: dict[str, list[float]] = {budget: [] for budget in EXPECTED_BUDGETS}

            for seed in EXPECTED_SEEDS:
                values: dict[str, float] = {}

                for budget in EXPECTED_BUDGETS:
                    row = row_by_key[(model, budget, seed)]
                    value = parse_float(row.get(metric_field))

                    if value is None:
                        missing_seed_trends.append(
                            {
                                "metric": metric,
                                "model": model,
                                "budget": budget,
                                "seed": seed,
                                "reason": "missing_metric_value",
                            }
                        )
                        continue

                    values[budget] = value
                    values_by_budget[budget].append(value)

                has_all_budgets = all(budget in values for budget in EXPECTED_BUDGETS)

                if not has_all_budgets:
                    continue

                x_values = [
                    math.log10(EXPECTED_BUDGET_TOKENS[budget]) for budget in EXPECTED_BUDGETS
                ]
                y_values = [values[budget] for budget in EXPECTED_BUDGETS]
                slope = simple_slope(x_values, y_values)

                delta_25m_minus_10m = values["25m"] - values["10m"]
                delta_50m_minus_25m = values["50m"] - values["25m"]
                delta_50m_minus_10m = values["50m"] - values["10m"]

                seed_slopes.append(slope)
                seed_delta_25m_minus_10m.append(delta_25m_minus_10m)
                seed_delta_50m_minus_25m.append(delta_50m_minus_25m)
                seed_delta_50m_minus_10m.append(delta_50m_minus_10m)

                if values["10m"] != 0.0:
                    relative_50m_vs_10m_percent = 100.0 * delta_50m_minus_10m / abs(values["10m"])
                else:
                    relative_50m_vs_10m_percent = None

                seed_trend_rows.append(
                    {
                        "metric": metric,
                        "metric_group": metric_group,
                        "metric_direction": metric_direction,
                        "model": model,
                        "seed": seed,
                        "value_10m": values["10m"],
                        "value_25m": values["25m"],
                        "value_50m": values["50m"],
                        "delta_25m_minus_10m": delta_25m_minus_10m,
                        "delta_50m_minus_25m": delta_50m_minus_25m,
                        "delta_50m_minus_10m": delta_50m_minus_10m,
                        "relative_50m_vs_10m_percent": relative_50m_vs_10m_percent,
                        "slope_per_log10_token": slope,
                        "monotone_decreasing": (values["10m"] >= values["25m"] >= values["50m"]),
                        "monotone_increasing": (values["10m"] <= values["25m"] <= values["50m"]),
                    }
                )

            for budget_start, budget_end in BUDGET_PAIRS:
                deltas = []
                relatives = []

                for seed in EXPECTED_SEEDS:
                    row_start = row_by_key[(model, budget_start, seed)]
                    row_end = row_by_key[(model, budget_end, seed)]

                    value_start = parse_float(row_start.get(metric_field))
                    value_end = parse_float(row_end.get(metric_field))

                    if value_start is None or value_end is None:
                        continue

                    delta = value_end - value_start
                    deltas.append(delta)

                    if value_start != 0.0:
                        relatives.append(100.0 * delta / abs(value_start))

                delta_summary = summarize_values(deltas)
                relative_summary = summarize_values(relatives)

                pair_delta_rows.append(
                    {
                        "metric": metric,
                        "metric_group": metric_group,
                        "metric_direction": metric_direction,
                        "model": model,
                        "budget_start": budget_start,
                        "budget_end": budget_end,
                        "budget_start_tokens": EXPECTED_BUDGET_TOKENS[budget_start],
                        "budget_end_tokens": EXPECTED_BUDGET_TOKENS[budget_end],
                        "paired_seed_count": len(deltas),
                        "mean_delta": delta_summary["mean"],
                        "std_delta": delta_summary["std"],
                        "standard_error_delta": delta_summary["standard_error"],
                        "ci95_low_delta": delta_summary["ci95_low"],
                        "ci95_high_delta": delta_summary["ci95_high"],
                        "median_delta": delta_summary["median"],
                        "min_delta": delta_summary["min"],
                        "max_delta": delta_summary["max"],
                        "mean_relative_percent": relative_summary["mean"],
                        "median_relative_percent": relative_summary["median"],
                        "directional_improvement_seed_count": directional_improvement_count(
                            deltas,
                            metric_direction,
                        ),
                        "directional_worsening_seed_count": directional_worsening_count(
                            deltas,
                            metric_direction,
                        ),
                        "unchanged_seed_count": sum(1 for value in deltas if value == 0.0),
                        "exact_sign_flip_p_value_for_delta": (
                            exact_sign_flip_p_value(
                                deltas,
                            )
                            if len(deltas) == EXPECTED_GROUP_SEED_COUNT
                            else None
                        ),
                    }
                )

            slope_summary = summarize_values(seed_slopes)
            delta_25m_10m_summary = summarize_values(seed_delta_25m_minus_10m)
            delta_50m_25m_summary = summarize_values(seed_delta_50m_minus_25m)
            delta_50m_10m_summary = summarize_values(seed_delta_50m_minus_10m)

            value_summaries = {
                budget: summarize_values(values) for budget, values in values_by_budget.items()
            }

            trend_summary_rows.append(
                {
                    "metric": metric,
                    "metric_group": metric_group,
                    "metric_direction": metric_direction,
                    "model": model,
                    "seed_count_with_complete_budget_trend": len(seed_slopes),
                    "is_primary_trend_metric": metric in PRIMARY_TREND_METRICS,
                    "mean_10m": value_summaries["10m"]["mean"],
                    "mean_25m": value_summaries["25m"]["mean"],
                    "mean_50m": value_summaries["50m"]["mean"],
                    "std_10m": value_summaries["10m"]["std"],
                    "std_25m": value_summaries["25m"]["std"],
                    "std_50m": value_summaries["50m"]["std"],
                    "mean_delta_25m_minus_10m": delta_25m_10m_summary["mean"],
                    "mean_delta_50m_minus_25m": delta_50m_25m_summary["mean"],
                    "mean_delta_50m_minus_10m": delta_50m_10m_summary["mean"],
                    "ci95_low_delta_50m_minus_10m": delta_50m_10m_summary["ci95_low"],
                    "ci95_high_delta_50m_minus_10m": delta_50m_10m_summary["ci95_high"],
                    "mean_slope_per_log10_token": slope_summary["mean"],
                    "std_slope_per_log10_token": slope_summary["std"],
                    "ci95_low_slope_per_log10_token": slope_summary["ci95_low"],
                    "ci95_high_slope_per_log10_token": slope_summary["ci95_high"],
                    "median_slope_per_log10_token": slope_summary["median"],
                    "exact_sign_flip_p_value_for_slope": (
                        exact_sign_flip_p_value(
                            seed_slopes,
                        )
                        if len(seed_slopes) == EXPECTED_GROUP_SEED_COUNT
                        else None
                    ),
                    "exact_sign_flip_p_value_for_50m_minus_10m_delta": (
                        exact_sign_flip_p_value(seed_delta_50m_minus_10m)
                        if len(seed_delta_50m_minus_10m) == EXPECTED_GROUP_SEED_COUNT
                        else None
                    ),
                    "directional_improvement_50m_vs_10m_seed_count": (
                        directional_improvement_count(
                            seed_delta_50m_minus_10m,
                            metric_direction,
                        )
                    ),
                    "directional_worsening_50m_vs_10m_seed_count": (
                        directional_worsening_count(
                            seed_delta_50m_minus_10m,
                            metric_direction,
                        )
                    ),
                }
            )

    expected_seed_trend_rows_for_primary = (
        len(PRIMARY_TREND_METRICS) * len(EXPECTED_MODELS) * len(EXPECTED_SEEDS)
    )
    actual_seed_trend_rows_for_primary = sum(
        1 for row in seed_trend_rows if row["metric"] in PRIMARY_TREND_METRICS
    )

    expected_trend_summary_rows = len(metrics) * len(EXPECTED_MODELS)
    expected_pair_delta_rows = len(metrics) * len(EXPECTED_MODELS) * len(BUDGET_PAIRS)

    primary_missing_seed_trends = [
        row for row in missing_seed_trends if row["metric"] in PRIMARY_TREND_METRICS
    ]

    trend_passed = (
        len(trend_summary_rows) == expected_trend_summary_rows
        and len(pair_delta_rows) == expected_pair_delta_rows
        and actual_seed_trend_rows_for_primary == expected_seed_trend_rows_for_primary
        and not primary_missing_seed_trends
    )

    OUT_TRENDS_CSV.parent.mkdir(parents=True, exist_ok=True)

    trend_fieldnames = [
        "metric",
        "metric_group",
        "metric_direction",
        "model",
        "seed_count_with_complete_budget_trend",
        "is_primary_trend_metric",
        "mean_10m",
        "mean_25m",
        "mean_50m",
        "std_10m",
        "std_25m",
        "std_50m",
        "mean_delta_25m_minus_10m",
        "mean_delta_50m_minus_25m",
        "mean_delta_50m_minus_10m",
        "ci95_low_delta_50m_minus_10m",
        "ci95_high_delta_50m_minus_10m",
        "mean_slope_per_log10_token",
        "std_slope_per_log10_token",
        "ci95_low_slope_per_log10_token",
        "ci95_high_slope_per_log10_token",
        "median_slope_per_log10_token",
        "exact_sign_flip_p_value_for_slope",
        "exact_sign_flip_p_value_for_50m_minus_10m_delta",
        "directional_improvement_50m_vs_10m_seed_count",
        "directional_worsening_50m_vs_10m_seed_count",
    ]

    pair_delta_fieldnames = [
        "metric",
        "metric_group",
        "metric_direction",
        "model",
        "budget_start",
        "budget_end",
        "budget_start_tokens",
        "budget_end_tokens",
        "paired_seed_count",
        "mean_delta",
        "std_delta",
        "standard_error_delta",
        "ci95_low_delta",
        "ci95_high_delta",
        "median_delta",
        "min_delta",
        "max_delta",
        "mean_relative_percent",
        "median_relative_percent",
        "directional_improvement_seed_count",
        "directional_worsening_seed_count",
        "unchanged_seed_count",
        "exact_sign_flip_p_value_for_delta",
    ]

    seed_trend_fieldnames = [
        "metric",
        "metric_group",
        "metric_direction",
        "model",
        "seed",
        "value_10m",
        "value_25m",
        "value_50m",
        "delta_25m_minus_10m",
        "delta_50m_minus_25m",
        "delta_50m_minus_10m",
        "relative_50m_vs_10m_percent",
        "slope_per_log10_token",
        "monotone_decreasing",
        "monotone_increasing",
    ]

    with OUT_TRENDS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trend_fieldnames)
        writer.writeheader()
        writer.writerows(trend_summary_rows)

    with OUT_PAIR_DELTAS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_delta_fieldnames)
        writer.writeheader()
        writer.writerows(pair_delta_rows)

    with OUT_SEED_TRENDS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_trend_fieldnames)
        writer.writeheader()
        writer.writerows(seed_trend_rows)

    write_json(OUT_TRENDS_JSON, trend_summary_rows)
    write_json(OUT_PAIR_DELTAS_JSON, pair_delta_rows)
    write_json(OUT_SEED_TRENDS_JSON, seed_trend_rows)

    audit = {
        "artifact_type": "balanced_10seed_matrix_budget_trends_audit",
        "input_flat_csv": str(INPUT_FLAT_CSV),
        "v1_controlled_design_guardrail": (
            "Budget trends are generated only from the balanced 10-seed flat "
            "artifact and summarize within-model behavior across token budgets."
        ),
        "scope_audit": scope_audit,
        "metrics": metrics,
        "metric_count": len(metrics),
        "primary_trend_metrics": PRIMARY_TREND_METRICS,
        "budget_pairs": BUDGET_PAIRS,
        "expected_trend_summary_rows": expected_trend_summary_rows,
        "trend_summary_row_count": len(trend_summary_rows),
        "expected_pair_delta_rows": expected_pair_delta_rows,
        "pair_delta_row_count": len(pair_delta_rows),
        "expected_seed_trend_rows_for_primary": expected_seed_trend_rows_for_primary,
        "actual_seed_trend_rows_for_primary": actual_seed_trend_rows_for_primary,
        "missing_seed_trend_count": len(missing_seed_trends),
        "primary_missing_seed_trend_count": len(primary_missing_seed_trends),
        "primary_missing_seed_trends": primary_missing_seed_trends[:20],
        "trends_csv": str(OUT_TRENDS_CSV),
        "trends_json": str(OUT_TRENDS_JSON),
        "pair_deltas_csv": str(OUT_PAIR_DELTAS_CSV),
        "pair_deltas_json": str(OUT_PAIR_DELTAS_JSON),
        "seed_trends_csv": str(OUT_SEED_TRENDS_CSV),
        "seed_trends_json": str(OUT_SEED_TRENDS_JSON),
        "audit_json": str(OUT_AUDIT_JSON),
        "trend_passed": trend_passed,
    }
    write_json(OUT_AUDIT_JSON, audit)

    print(json.dumps(audit, indent=2, sort_keys=True))

    if not trend_passed:
        raise ValueError("Budget trend artifact validation failed. See audit JSON.")


if __name__ == "__main__":
    main()
