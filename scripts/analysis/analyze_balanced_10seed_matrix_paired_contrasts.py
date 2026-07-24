from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

INPUT_FLAT_CSV = Path("results/analysis/balanced_10seed_matrix_summary_flat.csv")

OUT_CONTRASTS_CSV = Path(
    "results/analysis/balanced_10seed_matrix_paired_contrasts_full_precision.csv"
)
OUT_CONTRASTS_JSON = Path(
    "results/analysis/balanced_10seed_matrix_paired_contrasts_full_precision.json"
)
OUT_SEED_RECORDS_CSV = Path("results/analysis/balanced_10seed_matrix_paired_seed_records.csv")
OUT_SEED_RECORDS_JSON = Path("results/analysis/balanced_10seed_matrix_paired_seed_records.json")
OUT_AUDIT_JSON = Path("results/analysis/balanced_10seed_matrix_paired_contrasts_audit.json")

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
EXPECTED_PAIR_COUNT = 10

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260723

# Exact 97.5th percentile for Student t distribution with df=9.
T_CRITICAL_975_DF9 = 2.2621571627409915

TARGET_METRICS = [
    "validation_loss",
    "test_loss",
    "validation_perplexity",
    "test_perplexity",
    "train_loss",
    "lm_loss",
    "train_tokens_per_second",
    "peak_memory_bytes",
]

METRIC_GROUP = {
    "validation_loss": "quality",
    "test_loss": "quality",
    "validation_perplexity": "quality",
    "test_perplexity": "quality",
    "train_loss": "optimization",
    "lm_loss": "optimization",
    "train_tokens_per_second": "efficiency",
    "peak_memory_bytes": "efficiency",
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
}

TARGET_CONTRASTS = [
    {
        "contrast_id": "mla_vs_dense",
        "contrast_family": "attention_mechanism",
        "model_a": "dense_121m",
        "model_b": "mla_121m",
        "interpretation": "Effect of MLA attention relative to dense baseline.",
    },
    {
        "contrast_id": "moe_vs_dense",
        "contrast_family": "sparse_ffn_mechanism",
        "model_a": "dense_121m",
        "model_b": "moe_220m",
        "interpretation": "Effect of sparse MoE capacity relative to dense baseline.",
    },
    {
        "contrast_id": "mla_moe_vs_dense",
        "contrast_family": "composition_supporting",
        "model_a": "dense_121m",
        "model_b": "mla_moe_220m",
        "interpretation": (
            "Supporting comparison for combined MLA plus MoE relative to dense " "baseline."
        ),
    },
    {
        "contrast_id": "mla_moe_vs_mla",
        "contrast_family": "composition_mechanism",
        "model_a": "mla_121m",
        "model_b": "mla_moe_220m",
        "interpretation": "Effect of adding MoE capacity on top of MLA.",
    },
    {
        "contrast_id": "mla_moe_vs_moe",
        "contrast_family": "composition_mechanism",
        "model_a": "moe_220m",
        "model_b": "mla_moe_220m",
        "interpretation": "Effect of adding MLA attention inside the MoE setting.",
    },
    {
        "contrast_id": "v3_routing_vs_moe",
        "contrast_family": "routing_mechanism",
        "model_a": "moe_220m",
        "model_b": "v3_routing_220m",
        "interpretation": "Effect of V3-style routing relative to standard MoE.",
    },
    {
        "contrast_id": "mtp_vs_dense",
        "contrast_family": "objective_mechanism",
        "model_a": "dense_121m",
        "model_b": "mtp_121m",
        "interpretation": "Effect of MTP objective relative to next-token-only dense baseline.",
    },
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


def exact_sign_flip_p_value(differences: list[float]) -> float:
    nonzero_abs_differences = [abs(value) for value in differences if value != 0.0]

    if not nonzero_abs_differences:
        return 1.0

    observed_abs_mean = abs(statistics.fmean(differences))
    assignment_count = 0
    extreme_count = 0

    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero_abs_differences)):
        signed_values = [
            sign * value for sign, value in zip(signs, nonzero_abs_differences, strict=True)
        ]
        assigned_abs_mean = abs(statistics.fmean(signed_values))
        assignment_count += 1

        if assigned_abs_mean >= observed_abs_mean - 1e-15:
            extreme_count += 1

    return extreme_count / assignment_count


def bootstrap_mean_ci(
    differences: list[float],
    *,
    reps: int,
    seed_text: str,
) -> tuple[float, float]:
    rng = random.Random(f"{BOOTSTRAP_SEED}:{seed_text}")
    sample_size = len(differences)
    bootstrap_means = []

    for _ in range(reps):
        sample = [differences[rng.randrange(sample_size)] for _ in range(sample_size)]
        bootstrap_means.append(statistics.fmean(sample))

    bootstrap_means.sort()
    low_index = math.floor(0.025 * (reps - 1))
    high_index = math.ceil(0.975 * (reps - 1))
    return bootstrap_means[low_index], bootstrap_means[high_index]


def summarize_differences(
    differences: list[float],
    *,
    metric_direction: str,
) -> dict[str, Any]:
    pair_count = len(differences)

    mean_difference = statistics.fmean(differences)
    median_difference = statistics.median(differences)
    min_difference = min(differences)
    max_difference = max(differences)

    if pair_count <= 1:
        std_difference = None
        standard_error = None
        paired_t_ci95_low = None
        paired_t_ci95_high = None
        paired_cohens_dz = None
    else:
        std_difference_value = statistics.stdev(differences)
        std_difference = std_difference_value
        standard_error_value = std_difference_value / math.sqrt(pair_count)
        standard_error = standard_error_value
        paired_t_ci95_low = mean_difference - T_CRITICAL_975_DF9 * standard_error_value
        paired_t_ci95_high = mean_difference + T_CRITICAL_975_DF9 * standard_error_value
        paired_cohens_dz = (
            mean_difference / std_difference_value if std_difference_value != 0.0 else None
        )

    if metric_direction == "lower_is_better":
        model_b_better_seed_count = sum(1 for value in differences if value < 0.0)
        model_a_better_seed_count = sum(1 for value in differences if value > 0.0)
    elif metric_direction == "higher_is_better":
        model_b_better_seed_count = sum(1 for value in differences if value > 0.0)
        model_a_better_seed_count = sum(1 for value in differences if value < 0.0)
    else:
        raise ValueError(f"Unknown metric direction: {metric_direction}")

    tie_seed_count = sum(1 for value in differences if value == 0.0)

    return {
        "pair_count": pair_count,
        "mean_difference_model_b_minus_model_a": mean_difference,
        "median_difference_model_b_minus_model_a": median_difference,
        "std_difference": std_difference,
        "standard_error_difference": standard_error,
        "paired_t_ci95_low": paired_t_ci95_low,
        "paired_t_ci95_high": paired_t_ci95_high,
        "min_difference": min_difference,
        "max_difference": max_difference,
        "paired_cohens_dz": paired_cohens_dz,
        "model_b_better_seed_count": model_b_better_seed_count,
        "model_a_better_seed_count": model_a_better_seed_count,
        "tie_seed_count": tie_seed_count,
        "model_b_better_seed_fraction": model_b_better_seed_count / pair_count,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0 for _ in p_values]
    running_max = 0.0
    family_size = len(indexed)

    for rank_index, (original_index, p_value) in enumerate(indexed):
        multiplier = family_size - rank_index
        raw_adjusted = min(1.0, multiplier * p_value)
        running_max = max(running_max, raw_adjusted)
        adjusted[original_index] = running_max

    return adjusted


def add_holm_adjustments(
    contrast_rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
    output_key: str,
) -> None:
    grouped_indices: dict[tuple[Any, ...], list[int]] = {}

    for index, row in enumerate(contrast_rows):
        key = tuple(row[group_key] for group_key in group_keys)
        grouped_indices.setdefault(key, []).append(index)

    for indices in grouped_indices.values():
        p_values = [float(contrast_rows[index]["exact_sign_flip_p_value"]) for index in indices]
        adjusted_values = holm_adjust(p_values)

        for index, adjusted_value in zip(indices, adjusted_values, strict=True):
            contrast_rows[index][output_key] = adjusted_value


def main() -> None:
    rows = read_rows(INPUT_FLAT_CSV)
    scope_audit = validate_scope(rows)

    if not scope_audit["scope_passed"]:
        write_json(OUT_AUDIT_JSON, {"scope_audit": scope_audit})
        raise ValueError("Input flat artifact failed balanced 10-seed scope validation.")

    required_metric_fields = [f"metric__{metric}" for metric in TARGET_METRICS]
    missing_metric_fields = [field for field in required_metric_fields if field not in rows[0]]

    if missing_metric_fields:
        raise ValueError(f"Missing target metric fields: {missing_metric_fields}")

    row_by_key = {(row["model"], row["budget"], row["seed"]): row for row in rows}

    contrast_rows: list[dict[str, Any]] = []
    seed_record_rows: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, str]] = []

    for metric in TARGET_METRICS:
        metric_field = f"metric__{metric}"
        metric_direction = METRIC_DIRECTION[metric]

        for budget in EXPECTED_BUDGETS:
            for contrast in TARGET_CONTRASTS:
                model_a = str(contrast["model_a"])
                model_b = str(contrast["model_b"])
                differences = []
                model_a_values = []
                model_b_values = []

                for seed in EXPECTED_SEEDS:
                    row_a = row_by_key.get((model_a, budget, seed))
                    row_b = row_by_key.get((model_b, budget, seed))

                    if row_a is None or row_b is None:
                        missing_pairs.append(
                            {
                                "metric": metric,
                                "budget": budget,
                                "seed": seed,
                                "model_a": model_a,
                                "model_b": model_b,
                                "reason": "missing_model_seed_row",
                            }
                        )
                        continue

                    value_a = parse_float(row_a.get(metric_field))
                    value_b = parse_float(row_b.get(metric_field))

                    if value_a is None or value_b is None:
                        missing_pairs.append(
                            {
                                "metric": metric,
                                "budget": budget,
                                "seed": seed,
                                "model_a": model_a,
                                "model_b": model_b,
                                "reason": "missing_metric_value",
                            }
                        )
                        continue

                    difference = value_b - value_a
                    model_a_values.append(value_a)
                    model_b_values.append(value_b)
                    differences.append(difference)

                    if metric_direction == "lower_is_better":
                        model_b_better = difference < 0.0
                    else:
                        model_b_better = difference > 0.0

                    seed_record_rows.append(
                        {
                            "metric": metric,
                            "metric_group": METRIC_GROUP[metric],
                            "metric_direction": metric_direction,
                            "budget": budget,
                            "seed": seed,
                            "contrast_id": contrast["contrast_id"],
                            "contrast_family": contrast["contrast_family"],
                            "model_a": model_a,
                            "model_b": model_b,
                            "model_a_value": value_a,
                            "model_b_value": value_b,
                            "difference_model_b_minus_model_a": difference,
                            "model_b_better": model_b_better,
                        }
                    )

                if len(differences) != EXPECTED_PAIR_COUNT:
                    continue

                summary = summarize_differences(
                    differences,
                    metric_direction=metric_direction,
                )
                exact_p = exact_sign_flip_p_value(differences)
                bootstrap_low, bootstrap_high = bootstrap_mean_ci(
                    differences,
                    reps=BOOTSTRAP_REPS,
                    seed_text=f"{metric}:{budget}:{contrast['contrast_id']}",
                )

                model_a_mean = statistics.fmean(model_a_values)
                model_b_mean = statistics.fmean(model_b_values)
                mean_difference = float(summary["mean_difference_model_b_minus_model_a"])
                relative_change_percent = (
                    100.0 * mean_difference / abs(model_a_mean) if model_a_mean != 0.0 else None
                )

                contrast_rows.append(
                    {
                        "metric": metric,
                        "metric_group": METRIC_GROUP[metric],
                        "metric_direction": metric_direction,
                        "budget": budget,
                        "contrast_id": contrast["contrast_id"],
                        "contrast_family": contrast["contrast_family"],
                        "model_a": model_a,
                        "model_b": model_b,
                        "interpretation": contrast["interpretation"],
                        "model_a_mean": model_a_mean,
                        "model_b_mean": model_b_mean,
                        "relative_change_percent": relative_change_percent,
                        "exact_sign_flip_p_value": exact_p,
                        "bootstrap_reps": BOOTSTRAP_REPS,
                        "bootstrap_ci95_low": bootstrap_low,
                        "bootstrap_ci95_high": bootstrap_high,
                        **summary,
                    }
                )

    add_holm_adjustments(
        contrast_rows,
        group_keys=["metric", "budget"],
        output_key="holm_p_by_metric_budget",
    )
    add_holm_adjustments(
        contrast_rows,
        group_keys=["metric"],
        output_key="holm_p_by_metric_all_budgets",
    )
    add_holm_adjustments(
        contrast_rows,
        group_keys=["metric_group", "budget"],
        output_key="holm_p_by_metric_group_budget",
    )

    expected_contrast_rows = len(TARGET_METRICS) * len(EXPECTED_BUDGETS) * len(TARGET_CONTRASTS)
    expected_seed_record_rows = expected_contrast_rows * EXPECTED_PAIR_COUNT

    contrast_passed = (
        len(contrast_rows) == expected_contrast_rows
        and len(seed_record_rows) == expected_seed_record_rows
        and not missing_pairs
    )

    OUT_CONTRASTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    contrast_fieldnames = [
        "metric",
        "metric_group",
        "metric_direction",
        "budget",
        "contrast_id",
        "contrast_family",
        "model_a",
        "model_b",
        "interpretation",
        "pair_count",
        "model_a_mean",
        "model_b_mean",
        "mean_difference_model_b_minus_model_a",
        "median_difference_model_b_minus_model_a",
        "relative_change_percent",
        "std_difference",
        "standard_error_difference",
        "paired_t_ci95_low",
        "paired_t_ci95_high",
        "bootstrap_reps",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "min_difference",
        "max_difference",
        "paired_cohens_dz",
        "model_b_better_seed_count",
        "model_a_better_seed_count",
        "tie_seed_count",
        "model_b_better_seed_fraction",
        "exact_sign_flip_p_value",
        "holm_p_by_metric_budget",
        "holm_p_by_metric_all_budgets",
        "holm_p_by_metric_group_budget",
    ]

    with OUT_CONTRASTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=contrast_fieldnames)
        writer.writeheader()
        writer.writerows(contrast_rows)

    seed_record_fieldnames = [
        "metric",
        "metric_group",
        "metric_direction",
        "budget",
        "seed",
        "contrast_id",
        "contrast_family",
        "model_a",
        "model_b",
        "model_a_value",
        "model_b_value",
        "difference_model_b_minus_model_a",
        "model_b_better",
    ]

    with OUT_SEED_RECORDS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_record_fieldnames)
        writer.writeheader()
        writer.writerows(seed_record_rows)

    write_json(OUT_CONTRASTS_JSON, contrast_rows)
    write_json(OUT_SEED_RECORDS_JSON, seed_record_rows)

    audit = {
        "artifact_type": "balanced_10seed_matrix_paired_contrasts_audit",
        "input_flat_csv": str(INPUT_FLAT_CSV),
        "v1_controlled_design_guardrail": (
            "Paired contrasts are generated only from the balanced 10-seed flat "
            "artifact and only for pre-specified mechanism contrasts."
        ),
        "scope_audit": scope_audit,
        "target_metrics": TARGET_METRICS,
        "metric_direction": METRIC_DIRECTION,
        "target_contrasts": TARGET_CONTRASTS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "expected_contrast_rows": expected_contrast_rows,
        "contrast_row_count": len(contrast_rows),
        "expected_seed_record_rows": expected_seed_record_rows,
        "seed_record_row_count": len(seed_record_rows),
        "missing_pair_count": len(missing_pairs),
        "missing_pairs": missing_pairs[:20],
        "contrasts_csv": str(OUT_CONTRASTS_CSV),
        "contrasts_json": str(OUT_CONTRASTS_JSON),
        "seed_records_csv": str(OUT_SEED_RECORDS_CSV),
        "seed_records_json": str(OUT_SEED_RECORDS_JSON),
        "audit_json": str(OUT_AUDIT_JSON),
        "contrast_passed": contrast_passed,
    }
    write_json(OUT_AUDIT_JSON, audit)

    print(json.dumps(audit, indent=2, sort_keys=True))

    if not contrast_passed:
        raise ValueError("Paired contrast artifact validation failed. See audit JSON.")


if __name__ == "__main__":
    main()
