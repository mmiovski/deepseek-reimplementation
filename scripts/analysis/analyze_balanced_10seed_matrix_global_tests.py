from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

INPUT_FLAT_CSV = Path("results/analysis/balanced_10seed_matrix_summary_flat.csv")

OUT_GLOBAL_TESTS_CSV = Path(
    "results/analysis/balanced_10seed_matrix_global_tests_full_precision.csv"
)
OUT_GLOBAL_TESTS_JSON = Path(
    "results/analysis/balanced_10seed_matrix_global_tests_full_precision.json"
)
OUT_GLOBAL_SEED_RECORDS_CSV = Path(
    "results/analysis/balanced_10seed_matrix_global_seed_records.csv"
)
OUT_GLOBAL_SEED_RECORDS_JSON = Path(
    "results/analysis/balanced_10seed_matrix_global_seed_records.json"
)
OUT_AUDIT_JSON = Path("results/analysis/balanced_10seed_matrix_global_tests_audit.json")

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
EXPECTED_MODEL_COUNT = 6
EXPECTED_SEED_COUNT = 10

PERMUTATION_REPS = 20_000
PERMUTATION_SEED = 20260723

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


def repeated_measures_anova_f(matrix: list[list[float]]) -> tuple[float, float, float]:
    seed_count = len(matrix)
    model_count = len(matrix[0])
    values = [value for row in matrix for value in row]
    grand_total = sum(values)
    observation_count = seed_count * model_count
    correction = grand_total**2 / observation_count

    total_ss = sum(value**2 for value in values) - correction
    seed_ss = sum(sum(row) ** 2 / model_count for row in matrix) - correction
    model_totals = [
        sum(matrix[seed_index][model_index] for seed_index in range(seed_count))
        for model_index in range(model_count)
    ]
    model_ss = sum(total**2 / seed_count for total in model_totals) - correction
    error_ss = total_ss - seed_ss - model_ss

    model_df = model_count - 1
    error_df = (model_count - 1) * (seed_count - 1)

    model_ms = model_ss / model_df
    error_ms = error_ss / error_df

    if error_ms <= 0.0:
        return math.inf, model_ss, error_ss

    return model_ms / error_ms, model_ss, error_ss


def rank_values(values: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    index = 0

    while index < len(sorted_pairs):
        end_index = index + 1
        while (
            end_index < len(sorted_pairs) and sorted_pairs[end_index][1] == sorted_pairs[index][1]
        ):
            end_index += 1

        average_rank = (index + 1 + end_index) / 2.0

        for sorted_index in range(index, end_index):
            original_index = sorted_pairs[sorted_index][0]
            ranks[original_index] = average_rank

        index = end_index

    return ranks


def friedman_q(matrix: list[list[float]]) -> float:
    seed_count = len(matrix)
    model_count = len(matrix[0])
    rank_sums = [0.0 for _ in range(model_count)]

    for row in matrix:
        ranks = rank_values(row)
        for model_index, rank in enumerate(ranks):
            rank_sums[model_index] += rank

    return 12.0 / (seed_count * model_count * (model_count + 1)) * sum(
        rank_sum**2 for rank_sum in rank_sums
    ) - 3.0 * seed_count * (model_count + 1)


def permutation_p_values(
    matrix: list[list[float]],
    *,
    metric: str,
    budget: str,
) -> tuple[float, float]:
    observed_f, _, _ = repeated_measures_anova_f(matrix)
    observed_q = friedman_q(matrix)

    rng = random.Random(f"{PERMUTATION_SEED}:{metric}:{budget}")
    anova_extreme_count = 0
    friedman_extreme_count = 0

    for _ in range(PERMUTATION_REPS):
        permuted_matrix = [rng.sample(row, len(row)) for row in matrix]

        permuted_f, _, _ = repeated_measures_anova_f(permuted_matrix)
        permuted_q = friedman_q(permuted_matrix)

        if permuted_f >= observed_f - 1e-15:
            anova_extreme_count += 1

        if permuted_q >= observed_q - 1e-15:
            friedman_extreme_count += 1

    return (
        (anova_extreme_count + 1.0) / (PERMUTATION_REPS + 1.0),
        (friedman_extreme_count + 1.0) / (PERMUTATION_REPS + 1.0),
    )


def model_means(matrix: list[list[float]]) -> dict[str, float]:
    output = {}

    for model_index, model in enumerate(EXPECTED_MODELS):
        output[f"{model}__mean"] = statistics.fmean(row[model_index] for row in matrix)

    return output


def model_rank_means(matrix: list[list[float]]) -> dict[str, float]:
    rank_values_by_model: dict[str, list[float]] = {model: [] for model in EXPECTED_MODELS}

    for row in matrix:
        ranks = rank_values(row)
        for model, rank in zip(EXPECTED_MODELS, ranks, strict=True):
            rank_values_by_model[model].append(rank)

    return {
        f"{model}__mean_within_seed_rank": statistics.fmean(ranks)
        for model, ranks in rank_values_by_model.items()
    }


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

    global_test_rows: list[dict[str, Any]] = []
    seed_record_rows: list[dict[str, Any]] = []
    missing_values: list[dict[str, str]] = []

    for metric in TARGET_METRICS:
        metric_field = f"metric__{metric}"

        for budget in EXPECTED_BUDGETS:
            matrix: list[list[float]] = []

            for seed in EXPECTED_SEEDS:
                seed_values = []

                for model in EXPECTED_MODELS:
                    row = row_by_key.get((model, budget, seed))

                    if row is None:
                        missing_values.append(
                            {
                                "metric": metric,
                                "budget": budget,
                                "seed": seed,
                                "model": model,
                                "reason": "missing_model_budget_seed_row",
                            }
                        )
                        continue

                    value = parse_float(row.get(metric_field))

                    if value is None:
                        missing_values.append(
                            {
                                "metric": metric,
                                "budget": budget,
                                "seed": seed,
                                "model": model,
                                "reason": "missing_metric_value",
                            }
                        )
                        continue

                    seed_values.append(value)
                    seed_record_rows.append(
                        {
                            "metric": metric,
                            "metric_group": METRIC_GROUP[metric],
                            "metric_direction": METRIC_DIRECTION[metric],
                            "budget": budget,
                            "seed": seed,
                            "model": model,
                            "value": value,
                        }
                    )

                if len(seed_values) == EXPECTED_MODEL_COUNT:
                    matrix.append(seed_values)

            if len(matrix) != EXPECTED_SEED_COUNT:
                continue

            anova_f, anova_model_ss, anova_error_ss = repeated_measures_anova_f(matrix)
            friedman_statistic = friedman_q(matrix)
            anova_permutation_p, friedman_permutation_p = permutation_p_values(
                matrix,
                metric=metric,
                budget=budget,
            )

            global_test_rows.append(
                {
                    "metric": metric,
                    "metric_group": METRIC_GROUP[metric],
                    "metric_direction": METRIC_DIRECTION[metric],
                    "budget": budget,
                    "seed_count": EXPECTED_SEED_COUNT,
                    "model_count": EXPECTED_MODEL_COUNT,
                    "model_degrees_of_freedom": EXPECTED_MODEL_COUNT - 1,
                    "error_degrees_of_freedom": (
                        (EXPECTED_MODEL_COUNT - 1) * (EXPECTED_SEED_COUNT - 1)
                    ),
                    "repeated_measures_anova_f": anova_f,
                    "repeated_measures_anova_model_ss": anova_model_ss,
                    "repeated_measures_anova_error_ss": anova_error_ss,
                    "repeated_measures_anova_permutation_reps": PERMUTATION_REPS,
                    "repeated_measures_anova_permutation_p_value": (anova_permutation_p),
                    "friedman_q": friedman_statistic,
                    "friedman_permutation_reps": PERMUTATION_REPS,
                    "friedman_permutation_p_value": friedman_permutation_p,
                    **model_means(matrix),
                    **model_rank_means(matrix),
                }
            )

    expected_global_test_rows = len(TARGET_METRICS) * len(EXPECTED_BUDGETS)
    expected_seed_record_rows = (
        len(TARGET_METRICS) * len(EXPECTED_BUDGETS) * len(EXPECTED_SEEDS) * len(EXPECTED_MODELS)
    )

    global_tests_passed = (
        len(global_test_rows) == expected_global_test_rows
        and len(seed_record_rows) == expected_seed_record_rows
        and not missing_values
    )

    OUT_GLOBAL_TESTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    base_fieldnames = [
        "metric",
        "metric_group",
        "metric_direction",
        "budget",
        "seed_count",
        "model_count",
        "model_degrees_of_freedom",
        "error_degrees_of_freedom",
        "repeated_measures_anova_f",
        "repeated_measures_anova_model_ss",
        "repeated_measures_anova_error_ss",
        "repeated_measures_anova_permutation_reps",
        "repeated_measures_anova_permutation_p_value",
        "friedman_q",
        "friedman_permutation_reps",
        "friedman_permutation_p_value",
    ]
    model_mean_fields = [f"{model}__mean" for model in EXPECTED_MODELS]
    model_rank_fields = [f"{model}__mean_within_seed_rank" for model in EXPECTED_MODELS]
    global_test_fieldnames = base_fieldnames + model_mean_fields + model_rank_fields

    seed_record_fieldnames = [
        "metric",
        "metric_group",
        "metric_direction",
        "budget",
        "seed",
        "model",
        "value",
    ]

    with OUT_GLOBAL_TESTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=global_test_fieldnames)
        writer.writeheader()
        writer.writerows(global_test_rows)

    with OUT_GLOBAL_SEED_RECORDS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_record_fieldnames)
        writer.writeheader()
        writer.writerows(seed_record_rows)

    write_json(OUT_GLOBAL_TESTS_JSON, global_test_rows)
    write_json(OUT_GLOBAL_SEED_RECORDS_JSON, seed_record_rows)

    audit = {
        "artifact_type": "balanced_10seed_matrix_global_tests_audit",
        "input_flat_csv": str(INPUT_FLAT_CSV),
        "scope_audit": scope_audit,
        "target_metrics": TARGET_METRICS,
        "metric_direction": METRIC_DIRECTION,
        "permutation_reps": PERMUTATION_REPS,
        "permutation_seed": PERMUTATION_SEED,
        "expected_global_test_rows": expected_global_test_rows,
        "global_test_row_count": len(global_test_rows),
        "expected_seed_record_rows": expected_seed_record_rows,
        "seed_record_row_count": len(seed_record_rows),
        "missing_value_count": len(missing_values),
        "missing_values": missing_values[:20],
        "global_tests_csv": str(OUT_GLOBAL_TESTS_CSV),
        "global_tests_json": str(OUT_GLOBAL_TESTS_JSON),
        "global_seed_records_csv": str(OUT_GLOBAL_SEED_RECORDS_CSV),
        "global_seed_records_json": str(OUT_GLOBAL_SEED_RECORDS_JSON),
        "audit_json": str(OUT_AUDIT_JSON),
        "global_tests_passed": global_tests_passed,
    }
    write_json(OUT_AUDIT_JSON, audit)

    print(json.dumps(audit, indent=2, sort_keys=True))

    if not global_tests_passed:
        raise ValueError("Global test artifact validation failed. See audit JSON.")


if __name__ == "__main__":
    main()
