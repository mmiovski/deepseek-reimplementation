from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

INPUT_FLAT_CSV = Path("results/analysis/balanced_10seed_matrix_summary_flat.csv")
INPUT_DESCRIPTIVES_CSV = Path(
    "results/analysis/balanced_10seed_matrix_descriptives_full_precision.csv"
)

OUT_PROFILES_CSV = Path("results/analysis/balanced_10seed_matrix_mechanism_profiles.csv")
OUT_PROFILES_JSON = Path("results/analysis/balanced_10seed_matrix_mechanism_profiles.json")
OUT_METRIC_MAP_JSON = Path(
    "results/analysis/balanced_10seed_matrix_mechanism_profile_metric_map.json"
)
OUT_AUDIT_JSON = Path("results/analysis/balanced_10seed_matrix_mechanism_profiles_audit.json")

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

EXPECTED_FLAT_ROW_COUNT = 180
EXPECTED_PROFILE_ROW_COUNT = 18
EXPECTED_GROUP_SEED_COUNT = 10

MODEL_MECHANISMS: dict[str, dict[str, Any]] = {
    "dense_121m": {
        "mechanism_family": "dense_baseline",
        "attention_mechanism": "standard_attention",
        "ffn_mechanism": "dense_ffn",
        "routing_mechanism": "none",
        "objective_mechanism": "next_token_prediction",
        "has_mla": False,
        "has_moe": False,
        "has_v3_routing": False,
        "has_mtp": False,
    },
    "mla_121m": {
        "mechanism_family": "mla_attention",
        "attention_mechanism": "mla",
        "ffn_mechanism": "dense_ffn",
        "routing_mechanism": "none",
        "objective_mechanism": "next_token_prediction",
        "has_mla": True,
        "has_moe": False,
        "has_v3_routing": False,
        "has_mtp": False,
    },
    "mtp_121m": {
        "mechanism_family": "mtp_objective",
        "attention_mechanism": "standard_attention",
        "ffn_mechanism": "dense_ffn",
        "routing_mechanism": "none",
        "objective_mechanism": "multi_token_prediction",
        "has_mla": False,
        "has_moe": False,
        "has_v3_routing": False,
        "has_mtp": True,
    },
    "moe_220m": {
        "mechanism_family": "sparse_moe",
        "attention_mechanism": "standard_attention",
        "ffn_mechanism": "moe_ffn",
        "routing_mechanism": "standard_moe_routing",
        "objective_mechanism": "next_token_prediction",
        "has_mla": False,
        "has_moe": True,
        "has_v3_routing": False,
        "has_mtp": False,
    },
    "mla_moe_220m": {
        "mechanism_family": "mla_plus_moe",
        "attention_mechanism": "mla",
        "ffn_mechanism": "moe_ffn",
        "routing_mechanism": "standard_moe_routing",
        "objective_mechanism": "next_token_prediction",
        "has_mla": True,
        "has_moe": True,
        "has_v3_routing": False,
        "has_mtp": False,
    },
    "v3_routing_220m": {
        "mechanism_family": "v3_style_routing",
        "attention_mechanism": "standard_attention",
        "ffn_mechanism": "moe_ffn",
        "routing_mechanism": "v3_style_routing",
        "objective_mechanism": "next_token_prediction",
        "has_mla": False,
        "has_moe": True,
        "has_v3_routing": True,
        "has_mtp": False,
    },
}

PROFILE_METRIC_GROUPS: dict[str, list[str]] = {
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
    ],
    "parameterization": [
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

REQUIRED_PROFILE_METRICS = [
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input artifact: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float | None:
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


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        return None
    return numerator / denominator


def validate_flat_scope(rows: list[dict[str, str]]) -> dict[str, Any]:
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
        len(rows) == EXPECTED_FLAT_ROW_COUNT
        and models == sorted(EXPECTED_MODELS)
        and budgets == sorted(EXPECTED_BUDGETS)
        and seeds == sorted(EXPECTED_SEEDS)
        and not duplicate_cells
        and not bad_model_budget_counts
    )

    return {
        "row_count": len(rows),
        "expected_row_count": EXPECTED_FLAT_ROW_COUNT,
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
    for group_name, metrics in PROFILE_METRIC_GROUPS.items():
        if metric in metrics:
            return group_name
    return "other"


def build_descriptive_lookup(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}

    for row in rows:
        key = (row["metric"], row["budget"], row["model"])
        if key in lookup:
            raise ValueError(f"Duplicate descriptive key: {key}")
        lookup[key] = row

    return lookup


def metric_mean(profile_row: dict[str, Any], metric: str) -> float | None:
    return parse_float(profile_row.get(f"{metric}__mean"))


def main() -> None:
    flat_rows = read_csv(INPUT_FLAT_CSV)
    descriptive_rows = read_csv(INPUT_DESCRIPTIVES_CSV)

    flat_scope_audit = validate_flat_scope(flat_rows)
    if not flat_scope_audit["scope_passed"]:
        write_json(OUT_AUDIT_JSON, {"flat_scope_audit": flat_scope_audit})
        raise ValueError("Flat input failed balanced 10-seed scope validation.")

    descriptive_lookup = build_descriptive_lookup(descriptive_rows)
    all_profile_metrics = [
        metric for metrics in PROFILE_METRIC_GROUPS.values() for metric in metrics
    ]

    profile_rows: list[dict[str, Any]] = []
    profile_json: list[dict[str, Any]] = []
    missing_required_metric_groups: list[dict[str, Any]] = []
    missing_descriptive_keys: list[str] = []

    for budget in EXPECTED_BUDGETS:
        for model in EXPECTED_MODELS:
            mechanism = MODEL_MECHANISMS[model]

            profile_row: dict[str, Any] = {
                "model": model,
                "budget": budget,
                "budget_tokens": EXPECTED_BUDGET_TOKENS[budget],
                **mechanism,
            }

            nested_metrics: dict[str, dict[str, Any]] = {}

            for metric in all_profile_metrics:
                key = (metric, budget, model)
                descriptive_row = descriptive_lookup.get(key)

                if descriptive_row is None:
                    missing_descriptive_keys.append("|".join(key))
                    continue

                non_missing_seed_count = parse_int(descriptive_row["non_missing_seed_count"])
                mean_value = parse_float(descriptive_row["mean"])
                std_value = parse_float(descriptive_row["std"])
                ci95_low = parse_float(descriptive_row["ci95_low"])
                ci95_high = parse_float(descriptive_row["ci95_high"])

                profile_row[f"{metric}__non_missing_seed_count"] = non_missing_seed_count
                profile_row[f"{metric}__mean"] = mean_value
                profile_row[f"{metric}__std"] = std_value
                profile_row[f"{metric}__ci95_low"] = ci95_low
                profile_row[f"{metric}__ci95_high"] = ci95_high

                nested_metrics[metric] = {
                    "metric_group": metric_group_for(metric),
                    "non_missing_seed_count": non_missing_seed_count,
                    "mean": mean_value,
                    "std": std_value,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                }

                if (
                    metric in REQUIRED_PROFILE_METRICS
                    and non_missing_seed_count != EXPECTED_GROUP_SEED_COUNT
                ):
                    missing_required_metric_groups.append(
                        {
                            "metric": metric,
                            "budget": budget,
                            "model": model,
                            "non_missing_seed_count": non_missing_seed_count,
                        }
                    )

            total_parameters = metric_mean(profile_row, "total_parameters")
            trainable_parameters = metric_mean(profile_row, "trainable_parameters")
            activated_parameters = metric_mean(
                profile_row,
                "activated_parameters_per_token",
            )
            peak_memory_bytes = metric_mean(profile_row, "peak_memory_bytes")
            validation_loss = metric_mean(profile_row, "validation_loss")
            test_loss = metric_mean(profile_row, "test_loss")

            profile_row["activated_to_total_parameter_ratio_mean"] = safe_divide(
                activated_parameters,
                total_parameters,
            )
            profile_row["activated_to_trainable_parameter_ratio_mean"] = safe_divide(
                activated_parameters,
                trainable_parameters,
            )
            profile_row["total_to_activated_parameter_ratio_mean"] = safe_divide(
                total_parameters,
                activated_parameters,
            )
            profile_row["peak_memory_gib_mean"] = (
                peak_memory_bytes / (1024.0**3) if peak_memory_bytes is not None else None
            )
            profile_row["test_minus_validation_loss_gap_mean"] = (
                test_loss - validation_loss
                if test_loss is not None and validation_loss is not None
                else None
            )

            profile_rows.append(profile_row)
            profile_json.append(
                {
                    "model": model,
                    "budget": budget,
                    "budget_tokens": EXPECTED_BUDGET_TOKENS[budget],
                    "mechanism": mechanism,
                    "derived": {
                        "activated_to_total_parameter_ratio_mean": profile_row[
                            "activated_to_total_parameter_ratio_mean"
                        ],
                        "activated_to_trainable_parameter_ratio_mean": profile_row[
                            "activated_to_trainable_parameter_ratio_mean"
                        ],
                        "total_to_activated_parameter_ratio_mean": profile_row[
                            "total_to_activated_parameter_ratio_mean"
                        ],
                        "peak_memory_gib_mean": profile_row["peak_memory_gib_mean"],
                        "test_minus_validation_loss_gap_mean": profile_row[
                            "test_minus_validation_loss_gap_mean"
                        ],
                    },
                    "metrics": nested_metrics,
                }
            )

    base_fields = [
        "model",
        "budget",
        "budget_tokens",
        "mechanism_family",
        "attention_mechanism",
        "ffn_mechanism",
        "routing_mechanism",
        "objective_mechanism",
        "has_mla",
        "has_moe",
        "has_v3_routing",
        "has_mtp",
        "activated_to_total_parameter_ratio_mean",
        "activated_to_trainable_parameter_ratio_mean",
        "total_to_activated_parameter_ratio_mean",
        "peak_memory_gib_mean",
        "test_minus_validation_loss_gap_mean",
    ]

    metric_fields = []
    for metric in all_profile_metrics:
        metric_fields.extend(
            [
                f"{metric}__non_missing_seed_count",
                f"{metric}__mean",
                f"{metric}__std",
                f"{metric}__ci95_low",
                f"{metric}__ci95_high",
            ]
        )

    fieldnames = base_fields + metric_fields

    profile_passed = (
        len(profile_rows) == EXPECTED_PROFILE_ROW_COUNT
        and not missing_descriptive_keys
        and not missing_required_metric_groups
    )

    OUT_PROFILES_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PROFILES_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(profile_rows)

    write_json(OUT_PROFILES_JSON, profile_json)
    write_json(
        OUT_METRIC_MAP_JSON,
        {
            "profile_metric_groups": PROFILE_METRIC_GROUPS,
            "required_profile_metrics": REQUIRED_PROFILE_METRICS,
            "derived_fields": [
                "activated_to_total_parameter_ratio_mean",
                "activated_to_trainable_parameter_ratio_mean",
                "total_to_activated_parameter_ratio_mean",
                "peak_memory_gib_mean",
                "test_minus_validation_loss_gap_mean",
            ],
            "model_mechanisms": MODEL_MECHANISMS,
        },
    )

    audit = {
        "artifact_type": "balanced_10seed_matrix_mechanism_profiles_audit",
        "input_flat_csv": str(INPUT_FLAT_CSV),
        "input_descriptives_csv": str(INPUT_DESCRIPTIVES_CSV),
        "v1_controlled_design_guardrail": (
            "Mechanism profiles are built only from the balanced 10-seed flat "
            "artifact and validated model-by-budget descriptives."
        ),
        "flat_scope_audit": flat_scope_audit,
        "profile_row_count": len(profile_rows),
        "expected_profile_row_count": EXPECTED_PROFILE_ROW_COUNT,
        "metric_count": len(all_profile_metrics),
        "profile_metrics": all_profile_metrics,
        "required_profile_metrics": REQUIRED_PROFILE_METRICS,
        "missing_descriptive_key_count": len(missing_descriptive_keys),
        "missing_descriptive_keys": missing_descriptive_keys[:20],
        "missing_required_metric_group_count": len(missing_required_metric_groups),
        "missing_required_metric_groups": missing_required_metric_groups,
        "profiles_csv": str(OUT_PROFILES_CSV),
        "profiles_json": str(OUT_PROFILES_JSON),
        "metric_map_json": str(OUT_METRIC_MAP_JSON),
        "audit_json": str(OUT_AUDIT_JSON),
        "profile_passed": profile_passed,
    }
    write_json(OUT_AUDIT_JSON, audit)

    print(json.dumps(audit, indent=2, sort_keys=True))

    if not profile_passed:
        raise ValueError("Mechanism profile artifact validation failed. See audit JSON.")


if __name__ == "__main__":
    main()
