from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

INPUT_CSV = Path("results/analysis/main_large_50m_25seed_summary_flat.csv")
OUT_DIR = Path("results/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["dense_121m", "mtp_121m", "moe_220m", "v3_routing_220m"]

PRIMARY_METRICS = [
    "validation_loss",
    "test_loss",
    "validation_perplexity",
    "test_perplexity",
]

EFFICIENCY_METRICS = [
    "train_tokens_per_second",
    "peak_memory_bytes",
    "tokens_per_total_parameter",
    "tokens_per_trainable_parameter",
    "tokens_per_activated_parameter",
    "requested_tokens_per_total_parameter",
    "requested_tokens_per_trainable_parameter",
    "requested_tokens_per_activated_parameter",
]

ROUTING_METRICS = [
    "routing_stats.mean_aux_loss",
    "routing_stats.mean_expert_load_variance",
    "routing_stats.mean_router_probability",
    "routing_stats.expert_bias_std",
]

METRICS = PRIMARY_METRICS + EFFICIENCY_METRICS + ROUTING_METRICS

LOWER_IS_BETTER = {
    "validation_loss": True,
    "test_loss": True,
    "validation_perplexity": True,
    "test_perplexity": True,
    "peak_memory_bytes": True,
    "routing_stats.mean_aux_loss": True,
    "routing_stats.mean_expert_load_variance": True,
    "routing_stats.expert_bias_std": True,
    "train_tokens_per_second": False,
    "tokens_per_total_parameter": False,
    "tokens_per_trainable_parameter": False,
    "tokens_per_activated_parameter": False,
    "requested_tokens_per_total_parameter": False,
    "requested_tokens_per_trainable_parameter": False,
    "requested_tokens_per_activated_parameter": False,
    "routing_stats.mean_router_probability": False,
}

PLANNED_CONTRASTS = [
    ("mtp_121m", "dense_121m"),
    ("moe_220m", "dense_121m"),
    ("v3_routing_220m", "dense_121m"),
    ("v3_routing_220m", "moe_220m"),
    ("mtp_121m", "moe_220m"),
    ("mtp_121m", "v3_routing_220m"),
]

RNG = np.random.default_rng(20260707)
BOOTSTRAP_REPS = 20000
SIGN_FLIP_REPS = 20000


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * m
    running_max = 0.0

    for rank, (idx, p_value) in enumerate(indexed, start=1):
        if math.isnan(p_value):
            adjusted[idx] = math.nan
            continue
        candidate = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max

    return adjusted


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return math.nan, math.nan

    samples = RNG.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_p_value(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return math.nan

    observed = abs(float(values.mean()))
    signs = RNG.choice([-1.0, 1.0], size=(SIGN_FLIP_REPS, len(values)), replace=True)
    perm_means = np.abs((signs * values).mean(axis=1))
    return float((np.sum(perm_means >= observed) + 1) / (SIGN_FLIP_REPS + 1))


def rounded_value(value: Any, decimals: int) -> Any:
    try:
        if value is None:
            return value
        numeric = float(value)
        if math.isnan(numeric):
            return value
        return round(numeric, decimals)
    except (TypeError, ValueError):
        return value


def report_rounding(df: pd.DataFrame) -> pd.DataFrame:
    rounded = df.copy()

    for column in rounded.columns:
        lower = column.lower()

        if "p_value" in lower or lower.endswith("_p") or "pvalue" in lower:
            rounded[column] = rounded[column].map(lambda x: rounded_value(x, 6))
        elif "loss" in lower or "perplexity" in lower:
            rounded[column] = rounded[column].map(lambda x: rounded_value(x, 5))
        elif "effect" in lower or "cohen" in lower:
            rounded[column] = rounded[column].map(lambda x: rounded_value(x, 3))
        elif "memory" in lower and "bytes" in lower:
            rounded[column] = rounded[column].map(lambda x: rounded_value(x, 0))
        elif "tokens_per_second" in lower or "throughput" in lower:
            rounded[column] = rounded[column].map(lambda x: rounded_value(x, 1))
        elif "ratio" in lower or "parameter" in lower:
            rounded[column] = rounded[column].map(lambda x: rounded_value(x, 3))
        elif any(token in lower for token in ["mean", "std", "sem", "ci_", "delta"]):
            rounded[column] = rounded[column].map(lambda x: rounded_value(x, 5))

    return rounded


def descriptive_stats(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        for model in MODEL_ORDER:
            values = df.loc[df["short_name"] == model, metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue

            sd = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            sem = sd / math.sqrt(len(values)) if len(values) > 1 else math.nan
            tcrit = float(stats.t.ppf(0.975, len(values) - 1)) if len(values) > 1 else math.nan
            mean = float(np.mean(values))
            ci_low = mean - tcrit * sem if len(values) > 1 else math.nan
            ci_high = mean + tcrit * sem if len(values) > 1 else math.nan

            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "n": int(len(values)),
                    "mean": mean,
                    "std": sd,
                    "sem": sem,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )

    return pd.DataFrame(rows)


def global_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        wide = df.pivot(index="seed", columns="short_name", values=metric)
        wide = wide.reindex(columns=MODEL_ORDER).dropna()

        row: dict[str, Any] = {
            "metric": metric,
            "n_seed_blocks": int(len(wide)),
            "models": ",".join(MODEL_ORDER),
        }

        if len(wide) >= 2:
            long_df = wide.reset_index().melt(
                id_vars="seed",
                value_vars=MODEL_ORDER,
                var_name="model",
                value_name="value",
            )

            try:
                anova = AnovaRM(
                    long_df,
                    depvar="value",
                    subject="seed",
                    within=["model"],
                ).fit()
                anova_table = anova.anova_table.reset_index()
                model_row = anova_table.iloc[0]
                row["repeated_measures_anova_f"] = float(model_row["F Value"])
                row["repeated_measures_anova_df_num"] = float(model_row["Num DF"])
                row["repeated_measures_anova_df_den"] = float(model_row["Den DF"])
                row["repeated_measures_anova_p_value"] = float(model_row["Pr > F"])
            except Exception as exc:
                row["repeated_measures_anova_error"] = repr(exc)

            try:
                friedman = stats.friedmanchisquare(
                    *[wide[model].to_numpy(dtype=float) for model in MODEL_ORDER]
                )
                row["friedman_chi_square"] = float(friedman.statistic)
                row["friedman_p_value"] = float(friedman.pvalue)
            except Exception as exc:
                row["friedman_error"] = repr(exc)

        rows.append(row)

    return pd.DataFrame(rows)


def paired_contrasts(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        wide = df.pivot(index="seed", columns="short_name", values=metric)
        wide = wide.reindex(columns=MODEL_ORDER)

        metric_rows: list[dict[str, Any]] = []

        for model_a, model_b in PLANNED_CONTRASTS:
            paired = wide[[model_a, model_b]].dropna()
            a = paired[model_a].to_numpy(dtype=float)
            b = paired[model_b].to_numpy(dtype=float)
            diff = a - b

            n = len(diff)
            if n < 2:
                continue

            mean_a = float(np.mean(a))
            mean_b = float(np.mean(b))
            mean_diff = float(np.mean(diff))
            sd_diff = float(np.std(diff, ddof=1))
            sem_diff = sd_diff / math.sqrt(n)
            tcrit = float(stats.t.ppf(0.975, n - 1))
            ci_low = mean_diff - tcrit * sem_diff
            ci_high = mean_diff + tcrit * sem_diff
            boot_low, boot_high = bootstrap_mean_ci(diff)

            ttest = stats.ttest_1samp(diff, 0.0)

            try:
                wilcoxon = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                wilcoxon_stat = float(wilcoxon.statistic)
                wilcoxon_p = float(wilcoxon.pvalue)
            except Exception:
                wilcoxon_stat = math.nan
                wilcoxon_p = math.nan

            cohen_dz = mean_diff / sd_diff if sd_diff != 0 else math.nan

            lower_is_better = LOWER_IS_BETTER.get(metric, True)
            improvement = diff < 0 if lower_is_better else diff > 0

            metric_rows.append(
                {
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "comparison": f"{model_a} - {model_b}",
                    "n_seed_blocks": int(n),
                    "mean_model_a": mean_a,
                    "mean_model_b": mean_b,
                    "mean_difference_a_minus_b": mean_diff,
                    "std_difference": sd_diff,
                    "sem_difference": sem_diff,
                    "ci95_low_difference": ci_low,
                    "ci95_high_difference": ci_high,
                    "bootstrap_ci95_low_difference": boot_low,
                    "bootstrap_ci95_high_difference": boot_high,
                    "paired_t_statistic": float(ttest.statistic),
                    "paired_t_p_value": float(ttest.pvalue),
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p_value": wilcoxon_p,
                    "sign_flip_p_value": sign_flip_p_value(diff),
                    "cohen_dz": cohen_dz,
                    "lower_is_better": lower_is_better,
                    "model_a_improvement_count": int(np.sum(improvement)),
                    "model_b_or_tie_count": int(n - np.sum(improvement)),
                    "model_a_improvement_fraction": float(np.mean(improvement)),
                }
            )

        p_values = [float(row["paired_t_p_value"]) for row in metric_rows]
        adjusted = holm_adjust(p_values)
        for row, adjusted_p in zip(metric_rows, adjusted, strict=True):
            row["paired_t_p_value_holm_within_metric"] = adjusted_p

        rows.extend(metric_rows)

    return pd.DataFrame(rows)


def rankings(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        desc = descriptive_stats(df, [metric])
        if desc.empty:
            continue

        lower_is_better = LOWER_IS_BETTER.get(metric, True)
        desc = desc.sort_values("mean", ascending=lower_is_better).reset_index(drop=True)

        for idx, row in desc.iterrows():
            rows.append(
                {
                    "metric": metric,
                    "rank": int(idx + 1),
                    "model": row["model"],
                    "mean": row["mean"],
                    "ci95_low": row["ci95_low"],
                    "ci95_high": row["ci95_high"],
                    "lower_is_better": lower_is_better,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    available_metrics = [metric for metric in METRICS if metric in df.columns]
    df = coerce_numeric(df, ["seed", *available_metrics])

    # Keep only the intended 25-seed, 4-model paired subset.
    df = df[df["short_name"].isin(MODEL_ORDER)].copy()

    wide_counts = df.pivot_table(
        index="seed",
        columns="short_name",
        values="experiment_name",
        aggfunc="count",
        fill_value=0,
    ).reindex(columns=MODEL_ORDER)

    bad_blocks = wide_counts[wide_counts.isna().any(axis=1) | (wide_counts != 1).any(axis=1)]

    if len(df) != 100:
        raise RuntimeError(f"Expected 100 rows, found {len(df)}")
    if not bad_blocks.empty:
        raise RuntimeError(f"Bad seed blocks detected:\n{bad_blocks}")

    descriptives = descriptive_stats(df, available_metrics)
    globals_df = global_tests(df, PRIMARY_METRICS)
    contrasts = paired_contrasts(df, PRIMARY_METRICS)
    ranks = rankings(df, PRIMARY_METRICS + ["train_tokens_per_second", "peak_memory_bytes"])

    outputs = {
        "descriptives": descriptives,
        "global_tests": globals_df,
        "paired_contrasts": contrasts,
        "rankings": ranks,
    }

    for name, table in outputs.items():
        full_csv = OUT_DIR / f"main_large_50m_25seed_{name}_full_precision.csv"
        full_json = OUT_DIR / f"main_large_50m_25seed_{name}_full_precision.json"
        rounded_csv = OUT_DIR / f"main_large_50m_25seed_{name}_rounded.csv"
        rounded_json = OUT_DIR / f"main_large_50m_25seed_{name}_rounded.json"

        table.to_csv(full_csv, index=False)
        full_json.write_text(
            json.dumps(table.to_dict(orient="records"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        rounded = report_rounding(table)
        rounded.to_csv(rounded_csv, index=False)
        rounded_json.write_text(
            json.dumps(rounded.to_dict(orient="records"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summary = {
        "input_csv": str(INPUT_CSV),
        "row_count": int(len(df)),
        "seed_count": int(df["seed"].nunique()),
        "model_order": MODEL_ORDER,
        "primary_metrics": PRIMARY_METRICS,
        "efficiency_metrics": EFFICIENCY_METRICS,
        "routing_metrics": ROUTING_METRICS,
        "planned_contrasts": [
            {"model_a": a, "model_b": b, "difference": f"{a} - {b}"} for a, b in PLANNED_CONTRASTS
        ],
        "bootstrap_reps": BOOTSTRAP_REPS,
        "sign_flip_reps": SIGN_FLIP_REPS,
        "global_test_rows": int(len(globals_df)),
        "paired_contrast_rows": int(len(contrasts)),
        "descriptive_rows": int(len(descriptives)),
        "ranking_rows": int(len(ranks)),
        "rounding_policy": {
            "loss_and_perplexity": 5,
            "p_values": 6,
            "effect_sizes": 3,
            "throughput": 1,
            "memory_bytes": 0,
            "parameter_ratios": 3,
        },
        "outputs": {
            name: {
                "full_precision_csv": str(
                    OUT_DIR / f"main_large_50m_25seed_{name}_full_precision.csv"
                ),
                "rounded_csv": str(OUT_DIR / f"main_large_50m_25seed_{name}_rounded.csv"),
            }
            for name in outputs
        },
    }

    summary_path = OUT_DIR / "main_large_50m_25seed_statistical_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
