from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

INPUT_DESCRIPTIVES_CSV = Path(
    "results/analysis/balanced_10seed_matrix_descriptives_full_precision.csv"
)

OUT_DIR = Path("results/figures/balanced_10seed_matrix")
OUT_MANIFEST_CSV = OUT_DIR / "balanced_10seed_matrix_plot_manifest.csv"
OUT_MANIFEST_JSON = OUT_DIR / "balanced_10seed_matrix_plot_manifest.json"
OUT_AUDIT_JSON = OUT_DIR / "balanced_10seed_matrix_plot_audit.json"

EXPECTED_MODELS = [
    "dense_121m",
    "mla_121m",
    "mtp_121m",
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
]

MOE_MODELS = [
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
]

MTP_MODELS = [
    "mtp_121m",
]

EXPECTED_BUDGETS = ["10m", "25m", "50m"]

EXPECTED_BUDGET_TOKENS = {
    "10m": 10_000_000,
    "25m": 25_000_000,
    "50m": 50_000_000,
}

EXPECTED_SEED_COUNT = 10
EXPECTED_DESCRIPTIVE_ROWS = 468

MODEL_LABELS = {
    "dense_121m": "Dense",
    "mla_121m": "MLA",
    "mtp_121m": "MTP",
    "moe_220m": "MoE",
    "mla_moe_220m": "MLA+MoE",
    "v3_routing_220m": "V3 Routing",
}

# Okabe-Ito color-blind-friendly palette.
MODEL_COLORS = {
    "dense_121m": "#0072B2",
    "mla_121m": "#D55E00",
    "mtp_121m": "#009E73",
    "moe_220m": "#CC79A7",
    "mla_moe_220m": "#E69F00",
    "v3_routing_220m": "#56B4E9",
}

MODEL_MARKERS = {
    "dense_121m": "o",
    "mla_121m": "s",
    "mtp_121m": "^",
    "moe_220m": "D",
    "mla_moe_220m": "P",
    "v3_routing_220m": "X",
}

MODEL_LINESTYLES = {
    "dense_121m": "-",
    "mla_121m": "--",
    "mtp_121m": "-.",
    "moe_220m": ":",
    "mla_moe_220m": "-",
    "v3_routing_220m": "--",
}

PLOT_SPECS: list[dict[str, Any]] = [
    {
        "metric": "validation_loss",
        "filename": "validation_loss_by_budget.png",
        "title": "Validation Loss by Token Budget",
        "ylabel": "Validation loss",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "test_loss",
        "filename": "test_loss_by_budget.png",
        "title": "Test Loss by Token Budget",
        "ylabel": "Test loss",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "validation_perplexity",
        "filename": "validation_perplexity_by_budget.png",
        "title": "Validation Perplexity by Token Budget",
        "ylabel": "Validation perplexity",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "test_perplexity",
        "filename": "test_perplexity_by_budget.png",
        "title": "Test Perplexity by Token Budget",
        "ylabel": "Test perplexity",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "train_loss",
        "filename": "train_loss_by_budget.png",
        "title": "Training Loss by Token Budget",
        "ylabel": "Training loss",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "train_tokens_per_second",
        "filename": "train_tokens_per_second_by_budget.png",
        "title": "Training Throughput by Token Budget",
        "ylabel": "Tokens per second",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "peak_memory_bytes",
        "filename": "peak_memory_gib_by_budget.png",
        "title": "Peak Memory by Token Budget",
        "ylabel": "Peak memory (GiB)",
        "scale": 1.0 / (1024.0**3),
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "tokens_per_total_parameter",
        "filename": "tokens_per_total_parameter_by_budget.png",
        "title": "Tokens per Total Parameter by Token Budget",
        "ylabel": "Tokens / total parameter",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "tokens_per_trainable_parameter",
        "filename": "tokens_per_trainable_parameter_by_budget.png",
        "title": "Tokens per Trainable Parameter by Token Budget",
        "ylabel": "Tokens / trainable parameter",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "tokens_per_activated_parameter",
        "filename": "tokens_per_activated_parameter_by_budget.png",
        "title": "Tokens per Activated Parameter by Token Budget",
        "ylabel": "Tokens / activated parameter",
        "scale": 1.0,
        "models": EXPECTED_MODELS,
        "required": True,
    },
    {
        "metric": "mean_aux_loss",
        "filename": "mean_aux_loss_by_budget.png",
        "title": "Mean Auxiliary Loss by Token Budget",
        "ylabel": "Mean auxiliary loss",
        "scale": 1.0,
        "models": MOE_MODELS,
        "required": True,
    },
    {
        "metric": "mean_expert_load_variance",
        "filename": "mean_expert_load_variance_by_budget.png",
        "title": "Mean Expert Load Variance by Token Budget",
        "ylabel": "Mean expert load variance",
        "scale": 1.0,
        "models": MOE_MODELS,
        "required": True,
    },
    {
        "metric": "mean_routing_entropy",
        "filename": "mean_routing_entropy_by_budget.png",
        "title": "Mean Routing Entropy by Token Budget",
        "ylabel": "Mean routing entropy",
        "scale": 1.0,
        "models": MOE_MODELS,
        "required": True,
    },
    {
        "metric": "mtp_loss",
        "filename": "mtp_loss_by_budget.png",
        "title": "MTP Loss by Token Budget",
        "ylabel": "MTP loss",
        "scale": 1.0,
        "models": MTP_MODELS,
        "required": True,
    },
]

FIGURE_DPI = 300
FIGURE_SIZE = (6.4, 4.2)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
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


def validate_descriptives_scope(rows: list[dict[str, str]]) -> dict[str, Any]:
    models = sorted({row["model"] for row in rows})
    budgets = sorted({row["budget"] for row in rows})
    metrics = sorted({row["metric"] for row in rows})

    expected_rows_from_shape = len(EXPECTED_MODELS) * len(EXPECTED_BUDGETS) * len(metrics)

    scope_passed = (
        len(rows) == EXPECTED_DESCRIPTIVE_ROWS
        and len(rows) == expected_rows_from_shape
        and models == sorted(EXPECTED_MODELS)
        and budgets == sorted(EXPECTED_BUDGETS)
    )

    return {
        "row_count": len(rows),
        "expected_row_count": EXPECTED_DESCRIPTIVE_ROWS,
        "models": models,
        "expected_models": sorted(EXPECTED_MODELS),
        "budgets": budgets,
        "expected_budgets": sorted(EXPECTED_BUDGETS),
        "metric_count": len(metrics),
        "scope_passed": scope_passed,
    }


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def build_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}

    for row in rows:
        key = (row["metric"], row["budget"], row["model"])
        if key in lookup:
            raise ValueError(f"Duplicate descriptive key: {key}")
        lookup[key] = row

    return lookup


def scaled_value(row: dict[str, str], field: str, scale: float) -> float | None:
    value = parse_float(row.get(field))
    if value is None:
        return None
    return value * scale


def collect_model_series(
    lookup: dict[tuple[str, str, str], dict[str, str]],
    *,
    metric: str,
    model: str,
    scale: float,
) -> tuple[list[float], list[float], list[float], list[float], list[str]]:
    x_values: list[float] = []
    y_values: list[float] = []
    yerr_low: list[float] = []
    yerr_high: list[float] = []
    point_failures: list[str] = []

    for budget in EXPECTED_BUDGETS:
        row = lookup.get((metric, budget, model))

        if row is None:
            point_failures.append(f"{metric}|{budget}|{model}|missing_row")
            continue

        seed_count = parse_float(row.get("non_missing_seed_count"))
        mean = scaled_value(row, "mean", scale)
        ci_low = scaled_value(row, "ci95_low", scale)
        ci_high = scaled_value(row, "ci95_high", scale)

        if seed_count is None or int(seed_count) != EXPECTED_SEED_COUNT:
            point_failures.append(f"{metric}|{budget}|{model}|bad_seed_count")
            continue

        if mean is None or ci_low is None or ci_high is None:
            point_failures.append(f"{metric}|{budget}|{model}|missing_mean_or_ci")
            continue

        x_values.append(float(EXPECTED_BUDGET_TOKENS[budget]))
        y_values.append(mean)
        yerr_low.append(max(0.0, mean - ci_low))
        yerr_high.append(max(0.0, ci_high - mean))

    return x_values, y_values, yerr_low, yerr_high, point_failures


def make_metric_plot(
    lookup: dict[tuple[str, str, str], dict[str, str]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    metric = str(spec["metric"])
    output_path = OUT_DIR / str(spec["filename"])
    scale = float(spec["scale"])
    models_to_plot = [str(model) for model in spec["models"]]

    plotted_models: list[str] = []
    coverage_failures: list[str] = []

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for model in models_to_plot:
        x_values, y_values, yerr_low, yerr_high, point_failures = collect_model_series(
            lookup,
            metric=metric,
            model=model,
            scale=scale,
        )

        if point_failures:
            coverage_failures.extend(point_failures)
            continue

        if len(y_values) != len(EXPECTED_BUDGETS):
            coverage_failures.append(f"{metric}|{model}|incomplete_budget_series")
            continue

        plotted_models.append(model)

        ax.errorbar(
            x_values,
            y_values,
            yerr=[yerr_low, yerr_high],
            marker=MODEL_MARKERS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=1.6,
            markersize=4.5,
            capsize=2.5,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )

    ax.set_xscale("log")
    ax.set_xticks([EXPECTED_BUDGET_TOKENS[budget] for budget in EXPECTED_BUDGETS])
    ax.set_xticklabels(EXPECTED_BUDGETS)
    ax.set_xlabel("Training token budget")
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_title(str(spec["title"]))
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)

    expected_model_count = len(models_to_plot)
    coverage_passed = len(plotted_models) == expected_model_count and not coverage_failures

    return {
        "metric": metric,
        "filename": str(output_path),
        "required": bool(spec["required"]),
        "expected_model_count": expected_model_count,
        "plotted_model_count": len(plotted_models),
        "plotted_models": plotted_models,
        "coverage_failures": coverage_failures,
        "coverage_passed": coverage_passed,
        "file_exists": output_path.exists(),
        "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }


def main() -> None:
    rows = read_rows(INPUT_DESCRIPTIVES_CSV)
    scope_audit = validate_descriptives_scope(rows)

    if not scope_audit["scope_passed"]:
        write_json(OUT_AUDIT_JSON, {"scope_audit": scope_audit})
        raise ValueError("Descriptives artifact failed plot scope validation.")

    setup_matplotlib()
    lookup = build_lookup(rows)

    manifest_rows = [make_metric_plot(lookup, spec) for spec in PLOT_SPECS]

    plot_failures = [
        row
        for row in manifest_rows
        if row["required"]
        and (not row["file_exists"] or row["file_size_bytes"] <= 0 or not row["coverage_passed"])
    ]

    plot_generation_passed = not plot_failures

    OUT_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "metric",
        "filename",
        "required",
        "expected_model_count",
        "plotted_model_count",
        "plotted_models",
        "coverage_failures",
        "coverage_passed",
        "file_exists",
        "file_size_bytes",
    ]

    csv_rows = []
    for row in manifest_rows:
        csv_rows.append(
            {
                **row,
                "plotted_models": ";".join(row["plotted_models"]),
                "coverage_failures": ";".join(row["coverage_failures"]),
            }
        )

    with OUT_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    write_json(OUT_MANIFEST_JSON, manifest_rows)

    audit = {
        "artifact_type": "balanced_10seed_matrix_png_plot_audit",
        "input_descriptives_csv": str(INPUT_DESCRIPTIVES_CSV),
        "scope_audit": scope_audit,
        "plot_count": len(manifest_rows),
        "required_plot_count": sum(1 for row in manifest_rows if row["required"]),
        "plot_failure_count": len(plot_failures),
        "plot_failures": plot_failures,
        "color_palette": "Okabe-Ito",
        "uses_distinct_markers": True,
        "uses_distinct_line_styles": True,
        "figure_dpi": FIGURE_DPI,
        "plot_manifest_csv": str(OUT_MANIFEST_CSV),
        "plot_manifest_json": str(OUT_MANIFEST_JSON),
        "audit_json": str(OUT_AUDIT_JSON),
        "plot_generation_passed": plot_generation_passed,
    }
    write_json(OUT_AUDIT_JSON, audit)

    print(json.dumps(audit, indent=2, sort_keys=True))

    if not plot_generation_passed:
        raise ValueError("PNG plot generation validation failed. See audit JSON.")


if __name__ == "__main__":
    main()
