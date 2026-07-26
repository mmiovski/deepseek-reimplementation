"""Plot MTP loss by token budget for the balanced 10-seed matrix."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIVES_CSV = (
    ROOT / "results" / "analysis" / "balanced_10seed_matrix_descriptives_full_precision.csv"
)
OUT_DIR = ROOT / "results" / "figures" / "balanced_10seed_matrix_redesign"
OUT_PNG = OUT_DIR / "mtp_loss_by_budget.png"

METRIC_CANDIDATES = [
    "mtp_loss",
    "mean_mtp_loss",
    "train_mtp_loss",
    "mean_train_mtp_loss",
]

MODEL_ORDER = [
    "dense_121m",
    "mla_121m",
    "mtp_121m",
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
]

MODEL_LABELS = {
    "dense_121m": "Dense",
    "mla_121m": "MLA",
    "mtp_121m": "MTP",
    "moe_220m": "MoE",
    "mla_moe_220m": "MLA+MoE",
    "v3_routing_220m": "V3 Routing",
}

MODEL_COLORS = {
    "dense_121m": "#332288",
    "mla_121m": "#44AA99",
    "mtp_121m": "#117733",
    "moe_220m": "#E69F00",
    "mla_moe_220m": "#CC6677",
    "v3_routing_220m": "#882255",
}

MODEL_MARKERS = {
    "dense_121m": "o",
    "mla_121m": "^",
    "mtp_121m": "s",
    "moe_220m": "D",
    "mla_moe_220m": "X",
    "v3_routing_220m": "P",
}

MODEL_X_OFFSETS = {
    "dense_121m": -0.105,
    "mla_121m": -0.063,
    "mtp_121m": 0.0,
    "moe_220m": 0.021,
    "mla_moe_220m": 0.063,
    "v3_routing_220m": 0.105,
}

BUDGET_ORDER = ["10M", "25M", "50M"]
BUDGET_TO_X = {"10M": 0.0, "25M": 1.0, "50M": 2.0}

FIGURE_SIZE = (7.3, 5.9)
Y_PAD_FRACTION = 0.035
DPI = 300


@dataclass(frozen=True)
class PlotPoint:
    model: str
    budget: str
    mean: float
    ci_half_width: float


def normalize_budget(value: object) -> str:
    text = str(value).strip().lower().replace("_", "").replace(" ", "")
    mapping = {
        "10m": "10M",
        "10000000": "10M",
        "25m": "25M",
        "25000000": "25M",
        "50m": "50M",
        "50000000": "50M",
    }
    if text not in mapping:
        raise ValueError(f"Unrecognized budget value: {value!r}")
    return mapping[text]


def find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def required_column(fieldnames: list[str], candidates: list[str]) -> str:
    column = find_column(fieldnames, candidates)
    if column is None:
        raise ValueError("Could not find any expected column among: " + ", ".join(candidates))
    return column


def parse_optional_float(value: str) -> float | None:
    text = value.strip()
    if text == "":
        return None

    parsed = float(text)
    if not math.isfinite(parsed):
        return None

    return parsed


def load_points() -> tuple[str, list[PlotPoint]]:
    if not DESCRIPTIVES_CSV.exists():
        raise FileNotFoundError(f"Missing descriptives artifact: {DESCRIPTIVES_CSV}")

    points: list[PlotPoint] = []
    matched_metric_names: set[str] = set()
    skipped_blank_rows = 0

    with DESCRIPTIVES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {DESCRIPTIVES_CSV}")

        fieldnames = list(reader.fieldnames)
        model_col = required_column(
            fieldnames,
            ["model", "model_name", "model_id", "architecture", "config_name"],
        )
        budget_col = required_column(
            fieldnames,
            ["token_budget", "token_budget_label", "budget", "budget_label"],
        )
        metric_col = required_column(fieldnames, ["metric", "metric_name"])
        mean_col = required_column(fieldnames, ["mean", "mean_value"])

        half_width_col = find_column(
            fieldnames,
            ["ci95_half_width", "ci_half_width", "half_width"],
        )
        lower_col = find_column(
            fieldnames,
            ["ci95_low", "ci_lower", "lower_ci", "bootstrap_ci_low", "ci_low"],
        )
        upper_col = find_column(
            fieldnames,
            ["ci95_high", "ci_upper", "upper_ci", "bootstrap_ci_high", "ci_high"],
        )

        if half_width_col is None and (lower_col is None or upper_col is None):
            raise ValueError(
                "Could not find CI columns. Expected either a half-width column "
                "or lower/upper CI columns."
            )

        for row in reader:
            metric_name = row[metric_col].strip()
            if metric_name not in METRIC_CANDIDATES:
                continue

            model = row[model_col].strip()
            if model not in MODEL_ORDER:
                continue

            mean_value = parse_optional_float(row[mean_col])
            if mean_value is None:
                skipped_blank_rows += 1
                continue

            budget = normalize_budget(row[budget_col])

            if half_width_col is not None:
                ci_half_width = parse_optional_float(row[half_width_col])
                if ci_half_width is None:
                    skipped_blank_rows += 1
                    continue
            else:
                assert lower_col is not None
                assert upper_col is not None
                lower = parse_optional_float(row[lower_col])
                upper = parse_optional_float(row[upper_col])
                if lower is None or upper is None:
                    skipped_blank_rows += 1
                    continue
                ci_half_width = max(abs(mean_value - lower), abs(upper - mean_value))

            matched_metric_names.add(metric_name)
            points.append(
                PlotPoint(
                    model=model,
                    budget=budget,
                    mean=mean_value,
                    ci_half_width=ci_half_width,
                )
            )

    if not points:
        raise ValueError(
            "No nonblank MTP-loss rows found. Checked metric candidates: "
            + ", ".join(METRIC_CANDIDATES)
        )

    models_with_all_budgets: list[str] = []
    for model in MODEL_ORDER:
        observed_budgets = {point.budget for point in points if point.model == model}
        if observed_budgets == set(BUDGET_ORDER):
            models_with_all_budgets.append(model)

    if not models_with_all_budgets:
        raise ValueError("No model has complete nonblank MTP-loss values for all budgets.")

    filtered_points = [point for point in points if point.model in models_with_all_budgets]

    if len(matched_metric_names) != 1:
        raise ValueError(
            "Expected one MTP-loss metric name, found: " + ", ".join(sorted(matched_metric_names))
        )

    metric_name = next(iter(matched_metric_names))
    print("Using MTP-loss metric:", metric_name)
    print("Plotting models:", ", ".join(models_with_all_budgets))
    print("Skipped blank/non-finite rows:", skipped_blank_rows)

    return metric_name, filtered_points


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def plot(points: list[PlotPoint]) -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    y_values_for_limits: list[float] = []
    models_to_plot = [
        model
        for model in MODEL_ORDER
        if {point.budget for point in points if point.model == model} == set(BUDGET_ORDER)
    ]

    for model in models_to_plot:
        model_points = sorted(
            [point for point in points if point.model == model],
            key=lambda point: BUDGET_TO_X[point.budget],
        )
        xs = [BUDGET_TO_X[point.budget] + MODEL_X_OFFSETS[model] for point in model_points]
        ys = [point.mean for point in model_points]
        yerrs = [point.ci_half_width for point in model_points]

        for y_value, yerr in zip(ys, yerrs, strict=True):
            y_values_for_limits.append(y_value - yerr)
            y_values_for_limits.append(y_value + yerr)

        ax.errorbar(
            xs,
            ys,
            yerr=yerrs,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=1.95,
            markersize=6.4,
            elinewidth=0.95,
            capsize=3.6,
        )

    y_min = min(y_values_for_limits)
    y_max = max(y_values_for_limits)
    y_pad = max((y_max - y_min) * Y_PAD_FRACTION, 0.02)

    ax.set_xlim(-0.28, 2.28)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks([BUDGET_TO_X[budget] for budget in BUDGET_ORDER])
    ax.set_xticklabels(BUDGET_ORDER)
    ax.set_xlabel("Training Token Budget")
    ax.set_ylabel("MTP Loss")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_PNG}")


def main() -> None:
    _, points = load_points()
    plot(points)


if __name__ == "__main__":
    main()
