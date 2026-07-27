"""Plot report-ready routing behavior comparison."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIVES_CSV = (
    ROOT / "results" / "analysis" / "balanced_10seed_matrix_descriptives_full_precision.csv"
)
OUT_DIR = ROOT / "results" / "figures" / "balanced_10seed_matrix_report"
OUT_PNG = OUT_DIR / "report_routing_behavior_comparison.png"

ROUTING_MODELS = [
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
]

AUX_ZOOM_MODELS = [
    "moe_220m",
    "mla_moe_220m",
]

MODEL_LABELS = {
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
    "moe_220m": "D",
    "mla_moe_220m": "X",
    "v3_routing_220m": "P",
}

MODEL_X_OFFSETS = {
    "moe_220m": -0.045,
    "mla_moe_220m": 0.000,
    "v3_routing_220m": 0.045,
}

BUDGET_ORDER = ["10M", "25M", "50M"]
BUDGET_TO_X = {"10M": 0.0, "25M": 1.0, "50M": 2.0}

METRIC_PANELS = [
    ("mean_routing_entropy", "Routing Entropy", ROUTING_MODELS),
    ("mean_expert_load_variance", "Expert Load Variance", ROUTING_MODELS),
    ("mean_aux_loss", "Auxiliary Loss", ROUTING_MODELS),
    ("mean_aux_loss", "Auxiliary Loss (Zoom)", AUX_ZOOM_MODELS),
]

FIGURE_SIZE = (7.4, 6.35)
DPI = 300
TEXT_COLOR = "#000000"


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class MetricPoint:
    metric: str
    model: str
    budget: str
    summary: MetricSummary


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
        raise ValueError(
            "Could not find any expected column among "
            + str(candidates)
            + ". Available columns: "
            + str(fieldnames)
        )
    return column


def read_float(row: dict[str, str], column: str | None) -> float | None:
    if column is None:
        return None

    text = row[column].strip()
    if text == "":
        return None

    return float(text)


def extract_summary(
    row: dict[str, str],
    mean_col: str,
    half_width_col: str | None,
    lower_col: str | None,
    upper_col: str | None,
) -> MetricSummary:
    mean = float(row[mean_col])

    if half_width_col is not None:
        half_width = read_float(row, half_width_col)
        if half_width is not None:
            return MetricSummary(
                mean=mean,
                lower=mean - half_width,
                upper=mean + half_width,
            )

    lower = read_float(row, lower_col)
    upper = read_float(row, upper_col)

    return MetricSummary(mean=mean, lower=lower, upper=upper)


def load_points() -> list[MetricPoint]:
    if not DESCRIPTIVES_CSV.exists():
        raise FileNotFoundError(f"Missing descriptives artifact: {DESCRIPTIVES_CSV}")

    needed_metrics = {metric for metric, _, _ in METRIC_PANELS}
    points: list[MetricPoint] = []

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
            [
                "ci95_low",
                "ci95_lower",
                "bootstrap_ci95_low",
                "bootstrap_ci_low",
                "lower_ci",
                "ci_lower",
                "ci_low",
            ],
        )
        upper_col = find_column(
            fieldnames,
            [
                "ci95_high",
                "ci95_upper",
                "bootstrap_ci95_high",
                "bootstrap_ci_high",
                "upper_ci",
                "ci_upper",
                "ci_high",
            ],
        )

        for row in reader:
            model = row[model_col].strip()
            metric = row[metric_col].strip()

            if model not in ROUTING_MODELS:
                continue
            if metric not in needed_metrics:
                continue
            if row[mean_col].strip() == "":
                continue

            points.append(
                MetricPoint(
                    metric=metric,
                    model=model,
                    budget=normalize_budget(row[budget_col]),
                    summary=extract_summary(
                        row=row,
                        mean_col=mean_col,
                        half_width_col=half_width_col,
                        lower_col=lower_col,
                        upper_col=upper_col,
                    ),
                )
            )

    required_cells = {
        (metric, model, budget)
        for metric, _, models in METRIC_PANELS
        for model in models
        for budget in BUDGET_ORDER
    }
    observed_cells = {
        (point.metric, point.model, point.budget)
        for point in points
        if point.budget in BUDGET_ORDER
    }
    missing_cells = sorted(required_cells - observed_cells)

    if missing_cells:
        raise ValueError(f"Missing routing metric cells: {missing_cells}")

    return points


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def enforce_black_text(ax: plt.Axes) -> None:
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.tick_params(axis="both", colors=TEXT_COLOR)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TEXT_COLOR)

    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)


def vertical_error(summary: MetricSummary) -> tuple[list[float], list[float]] | None:
    if summary.lower is None or summary.upper is None:
        return None

    return ([summary.mean - summary.lower], [summary.upper - summary.mean])


def plot_panel(
    ax: plt.Axes,
    points: list[MetricPoint],
    metric: str,
    y_label: str,
    models: list[str],
) -> None:
    for model in models:
        model_points = sorted(
            [point for point in points if point.metric == metric and point.model == model],
            key=lambda point: BUDGET_TO_X[point.budget],
        )

        xs = [BUDGET_TO_X[point.budget] + MODEL_X_OFFSETS[model] for point in model_points]
        ys = [point.summary.mean for point in model_points]

        ax.plot(
            xs,
            ys,
            color=MODEL_COLORS[model],
            linewidth=1.45,
            alpha=0.82,
            zorder=2,
        )

        for x_value, point in zip(xs, model_points, strict=True):
            ax.errorbar(
                x_value,
                point.summary.mean,
                yerr=vertical_error(point.summary),
                fmt=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor=TEXT_COLOR,
                markeredgewidth=0.65,
                markersize=5.6,
                linewidth=0,
                elinewidth=0.7,
                capsize=2.0,
                capthick=0.7,
                alpha=0.88,
                zorder=3,
            )

    y_values = [
        value
        for point in points
        if point.metric == metric and point.model in models
        for value in (
            point.summary.lower if point.summary.lower is not None else point.summary.mean,
            point.summary.upper if point.summary.upper is not None else point.summary.mean,
        )
    ]

    y_span = max(y_values) - min(y_values)
    ax.set_ylim(
        min(y_values) - max(y_span * 0.12, 0.00001),
        max(y_values) + max(y_span * 0.12, 0.00001),
    )

    if y_label == "Expert Load Variance":
        ax.set_ylim(0.00025, 0.00085)

    ax.set_xlim(-0.20, 2.20)
    ax.set_xticks([BUDGET_TO_X[budget] for budget in BUDGET_ORDER])
    ax.set_xticklabels(BUDGET_ORDER)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    enforce_black_text(ax)


def plot(points: list[MetricPoint]) -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=FIGURE_SIZE,
        sharex=True,
    )
    flat_axes = [axis for row in axes for axis in row]

    for ax, (metric, y_label, models) in zip(
        flat_axes,
        METRIC_PANELS,
        strict=True,
    ):
        plot_panel(
            ax=ax,
            points=points,
            metric=metric,
            y_label=y_label,
            models=models,
        )

    for ax in flat_axes:
        ax.tick_params(axis="x", labelbottom=True)

    for ax in flat_axes[2:]:
        ax.set_xlabel("Training Token Budget")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=1.45,
            markersize=5.8,
            label=MODEL_LABELS[model],
        )
        for model in ROUTING_MODELS
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        handletextpad=0.7,
        columnspacing=1.8,
    )

    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        bottom=0.095,
        top=0.895,
        wspace=0.30,
        hspace=0.34,
    )

    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_PNG}")


def main() -> None:
    points = load_points()
    plot(points)


if __name__ == "__main__":
    main()
