"""Plot report-ready test loss versus tokens per activated parameter."""

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
OUT_PNG = OUT_DIR / "report_test_loss_vs_tokens_per_activated_parameter.png"

X_METRIC = "tokens_per_activated_parameter"
Y_METRIC = "test_loss"

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

BUDGET_ORDER = ["10M", "25M", "50M"]
BUDGET_RANK = {"10M": 0, "25M": 1, "50M": 2}

# Display-only horizontal offsets to reduce overplotting.
# These do not alter the underlying artifact values.
X_VISUAL_OFFSETS = {
    "dense_121m": -0.026,
    "mla_121m": -0.016,
    "mtp_121m": -0.006,
    "moe_220m": 0.006,
    "mla_moe_220m": 0.016,
    "v3_routing_220m": 0.026,
}

FIGURE_SIZE = (7.3, 5.9)
DPI = 300
TEXT_COLOR = "#000000"


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class RegimePoint:
    model: str
    budget: str
    tokens_per_activated_parameter: MetricSummary
    test_loss: MetricSummary


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


def load_points() -> list[RegimePoint]:
    if not DESCRIPTIVES_CSV.exists():
        raise FileNotFoundError(f"Missing descriptives artifact: {DESCRIPTIVES_CSV}")

    summaries: dict[tuple[str, str, str], MetricSummary] = {}

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

            if model not in MODEL_ORDER:
                continue
            if metric not in {X_METRIC, Y_METRIC}:
                continue
            if row[mean_col].strip() == "":
                continue

            budget = normalize_budget(row[budget_col])
            summaries[(model, budget, metric)] = extract_summary(
                row=row,
                mean_col=mean_col,
                half_width_col=half_width_col,
                lower_col=lower_col,
                upper_col=upper_col,
            )

    points: list[RegimePoint] = []
    for model in MODEL_ORDER:
        for budget in BUDGET_ORDER:
            x_summary = summaries.get((model, budget, X_METRIC))
            y_summary = summaries.get((model, budget, Y_METRIC))

            if x_summary is None or y_summary is None:
                raise ValueError(
                    f"Missing summaries for {model}, {budget}: "
                    f"{X_METRIC}={x_summary}, {Y_METRIC}={y_summary}"
                )

            points.append(
                RegimePoint(
                    model=model,
                    budget=budget,
                    tokens_per_activated_parameter=x_summary,
                    test_loss=y_summary,
                )
            )

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
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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


def plot(points: list[RegimePoint]) -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for model in MODEL_ORDER:
        model_points = sorted(
            [point for point in points if point.model == model],
            key=lambda point: BUDGET_RANK[point.budget],
        )

        xs_raw = [point.tokens_per_activated_parameter.mean for point in model_points]
        xs_display = [x_value + X_VISUAL_OFFSETS[model] for x_value in xs_raw]
        ys = [point.test_loss.mean for point in model_points]

        ax.plot(
            xs_display,
            ys,
            color=MODEL_COLORS[model],
            linewidth=1.55,
            alpha=0.82,
            zorder=2,
        )

        for x_display, point in zip(xs_display, model_points, strict=True):
            ax.errorbar(
                x_display,
                point.test_loss.mean,
                yerr=vertical_error(point.test_loss),
                fmt=MODEL_MARKERS[model],
                color=MODEL_COLORS[model],
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor=TEXT_COLOR,
                markeredgewidth=0.75,
                markersize=6.4,
                linewidth=0,
                elinewidth=0.7,
                capsize=2.1,
                capthick=0.7,
                alpha=0.88,
                zorder=3,
            )

    x_values = [
        point.tokens_per_activated_parameter.mean + X_VISUAL_OFFSETS[point.model]
        for point in points
    ]
    y_values = [
        value
        for point in points
        for value in (
            point.test_loss.lower if point.test_loss.lower is not None else point.test_loss.mean,
            point.test_loss.upper if point.test_loss.upper is not None else point.test_loss.mean,
        )
    ]

    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)

    ax.set_xlim(
        min(x_values) - max(x_span * 0.08, 0.02),
        max(x_values) + max(x_span * 0.08, 0.02),
    )
    ax.set_ylim(
        min(y_values) - max(y_span * 0.08, 0.02),
        max(y_values) + max(y_span * 0.08, 0.02),
    )

    ax.set_xlabel("Tokens/Activated Parameters")
    ax.set_ylabel("Test Loss")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(alpha=0.22, linewidth=0.8)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linewidth=1.55,
            markersize=6.4,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )

    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    enforce_black_text(ax)

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_PNG}")


def main() -> None:
    points = load_points()
    plot(points)


if __name__ == "__main__":
    main()
