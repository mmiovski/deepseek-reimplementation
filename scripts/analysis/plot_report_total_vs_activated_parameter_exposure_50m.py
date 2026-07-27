"""Plot report-ready total versus activated parameter exposure at 50M tokens."""

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
OUT_PNG = OUT_DIR / "report_total_vs_activated_parameter_exposure_50m.png"

BUDGET_LABEL = "50M"
TOTAL_METRIC = "tokens_per_total_parameter"
ACTIVATED_METRIC = "tokens_per_activated_parameter"

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

FIGURE_SIZE = (7.3, 5.9)
DPI = 300
TEXT_COLOR = "#000000"
TOTAL_Y_OFFSET = -0.095
ACTIVATED_Y_OFFSET = 0.095


@dataclass(frozen=True)
class ExposurePoint:
    model: str
    tokens_per_total_parameter: float
    tokens_per_activated_parameter: float


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


def load_points() -> list[ExposurePoint]:
    if not DESCRIPTIVES_CSV.exists():
        raise FileNotFoundError(f"Missing descriptives artifact: {DESCRIPTIVES_CSV}")

    values: dict[tuple[str, str], float] = {}

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

        for row in reader:
            model = row[model_col].strip()
            metric = row[metric_col].strip()

            if model not in MODEL_ORDER:
                continue
            if normalize_budget(row[budget_col]) != BUDGET_LABEL:
                continue
            if metric not in {TOTAL_METRIC, ACTIVATED_METRIC}:
                continue
            if row[mean_col].strip() == "":
                continue

            values[(model, metric)] = float(row[mean_col])

    points: list[ExposurePoint] = []
    for model in MODEL_ORDER:
        total_value = values.get((model, TOTAL_METRIC))
        activated_value = values.get((model, ACTIVATED_METRIC))

        if total_value is None or activated_value is None:
            raise ValueError(
                f"Missing exposure values for {model}: "
                f"{TOTAL_METRIC}={total_value}, {ACTIVATED_METRIC}={activated_value}"
            )

        points.append(
            ExposurePoint(
                model=model,
                tokens_per_total_parameter=total_value,
                tokens_per_activated_parameter=activated_value,
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


def plot(points: list[ExposurePoint]) -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    y_positions = list(range(len(points)))

    for y_position, point in zip(y_positions, points, strict=True):
        color = MODEL_COLORS[point.model]
        total_y = y_position + TOTAL_Y_OFFSET
        activated_y = y_position + ACTIVATED_Y_OFFSET

        ax.plot(
            [
                point.tokens_per_total_parameter,
                point.tokens_per_activated_parameter,
            ],
            [total_y, activated_y],
            color=color,
            linewidth=1.2,
            alpha=0.55,
            zorder=1,
        )

        ax.scatter(
            point.tokens_per_total_parameter,
            total_y,
            marker="o",
            s=54,
            facecolors="white",
            edgecolors=color,
            linewidths=1.6,
            zorder=3,
        )

        ax.scatter(
            point.tokens_per_activated_parameter,
            activated_y,
            marker="^",
            s=62,
            facecolors=color,
            edgecolors=TEXT_COLOR,
            linewidths=0.65,
            zorder=3,
        )

    all_x_values = [
        value
        for point in points
        for value in (
            point.tokens_per_total_parameter,
            point.tokens_per_activated_parameter,
        )
    ]
    x_span = max(all_x_values) - min(all_x_values)

    ax.set_xlim(
        min(all_x_values) - max(x_span * 0.10, 0.025),
        max(all_x_values) + max(x_span * 0.10, 0.025),
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([MODEL_LABELS[point.model] for point in points])
    ax.invert_yaxis()

    ax.set_xlabel("Tokens/Parameter")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=TEXT_COLOR,
            markerfacecolor="white",
            markeredgecolor=TEXT_COLOR,
            linestyle="None",
            markersize=6.8,
            label="Total Parameters",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=TEXT_COLOR,
            markerfacecolor=TEXT_COLOR,
            markeredgecolor=TEXT_COLOR,
            linestyle="None",
            markersize=7.2,
            label="Activated Parameters",
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        borderaxespad=0.5,
        handletextpad=0.8,
        labelspacing=0.5,
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
