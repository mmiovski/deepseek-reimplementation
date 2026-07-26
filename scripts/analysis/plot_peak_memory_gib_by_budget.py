"""Plot peak GPU memory by token budget for the balanced 10-seed matrix."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIVES_CSV = (
    ROOT / "results" / "analysis" / "balanced_10seed_matrix_descriptives_full_precision.csv"
)
OUT_DIR = ROOT / "results" / "figures" / "balanced_10seed_matrix_redesign"
OUT_PNG = OUT_DIR / "peak_memory_gib_by_budget.png"

METRIC_CANDIDATES = [
    "peak_memory_bytes",
    "peak_memory_allocated_bytes",
    "max_memory_allocated_bytes",
    "peak_memory_gib",
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
    "mtp_121m": -0.021,
    "moe_220m": 0.021,
    "mla_moe_220m": 0.063,
    "v3_routing_220m": 0.105,
}

BUDGET_ORDER = ["10M", "25M", "50M"]
BUDGET_TO_X = {"10M": 0.0, "25M": 1.0, "50M": 2.0}

FIGURE_SIZE = (7.3, 5.9)
Y_PAD_FRACTION = 0.035
DPI = 300
BYTES_PER_GIB = 1024.0**3


@dataclass(frozen=True)
class PlotPoint:
    model: str
    budget: str
    mean_gib: float
    ci_half_width_gib: float


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


def get_float(row: dict[str, str], column: str) -> float:
    return float(row[column])


def convert_to_gib(value: float, metric_name: str) -> float:
    if metric_name.endswith("_gib"):
        return value
    return value / BYTES_PER_GIB


def load_points() -> list[PlotPoint]:
    if not DESCRIPTIVES_CSV.exists():
        raise FileNotFoundError(f"Missing descriptives artifact: {DESCRIPTIVES_CSV}")

    points: list[PlotPoint] = []

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

            budget = normalize_budget(row[budget_col])
            mean_gib = convert_to_gib(get_float(row, mean_col), metric_name)

            if half_width_col is not None:
                ci_half_width_gib = convert_to_gib(
                    get_float(row, half_width_col),
                    metric_name,
                )
            else:
                assert lower_col is not None
                assert upper_col is not None
                lower_gib = convert_to_gib(get_float(row, lower_col), metric_name)
                upper_gib = convert_to_gib(get_float(row, upper_col), metric_name)
                ci_half_width_gib = max(
                    abs(mean_gib - lower_gib),
                    abs(upper_gib - mean_gib),
                )

            points.append(
                PlotPoint(
                    model=model,
                    budget=budget,
                    mean_gib=mean_gib,
                    ci_half_width_gib=ci_half_width_gib,
                )
            )

    expected = {(model, budget) for model in MODEL_ORDER for budget in BUDGET_ORDER}
    observed = {(point.model, point.budget) for point in points}
    missing = sorted(expected - observed)

    if missing:
        missing_text = ", ".join(f"{model}/{budget}" for model, budget in missing)
        raise ValueError(f"Missing peak-memory descriptives for: {missing_text}")

    return points


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

    for model in MODEL_ORDER:
        model_points = [point for point in points if point.model == model]
        xs = [BUDGET_TO_X[point.budget] + MODEL_X_OFFSETS[model] for point in model_points]
        ys = [point.mean_gib for point in model_points]
        yerrs = [point.ci_half_width_gib for point in model_points]

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
    ax.set_ylabel("Peak Memory (GiB)")
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
    plot(load_points())


if __name__ == "__main__":
    main()
