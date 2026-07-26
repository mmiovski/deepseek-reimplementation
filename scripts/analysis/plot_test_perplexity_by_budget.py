from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[2]

INPUT_DESCRIPTIVES_CSV = (
    ROOT / "results" / "analysis" / "balanced_10seed_matrix_descriptives_full_precision.csv"
)
INPUT_SUMMARY_FLAT_CSV = ROOT / "results" / "analysis" / "balanced_10seed_matrix_summary_flat.csv"

OUT_DIR = ROOT / "results" / "figures" / "balanced_10seed_matrix_redesign"
OUT_PNG = OUT_DIR / "test_perplexity_by_budget.png"

METRIC_NAME = "test_perplexity"

BUDGET_ORDER = ["10M", "25M", "50M"]
BUDGET_POSITIONS = {"10M": 0.0, "25M": 1.0, "50M": 2.0}

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

# Same broad design logic as before:
# baseline/dense family first, sparse family second, routing/combined variants after.
MODEL_COLORS = {
    "dense_121m": "#332288",  # Dense baseline: dark indigo / primary anchor
    "mla_121m": "#44AA99",  # MLA: teal / dense-attention variant
    "mtp_121m": "#117733",  # MTP: green / auxiliary-objective variant
    "moe_220m": "#E69F00",  # MoE: orange / sparse baseline
    "mla_moe_220m": "#CC6677",  # MLA+MoE: rose / combined mechanism
    "v3_routing_220m": "#882255",  # V3 Routing: wine-purple / routing variant
}

MODEL_MARKERS = {
    "dense_121m": "o",
    "mla_121m": "^",
    "mtp_121m": "s",
    "moe_220m": "D",
    "mla_moe_220m": "X",
    "v3_routing_220m": "P",
}

# Small offsets prevent six error bars from stacking exactly on the same x-coordinate.
MODEL_X_OFFSETS = {
    "dense_121m": -0.105,
    "mla_121m": -0.063,
    "mtp_121m": -0.021,
    "moe_220m": 0.021,
    "mla_moe_220m": 0.063,
    "v3_routing_220m": 0.105,
}

FIGURE_SIZE = (7.3, 5.9)
FIGURE_DPI = 300

LINE_WIDTH = 1.95
MARKER_SIZE = 6.4
ERRORBAR_LINE_WIDTH = 0.95
ERRORBAR_CAP_SIZE = 3.6

Y_PAD_FRACTION = 0.035


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def find_column(columns: set[str], candidates: list[str]) -> str | None:
    normalized_lookup = {normalize_key(column): column for column in columns}
    for candidate in candidates:
        normalized = normalize_key(candidate)
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]
    return None


def normalize_budget(value: str) -> str | None:
    raw = value.strip()
    compact = raw.lower().replace("_", "").replace("-", "").replace(",", "").replace(" ", "")

    if compact in {"10m", "10000000", "10.0m"} or "10m" in compact:
        return "10M"
    if compact in {"25m", "25000000", "25.0m"} or "25m" in compact:
        return "25M"
    if compact in {"50m", "50000000", "50.0m"} or "50m" in compact:
        return "50M"

    try:
        numeric = float(compact)
    except ValueError:
        return None

    if math.isclose(numeric, 10_000_000):
        return "10M"
    if math.isclose(numeric, 25_000_000):
        return "25M"
    if math.isclose(numeric, 50_000_000):
        return "50M"

    return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped or stripped.lower() in {"nan", "none", "null"}:
        return None

    return float(stripped)


def load_from_descriptives() -> dict[tuple[str, str], tuple[float, float]]:
    rows = read_csv(INPUT_DESCRIPTIVES_CSV)
    if not rows:
        raise ValueError(f"No rows found in {INPUT_DESCRIPTIVES_CSV}")

    columns = set(rows[0].keys())

    model_col = find_column(columns, ["model_id", "model", "model_name"])
    budget_col = find_column(
        columns,
        [
            "token_budget_label",
            "training_token_budget_label",
            "budget_label",
            "budget",
            "token_budget",
            "training_tokens",
        ],
    )
    metric_col = find_column(columns, ["metric_name", "metric"])
    mean_col = find_column(columns, ["mean", "value_mean", "mean_value"])
    sem_col = find_column(columns, ["sem", "se", "standard_error"])
    std_col = find_column(columns, ["std", "standard_deviation"])
    n_col = find_column(columns, ["n", "count", "seed_count"])

    ci_low_col = find_column(
        columns,
        ["ci95_low", "mean_ci95_low", "bootstrap_ci_low", "ci_low"],
    )
    ci_high_col = find_column(
        columns,
        ["ci95_high", "mean_ci95_high", "bootstrap_ci_high", "ci_high"],
    )

    required = {
        "model_col": model_col,
        "budget_col": budget_col,
        "metric_col": metric_col,
        "mean_col": mean_col,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Could not read descriptives artifact with expected columns. "
            f"Missing: {missing}. Columns found: {sorted(columns)}"
        )

    data: dict[tuple[str, str], tuple[float, float]] = {}

    for row in rows:
        metric = normalize_key(row[metric_col])  # type: ignore[index]
        if metric != METRIC_NAME:
            continue

        model = row[model_col].strip()  # type: ignore[index]
        budget = normalize_budget(row[budget_col])  # type: ignore[index]
        if model not in MODEL_ORDER or budget not in BUDGET_ORDER:
            continue

        value_mean = parse_float(row[mean_col])  # type: ignore[index]
        if value_mean is None:
            continue

        yerr = 0.0
        ci_low = parse_float(row[ci_low_col]) if ci_low_col else None
        ci_high = parse_float(row[ci_high_col]) if ci_high_col else None
        sem = parse_float(row[sem_col]) if sem_col else None
        std = parse_float(row[std_col]) if std_col else None
        n = parse_float(row[n_col]) if n_col else None

        if ci_low is not None and ci_high is not None:
            yerr = max(abs(value_mean - ci_low), abs(ci_high - value_mean))
        elif sem is not None:
            yerr = 1.96 * sem
        elif std is not None and n is not None and n > 1:
            yerr = 1.96 * std / math.sqrt(n)

        data[(model, budget)] = (value_mean, yerr)

    if data:
        return data

    raise ValueError("No usable test_perplexity rows found in descriptives artifact.")


def load_from_summary_flat() -> dict[tuple[str, str], tuple[float, float]]:
    rows = read_csv(INPUT_SUMMARY_FLAT_CSV)
    if not rows:
        raise ValueError(f"No rows found in {INPUT_SUMMARY_FLAT_CSV}")

    columns = set(rows[0].keys())

    model_col = find_column(columns, ["model_id", "model", "model_name"])
    budget_col = find_column(
        columns,
        [
            "token_budget_label",
            "training_token_budget_label",
            "budget_label",
            "budget",
            "token_budget",
            "training_tokens",
        ],
    )
    metric_col = find_column(columns, [METRIC_NAME])

    required = {
        "model_col": model_col,
        "budget_col": budget_col,
        "metric_col": metric_col,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Could not read summary-flat artifact with expected columns. "
            f"Missing: {missing}. Columns found: {sorted(columns)}"
        )

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in rows:
        model = row[model_col].strip()  # type: ignore[index]
        budget = normalize_budget(row[budget_col])  # type: ignore[index]
        if model not in MODEL_ORDER or budget not in BUDGET_ORDER:
            continue

        value = parse_float(row[metric_col])  # type: ignore[index]
        if value is not None:
            grouped[(model, budget)].append(value)

    data: dict[tuple[str, str], tuple[float, float]] = {}

    for key, values in grouped.items():
        value_mean = mean(values)
        if len(values) > 1:
            yerr = 1.96 * stdev(values) / math.sqrt(len(values))
        else:
            yerr = 0.0
        data[key] = (value_mean, yerr)

    if data:
        return data

    raise ValueError("No usable test_perplexity values found in summary-flat artifact.")


def load_plot_data() -> dict[tuple[str, str], tuple[float, float]]:
    try:
        return load_from_descriptives()
    except Exception as descriptives_error:
        print(f"Descriptives read failed; falling back to summary flat: {descriptives_error}")
        return load_from_summary_flat()


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "CMU Serif",
                "Computer Modern Roman",
                "Latin Modern Roman",
                "DejaVu Serif",
            ],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
        }
    )


def configure_axes(ax: Axes, plotted_values: list[tuple[float, float]]) -> None:
    ax.set_xlabel("Training Token Budget")
    ax.set_ylabel("Test Perplexity")

    ax.set_xticks([BUDGET_POSITIONS[budget] for budget in BUDGET_ORDER])
    ax.set_xticklabels(BUDGET_ORDER)
    ax.set_xlim(-0.28, 2.28)

    lows = [value - yerr for value, yerr in plotted_values]
    highs = [value + yerr for value, yerr in plotted_values]

    y_low = min(lows)
    y_high = max(highs)
    spread = y_high - y_low
    pad = max(spread * Y_PAD_FRACTION, 0.0025)

    ax.set_ylim(y_low - pad, y_high + pad)

    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)


def plot() -> None:
    setup_matplotlib()
    data = load_plot_data()

    missing = [
        (model, budget)
        for model in MODEL_ORDER
        for budget in BUDGET_ORDER
        if (model, budget) not in data
    ]
    if missing:
        raise ValueError(f"Missing model-budget cells for plot: {missing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    plotted_values: list[tuple[float, float]] = []

    for model in MODEL_ORDER:
        xs = [BUDGET_POSITIONS[budget] + MODEL_X_OFFSETS[model] for budget in BUDGET_ORDER]
        ys = [data[(model, budget)][0] for budget in BUDGET_ORDER]
        yerrs = [data[(model, budget)][1] for budget in BUDGET_ORDER]

        plotted_values.extend(zip(ys, yerrs, strict=True))

        ax.errorbar(
            xs,
            ys,
            yerr=yerrs,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgecolor="white",
            markeredgewidth=0.55,
            elinewidth=ERRORBAR_LINE_WIDTH,
            capsize=ERRORBAR_CAP_SIZE,
            capthick=ERRORBAR_LINE_WIDTH,
            alpha=0.96,
        )

    configure_axes(ax, plotted_values)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=2.0,
        borderaxespad=0.0,
    )

    fig.savefig(OUT_PNG)
    plt.close(fig)

    print(f"Wrote: {OUT_PNG}")


if __name__ == "__main__":
    plot()
