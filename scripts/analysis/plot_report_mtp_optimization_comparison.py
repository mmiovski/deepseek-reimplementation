"""Plot report-ready MTP optimization comparison."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
PAIRED_CONTRAST_CANDIDATES = [
    ROOT / "results" / "analysis" / "balanced_10seed_matrix_paired_contrasts_full_precision.csv",
    ROOT / "results" / "analysis" / "balanced_10seed_matrix_paired_contrasts.csv",
]
DESCRIPTIVES_CSV = (
    ROOT / "results" / "analysis" / "balanced_10seed_matrix_descriptives_full_precision.csv"
)
OUT_DIR = ROOT / "results" / "figures" / "balanced_10seed_matrix_report"
OUT_PNG = OUT_DIR / "report_mtp_optimization_comparison.png"

BUDGET_ORDER = ["10M", "25M", "50M"]
BUDGET_TO_X = {"10M": 0.0, "25M": 1.0, "50M": 2.0}

DENSE_MODEL = "dense_121m"
MTP_MODEL = "mtp_121m"
TEST_LOSS_METRIC = "test_loss"
MTP_LOSS_CANDIDATES = [
    "mtp_loss",
    "mean_mtp_loss",
    "train_mtp_loss",
    "mean_train_mtp_loss",
]

MTP_COLOR = "#117733"
TEXT_COLOR = "#000000"
FIGURE_SIZE = (7.4, 3.6)
DPI = 300


@dataclass(frozen=True)
class IntervalPoint:
    budget: str
    mean: float
    lower: float
    upper: float


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


def find_first_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path

    checked = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Missing paired contrast artifact. Checked: {checked}")


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


def choose_metric(available_metrics: set[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in available_metrics:
            return candidate

    raise ValueError("Could not find MTP loss metric among candidates: " + str(candidates))


def load_mtp_test_loss_differences() -> list[IntervalPoint]:
    source_csv = find_first_existing_path(PAIRED_CONTRAST_CANDIDATES)

    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {source_csv}")

        fieldnames = list(reader.fieldnames)
        metric_col = required_column(fieldnames, ["metric", "metric_name"])
        budget_col = required_column(
            fieldnames,
            ["token_budget", "token_budget_label", "budget", "budget_label"],
        )
        model_a_col = required_column(fieldnames, ["model_a", "baseline_model"])
        model_b_col = required_column(fieldnames, ["model_b", "comparison_model"])
        estimate_col = required_column(
            fieldnames,
            [
                "mean_difference_model_b_minus_model_a",
                "mean_difference",
                "mean_diff",
                "delta_mean",
                "estimate",
            ],
        )
        lower_col = required_column(
            fieldnames,
            [
                "bootstrap_ci95_low",
                "bootstrap_ci_lower",
                "bootstrap_ci_low",
                "paired_t_ci95_low",
                "ci95_lower",
                "ci95_low",
                "lower_ci",
                "ci_lower",
                "ci_low",
            ],
        )
        upper_col = required_column(
            fieldnames,
            [
                "bootstrap_ci95_high",
                "bootstrap_ci_upper",
                "bootstrap_ci_high",
                "paired_t_ci95_high",
                "ci95_upper",
                "ci95_high",
                "upper_ci",
                "ci_upper",
                "ci_high",
            ],
        )

        matched: dict[str, IntervalPoint] = {}

        for row in reader:
            if row[metric_col].strip() != TEST_LOSS_METRIC:
                continue

            budget = normalize_budget(row[budget_col])
            if budget not in BUDGET_ORDER:
                continue

            row_a = row[model_a_col].strip()
            row_b = row[model_b_col].strip()

            if row_a == DENSE_MODEL and row_b == MTP_MODEL:
                mean = float(row[estimate_col])
                lower = float(row[lower_col])
                upper = float(row[upper_col])
            elif row_a == MTP_MODEL and row_b == DENSE_MODEL:
                mean = -float(row[estimate_col])
                lower = -float(row[upper_col])
                upper = -float(row[lower_col])
            else:
                continue

            matched[budget] = IntervalPoint(
                budget=budget,
                mean=mean,
                lower=lower,
                upper=upper,
            )

    missing = [budget for budget in BUDGET_ORDER if budget not in matched]
    if missing:
        raise ValueError(f"Missing MTP - Dense test-loss contrasts: {missing}")

    return [matched[budget] for budget in BUDGET_ORDER]


def load_mtp_loss() -> list[IntervalPoint]:
    if not DESCRIPTIVES_CSV.exists():
        raise FileNotFoundError(f"Missing descriptives artifact: {DESCRIPTIVES_CSV}")

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

        rows = list(reader)

    available_metrics = {row[metric_col].strip() for row in rows}
    mtp_loss_metric = choose_metric(available_metrics, MTP_LOSS_CANDIDATES)

    matched: dict[str, IntervalPoint] = {}

    for row in rows:
        if row[model_col].strip() != MTP_MODEL:
            continue
        if row[metric_col].strip() != mtp_loss_metric:
            continue
        if row[mean_col].strip() == "":
            continue

        budget = normalize_budget(row[budget_col])
        if budget not in BUDGET_ORDER:
            continue

        mean = float(row[mean_col])

        if half_width_col is not None:
            half_width = read_float(row, half_width_col)
            if half_width is not None:
                lower = mean - half_width
                upper = mean + half_width
            else:
                lower = read_float(row, lower_col)
                upper = read_float(row, upper_col)
        else:
            lower = read_float(row, lower_col)
            upper = read_float(row, upper_col)

        if lower is None or upper is None:
            lower = mean
            upper = mean

        matched[budget] = IntervalPoint(
            budget=budget,
            mean=mean,
            lower=lower,
            upper=upper,
        )

    missing = [budget for budget in BUDGET_ORDER if budget not in matched]
    if missing:
        raise ValueError(f"Missing MTP loss values: {missing}")

    print(f"MTP loss metric used: {mtp_loss_metric}")
    return [matched[budget] for budget in BUDGET_ORDER]


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
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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


def y_error(point: IntervalPoint) -> tuple[list[float], list[float]]:
    return ([point.mean - point.lower], [point.upper - point.mean])


def plot_metric_panel(
    ax: plt.Axes,
    points: list[IntervalPoint],
    y_label: str,
    include_zero_line: bool,
) -> None:
    xs = [BUDGET_TO_X[point.budget] for point in points]
    ys = [point.mean for point in points]

    ax.plot(
        xs,
        ys,
        color=MTP_COLOR,
        linewidth=1.65,
        alpha=0.86,
        zorder=2,
    )

    for x_value, point in zip(xs, points, strict=True):
        ax.errorbar(
            x_value,
            point.mean,
            yerr=y_error(point),
            fmt="s",
            color=MTP_COLOR,
            markerfacecolor=MTP_COLOR,
            markeredgecolor=TEXT_COLOR,
            markeredgewidth=0.7,
            markersize=6.2,
            linewidth=0,
            elinewidth=0.8,
            capsize=2.4,
            capthick=0.8,
            alpha=0.9,
            zorder=3,
        )

    if include_zero_line:
        ax.axhline(0.0, color=TEXT_COLOR, linewidth=0.85, alpha=0.72)

    y_values = [value for point in points for value in (point.lower, point.upper, point.mean)]
    y_span = max(y_values) - min(y_values)
    ax.set_ylim(
        min(y_values) - max(y_span * 0.14, 0.005),
        max(y_values) + max(y_span * 0.14, 0.005),
    )

    ax.set_xlim(-0.18, 2.18)
    ax.set_xticks([BUDGET_TO_X[budget] for budget in BUDGET_ORDER])
    ax.set_xticklabels(BUDGET_ORDER)
    ax.set_xlabel("Training Token Budget")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    enforce_black_text(ax)


def plot(
    mtp_difference_points: list[IntervalPoint],
    mtp_loss_points: list[IntervalPoint],
) -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    plot_metric_panel(
        ax=axes[0],
        points=mtp_difference_points,
        y_label="Test Loss Difference (MTP - Dense)",
        include_zero_line=True,
    )
    plot_metric_panel(
        ax=axes[1],
        points=mtp_loss_points,
        y_label="MTP Loss",
        include_zero_line=False,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.18,
        top=0.96,
        wspace=0.35,
    )

    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_PNG}")


def main() -> None:
    mtp_difference_points = load_mtp_test_loss_differences()
    mtp_loss_points = load_mtp_loss()
    plot(
        mtp_difference_points=mtp_difference_points,
        mtp_loss_points=mtp_loss_points,
    )


if __name__ == "__main__":
    main()
