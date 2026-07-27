"""Plot report-ready paired test-loss contrasts at 50M tokens."""

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
OUT_DIR = ROOT / "results" / "figures" / "balanced_10seed_matrix_report"
OUT_PNG = OUT_DIR / "report_paired_test_loss_contrasts_50m.png"

METRIC_NAME = "test_loss"
BUDGET_LABEL = "50M"

CONTRASTS_TO_PLOT = [
    ("dense_121m", "mla_121m", "MLA - Dense"),
    ("dense_121m", "mtp_121m", "MTP - Dense"),
    ("dense_121m", "moe_220m", "MoE - Dense"),
    ("dense_121m", "mla_moe_220m", "MLA+MoE - Dense"),
    ("mla_121m", "mla_moe_220m", "MLA+MoE - MLA"),
    ("moe_220m", "mla_moe_220m", "MLA+MoE - MoE"),
    ("moe_220m", "v3_routing_220m", "V3 Routing - MoE"),
]

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


@dataclass(frozen=True)
class ContrastPoint:
    model_a: str
    model_b: str
    label: str
    estimate: float
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


def load_contrast_rows() -> list[ContrastPoint]:
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

        candidate_rows = []
        for row in reader:
            if row[metric_col].strip() != METRIC_NAME:
                continue
            if normalize_budget(row[budget_col]) != BUDGET_LABEL:
                continue
            candidate_rows.append(row)

    points: list[ContrastPoint] = []

    for model_a, model_b, label in CONTRASTS_TO_PLOT:
        matched_row: dict[str, str] | None = None
        reversed_direction = False

        for row in candidate_rows:
            row_a = row[model_a_col].strip()
            row_b = row[model_b_col].strip()

            if row_a == model_a and row_b == model_b:
                matched_row = row
                reversed_direction = False
                break

            if row_a == model_b and row_b == model_a:
                matched_row = row
                reversed_direction = True
                break

        if matched_row is None:
            raise ValueError(
                f"Missing paired contrast for {model_b} minus {model_a} " f"at {BUDGET_LABEL}."
            )

        estimate = float(matched_row[estimate_col])
        lower = float(matched_row[lower_col])
        upper = float(matched_row[upper_col])

        if reversed_direction:
            estimate, lower, upper = -estimate, -upper, -lower

        points.append(
            ContrastPoint(
                model_a=model_a,
                model_b=model_b,
                label=label,
                estimate=estimate,
                lower=lower,
                upper=upper,
            )
        )

    print(f"Read paired contrasts from: {source_csv}")
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


def plot(points: list[ContrastPoint]) -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    y_positions = list(range(len(points)))

    for y_position, point in zip(y_positions, points, strict=True):
        lower_error = point.estimate - point.lower
        upper_error = point.upper - point.estimate

        ax.errorbar(
            point.estimate,
            y_position,
            xerr=[[lower_error], [upper_error]],
            fmt="o",
            color=MODEL_COLORS[point.model_b],
            markersize=6.8,
            linewidth=1.9,
            elinewidth=1.05,
            capsize=3.8,
        )

    all_interval_values = [
        value for point in points for value in (point.lower, point.upper, point.estimate)
    ]
    x_min = min(all_interval_values)
    x_max = max(all_interval_values)
    x_span = x_max - x_min
    x_pad = max(x_span * 0.12, 0.01)

    ax.axvline(0.0, color="#000000", linewidth=0.9, alpha=0.75)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([point.label for point in points])
    ax.invert_yaxis()
    ax.set_xlabel("Test Loss Difference (Model B - Model A)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)

    enforce_black_text(ax)

    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.13, top=0.98)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_PNG}")


def main() -> None:
    points = load_contrast_rows()
    plot(points)


if __name__ == "__main__":
    main()
