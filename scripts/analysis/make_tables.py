"""Build report-ready comparison tables from experiment summary JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    """Load one summary JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return payload


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _megabytes(num_bytes: Any) -> float | None:
    value = _safe_float(num_bytes)
    if value is None:
        return None
    return value / 1_000_000.0


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_row(summary: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    """Normalize one training summary into a flat table row."""
    model_config = summary.get("model_config") or {}
    activated = summary.get("activated_parameters") or {}
    routing = summary.get("routing_stats") or {}

    if not isinstance(model_config, dict):
        model_config = {}
    if not isinstance(activated, dict):
        activated = {}
    if not isinstance(routing, dict):
        routing = {}

    routing_modes = routing.get("routing_modes")
    routing_mode = None
    if isinstance(routing_modes, list) and routing_modes:
        unique_modes = sorted({str(mode) for mode in routing_modes})
        routing_mode = ",".join(unique_modes)

    return {
        "summary_path": str(source_path),
        "experiment_name": summary.get("experiment_name"),
        "model_name": summary.get("model_name"),
        "seed": summary.get("seed"),
        "attention_type": model_config.get("attention_type"),
        "ffn_type": model_config.get("ffn_type"),
        "routing_mode": routing_mode,
        "mtp_enabled": summary.get("mtp_enabled"),
        "mtp_num_future_tokens": summary.get("mtp_num_future_tokens"),
        "mtp_loss_weight": summary.get("mtp_loss_weight"),
        "steps": summary.get("steps"),
        "train_tokens": summary.get("train_tokens"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "train_tokens_per_second": summary.get("train_tokens_per_second"),
        "total_parameters": summary.get("total_parameters"),
        "trainable_parameters": summary.get("trainable_parameters"),
        "activated_parameters_per_token": activated.get("activated_parameters_per_token"),
        "activated_to_total_ratio": activated.get("activated_to_total_ratio"),
        "final_train_loss": summary.get("final_train_loss"),
        "final_lm_loss": summary.get("final_lm_loss"),
        "final_mtp_loss": summary.get("final_mtp_loss"),
        "validation_loss": summary.get("validation_loss"),
        "validation_perplexity": summary.get("validation_perplexity"),
        "test_loss": summary.get("test_loss"),
        "test_perplexity": summary.get("test_perplexity"),
        "peak_memory_mb": _megabytes(summary.get("peak_memory_bytes")),
        "mean_routing_entropy": routing.get("mean_routing_entropy"),
        "mean_expert_load_variance": routing.get("mean_expert_load_variance"),
        "mean_aux_loss": routing.get("mean_aux_loss"),
    }


def build_rows(summary_paths: list[Path]) -> list[dict[str, Any]]:
    """Build normalized rows for multiple summary files."""
    return [build_row(load_summary(path), source_path=path) for path in summary_paths]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV."""
    if not rows:
        raise ValueError("Cannot write table with zero rows")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as a Markdown table."""
    if not rows:
        raise ValueError("Cannot write table with zero rows")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[field]) for field in fieldnames) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = value
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as a compact LaTeX tabular artifact."""
    if not rows:
        raise ValueError("Cannot write table with zero rows")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    column_spec = "l" * len(fieldnames)

    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(_escape_latex(field) for field in fieldnames) + r" \\",
        r"\hline",
    ]

    for row in rows:
        lines.append(
            " & ".join(_escape_latex(_format_cell(row[field])) for field in fieldnames) + r" \\"
        )

    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        required=True,
        help="Path to a summary.json file. Repeat for multiple experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory where comparison tables will be written.",
    )
    parser.add_argument(
        "--basename",
        default="model_comparison",
        help="Output filename stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.summary)

    output_dir = args.output_dir
    write_csv(output_dir / f"{args.basename}.csv", rows)
    write_markdown(output_dir / f"{args.basename}.md", rows)
    write_latex(output_dir / f"{args.basename}.tex", rows)

    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
