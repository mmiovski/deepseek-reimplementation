"""Print and optionally save a compact comparison of two experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return payload


def build_row(summary: dict[str, Any]) -> dict[str, Any]:
    activated = summary["activated_parameters"]
    return {
        "experiment": summary["experiment_name"],
        "model": summary["model_name"],
        "total_parameters": summary["total_parameters"],
        "activated_parameters_per_token": activated["activated_parameters_per_token"],
        "activated_to_total_ratio": activated["activated_to_total_ratio"],
        "train_tokens_per_second": summary["train_tokens_per_second"],
        "final_train_loss": summary["final_train_loss"],
        "validation_loss": summary["validation_loss"],
        "routing_stats_present": summary["routing_stats"] is not None,
    }


def compare_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_total_parameter_ratio": (
            candidate["total_parameters"] / baseline["total_parameters"]
        ),
        "candidate_activated_parameter_ratio": (
            candidate["activated_parameters_per_token"] / baseline["activated_parameters_per_token"]
        ),
        "candidate_tokens_per_second_ratio": (
            candidate["train_tokens_per_second"] / baseline["train_tokens_per_second"]
        ),
        "candidate_final_train_loss_delta": (
            candidate["final_train_loss"] - baseline["final_train_loss"]
        ),
        "candidate_validation_loss_delta": (
            candidate["validation_loss"] - baseline["validation_loss"]
        ),
    }


def format_markdown(comparison: dict[str, Any]) -> str:
    rows = [comparison["baseline"], comparison["candidate"]]
    header = (
        "| Experiment | Model | Total params | Activated/token | Activated ratio | "
        "Tokens/sec | Final train loss | Validation loss | Routing stats |"
    )
    lines = [
        header,
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['experiment']} | "
            f"{row['model']} | "
            f"{row['total_parameters']} | "
            f"{row['activated_parameters_per_token']} | "
            f"{row['activated_to_total_ratio']:.6f} | "
            f"{row['train_tokens_per_second']:.6f} | "
            f"{row['final_train_loss']:.6f} | "
            f"{row['validation_loss']:.6f} | "
            f"{row['routing_stats_present']} |"
        )

    lines.extend(
        [
            "",
            "Derived candidate/baseline metrics:",
            "",
            f"- Total parameter ratio: {comparison['candidate_total_parameter_ratio']:.6f}",
            f"- Activated-parameter ratio: {comparison['candidate_activated_parameter_ratio']:.6f}",
            f"- Tokens/sec ratio: {comparison['candidate_tokens_per_second_ratio']:.6f}",
            f"- Final train loss delta: {comparison['candidate_final_train_loss_delta']:.6f}",
            f"- Validation loss delta: {comparison['candidate_validation_loss_delta']:.6f}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline = build_row(load_summary(args.baseline_summary))
    candidate = build_row(load_summary(args.candidate_summary))
    comparison = compare_rows(baseline, candidate)
    markdown = format_markdown(comparison)

    print(markdown)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
