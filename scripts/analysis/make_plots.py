"""Build report-ready PNG plots from training logs and summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a training log."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                msg = f"Expected JSON object record in {path}"
                raise TypeError(msg)
            records.append(payload)
    return records


def load_summary(path: Path) -> dict[str, Any]:
    """Load one summary JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return payload


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _records_by_type(records: list[dict[str, Any]], record_type: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("record_type") == record_type]


def plot_training_loss(
    labeled_logs: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    """Plot training loss against processed tokens."""
    plt.figure(figsize=(6.5, 4.0))

    for label, records in labeled_logs:
        train_records = _records_by_type(records, "train")
        tokens = [
            record["tokens"] for record in train_records if record.get("train_loss") is not None
        ]
        losses = [
            record["train_loss"] for record in train_records if record.get("train_loss") is not None
        ]
        if tokens and losses:
            plt.plot(tokens, losses, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.xlabel("Training tokens")
    plt.ylabel("Training loss")
    plt.title("Training loss by token budget")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_png(output_path)


def plot_validation_loss(
    labeled_logs: list[tuple[str, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    """Plot validation loss against processed tokens."""
    plt.figure(figsize=(6.5, 4.0))

    for label, records in labeled_logs:
        eval_records = [
            record
            for record in _records_by_type(records, "eval")
            if record.get("split") == "validation" and record.get("loss") is not None
        ]
        tokens = [record["tokens"] for record in eval_records]
        losses = [record["loss"] for record in eval_records]
        if tokens and losses:
            plt.plot(tokens, losses, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.xlabel("Training tokens")
    plt.ylabel("Validation loss")
    plt.title("Validation loss by token budget")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_png(output_path)


def plot_tokens_per_second(
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot final training throughput by experiment."""
    labels = [str(summary.get("experiment_name")) for summary in summaries]
    values = [float(summary.get("train_tokens_per_second", 0.0)) for summary in summaries]

    plt.figure(figsize=(7.0, 4.0))
    plt.bar(labels, values)
    plt.xlabel("Experiment")
    plt.ylabel("Training tokens/sec")
    plt.title("Training throughput by experiment")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    _save_png(output_path)


def plot_final_test_loss(
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot final held-out test loss by experiment."""
    labels = []
    values = []

    for summary in summaries:
        test_loss = summary.get("test_loss")
        if test_loss is None:
            continue
        labels.append(str(summary.get("experiment_name")))
        values.append(float(test_loss))

    plt.figure(figsize=(7.0, 4.0))
    plt.bar(labels, values)
    plt.xlabel("Experiment")
    plt.ylabel("Held-out test loss")
    plt.title("Final held-out test loss by experiment")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    _save_png(output_path)


def plot_final_test_perplexity(
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot final held-out test perplexity by experiment."""
    labels = []
    values = []

    for summary in summaries:
        test_perplexity = summary.get("test_perplexity")
        if test_perplexity is None:
            continue
        labels.append(str(summary.get("experiment_name")))
        values.append(float(test_perplexity))

    plt.figure(figsize=(7.0, 4.0))
    plt.bar(labels, values)
    plt.xlabel("Experiment")
    plt.ylabel("Held-out test perplexity")
    plt.title("Final held-out test perplexity by experiment")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    _save_png(output_path)


def plot_validation_loss_vs_activated_parameters(
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot validation loss against estimated activated parameters per token."""
    x_values: list[float] = []
    y_values: list[float] = []
    labels: list[str] = []

    for summary in summaries:
        activated = summary.get("activated_parameters") or {}
        if not isinstance(activated, dict):
            continue

        activated_params = activated.get("activated_parameters_per_token")
        validation_loss = summary.get("validation_loss")
        if activated_params is None or validation_loss is None:
            continue

        x_values.append(float(activated_params))
        y_values.append(float(validation_loss))
        labels.append(str(summary.get("experiment_name")))

    plt.figure(figsize=(6.5, 4.0))
    plt.scatter(x_values, y_values)

    for x_value, y_value, label in zip(x_values, y_values, labels, strict=True):
        plt.annotate(label, (x_value, y_value), textcoords="offset points", xytext=(4, 4))

    plt.xlabel("Estimated activated parameters/token")
    plt.ylabel("Validation loss")
    plt.title("Validation loss vs activated parameters")
    plt.grid(True, alpha=0.3)
    _save_png(output_path)


def parse_labeled_paths(values: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated label=path CLI arguments."""
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            msg = f"Expected LABEL=PATH, got {value!r}"
            raise ValueError(msg)
        label, raw_path = value.split("=", 1)
        if not label:
            msg = f"Label cannot be empty in {value!r}"
            raise ValueError(msg)
        parsed.append((label, Path(raw_path)))
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-log",
        action="append",
        default=[],
        help="Training log in LABEL=PATH form. Repeat for multiple experiments.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        action="append",
        default=[],
        help="Path to a summary.json file. Repeat for multiple experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory where PNG plots will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_output_dir(args.output_dir)

    labeled_logs = [
        (label, load_jsonl(path)) for label, path in parse_labeled_paths(args.train_log)
    ]
    summaries = [load_summary(path) for path in args.summary]

    if labeled_logs:
        plot_training_loss(labeled_logs, args.output_dir / "training_loss.png")
        plot_validation_loss(labeled_logs, args.output_dir / "validation_loss.png")

    if summaries:
        plot_tokens_per_second(summaries, args.output_dir / "tokens_per_second.png")
        plot_final_test_loss(summaries, args.output_dir / "test_loss.png")
        plot_final_test_perplexity(summaries, args.output_dir / "test_perplexity.png")
        plot_validation_loss_vs_activated_parameters(
            summaries,
            args.output_dir / "validation_loss_vs_activated_parameters.png",
        )

    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
