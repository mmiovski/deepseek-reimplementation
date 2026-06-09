from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.make_plots import (
    load_jsonl,
    parse_labeled_paths,
    plot_final_test_loss,
    plot_final_test_perplexity,
    plot_tokens_per_second,
    plot_training_loss,
    plot_validation_loss,
    plot_validation_loss_vs_activated_parameters,
)


def test_load_jsonl_reads_records(tmp_path: Path) -> None:
    path = tmp_path / "train_log.jsonl"
    path.write_text(
        json.dumps({"record_type": "train", "tokens": 10, "train_loss": 2.0}) + "\n",
        encoding="utf-8",
    )

    records = load_jsonl(path)

    assert records == [{"record_type": "train", "tokens": 10, "train_loss": 2.0}]


def test_parse_labeled_paths_parses_label_equals_path() -> None:
    parsed = parse_labeled_paths(["baseline=results/raw_logs/00/train_log.jsonl"])

    assert parsed == [("baseline", Path("results/raw_logs/00/train_log.jsonl"))]


def test_plot_functions_write_png_files(tmp_path: Path) -> None:
    records = [
        {"record_type": "train", "tokens": 10, "train_loss": 2.5},
        {"record_type": "train", "tokens": 20, "train_loss": 2.0},
        {
            "record_type": "eval",
            "split": "validation",
            "tokens": 20,
            "loss": 2.1,
            "perplexity": 8.17,
        },
    ]
    summaries = [
        {
            "experiment_name": "00_baseline_report",
            "train_tokens_per_second": 100.0,
            "validation_loss": 2.1,
            "test_loss": 2.2,
            "test_perplexity": 9.025,
            "activated_parameters": {"activated_parameters_per_token": 1000},
        },
        {
            "experiment_name": "01_mla_report",
            "train_tokens_per_second": 90.0,
            "validation_loss": 2.0,
            "test_loss": 2.05,
            "test_perplexity": 7.77,
            "activated_parameters": {"activated_parameters_per_token": 900},
        },
    ]

    training_path = tmp_path / "training_loss.png"
    validation_path = tmp_path / "validation_loss.png"
    throughput_path = tmp_path / "tokens_per_second.png"
    test_loss_path = tmp_path / "test_loss.png"
    test_perplexity_path = tmp_path / "test_perplexity.png"
    pareto_path = tmp_path / "validation_loss_vs_activated_parameters.png"

    plot_training_loss([("baseline", records)], training_path)
    plot_validation_loss([("baseline", records)], validation_path)
    plot_tokens_per_second(summaries, throughput_path)
    plot_final_test_loss(summaries, test_loss_path)
    plot_final_test_perplexity(summaries, test_perplexity_path)
    plot_validation_loss_vs_activated_parameters(summaries, pareto_path)

    for path in (
        training_path,
        validation_path,
        throughput_path,
        test_loss_path,
        test_perplexity_path,
        pareto_path,
    ):
        assert path.exists()
        assert path.stat().st_size > 0
