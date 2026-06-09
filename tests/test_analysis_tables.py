from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analysis.make_tables import build_rows, write_csv, write_latex, write_markdown


def _write_summary(path: Path, *, experiment_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "model_name": "baseline_gpt",
                "seed": 1337,
                "model_config": {
                    "attention_type": "dense",
                    "ffn_type": "swiglu",
                },
                "steps": 2,
                "train_tokens": 128,
                "elapsed_seconds": 1.5,
                "train_tokens_per_second": 85.333333,
                "total_parameters": 1000,
                "trainable_parameters": 1000,
                "activated_parameters": {
                    "activated_parameters_per_token": 1000,
                    "activated_to_total_ratio": 1.0,
                },
                "final_train_loss": 2.0,
                "final_lm_loss": 2.0,
                "final_mtp_loss": None,
                "validation_loss": 2.1,
                "validation_perplexity": 8.16617,
                "test_loss": 2.2,
                "test_perplexity": 9.02501,
                "peak_memory_bytes": 2_000_000,
                "routing_stats": None,
                "mtp_enabled": False,
                "mtp_num_future_tokens": 0,
                "mtp_loss_weight": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_rows_flattens_summary_for_report_table(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, experiment_name="00_baseline_report")

    rows = build_rows([summary_path])

    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_name"] == "00_baseline_report"
    assert row["attention_type"] == "dense"
    assert row["ffn_type"] == "swiglu"
    assert row["activated_parameters_per_token"] == 1000
    assert row["peak_memory_mb"] == 2.0


def test_write_report_tables_creates_csv_markdown_and_latex(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, experiment_name="00_baseline_report")
    rows = build_rows([summary_path])

    csv_path = tmp_path / "tables" / "comparison.csv"
    markdown_path = tmp_path / "tables" / "comparison.md"
    latex_path = tmp_path / "tables" / "comparison.tex"

    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows)
    write_latex(latex_path, rows)

    with csv_path.open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))

    assert csv_rows[0]["experiment_name"] == "00_baseline_report"
    assert "| experiment_name |" in markdown_path.read_text(encoding="utf-8")
    assert r"\begin{tabular}" in latex_path.read_text(encoding="utf-8")
