from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_export_main_large_batch_sweep_summary_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/analysis/export_main_large_batch_sweep_summary.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "main_large_batch_size_sweep" in result.stdout


def test_main_large_batch_sweep_summary_selects_batch_size_4() -> None:
    summary_path = Path("results/analysis/main_large_batch_sweep_summary.json")
    csv_path = Path("results/analysis/main_large_batch_sweep_summary.csv")

    assert summary_path.exists()
    assert csv_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "main_large_batch_size_sweep"
    assert payload["num_runs"] == 12
    assert payload["decision"]["selected_batch_size"] == 4

    assert payload["summary_by_batch"]["2"]["runs"] == 6
    assert payload["summary_by_batch"]["4"]["runs"] == 6
    assert payload["summary_by_batch"]["4"]["max_peak_memory_gb"] < 5.25

    expected_models = {
        "baseline_gpt",
        "mla_gpt",
        "mtp_gpt",
        "moe_gpt",
        "mla_moe_gpt",
        "v3_routing_gpt",
    }

    for batch_size in {2, 4}:
        rows = [row for row in payload["runs"] if row["batch_size"] == batch_size]
        assert len(rows) == 6
        assert {row["model_name"] for row in rows} == expected_models
        for row in rows:
            assert row["train_tokens_per_second"] is not None
            assert row["peak_memory_bytes"] is not None
            assert row["peak_memory_gb"] is not None
            assert row["total_parameters"] > 0
            assert row["activated_parameters_per_token"] > 0


def test_main_large_train_configs_use_selected_standardized_batch_size() -> None:
    for budget in ["10m", "25m", "50m"]:
        path = Path("configs/train") / f"main_large_{budget}.yaml"
        wrapper = yaml.safe_load(path.read_text(encoding="utf-8"))
        train = wrapper["train"]
        assert train["batch_size"] == 4
        assert train["block_size"] == 256
        assert train["device"] == "cuda"
        assert train["precision"] == "fp32"
        assert train["max_steps"] is None


def test_main_large_manifest_records_batch_policy() -> None:
    manifest = json.loads(
        Path("configs/experiment/main_large_matrix_manifest.json").read_text(encoding="utf-8")
    )
    policy = manifest["training_policy"]
    assert policy["selected_batch_size"] == 4
    assert policy["block_size"] == 256
    assert policy["precision"] == "fp32"
    assert policy["selection_artifact"] == ("results/analysis/main_large_batch_sweep_summary.json")
