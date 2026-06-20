from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_export_main_large_50m_multiseed_summary_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/analysis/export_main_large_50m_multiseed_summary.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "main_large_50m_targeted_multiseed_results" in result.stdout


def test_main_large_50m_multiseed_summary_has_expected_runs() -> None:
    summary_path = Path("results/analysis/main_large_50m_multiseed_summary.json")
    csv_path = Path("results/analysis/main_large_50m_multiseed_summary.csv")

    assert summary_path.exists()
    assert csv_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "main_large_50m_targeted_multiseed_results"
    assert payload["requested_train_tokens"] == 50_000_000
    assert payload["additional_seeds"] == [2027, 31415]
    assert payload["num_runs"] == 8

    expected_models = {
        "baseline_gpt",
        "mtp_gpt",
        "moe_gpt",
        "v3_routing_gpt",
    }
    observed_models = {row["model_name"] for row in payload["runs"]}
    assert observed_models == expected_models

    observed_seeds = {row["seed"] for row in payload["runs"]}
    assert observed_seeds == {2027, 31415}

    for model_name in expected_models:
        rows = [row for row in payload["runs"] if row["model_name"] == model_name]
        assert len(rows) == 2

    for row in payload["runs"]:
        assert row["requested_train_tokens"] == 50_000_000
        assert row["train_tokens"] >= 50_000_000
        assert row["batch_size"] == 4
        assert row["block_size"] == 256
        assert row["steps"] > 0
        assert row["train_tokens_per_second"] is not None
        assert row["peak_memory_bytes"] is not None
        assert row["validation_loss"] is not None
        assert row["test_loss"] is not None
        assert row["total_parameters"] > 0
        assert row["activated_parameters_per_token"] > 0

    for row in payload["runs"]:
        if row["model_name"] == "mtp_gpt":
            assert row["mtp_enabled"] is True
            assert row["final_mtp_loss"] is not None
        if row["model_name"] in {"moe_gpt", "v3_routing_gpt"}:
            assert row["routing_stats_present"] is True
            assert row["mean_routing_entropy"] is not None
            assert row["mean_expert_load_variance"] is not None


def test_main_large_50m_multiseed_exact_parameter_accounting() -> None:
    payload = json.loads(
        Path("results/analysis/main_large_50m_multiseed_summary.json").read_text(encoding="utf-8")
    )

    expected = {
        "baseline_gpt": (120_945_408, 120_945_408, 120_945_408),
        "mtp_gpt": (136_305_408, 136_305_408, 136_305_408),
        "moe_gpt": (220_146_432, 220_146_432, 78_588_672),
        "v3_routing_gpt": (220_146_432, 220_146_432, 78_588_672),
    }

    for row in payload["runs"]:
        assert (
            row["total_parameters"],
            row["trainable_parameters"],
            row["activated_parameters_per_token"],
        ) == expected[row["model_name"]]
