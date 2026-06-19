from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_export_main_large_50m_summary_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/analysis/export_main_large_50m_summary.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "main_large_50m_fixed_budget_results" in result.stdout


def test_main_large_50m_summary_has_all_six_main_runs() -> None:
    summary_path = Path("results/analysis/main_large_50m_summary.json")
    csv_path = Path("results/analysis/main_large_50m_summary.csv")

    assert summary_path.exists()
    assert csv_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "main_large_50m_fixed_budget_results"
    assert payload["requested_train_tokens"] == 50_000_000
    assert payload["num_runs"] == 6

    expected_models = {
        "baseline_gpt",
        "mla_gpt",
        "mtp_gpt",
        "moe_gpt",
        "mla_moe_gpt",
        "v3_routing_gpt",
    }
    observed = {row["model_name"] for row in payload["runs"]}
    assert observed == expected_models

    for row in payload["runs"]:
        assert row["requested_train_tokens"] == 50_000_000
        assert row["train_tokens"] >= 50_000_000
        assert row["batch_size"] == 4
        assert row["block_size"] == 256
        assert row["steps"] > 0
        assert row["train_tokens_per_second"] is not None
        assert row["peak_memory_bytes"] is not None
        assert row["total_parameters"] > 0
        assert row["trainable_parameters"] > 0
        assert row["activated_parameters_per_token"] > 0
        assert row["tokens_per_total_parameter"] is not None
        assert row["tokens_per_trainable_parameter"] is not None
        assert row["tokens_per_activated_parameter"] is not None
        assert row["validation_loss"] is not None
        assert row["test_loss"] is not None

    mtp = [row for row in payload["runs"] if row["model_name"] == "mtp_gpt"][0]
    assert mtp["mtp_enabled"] is True
    assert mtp["final_mtp_loss"] is not None

    for model_name in {"moe_gpt", "mla_moe_gpt", "v3_routing_gpt"}:
        row = [run for run in payload["runs"] if run["model_name"] == model_name][0]
        assert row["routing_stats_present"] is True
        assert row["mean_routing_entropy"] is not None
        assert row["mean_expert_load_variance"] is not None


def test_main_large_50m_summary_uses_exact_model_specific_parameters() -> None:
    payload = json.loads(
        Path("results/analysis/main_large_50m_summary.json").read_text(encoding="utf-8")
    )

    params_by_model = {
        row["model_name"]: (
            row["total_parameters"],
            row["trainable_parameters"],
            row["activated_parameters_per_token"],
        )
        for row in payload["runs"]
    }

    assert params_by_model["baseline_gpt"] == (120_945_408, 120_945_408, 120_945_408)
    assert params_by_model["mla_gpt"] == (137_460_480, 137_460_480, 137_460_480)
    assert params_by_model["mtp_gpt"] == (136_305_408, 136_305_408, 136_305_408)
    assert params_by_model["moe_gpt"] == (220_146_432, 220_146_432, 78_588_672)
    assert params_by_model["mla_moe_gpt"] == (236_661_504, 236_661_504, 95_103_744)
    assert params_by_model["v3_routing_gpt"] == (220_146_432, 220_146_432, 78_588_672)
