from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_export_main_large_feasibility_summary_cli() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/analysis/export_main_large_feasibility_summary.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "main_large_feasibility_probe" in result.stdout


def test_main_large_feasibility_summary_tracks_all_six_large_variants() -> None:
    summary_path = Path("results/analysis/main_large_feasibility_summary.json")
    csv_path = Path("results/analysis/main_large_feasibility_summary.csv")

    assert summary_path.exists()
    assert csv_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "main_large_feasibility_probe"
    assert payload["num_runs"] == 6

    expected = {
        "baseline_gpt",
        "mla_gpt",
        "mtp_gpt",
        "moe_gpt",
        "mla_moe_gpt",
        "v3_routing_gpt",
    }
    observed = {row["model_name"] for row in payload["runs"]}
    assert observed == expected

    for row in payload["runs"]:
        assert row["train_tokens"] == 8192
        assert row["steps"] == 32
        assert row["total_parameters"] > 0
        assert row["trainable_parameters"] > 0
        assert row["activated_parameters_per_token"] > 0
        assert row["tokens_per_total_parameter"] is not None
        assert row["tokens_per_trainable_parameter"] is not None
        assert row["tokens_per_activated_parameter"] is not None
        assert row["requested_tokens_per_total_parameter"] is not None
        assert row["requested_tokens_per_trainable_parameter"] is not None
        assert row["requested_tokens_per_activated_parameter"] is not None
        assert row["train_tokens_per_second"] is not None
        assert row["peak_memory_bytes"] is not None
        assert row["peak_memory_gb"] is not None

    mtp = [row for row in payload["runs"] if row["model_name"] == "mtp_gpt"][0]
    assert mtp["mtp_enabled"] is True
    assert mtp["mtp_num_future_tokens"] == 2
    assert mtp["final_mtp_loss"] is not None
    assert mtp["final_mtp_per_horizon_losses"] is not None

    for model_name in {"moe_gpt", "mla_moe_gpt", "v3_routing_gpt"}:
        row = [run for run in payload["runs"] if run["model_name"] == model_name][0]
        assert row["routing_stats_present"] is True
        assert row["mean_routing_entropy"] is not None
        assert row["mean_expert_load_variance"] is not None

    v3 = [row for row in payload["runs"] if row["model_name"] == "v3_routing_gpt"][0]
    assert v3["mean_aux_loss"] == 0.0
    assert v3["routing_modes"] is not None
    assert set(v3["routing_modes"]) == {"aux_loss_free_bias"}


def test_main_large_feasibility_summary_does_not_use_stale_field_names() -> None:
    source = Path("scripts/analysis/export_main_large_feasibility_summary.py").read_text(
        encoding="utf-8"
    )

    forbidden_fragments = [
        '"tokens_per_second"',
        '"peak_memory_mb"',
        '"train_steps"',
        'data.get("tokens_per_second")',
        'data.get("peak_memory_mb")',
        'data.get("train_steps")',
    ]
    assert [fragment for fragment in forbidden_fragments if fragment in source] == []
