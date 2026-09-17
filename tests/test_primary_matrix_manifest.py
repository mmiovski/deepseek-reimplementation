from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

MODELS = {
    "dense_121m",
    "mla_121m",
    "mtp_121m",
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
}
BUDGETS = {"10m": 10_000_000, "25m": 25_000_000, "50m": 50_000_000}
SEEDS = {1337, 2027, 31415, 4441, 5501, 6173, 8191, 10007, 11213, 12721}


def test_primary_manifest_is_complete_unique_and_counterbalanced() -> None:
    rows = json.loads(
        Path("results/analysis/balanced_10seed_matrix_manifest.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 180
    identities = {(row["model"], row["budget"], row["seed"]) for row in rows}
    assert identities == {
        (model, budget, seed) for model in MODELS for budget in BUDGETS for seed in SEEDS
    }
    positions = Counter((row["model"], row["planned_order_position"]) for row in rows)
    assert all(positions[(model, position)] == 5 for model in MODELS for position in range(1, 7))


def test_primary_manifest_configs_match_every_identity() -> None:
    rows = json.loads(
        Path("results/analysis/balanced_10seed_matrix_manifest.json").read_text(encoding="utf-8")
    )
    for row in rows:
        experiment = yaml.safe_load(Path(row["experiment_config"]).read_text(encoding="utf-8"))[
            "experiment"
        ]
        train = yaml.safe_load(Path(row["train_config"]).read_text(encoding="utf-8"))["train"]
        assert experiment["protocol_id"] == "primary_matrix_2026"
        assert experiment["name"] == row["experiment_name"]
        assert experiment["variant"] == row["model"]
        assert experiment["budget"] == row["budget"]
        assert experiment["seed"] == row["seed"] == train["seed"]
        assert row["max_tokens"] == BUDGETS[row["budget"]] == train["max_tokens"]
        run_root = f"results/runs/{row['experiment_name']}"
        assert experiment["output_dir"] == f"{run_root}/logs"
        assert experiment["metrics_dir"] == f"{run_root}/metrics"
        assert experiment["checkpoint_dir"] == f"{run_root}/checkpoints"
        assert train["eval_batches"] == 100
        assert train["num_workers"] == 0
        assert train["deterministic"] is True
        assert isinstance(train["checkpoint_interval"], int)
