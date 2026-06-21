from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepseek_reimpl.utils.config import load_yaml_config

EXPECTED_EXISTING_SEEDS = [1337, 2027, 31415]
EXPECTED_ADDITIONAL_SEEDS = [
    4441,
    5501,
    6173,
    8191,
    10007,
    11213,
    12721,
    14563,
    16001,
    17749,
    19937,
    22027,
    24103,
    26557,
    28661,
    30757,
    33191,
    35591,
    38039,
    40543,
    43103,
    45641,
]
EXPECTED_ALL_SEEDS = EXPECTED_EXISTING_SEEDS + EXPECTED_ADDITIONAL_SEEDS
EXPECTED_MODELS = {"dense_121m", "mtp_121m", "moe_220m", "v3_routing_220m"}


def _load_json(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_main_large_50m_25seed_manifest_contract() -> None:
    manifest = _load_json("configs/experiment/main_large_50m_25seed_manifest.json")

    assert manifest["scope"] == "main_large_50m_25seed_targeted_paired_replication"
    assert manifest["requested_train_tokens"] == 50_000_000
    assert manifest["target_total_aligned_seeds"] == 25
    assert manifest["existing_completed_seeds"] == EXPECTED_EXISTING_SEEDS
    assert manifest["additional_seeds"] == EXPECTED_ADDITIONAL_SEEDS
    assert manifest["all_aligned_seeds"] == EXPECTED_ALL_SEEDS
    assert set(manifest["replicated_models"]) == EXPECTED_MODELS

    experiments = manifest["experiments"]
    assert len(experiments) == 100

    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in experiments:
        by_seed.setdefault(int(row["seed"]), []).append(row)

    assert sorted(by_seed) == sorted(EXPECTED_ALL_SEEDS)

    for seed in EXPECTED_ALL_SEEDS:
        rows = by_seed[seed]
        assert len(rows) == 4
        assert {row["short_name"] for row in rows} == EXPECTED_MODELS


def test_main_large_50m_25seed_train_configs() -> None:
    for seed in EXPECTED_ADDITIONAL_SEEDS:
        path = Path(f"configs/train/main_large_50m_seed{seed}.yaml")
        assert path.exists(), path

        wrapper = load_yaml_config(path)
        train = wrapper["train"]

        assert train["seed"] == seed
        assert train["device"] == "cuda"
        assert train["batch_size"] == 4
        assert train["block_size"] == 256
        assert train["max_steps"] is None
        assert train["max_tokens"] == 50_000_000
        assert train["eval_interval"] == 5000
        assert train["eval_batches"] == 100
        assert train["learning_rate"] == 0.0003
        assert train["weight_decay"] == 0.1
        assert train["checkpoint_interval"] is None
        assert train["log_interval"] == 500
        assert train["precision"] == "fp32"


def test_main_large_50m_25seed_experiment_configs() -> None:
    manifest = _load_json("configs/experiment/main_large_50m_25seed_manifest.json")

    for row in manifest["experiments"]:
        config_path = Path(row["experiment_config"])
        assert config_path.exists(), config_path

        wrapper = load_yaml_config(config_path)
        experiment = wrapper["experiment"]

        assert experiment["name"] == row["experiment_name"]
        assert experiment["model_config"] == row["model_config"]
        assert experiment["data_config"] == "configs/data/fineweb_edu_10bt.yaml"
        assert (
            experiment["tokenizer_config"]
            == "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml"
        )
        assert experiment["train_config"] == row["train_config"]
        assert experiment["output_dir"] == f"results/raw_logs/{row['experiment_name']}"
        assert experiment["metrics_dir"] == f"results/metrics/{row['experiment_name']}"
        assert experiment["checkpoint_dir"] == f"results/checkpoints/{row['experiment_name']}"


def test_main_large_50m_25seed_queue_contains_only_new_runs() -> None:
    queue = _load_json("configs/experiment/main_large_50m_25seed_queue_new_runs.json")

    assert queue["scope"] == "main_large_50m_25seed_new_run_queue"
    assert queue["total_new_runs"] == 88
    assert queue["additional_seeds"] == EXPECTED_ADDITIONAL_SEEDS
    assert set(queue["replicated_models"]) == EXPECTED_MODELS

    experiments = queue["experiments"]
    assert len(experiments) == 88
    assert {int(row["seed"]) for row in experiments} == set(EXPECTED_ADDITIONAL_SEEDS)
    assert all(row["included_in_training_queue"] is True for row in experiments)
    assert all(row["completed_at_generation"] is False for row in experiments)

    for row in experiments:
        assert Path(row["experiment_config"]).exists()
        assert Path(row["train_config"]).exists()


def test_main_large_50m_25seed_queue_script_exists() -> None:
    script_path = Path("scripts/train/run_main_large_50m_25seed_queue.ps1")
    assert script_path.exists()
    text = script_path.read_text(encoding="utf-8")

    assert "SetThreadExecutionState" in text
    assert "main_large_50m_25seed_queue_new_runs.json" in text
    assert "scripts\\train\\run_pretrain.py" in text
    assert "skipped_existing_summary" in text
