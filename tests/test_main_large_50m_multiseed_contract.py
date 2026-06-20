from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_main_large_50m_multiseed_manifest_contract() -> None:
    manifest_path = Path("configs/experiment/main_large_50m_multiseed_manifest.json")
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scope"] == "main_large_50m_targeted_multiseed_replication"
    assert manifest["base_seed"] == 1337
    assert manifest["additional_seeds"] == [2027, 31415]
    assert manifest["requested_train_tokens"] == 50_000_000
    assert manifest["batch_size"] == 4
    assert manifest["block_size"] == 256
    assert manifest["precision"] == "fp32"

    expected_models = {
        "dense_121m",
        "mtp_121m",
        "moe_220m",
        "v3_routing_220m",
    }
    assert set(manifest["replicated_models"]) == expected_models
    assert len(manifest["experiments"]) == 8


def test_main_large_50m_multiseed_train_configs() -> None:
    for seed in [2027, 31415]:
        path = Path(f"configs/train/main_large_50m_seed{seed}.yaml")
        assert path.exists()

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        train = config["train"]

        assert train["seed"] == seed
        assert traing="utf-8"))
        experiment = config["experiment"]

        assert experiment["name"] == row["experiment_name"]
        assert experiment["model_config"] in expected_model_configs
        assert experiment["data_config"] == "configs/data/fineweb_edu_10bt.yaml"
        assert (
            experiment["tokenizer_config"]
            == "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml"
        )
        assert experiment["train_config"] == row["train_config"]
        assert experiment["metrics_dir"] == f"results/metrics/{experiment['name']}"
        assert experiment["output_dir"] == f"results/raw_logs/{experiment['name']}"

        seen_names.add(experiment["name"])
        seen_metric_dirs.add(experiment["metrics_dir"])

    assert len(seen_names) == 8
    assert len(seen_metric_dirs) == 8
