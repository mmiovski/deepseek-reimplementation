from __future__ import annotations

from pathlib import Path

from deepseek_reimpl.utils.config import load_yaml_config

EXPECTED_EXPERIMENTS = {
    "local_10m_00_baseline.yaml": "configs/model/baseline_gpt.yaml",
    "local_10m_01_mla.yaml": "configs/model/mla.yaml",
    "local_10m_02_moe.yaml": "configs/model/moe.yaml",
    "local_10m_03_mla_moe.yaml": "configs/model/mla_moe.yaml",
    "local_10m_04_v3_routing.yaml": "configs/model/v3_routing.yaml",
    "local_10m_05_mtp.yaml": "configs/model/mtp.yaml",
}


def test_local_10m_train_config_uses_fixed_token_budget() -> None:
    config = load_yaml_config("configs/train/local_experiment_10m.yaml")

    assert config["device"] == "cuda"
    assert config["batch_size"] == 8
    assert config["block_size"] == 128
    assert config["max_steps"] is None
    assert config["max_tokens"] == 10_000_000
    assert config["eval_interval"] == 1000
    assert config["eval_batches"] == 50
    assert config["precision"] == "fp32"


def test_local_10m_experiment_configs_use_local_experiment_tokenizer() -> None:
    for filename, model_config in EXPECTED_EXPERIMENTS.items():
        path = Path("configs/experiment") / filename
        config = load_yaml_config(path)

        name = filename.removesuffix(".yaml")
        assert config["experiment"]["name"] == name
        assert config["model_config"] == model_config
        assert config["data_config"] == "configs/data/fineweb_edu_10bt.yaml"
        assert (
            config["tokenizer_config"]
            == "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml"
        )
        assert config["train_config"] == "configs/train/local_experiment_10m.yaml"
        assert config["experiment"]["output_dir"] == f"results/raw_logs/{name}"
        assert config["experiment"]["metrics_dir"] == f"results/metrics/{name}"
        assert config["experiment"]["checkpoint_dir"] == f"results/checkpoints/{name}"


def test_local_10m_model_vocab_sizes_match_local_tokenizer_vocab() -> None:
    tokenizer_config = load_yaml_config(
        "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml"
    )
    vocab_size = tokenizer_config["tokenizer"]["vocab_size"]

    for model_config_path in EXPECTED_EXPERIMENTS.values():
        model_config = load_yaml_config(model_config_path)
        assert model_config["vocab_size"] == vocab_size
