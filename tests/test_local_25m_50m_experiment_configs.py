from __future__ import annotations

from pathlib import Path
from typing import Any

from deepseek_reimpl.utils.config import load_yaml_config

MODEL_CONFIGS = {
    "00_baseline": "configs/model/baseline_gpt.yaml",
    "01_mla": "configs/model/mla.yaml",
    "02_moe": "configs/model/moe.yaml",
    "03_mla_moe": "configs/model/mla_moe.yaml",
    "04_v3_routing": "configs/model/v3_routing.yaml",
    "05_mtp": "configs/model/mtp.yaml",
}

BUDGETS = {
    "25m": {
        "max_tokens": 25_000_000,
        "eval_interval": 2500,
        "eval_batches": 75,
        "log_interval": 250,
    },
    "50m": {
        "max_tokens": 50_000_000,
        "eval_interval": 5000,
        "eval_batches": 100,
        "log_interval": 500,
    },
}


def _vocab_size(config: dict[str, Any]) -> int:
    model = config.get("model")
    if isinstance(model, dict) and "vocab_size" in model:
        return int(model["vocab_size"])
    if "vocab_size" in config:
        return int(config["vocab_size"])
    raise AssertionError("Model config does not expose vocab_size")


def test_local_25m_50m_train_configs_use_fixed_token_budgets() -> None:
    for tag, expected in BUDGETS.items():
        wrapper = load_yaml_config(f"configs/train/local_experiment_{tag}.yaml")
        assert set(wrapper.keys()) == {"train"}

        train = wrapper["train"]
        assert train["device"] == "cuda"
        assert train["batch_size"] == 8
        assert train["block_size"] == 128
        assert train["max_steps"] is None
        assert train["max_tokens"] == expected["max_tokens"]
        assert train["eval_interval"] == expected["eval_interval"]
        assert train["eval_batches"] == expected["eval_batches"]
        assert train["log_interval"] == expected["log_interval"]
        assert train["precision"] == "fp32"


def test_local_25m_50m_experiment_configs_use_runtime_schema() -> None:
    for tag in BUDGETS:
        for suffix, model_config in MODEL_CONFIGS.items():
            name = f"local_{tag}_{suffix}"
            config = load_yaml_config(Path("configs/experiment") / f"{name}.yaml")
            assert set(config.keys()) == {"experiment"}

            experiment = config["experiment"]
            assert experiment["name"] == name
            assert experiment["model_config"] == model_config
            assert experiment["data_config"] == "configs/data/fineweb_edu_10bt.yaml"
            assert (
                experiment["tokenizer_config"]
                == "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml"
            )
            assert experiment["train_config"] == f"configs/train/local_experiment_{tag}.yaml"
            assert experiment["output_dir"] == f"results/raw_logs/{name}"
            assert experiment["metrics_dir"] == f"results/metrics/{name}"
            assert experiment["checkpoint_dir"] == f"results/checkpoints/{name}"


def test_local_25m_50m_model_vocab_sizes_match_local_tokenizer_vocab() -> None:
    tokenizer_config = load_yaml_config(
        "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml"
    )
    vocab_size = int(tokenizer_config["tokenizer"]["vocab_size"])

    for model_config_path in MODEL_CONFIGS.values():
        model_config = load_yaml_config(model_config_path)
        assert _vocab_size(model_config) == vocab_size
