from __future__ import annotations

from pathlib import Path

from deepseek_reimpl.utils.config import load_yaml_config

FINEWEB_PILOT_CONFIGS = {
    "fineweb_pilot_00_baseline.yaml": "configs/model/baseline_gpt.yaml",
    "fineweb_pilot_01_mla.yaml": "configs/model/mla.yaml",
    "fineweb_pilot_02_moe.yaml": "configs/model/moe.yaml",
    "fineweb_pilot_03_mla_moe.yaml": "configs/model/mla_moe.yaml",
    "fineweb_pilot_04_v3_routing.yaml": "configs/model/v3_routing.yaml",
    "fineweb_pilot_05_mtp.yaml": "configs/model/mtp.yaml",
}


def test_fineweb_pilot_experiment_configs_are_consistent() -> None:
    for filename, model_config in FINEWEB_PILOT_CONFIGS.items():
        path = Path("configs/experiment") / filename
        experiment = load_yaml_config(path)["experiment"]

        assert experiment["name"] == filename.removesuffix(".yaml")
        assert experiment["model_config"] == model_config
        assert experiment["data_config"] == "configs/data/fineweb_edu_10bt.yaml"
        assert experiment["tokenizer_config"] == "configs/tokenizer/bpe_fineweb_edu_10bt.yaml"
        assert experiment["train_config"] == "configs/train/local_pilot_fixed_budget.yaml"
        assert experiment["output_dir"] == f"results/raw_logs/{experiment['name']}"
        assert experiment["metrics_dir"] == f"results/metrics/{experiment['name']}"
        assert experiment["checkpoint_dir"] == f"checkpoints/{experiment['name']}"


def test_fineweb_pilot_model_vocab_sizes_match_tokenizer() -> None:
    tokenizer_config = load_yaml_config("configs/tokenizer/bpe_fineweb_edu_10bt.yaml")
    vocab_size = tokenizer_config["tokenizer"]["vocab_size"]

    for model_config in FINEWEB_PILOT_CONFIGS.values():
        model = load_yaml_config(model_config)["model"]
        assert model["vocab_size"] == vocab_size
