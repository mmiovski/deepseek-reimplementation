from __future__ import annotations

from pathlib import Path

from deepseek_reimpl.utils.config import load_yaml_config, require_keys, validate_relative_paths

REPORT_EXPERIMENTS = (
    "report_00_baseline",
    "report_01_mla",
    "report_02_moe",
    "report_03_mla_moe",
    "report_04_v3_routing",
    "report_05_mtp",
)


def _assert_valid_train_config(path: str) -> None:
    config = load_yaml_config(path)
    require_keys(config, {"train"}, name=path)
    train = config["train"]

    require_keys(
        train,
        {
            "seed",
            "device",
            "batch_size",
            "block_size",
            "max_steps",
            "max_tokens",
            "eval_interval",
            "eval_batches",
            "learning_rate",
            "weight_decay",
            "betas",
            "grad_clip",
            "num_workers",
            "checkpoint_interval",
            "log_interval",
            "precision",
        },
        name=f"{path} train",
    )

    assert train["device"] == "cuda"
    assert train["precision"] == "fp32"
    assert train["max_steps"] is None
    assert train["max_tokens"] > 0
    assert train["batch_size"] > 0
    assert train["block_size"] == 128
    assert train["eval_interval"] > 0
    assert train["eval_batches"] > 0
    assert train["log_interval"] > 0


def test_controlled_train_configs_load_and_validate() -> None:
    for path in (
        "configs/train/local_pilot_fixed_budget.yaml",
        "configs/train/local_report_fixed_budget.yaml",
    ):
        _assert_valid_train_config(path)


def test_report_experiment_configs_use_wikitext2_and_report_layout() -> None:
    expected_models = {
        "report_00_baseline": "configs/model/baseline_gpt.yaml",
        "report_01_mla": "configs/model/mla.yaml",
        "report_02_moe": "configs/model/moe.yaml",
        "report_03_mla_moe": "configs/model/mla_moe.yaml",
        "report_04_v3_routing": "configs/model/v3_routing.yaml",
        "report_05_mtp": "configs/model/mtp.yaml",
    }

    for name in REPORT_EXPERIMENTS:
        path = f"configs/experiment/{name}.yaml"
        config = load_yaml_config(path)
        require_keys(config, {"experiment"}, name=name)
        experiment = config["experiment"]

        require_keys(
            experiment,
            {
                "name",
                "description",
                "model_config",
                "data_config",
                "tokenizer_config",
                "train_config",
                "output_dir",
                "metrics_dir",
                "checkpoint_dir",
            },
            name=name,
        )

        assert experiment["name"] == name
        assert experiment["model_config"] == expected_models[name]
        assert experiment["data_config"] == "configs/data/wikitext2.yaml"
        assert experiment["tokenizer_config"] == "configs/tokenizer/bpe_wikitext2.yaml"
        assert experiment["train_config"] == "configs/train/local_report_fixed_budget.yaml"
        assert experiment["output_dir"] == f"results/raw_logs/{name}"
        assert experiment["metrics_dir"] == f"results/metrics/{name}"
        assert experiment["checkpoint_dir"] == f"checkpoints/{name}"

        validate_relative_paths(
            experiment,
            (
                "model_config",
                "data_config",
                "tokenizer_config",
                "train_config",
                "output_dir",
                "metrics_dir",
                "checkpoint_dir",
            ),
        )


def test_report_configs_do_not_replace_smoke_configs() -> None:
    for name in REPORT_EXPERIMENTS:
        assert Path(f"configs/experiment/{name}.yaml").exists()

    assert Path("configs/experiment/00_baseline.yaml").exists()
    assert Path("configs/experiment/00_baseline_gpu_smoke.yaml").exists()
