from __future__ import annotations

from deepseek_reimpl.utils.config import load_yaml_config, require_keys, validate_relative_paths

PILOT_EXPERIMENTS = (
    "pilot_00_baseline",
    "pilot_01_mla",
    "pilot_02_moe",
    "pilot_03_mla_moe",
    "pilot_04_v3_routing",
    "pilot_05_mtp",
)


def test_pilot_experiment_configs_use_wikitext2_and_pilot_layout() -> None:
    expected_models = {
        "pilot_00_baseline": "configs/model/baseline_gpt.yaml",
        "pilot_01_mla": "configs/model/mla.yaml",
        "pilot_02_moe": "configs/model/moe.yaml",
        "pilot_03_mla_moe": "configs/model/mla_moe.yaml",
        "pilot_04_v3_routing": "configs/model/v3_routing.yaml",
        "pilot_05_mtp": "configs/model/mtp.yaml",
    }

    for name in PILOT_EXPERIMENTS:
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
        assert experiment["train_config"] == "configs/train/local_pilot_fixed_budget.yaml"
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


def test_pilot_configs_do_not_replace_report_or_smoke_configs() -> None:
    for name in PILOT_EXPERIMENTS:
        assert load_yaml_config(f"configs/experiment/{name}.yaml")["experiment"]["name"] == name

    assert (
        load_yaml_config("configs/experiment/report_00_baseline.yaml")["experiment"]["name"]
        == "report_00_baseline"
    )
    assert (
        load_yaml_config("configs/experiment/00_baseline_gpu_smoke.yaml")["experiment"]["name"]
        == "00_baseline_gpu_smoke"
    )
