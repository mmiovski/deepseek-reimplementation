from __future__ import annotations

from pathlib import Path

import pytest

from deepseek_reimpl.model.config import GPTConfig
from deepseek_reimpl.utils.config import (
    load_yaml_config,
    require_keys,
    validate_relative_paths,
)
from deepseek_reimpl.utils.paths import ensure_project_dirs, find_project_root, project_path


def test_find_project_root_contains_pyproject() -> None:
    root = find_project_root()
    assert (root / "pyproject.toml").exists()


def test_project_path_resolves_inside_repo() -> None:
    root = find_project_root()
    config_path = project_path("configs", "data", "tinystories.yaml")
    assert config_path == root / "configs" / "data" / "tinystories.yaml"


def test_ensure_project_dirs_creates_expected_directories(tmp_path: Path) -> None:
    dirs = ensure_project_dirs(root=tmp_path)

    expected_keys = {
        "data_raw",
        "data_interim",
        "data_processed",
        "data_tokenized",
        "tokenizers_trained",
        "tokenizers_metadata",
        "results",
        "checkpoints",
    }

    assert set(dirs) == expected_keys
    assert all(path.exists() and path.is_dir() for path in dirs.values())


def test_tinystories_config_loads() -> None:
    config = load_yaml_config("configs/data/tinystories.yaml")

    require_keys(
        config,
        {"dataset", "splits", "validation_test_split", "paths", "preprocessing", "artifacts"},
        name="TinyStories data",
    )

    assert config["dataset"]["name"] == "tinystories"
    assert config["dataset"]["source"] == "huggingface"
    assert config["dataset"]["text_field"] == "text"
    assert config["splits"]["train"] == "train"
    assert config["splits"]["validation_source"] == "validation"
    assert config["validation_test_split"]["method"] == "deterministic_ordered_fraction"
    assert config["validation_test_split"]["validation_fraction"] == 0.5

    validate_relative_paths(
        config["paths"],
        ("raw_dir", "interim_dir", "processed_dir", "tokenized_dir"),
    )
    validate_relative_paths(
        config["artifacts"],
        ("train_text", "validation_text", "test_text"),
    )


def test_bpe_tiny_tokenizer_config_loads() -> None:
    config = load_yaml_config("configs/tokenizer/bpe_tiny.yaml")

    require_keys(
        config,
        {"tokenizer", "special_tokens", "training", "artifacts"},
        name="BPE tokenizer",
    )

    assert config["tokenizer"]["name"] == "bpe_tiny"
    assert config["tokenizer"]["type"] == "byte_level_bpe"
    assert config["tokenizer"]["vocab_size"] > 0

    validate_relative_paths(
        config["artifacts"],
        ("tokenizer_json", "metadata_json"),
    )

    for path_value in config["training"]["input_text_files"]:
        assert not Path(path_value).is_absolute()


def _assert_valid_train_config(path: str) -> None:
    config = load_yaml_config(path)

    require_keys(config, {"train"}, name="training")
    train_config = config["train"]

    required_train_keys = {
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
    }
    require_keys(train_config, required_train_keys, name=f"{path} train")

    assert train_config["device"] in {"cpu", "cuda", "auto"}
    assert train_config["precision"] == "fp32"
    assert train_config["batch_size"] > 0
    assert train_config["block_size"] > 0
    assert train_config["learning_rate"] > 0
    assert train_config["weight_decay"] >= 0
    assert len(train_config["betas"]) == 2
    assert train_config["max_steps"] is not None or train_config["max_tokens"] is not None


def test_phase3_train_configs_load_and_validate() -> None:
    for path in (
        "configs/train/cpu_smoke.yaml",
        "configs/train/gpu_smoke.yaml",
        "configs/train/main_fixed_budget.yaml",
    ):
        _assert_valid_train_config(path)


def test_baseline_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/00_baseline.yaml")

    require_keys(config, {"experiment"}, name="baseline experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="baseline experiment",
    )

    assert experiment_config["name"] == "00_baseline"

    validate_relative_paths(
        experiment_config,
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


def test_gpu_smoke_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/00_baseline_gpu_smoke.yaml")

    require_keys(config, {"experiment"}, name="GPU smoke experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="GPU smoke experiment",
    )

    assert experiment_config["name"] == "00_baseline_gpu_smoke"
    assert experiment_config["train_config"] == "configs/train/gpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_mla_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/01_mla.yaml")

    require_keys(config, {"experiment"}, name="MLA experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="MLA experiment",
    )

    assert experiment_config["name"] == "01_mla"
    assert experiment_config["model_config"] == "configs/model/mla.yaml"
    assert experiment_config["train_config"] == "configs/train/cpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_mla_gpu_smoke_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/01_mla_gpu_smoke.yaml")

    require_keys(config, {"experiment"}, name="MLA GPU smoke experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="MLA GPU smoke experiment",
    )

    assert experiment_config["name"] == "01_mla_gpu_smoke"
    assert experiment_config["model_config"] == "configs/model/mla.yaml"
    assert experiment_config["train_config"] == "configs/train/gpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_moe_model_config_loads() -> None:
    config_dict = load_yaml_config("configs/model/moe.yaml")
    config = config_dict["model"]

    require_keys(
        config,
        {
            "name",
            "vocab_size",
            "block_size",
            "n_layers",
            "n_heads",
            "d_model",
            "d_ff",
            "dropout",
            "norm_type",
            "positional_encoding",
            "ffn_type",
            "attention_type",
            "n_routed_experts",
            "n_shared_experts",
            "moe_top_k",
            "moe_expert_d_ff",
            "moe_router_score",
            "moe_normalize_top_k_weights",
            "moe_aux_loss_weight",
            "moe_drop_tokens",
            "tie_embeddings",
        },
        name="MoE model config",
    )

    assert config["name"] == "moe_gpt"
    assert config["vocab_size"] == 10000
    assert config["ffn_type"] == "moe"
    assert config["attention_type"] == "dense"
    assert config["n_routed_experts"] == 8
    assert config["n_shared_experts"] == 1
    assert config["moe_top_k"] == 2
    assert config["moe_expert_d_ff"] == 256
    assert config["moe_aux_loss_weight"] == 0.01
    assert config["moe_drop_tokens"] is False


def test_moe_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/02_moe.yaml")

    require_keys(config, {"experiment"}, name="MoE experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="MoE experiment",
    )

    assert experiment_config["name"] == "02_moe"
    assert experiment_config["model_config"] == "configs/model/moe.yaml"
    assert experiment_config["train_config"] == "configs/train/cpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_moe_gpu_smoke_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/02_moe_gpu_smoke.yaml")

    require_keys(config, {"experiment"}, name="MoE GPU smoke experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="MoE GPU smoke experiment",
    )

    assert experiment_config["name"] == "02_moe_gpu_smoke"
    assert experiment_config["model_config"] == "configs/model/moe.yaml"
    assert experiment_config["train_config"] == "configs/train/gpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_mla_moe_model_config_loads() -> None:
    config_dict = load_yaml_config("configs/model/mla_moe.yaml")
    config = config_dict["model"]

    require_keys(
        config,
        {
            "name",
            "vocab_size",
            "block_size",
            "n_layers",
            "n_heads",
            "d_model",
            "d_ff",
            "dropout",
            "norm_type",
            "positional_encoding",
            "ffn_type",
            "attention_type",
            "mla_kv_latent_dim",
            "mla_q_rope_dim",
            "n_routed_experts",
            "n_shared_experts",
            "moe_top_k",
            "moe_expert_d_ff",
            "moe_router_score",
            "moe_normalize_top_k_weights",
            "moe_aux_loss_weight",
            "moe_drop_tokens",
            "tie_embeddings",
        },
        name="MLA+MoE model config",
    )

    assert config["name"] == "mla_moe_gpt"
    assert config["vocab_size"] == 10000
    assert config["attention_type"] == "mla"
    assert config["ffn_type"] == "moe"
    assert config["mla_kv_latent_dim"] == 64
    assert config["mla_q_rope_dim"] == 32
    assert config["n_routed_experts"] == 8
    assert config["n_shared_experts"] == 1
    assert config["moe_top_k"] == 2
    assert config["moe_expert_d_ff"] == 256
    assert config["moe_aux_loss_weight"] == 0.01
    assert config["moe_drop_tokens"] is False


def test_mla_moe_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/03_mla_moe.yaml")

    require_keys(config, {"experiment"}, name="MLA+MoE experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="MLA+MoE experiment",
    )

    assert experiment_config["name"] == "03_mla_moe"
    assert experiment_config["model_config"] == "configs/model/mla_moe.yaml"
    assert experiment_config["train_config"] == "configs/train/cpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_mla_moe_gpu_smoke_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/03_mla_moe_gpu_smoke.yaml")

    require_keys(config, {"experiment"}, name="MLA+MoE GPU smoke experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="MLA+MoE GPU smoke experiment",
    )

    assert experiment_config["name"] == "03_mla_moe_gpu_smoke"
    assert experiment_config["model_config"] == "configs/model/mla_moe.yaml"
    assert experiment_config["train_config"] == "configs/train/gpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_v3_routing_model_config_loads() -> None:
    config_dict = load_yaml_config("configs/model/v3_routing.yaml")
    config = config_dict["model"]

    require_keys(
        config,
        {
            "name",
            "vocab_size",
            "block_size",
            "n_layers",
            "n_heads",
            "d_model",
            "d_ff",
            "dropout",
            "norm_type",
            "positional_encoding",
            "ffn_type",
            "attention_type",
            "n_routed_experts",
            "n_shared_experts",
            "moe_top_k",
            "moe_expert_d_ff",
            "moe_router_score",
            "moe_normalize_top_k_weights",
            "moe_aux_loss_weight",
            "moe_drop_tokens",
            "moe_routing_mode",
            "moe_use_expert_bias",
            "moe_expert_bias_update_rate",
            "moe_expert_bias_update_interval",
            "moe_expert_bias_min",
            "moe_expert_bias_max",
            "tie_embeddings",
        },
        name="V3 routing model config",
    )

    assert config["name"] == "v3_routing_gpt"
    assert config["vocab_size"] == 10000
    assert config["attention_type"] == "dense"
    assert config["ffn_type"] == "moe"
    assert config["n_routed_experts"] == 8
    assert config["n_shared_experts"] == 1
    assert config["moe_top_k"] == 2
    assert config["moe_expert_d_ff"] == 256
    assert config["moe_aux_loss_weight"] == 0.0
    assert config["moe_routing_mode"] == "aux_loss_free_bias"
    assert config["moe_use_expert_bias"] is True
    assert config["moe_expert_bias_update_rate"] == 0.001
    assert config["moe_expert_bias_update_interval"] == 1
    assert config["moe_expert_bias_min"] == -1.0
    assert config["moe_expert_bias_max"] == 1.0
    assert config["moe_drop_tokens"] is False


def test_v3_routing_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/04_v3_routing.yaml")

    require_keys(config, {"experiment"}, name="V3 routing experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="V3 routing experiment",
    )

    assert experiment_config["name"] == "04_v3_routing"
    assert experiment_config["model_config"] == "configs/model/v3_routing.yaml"
    assert experiment_config["train_config"] == "configs/train/cpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_v3_routing_gpu_smoke_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/04_v3_routing_gpu_smoke.yaml")

    require_keys(config, {"experiment"}, name="V3 routing GPU smoke experiment")
    experiment_config = config["experiment"]

    require_keys(
        experiment_config,
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
        name="V3 routing GPU smoke experiment",
    )

    assert experiment_config["name"] == "04_v3_routing_gpu_smoke"
    assert experiment_config["model_config"] == "configs/model/v3_routing.yaml"
    assert experiment_config["train_config"] == "configs/train/gpu_smoke.yaml"

    validate_relative_paths(
        experiment_config,
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


def test_gpt_config_accepts_valid_mtp_config() -> None:
    config = GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        d_ff=64,
        mtp_enabled=True,
        mtp_num_future_tokens=2,
        mtp_loss_weight=0.5,
        mtp_share_lm_head=False,
    )

    assert config.mtp_enabled is True
    assert config.mtp_num_future_tokens == 2
    assert config.mtp_loss_weight == 0.5
    assert config.mtp_share_lm_head is False


def test_gpt_config_rejects_disabled_mtp_with_future_tokens() -> None:
    with pytest.raises(ValueError, match="mtp_num_future_tokens must be 0"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_num_future_tokens=2,
        )


def test_gpt_config_rejects_enabled_mtp_without_positive_horizons() -> None:
    with pytest.raises(ValueError, match="mtp_num_future_tokens must be positive"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_enabled=True,
            mtp_num_future_tokens=0,
            mtp_loss_weight=0.5,
        )


def test_gpt_config_rejects_enabled_mtp_with_invalid_loss_weight() -> None:
    with pytest.raises(ValueError, match="mtp_loss_weight must be positive"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_enabled=True,
            mtp_num_future_tokens=2,
            mtp_loss_weight=0.0,
        )


def test_gpt_config_rejects_mtp_horizon_at_or_above_block_size() -> None:
    with pytest.raises(ValueError, match="mtp_num_future_tokens must be smaller than block_size"):
        GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=64,
            mtp_enabled=True,
            mtp_num_future_tokens=16,
            mtp_loss_weight=0.5,
        )


def test_mtp_model_config_loads() -> None:
    config = load_yaml_config("configs/model/mtp.yaml")
    model_config = config["model"]

    assert model_config["name"] == "mtp_gpt"
    assert model_config["vocab_size"] == 10000
    assert model_config["attention_type"] == "dense"
    assert model_config["ffn_type"] == "swiglu"
    assert model_config["mtp_enabled"] is True
    assert model_config["mtp_num_future_tokens"] == 2
    assert model_config["mtp_loss_weight"] == 0.3
    assert model_config["mtp_share_lm_head"] is False


def test_mtp_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/05_mtp.yaml")
    experiment = config["experiment"]

    assert experiment["name"] == "05_mtp"
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


def test_mtp_gpu_smoke_experiment_config_loads() -> None:
    config = load_yaml_config("configs/experiment/05_mtp_gpu_smoke.yaml")
    experiment = config["experiment"]

    assert experiment["name"] == "05_mtp_gpu_smoke"
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


def test_core_model_configs_share_controlled_base_hyperparameters() -> None:
    model_paths = (
        "configs/model/baseline_gpt.yaml",
        "configs/model/mla.yaml",
        "configs/model/moe.yaml",
        "configs/model/mla_moe.yaml",
        "configs/model/v3_routing.yaml",
        "configs/model/mtp.yaml",
    )

    controlled_keys = {
        "vocab_size",
        "block_size",
        "n_layers",
        "n_heads",
        "d_model",
        "dropout",
        "norm_type",
        "positional_encoding",
        "tie_embeddings",
    }

    loaded = {path: load_yaml_config(path)["model"] for path in model_paths}
    baseline = loaded["configs/model/baseline_gpt.yaml"]

    for path, model_config in loaded.items():
        for key in controlled_keys:
            assert model_config[key] == baseline[key], f"{path} differs on controlled key {key}"


def test_dense_non_moe_model_configs_share_feedforward_width() -> None:
    model_paths = (
        "configs/model/baseline_gpt.yaml",
        "configs/model/mla.yaml",
        "configs/model/mtp.yaml",
    )

    loaded = {path: load_yaml_config(path)["model"] for path in model_paths}
    baseline = loaded["configs/model/baseline_gpt.yaml"]

    for path, model_config in loaded.items():
        assert model_config["ffn_type"] != "moe"
        assert model_config["d_ff"] == baseline["d_ff"], f"{path} differs on dense FFN width"


def test_standard_experiment_configs_use_consistent_artifact_layout() -> None:
    experiment_paths = (
        "configs/experiment/00_baseline.yaml",
        "configs/experiment/01_mla.yaml",
        "configs/experiment/02_moe.yaml",
        "configs/experiment/03_mla_moe.yaml",
        "configs/experiment/04_v3_routing.yaml",
        "configs/experiment/05_mtp.yaml",
    )

    for path in experiment_paths:
        experiment = load_yaml_config(path)["experiment"]
        name = experiment["name"]

        assert experiment["output_dir"] == f"results/raw_logs/{name}"
        assert experiment["metrics_dir"] == f"results/metrics/{name}"
        assert experiment["checkpoint_dir"] == f"checkpoints/{name}"
        assert experiment["train_config"] == "configs/train/cpu_smoke.yaml"


def test_gpu_smoke_experiment_configs_use_gpu_smoke_train_config() -> None:
    experiment_paths = (
        "configs/experiment/00_baseline_gpu_smoke.yaml",
        "configs/experiment/01_mla_gpu_smoke.yaml",
        "configs/experiment/02_moe_gpu_smoke.yaml",
        "configs/experiment/03_mla_moe_gpu_smoke.yaml",
        "configs/experiment/04_v3_routing_gpu_smoke.yaml",
        "configs/experiment/05_mtp_gpu_smoke.yaml",
    )

    for path in experiment_paths:
        experiment = load_yaml_config(path)["experiment"]
        assert experiment["train_config"] == "configs/train/gpu_smoke.yaml"
