from __future__ import annotations

from pathlib import Path

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
