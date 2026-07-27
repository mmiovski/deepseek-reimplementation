from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from deepseek_reimpl.utils.config import load_yaml_config, require_keys, validate_relative_paths
from deepseek_reimpl.utils.paths import project_path


def test_fineweb_edu_streaming_config_loads() -> None:
    config = load_yaml_config("configs/data/fineweb_edu_10bt.yaml")

    require_keys(
        config,
        {"dataset", "splits", "paths", "streaming", "preprocessing", "artifacts"},
        name="FineWeb-Edu streaming data",
    )

    assert config["dataset"]["name"] == "fineweb_edu_10bt"
    assert config["dataset"]["source"] == "huggingface_streaming"
    assert config["dataset"]["hf_dataset_name"] == "HuggingFaceFW/fineweb-edu"
    assert config["dataset"]["hf_dataset_config_name"] == "sample-10BT"
    assert config["dataset"]["text_field"] == "text"

    assert config["splits"]["source"] == "train"
    assert config["splits"]["train"] == "train"
    assert config["splits"]["validation"] == "validation"
    assert config["splits"]["test"] == "test"

    assert config["streaming"]["enabled"] is True
    assert config["streaming"]["require_explicit_caps"] is True
    assert config["streaming"]["shuffle"] is True
    assert config["streaming"]["shuffle_seed"] == 1337
    assert config["streaming"]["shuffle_buffer_size"] > 0

    validate_relative_paths(
        config["paths"],
        ("raw_dir", "interim_dir", "processed_dir", "tokenized_dir"),
    )
    validate_relative_paths(
        config["artifacts"],
        (
            "train_text",
            "validation_text",
            "test_text",
            "train_token_ids",
            "validation_token_ids",
            "test_token_ids",
            "metadata",
            "tokenized_metadata",
        ),
    )


def test_hf_streaming_script_rejects_missing_caps_before_streaming() -> None:
    script_path = project_path("scripts", "data", "prepare_hf_streaming_text.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            "configs/data/fineweb_edu_10bt.yaml",
            "--no-shuffle",
        ],
        cwd=project_path(),
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert "has no explicit cap" in result.stderr
    assert "before streaming" in result.stderr


def test_bpe_fineweb_edu_tokenizer_config_loads() -> None:
    config = load_yaml_config("configs/tokenizer/bpe_fineweb_edu_10bt.yaml")

    require_keys(
        config,
        {"tokenizer", "special_tokens", "training", "artifacts"},
        name="FineWeb-Edu BPE tokenizer",
    )

    assert config["tokenizer"]["name"] == "bpe_fineweb_edu_10bt"
    assert config["tokenizer"]["type"] == "byte_level_bpe"
    assert config["tokenizer"]["vocab_size"] == 10000
    assert config["tokenizer"]["min_frequency"] == 2

    assert config["training"]["max_training_chars"] == 50000000

    validate_relative_paths(
        config["artifacts"],
        ("tokenizer_json", "metadata_json"),
    )

    for path_value in config["training"]["input_text_files"]:
        assert not Path(path_value).is_absolute()
        assert "fineweb_edu_10bt" in path_value
