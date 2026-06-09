from __future__ import annotations

from pathlib import Path

from deepseek_reimpl.utils.config import load_yaml_config, require_keys, validate_relative_paths


def test_wikitext2_config_loads() -> None:
    config = load_yaml_config("configs/data/wikitext2.yaml")

    require_keys(
        config,
        {"dataset", "splits", "paths", "preprocessing", "artifacts"},
        name="WikiText-2 data",
    )

    assert config["dataset"]["name"] == "wikitext2"
    assert config["dataset"]["source"] == "huggingface"
    assert config["dataset"]["hf_dataset_name"] == "Salesforce/wikitext"
    assert config["dataset"]["hf_dataset_config_name"] == "wikitext-2-raw-v1"
    assert config["dataset"]["text_field"] == "text"

    assert config["splits"]["train"] == "train"
    assert config["splits"]["validation"] == "validation"
    assert config["splits"]["test"] == "test"

    validate_relative_paths(
        config["paths"],
        ("raw_dir", "interim_dir", "processed_dir", "tokenized_dir"),
    )
    validate_relative_paths(
        config["artifacts"],
        ("train_text", "validation_text", "test_text"),
    )


def test_bpe_wikitext2_tokenizer_config_loads() -> None:
    config = load_yaml_config("configs/tokenizer/bpe_wikitext2.yaml")

    require_keys(
        config,
        {"tokenizer", "special_tokens", "training", "artifacts"},
        name="WikiText-2 BPE tokenizer",
    )

    assert config["tokenizer"]["name"] == "bpe_wikitext2"
    assert config["tokenizer"]["type"] == "byte_level_bpe"
    assert config["tokenizer"]["vocab_size"] == 10000

    validate_relative_paths(
        config["artifacts"],
        ("tokenizer_json", "metadata_json"),
    )

    for path_value in config["training"]["input_text_files"]:
        assert not Path(path_value).is_absolute()
        assert "wikitext2" in path_value
