from __future__ import annotations

from deepseek_reimpl.utils.config import load_yaml_config


def test_fineweb_local_experiment_tokenizer_config_is_uncapped() -> None:
    config = load_yaml_config("configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml")

    assert config["tokenizer"]["name"] == "bpe_fineweb_edu_10bt_local_experiment"
    assert config["tokenizer"]["type"] == "byte_level_bpe"
    assert config["tokenizer"]["vocab_size"] == 10000
    assert config["training"]["input_text_files"] == ["data/processed/fineweb_edu_10bt/train.txt"]
    assert config["training"]["max_training_chars"] is None
    assert (
        config["artifacts"]["tokenizer_json"]
        == "tokenizers/trained/bpe_fineweb_edu_10bt_local_experiment.json"
    )
    assert (
        config["artifacts"]["metadata_json"]
        == "tokenizers/metadata/bpe_fineweb_edu_10bt_local_experiment_metadata.json"
    )
