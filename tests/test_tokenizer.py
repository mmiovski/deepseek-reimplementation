from __future__ import annotations

import json
from pathlib import Path

from deepseek_reimpl.tokenizer.load_tokenizer import load_tokenizer
from deepseek_reimpl.tokenizer.tokenizer_utils import save_tokenizer
from deepseek_reimpl.tokenizer.train_tokenizer import train_byte_level_bpe_tokenizer


def test_train_byte_level_bpe_tokenizer_on_tiny_corpus(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny_corpus.txt"
    corpus_path.write_text(
        "Once upon a time there was a small cat.\n"
        "The small cat liked stories.\n"
        "Stories help test tokenizers.\n",
        encoding="utf-8",
    )

    special_tokens = {
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }

    tokenizer, effective_training_chars, was_capped = train_byte_level_bpe_tokenizer(
        input_text_files=[corpus_path],
        vocab_size=128,
        min_frequency=1,
        special_tokens_config=special_tokens,
    )

    encoded = tokenizer.encode("small cat")
    decoded = tokenizer.decode(encoded.ids)

    assert effective_training_chars == len(corpus_path.read_text(encoding="utf-8"))
    assert was_capped is False
    assert tokenizer.get_vocab_size() <= 128
    assert tokenizer.token_to_id("<unk>") is not None
    assert tokenizer.token_to_id("<bos>") is not None
    assert tokenizer.token_to_id("<eos>") is not None
    assert tokenizer.token_to_id("<pad>") is not None
    assert len(encoded.ids) > 0
    assert "small" in decoded
    assert "cat" in decoded


def test_save_and_load_tokenizer_roundtrip(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny_corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"

    corpus_path.write_text(
        "A tiny tokenizer test corpus.\n" "Another tiny sentence for testing.\n",
        encoding="utf-8",
    )

    special_tokens = {
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }

    tokenizer, _, _ = train_byte_level_bpe_tokenizer(
        input_text_files=[corpus_path],
        vocab_size=128,
        min_frequency=1,
        special_tokens_config=special_tokens,
    )

    saved_path = save_tokenizer(tokenizer, tokenizer_path)
    loaded = load_tokenizer(saved_path)

    assert saved_path.exists()
    assert loaded.encode("tiny test").ids == tokenizer.encode("tiny test").ids


def test_project_tokenizer_artifact_fits_model_vocab() -> None:
    from deepseek_reimpl.utils.config import load_yaml_config
    from tokenizers import Tokenizer

    tokenizer_config = load_yaml_config("configs/tokenizer/bpe_tiny.yaml")
    baseline_config = load_yaml_config("configs/model/baseline_gpt.yaml")
    mla_config = load_yaml_config("configs/model/mla.yaml")

    requested_vocab_size = tokenizer_config["tokenizer"]["vocab_size"]
    tokenizer_path = tokenizer_config["artifacts"]["tokenizer_json"]
    actual_vocab_size = Tokenizer.from_file(tokenizer_path).get_vocab_size()

    assert requested_vocab_size == 10000
    assert actual_vocab_size <= requested_vocab_size
    assert baseline_config["model"]["vocab_size"] == requested_vocab_size
    assert mla_config["model"]["vocab_size"] == requested_vocab_size
    assert actual_vocab_size <= baseline_config["model"]["vocab_size"]
    assert actual_vocab_size <= mla_config["model"]["vocab_size"]


def test_train_byte_level_bpe_tokenizer_respects_character_cap(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny_corpus.txt"
    corpus_path.write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    special_tokens = {
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }

    tokenizer, effective_training_chars, was_capped = train_byte_level_bpe_tokenizer(
        input_text_files=[corpus_path],
        vocab_size=128,
        min_frequency=1,
        special_tokens_config=special_tokens,
        max_training_chars=10,
    )

    assert tokenizer.get_vocab_size() <= 128
    assert effective_training_chars == 10
    assert was_capped is True


def test_train_tokenizer_from_config_writes_metadata(tmp_path: Path) -> None:
    from deepseek_reimpl.tokenizer.train_tokenizer import train_tokenizer_from_config

    corpus_path = tmp_path / "tiny_corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    metadata_path = tmp_path / "metadata.json"

    corpus_path.write_text("alpha beta gamma delta epsilon", encoding="utf-8")

    config = {
        "tokenizer": {
            "name": "unit_test_bpe",
            "type": "byte_level_bpe",
            "vocab_size": 128,
            "min_frequency": 1,
        },
        "special_tokens": {
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        },
        "training": {
            "input_text_files": [str(corpus_path)],
            "max_training_chars": None,
        },
        "artifacts": {
            "tokenizer_json": str(tokenizer_path),
            "metadata_json": str(metadata_path),
        },
    }

    saved_path = train_tokenizer_from_config(config)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert saved_path == tokenizer_path
    assert tokenizer_path.exists()
    assert metadata["training"]["max_training_chars"] is None
    assert metadata["training"]["was_capped"] is False
    assert metadata["training"]["effective_training_chars"] == len(
        corpus_path.read_text(encoding="utf-8")
    )
    assert metadata["actual_vocab_size"] <= 128


def test_train_tokenizer_cli_help_runs_from_script_path() -> None:
    import subprocess
    import sys
    from pathlib import Path

    script_path = Path("scripts/tokenizer/train_tokenizer.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
