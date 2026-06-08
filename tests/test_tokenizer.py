from __future__ import annotations

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

    tokenizer = train_byte_level_bpe_tokenizer(
        input_text_files=[corpus_path],
        vocab_size=128,
        min_frequency=1,
        special_tokens_config=special_tokens,
    )

    encoded = tokenizer.encode("small cat")
    decoded = tokenizer.decode(encoded.ids)

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

    tokenizer = train_byte_level_bpe_tokenizer(
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
