"""Byte-level BPE tokenizer training utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from deepseek_reimpl.tokenizer.tokenizer_utils import (
    get_special_tokens,
    resolve_tokenizer_artifact_path,
    save_tokenizer,
)
from deepseek_reimpl.utils.paths import project_path
from tokenizers import Tokenizer


def _validate_max_training_chars(max_training_chars: int | None) -> None:
    if max_training_chars is not None and max_training_chars <= 0:
        raise ValueError("max_training_chars must be positive or null")


def _read_training_text(
    input_text_files: list[str | Path],
    *,
    max_training_chars: int | None,
) -> tuple[str, int, bool]:
    """Read tokenizer training text, optionally capped by character count."""
    _validate_max_training_chars(max_training_chars)

    chunks: list[str] = []
    remaining = max_training_chars
    total_chars = 0
    was_capped = False

    for input_path in input_text_files:
        path = project_path(input_path)
        text = path.read_text(encoding="utf-8")

        if remaining is None:
            chunks.append(text)
            total_chars += len(text)
            continue

        if remaining <= 0:
            was_capped = True
            break

        chunk = text[:remaining]
        chunks.append(chunk)
        total_chars += len(chunk)
        remaining -= len(chunk)

        if len(chunk) < len(text):
            was_capped = True
            break

    return "".join(chunks), total_chars, was_capped


def _write_tokenizer_metadata(
    *,
    metadata_path: str | Path,
    tokenizer_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    artifacts_cfg: dict[str, Any],
    effective_training_chars: int,
    was_capped: bool,
    vocab_size: int,
) -> Path:
    """Write tokenizer training metadata for reproducibility."""
    resolved = resolve_tokenizer_artifact_path(metadata_path)
    payload = {
        "tokenizer": tokenizer_cfg,
        "training": {
            "input_text_files": training_cfg["input_text_files"],
            "max_training_chars": training_cfg.get("max_training_chars"),
            "effective_training_chars": effective_training_chars,
            "was_capped": was_capped,
        },
        "artifacts": artifacts_cfg,
        "actual_vocab_size": vocab_size,
    }

    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def train_byte_level_bpe_tokenizer(
    *,
    input_text_files: list[str | Path],
    vocab_size: int,
    min_frequency: int,
    special_tokens_config: dict[str, str],
    max_training_chars: int | None = None,
) -> tuple[Tokenizer, int, bool]:
    """Train a byte-level BPE tokenizer from local text files.

    Returns the tokenizer, the number of training characters actually used,
    and whether the configured character cap truncated the source text.
    """
    training_text, effective_training_chars, was_capped = _read_training_text(
        input_text_files,
        max_training_chars=max_training_chars,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_corpus = Path(tmp_dir) / "tokenizer_training_corpus.txt"
        training_corpus.write_text(training_text, encoding="utf-8")

        tokenizer = Tokenizer(BPE(unk_token=special_tokens_config["unk_token"]))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=get_special_tokens(special_tokens_config),
        )

        tokenizer.train(files=[str(training_corpus)], trainer=trainer)

    bos_token = special_tokens_config["bos_token"]
    eos_token = special_tokens_config["eos_token"]
    bos_id = tokenizer.token_to_id(bos_token)
    eos_id = tokenizer.token_to_id(eos_token)

    if bos_id is None or eos_id is None:
        raise ValueError("BOS/EOS special tokens must exist after tokenizer training.")

    tokenizer.post_processor = TemplateProcessing(
        single=f"{bos_token} $A {eos_token}",
        pair=f"{bos_token} $A {eos_token} $B:1 {eos_token}:1",
        special_tokens=[(bos_token, bos_id), (eos_token, eos_id)],
    )

    return tokenizer, effective_training_chars, was_capped


def train_tokenizer_from_config(config: dict[str, Any]) -> Path:
    """Train, save, and document a tokenizer from a tokenizer config dictionary."""
    tokenizer_cfg = config["tokenizer"]
    special_tokens_cfg = config["special_tokens"]
    training_cfg = config["training"]
    artifacts_cfg = config["artifacts"]

    tokenizer, effective_training_chars, was_capped = train_byte_level_bpe_tokenizer(
        input_text_files=training_cfg["input_text_files"],
        vocab_size=tokenizer_cfg["vocab_size"],
        min_frequency=tokenizer_cfg["min_frequency"],
        special_tokens_config=special_tokens_cfg,
        max_training_chars=training_cfg.get("max_training_chars"),
    )

    tokenizer_path = save_tokenizer(tokenizer, artifacts_cfg["tokenizer_json"])

    metadata_json = artifacts_cfg.get("metadata_json")
    if metadata_json is not None:
        _write_tokenizer_metadata(
            metadata_path=metadata_json,
            tokenizer_cfg=tokenizer_cfg,
            training_cfg=training_cfg,
            artifacts_cfg=artifacts_cfg,
            effective_training_chars=effective_training_chars,
            was_capped=was_capped,
            vocab_size=tokenizer.get_vocab_size(),
        )

    return tokenizer_path
