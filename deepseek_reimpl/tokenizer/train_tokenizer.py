"""Byte-level BPE tokenizer training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from deepseek_reimpl.tokenizer.tokenizer_utils import get_special_tokens, save_tokenizer
from deepseek_reimpl.utils.paths import project_path
from tokenizers import Tokenizer


def train_byte_level_bpe_tokenizer(
    *,
    input_text_files: list[str | Path],
    vocab_size: int,
    min_frequency: int,
    special_tokens_config: dict[str, str],
) -> Tokenizer:
    """Train a byte-level BPE tokenizer from local text files."""
    input_paths = [str(project_path(path)) for path in input_text_files]

    tokenizer = Tokenizer(BPE(unk_token=special_tokens_config["unk_token"]))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=get_special_tokens(special_tokens_config),
    )

    tokenizer.train(files=input_paths, trainer=trainer)

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

    return tokenizer


def train_tokenizer_from_config(config: dict[str, Any]) -> Path:
    """Train and save a tokenizer from a tokenizer config dictionary."""
    tokenizer_cfg = config["tokenizer"]
    special_tokens_cfg = config["special_tokens"]
    training_cfg = config["training"]
    artifacts_cfg = config["artifacts"]

    tokenizer = train_byte_level_bpe_tokenizer(
        input_text_files=training_cfg["input_text_files"],
        vocab_size=tokenizer_cfg["vocab_size"],
        min_frequency=tokenizer_cfg["min_frequency"],
        special_tokens_config=special_tokens_cfg,
    )

    return save_tokenizer(tokenizer, artifacts_cfg["tokenizer_json"])
