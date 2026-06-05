from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from deepseek_reimpl.train.pretrain import _build_lm_dataloader, _require_file
from tokenizers import Tokenizer


def _write_tiny_wordlevel_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def test_require_file_returns_existing_absolute_path(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.txt"
    file_path.write_text("content", encoding="utf-8")

    assert _require_file(file_path, purpose="test artifact", remediation="create it") == file_path


def test_require_file_raises_clear_error_for_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Missing test artifact"):
        _require_file(missing_path, purpose="test artifact", remediation="create it")


def test_build_lm_dataloader_from_text_and_tokenizer_artifacts(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    text_path = tmp_path / "train.txt"

    _write_tiny_wordlevel_tokenizer(tokenizer_path)
    text_path.write_text("one two three four five six seven eight", encoding="utf-8")

    dataloader = _build_lm_dataloader(
        text_path=text_path,
        tokenizer_path=tokenizer_path,
        block_size=3,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )

    input_ids, targets = next(iter(dataloader))

    assert input_ids.shape == (2, 3)
    assert targets.shape == (2, 3)
    assert input_ids.dtype == torch.long
    assert targets.dtype == torch.long
