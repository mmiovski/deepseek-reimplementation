"""Tokenizer loading utilities."""

from __future__ import annotations

from pathlib import Path

from deepseek_reimpl.utils.paths import project_path
from tokenizers import Tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a tokenizer JSON artifact from a repo-relative or absolute path."""
    tokenizer_path = Path(path)

    if not tokenizer_path.is_absolute():
        tokenizer_path = project_path(tokenizer_path)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file does not exist: {tokenizer_path}")

    return Tokenizer.from_file(str(tokenizer_path))
