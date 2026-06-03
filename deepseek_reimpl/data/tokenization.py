"""Tokenization helpers for language-modeling data."""

from __future__ import annotations

from pathlib import Path

from deepseek_reimpl.utils.paths import ensure_dir, project_path
from tokenizers import Tokenizer


def encode_text(text: str, tokenizer: Tokenizer) -> list[int]:
    """Encode text into token IDs."""
    return tokenizer.encode(text).ids


def encode_text_file(input_path: str | Path, tokenizer: Tokenizer) -> list[int]:
    """Read a text file and encode it into token IDs."""
    path = Path(input_path)

    if not path.is_absolute():
        path = project_path(path)

    text = path.read_text(encoding="utf-8")
    return encode_text(text, tokenizer)


def write_token_ids(token_ids: list[int], output_path: str | Path) -> Path:
    """Write token IDs to a plain text artifact, one integer per line."""
    path = Path(output_path)

    if not path.is_absolute():
        path = project_path(path)

    ensure_dir(path.parent)
    path.write_text("\n".join(str(token_id) for token_id in token_ids), encoding="utf-8")
    return path


def read_token_ids(input_path: str | Path) -> list[int]:
    """Read token IDs from a plain text artifact."""
    path = Path(input_path)

    if not path.is_absolute():
        path = project_path(path)

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        return []

    return [int(line) for line in content.splitlines()]
