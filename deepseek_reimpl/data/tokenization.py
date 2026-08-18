"""Tokenization helpers for language-modeling data."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from deepseek_reimpl.utils.paths import ensure_dir, project_path
from tokenizers import Tokenizer


def encode_text(text: str, tokenizer: Tokenizer) -> list[int]:
    """Encode raw LM text without injecting tokenizer post-processor tokens."""
    return tokenizer.encode(text, add_special_tokens=False).ids


def encode_text_file(input_path: str | Path, tokenizer: Tokenizer) -> list[int]:
    """Read a text file and encode it into token IDs.

    This helper is retained for small smoke tests. Large corpora should use
    encode_text_file_to_int32_bin instead.
    """
    path = Path(input_path)

    if not path.is_absolute():
        path = project_path(path)

    text = path.read_text(encoding="utf-8")
    return encode_text(text, tokenizer)


def _write_encoded_lines(
    lines: list[str],
    *,
    tokenizer: Tokenizer,
    output_file: Any,
) -> int:
    encodings = tokenizer.encode_batch(lines, add_special_tokens=False)
    token_ids: list[int] = []
    for encoding in encodings:
        token_ids.extend(encoding.ids)

    if not token_ids:
        return 0

    token_array = np.asarray(token_ids, dtype=np.int32)
    token_array.tofile(output_file)
    return int(token_array.size)


def encode_text_file_to_int32_bin(
    input_path: str | Path,
    tokenizer: Tokenizer,
    output_path: str | Path,
    *,
    batch_lines: int = 2048,
) -> int:
    """Atomically encode raw text into a continuous int32 LM token stream.

    Tokenizer post-processors are intentionally disabled. In particular, BOS/EOS
    tokens must not be inserted around physical lines in a multi-line document.
    """
    if batch_lines <= 0:
        raise ValueError("batch_lines must be positive.")

    input_file = Path(input_path)
    if not input_file.is_absolute():
        input_file = project_path(input_file)

    output_file = Path(output_path)
    if not output_file.is_absolute():
        output_file = project_path(output_file)

    ensure_dir(output_file.parent)

    total_tokens = 0
    line_batch: list[str] = []
    temporary_path: Path | None = None

    try:
        with (
            input_file.open("r", encoding="utf-8") as src,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_file.parent,
                prefix=f".{output_file.name}.",
                suffix=".tmp",
                delete=False,
            ) as dst,
        ):
            temporary_path = Path(dst.name)

            for line in src:
                line_batch.append(line)
                if len(line_batch) >= batch_lines:
                    total_tokens += _write_encoded_lines(
                        line_batch,
                        tokenizer=tokenizer,
                        output_file=dst,
                    )
                    line_batch.clear()

            if line_batch:
                total_tokens += _write_encoded_lines(
                    line_batch,
                    tokenizer=tokenizer,
                    output_file=dst,
                )

            dst.flush()
            os.fsync(dst.fileno())

        os.replace(temporary_path, output_file)
        temporary_path = None
        return total_tokens
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
