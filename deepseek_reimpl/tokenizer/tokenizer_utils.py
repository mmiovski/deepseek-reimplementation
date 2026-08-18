"""Shared tokenizer utility functions."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from deepseek_reimpl.utils.paths import ensure_dir, project_path
from tokenizers import Tokenizer


def get_special_tokens(special_tokens_config: Mapping[str, str]) -> list[str]:
    """Return special token strings in config order."""
    return list(special_tokens_config.values())


def resolve_tokenizer_artifact_path(path: str | Path) -> Path:
    """Resolve a tokenizer artifact path relative to the project root."""
    resolved = project_path(path)
    ensure_dir(resolved.parent)
    return resolved


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> Path:
    """Atomically save a tokenizer JSON artifact."""
    resolved = resolve_tokenizer_artifact_path(path)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)

        tokenizer.save(str(temporary_path))
        with temporary_path.open("r+b") as file:
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, resolved)
        temporary_path = None
        return resolved
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
