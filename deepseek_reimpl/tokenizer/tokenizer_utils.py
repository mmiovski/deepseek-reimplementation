"""Shared tokenizer utility functions."""

from __future__ import annotations

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
    """Save a tokenizer JSON artifact."""
    resolved = resolve_tokenizer_artifact_path(path)
    tokenizer.save(str(resolved))
    return resolved
