"""Text preprocessing utilities for language-modeling datasets."""

from __future__ import annotations


def normalize_text(
    text: str,
    *,
    normalize_newlines: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """Normalize a single text example."""
    if normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    if strip_whitespace:
        text = text.strip()

    return text


def keep_text(text: str, *, min_chars: int = 1) -> bool:
    """Return whether a normalized text example should be kept."""
    return len(text) >= min_chars


def format_lm_text(texts: list[str], *, eos_text: str = "\n\n") -> str:
    """Join text examples into one language-modeling text stream."""
    return eos_text.join(texts)
