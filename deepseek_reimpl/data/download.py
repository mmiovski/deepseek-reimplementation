"""Dataset loading and preparation utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset

from deepseek_reimpl.data.preprocess import format_lm_text, keep_text, normalize_text
from deepseek_reimpl.utils.paths import ensure_dir, project_path


def iter_texts_from_records(
    records: Iterable[dict[str, Any]],
    *,
    text_field: str,
    normalize_newlines: bool = True,
    strip_whitespace: bool = True,
    min_chars: int = 1,
) -> list[str]:
    """Extract, normalize, and filter text values from dataset records."""
    texts: list[str] = []

    for record in records:
        value = record.get(text_field)

        if not isinstance(value, str):
            continue

        text = normalize_text(
            value,
            normalize_newlines=normalize_newlines,
            strip_whitespace=strip_whitespace,
        )

        if keep_text(text, min_chars=min_chars):
            texts.append(text)

    return texts


def write_text_stream(texts: list[str], output_path: str | Path) -> Path:
    """Write a list of text examples as one LM text stream."""
    path = project_path(output_path)
    ensure_dir(path.parent)
    path.write_text(format_lm_text(texts), encoding="utf-8")
    return path


def load_huggingface_texts(
    *,
    hf_dataset_name: str,
    split: str,
    text_field: str,
    normalize_newlines: bool = True,
    strip_whitespace: bool = True,
    min_chars: int = 1,
    max_examples: int | None = None,
) -> list[str]:
    """Load, normalize, and filter text examples from a Hugging Face split."""
    dataset = load_dataset(hf_dataset_name, split=split)

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    return iter_texts_from_records(
        dataset,
        text_field=text_field,
        normalize_newlines=normalize_newlines,
        strip_whitespace=strip_whitespace,
        min_chars=min_chars,
    )


def prepare_huggingface_text_dataset(
    *,
    hf_dataset_name: str,
    split: str,
    text_field: str,
    output_path: str | Path,
    normalize_newlines: bool = True,
    strip_whitespace: bool = True,
    min_chars: int = 1,
    max_examples: int | None = None,
) -> Path:
    """Load a Hugging Face text dataset split and write a local LM text file."""
    texts = load_huggingface_texts(
        hf_dataset_name=hf_dataset_name,
        split=split,
        text_field=text_field,
        normalize_newlines=normalize_newlines,
        strip_whitespace=strip_whitespace,
        min_chars=min_chars,
        max_examples=max_examples,
    )

    return write_text_stream(texts, output_path)
