"""Lightweight text-quality audit utilities for language-model corpora."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from deepseek_reimpl.utils.artifacts import sha256_text

MOJIBAKE_MARKERS = (
    "â€™",
    "â€œ",
    "â€\x9d",
    "â€",
    "Ã",
    "Â",
    "�",
)


def split_lm_documents(text: str, *, separator: str = "\n\n") -> list[str]:
    """Split LM text into non-empty documents using the corpus separator."""
    return [document.strip() for document in text.split(separator) if document.strip()]


def _percentile(sorted_values: list[int], percentile: float) -> float:
    if not sorted_values:
        return 0.0

    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])

    index = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower

    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def count_mojibake_markers(text: str) -> dict[str, int]:
    """Count common mojibake / replacement-character markers."""
    return {marker: text.count(marker) for marker in MOJIBAKE_MARKERS}


def _documents_from_record_manifest(text: str, manifest_path: Path) -> list[str]:
    documents: list[str] = []
    previous_end = 0
    for expected_ordinal, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        if record.get("ordinal") != expected_ordinal:
            raise ValueError(f"Non-contiguous record ordinal in {manifest_path}")
        start = record.get("start_char")
        end = record.get("end_char")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < previous_end
            or end < start
        ):
            raise ValueError(f"Invalid record offsets in {manifest_path}")
        document = text[start:end]
        if sha256_text(document) != record.get("text_sha256"):
            raise ValueError(f"Record content hash mismatch in {manifest_path}")
        documents.append(document)
        previous_end = end
    return documents


def compute_text_quality_report(
    path: str | Path,
    *,
    separator: str = "\n\n",
    record_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute a compact local text-quality report for one LM corpus split."""
    resolved_path = Path(path)
    text = resolved_path.read_text(encoding="utf-8")
    manifest = None if record_manifest_path is None else Path(record_manifest_path)
    documents = (
        split_lm_documents(text, separator=separator)
        if manifest is None
        else _documents_from_record_manifest(text, manifest)
    )
    document_lengths = sorted(len(document) for document in documents)

    line_count = text.count("\n") + int(bool(text))
    blank_line_count = sum(1 for line in text.splitlines() if not line.strip())
    non_ascii_chars = sum(1 for char in text if ord(char) > 127)
    control_chars = sum(1 for char in text if ord(char) < 32 and char not in {"\n", "\r", "\t"})

    marker_counts = count_mojibake_markers(text)
    total_mojibake_markers = sum(marker_counts.values())

    length_report = {
        "min": int(document_lengths[0]) if document_lengths else 0,
        "max": int(document_lengths[-1]) if document_lengths else 0,
        "mean": float(mean(document_lengths)) if document_lengths else 0.0,
        "median": float(median(document_lengths)) if document_lengths else 0.0,
        "p95": _percentile(document_lengths, 95),
        "p99": _percentile(document_lengths, 99),
    }

    top_mojibake_markers = Counter(marker_counts).most_common()

    return {
        "path": str(resolved_path),
        "record_manifest_path": None if manifest is None else str(manifest),
        "record_boundaries_verified": manifest is not None,
        "bytes": resolved_path.stat().st_size,
        "chars": len(text),
        "lines": line_count,
        "blank_lines": blank_line_count,
        "documents": len(documents),
        "document_length_chars": length_report,
        "non_ascii_chars": non_ascii_chars,
        "control_chars": control_chars,
        "mojibake_markers": marker_counts,
        "total_mojibake_markers": total_mojibake_markers,
        "top_mojibake_markers": [
            {"marker": marker, "count": count}
            for marker, count in top_mojibake_markers
            if count > 0
        ],
    }
