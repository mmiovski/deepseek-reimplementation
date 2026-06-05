"""Structured metric logging utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON payload to disk, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")

    return output_path


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one JSON record to a JSONL file, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        json.dump(record, file, sort_keys=True)
        file.write("\n")

    return output_path
