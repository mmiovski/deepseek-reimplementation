"""Structured metric logging utilities."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically write a JSON payload, preserving any prior complete artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
        return output_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one JSON record to a JSONL file, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        json.dump(record, file, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    return output_path


def truncate_jsonl_to_step(path: str | Path, *, max_step: int) -> Path:
    """Atomically discard records newer than a resumable checkpoint."""
    output_path = Path(path)
    if not output_path.exists():
        return output_path
    retained: list[dict[str, Any]] = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        step = record.get("step")
        if not isinstance(step, int) or step <= max_step:
            retained.append(record)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            for record in retained:
                json.dump(record, file, sort_keys=True)
                file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
        return output_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
