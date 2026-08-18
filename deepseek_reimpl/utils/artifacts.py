"""Integrity and publication helpers for reproducible artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically for hashing."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(payload: Any) -> str:
    """Return a canonical SHA-256 digest for a JSON-compatible value."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Durably replace UTF-8 text through a same-directory temporary file."""
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
            file.write(text)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
        return output_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Durably replace JSON data through a same-directory temporary file."""
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
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, output_path)
        temporary_path = None
        return output_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def staged_directory(target: str | Path) -> Iterator[Path]:
    """Build a new immutable artifact directory and publish it atomically.

    Existing target directories are never replaced. This makes a published
    corpus/tokenizer generation immutable and guarantees that preparation
    failures cannot damage an earlier valid generation.
    """
    target_path = Path(target).resolve()
    if target_path.exists():
        raise FileExistsError(f"Refusing to replace published artifact directory: {target_path}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path = Path(
        tempfile.mkdtemp(
            dir=target_path.parent,
            prefix=f".{target_path.name}.stage-",
        )
    ).resolve()

    if stage_path.parent != target_path.parent:
        raise RuntimeError("Staging directory must share the target parent.")

    published = False
    try:
        yield stage_path
        if target_path.exists():
            raise FileExistsError(f"Artifact target appeared during preparation: {target_path}")
        # The target is guaranteed absent, so rename is the correct atomic
        # primitive. On Windows, os.replace can deny directory publication even
        # when no target exists.
        os.rename(stage_path, target_path)
        published = True
    finally:
        if not published and stage_path.exists():
            if stage_path.parent != target_path.parent or not stage_path.name.startswith(
                f".{target_path.name}.stage-"
            ):
                raise RuntimeError(f"Refusing to clean unexpected staging path: {stage_path}")
            shutil.rmtree(stage_path)
