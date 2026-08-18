"""Prepare capped text artifacts from a Hugging Face streaming dataset.

This script is intentionally cap-first. It refuses to run unless each output split
has an explicit example cap or character cap, so large streaming datasets cannot
be consumed accidentally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset  # noqa: E402

from deepseek_reimpl.data.preprocess import keep_text, normalize_text  # noqa: E402
from deepseek_reimpl.utils.artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    sha256_text,
    staged_directory,
)
from deepseek_reimpl.utils.config import load_yaml_config  # noqa: E402
from deepseek_reimpl.utils.paths import project_path  # noqa: E402


@dataclass
class SplitWriter:
    """State for one output split."""

    name: str
    path: Path
    max_examples: int | None
    max_chars: int | None
    examples: int = 0
    chars: int = 0
    serialized_chars: int = 0
    serialized_bytes: int = 0

    def is_done(self) -> bool:
        """Return whether this split has reached one of its explicit caps."""
        examples_done = self.max_examples is not None and self.examples >= self.max_examples
        chars_done = self.max_chars is not None and self.chars >= self.max_chars
        return examples_done or chars_done

    def write(self, text: str, file: TextIO, records: TextIO, *, source_index: int) -> None:
        """Append one normalized document to this split."""
        separator = "\n\n" if self.examples > 0 else ""
        file.write(separator)
        start_char = self.serialized_chars + len(separator)
        start_byte = self.serialized_bytes + len(separator.encode("utf-8"))
        text_bytes = text.encode("utf-8")
        end_char = start_char + len(text)
        end_byte = start_byte + len(text_bytes)

        records.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "split": self.name,
                    "ordinal": self.examples,
                    "source_stream_index": source_index,
                    "text_sha256": sha256_text(text),
                    "chars": len(text),
                    "bytes": len(text_bytes),
                    "start_char": start_char,
                    "end_char": end_char,
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                },
                sort_keys=True,
            )
            + "\n"
        )
        file.write(text)

        self.examples += 1
        self.chars += len(text)
        self.serialized_chars = end_char
        self.serialized_bytes = end_byte


def _positive_int_or_none(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{name} must be positive when provided, got {value}")
    return value


def _resolve_cap(cli_value: int | None, config_value: Any, *, name: str) -> int | None:
    value = cli_value if cli_value is not None else config_value
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer when provided, got {type(value)}")
    return _positive_int_or_none(value, name=name)


def _require_split_cap(split: SplitWriter) -> None:
    if split.max_examples is None and split.max_chars is None:
        raise ValueError(
            f"Split {split.name!r} has no explicit cap. "
            "Provide an example cap or character cap before streaming."
        )


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = project_path(path)
    return path


def _next_unfinished_split(split_writers: list[SplitWriter]) -> SplitWriter | None:
    for split in split_writers:
        if not split.is_done():
            return split
    return None


def _build_stream(
    *, dataset_cfg: dict[str, Any], splits_cfg: dict[str, Any], stream_cfg: dict[str, Any]
):
    dataset_kwargs: dict[str, Any] = {
        "path": dataset_cfg["hf_dataset_name"],
        "split": splits_cfg["source"],
        "streaming": True,
        "revision": dataset_cfg["hf_dataset_revision"],
    }

    dataset_config_name = dataset_cfg.get("hf_dataset_config_name")
    if dataset_config_name:
        dataset_kwargs["name"] = dataset_config_name

    stream = load_dataset(**dataset_kwargs)

    if stream_cfg.get("shuffle", False):
        stream = stream.shuffle(
            seed=int(stream_cfg.get("shuffle_seed", 42)),
            buffer_size=int(stream_cfg.get("shuffle_buffer_size", 10000)),
        )

    return stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/fineweb_edu_10bt.yaml"),
        help="Path to Hugging Face streaming data config.",
    )
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--max-train-chars", type=int, default=None)
    parser.add_argument("--max-validation-chars", type=int, default=None)
    parser.add_argument("--max-test-chars", type=int, default=None)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable streaming shuffle even if the config enables it.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=None,
        help="Override config shuffle buffer size.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Override config shuffle seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_cfg = config["dataset"]
    splits_cfg = config["splits"]
    preprocessing_cfg = config["preprocessing"]
    artifacts_cfg = config["artifacts"]
    stream_cfg = dict(config["streaming"])

    if not stream_cfg.get("enabled", False):
        raise ValueError("Streaming config must set streaming.enabled: true")

    if args.no_shuffle:
        stream_cfg["shuffle"] = False
    if args.shuffle_buffer_size is not None:
        stream_cfg["shuffle_buffer_size"] = _positive_int_or_none(
            args.shuffle_buffer_size,
            name="shuffle_buffer_size",
        )
    if args.shuffle_seed is not None:
        stream_cfg["shuffle_seed"] = args.shuffle_seed

    cap_cfg = config.get("caps", {})

    split_writers = [
        SplitWriter(
            name="validation",
            path=_resolve_path(artifacts_cfg["validation_text"]),
            max_examples=_resolve_cap(
                args.max_validation_examples,
                cap_cfg.get("validation_examples"),
                name="validation_examples",
            ),
            max_chars=_resolve_cap(
                args.max_validation_chars,
                cap_cfg.get("validation_chars"),
                name="validation_chars",
            ),
        ),
        SplitWriter(
            name="test",
            path=_resolve_path(artifacts_cfg["test_text"]),
            max_examples=_resolve_cap(
                args.max_test_examples,
                cap_cfg.get("test_examples"),
                name="test_examples",
            ),
            max_chars=_resolve_cap(
                args.max_test_chars,
                cap_cfg.get("test_chars"),
                name="test_chars",
            ),
        ),
        SplitWriter(
            name="train",
            path=_resolve_path(artifacts_cfg["train_text"]),
            max_examples=_resolve_cap(
                args.max_train_examples,
                cap_cfg.get("train_examples"),
                name="train_examples",
            ),
            max_chars=_resolve_cap(
                args.max_train_chars,
                cap_cfg.get("train_chars"),
                name="train_chars",
            ),
        ),
    ]

    if stream_cfg.get("require_explicit_caps", True):
        for split in split_writers:
            _require_split_cap(split)

    stream = _build_stream(
        dataset_cfg=dataset_cfg,
        splits_cfg=splits_cfg,
        stream_cfg=stream_cfg,
    )

    record_path_keys = {
        "train": "train_records",
        "validation": "validation_records",
        "test": "test_records",
    }
    publication_paths = [split.path for split in split_writers]
    publication_paths.extend(_resolve_path(artifacts_cfg[key]) for key in record_path_keys.values())
    metadata_path = _resolve_path(artifacts_cfg["metadata"])
    publication_paths.append(metadata_path)
    publication_parents = {path.parent for path in publication_paths}
    if len(publication_parents) != 1:
        raise ValueError("All prepared corpus artifacts must share one directory.")
    target_dir = publication_parents.pop()

    text_field = dataset_cfg["text_field"]
    with staged_directory(target_dir) as stage_dir, ExitStack() as stack:
        for split in split_writers:
            split.path = stage_dir / split.path.name
        output_files = {
            split.name: stack.enter_context(split.path.open("w", encoding="utf-8", newline="\n"))
            for split in split_writers
        }
        record_paths = {
            name: stage_dir / Path(artifacts_cfg[key]).name
            for name, key in record_path_keys.items()
        }
        record_files = {
            name: stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
            for name, path in record_paths.items()
        }

        for source_index, record in enumerate(stream):
            current_split = _next_unfinished_split(split_writers)
            if current_split is None:
                break

            value = record.get(text_field)
            if not isinstance(value, str):
                continue

            text = normalize_text(
                value,
                normalize_newlines=preprocessing_cfg["normalize_newlines"],
                strip_whitespace=preprocessing_cfg["strip_whitespace"],
            )
            if not keep_text(text, min_chars=preprocessing_cfg["min_chars"]):
                continue

            current_split.write(
                text,
                output_files[current_split.name],
                record_files[current_split.name],
                source_index=source_index,
            )

        for file in [*output_files.values(), *record_files.values()]:
            file.flush()
            os.fsync(file.fileno())

        unfinished = [split.name for split in split_writers if not split.is_done()]
        if unfinished:
            raise RuntimeError(f"Streaming dataset ended before caps were reached: {unfinished}")

        metadata = {
            "artifact_type": "prepared_corpus_metadata",
            "schema_version": 2,
            "data_config": str(args.config),
            "data_config_sha256": sha256_file(_resolve_path(args.config)),
            "dataset": {
                "name": dataset_cfg["name"],
                "hf_dataset_name": dataset_cfg["hf_dataset_name"],
                "hf_dataset_config_name": dataset_cfg.get("hf_dataset_config_name"),
                "hf_dataset_revision": dataset_cfg["hf_dataset_revision"],
                "source_split": splits_cfg["source"],
                "text_field": text_field,
            },
            "streaming": {
                "enabled": True,
                "shuffle": bool(stream_cfg.get("shuffle", False)),
                "shuffle_seed": stream_cfg.get("shuffle_seed"),
                "shuffle_buffer_size": stream_cfg.get("shuffle_buffer_size"),
                "require_explicit_caps": bool(stream_cfg.get("require_explicit_caps", True)),
            },
            "preprocessing": preprocessing_cfg,
            "splits": {
                split.name: {
                    "text_path": str(Path(artifacts_cfg[f"{split.name}_text"])),
                    "text_sha256": sha256_file(split.path),
                    "record_manifest_path": str(Path(artifacts_cfg[record_path_keys[split.name]])),
                    "record_manifest_sha256": sha256_file(record_paths[split.name]),
                    "examples": split.examples,
                    "chars": split.chars,
                    "max_examples": split.max_examples,
                    "max_chars": split.max_chars,
                    "bytes": split.serialized_bytes,
                    "serialized_chars": split.serialized_chars,
                }
                for split in split_writers
            },
        }
        atomic_write_json(stage_dir / metadata_path.name, metadata)

    for split in split_writers:
        print(f"Prepared {split.name}: {split.examples} examples, {split.chars} chars")
    print(f"Published prepared corpus to {target_dir}")


if __name__ == "__main__":
    main()
