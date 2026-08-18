"""Tokenize processed LM text artifacts into reusable int32 token-ID files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from deepseek_reimpl.data.tokenization import encode_text_file_to_int32_bin  # noqa: E402
from deepseek_reimpl.tokenizer.load_tokenizer import load_tokenizer  # noqa: E402
from deepseek_reimpl.utils.artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    staged_directory,  # noqa: E402
)
from deepseek_reimpl.utils.config import load_yaml_config  # noqa: E402
from deepseek_reimpl.utils.paths import project_path  # noqa: E402


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = project_path(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-config",
        type=Path,
        required=True,
        help="Path to data YAML config with text and tokenized artifacts.",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        required=True,
        help="Path to tokenizer YAML config.",
    )
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=2048,
        help="Number of text lines to encode per tokenizer batch.",
    )
    return parser.parse_args()


def _required_artifact(artifacts: dict[str, Any], key: str) -> str:
    value = artifacts.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required artifact path: {key}")
    return value


def _tokenize_split(
    *,
    split_name: str,
    text_path: str,
    token_ids_path: Path,
    logical_token_ids_path: str,
    record_manifest_path: str,
    tokenizer,
    special_token_ids: dict[str, int],
    batch_lines: int,
) -> dict[str, Any]:
    resolved_text_path = _resolve(text_path)
    if not resolved_text_path.exists():
        raise FileNotFoundError(f"Missing processed text artifact: {resolved_text_path}")

    resolved_record_manifest = _resolve(record_manifest_path)
    if not resolved_record_manifest.exists():
        raise FileNotFoundError(f"Missing source record manifest: {resolved_record_manifest}")

    num_tokens = encode_text_file_to_int32_bin(
        resolved_text_path,
        tokenizer,
        token_ids_path,
        batch_lines=batch_lines,
    )

    token_ids = np.memmap(token_ids_path, mode="r", dtype=np.int32)
    special_token_counts = {
        name: int(np.count_nonzero(token_ids == token_id))
        for name, token_id in special_token_ids.items()
    }
    del token_ids

    forbidden_counts = {
        name: count
        for name, count in special_token_counts.items()
        if name in {"unk_token", "bos_token", "eos_token", "pad_token"} and count != 0
    }
    if forbidden_counts:
        raise RuntimeError(
            f"Unexpected special-token IDs in raw {split_name} stream: {forbidden_counts}"
        )

    return {
        "split": split_name,
        "text_path": text_path,
        "text_sha256": sha256_file(resolved_text_path),
        "record_manifest_path": record_manifest_path,
        "record_manifest_sha256": sha256_file(resolved_record_manifest),
        "token_ids_path": logical_token_ids_path,
        "token_ids_sha256": sha256_file(token_ids_path),
        "source_bytes": resolved_text_path.stat().st_size,
        "num_tokens": num_tokens,
        "token_bytes": token_ids_path.stat().st_size,
        "dtype": "int32",
        "special_token_counts": special_token_counts,
    }


def main() -> None:
    args = parse_args()

    data_config = load_yaml_config(args.data_config)
    tokenizer_config = load_yaml_config(args.tokenizer_config)

    artifacts = data_config["artifacts"]
    tokenizer_path = tokenizer_config["artifacts"]["tokenizer_json"]
    tokenizer = load_tokenizer(tokenizer_path)

    special_token_ids: dict[str, int] = {}
    for name, token in tokenizer_config["special_tokens"].items():
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Configured special token is absent from tokenizer: {token}")
        special_token_ids[name] = token_id

    split_specs = {
        "train": ("train_text", "train_records", "train_token_ids"),
        "validation": ("validation_text", "validation_records", "validation_token_ids"),
        "test": ("test_text", "test_records", "test_token_ids"),
    }

    logical_token_paths = {
        split_name: _required_artifact(artifacts, token_key)
        for split_name, (_, _, token_key) in split_specs.items()
    }
    logical_metadata_path = _required_artifact(artifacts, "tokenized_metadata")
    target_paths = [
        *(_resolve(path) for path in logical_token_paths.values()),
        _resolve(logical_metadata_path),
    ]
    target_parents = {path.parent for path in target_paths}
    if len(target_parents) != 1:
        raise ValueError("All tokenized artifacts must share one publication directory.")
    target_dir = target_parents.pop()

    with staged_directory(target_dir) as stage_dir:
        split_metadata: dict[str, Any] = {}
        for split_name, (text_key, records_key, _) in split_specs.items():
            staged_token_path = stage_dir / Path(logical_token_paths[split_name]).name
            split_metadata[split_name] = _tokenize_split(
                split_name=split_name,
                text_path=_required_artifact(artifacts, text_key),
                record_manifest_path=_required_artifact(artifacts, records_key),
                token_ids_path=staged_token_path,
                logical_token_ids_path=logical_token_paths[split_name],
                tokenizer=tokenizer,
                special_token_ids=special_token_ids,
                batch_lines=args.batch_lines,
            )
            print(
                f"Prepared {split_name} token IDs "
                f"({split_metadata[split_name]['num_tokens']} tokens)"
            )

        metadata = {
            "artifact_type": "tokenized_corpus_metadata",
            "schema_version": 2,
            "data_config": str(args.data_config),
            "data_config_sha256": sha256_file(_resolve(args.data_config)),
            "tokenizer_config": str(args.tokenizer_config),
            "tokenizer_config_sha256": sha256_file(_resolve(args.tokenizer_config)),
            "tokenizer_json": tokenizer_path,
            "tokenizer_sha256": sha256_file(_resolve(tokenizer_path)),
            "encoding": {
                "add_special_tokens": False,
                "batch_lines": args.batch_lines,
                "encoding_unit": "physical_line",
                "continuous_stream": True,
            },
            "dtype": "int32",
            "special_token_ids": special_token_ids,
            "splits": split_metadata,
        }
        atomic_write_json(stage_dir / Path(logical_metadata_path).name, metadata)

    print(f"Published tokenized artifact set to {target_dir}")


if __name__ == "__main__":
    main()
