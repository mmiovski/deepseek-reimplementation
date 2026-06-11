"""Tokenize processed LM text artifacts into reusable int32 token-ID files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseek_reimpl.data.tokenization import encode_text_file_to_int32_bin  # noqa: E402
from deepseek_reimpl.tokenizer.load_tokenizer import load_tokenizer  # noqa: E402
from deepseek_reimpl.utils.config import load_yaml_config  # noqa: E402
from deepseek_reimpl.utils.paths import ensure_dir, project_path  # noqa: E402


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
    token_ids_path: str,
    tokenizer,
    batch_lines: int,
) -> dict[str, Any]:
    resolved_text_path = _resolve(text_path)
    resolved_token_ids_path = _resolve(token_ids_path)

    if not resolved_text_path.exists():
        raise FileNotFoundError(f"Missing processed text artifact: {resolved_text_path}")

    num_tokens = encode_text_file_to_int32_bin(
        resolved_text_path,
        tokenizer,
        resolved_token_ids_path,
        batch_lines=batch_lines,
    )

    return {
        "split": split_name,
        "text_path": text_path,
        "token_ids_path": token_ids_path,
        "source_bytes": resolved_text_path.stat().st_size,
        "num_tokens": num_tokens,
        "dtype": "int32",
    }


def main() -> None:
    args = parse_args()

    data_config = load_yaml_config(args.data_config)
    tokenizer_config = load_yaml_config(args.tokenizer_config)

    artifacts = data_config["artifacts"]
    tokenizer_path = tokenizer_config["artifacts"]["tokenizer_json"]
    tokenizer = load_tokenizer(tokenizer_path)

    split_specs = {
        "train": ("train_text", "train_token_ids"),
        "validation": ("validation_text", "validation_token_ids"),
        "test": ("test_text", "test_token_ids"),
    }

    split_metadata: dict[str, Any] = {}
    for split_name, (text_key, token_key) in split_specs.items():
        split_metadata[split_name] = _tokenize_split(
            split_name=split_name,
            text_path=_required_artifact(artifacts, text_key),
            token_ids_path=_required_artifact(artifacts, token_key),
            tokenizer=tokenizer,
            batch_lines=args.batch_lines,
        )
        print(
            f"Wrote {split_name} token IDs to "
            f"{split_metadata[split_name]['token_ids_path']} "
            f"({split_metadata[split_name]['num_tokens']} tokens)"
        )

    metadata_path = _resolve(_required_artifact(artifacts, "tokenized_metadata"))
    ensure_dir(metadata_path.parent)

    metadata = {
        "data_config": str(args.data_config),
        "tokenizer_config": str(args.tokenizer_config),
        "tokenizer_json": tokenizer_path,
        "batch_lines": args.batch_lines,
        "dtype": "int32",
        "splits": split_metadata,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote tokenized metadata to {metadata_path}")


if __name__ == "__main__":
    main()
