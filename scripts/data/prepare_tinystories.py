"""Prepare TinyStories text artifacts from the configured dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_reimpl.data.download import load_huggingface_texts, write_text_stream
from deepseek_reimpl.data.splitting import split_sequence_by_fraction
from deepseek_reimpl.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TinyStories text files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/tinystories.yaml"),
        help="Path to TinyStories data config.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap per source split for local smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_cfg = config["dataset"]
    splits_cfg = config["splits"]
    split_policy_cfg = config["validation_test_split"]
    preprocessing_cfg = config["preprocessing"]
    artifacts_cfg = config["artifacts"]

    train_texts = load_huggingface_texts(
        hf_dataset_name=dataset_cfg["hf_dataset_name"],
        split=splits_cfg["train"],
        text_field=dataset_cfg["text_field"],
        normalize_newlines=preprocessing_cfg["normalize_newlines"],
        strip_whitespace=preprocessing_cfg["strip_whitespace"],
        min_chars=preprocessing_cfg["min_chars"],
        max_examples=args.max_examples,
    )
    train_path = write_text_stream(train_texts, artifacts_cfg["train_text"])
    print(f"Wrote train text to {train_path}")

    validation_source_texts = load_huggingface_texts(
        hf_dataset_name=dataset_cfg["hf_dataset_name"],
        split=splits_cfg["validation_source"],
        text_field=dataset_cfg["text_field"],
        normalize_newlines=preprocessing_cfg["normalize_newlines"],
        strip_whitespace=preprocessing_cfg["strip_whitespace"],
        min_chars=preprocessing_cfg["min_chars"],
        max_examples=args.max_examples,
    )
    validation_test_split = split_sequence_by_fraction(
        validation_source_texts,
        validation_fraction=split_policy_cfg["validation_fraction"],
    )

    validation_path = write_text_stream(
        validation_test_split.validation_items,
        artifacts_cfg["validation_text"],
    )
    print(f"Wrote validation text to {validation_path}")

    test_path = write_text_stream(
        validation_test_split.test_items,
        artifacts_cfg["test_text"],
    )
    print(f"Wrote test text to {test_path}")


if __name__ == "__main__":
    main()
