"""Prepare WikiText-2 text artifacts from the configured Hugging Face dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset

from deepseek_reimpl.data.download import iter_texts_from_records, write_text_stream
from deepseek_reimpl.utils.config import load_yaml_config


def _load_wikitext_split(
    *,
    dataset_cfg: dict[str, Any],
    split: str,
    preprocessing_cfg: dict[str, Any],
    max_examples: int | None,
) -> list[str]:
    """Load and normalize one WikiText split."""
    dataset = load_dataset(
        dataset_cfg["hf_dataset_name"],
        dataset_cfg["hf_dataset_config_name"],
        split=split,
    )

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    return iter_texts_from_records(
        dataset,
        text_field=dataset_cfg["text_field"],
        normalize_newlines=preprocessing_cfg["normalize_newlines"],
        strip_whitespace=preprocessing_cfg["strip_whitespace"],
        min_chars=preprocessing_cfg["min_chars"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 text files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/wikitext2.yaml"),
        help="Path to WikiText-2 data config.",
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
    preprocessing_cfg = config["preprocessing"]
    artifacts_cfg = config["artifacts"]

    train_texts = _load_wikitext_split(
        dataset_cfg=dataset_cfg,
        split=splits_cfg["train"],
        preprocessing_cfg=preprocessing_cfg,
        max_examples=args.max_examples,
    )
    train_path = write_text_stream(train_texts, artifacts_cfg["train_text"])
    print(f"Wrote train text to {train_path}")

    validation_texts = _load_wikitext_split(
        dataset_cfg=dataset_cfg,
        split=splits_cfg["validation"],
        preprocessing_cfg=preprocessing_cfg,
        max_examples=args.max_examples,
    )
    validation_path = write_text_stream(validation_texts, artifacts_cfg["validation_text"])
    print(f"Wrote validation text to {validation_path}")

    test_texts = _load_wikitext_split(
        dataset_cfg=dataset_cfg,
        split=splits_cfg["test"],
        preprocessing_cfg=preprocessing_cfg,
        max_examples=args.max_examples,
    )
    test_path = write_text_stream(test_texts, artifacts_cfg["test_text"])
    print(f"Wrote test text to {test_path}")


if __name__ == "__main__":
    main()
