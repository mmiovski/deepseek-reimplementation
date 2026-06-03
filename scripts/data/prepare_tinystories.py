"""Prepare TinyStories text artifacts from the configured dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_reimpl.data.download import prepare_huggingface_text_dataset
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
        help="Optional cap per split for local smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_cfg = config["dataset"]
    splits_cfg = config["splits"]
    preprocessing_cfg = config["preprocessing"]
    artifacts_cfg = config["artifacts"]

    for split_name, hf_split in splits_cfg.items():
        output_key = f"{split_name}_text"
        output_path = artifacts_cfg[output_key]

        written_path = prepare_huggingface_text_dataset(
            hf_dataset_name=dataset_cfg["hf_dataset_name"],
            split=hf_split,
            text_field=dataset_cfg["text_field"],
            output_path=output_path,
            normalize_newlines=preprocessing_cfg["normalize_newlines"],
            strip_whitespace=preprocessing_cfg["strip_whitespace"],
            min_chars=preprocessing_cfg["min_chars"],
            max_examples=args.max_examples,
        )

        print(f"Wrote {split_name} text to {written_path}")


if __name__ == "__main__":
    main()
