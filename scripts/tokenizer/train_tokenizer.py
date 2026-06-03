"""Train a tokenizer from a tokenizer config."""

from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_reimpl.tokenizer.train_tokenizer import train_tokenizer_from_config
from deepseek_reimpl.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tokenizer.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tokenizer/bpe_tiny.yaml"),
        help="Path to tokenizer config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    tokenizer_path = train_tokenizer_from_config(config)

    print(f"Wrote tokenizer to {tokenizer_path}")


if __name__ == "__main__":
    main()
