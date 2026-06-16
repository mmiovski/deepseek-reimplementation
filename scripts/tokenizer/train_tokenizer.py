"""Train a tokenizer from a tokenizer config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseek_reimpl.tokenizer.train_tokenizer import train_tokenizer_from_config  # noqa: E402
from deepseek_reimpl.utils.config import load_yaml_config  # noqa: E402


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
