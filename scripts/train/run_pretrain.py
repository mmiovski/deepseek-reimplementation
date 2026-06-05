"""Run a configured pretraining smoke/control experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseek_reimpl.train.pretrain import run_pretraining_from_experiment_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pretraining_from_experiment_config(args.experiment_config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
