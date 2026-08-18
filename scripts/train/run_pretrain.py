"""Run a configured pretraining smoke/control experiment."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--runner-pid-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def publish_runner_pid(path: Path | None) -> None:
    """Atomically identify the real interpreter behind a virtualenv launcher."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(f"{os.getpid()}\n", encoding="ascii")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    publish_runner_pid(args.runner_pid_file)
    summary = run_pretraining_from_experiment_config(args.experiment_config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
