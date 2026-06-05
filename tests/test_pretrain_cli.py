from __future__ import annotations

import subprocess
import sys


def test_run_pretrain_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train/run_pretrain.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--experiment-config" in result.stdout
