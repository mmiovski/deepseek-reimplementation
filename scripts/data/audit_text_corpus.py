"""Audit local LM text artifacts for coarse quality issues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseek_reimpl.data.text_quality import compute_text_quality_report  # noqa: E402
from deepseek_reimpl.utils.paths import ensure_dir, project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more local text files to audit.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for the combined audit report.",
    )
    parser.add_argument(
        "--separator",
        default="\n\n",
        help="Document separator used by prepared LM text artifacts.",
    )
    return parser.parse_args()


def _resolve_local_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_path(path)


def main() -> None:
    args = parse_args()

    reports: list[dict[str, Any]] = []
    for input_path in args.inputs:
        report = compute_text_quality_report(
            _resolve_local_path(input_path),
            separator=args.separator,
        )
        reports.append(report)

    payload = {"reports": reports}

    if args.output_json is not None:
        output_path = _resolve_local_path(args.output_json)
        ensure_dir(output_path.parent)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote text-quality audit to {output_path}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
