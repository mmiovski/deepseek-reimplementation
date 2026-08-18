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
        "--record-manifests",
        nargs="+",
        default=None,
        help="Record manifests corresponding one-to-one with --inputs.",
    )
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
    if args.record_manifests is not None and len(args.record_manifests) != len(args.inputs):
        raise ValueError("--record-manifests must have the same length as --inputs")
    manifests = args.record_manifests or [None] * len(args.inputs)
    for input_path, manifest_path in zip(args.inputs, manifests, strict=True):
        report = compute_text_quality_report(
            _resolve_local_path(input_path),
            separator=args.separator,
            record_manifest_path=(
                None if manifest_path is None else _resolve_local_path(manifest_path)
            ),
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
