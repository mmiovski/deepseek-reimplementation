"""Build and verify the canonical 10-seed evidence index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = Path("results/analysis/" "balanced_10seed_matrix_evidence_index.json")

EXPECTED_MODELS = (
    "dense_121m",
    "mla_121m",
    "mtp_121m",
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
)

EXPECTED_BUDGETS = (
    "10m",
    "25m",
    "50m",
)

EXPECTED_SEEDS = (
    1337,
    2027,
    4441,
    5501,
    6173,
    8191,
    10007,
    11213,
    12721,
    31415,
)

ANALYSIS_GROUPS: tuple[
    tuple[str, tuple[str, ...]],
    ...,
] = (
    (
        "canonical_matrix_and_audits",
        (
            "results/analysis/" "balanced_10seed_matrix_manifest.csv",
            "results/analysis/" "balanced_10seed_matrix_manifest.json",
            "results/analysis/" "balanced_10seed_matrix_generation_summary.json",
            "results/analysis/" "balanced_10seed_matrix_completion_audit.json",
            "results/analysis/" "balanced_10seed_matrix_completion_audit_rows.csv",
            "results/analysis/" "balanced_10seed_matrix_extraction_audit.json",
            "results/analysis/" "balanced_10seed_matrix_summary_flat.csv",
            "results/analysis/" "balanced_10seed_matrix_summary_flat.json",
            "results/analysis/" "balanced_10seed_matrix_summary_schema.json",
        ),
    ),
    (
        "descriptive_statistics",
        (
            "results/analysis/" "balanced_10seed_matrix_descriptives_audit.json",
            "results/analysis/" "balanced_10seed_matrix_descriptives_full_precision.csv",
            "results/analysis/" "balanced_10seed_matrix_descriptives_full_precision.json",
            "results/analysis/" "balanced_10seed_matrix_metric_availability.csv",
            "results/analysis/" "balanced_10seed_matrix_metric_availability.json",
        ),
    ),
    (
        "global_tests",
        (
            "results/analysis/" "balanced_10seed_matrix_global_seed_records.csv",
            "results/analysis/" "balanced_10seed_matrix_global_seed_records.json",
            "results/analysis/" "balanced_10seed_matrix_global_tests_audit.json",
            "results/analysis/" "balanced_10seed_matrix_global_tests_full_precision.csv",
            "results/analysis/" "balanced_10seed_matrix_global_tests_full_precision.json",
        ),
    ),
    (
        "paired_contrasts",
        (
            "results/analysis/" "balanced_10seed_matrix_paired_seed_records.csv",
            "results/analysis/" "balanced_10seed_matrix_paired_seed_records.json",
            "results/analysis/" "balanced_10seed_matrix_paired_contrasts_audit.json",
            "results/analysis/" "balanced_10seed_matrix_paired_contrasts_full_precision.csv",
            "results/analysis/" "balanced_10seed_matrix_paired_contrasts_full_precision.json",
        ),
    ),
    (
        "budget_trends",
        (
            "results/analysis/" "balanced_10seed_matrix_budget_pair_deltas_full_precision.csv",
            "results/analysis/" "balanced_10seed_matrix_budget_pair_deltas_full_precision.json",
            "results/analysis/" "balanced_10seed_matrix_budget_trend_seed_records.csv",
            "results/analysis/" "balanced_10seed_matrix_budget_trend_seed_records.json",
            "results/analysis/" "balanced_10seed_matrix_budget_trends_audit.json",
            "results/analysis/" "balanced_10seed_matrix_budget_trends_full_precision.csv",
            "results/analysis/" "balanced_10seed_matrix_budget_trends_full_precision.json",
        ),
    ),
    (
        "mechanism_profiles",
        (
            "results/analysis/" "balanced_10seed_matrix_mechanism_profile_metric_map.json",
            "results/analysis/" "balanced_10seed_matrix_mechanism_profiles.csv",
            "results/analysis/" "balanced_10seed_matrix_mechanism_profiles.json",
            "results/analysis/" "balanced_10seed_matrix_mechanism_profiles_audit.json",
        ),
    ),
    (
        "operational_provenance",
        (
            "results/analysis/" "balanced_10seed_matrix_queue.txt",
            "results/analysis/" "balanced_10seed_matrix_queue_progress.jsonl",
        ),
    ),
    (
        "data_and_tokenizer_provenance",
        ("results/analysis/" "final_data_tokenizer_provenance.json",),
    ),
)

CORE_SCRIPT_ROLES: dict[str, tuple[str, ...]] = {
    ("scripts/analysis/" "build_balanced_10seed_matrix_manifest.py"): (
        "manifest_builder",
        "queue_script_generator",
    ),
    ("scripts/analysis/" "extract_balanced_10seed_matrix_artifacts.py"): ("canonical_extractor",),
    ("scripts/analysis/" "summarize_balanced_10seed_matrix_descriptives.py"): (
        "descriptive_statistics",
    ),
    ("scripts/analysis/" "analyze_balanced_10seed_matrix_global_tests.py"): ("global_tests",),
    ("scripts/analysis/" "analyze_balanced_10seed_matrix_paired_contrasts.py"): (
        "paired_contrasts",
    ),
    ("scripts/analysis/" "analyze_balanced_10seed_matrix_budget_trends.py"): ("budget_trends",),
    ("scripts/analysis/" "build_balanced_10seed_mechanism_profiles.py"): ("mechanism_profiles",),
    ("scripts/analysis/" "verify_final_data_tokenizer_provenance.py"): (
        "data_tokenizer_provenance_verifier",
    ),
    ("scripts/analysis/" "build_balanced_10seed_evidence_index.py"): ("evidence_index_builder",),
    ("scripts/run_balanced_10seed_matrix_queue.ps1"): ("training_queue_runner",),
}

FIGURE_DIRECTORIES: tuple[
    tuple[str, str],
    ...,
] = (
    (
        "diagnostic",
        "results/figures/" "balanced_10seed_matrix_redesign",
    ),
    (
        "report_ready",
        "results/figures/" "balanced_10seed_matrix_report",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _file_record(
    root: Path,
    path_value: str,
) -> dict[str, Any]:
    path = root / path_value

    if not path.is_file():
        raise FileNotFoundError(f"Missing evidence file: {path_value}")

    return {
        "path": path_value,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_csv(
    root: Path,
    path_value: str,
) -> tuple[list[str], list[dict[str, str]]]:
    path = root / path_value

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    return list(reader.fieldnames or []), rows


def _read_progress_records(
    root: Path,
) -> list[dict[str, Any]]:
    path = root / "results/analysis/" "balanced_10seed_matrix_queue_progress.jsonl"
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            payload = json.loads(stripped)

            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object.")

            if not isinstance(
                payload.get("experiment_config"),
                str,
            ):
                raise TypeError(
                    f"{path}:{line_number} does not contain " "a plain experiment-config path."
                )

            records.append(payload)

    return records


def _analysis_artifacts(
    root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[dict[str, Any]] = []
    all_paths: list[str] = []

    for category, paths in ANALYSIS_GROUPS:
        records = [_file_record(root, path) for path in paths]
        groups.append(
            {
                "category": category,
                "file_count": len(records),
                "files": records,
            }
        )
        all_paths.extend(paths)

    if len(all_paths) != len(set(all_paths)):
        raise RuntimeError("The analysis evidence contract contains duplicates.")

    if len(all_paths) != 38:
        raise RuntimeError("Expected 38 indexed analysis artifacts, " f"found {len(all_paths)}.")

    return groups, all_paths


def _figure_artifacts(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, set[str]],
]:
    figures: list[dict[str, Any]] = []
    script_roles: dict[str, set[str]] = {
        path: set(roles) for path, roles in CORE_SCRIPT_ROLES.items()
    }

    for classification, directory_value in FIGURE_DIRECTORIES:
        directory = root / directory_value

        if not directory.is_dir():
            raise FileNotFoundError(f"Missing figure directory: {directory_value}")

        for figure_path in sorted(directory.glob("*.png")):
            relative_figure = figure_path.relative_to(root).as_posix()
            producer = "scripts/analysis/" f"plot_{figure_path.stem}.py"

            if not (root / producer).is_file():
                raise FileNotFoundError(
                    f"No unique producer for {relative_figure}: " f"expected {producer}."
                )

            script_roles.setdefault(
                producer,
                set(),
            ).add("figure_producer")

            figures.append(
                {
                    **_file_record(
                        root,
                        relative_figure,
                    ),
                    "classification": classification,
                    "producer_script": producer,
                }
            )

    if len(figures) != 22:
        raise RuntimeError("Expected 22 canonical PNG figures, " f"found {len(figures)}.")

    return figures, script_roles


def _script_records(
    root: Path,
    script_roles: dict[str, set[str]],
) -> list[dict[str, Any]]:
    return [
        {
            **_file_record(root, path),
            "roles": sorted(roles),
        }
        for path, roles in sorted(script_roles.items())
    ]


def _serialization_pairs(
    indexed_paths: list[str],
) -> list[str]:
    suffixes_by_stem: dict[str, set[str]] = {}

    for path_value in indexed_paths:
        path = Path(path_value)
        suffixes_by_stem.setdefault(
            path.stem,
            set(),
        ).add(path.suffix)

    pairs = sorted(
        stem for stem, suffixes in suffixes_by_stem.items() if {".csv", ".json"}.issubset(suffixes)
    )

    if len(pairs) != 12:
        raise RuntimeError("Expected 12 intentional CSV/JSON pairs, " f"found {len(pairs)}.")

    return pairs


def build_evidence_index(
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build the deterministic final evidence index."""
    analysis_groups, indexed_paths = _analysis_artifacts(root)
    figures, script_roles = _figure_artifacts(root)

    manifest_columns, manifest_rows = _read_csv(
        root,
        ("results/analysis/" "balanced_10seed_matrix_manifest.csv"),
    )
    flat_columns, flat_rows = _read_csv(
        root,
        ("results/analysis/" "balanced_10seed_matrix_summary_flat.csv"),
    )

    if len(manifest_rows) != 180:
        raise RuntimeError("Expected 180 canonical manifest rows.")

    if len(flat_rows) != 180:
        raise RuntimeError("Expected 180 canonical flat-summary rows.")

    manifest_keys = {
        (
            row["budget"],
            row["model"],
            int(row["seed"]),
        )
        for row in manifest_rows
    }

    if len(manifest_keys) != 180:
        raise RuntimeError("Canonical manifest keys are not unique.")

    observed_models = tuple(sorted({row["model"] for row in manifest_rows}))
    observed_budgets = tuple(sorted({row["budget"] for row in manifest_rows}))
    observed_seeds = tuple(sorted({int(row["seed"]) for row in manifest_rows}))

    if observed_models != tuple(sorted(EXPECTED_MODELS)):
        raise RuntimeError(f"Unexpected canonical models: {observed_models}.")

    if observed_budgets != tuple(sorted(EXPECTED_BUDGETS)):
        raise RuntimeError(f"Unexpected canonical budgets: {observed_budgets}.")

    if observed_seeds != tuple(sorted(EXPECTED_SEEDS)):
        raise RuntimeError(f"Unexpected canonical seeds: {observed_seeds}.")

    progress_records = _read_progress_records(root)
    status_counts = Counter(str(record.get("status")) for record in progress_records)
    config_counts = Counter(str(record["experiment_config"]) for record in progress_records)

    if len(progress_records) != 252:
        raise RuntimeError("Expected 252 queue progress records.")

    if status_counts != Counter(
        {
            "started": 126,
            "completed": 126,
        }
    ):
        raise RuntimeError(f"Unexpected queue statuses: {status_counts}.")

    if len(config_counts) != 126 or set(config_counts.values()) != {2}:
        raise RuntimeError(
            "Queue progress does not contain exactly two " "events for each of 126 queued configs."
        )

    serialization_pairs = _serialization_pairs(indexed_paths)

    return {
        "artifact_type": ("balanced_10seed_matrix_evidence_index"),
        "schema_version": 1,
        "scope": {
            "description": (
                "Canonical balanced efficiency study: "
                "6 models x 3 token budgets x "
                "10 aligned seeds."
            ),
            "model_count": len(EXPECTED_MODELS),
            "models": list(EXPECTED_MODELS),
            "budget_count": len(EXPECTED_BUDGETS),
            "budgets": list(EXPECTED_BUDGETS),
            "seed_count": len(EXPECTED_SEEDS),
            "seeds": list(EXPECTED_SEEDS),
            "expected_matrix_rows": 180,
        },
        "matrix_contract": {
            "manifest_path": ("results/analysis/" "balanced_10seed_matrix_manifest.csv"),
            "manifest_row_count": len(manifest_rows),
            "manifest_column_count": len(manifest_columns),
            "unique_matrix_key_count": len(manifest_keys),
            "canonical_downstream_input": (
                "results/analysis/" "balanced_10seed_matrix_summary_flat.csv"
            ),
            "flat_summary_row_count": len(flat_rows),
            "flat_summary_column_count": len(flat_columns),
        },
        "analysis_artifact_count": len(indexed_paths),
        "analysis_artifact_groups": analysis_groups,
        "intentional_csv_json_pair_count": len(serialization_pairs),
        "intentional_csv_json_pairs": (serialization_pairs),
        "queue_progress_contract": {
            "record_count": len(progress_records),
            "unique_experiment_config_count": len(config_counts),
            "status_counts": dict(sorted(status_counts.items())),
            "experiment_config_representation": ("plain repository-relative string"),
        },
        "figure_count": len(figures),
        "figures": figures,
        "supporting_script_count": len(script_roles),
        "supporting_scripts": _script_records(
            root,
            script_roles,
        ),
        "excluded_artifacts": [
            {
                "path": ("results/analysis/" "balanced_10seed_matrix_inventory.json"),
                "reason": (
                    "Removed stale whole-repository inventory "
                    "that mixed canonical and historical scope."
                ),
            }
        ],
        "determinism": {
            "contains_generation_timestamp": False,
            "hash_algorithm": "SHA-256",
            "index_includes_own_hash": False,
        },
    }


def render_evidence_index(
    payload: dict[str, Any],
) -> str:
    """Render the canonical deterministic JSON representation."""
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Build or verify the canonical balanced " "10-seed evidence index.")
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=("Verify the tracked index instead of " "rewriting it."),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    root = ROOT if args.root is None else args.root
    payload = build_evidence_index(root)
    rendered = render_evidence_index(payload)
    index_path = root / INDEX_PATH

    if args.check:
        if not index_path.is_file():
            print(f"Missing evidence index: {INDEX_PATH}")
            return 1

        current = index_path.read_text(encoding="utf-8")

        if current != rendered:
            print("Evidence index does not match the " "current canonical evidence tree.")
            return 1

        print(
            json.dumps(
                {
                    "analysis_artifact_count": payload["analysis_artifact_count"],
                    "figure_count": payload["figure_count"],
                    "matrix_rows": payload["matrix_contract"]["manifest_row_count"],
                    "queue_progress_records": payload["queue_progress_contract"]["record_count"],
                    "verification_passed": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    index_path.write_text(
        rendered,
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            {
                "analysis_artifact_count": payload["analysis_artifact_count"],
                "figure_count": payload["figure_count"],
                "index_path": INDEX_PATH.as_posix(),
                "matrix_rows": payload["matrix_contract"]["manifest_row_count"],
                "queue_progress_records": payload["queue_progress_contract"]["record_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
