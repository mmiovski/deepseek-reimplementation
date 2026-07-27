from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from scripts.analysis.build_balanced_10seed_evidence_index import (
    INDEX_PATH,
    build_evidence_index,
)

ROOT = Path(__file__).resolve().parents[1]

STALE_INVENTORY_PATH = Path("results/analysis/" "balanced_10seed_matrix_inventory.json")

PROGRESS_PATH = Path("results/analysis/" "balanced_10seed_matrix_queue_progress.jsonl")

MANIFEST_PATH = Path("results/analysis/" "balanced_10seed_matrix_manifest.csv")


def _load_index() -> dict[str, object]:
    payload = json.loads((ROOT / INDEX_PATH).read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
    return payload


def _load_progress() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    with (ROOT / PROGRESS_PATH).open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line in file:
            stripped = line.strip()

            if not stripped:
                continue

            payload = json.loads(stripped)
            assert isinstance(payload, dict)
            records.append(payload)

    return records


def test_evidence_index_is_deterministic() -> None:
    assert _load_index() == build_evidence_index(ROOT)


def test_final_balanced_matrix_contract() -> None:
    payload = _load_index()
    scope = payload["scope"]
    matrix = payload["matrix_contract"]

    assert isinstance(scope, dict)
    assert isinstance(matrix, dict)

    assert scope["model_count"] == 6
    assert scope["budget_count"] == 3
    assert scope["seed_count"] == 10
    assert scope["expected_matrix_rows"] == 180

    assert matrix["manifest_row_count"] == 180
    assert matrix["unique_matrix_key_count"] == 180
    assert matrix["flat_summary_row_count"] == 180


def test_all_figures_have_one_named_producer() -> None:
    payload = _load_index()
    figures = payload["figures"]

    assert isinstance(figures, list)
    assert len(figures) == 22

    diagnostic_count = 0
    report_count = 0

    for figure in figures:
        assert isinstance(figure, dict)

        figure_path = Path(str(figure["path"]))
        producer_path = Path(str(figure["producer_script"]))

        assert (ROOT / figure_path).is_file()
        assert (ROOT / producer_path).is_file()
        assert producer_path.stem == f"plot_{figure_path.stem}"

        if figure["classification"] == "diagnostic":
            diagnostic_count += 1
        elif figure["classification"] == "report_ready":
            report_count += 1
        else:
            raise AssertionError("Unexpected figure classification.")

    assert diagnostic_count == 14
    assert report_count == 8


def test_progress_history_is_sanitized_and_complete() -> None:
    records = _load_progress()

    assert len(records) == 252

    status_counts = Counter(str(record["status"]) for record in records)
    config_counts = Counter(str(record["experiment_config"]) for record in records)

    assert status_counts == Counter(
        {
            "started": 126,
            "completed": 126,
        }
    )
    assert len(config_counts) == 126
    assert set(config_counts.values()) == {2}

    for record in records:
        experiment_config = record["experiment_config"]

        assert isinstance(experiment_config, str)
        assert experiment_config.startswith("configs/experiment/")
        assert "\\" not in experiment_config

    progress_text = (ROOT / PROGRESS_PATH).read_text(encoding="utf-8")

    for forbidden_term in (
        "PSDrive",
        "PSPath",
        "PSProvider",
        "PSParentPath",
        "ReadCount",
        "Microsoft.PowerShell.Core",
        "C:\\Users\\",
    ):
        assert forbidden_term not in progress_text


def test_stale_inventory_is_removed() -> None:
    assert not (ROOT / STALE_INVENTORY_PATH).exists()


def test_legacy_50m_naming_matches_manifest() -> None:
    from scripts.analysis.build_balanced_10seed_matrix_manifest import (
        BUDGETS,
        MODELS,
        expected_experiment_name,
    )

    manifest_rows: list[dict[str, str]]

    with (ROOT / MANIFEST_PATH).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        manifest_rows = list(csv.DictReader(file))

    budget = next(item for item in BUDGETS if item.label == "50m")

    expected_slots = {
        "dense_121m": "00",
        "mtp_121m": "01",
        "moe_220m": "02",
        "v3_routing_220m": "03",
    }

    for model_name, slot in expected_slots.items():
        model = next(item for item in MODELS if item.short_name == model_name)
        name = expected_experiment_name(
            budget,
            2027,
            model,
        )

        assert name == (f"main_large_50m_seed2027_" f"{slot}_{model_name}")

        matching_rows = [
            row
            for row in manifest_rows
            if row["budget"] == "50m" and row["seed"] == "2027" and row["model"] == model_name
        ]

        assert len(matching_rows) == 1
        assert matching_rows[0]["experiment_name"] == name

    source = (ROOT / "scripts/analysis/" "build_balanced_10seed_matrix_manifest.py").read_text(
        encoding="utf-8"
    )

    assert "25-seed" not in source
    assert "FOUR_MODEL" not in source
    assert "LEGACY_50M_CANONICAL_SLOTS" in source


def test_index_covers_intended_evidence_contract() -> None:
    payload = _load_index()

    assert payload["analysis_artifact_count"] == 38
    assert payload["figure_count"] == 22
    assert payload["intentional_csv_json_pair_count"] == 12

    groups = payload["analysis_artifact_groups"]
    assert isinstance(groups, list)

    indexed_paths = {
        str(file_record["path"])
        for group in groups
        if isinstance(group, dict)
        for file_record in group["files"]
        if isinstance(file_record, dict)
    }

    assert "results/analysis/" "balanced_10seed_matrix_inventory.json" not in indexed_paths
    assert "results/analysis/" "balanced_10seed_matrix_summary_flat.csv" in indexed_paths
    assert "results/analysis/" "balanced_10seed_matrix_completion_audit.json" in indexed_paths
    assert "results/analysis/" "final_data_tokenizer_provenance.json" in indexed_paths
