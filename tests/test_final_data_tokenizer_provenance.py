from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.analysis.verify_final_data_tokenizer_provenance import (
    verify_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = Path("results/analysis/final_data_tokenizer_provenance.json")
CONFIG_PATHS = (
    Path("configs/data/fineweb_edu_10bt.yaml"),
    Path("configs/tokenizer/" "bpe_fineweb_edu_10bt_local_experiment.yaml"),
)


def _copy_contract_files(target_root: Path) -> None:
    for relative_path in (
        PROVENANCE_PATH,
        *CONFIG_PATHS,
    ):
        destination = target_root / relative_path
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(
            ROOT / relative_path,
            destination,
        )


def test_final_provenance_matches_tracked_configs() -> None:
    summary = verify_provenance(
        PROVENANCE_PATH,
        root=ROOT,
    )

    assert summary["config_validation_passed"] is True
    assert summary["artifact_mismatch_count"] == 0
    assert summary["passed"] is True


def test_missing_local_artifacts_are_optional_by_default(
    tmp_path: Path,
) -> None:
    _copy_contract_files(tmp_path)

    optional_summary = verify_provenance(
        PROVENANCE_PATH,
        root=tmp_path,
    )
    required_summary = verify_provenance(
        PROVENANCE_PATH,
        root=tmp_path,
        require_local_artifacts=True,
    )

    assert optional_summary["artifact_contract_count"] == 10
    assert optional_summary["verified_artifact_count"] == 0
    assert optional_summary["missing_artifact_count"] == 10
    assert optional_summary["passed"] is True

    assert required_summary["missing_artifact_count"] == 10
    assert required_summary["passed"] is False


def test_present_artifacts_are_hash_verified(
    tmp_path: Path,
) -> None:
    _copy_contract_files(tmp_path)

    provenance_file = tmp_path / PROVENANCE_PATH
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))

    for path_value in provenance["artifacts"]:
        artifact_path = tmp_path / path_value
        artifact_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        content = path_value.encode("utf-8")
        artifact_path.write_bytes(content)

        provenance["artifacts"][path_value] = {
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    provenance_file.write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    matching_summary = verify_provenance(
        PROVENANCE_PATH,
        root=tmp_path,
        require_local_artifacts=True,
    )

    assert matching_summary["verified_artifact_count"] == 10
    assert matching_summary["missing_artifact_count"] == 0
    assert matching_summary["artifact_mismatch_count"] == 0
    assert matching_summary["passed"] is True

    first_path = next(iter(provenance["artifacts"]))
    (tmp_path / first_path).write_bytes(b"modified")

    mismatching_summary = verify_provenance(
        PROVENANCE_PATH,
        root=tmp_path,
        require_local_artifacts=True,
    )

    assert mismatching_summary["artifact_mismatch_count"] == 1
    assert mismatching_summary["passed"] is False
