"""Verify the final corpus and tokenizer provenance contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROVENANCE_PATH = Path("results/analysis/final_data_tokenizer_provenance.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")

    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")

    return payload


def _resolve(root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return root / path


def verify_provenance(
    provenance_path: str | Path = DEFAULT_PROVENANCE_PATH,
    *,
    root: str | Path | None = None,
    require_local_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate configs and any locally present retained artifacts."""
    root_path = Path.cwd() if root is None else Path(root)
    resolved_provenance = _resolve(root_path, provenance_path)
    provenance = _load_json(resolved_provenance)

    config_errors: list[str] = []
    missing_artifacts: list[str] = []
    artifact_mismatches: list[dict[str, Any]] = []
    verified_artifacts: list[str] = []

    if provenance.get("artifact_type") != ("final_data_tokenizer_provenance"):
        config_errors.append("Unexpected artifact_type.")

    if provenance.get("schema_version") != 1:
        config_errors.append("Unexpected schema_version.")

    data_config_path = str(provenance.get("data_config", ""))
    tokenizer_config_path = str(provenance.get("tokenizer_config", ""))

    if not data_config_path:
        config_errors.append("Missing data_config path.")

    if not tokenizer_config_path:
        config_errors.append("Missing tokenizer_config path.")

    data_config = _load_yaml(_resolve(root_path, data_config_path))
    tokenizer_config = _load_yaml(_resolve(root_path, tokenizer_config_path))

    expected_dataset = provenance.get("dataset")
    observed_dataset = data_config.get("dataset")

    if not isinstance(expected_dataset, dict):
        config_errors.append("Provenance dataset must be an object.")
    elif not isinstance(observed_dataset, dict):
        config_errors.append("Data config dataset must be a mapping.")
    else:
        dataset_fields = (
            "source",
            "hf_dataset_name",
            "hf_dataset_config_name",
            "text_field",
        )

        for field in dataset_fields:
            if observed_dataset.get(field) != expected_dataset.get(field):
                config_errors.append(f"Dataset field mismatch: {field}.")

    expected_streaming = provenance.get("streaming")
    observed_streaming = data_config.get("streaming")

    if expected_streaming != observed_streaming:
        config_errors.append("Streaming configuration does not match provenance.")

    if provenance.get("preprocessing") != data_config.get("preprocessing"):
        config_errors.append("Preprocessing configuration does not match provenance.")

    expected_tokenizer = provenance.get("tokenizer")
    observed_tokenizer = tokenizer_config.get("tokenizer")

    if not isinstance(expected_tokenizer, dict):
        config_errors.append("Provenance tokenizer must be an object.")
    elif not isinstance(observed_tokenizer, dict):
        config_errors.append("Tokenizer config tokenizer must be a mapping.")
    else:
        tokenizer_fields = (
            ("name", "name"),
            ("type", "type"),
            ("configured_vocab_size", "vocab_size"),
            ("min_frequency", "min_frequency"),
        )

        for provenance_field, config_field in tokenizer_fields:
            if expected_tokenizer.get(provenance_field) != observed_tokenizer.get(config_field):
                config_errors.append("Tokenizer field mismatch: " f"{config_field}.")

    if provenance.get("tokenizer", {}).get("special_tokens") != tokenizer_config.get(
        "special_tokens"
    ):
        config_errors.append("Tokenizer special tokens do not match provenance.")

    expected_training = tokenizer_config.get("training", {})

    if expected_training.get("max_training_chars") != provenance.get("tokenizer", {}).get(
        "max_training_chars"
    ):
        config_errors.append("Tokenizer max_training_chars does not match.")

    configured_artifact_paths = set()

    for value in data_config.get("artifacts", {}).values():
        if isinstance(value, str):
            configured_artifact_paths.add(Path(value).as_posix())

    for value in tokenizer_config.get(
        "artifacts",
        {},
    ).values():
        if isinstance(value, str):
            configured_artifact_paths.add(Path(value).as_posix())

    artifact_contract = provenance.get("artifacts")

    if not isinstance(artifact_contract, dict):
        config_errors.append("Provenance artifacts must be an object.")
        artifact_contract = {}

    provenance_artifact_paths = {Path(path).as_posix() for path in artifact_contract}

    if configured_artifact_paths != provenance_artifact_paths:
        config_errors.append(
            "Configured artifact paths do not exactly match " "the provenance artifact set."
        )

    for path_value, expected in sorted(artifact_contract.items()):
        if not isinstance(expected, dict):
            config_errors.append(f"Artifact record is not an object: {path_value}.")
            continue

        expected_size = expected.get("size_bytes")
        expected_hash = expected.get("sha256")

        if not isinstance(expected_size, int):
            config_errors.append(f"Invalid artifact size contract: {path_value}.")
            continue

        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            config_errors.append(f"Invalid artifact hash contract: {path_value}.")
            continue

        artifact_path = _resolve(root_path, path_value)

        if not artifact_path.is_file():
            missing_artifacts.append(path_value)
            continue

        observed_size = artifact_path.stat().st_size
        observed_hash = _sha256(artifact_path)

        if observed_size != expected_size or observed_hash != expected_hash:
            artifact_mismatches.append(
                {
                    "path": path_value,
                    "expected_size_bytes": expected_size,
                    "observed_size_bytes": observed_size,
                    "expected_sha256": expected_hash,
                    "observed_sha256": observed_hash,
                }
            )
            continue

        verified_artifacts.append(path_value)

    passed = (
        not config_errors
        and not artifact_mismatches
        and (not require_local_artifacts or not missing_artifacts)
    )

    return {
        "artifact_type": ("final_data_tokenizer_provenance_verification"),
        "provenance_path": Path(provenance_path).as_posix(),
        "config_validation_passed": not config_errors,
        "config_errors": config_errors,
        "artifact_contract_count": len(artifact_contract),
        "verified_artifact_count": len(verified_artifacts),
        "verified_artifacts": verified_artifacts,
        "missing_artifact_count": len(missing_artifacts),
        "missing_artifacts": missing_artifacts,
        "artifact_mismatch_count": len(artifact_mismatches),
        "artifact_mismatches": artifact_mismatches,
        "require_local_artifacts": require_local_artifacts,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=("Verify final corpus and tokenizer provenance."))
    parser.add_argument(
        "--provenance",
        type=Path,
        default=DEFAULT_PROVENANCE_PATH,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--require-local-artifacts",
        action="store_true",
    )
    args = parser.parse_args()

    summary = verify_provenance(
        args.provenance,
        root=args.root,
        require_local_artifacts=(args.require_local_artifacts),
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
