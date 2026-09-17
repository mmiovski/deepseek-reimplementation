"""Fail-fast integrity and identity checks for the complete primary matrix."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from deepseek_reimpl.train.pretrain import _code_fingerprint, _runtime_metadata  # noqa: E402
from deepseek_reimpl.train.train_utils import configure_determinism  # noqa: E402
from deepseek_reimpl.utils.artifacts import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from deepseek_reimpl.utils.config import load_yaml_config  # noqa: E402
from scripts.analysis.verify_final_data_tokenizer_provenance import (  # noqa: E402
    verify_provenance,
)

MANIFEST = Path("results/analysis/balanced_10seed_matrix_manifest.json")
OUTPUT = Path("results/analysis/primary_matrix_preflight.json")
PROTOCOL_ID = "primary_matrix_2026"
EXPECTED_MODELS = {
    "dense_121m",
    "mla_121m",
    "mtp_121m",
    "moe_220m",
    "mla_moe_220m",
    "v3_routing_220m",
}
EXPECTED_BUDGETS = {"10m": 10_000_000, "25m": 25_000_000, "50m": 50_000_000}
EXPECTED_SEEDS = {1337, 2027, 31415, 4441, 5501, 6173, 8191, 10007, 11213, 12721}


def _fixed_starts(*, num_tokens: int, block_size: int = 256, samples: int = 400) -> list[int]:
    source_length = num_tokens - block_size
    slots = ((source_length - 1) // block_size) + 1
    if samples > slots:
        raise ValueError("Held-out split cannot supply 400 non-overlapping windows")
    last_slot = slots - 1
    return [index * last_slot // (samples - 1) * block_size for index in range(samples)]


def _validate_summary(
    summary_path: Path,
    *,
    row: dict[str, Any],
    experiment: dict[str, Any],
    current_code: dict[str, Any],
    current_environment: dict[str, Any],
    current_artifacts: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Unreadable completed summary {summary_path}: {exc}"]
    expected_fields = {
        "schema_version": 3,
        "protocol_id": PROTOCOL_ID,
        "run_completed": True,
        "experiment_name": row["experiment_name"],
        "variant": row["model"],
        "budget": row["budget"],
        "seed": row["seed"],
        "max_tokens": row["max_tokens"],
    }
    for field, expected in expected_fields.items():
        if summary.get(field) != expected:
            errors.append(f"{summary_path}: field {field} does not match manifest")
    batch_size = int(summary.get("batch_size", 0))
    block_size = int(summary.get("block_size", 0))
    if batch_size <= 0 or block_size <= 0:
        errors.append(f"{summary_path}: invalid batch/block size")
    else:
        expected_steps = math.ceil(int(row["max_tokens"]) / (batch_size * block_size))
        expected_train_tokens = expected_steps * batch_size * block_size
        if (
            summary.get("steps") != expected_steps
            or summary.get("train_tokens") != expected_train_tokens
        ):
            errors.append(f"{summary_path}: completed step/token accounting mismatch")
    run_identity = summary.get("run_identity")
    if not isinstance(run_identity, dict) or sha256_json(run_identity) != summary.get(
        "run_identity_sha256"
    ):
        errors.append(f"{summary_path}: invalid self-authenticating run identity")
    if summary.get("code_fingerprint") != current_code:
        errors.append(f"{summary_path}: code fingerprint differs from current source")
    if summary.get("runtime", {}).get("environment_sha256") != current_environment.get(
        "environment_sha256"
    ):
        errors.append(f"{summary_path}: environment fingerprint mismatch")
    recorded_artifacts = summary.get("artifact_fingerprint", {})
    for field, expected_hash in current_artifacts.items():
        if recorded_artifacts.get(field) != expected_hash:
            errors.append(f"{summary_path}: artifact fingerprint mismatch for {field}")
    config_paths = {
        "experiment": Path(row["experiment_config"]),
        "model": Path(row["model_config"]),
        "data": Path(experiment["data_config"]),
        "tokenizer": Path(experiment["tokenizer_config"]),
        "train": Path(row["train_config"]),
    }
    recorded_hashes = summary.get("config_sha256", {})
    for name, relative in config_paths.items():
        if recorded_hashes.get(name) != sha256_file(PROJECT_ROOT / relative):
            errors.append(f"{summary_path}: {name} config hash mismatch")
    log_path = PROJECT_ROOT / experiment["output_dir"] / "train_log.jsonl"
    if not log_path.is_file() or summary.get("train_log_sha256") != sha256_file(log_path):
        errors.append(f"{summary_path}: training-log hash mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []

    rows = json.loads((PROJECT_ROOT / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 180:
        errors.append("Manifest must contain exactly 180 rows")
        rows = []
    identities = {(row["model"], row["budget"], int(row["seed"])) for row in rows}
    expected_identities = {
        (model, budget, seed)
        for model in EXPECTED_MODELS
        for budget in EXPECTED_BUDGETS
        for seed in EXPECTED_SEEDS
    }
    if identities != expected_identities:
        errors.append("Manifest identities do not exactly cover the 6 x 3 x 10 design")

    position_counts = Counter((row["model"], int(row["planned_order_position"])) for row in rows)
    if any(
        position_counts[(model, position)] != 5
        for model in EXPECTED_MODELS
        for position in range(1, 7)
    ):
        errors.append("Queue order is not exactly counterbalanced across model positions")

    completed = 0
    current_code = _code_fingerprint()
    configure_determinism(enabled=True)
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable in the primary runtime")
        current_environment: dict[str, Any] = {}
    else:
        current_environment = _runtime_metadata(torch.device("cuda"))
    current_artifacts = {
        "tokenizer_sha256": sha256_file(
            PROJECT_ROOT / "tokenizers/fineweb_edu_10bt/tokenizer.json"
        ),
        "tokenized_metadata_sha256": sha256_file(
            PROJECT_ROOT / "data/tokens/fineweb_edu_10bt/metadata.json"
        ),
    }
    for row in rows:
        experiment_path = PROJECT_ROOT / row["experiment_config"]
        experiment = load_yaml_config(experiment_path)["experiment"]
        train = load_yaml_config(PROJECT_ROOT / row["train_config"])["train"]
        model = load_yaml_config(PROJECT_ROOT / row["model_config"])["model"]
        expected_experiment = {
            "name": row["experiment_name"],
            "protocol_id": PROTOCOL_ID,
            "variant": row["model"],
            "budget": row["budget"],
            "seed": row["seed"],
            "requested_train_tokens": row["max_tokens"],
        }
        for field, expected in expected_experiment.items():
            if experiment.get(field) != expected:
                errors.append(f"{experiment_path}: identity field {field} mismatch")
        if (
            train.get("seed") != row["seed"]
            or train.get("max_tokens") != row["max_tokens"]
            or train.get("batch_size") != 4
            or train.get("block_size") != 256
            or train.get("eval_batches") != 100
            or train.get("num_workers") != 0
            or train.get("precision") != "fp32"
            or train.get("deterministic") is not True
            or not isinstance(train.get("checkpoint_interval"), int)
        ):
            errors.append(f"{row['train_config']}: training protocol mismatch")
        if model.get("vocab_size") != 10_000 or model.get("block_size") != 256:
            errors.append(f"{row['model_config']}: model/token protocol mismatch")
        summary_path = PROJECT_ROOT / row["summary_path"]
        if summary_path.exists():
            completed += 1
            errors.extend(
                _validate_summary(
                    summary_path,
                    row=row,
                    experiment=experiment,
                    current_code=current_code,
                    current_environment=current_environment,
                    current_artifacts=current_artifacts,
                )
            )

    if args.require_complete and completed != len(rows):
        errors.append(
            f"Complete-matrix validation requires {len(rows)} summaries; found {completed}"
        )

    provenance = verify_provenance(root=PROJECT_ROOT, require_local_artifacts=True)
    if not provenance["passed"]:
        errors.append("Data/tokenizer provenance verification failed")
    token_metadata_path = PROJECT_ROOT / "data/tokens/fineweb_edu_10bt/metadata.json"
    token_metadata = json.loads(token_metadata_path.read_text(encoding="utf-8"))
    evaluation_windows = {}
    for split in ("validation", "test"):
        starts = _fixed_starts(num_tokens=int(token_metadata["splits"][split]["num_tokens"]))
        evaluation_windows[split] = {
            "samples": len(starts),
            "tokens": len(starts) * 256,
            "start_indices_sha256": sha256_json(starts),
        }

    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source_prefixes = ("deepseek_reimpl/", "scripts/", "configs/")
    source_names = {
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-cuda.txt",
    }
    source_dirty = any(
        (
            (path_value := line[3:].split(" -> ")[-1].replace("\\", "/")).startswith(
                source_prefixes
            )
            or path_value in source_names
        )
        for line in git_status.splitlines()
        if len(line) >= 4
    )
    if args.require_clean_git and source_dirty:
        errors.append("Primary launch requires a clean committed Git worktree")

    payload = {
        "artifact_type": "primary_matrix_preflight",
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "manifest": MANIFEST.as_posix(),
        "manifest_sha256": sha256_file(PROJECT_ROOT / MANIFEST),
        "target_runs": len(rows),
        "validated_completed_runs": completed,
        "pending_runs": len(rows) - completed,
        "queue_order_position_counts": {
            model: [position_counts[(model, position)] for position in range(1, 7)]
            for model in sorted(EXPECTED_MODELS)
        },
        "evaluation_windows": evaluation_windows,
        "data_tokenizer_provenance_passed": provenance["passed"],
        "source_git_dirty": source_dirty,
        "require_clean_git": args.require_clean_git,
        "require_complete": args.require_complete,
    }
    atomic_write_json(PROJECT_ROOT / OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
