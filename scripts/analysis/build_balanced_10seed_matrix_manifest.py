"""Build the fixed 6-model, 3-budget, 10-seed experiment matrix."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepseek_reimpl.utils.artifacts import atomic_write_json, atomic_write_text  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    slot: str
    short_name: str
    model_config: str


@dataclass(frozen=True)
class BudgetSpec:
    label: str
    max_tokens: int
    base_train_config: str


SEEDS = [1337, 2027, 31415, 4441, 5501, 6173, 8191, 10007, 11213, 12721]

MODELS = [
    ModelSpec("00", "dense_121m", "configs/model/dense_121m.yaml"),
    ModelSpec("01", "mla_121m", "configs/model/mla_121m.yaml"),
    ModelSpec("02", "mtp_121m", "configs/model/mtp_121m.yaml"),
    ModelSpec("03", "moe_220m", "configs/model/moe_220m.yaml"),
    ModelSpec("04", "mla_moe_220m", "configs/model/mla_moe_220m.yaml"),
    ModelSpec("05", "v3_routing_220m", "configs/model/v3_routing_220m.yaml"),
]

BUDGETS = [
    BudgetSpec("10m", 10_000_000, "configs/train/main_large_10m.yaml"),
    BudgetSpec("25m", 25_000_000, "configs/train/main_large_25m.yaml"),
    BudgetSpec("50m", 50_000_000, "configs/train/main_large_50m.yaml"),
]

# Preserve established filenames for the pre-existing 50M configurations.
ESTABLISHED_50M_CANONICAL_SLOTS = {
    "dense_121m": "00",
    "mtp_121m": "01",
    "moe_220m": "02",
    "v3_routing_220m": "03",
}

DATA_CONFIG = "configs/data/fineweb_edu_10bt.yaml"
TOKENIZER_CONFIG = "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml"
OUT_DIR = Path("results/analysis")
MANIFEST_CSV = OUT_DIR / "balanced_10seed_matrix_manifest.csv"
MANIFEST_JSON = OUT_DIR / "balanced_10seed_matrix_manifest.json"
QUEUE_TXT = OUT_DIR / "balanced_10seed_matrix_queue.txt"
SUMMARY_JSON = OUT_DIR / "balanced_10seed_matrix_generation_summary.json"
RUN_SCRIPT = Path("scripts/run_balanced_10seed_matrix_queue.ps1")


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def require_file(path: str | Path) -> None:
    file_path = _absolute(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing required file: {file_path}")


def require_dir(path: str | Path) -> None:
    directory = _absolute(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing required directory: {directory}")


def train_config_path(budget: BudgetSpec, seed: int) -> Path:
    if seed == 1337:
        return Path(budget.base_train_config)
    return Path(f"configs/train/main_large_{budget.label}_seed{seed}.yaml")


def expected_experiment_name(budget: BudgetSpec, seed: int, model: ModelSpec) -> str:
    if seed == 1337:
        return f"main_large_{budget.label}_{model.slot}_{model.short_name}"
    if budget.label == "50m" and model.short_name in ESTABLISHED_50M_CANONICAL_SLOTS:
        slot = ESTABLISHED_50M_CANONICAL_SLOTS[model.short_name]
        return f"main_large_50m_seed{seed}_{slot}_{model.short_name}"
    return f"main_large_{budget.label}_seed{seed}_{model.slot}_{model.short_name}"


def experiment_config_path(name: str) -> Path:
    return Path("configs/experiment") / f"{name}.yaml"


def summary_path(name: str) -> Path:
    return Path("results/runs") / name / "metrics" / "summary.json"


def write_if_missing_or_identical(path: Path, text: str, *, refresh: bool = False) -> bool:
    resolved = _absolute(path)
    normalized = text.rstrip() + "\n"
    if resolved.exists():
        current = resolved.read_text(encoding="utf-8")
        if current.strip() != text.strip():
            if refresh:
                atomic_write_text(resolved, normalized)
                return True
            raise RuntimeError(f"Existing file differs from generated content: {path}")
        return False
    atomic_write_text(resolved, normalized)
    return True


def make_train_config(budget: BudgetSpec, seed: int, *, refresh: bool) -> bool:
    target = train_config_path(budget, seed)
    base = _absolute(budget.base_train_config)
    require_file(base)
    text = base.read_text(encoding="utf-8")
    if "  seed: 1337\n" not in text:
        raise RuntimeError(f"Could not locate seed line in {base}")
    if f"  max_tokens: {budget.max_tokens}\n" not in text:
        raise RuntimeError(f"Unexpected max_tokens in {base}")
    text = text.replace("  seed: 1337\n", f"  seed: {seed}\n", 1)
    return write_if_missing_or_identical(target, text, refresh=refresh)


def experiment_config_text(
    name: str,
    model: ModelSpec,
    train_config: Path,
    *,
    budget: BudgetSpec,
    seed: int,
) -> str:
    return "\n".join(
        [
            "experiment:",
            f"  name: {name}",
            "  protocol_id: primary_matrix_2026",
            f"  variant: {model.short_name}",
            f"  budget: {budget.label}",
            f"  seed: {seed}",
            f"  requested_train_tokens: {budget.max_tokens}",
            f"  output_dir: results/runs/{name}/logs",
            f"  metrics_dir: results/runs/{name}/metrics",
            f"  checkpoint_dir: results/runs/{name}/checkpoints",
            f"  model_config: {model.model_config}",
            f"  data_config: {DATA_CONFIG}",
            f"  tokenizer_config: {TOKENIZER_CONFIG}",
            f"  train_config: {train_config.as_posix()}",
            "",
        ]
    )


def make_experiment_config(
    name: str,
    model: ModelSpec,
    train_config: Path,
    *,
    budget: BudgetSpec,
    seed: int,
    refresh: bool,
) -> bool:
    text = experiment_config_text(name, model, train_config, budget=budget, seed=seed)
    return write_if_missing_or_identical(experiment_config_path(name), text, refresh=refresh)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write empty manifest.")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(_absolute(path), buffer.getvalue())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-generated-configs",
        action="store_true",
        help="Rewrite generated children from the canonical base configurations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in ("configs/model", "configs/train", "configs/experiment"):
        require_dir(directory)
    for path in (DATA_CONFIG, TOKENIZER_CONFIG, RUN_SCRIPT):
        require_file(path)
    for model in MODELS:
        require_file(model.model_config)
    for budget in BUDGETS:
        require_file(budget.base_train_config)

    rows: list[dict[str, Any]] = []
    queue: list[str] = []
    created_train_configs: list[str] = []
    created_experiment_configs: list[str] = []
    seen_names: set[str] = set()
    seen_configs: set[str] = set()

    for budget_index, budget in enumerate(BUDGETS):
        for seed_index, seed in enumerate(SEEDS):
            train_path = train_config_path(budget, seed)
            if make_train_config(budget, seed, refresh=args.refresh_generated_configs):
                created_train_configs.append(train_path.as_posix())

            order_offset = (budget_index * len(SEEDS) + seed_index) % len(MODELS)
            ordered_models = MODELS[order_offset:] + MODELS[:order_offset]
            for order_position, model in enumerate(ordered_models, start=1):
                name = expected_experiment_name(budget, seed, model)
                config_path = experiment_config_path(name)
                summary = summary_path(name)
                if name in seen_names or config_path.as_posix() in seen_configs:
                    raise RuntimeError(f"Duplicate experiment identity: {name}")
                seen_names.add(name)
                seen_configs.add(config_path.as_posix())

                if make_experiment_config(
                    name,
                    model,
                    train_path,
                    budget=budget,
                    seed=seed,
                    refresh=args.refresh_generated_configs,
                ):
                    created_experiment_configs.append(config_path.as_posix())

                is_complete = _absolute(summary).is_file()
                rows.append(
                    {
                        "budget": budget.label,
                        "max_tokens": budget.max_tokens,
                        "seed": seed,
                        "model": model.short_name,
                        "slot": model.slot,
                        "planned_order_position": order_position,
                        "experiment_name": name,
                        "experiment_config": config_path.as_posix(),
                        "train_config": train_path.as_posix(),
                        "model_config": model.model_config,
                        "summary_path": summary.as_posix(),
                        "status": (
                            "complete_existing_summary" if is_complete else "queued_missing_summary"
                        ),
                    }
                )
                if not is_complete:
                    queue.append(config_path.as_posix())

    completed = [row for row in rows if row["status"] == "complete_existing_summary"]
    missing = [row for row in rows if row["status"] == "queued_missing_summary"]
    if len(rows) != 180 or len(completed) + len(missing) != 180 or len(queue) != len(missing):
        raise RuntimeError("Generated matrix does not exactly cover the 180-cell design")

    write_csv(MANIFEST_CSV, rows)
    atomic_write_json(_absolute(MANIFEST_JSON), rows)
    queue_text = "\n".join(queue)
    atomic_write_text(_absolute(QUEUE_TXT), queue_text + ("\n" if queue_text else ""))

    counts: dict[str, Any] = {
        "target_cells": len(rows),
        "completed_existing_summary": len(completed),
        "queued_missing_summary": len(missing),
        "queue_entries": len(queue),
        "created_train_config_count": len(created_train_configs),
        "created_experiment_config_count": len(created_experiment_configs),
        "created_run_script": False,
        "seeds": SEEDS,
        "models": [model.short_name for model in MODELS],
        "budgets": [budget.label for budget in BUDGETS],
        "created_train_configs": created_train_configs,
        "created_experiment_configs": created_experiment_configs,
        "manifest_csv": MANIFEST_CSV.as_posix(),
        "manifest_json": MANIFEST_JSON.as_posix(),
        "queue_txt": QUEUE_TXT.as_posix(),
        "run_script": RUN_SCRIPT.as_posix(),
    }
    atomic_write_json(_absolute(SUMMARY_JSON), counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
