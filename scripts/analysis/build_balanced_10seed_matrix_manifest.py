from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


SEEDS = [
    1337,
    2027,
    31415,
    4441,
    5501,
    6173,
    8191,
    10007,
    11213,
    12721,
]

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

# Preserve established canonical filenames for the pre-existing 50M runs.
LEGACY_50M_CANONICAL_SLOTS = {
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


def require_file(path: str | Path) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")


def require_dir(path: str | Path) -> None:
    dir_path = Path(path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Missing required directory: {dir_path}")


def train_config_path(budget: BudgetSpec, seed: int) -> Path:
    if seed == 1337:
        return Path(budget.base_train_config)
    return Path(f"configs/train/main_large_{budget.label}_seed{seed}.yaml")


def expected_experiment_name(budget: BudgetSpec, seed: int, model: ModelSpec) -> str:
    if seed == 1337:
        return f"main_large_{budget.label}_{model.slot}_{model.short_name}"

    if budget.label == "50m" and model.short_name in LEGACY_50M_CANONICAL_SLOTS:
        slot = LEGACY_50M_CANONICAL_SLOTS[model.short_name]
        return f"main_large_50m_seed{seed}_{slot}_{model.short_name}"

    return f"main_large_{budget.label}_seed{seed}_{model.slot}_{model.short_name}"


def experiment_config_path(name: str) -> Path:
    return Path("configs/experiment") / f"{name}.yaml"


def summary_path(name: str) -> Path:
    return Path("results/metrics") / name / "summary.json"


def write_if_missing_or_identical(path: Path, text: str) -> bool:
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current.strip() != text.strip():
            raise RuntimeError(f"Existing file differs from generated content: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return True


def make_train_config(budget: BudgetSpec, seed: int) -> bool:
    target = train_config_path(budget, seed)
    base = Path(budget.base_train_config)
    require_file(base)

    text = base.read_text(encoding="utf-8")
    if "  seed: 1337\n" not in text:
        raise RuntimeError(f"Could not locate seed line in {base}")
    if f"  max_tokens: {budget.max_tokens}\n" not in text:
        raise RuntimeError(f"Unexpected max_tokens in {base}")

    text = text.replace("  seed: 1337\n", f"  seed: {seed}\n", 1)

    if target.exists():
        current = target.read_text(encoding="utf-8")
        if f"  seed: {seed}\n" not in current:
            raise RuntimeError(f"Existing train config has wrong seed: {target}")
        if f"  max_tokens: {budget.max_tokens}\n" not in current:
            raise RuntimeError(f"Existing train config has wrong max_tokens: {target}")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")
    return True


def experiment_config_text(name: str, model: ModelSpec, train_config: Path) -> str:
    return "\n".join(
        [
            "experiment:",
            f"  name: {name}",
            f"  output_dir: results/raw_logs/{name}",
            f"  metrics_dir: results/metrics/{name}",
            f"  checkpoint_dir: results/checkpoints/{name}",
            f"  model_config: {model.model_config}",
            f"  data_config: {DATA_CONFIG}",
            f"  tokenizer_config: {TOKENIZER_CONFIG}",
            f"  train_config: {train_config.as_posix()}",
            "",
        ]
    )


def make_experiment_config(name: str, model: ModelSpec, train_config: Path) -> bool:
    text = experiment_config_text(name, model, train_config)
    return write_if_missing_or_identical(experiment_config_path(name), text)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write empty manifest.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run_script() -> bool:
    text = r"""
param(
    [string]$QueuePath = "results\analysis\balanced_10seed_matrix_queue.txt",
    [switch]$AllowPendingReboot
)

$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing virtual environment Python at .\.venv\Scripts\python.exe"
}

if (-not (Test-Path $QueuePath)) {
    throw "Missing queue file: $QueuePath"
}

function Test-PendingReboot {
    $RebootKeys = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
    )

    foreach ($Key in $RebootKeys) {
        if (Test-Path $Key) {
            return $true
        }
    }

    $SessionManager = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager"
    try {
        $PendingFileRename = Get-ItemProperty `
            -Path $SessionManager `
            -Name PendingFileRenameOperations `
            -ErrorAction Stop
        if ($null -ne $PendingFileRename) {
            return $true
        }
    } catch {
    }

    return $false
}

function Enable-LongRunPowerGuard {
    "Enabling long-run power guard for AC power."

    powercfg -change -standby-timeout-ac 0
    powercfg -change -hibernate-timeout-ac 0
    powercfg -change -monitor-timeout-ac 0

    $Code = @"
using System;
using System.Runtime.InteropServices;

public static class SleepUtil {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

    if ($null -eq ("SleepUtil" -as [type])) {
        Add-Type -TypeDefinition $Code
    }

    $ES_CONTINUOUS = [uint32]"0x80000000"
    $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
    $ES_DISPLAY_REQUIRED = [uint32]"0x00000002"
    $ES_AWAYMODE_REQUIRED = [uint32]"0x00000040"

    $Flags = $ES_CONTINUOUS `
        -bor $ES_SYSTEM_REQUIRED `
        -bor $ES_DISPLAY_REQUIRED `
        -bor $ES_AWAYMODE_REQUIRED

    [SleepUtil]::SetThreadExecutionState($Flags) | Out-Null

    "Power guard enabled. AC sleep, hibernate, and display timeout set to never."
    "Current blocking requests:"
    powercfg /requests
}

function Refresh-LongRunPowerGuard {
    $ES_CONTINUOUS = [uint32]"0x80000000"
    $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
    $ES_DISPLAY_REQUIRED = [uint32]"0x00000002"
    $ES_AWAYMODE_REQUIRED = [uint32]"0x00000040"

    $Flags = $ES_CONTINUOUS `
        -bor $ES_SYSTEM_REQUIRED `
        -bor $ES_DISPLAY_REQUIRED `
        -bor $ES_AWAYMODE_REQUIRED

    [SleepUtil]::SetThreadExecutionState($Flags) | Out-Null
}

function Disable-LongRunPowerGuard {
    if ($null -ne ("SleepUtil" -as [type])) {
        $ES_CONTINUOUS = [uint32]"0x80000000"
        [SleepUtil]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    }
}

if ((Test-PendingReboot) -and (-not $AllowPendingReboot)) {
    throw "Pending reboot detected. Restart Windows first, then launch again."
}

Enable-LongRunPowerGuard

$ProgressPath = "results\analysis\balanced_10seed_matrix_queue_progress.jsonl"
$Queue = Get-Content $QueuePath | Where-Object { $_.Trim().Length -gt 0 }
$Total = $Queue.Count
$Index = 0

"Queue entries: $Total"
"Progress log: $ProgressPath"

try {
    foreach ($ExperimentConfig in $Queue) {
        $ExperimentConfig = [string]$ExperimentConfig
        $Index += 1
        Refresh-LongRunPowerGuard

        if ((Test-PendingReboot) -and (-not $AllowPendingReboot)) {
            throw "Pending reboot appeared before run $Index of $Total. Stopping safely."
        }

        if (-not (Test-Path $ExperimentConfig)) {
            throw "Missing experiment config: $ExperimentConfig"
        }

        $ConfigLines = Get-Content $ExperimentConfig
        $NameLine = $ConfigLines |
            Where-Object { $_ -match "^\s+name:\s+" } |
            Select-Object -First 1
        $MetricsLine = $ConfigLines |
            Where-Object { $_ -match "^\s+metrics_dir:\s+" } |
            Select-Object -First 1

        if ($null -eq $NameLine) {
            throw "Could not parse experiment name from $ExperimentConfig"
        }
        if ($null -eq $MetricsLine) {
            throw "Could not parse metrics_dir from $ExperimentConfig"
        }

        $ExperimentName = ($NameLine -replace "^\s+name:\s*", "").Trim()
        $MetricsDir = ($MetricsLine -replace "^\s+metrics_dir:\s*", "").Trim()
        $SummaryPath = Join-Path $MetricsDir "summary.json"
        $Ticker = "[$Index/$Total]"

        if (Test-Path $SummaryPath) {
            $Record = [ordered]@{
                timestamp = (Get-Date).ToString("o")
                queue_index = $Index
                queue_total = $Total
                experiment_config = $ExperimentConfig
                experiment_name = $ExperimentName
                status = "skipped_existing_summary"
                summary_path = $SummaryPath
            }
            ($Record | ConvertTo-Json -Compress) | Add-Content $ProgressPath
            "$Ticker SKIP existing summary: $ExperimentName"
            continue
        }

        $RunStart = Get-Date
        $StartRecord = [ordered]@{
            timestamp = $RunStart.ToString("o")
            queue_index = $Index
            queue_total = $Total
            experiment_config = $ExperimentConfig
            experiment_name = $ExperimentName
            status = "started"
            summary_path = $SummaryPath
        }
        ($StartRecord | ConvertTo-Json -Compress) | Add-Content $ProgressPath

        "$Ticker START: $ExperimentName"
        & $Python scripts\train\run_pretrain.py --experiment-config $ExperimentConfig
        $ExitCode = $LASTEXITCODE
        $RunEnd = Get-Date
        $DurationMinutes = [Math]::Round(
            ($RunEnd - $RunStart).TotalMinutes,
            2
        )

        if ($ExitCode -ne 0) {
            $FailRecord = [ordered]@{
                timestamp = $RunEnd.ToString("o")
                queue_index = $Index
                queue_total = $Total
                experiment_config = $ExperimentConfig
                experiment_name = $ExperimentName
                status = "failed"
                exit_code = $ExitCode
                duration_minutes = $DurationMinutes
                summary_path = $SummaryPath
            }
            ($FailRecord | ConvertTo-Json -Compress) | Add-Content $ProgressPath
            throw "$Ticker Experiment failed with exit code $ExitCode`: $ExperimentName"
        }

        if (-not (Test-Path $SummaryPath)) {
            $MissingRecord = [ordered]@{
                timestamp = $RunEnd.ToString("o")
                queue_index = $Index
                queue_total = $Total
                experiment_config = $ExperimentConfig
                experiment_name = $ExperimentName
                status = "completed_but_missing_summary"
                duration_minutes = $DurationMinutes
                summary_path = $SummaryPath
            }
            ($MissingRecord | ConvertTo-Json -Compress) | Add-Content $ProgressPath
            throw "$Ticker Completed but summary.json was not found: $ExperimentName"
        }

        $PctComplete = [Math]::Round(
            ($Index / [Math]::Max($Total, 1)) * 100,
            2
        )
        $DoneRecord = [ordered]@{
            timestamp = $RunEnd.ToString("o")
            queue_index = $Index
            queue_total = $Total
            percent_complete = $PctComplete
            experiment_config = $ExperimentConfig
            experiment_name = $ExperimentName
            status = "completed"
            duration_minutes = $DurationMinutes
            summary_path = $SummaryPath
        }
        ($DoneRecord | ConvertTo-Json -Compress) | Add-Content $ProgressPath
        "$Ticker DONE: $ExperimentName in $DurationMinutes minutes ($PctComplete%)"
    }
} finally {
    Disable-LongRunPowerGuard
}
""".strip()
    return write_if_missing_or_identical(RUN_SCRIPT, text)


def main() -> None:
    require_dir("configs/model")
    require_dir("configs/train")
    require_dir("configs/experiment")
    require_file(DATA_CONFIG)
    require_file(TOKENIZER_CONFIG)

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

    for budget in BUDGETS:
        for seed in SEEDS:
            train_path = train_config_path(budget, seed)
            created_train = make_train_config(budget, seed)
            if created_train:
                created_train_configs.append(train_path.as_posix())

            for model in MODELS:
                name = expected_experiment_name(budget, seed, model)
                config_path = experiment_config_path(name)
                summary = summary_path(name)

                if name in seen_names:
                    raise RuntimeError(f"Duplicate experiment name in target matrix: {name}")
                if config_path.as_posix() in seen_configs:
                    raise RuntimeError(f"Duplicate experiment config path: {config_path}")

                seen_names.add(name)
                seen_configs.add(config_path.as_posix())

                created_experiment = make_experiment_config(name, model, train_path)
                if created_experiment:
                    created_experiment_configs.append(config_path.as_posix())

                is_complete = summary.exists()
                status = "complete_existing_summary" if is_complete else "queued_missing_summary"

                row = {
                    "budget": budget.label,
                    "max_tokens": budget.max_tokens,
                    "seed": seed,
                    "model": model.short_name,
                    "slot": model.slot,
                    "experiment_name": name,
                    "experiment_config": config_path.as_posix(),
                    "train_config": train_path.as_posix(),
                    "model_config": model.model_config,
                    "summary_path": summary.as_posix(),
                    "status": status,
                }
                rows.append(row)

                if not is_complete:
                    queue.append(config_path.as_posix())

    completed = [row for row in rows if row["status"] == "complete_existing_summary"]
    missing = [row for row in rows if row["status"] == "queued_missing_summary"]

    if len(rows) != 180:
        raise RuntimeError(f"Expected 180 target cells, found {len(rows)}")
    if len(completed) + len(missing) != len(rows):
        raise RuntimeError(
            "Completed and missing partitions do not " "cover the complete manifest."
        )
    if len(queue) != len(missing):
        raise RuntimeError("Queue size does not match the " "missing-summary count.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(MANIFEST_CSV, rows)
    MANIFEST_JSON.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    queue_text = "\n".join(queue)
    QUEUE_TXT.write_text(
        queue_text + ("\n" if queue_text else ""),
        encoding="utf-8",
    )
    created_run_script = write_run_script()

    counts: dict[str, Any] = {
        "target_cells": len(rows),
        "completed_existing_summary": len(completed),
        "queued_missing_summary": len(missing),
        "queue_entries": len(queue),
        "created_train_config_count": len(created_train_configs),
        "created_experiment_config_count": len(created_experiment_configs),
        "created_run_script": created_run_script,
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
    SUMMARY_JSON.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
