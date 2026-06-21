from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path.cwd()

EXISTING_COMPLETED_SEEDS = [1337, 2027, 31415]

ADDITIONAL_SEEDS = [
    4441,
    5501,
    6173,
    8191,
    10007,
    11213,
    12721,
    14563,
    16001,
    17749,
    19937,
    22027,
    24103,
    26557,
    28661,
    30757,
    33191,
    35591,
    38039,
    40543,
    43103,
    45641,
]

ALL_SEEDS = EXISTING_COMPLETED_SEEDS + ADDITIONAL_SEEDS

MODELS = [
    {
        "short_name": "dense_121m",
        "slot": "00",
        "model_config": "configs/model/dense_121m.yaml",
        "base_experiment_name": "main_large_50m_00_dense_121m",
        "base_experiment_config": "configs/experiment/main_large_50m_00_dense_121m.yaml",
    },
    {
        "short_name": "mtp_121m",
        "slot": "01",
        "model_config": "configs/model/mtp_121m.yaml",
        "base_experiment_name": "main_large_50m_02_mtp_121m",
        "base_experiment_config": "configs/experiment/main_large_50m_02_mtp_121m.yaml",
    },
    {
        "short_name": "moe_220m",
        "slot": "02",
        "model_config": "configs/model/moe_220m.yaml",
        "base_experiment_name": "main_large_50m_03_moe_220m",
        "base_experiment_config": "configs/experiment/main_large_50m_03_moe_220m.yaml",
    },
    {
        "short_name": "v3_routing_220m",
        "slot": "03",
        "model_config": "configs/model/v3_routing_220m.yaml",
        "base_experiment_name": "main_large_50m_05_v3_routing_220m",
        "base_experiment_config": "configs/experiment/main_large_50m_05_v3_routing_220m.yaml",
    },
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def write_text_if_new_or_same(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.rstrip() + "\n"

    if path.exists():
        existing = path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip() + "\n"
        if existing != normalized:
            raise RuntimeError(f"Refusing to overwrite existing different file: {path}")
        return

    path.write_text(normalized, encoding="utf-8", newline="\n")


def train_config_path(seed: int) -> str:
    if seed == 1337:
        return "configs/train/main_large_50m.yaml"
    return f"configs/train/main_large_50m_seed{seed}.yaml"


def experiment_name(seed: int, model: dict[str, str]) -> str:
    if seed == 1337:
        return model["base_experiment_name"]
    return f"main_large_50m_seed{seed}_{model['slot']}_{model['short_name']}"


def experiment_config_path(seed: int, model: dict[str, str]) -> str:
    if seed == 1337:
        return model["base_experiment_config"]
    return f"configs/experiment/{experiment_name(seed, model)}.yaml"


def train_yaml(seed: int) -> str:
    return f"""train:
  seed: {seed}
  device: cuda
  batch_size: 4
  block_size: 256
  max_steps: null
  max_tokens: 50000000
  eval_interval: 5000
  eval_batches: 100
  learning_rate: 0.0003
  weight_decay: 0.1
  betas:
  - 0.9
  - 0.95
  grad_clip: 1.0
  num_workers: 0
  checkpoint_interval: null
  log_interval: 500
  precision: fp32
"""


def experiment_yaml(seed: int, model: dict[str, str]) -> str:
    name = experiment_name(seed, model)
    return f"""experiment:
  name: {name}
  output_dir: results/raw_logs/{name}
  metrics_dir: results/metrics/{name}
  checkpoint_dir: results/checkpoints/{name}
  model_config: {model["model_config"]}
  data_config: configs/data/fineweb_edu_10bt.yaml
  tokenizer_config: configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml
  train_config: {train_config_path(seed)}
"""


def experiment_row(seed: int, model: dict[str, str]) -> dict[str, Any]:
    name = experiment_name(seed, model)
    metrics_summary = f"results/metrics/{name}/summary.json"
    return {
        "seed": seed,
        "short_name": model["short_name"],
        "slot": model["slot"],
        "experiment_name": name,
        "experiment_config": experiment_config_path(seed, model),
        "train_config": train_config_path(seed),
        "model_config": model["model_config"],
        "data_config": "configs/data/fineweb_edu_10bt.yaml",
        "tokenizer_config": "configs/tokenizer/bpe_fineweb_edu_10bt_local_experiment.yaml",
        "requested_train_tokens": 50_000_000,
        "metrics_summary": metrics_summary,
        "raw_log": f"results/raw_logs/{name}/train_log.jsonl",
        "checkpoint_dir": f"results/checkpoints/{name}",
        "completed_at_generation": Path(metrics_summary).exists(),
        "included_in_training_queue": seed in ADDITIONAL_SEEDS,
    }


def queue_script_text() -> str:
    return r"""param(
    [string]$QueuePath = "configs\experiment\main_large_50m_25seed_queue_new_runs.json",
    [string]$PythonPath = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

Set-Location "C:\Projects\deepseek-reimplementation"

if (-not (Test-Path $PythonPath)) {
    throw "Expected virtual environment Python not found: $PythonPath"
}

New-Item -ItemType Directory -Force -Path "results\analysis" | Out-Null

$StartedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = "results\analysis\main_large_50m_25seed_queue_$StartedAt.transcript.txt"
$ProgressPath = "results\analysis\main_large_50m_25seed_queue_progress.jsonl"

Start-Transcript -Path $TranscriptPath

try {
    Write-Host "`n=== MAIN LARGE 50M 25-SEED QUEUE START ==="
    Write-Host "Queue path: $QueuePath"
    Write-Host "Python path: $PythonPath"
    & $PythonPath --version

    $Preflight = @'
import sys
import torch

print("sys.executable=" + sys.executable)
print("sys.prefix=" + sys.prefix)
print("torch=" + torch.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available from selected Python.")
print("cuda_device=" + torch.cuda.get_device_name(0))
'@
    $PreflightPath = Join-Path $env:TEMP "deepseek_queue_preflight.py"
    $Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($PreflightPath, $Preflight, $Utf8NoBom)
    & $PythonPath $PreflightPath
    Remove-Item $PreflightPath -Force

    Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class SleepControl {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

    $ES_CONTINUOUS = [uint32]"0x80000000"
    $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
    $ES_AWAYMODE_REQUIRED = [uint32]"0x00000040"

    $SleepFlags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED
    [SleepControl]::SetThreadExecutionState($SleepFlags) | Out-Null
    Write-Host "`nSession keep-awake enabled."

    $Payload = Get-Content $QueuePath -Raw | ConvertFrom-Json
    $Runs = $Payload.experiments
    $Total = $Runs.Count
    $Index = 0

    foreach ($Run in $Runs) {
        $Index += 1

        $ExperimentConfig = [string]$Run.experiment_config
        $ExperimentName = [string]$Run.experiment_name
        $SummaryPath = [string]$Run.metrics_summary

        Write-Host "`n=== [$Index / $Total] $ExperimentName ==="
        Write-Host "Config: $ExperimentConfig"
        Write-Host "Summary: $SummaryPath"

        if (Test-Path $SummaryPath) {
            Write-Host "SKIP: summary already exists."
            $Record = [ordered]@{
                timestamp = (Get-Date).ToString("o")
                experiment_name = $ExperimentName
                experiment_config = $ExperimentConfig
                status = "skipped_existing_summary"
                exit_code = 0
            }
            ($Record | ConvertTo-Json -Compress) | Add-Content -Path $ProgressPath
            continue
        }

        $RunStarted = Get-Date
        & $PythonPath scripts\train\run_pretrain.py --experiment-config $ExperimentConfig
        $ExitCode = $LASTEXITCODE
        $RunEnded = Get-Date
        $ElapsedSeconds = ($RunEnded - $RunStarted).TotalSeconds

        if ($ExitCode -ne 0) {
            $Record = [ordered]@{
                timestamp = (Get-Date).ToString("o")
                experiment_name = $ExperimentName
                experiment_config = $ExperimentConfig
                status = "failed"
                exit_code = $ExitCode
                elapsed_seconds = $ElapsedSeconds
            }
            ($Record | ConvertTo-Json -Compress) | Add-Content -Path $ProgressPath
            throw "Run failed: $ExperimentName with exit code $ExitCode"
        }

        if (-not (Test-Path $SummaryPath)) {
            $Record = [ordered]@{
                timestamp = (Get-Date).ToString("o")
                experiment_name = $ExperimentName
                experiment_config = $ExperimentConfig
                status = "failed_missing_summary"
                exit_code = $ExitCode
                elapsed_seconds = $ElapsedSeconds
            }
            ($Record | ConvertTo-Json -Compress) | Add-Content -Path $ProgressPath
            throw "Run completed but summary is missing: $SummaryPath"
        }

        $Record = [ordered]@{
            timestamp = (Get-Date).ToString("o")
            experiment_name = $ExperimentName
            experiment_config = $ExperimentConfig
            status = "completed"
            exit_code = $ExitCode
            elapsed_seconds = $ElapsedSeconds
        }
        ($Record | ConvertTo-Json -Compress) | Add-Content -Path $ProgressPath

        Write-Host "COMPLETED: $ExperimentName"
    }

    Write-Host "`n=== MAIN LARGE 50M 25-SEED QUEUE COMPLETE ==="
}
finally {
    $ES_CONTINUOUS = [uint32]"0x80000000"
    [SleepControl]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    Write-Host "`nSession keep-awake reset."
    Stop-Transcript
}
"""


def main() -> None:
    if len(ALL_SEEDS) != 25:
        raise AssertionError(f"Expected 25 total seeds, got {len(ALL_SEEDS)}")
    if len(set(ALL_SEEDS)) != 25:
        raise AssertionError("Seed list contains duplicates.")

    experiments: list[dict[str, Any]] = []

    for seed in ALL_SEEDS:
        if seed != 1337:
            write_text_if_new_or_same(ROOT / train_config_path(seed), train_yaml(seed))

        for model in MODELS:
            row = experiment_row(seed, model)
            experiments.append(row)

            if seed != 1337:
                write_text_if_new_or_same(
                    ROOT / row["experiment_config"],
                    experiment_yaml(seed, model),
                )

    additional_experiments = [row for row in experiments if row["seed"] in ADDITIONAL_SEEDS]

    manifest = {
        "scope": "main_large_50m_25seed_targeted_paired_replication",
        "description": (
            "Twenty-five aligned seeds for the central 50M-token paired comparison. "
            "Seeds are paired blocks across dense_121m, mtp_121m, moe_220m, and v3_routing_220m."
        ),
        "requested_train_tokens": 50_000_000,
        "target_total_aligned_seeds": 25,
        "existing_completed_seeds": EXISTING_COMPLETED_SEEDS,
        "additional_seeds": ADDITIONAL_SEEDS,
        "all_aligned_seeds": ALL_SEEDS,
        "replicated_models": [model["short_name"] for model in MODELS],
        "batch_size": 4,
        "block_size": 256,
        "precision": "fp32",
        "statistical_design": {
            "design": "paired_seed_blocked_model_comparison",
            "blocking_variable": "seed",
            "primary_budget": "50M requested train tokens",
            "primary_models": [model["short_name"] for model in MODELS],
            "primary_comparisons": [
                "mtp_121m - dense_121m",
                "moe_220m - dense_121m",
                "v3_routing_220m - dense_121m",
                "v3_routing_220m - moe_220m",
                "mtp_121m - moe_220m",
                "mtp_121m - v3_routing_220m",
            ],
            "planned_statistics": [
                "paired mean differences",
                "paired 95% confidence intervals",
                "paired effect sizes",
                "bootstrap confidence intervals",
                "Wilcoxon signed-rank diagnostics",
                "sign consistency",
                "multiple-comparison-aware reporting",
                (
                    "Pareto/frontier analysis across quality, speed, memory, "
                    "and activated-parameter efficiency"
                ),
            ],
        },
        "experiments": experiments,
    }

    queue = {
        "scope": "main_large_50m_25seed_new_run_queue",
        "description": (
            "Queue containing only the 88 new runs needed to expand " "from 3 to 25 aligned seeds."
        ),
        "resume_policy": "Skip a run if its metrics summary JSON already exists.",
        "total_new_runs": len(additional_experiments),
        "additional_seeds": ADDITIONAL_SEEDS,
        "replicated_models": [model["short_name"] for model in MODELS],
        "experiments": additional_experiments,
    }

    if len(experiments) != 100:
        raise AssertionError(f"Expected 100 total manifest experiments, got {len(experiments)}")
    if len(additional_experiments) != 88:
        raise AssertionError(
            f"Expected 88 new queue experiments, got {len(additional_experiments)}"
        )

    manifest_path = ROOT / "configs/experiment/main_large_50m_25seed_manifest.json"
    queue_path = ROOT / "configs/experiment/main_large_50m_25seed_queue_new_runs.json"
    script_path = ROOT / "scripts/train/run_main_large_50m_25seed_queue.ps1"

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    queue_path.write_text(
        json.dumps(queue, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_text_if_new_or_same(script_path, queue_script_text())

    print(
        json.dumps(
            {
                "manifest": rel(manifest_path),
                "queue": rel(queue_path),
                "queue_script": rel(script_path),
                "total_aligned_seeds": len(ALL_SEEDS),
                "total_manifest_experiments": len(experiments),
                "new_queue_experiments": len(additional_experiments),
                "additional_seeds": ADDITIONAL_SEEDS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
