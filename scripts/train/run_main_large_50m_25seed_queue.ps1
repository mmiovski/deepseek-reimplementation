param(
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
        & $PythonPath scripts\train\run_pretrain.py $ExperimentConfig
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
