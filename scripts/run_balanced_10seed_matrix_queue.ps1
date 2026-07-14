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
