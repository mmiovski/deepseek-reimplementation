param(
    [string]$QueuePath = "results\analysis\balanced_10seed_matrix_queue.txt"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ProgressPath = Join-Path $ProjectRoot "results\analysis\balanced_10seed_matrix_queue_progress.jsonl"
$SystemStatePath = Join-Path $ProjectRoot "results\analysis\long_run_system_state.json"
$TrainingPidPath = Join-Path $ProjectRoot "tmp\long_run_training.pid"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing repaired Python environment: $Python"
}

$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = [Security.Principal.WindowsPrincipal]::new($Identity)
if (-not $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Launch this runner from an elevated PowerShell window."
}

function Write-DurableJsonLine {
    param([hashtable]$Record)
    $Parent = Split-Path -Parent $ProgressPath
    New-Item -ItemType Directory -Path $Parent -Force | Out-Null
    $Line = ($Record | ConvertTo-Json -Compress) + [Environment]::NewLine
    $Bytes = [Text.Encoding]::UTF8.GetBytes($Line)
    $Stream = [IO.FileStream]::new(
        $ProgressPath,
        [IO.FileMode]::Append,
        [IO.FileAccess]::Write,
        [IO.FileShare]::Read,
        4096,
        [IO.FileOptions]::WriteThrough
    )
    try {
        $Stream.Write($Bytes, 0, $Bytes.Length)
        $Stream.Flush($true)
    } finally {
        $Stream.Dispose()
    }
}

function Set-PowerGuard {
    if ($null -eq ("LongRunExecutionState" -as [type])) {
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class LongRunExecutionState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
    }
    # Windows PowerShell 5.1 parses 0x80000000 as a negative Int32 before a
    # cast is applied. Convert the complete bit pattern from hexadecimal so
    # the native call receives the intended unsigned execution-state flags.
    $Flags = [Convert]::ToUInt32("80000041", 16)
    if ([LongRunExecutionState]::SetThreadExecutionState($Flags) -eq 0) {
        throw "Windows rejected the long-run power guard."
    }
}

function Clear-PowerGuard {
    if ($null -ne ("LongRunExecutionState" -as [type])) {
        $ContinuousFlag = [Convert]::ToUInt32("80000000", 16)
        [LongRunExecutionState]::SetThreadExecutionState($ContinuousFlag) | Out-Null
    }
}

function Enable-UpdateGuard {
    $PolicyPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU"
    $Previous = $null
    $Existed = $false
    if (Test-Path -LiteralPath $PolicyPath) {
        $Property = Get-ItemProperty -LiteralPath $PolicyPath -Name NoAutoUpdate -ErrorAction SilentlyContinue
        if ($null -ne $Property) {
            $Existed = $true
            $Previous = [int]$Property.NoAutoUpdate
        }
    }
    @{
        policy_path = $PolicyPath
        value_existed = $Existed
        previous_value = $Previous
        enabled_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $SystemStatePath -Encoding UTF8
    New-Item -Path $PolicyPath -Force | Out-Null
    New-ItemProperty -LiteralPath $PolicyPath -Name NoAutoUpdate -PropertyType DWord -Value 1 -Force | Out-Null
}

function Stop-UpdateActivity {
    foreach ($Name in @("wuauserv", "UsoSvc", "BITS", "DoSvc")) {
        $Service = Get-Service -Name $Name -ErrorAction SilentlyContinue
        if ($null -ne $Service -and $Service.Status -ne "Stopped") {
            Stop-Service -Name $Name -Force -ErrorAction SilentlyContinue
        }
    }
}

function Disable-UpdateGuard {
    if (-not (Test-Path -LiteralPath $SystemStatePath)) {
        return
    }
    $State = Get-Content -LiteralPath $SystemStatePath -Raw | ConvertFrom-Json
    if ($State.value_existed) {
        New-ItemProperty `
            -LiteralPath $State.policy_path `
            -Name NoAutoUpdate `
            -PropertyType DWord `
            -Value ([int]$State.previous_value) `
            -Force | Out-Null
    } else {
        Remove-ItemProperty -LiteralPath $State.policy_path -Name NoAutoUpdate -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $SystemStatePath -Force
}

function Assert-NoCompetingInteractiveApps {
    $Blocked = @("Codex", "ChatGPT", "chrome", "msedge", "firefox", "Discord", "Teams", "steam")
    $Running = Get-Process -Name $Blocked -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty ProcessName -Unique
    if ($Running) {
        throw "Close competing interactive apps before launch: $($Running -join ', ')"
    }
}

function Assert-NoPendingReboot {
    $PendingKeys = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
    )
    if ($PendingKeys | Where-Object { Test-Path -LiteralPath $_ }) {
        throw "Windows reports a pending reboot; restart before the long run."
    }
}

function Assert-AcPower {
    $Batteries = @(Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue)
    if ($Batteries | Where-Object { [int]$_.BatteryStatus -eq 1 }) {
        throw "The system is running on battery power. Connect AC power before continuing."
    }
}

function Assert-FreeDiskSpace {
    $Drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($ProjectRoot))
    if ($Drive.AvailableFreeSpace -lt 20GB) {
        throw "Less than 20 GiB of free workspace drive capacity remains."
    }
}

function Wait-TrainingPidMarker {
    param(
        [Diagnostics.Process]$LauncherProcess,
        [string]$MarkerPath,
        [int]$TimeoutSeconds = 120
    )

    $Deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while (-not (Test-Path -LiteralPath $MarkerPath -PathType Leaf)) {
        $LauncherProcess.Refresh()
        if ($LauncherProcess.HasExited) {
            throw "Training launcher exited before publishing its interpreter PID."
        }
        if ([DateTime]::UtcNow -ge $Deadline) {
            throw "Training interpreter did not publish its PID within $TimeoutSeconds seconds."
        }
        Start-Sleep -Milliseconds 250
    }

    $RawPid = (Get-Content -LiteralPath $MarkerPath -Raw).Trim()
    $TrainingPid = 0
    if (-not [int]::TryParse($RawPid, [ref]$TrainingPid) -or $TrainingPid -le 0) {
        throw "Training interpreter published an invalid PID marker: $RawPid"
    }
    if ($null -eq (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue)) {
        throw "Training interpreter PID $TrainingPid is no longer active."
    }
    return $TrainingPid
}

function Stop-OwnedProcesses {
    param([int[]]$OwnedPids)

    foreach ($OwnedProcessId in @($OwnedPids | Select-Object -Unique)) {
        if ($OwnedProcessId -gt 0 -and $OwnedProcessId -ne $PID) {
            Stop-Process -Id $OwnedProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}

function Get-CompetingGpuComputeProcesses {
    param([int[]]$AllowedPids = @())

    $AllowedPidSet = [Collections.Generic.HashSet[int]]::new()
    foreach ($AllowedProcessId in $AllowedPids) {
        [void]$AllowedPidSet.Add($AllowedProcessId)
    }
    $Rows = & nvidia-smi `
        --query-compute-apps=pid,process_name `
        --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi compute-process query failed."
    }
    $Competitors = @()
    foreach ($Row in $Rows) {
        if ([string]::IsNullOrWhiteSpace($Row)) { continue }
        $Parts = $Row -split ",", 2
        $PidValue = [int]$Parts[0].Trim()
        if (-not $AllowedPidSet.Contains($PidValue)) {
            $Competitors += $Row.Trim()
        }
    }
    return $Competitors
}

if (Test-Path -LiteralPath $SystemStatePath) {
    Disable-UpdateGuard
}

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:PYTHONHASHSEED = "0"

& $Python scripts\analysis\build_balanced_10seed_matrix_manifest.py
if ($LASTEXITCODE -ne 0) { throw "Configuration/manifest generation failed." }
& $Python scripts\validation\preflight_primary_matrix.py --require-clean-git
if ($LASTEXITCODE -ne 0) { throw "Primary-matrix preflight failed." }

if (-not (Test-Path -LiteralPath $QueuePath -PathType Leaf)) {
    throw "Missing queue file: $QueuePath"
}
$Queue = @(Get-Content -LiteralPath $QueuePath | Where-Object { $_.Trim() })
$ExistingCheckpoints = @(Get-ChildItem `
    -LiteralPath (Join-Path $ProjectRoot "results\runs") `
    -Filter "checkpoint.pt" `
    -File `
    -Recurse `
    -ErrorAction SilentlyContinue)
if ($Queue.Count -eq 180 -and $ExistingCheckpoints.Count -eq 0) {
    [IO.File]::WriteAllBytes($ProgressPath, [byte[]]@())
}

$CompletedProgressConfigs = [Collections.Generic.HashSet[string]]::new()
$StartedProgressConfigs = [Collections.Generic.HashSet[string]]::new()
if (Test-Path -LiteralPath $ProgressPath -PathType Leaf) {
    foreach ($Line in Get-Content -LiteralPath $ProgressPath) {
        if ([string]::IsNullOrWhiteSpace($Line)) { continue }
        $Record = $Line | ConvertFrom-Json
        if ($Record.status -eq "completed") {
            [void]$CompletedProgressConfigs.Add([string]$Record.experiment_config)
        }
        if ($Record.status -eq "started") {
            [void]$StartedProgressConfigs.Add([string]$Record.experiment_config)
        }
    }
}
$ManifestRows = Get-Content `
    -LiteralPath (Join-Path $ProjectRoot "results\analysis\balanced_10seed_matrix_manifest.json") `
    -Raw | ConvertFrom-Json
foreach ($Row in $ManifestRows) {
    if ($Row.status -ne "complete_existing_summary") { continue }
    $ConfigPath = [string]$Row.experiment_config
    if (-not $StartedProgressConfigs.Contains($ConfigPath)) {
        Write-DurableJsonLine @{
            timestamp = (Get-Date).ToString("o")
            queue_index = 0
            queue_total = 180
            experiment_config = $ConfigPath
            status = "started"
            reconciled_from_validated_summary = $true
        }
        [void]$StartedProgressConfigs.Add($ConfigPath)
    }
    if (-not $CompletedProgressConfigs.Contains($ConfigPath)) {
        Write-DurableJsonLine @{
            timestamp = (Get-Date).ToString("o")
            queue_index = 0
            queue_total = 180
            experiment_config = $ConfigPath
            status = "completed"
            reconciled_from_validated_summary = $true
        }
        [void]$CompletedProgressConfigs.Add($ConfigPath)
    }
}

if ($Queue.Count -eq 0) {
    & $Python scripts\validation\preflight_primary_matrix.py --require-clean-git --require-complete
    if ($LASTEXITCODE -ne 0) { throw "Completed-matrix validation failed." }
    & $Python scripts\analysis\run_balanced_10seed_pipeline.py
    if ($LASTEXITCODE -ne 0) { throw "Analysis/figure regeneration failed." }
    "All primary runs and analysis artifacts are complete and validated."
    exit 0
}

Assert-NoCompetingInteractiveApps
Assert-NoPendingReboot
Assert-AcPower
Assert-FreeDiskSpace
if ((Get-CompetingGpuComputeProcesses).Count -gt 0) {
    throw "A competing CUDA compute process is already active."
}

try {
    Set-PowerGuard
    Enable-UpdateGuard
    Stop-UpdateActivity

    for ($Index = 0; $Index -lt $Queue.Count; $Index++) {
        $ExperimentConfig = [string]$Queue[$Index]
        $QueueIndex = $Index + 1
        Set-PowerGuard
        Stop-UpdateActivity
        Assert-NoCompetingInteractiveApps
        Assert-AcPower
        Assert-FreeDiskSpace

        $StartedAt = Get-Date
        Write-DurableJsonLine @{
            timestamp = $StartedAt.ToString("o")
            queue_index = $QueueIndex
            queue_total = $Queue.Count
            experiment_config = $ExperimentConfig
            status = "started"
        }

        $Process = $null
        $TrainingPid = $null
        Remove-Item -LiteralPath $TrainingPidPath -Force -ErrorAction SilentlyContinue
        try {
            $Process = Start-Process `
                -FilePath $Python `
                -ArgumentList @(
                    "scripts\train\run_pretrain.py",
                    "--experiment-config",
                    $ExperimentConfig,
                    "--runner-pid-file",
                    $TrainingPidPath
                ) `
                -WorkingDirectory $ProjectRoot `
                -NoNewWindow `
                -PassThru

            $TrainingPid = Wait-TrainingPidMarker `
                -LauncherProcess $Process `
                -MarkerPath $TrainingPidPath

            $MonitorIterations = 0
            while (-not $Process.HasExited) {
                Start-Sleep -Seconds 30
                $Process.Refresh()
                $MonitorIterations += 1
                Set-PowerGuard
                Stop-UpdateActivity
                if ($MonitorIterations % 20 -eq 0) { Assert-AcPower }
                $OwnedPids = @($Process.Id, $TrainingPid)
                $Competitors = @(Get-CompetingGpuComputeProcesses -AllowedPids $OwnedPids)
                if ($Competitors.Count -gt 0) {
                    throw "Competing CUDA process detected: $($Competitors -join '; ')"
                }
            }

            if ($Process.ExitCode -ne 0) {
                Write-DurableJsonLine @{
                    timestamp = (Get-Date).ToString("o")
                    queue_index = $QueueIndex
                    queue_total = $Queue.Count
                    experiment_config = $ExperimentConfig
                    status = "failed"
                    exit_code = $Process.ExitCode
                }
                throw "Training failed for $ExperimentConfig with exit code $($Process.ExitCode)."
            }
        } finally {
            if ($null -ne $Process -and -not $Process.HasExited) {
                Stop-OwnedProcesses -OwnedPids @($TrainingPid, $Process.Id)
                $Process.WaitForExit()
            }
            Remove-Item -LiteralPath $TrainingPidPath -Force -ErrorAction SilentlyContinue
        }

        & $Python scripts\validation\preflight_primary_matrix.py --require-clean-git
        if ($LASTEXITCODE -ne 0) {
            throw "Post-run semantic validation failed for $ExperimentConfig."
        }

        Write-DurableJsonLine @{
            timestamp = (Get-Date).ToString("o")
            queue_index = $QueueIndex
            queue_total = $Queue.Count
            experiment_config = $ExperimentConfig
            status = "completed"
            duration_minutes = [Math]::Round(((Get-Date) - $StartedAt).TotalMinutes, 2)
        }
    }
} finally {
    Disable-UpdateGuard
    Clear-PowerGuard
}

& $Python scripts\analysis\build_balanced_10seed_matrix_manifest.py
if ($LASTEXITCODE -ne 0) { throw "Final manifest validation failed." }
& $Python scripts\validation\preflight_primary_matrix.py --require-clean-git --require-complete
if ($LASTEXITCODE -ne 0) { throw "Final completed-matrix validation failed." }
& $Python scripts\analysis\run_balanced_10seed_pipeline.py
if ($LASTEXITCODE -ne 0) { throw "Final analysis/figure regeneration failed." }
"All 180 primary runs completed and validated."
