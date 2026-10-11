$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
$logRoot = Join-Path $PSScriptRoot 'run_logs'
$statusPath = Join-Path $logRoot 'Tawan_run_status.txt'
$alreadyRunningPid = 39524

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Add-Content -LiteralPath $statusPath -Value "Waiting for E2_1 process $alreadyRunningPid at $(Get-Date -Format o)"
try { Wait-Process -Id $alreadyRunningPid -ErrorAction SilentlyContinue } catch { }

$experiments = @(
    'Tawan_E2_2_global_multi_head.py',
    'Tawan_E2_7_regional_single_head.py',
    'Tawan_E2_8_regional_multi_head.py'
)

foreach ($scriptName in $experiments) {
    $logName = [System.IO.Path]::GetFileNameWithoutExtension($scriptName)
    $stdoutPath = Join-Path $logRoot "$logName.stdout.log"
    $stderrPath = Join-Path $logRoot "$logName.stderr.log"
    Add-Content -LiteralPath $statusPath -Value "Starting $scriptName at $(Get-Date -Format o)"
    $process = Start-Process -FilePath $pythonPath `
        -ArgumentList @((Join-Path $PSScriptRoot $scriptName), '--train', '--source', 'local') `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru
    $process.WaitForExit()
    Add-Content -LiteralPath $statusPath -Value "Finished $scriptName with exit code $($process.ExitCode) at $(Get-Date -Format o)"
}
