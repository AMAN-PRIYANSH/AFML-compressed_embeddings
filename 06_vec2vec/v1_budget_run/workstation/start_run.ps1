# Start one vec2vec run in the background (hidden, keeps running if this window is closed).
# Usage:  powershell -ExecutionPolicy Bypass -File .\start_run.ps1 -Scheme fp32 -Hours 66
#         powershell -ExecutionPolicy Bypass -File .\start_run.ps1 -Scheme fp32 -Hours 0.1 -Smoke
param(
    [Parameter(Mandatory = $true)][ValidateSet('fp32', 'pq128x8')][string]$Scheme,
    [Parameter(Mandatory = $true)][double]$Hours,
    [switch]$Smoke
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$name = if ($Smoke) { "${Scheme}_smoke" } else { $Scheme }
$out = Join-Path $here "outputs\$name"

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_v2v_local.py*--scheme $Scheme*" }
if ($running) { Write-Host "A $Scheme run is already running (pid $($running.ProcessId)). Not starting a second one."; exit 1 }

New-Item -ItemType Directory -Force -Path $out | Out-Null
$argList = @('run', 'python', 'run_v2v_local.py', '--scheme', $Scheme, '--hours', "$Hours")
if ($Smoke) { $argList += '--smoke' }
$p = Start-Process -FilePath 'uv' -ArgumentList $argList -WorkingDirectory $here `
    -RedirectStandardOutput (Join-Path $out 'stdout.txt') -RedirectStandardError (Join-Path $out 'stderr.txt') `
    -WindowStyle Hidden -PassThru
Write-Host "Started $name in the background (launcher pid $($p.Id))."
Write-Host "Check progress:  powershell -ExecutionPolicy Bypass -File .\status.ps1"
