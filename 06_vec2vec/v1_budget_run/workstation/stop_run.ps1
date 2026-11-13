# Stop one run early (the files written so far stay in outputs\<scheme>).
# Usage:  powershell -ExecutionPolicy Bypass -File .\stop_run.ps1 -Scheme fp32
param([Parameter(Mandatory = $true)][ValidateSet('fp32', 'pq128x8')][string]$Scheme)
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe'" |
    Where-Object { $_.CommandLine -like "*run_v2v_local.py*--scheme $Scheme*" }
if (-not $procs) { Write-Host "No $Scheme run found."; exit 0 }
$procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host "Stopped pid $($_.ProcessId)" }
