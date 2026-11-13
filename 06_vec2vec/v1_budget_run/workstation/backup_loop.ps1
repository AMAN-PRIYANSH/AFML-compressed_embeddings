# Copy outputs\ to a cloud-synced folder every hour (for example your OneDrive folder).
# Start it hidden in the background:
#   powershell -ExecutionPolicy Bypass -Command "Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','backup_loop.ps1','-Target','C:\Users\Student5\OneDrive\vec2vec-backup'"
param([Parameter(Mandatory = $true)][string]$Target, [int]$EveryMinutes = 60)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
New-Item -ItemType Directory -Force -Path $Target | Out-Null
while ($true) {
    # /E all subfolders, /XO skip files that are not newer, /R:2 /W:5 retry briefly on locked files
    robocopy (Join-Path $here 'outputs') $Target /E /XO /R:2 /W:5 /NP /NFL /NDL "/LOG+:$(Join-Path $here 'backup_log.txt')" | Out-Null
    Add-Content (Join-Path $here 'backup_log.txt') "backup finished $(Get-Date -Format s)"
    $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_v2v_local.py*' }
    if (-not $alive) {   # runs are over: one last copy, then stop
        robocopy (Join-Path $here 'outputs') $Target /E /XO /R:2 /W:5 /NP /NFL /NDL "/LOG+:$(Join-Path $here 'backup_log.txt')" | Out-Null
        break
    }
    Start-Sleep -Seconds ($EveryMinutes * 60)
}
