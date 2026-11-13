# Show which runs are alive, their last 3 epochs, any errors, and GPU use.
# Usage:  powershell -ExecutionPolicy Bypass -File .\status.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_v2v_local.py*' }
if ($procs) { $procs | ForEach-Object { Write-Host "RUNNING  pid $($_.ProcessId)  $($_.CommandLine)" -ForegroundColor Green } }
else { Write-Host "No vec2vec run is running right now." -ForegroundColor Yellow }

Get-ChildItem (Join-Path $here 'outputs') -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "`n== $($_.Name)"
    $log = Join-Path $_.FullName 'v2v_log.jsonl'
    if (Test-Path $log) {
        Get-Content $log | Where-Object { $_ -like '*"kind": "ours"*' } | Select-Object -Last 3 | ForEach-Object {
            $r = $_ | ConvertFrom-Json
            "  epoch {0,5}   {1,7:N2} h   top1 = {2:N4}   avg_rank = {3,8:N1}   cos = {4:N3}" -f $r.epoch, $r.t_hours, $r.top1, $r.avg_rank, $r.mean_cos
        }
    } else { Write-Host "  no epoch finished yet" }
    $summary = Get-ChildItem $_.FullName -Filter 'summary_*.json' -ErrorAction SilentlyContinue
    if ($summary) { Write-Host "  FINISHED: $($summary.Name) written" -ForegroundColor Green }
    $err = Join-Path $_.FullName 'stderr.txt'
    if ((Test-Path $err) -and (Select-String -Path $err -Pattern 'Traceback' -Quiet)) {
        Write-Host "  !! a Traceback is in $err - send that file to Claude Code" -ForegroundColor Red
    }
}
Write-Host ""
& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
