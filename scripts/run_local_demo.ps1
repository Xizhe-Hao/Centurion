# Local DiLoCo demo: coordinator + 2 PyTorch workers, each as its own process.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\run_local_demo.ps1
#
# This is the single-machine version of the real Windows<->Mac setup. On the
# real setup you run the coordinator on one machine and point each worker's
# --coord-url at that machine's LAN IP.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$port = 9100
$rounds = 10
$localSteps = 20

Write-Host "Starting coordinator on 127.0.0.1:$port (world_size=2, rounds=$rounds)"
$coord = Start-Process -FilePath $py -PassThru -NoNewWindow `
    -ArgumentList "-m","centurion_coord.server","--host","127.0.0.1","--port","$port","--world-size","2","--rounds","$rounds" `
    -RedirectStandardOutput "$logs\coord.out.log" -RedirectStandardError "$logs\coord.err.log" `
    -WorkingDirectory $root

Start-Sleep -Seconds 3

$workers = @()
foreach ($i in 0,1) {
    Write-Host "Starting worker w$i (data-seed=$i)"
    $w = Start-Process -FilePath $py -PassThru -NoNewWindow `
        -ArgumentList "-m","centurion_worker.diloco_client","--coord-url","http://127.0.0.1:$port","--worker-id","w$i","--framework","pytorch","--data-seed","$i","--local-steps","$localSteps","--rounds","$rounds" `
        -RedirectStandardOutput "$logs\w$i.out.log" -RedirectStandardError "$logs\w$i.err.log" `
        -WorkingDirectory $root
    $workers += $w
}

Write-Host "Waiting for workers to finish..."
$workers | ForEach-Object { $_.WaitForExit() }

Write-Host "Stopping coordinator"
if (-not $coord.HasExited) { Stop-Process -Id $coord.Id -Force }

Write-Host "`n===== worker w0 log ====="
Get-Content "$logs\w0.out.log"
Write-Host "`n===== worker w1 log ====="
Get-Content "$logs\w1.out.log"
