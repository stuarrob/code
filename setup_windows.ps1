# setup_windows.ps1 — build the conda env, install the package, verify.
# Run AFTER the Option-A import refactor is in place.
$ErrorActionPreference = "Stop"
$RepoRoot  = $PSScriptRoot
$CondaRoot = Join-Path $env:APPDATA "miniconda3"
$EnvName   = "etftrader"
$DataRoot  = "C:\Users\stuar\trade_data\ETFTrader"
$CondaExe  = Join-Path $CondaRoot "Scripts\conda.exe"
if (-not (Test-Path $CondaExe)) { Write-Error "conda not found at $CondaExe - install Miniconda first." }
if (& $CondaExe env list | Select-String -Pattern "^\s*$EnvName\s") {
    & $CondaExe env update -n $EnvName -f (Join-Path $RepoRoot "environment.yml") --prune
} else {
    & $CondaExe env create -n $EnvName -f (Join-Path $RepoRoot "environment.yml")
}
$EnvPy = Join-Path $CondaRoot "envs\$EnvName\python.exe"
& $EnvPy -m pip install -e $RepoRoot
foreach ($d in @("raw","processed","logs","indicators","signals","ib_historical")) {
    $p = Join-Path $DataRoot $d
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}
& $EnvPy -c "import numpy, pandas, cvxpy, ib_insync, yfinance, src; import os; print('imports OK; DATA_DIR =', os.getenv('DATA_DIR'))"
& $EnvPy -m pytest -q -m "not slow and not requires_data and not integration"
Write-Host "Done."
