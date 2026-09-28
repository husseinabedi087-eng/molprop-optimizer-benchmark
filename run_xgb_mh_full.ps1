# Main run: Random, TPE, GWO-a0, WOA-a0, PSO; 5 datasets x 2 experiments x 30 seeds x 100 calls.
# Resumable: rerun this same command after a shutdown, hibernate or Ctrl+C; finished runs are skipped.
# Progress: results\xgb_mh\full\full_run.log
# Python: set $env:XGB_MH_PYTHON to a specific interpreter, otherwise `python` on PATH is used.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = if ($env:XGB_MH_PYTHON) { $env:XGB_MH_PYTHON } else { 'python' }
$env:PYTHONPATH = $PSScriptRoot
$env:PYTHONWARNINGS = 'ignore'
& $python -m xgb_mh.full_run  @args
exit $LASTEXITCODE
