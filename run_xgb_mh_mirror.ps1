# Mirrored-encoding robustness check (pre-registered: results\xgb_mh\mirror\prediction.md).
# Resumable: rerun this same command after a shutdown, hibernate or Ctrl+C; finished runs are skipped.
# Progress: results\xgb_mh\mirror\full_run.log
# Python: set $env:XGB_MH_PYTHON to a specific interpreter, otherwise `python` on PATH is used.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = if ($env:XGB_MH_PYTHON) { $env:XGB_MH_PYTHON } else { 'python' }
$env:PYTHONPATH = $PSScriptRoot
$env:PYTHONWARNINGS = 'ignore'
& $python -m xgb_mh.full_run --out-dir results/xgb_mh/mirror --mirror --experiments B --datasets BBBP,FreeSolv --optimizers random,tpe,gwo_ref,woa_ref,pso @args
exit $LASTEXITCODE
