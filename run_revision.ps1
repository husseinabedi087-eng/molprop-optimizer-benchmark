# Revision experiments (IJIES review). Run from the repository root with the virtual environment active.
# Resumable: stop at any time (Ctrl+C) and run the same command again; finished tasks are skipped.
# Example: powershell -File run_revision.ps1 -Jobs J1          (one job)
#          powershell -File run_revision.ps1                   (all jobs in order)
param([string[]]$Jobs = @("J1", "J3", "J4", "J5", "J2", "J6"), [int]$Workers = 10)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path results\revision | Out-Null
foreach ($j in $Jobs) {
    python -W ignore -m revision.jobs $j --workers $Workers 2>&1 | Tee-Object -FilePath results\revision\run.log -Append
}
python -m revision.status
