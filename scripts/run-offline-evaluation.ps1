$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
New-Item -ItemType Directory -Force artifacts/evaluation | Out-Null
python -m medication_review_agent.evaluation `
  --cases evaluation/cases.jsonl `
  --output artifacts/evaluation/offline-regression-report.json
