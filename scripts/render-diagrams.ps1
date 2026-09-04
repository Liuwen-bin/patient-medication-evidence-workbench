$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$assets = Join-Path $root "docs/portfolio/assets"
New-Item -ItemType Directory -Force $assets | Out-Null

npx -y @mermaid-js/mermaid-cli@11.4.2 `
  -i (Join-Path $root "docs/portfolio/architecture.mmd") `
  -o (Join-Path $assets "architecture.png") `
  -w 1600 -H 1000 -b transparent
if ($LASTEXITCODE -ne 0) { throw "Architecture diagram rendering failed." }

npx -y @mermaid-js/mermaid-cli@11.4.2 `
  -i (Join-Path $root "docs/portfolio/state-machine.mmd") `
  -o (Join-Path $assets "state-machine.png") `
  -w 1600 -H 1000 -b transparent
if ($LASTEXITCODE -ne 0) { throw "State diagram rendering failed." }
