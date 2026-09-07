param(
  [Parameter(Mandatory=$true)][string]$DataRoot,
  [string]$OutputDir = "outputs\aitodv2_local8gb",
  [string]$Python = 'D:\venv\AQFC-DETR\Scripts\python.exe',
  [string]$Config = 'configs/aitodv2/aqfc_r50_5scale_local8gb.py',
  [string]$Pretrained = '',
  [string]$Resume = '',
  [int]$NumWorkers = 0
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw "Environment not found: $Python" }
if ($Pretrained -and $Resume) { throw 'Choose Pretrained or Resume, not both.' }
$trainArgs = @('main.py', '--config', $Config, '--data-root', $DataRoot,
  '--output-dir', $OutputDir, '--num_workers', "$NumWorkers")
if ($Pretrained) { $trainArgs += @('--pretrained', $Pretrained) }
if ($Resume) { $trainArgs += @('--resume', $Resume) }
Push-Location (Join-Path $PSScriptRoot '..')
try {
  & $Python @trainArgs
  if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
