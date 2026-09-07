param(
    [string]$BasePython = 'D:\python\python.exe',
    [string]$EnvironmentPath = 'D:\venv\AQFC-DETR'
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $EnvironmentPath) {
    throw "Environment already exists; refusing to reuse or overwrite it: $EnvironmentPath"
}
$projectRoot = Split-Path -Parent $PSScriptRoot
function Invoke-CheckedPython([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}
Invoke-CheckedPython $BasePython @('-m','venv','--prompt','AQFC-DETR',$EnvironmentPath)
$Python = Join-Path $EnvironmentPath 'Scripts\python.exe'
Push-Location $projectRoot
try {
    Invoke-CheckedPython $Python @('-m','pip','install','torch==2.7.1','torchvision==0.22.1',
        '--index-url','https://download.pytorch.org/whl/cu118')
    Invoke-CheckedPython $Python @('-m','pip','install','-r','requirements-runtime.txt')
    Invoke-CheckedPython $Python @('-m','pip','install',
        'git+https://github.com/cocodataset/panopticapi.git')
    & (Join-Path $PSScriptRoot 'build_ops.ps1') -Python $Python `
        -OpsDirectory (Join-Path $projectRoot 'cocoapi-aitod\aitodpycocotools')
    & (Join-Path $PSScriptRoot 'build_ops.ps1') -Python $Python
    Invoke-CheckedPython $Python @('-m','pip','check')
} finally { Pop-Location }
