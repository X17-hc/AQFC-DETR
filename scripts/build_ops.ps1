param(
    [string]$Python = 'D:\venv\AQFC-DETR\Scripts\python.exe',
    [string]$OpsDirectory = (Join-Path $PSScriptRoot '..\models\aqfcdetr\ops'),
    [string]$WheelDirectory = '',
    [switch]$WheelOnly
)
$ErrorActionPreference = 'Stop'
$vcRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133'
$sdkRoot = 'C:\Program Files (x86)\Windows Kits\10'
$sdkVersion = '10.0.19041.0'
if (-not (Test-Path -LiteralPath "$vcRoot\include")) { throw 'Configure vcRoot for the installed MSVC compiler.' }
$buildVariables = @('DISTUTILS_USE_SDK','MSSdk','VSCMD_ARG_TGT_ARCH','VSLANG',
    'MAX_JOBS','TORCH_CUDA_ARCH_LIST','PYTORCH_NVCC','INCLUDE','LIB','PATH')
$savedBuildEnvironment = @{}
foreach ($key in $buildVariables) { $savedBuildEnvironment[$key] = [Environment]::GetEnvironmentVariable($key) }
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
$env:VSCMD_ARG_TGT_ARCH = 'x64'
$env:VSLANG = '1033'
$env:MAX_JOBS = '2'
$env:TORCH_CUDA_ARCH_LIST = '8.9'
$env:PYTORCH_NVCC = '"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe"'
$env:INCLUDE = "$vcRoot\include;$sdkRoot\Include\$sdkVersion\ucrt;$sdkRoot\Include\$sdkVersion\shared;$sdkRoot\Include\$sdkVersion\um;$sdkRoot\Include\$sdkVersion\winrt"
$env:LIB = "$vcRoot\lib\x64;$sdkRoot\Lib\$sdkVersion\ucrt\x64;$sdkRoot\Lib\$sdkVersion\um\x64"
$env:PATH = "$(Split-Path -Parent $Python);$vcRoot\bin\Hostx64\x64;$sdkRoot\bin\$sdkVersion\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;$env:SystemRoot\System32;$env:SystemRoot"
Push-Location $OpsDirectory
try {
    if ($WheelOnly) {
        & $Python -m pip wheel --no-cache-dir --no-build-isolation --no-deps --wheel-dir $WheelDirectory .
    } else {
        & $Python -m pip install --no-build-isolation --no-deps .
    }
    if ($LASTEXITCODE -ne 0) { throw 'CUDA extension compilation failed.' }
} finally {
    Pop-Location
    foreach ($key in $buildVariables) { [Environment]::SetEnvironmentVariable($key, $savedBuildEnvironment[$key]) }
}
