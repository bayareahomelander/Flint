# Create or update Flint's Python environment (.venv) on the Windows laptop.
# Needs Python 3.12, the CUDA 12.4 toolkit, and Visual Studio 2022 Build Tools.
# Run from anywhere: powershell -ExecutionPolicy Bypass -File scripts\setup_laptop.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    py -3.12 -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $python -m pip install --no-input --disable-pip-version-check pip==26.1.2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m pip install --no-input --disable-pip-version-check -r (Join-Path $root 'requirements-windows.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name())"
exit $LASTEXITCODE
