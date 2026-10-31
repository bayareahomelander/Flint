# Configure, build, and test Flint on the Windows laptop. Extra arguments go
# to pytest. Run from anywhere:
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\check.ps1 [pytest args]
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'dev_shell.ps1')

Push-Location $root
try {
    cmake --preset release
    if ($LASTEXITCODE -eq 0) { cmake --build --preset release }
    if ($LASTEXITCODE -eq 0) { python -m pytest @args }
} finally {
    Pop-Location
}
exit $LASTEXITCODE
