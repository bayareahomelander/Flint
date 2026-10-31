# Enter Flint's laptop build shell: the Visual Studio x64 developer environment
# (cl.exe, link.exe, the Windows SDK), the project venv (cmake, ninja, pytest,
# torch), and the newest installed Nsight Compute and Nsight Systems.
# Dot-source it in PowerShell from the Flint folder:  . .\scripts\dev_shell.ps1
$root = Split-Path -Parent $PSScriptRoot

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio 2022 with the C++ x64 build tools was not found' }
# VsDevCmd's own scripts call vswhere by name.
$env:PATH = "$(Split-Path $vswhere);$env:PATH"
Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null

# Activated last, so the venv's pinned cmake and ninja win over Visual Studio's.
& (Join-Path $root '.venv\Scripts\Activate.ps1')

$nvidia = 'C:\Program Files\NVIDIA Corporation'
$tools = @(
    (Join-Path $env:CUDA_PATH 'compute-sanitizer'),
    (Get-ChildItem $nvidia -Directory -Filter 'Nsight Compute *' -ErrorAction SilentlyContinue |
        Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName),
    (Get-ChildItem $nvidia -Directory -Filter 'Nsight Systems *' -ErrorAction SilentlyContinue |
        Sort-Object Name | Select-Object -Last 1 |
        ForEach-Object { Join-Path $_.FullName 'target-windows-x64' })
) | Where-Object { $_ -and (Test-Path $_) }
$env:PATH = (($tools + $env:PATH) -join ';')
