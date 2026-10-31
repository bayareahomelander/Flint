"""The runtime library links only the static CUDA runtime.

Flint never depends on cuBLAS, cuBLASLt, cuDNN, NVML, or the shared CUDA
runtime, so its kernels are all its own. One check reads the library's list of
dependent libraries with the platform's tool; another searches its bytes for
those libraries' names, which a static link would leave behind.
"""

import re
import shutil
import subprocess
import sys

import pytest

import flint_lib

FORBIDDEN = re.compile(r"cublas|cudnn|nvidia-ml|nvml|cudart", re.IGNORECASE)


def run_tool(name, *args):
    path = shutil.which(name)
    if path is None:
        return None
    result = subprocess.run(
        [path, *args], capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace"
    )
    assert result.returncode == 0, f"{name} failed: {result.stderr or result.stdout}"
    return result.stdout


def dependencies(library):
    """The names of the shared libraries the library needs."""
    if sys.platform == "win32":
        output = run_tool("dumpbin", "/DEPENDENTS", str(library))
        if output is None:
            pytest.fail(
                "dumpbin not found on PATH; run in the dev shell "
                "(`. scripts/dev_shell.ps1`), which adds Visual Studio's tools"
            )
        # The dependencies are the indented *.dll lines after the header.
        return re.findall(r"^\s+(\S+\.dll)\s*$", output, re.IGNORECASE | re.MULTILINE)
    output = run_tool("readelf", "-d", str(library))
    if output is not None:
        return re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", output)
    output = run_tool("objdump", "-p", str(library))
    if output is None:
        pytest.fail("neither readelf nor objdump found on PATH; install binutils")
    return re.findall(r"^\s*NEEDED\s+(\S+)", output, re.MULTILINE)


def built(path):
    if not path.is_file():
        pytest.fail(
            f"{path} not found; build it with "
            "`cmake --preset release && cmake --build --preset release`"
        )
    return path


def test_library_links_no_cuda_libraries():
    library = built(flint_lib.library_path())
    found = dependencies(library)
    # Every library needs at least the C runtime, so an empty list means the
    # tool's output wasn't parsed.
    assert found, f"no dependencies parsed from {library}"
    banned = [name for name in found if FORBIDDEN.search(name)]
    assert not banned, f"{library.name} depends on {banned}; all dependencies: {found}"


# Names a statically linked cuBLAS, cuDNN or NVML would leave in the binary,
# in its symbols and messages.
STATIC_NAMES = re.compile(rb"cublas|cudnn|nvml", re.IGNORECASE)


def library_names_in(path):
    return sorted({match.group(0).lower() for match in STATIC_NAMES.finditer(path.read_bytes())})


def test_library_contains_no_cuda_library_code():
    library = built(flint_lib.library_path())
    # flint_bench links cuBLAS and NVML, so the search must find them there,
    # or it proves nothing about the library.
    bench = built(library.parent / ("flint_bench.exe" if sys.platform == "win32" else "flint_bench"))
    assert library_names_in(bench) == [b"cublas", b"nvml"]
    assert library_names_in(library) == [], f"{library.name} contains cuBLAS, cuDNN or NVML names"
