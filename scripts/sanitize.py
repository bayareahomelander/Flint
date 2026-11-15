"""Runs the test suite under NVIDIA compute-sanitizer, and proves it sees Flint's kernels.

Usage: python scripts/sanitize.py [--tools memcheck,racecheck,synccheck,initcheck]
       [pytest args...]

Each tool runs the whole suite and must report zero errors, and racecheck zero
warnings too. Two canaries then run deliberately broken calls and must be
caught: an out-of-bounds read under memcheck, and an uninitialized read under
initcheck. Unrecognized arguments go to pytest. Exits nonzero if any run fails.

A canary counts as caught only when its run exits nonzero and prints the
expected message. The tool is the one on PATH, or the executable that the
environment variable FLINT_COMPUTE_SANITIZER names; the log starts with its
path and version. On the 2080 Ti server, compute-sanitizer 2024.1.1 crashes
intermittently while reporting an error it caught, so such a run proves nothing
about Flint's kernels. The server's precheck therefore runs compute-sanitizer
2026.1.1 from the `sancheck` micromamba environment (scripts/setup_server.sh).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ["memcheck", "racecheck", "synccheck", "initcheck"]

# A substring of Flint's mangled namespace (_ZN5flint...), so only Flint's
# kernels are checked. initcheck can't use it: it must see torch's writes too.
FLINT_KERNELS = ["--kernel-name", "kns=5flint"]
FILTERED_TOOLS = ["memcheck", "racecheck", "synccheck"]
# Padding after each allocation lets memcheck catch reads that run into a
# neighbouring allocation; plain cudaMalloc often packs them back to back.
MEMCHECK_PADDING = ["--padding", "65536"]
# On the server, compute-sanitizer 2024.1.1 (driver 595, Turing) intermittently
# crashes while reporting a device error. That is the tool's fault, and 2026.1.1
# doesn't do it. "kernel" mode once kept 2024.1.1 reporting through some of those
# crashes; it does no harm, so it stays.
# 2026.1.1's default for --report-api-errors is "extended", which also reports CUDA's
# log messages, such as torch's internal kernels that aren't found in their library.
# "explicit" reports only failed explicit API calls, which was 2024.1.1's default.
COMMON_OPTIONS = [
    "--error-exitcode", "1",
    "--destroy-on-device-error", "kernel",
    "--report-api-errors", "explicit",
]  # fmt: skip

# What each canary does, and the output that proves the tool caught it.
CANARIES = {
    "out-of-bounds": ("memcheck", "Invalid __global__ read"),
    "uninitialized": ("initcheck", "Uninitialized __global__ memory read"),
}
# How compute-sanitizer 2024.1.1 crashes on the server's driver while reporting.
# A canary that crashes this way didn't report, so it fails; the explanation
# below is printed with it, since the cause is the tool and not Flint.
REPORT_CRASH = (
    "Internal Sanitizer Error: The Sanitizer failed to handle a hardware exception"
)
CRASH_NOTE = (
    "this is the tool's crash, not Flint's: compute-sanitizer 2024.1.1 does this on the "
    "server's driver; the server's precheck runs 2026.1.1 from the sancheck environment"
)
# Names the compute-sanitizer to run, instead of the one on PATH.
SANITIZER_VARIABLE = "FLINT_COMPUTE_SANITIZER"


def canary_child(name):
    sys.path.insert(0, str(ROOT / "tests"))
    import torch

    import flint_lib

    flint = flint_lib.Flint()
    n = 4096
    if name == "out-of-bounds":
        # memcheck's padding keeps the two allocations apart, so the over-long
        # range neither overlaps dst nor reaches another live allocation.
        src = torch.ones(n, dtype=torch.uint8, device="cuda")
        dst = torch.empty(n + 512, dtype=torch.uint8, device="cuda")
        flint.copy(dst, src, n + 512)  # reads 512 bytes past the end of src
    else:
        src = torch.empty(n, dtype=torch.uint8, device="cuda")  # never written
        dst = torch.zeros(n, dtype=torch.uint8, device="cuda")
        flint.copy(dst, src, n)
    torch.cuda.synchronize()


def find_sanitizer():
    """The compute-sanitizer to run: the file FLINT_COMPUTE_SANITIZER names, else the one on
    PATH. Returns its path, or None after printing why there is none."""
    named = os.environ.get(SANITIZER_VARIABLE)
    if named:
        if not (os.path.isfile(named) and os.access(named, os.X_OK)):
            print(f"{SANITIZER_VARIABLE} is {named!r}, which is not an existing executable file")
            return None
        return named
    found = shutil.which("compute-sanitizer")
    if found is None:
        print(
            f"compute-sanitizer not found on PATH and {SANITIZER_VARIABLE} is not set "
            "(enter the dev shell, or install the CUDA toolkit)"
        )
    return found


def tool_version(sanitizer):
    """The last line of the tool's --version output, or a note that it didn't answer. The run
    doesn't depend on it, so a failure here is not one."""
    try:
        result = subprocess.run(
            [sanitizer, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "version unknown"
    lines = result.stdout.strip().splitlines()
    return lines[-1].strip() if lines else "version unknown"


def announce_sanitizer(sanitizer):
    # Printed once, so a log shows which tool ran.
    print(f"compute-sanitizer: {sanitizer} ({tool_version(sanitizer)})", flush=True)


def sanitizer_command(sanitizer, tool, child):
    options = FLINT_KERNELS if tool in FILTERED_TOOLS else []
    if tool == "memcheck":
        options = options + MEMCHECK_PADDING
    return [sanitizer, "--tool", tool, *COMMON_OPTIONS, *options, *child]


def canary_needed(tool, tools):
    # The memcheck canary also proves the kernel filter that racecheck and
    # synccheck share, so it runs whenever any filtered tool does.
    if tool == "memcheck":
        return any(t in FILTERED_TOOLS for t in tools)
    return tool in tools


def run(command):
    # FLINT_SANITIZER tells tests that timing is distorted under the tools.
    environment = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1", FLINT_SANITIZER="1")
    start = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    return result.returncode, result.stdout, time.monotonic() - start


def error_count(output):
    # racecheck ends with "RACECHECK SUMMARY: ... (N errors, M warnings)" instead.
    # Past the print limit, a second "ERROR SUMMARY: N errors were not printed"
    # line follows the total; skip it.
    counts = re.findall(
        r"(?:ERROR SUMMARY: |RACECHECK SUMMARY: .*\()(\d+) errors?\b(?! were not printed)",
        output,
    )
    return int(counts[-1]) if counts else None


def warning_count(output):
    # Only racecheck counts warnings, in its summary's "(N errors, M warnings)".
    counts = re.findall(r"RACECHECK SUMMARY: .*\(\d+ errors?, (\d+) warnings?\)", output)
    return int(counts[-1]) if counts else None


def report(name, passed, errors, seconds, output, warnings=None, tool=None):
    shown = "-" if errors is None else errors
    counts = f"errors={shown}"
    if tool == "racecheck":
        counts += f" warnings={'-' if warnings is None else warnings}"
    print(f"{name:<24} {'PASS' if passed else 'FAIL'}  {counts}  {seconds:.1f}s", flush=True)
    if not passed:
        print("\n".join(output.splitlines()[-40:]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tools", default=",".join(TOOLS))
    parser.add_argument("--canary", choices=sorted(CANARIES), help=argparse.SUPPRESS)
    args, pytest_args = parser.parse_known_args()

    if args.canary:
        canary_child(args.canary)
        return 0

    tools = args.tools.split(",")
    unknown = [tool for tool in tools if tool not in TOOLS]
    if unknown:
        parser.error(f"unknown tools {unknown}; choose from {TOOLS}")
    sanitizer = find_sanitizer()
    if sanitizer is None:
        return 1
    announce_sanitizer(sanitizer)

    failed = False
    for tool in tools:
        pytest_command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *pytest_args]
        code, output, seconds = run(sanitizer_command(sanitizer, tool, pytest_command))
        errors = error_count(output)
        # racecheck's summary counts warnings apart from errors; either fails.
        warnings = warning_count(output) if tool == "racecheck" else 0
        passed = code == 0 and errors == 0 and warnings == 0
        failed |= not passed
        report(f"{tool} (suite)", passed, errors, seconds, output, warnings, tool)

    script = str(Path(__file__).resolve())
    for name, (tool, message) in CANARIES.items():
        if not canary_needed(tool, tools):
            continue
        child = [sys.executable, script, "--canary", name]
        code, output, seconds = run(sanitizer_command(sanitizer, tool, child))
        caught = code != 0 and message in output
        failed |= not caught
        report(f"{tool} canary ({name})", caught, error_count(output), seconds, output)
        if not caught and REPORT_CRASH in output:
            print(CRASH_NOTE, flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
