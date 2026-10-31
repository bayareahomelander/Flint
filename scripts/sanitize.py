"""Runs the test suite under NVIDIA compute-sanitizer, and proves it sees Flint's kernels.

Usage: python scripts/sanitize.py [--tools memcheck,racecheck,synccheck,initcheck]
       [pytest args...]

Each tool runs the whole suite and must report zero errors, and racecheck zero
warnings too. Two canaries then run deliberately broken calls and must be
caught: an out-of-bounds read under memcheck, and an uninitialized read under
initcheck. Unrecognized arguments go to pytest. Exits nonzero if any run fails.

On the 2080 Ti server, memcheck and initcheck intermittently crash while
reporting an error they caught, so a canary also counts as caught when its run
fails with that crash. Diagnose such errors on the laptop.
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
# On the server (compute-sanitizer 2024.1.1, driver 595, Turing), reporting a
# device error intermittently crashes the tool; "kernel" mode kept memcheck
# reporting through one such spell, though not through all of them.
COMMON_OPTIONS = ["--error-exitcode", "1", "--destroy-on-device-error", "kernel"]

# What each canary does, and the output that proves the tool caught it.
CANARIES = {
    "out-of-bounds": ("memcheck", "Invalid __global__ read"),
    "uninitialized": ("initcheck", "Uninitialized __global__ memory read"),
}
# How a report crashes on the server. A canary run that fails with it still
# proves the tool trapped on Flint's kernel: memcheck's padding keeps the
# over-read in mapped memory, and the initcheck canary's kernel is valid, so
# neither can fault on its own.
REPORT_CRASH = (
    "Internal Sanitizer Error: The Sanitizer failed to handle a hardware exception"
)


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
    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        print("compute-sanitizer not found on PATH (enter the dev shell, or install the CUDA toolkit)")
        return 1

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
        reported = message in output
        unreported = REPORT_CRASH in output
        caught = code != 0 and (reported or unreported)
        failed |= not caught
        label = f"{tool} canary ({name}{', unreported' if caught and not reported else ''})"
        report(label, caught, error_count(output), seconds, output)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
