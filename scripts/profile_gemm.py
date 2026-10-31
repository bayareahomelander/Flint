"""Profiles flint_bench gemm cases with Nsight Compute, one JSON per case.

Usage: python scripts/profile_gemm.py --layout {nt,nn,tn} --shapes MxNxK[,MxNxK...]
           --out-dir DIR [--reports-dir DIR] [--exe PATH] [--ncu PATH]

For every shape it runs flint_bench under ncu three times plus once bare:
a bare 1-round run (for the device, git state, and the bench's own ratio), a
cheap ncu run that lists the kernels each side launches and in what order, a
"flint" pass (only kernels in namespace flint, `--set full`), and a "baseline"
pass (every other kernel, `--set full` minus SourceCounters, so no instruction
data of a baseline kernel is collected). Each pass profiles exactly one launch
of every distinct kernel of its side, never that kernel's first launch. The
.ncu-rep reports go to --reports-dir and stay local: they record host, user,
paths, and, because ncu embeds each profiled module's cubin in a report, the
cuBLAS module's SASS too. Only the JSON is committable. It holds the METRICS
below for each kernel, and for Flint's kernels the per-instruction SASS with
its stall samples (zero stall counts left out), under the schema flint-ncu/1.
ncu runs with its defaults, clocks locked to base and caches flushed, so its
durations are not the benchmark's. The laptop doesn't always hold the base
clock, so the summary compares the sides by elapsed SM cycles as well as time.
"""

import argparse
import copy
import csv
import getpass
import json
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCHEMA = "flint-ncu/1"
CLOCK_CONTROL = "base"  # ncu's defaults, passed explicitly so the JSON records them
CACHE_CONTROL = "all"
SIZE_LIMIT = 1_000_000  # bytes per case; zero-count SASS rows go when that saves it

FLINT_FILTER = "regex:^flint::"
BASELINE_FILTER = "regex:^(?!flint::)"

# `ncu --list-sets` on 2024.1.1 shows the full set as these sections plus
# SourceCounters, which is the one that collects per-instruction data.
BASELINE_SECTIONS = [
    "ComputeWorkloadAnalysis",
    "InstructionStats",
    "LaunchStats",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Chart",
    "MemoryWorkloadAnalysis_Tables",
    "NumaAffinity",
    "Nvlink_Tables",
    "Nvlink_Topology",
    "Occupancy",
    "PmSampling",
    "SchedulerStats",
    "SpeedOfLight",
    "SpeedOfLight_RooflineChart",
    "WarpStateStats",
    "WorkloadDistribution",
]

TENSOR_ACTIVE = "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"

# Metric names as ncu 2024.1.1 reports them on sm_89. A name missing from a raw
# page is an error. The stall reasons are added by pattern, not listed here.
METRICS = [
    # time and clocks
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg.per_second",
    "sm__cycles_elapsed.max",
    "sm__cycles_active.avg",
    # launch and occupancy
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__block_size",
    "launch__grid_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "launch__waves_per_multiprocessor",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # tensor pipe and issue
    TENSOR_ACTIVE,
    "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.per_cycle_active",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    # memory
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__inst_executed_op_ldsm.sum",
    # throughput summaries
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
]
STALL_PATTERN = re.compile(r"smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio")
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"
SELECTED = STALL_PREFIX + "selected" + STALL_SUFFIX  # issuing, not stalled: left out of the summary

# The columns of the source page kept per SASS instruction, and their JSON keys.
SOURCE_COLUMNS = {
    "Warp Stall Sampling (All Samples)": "samples",
    "Warp Stall Sampling (Not-issued Samples)": "samples_not_issued",
    "Instructions Executed": "executed",
    "L1 Wavefronts Shared": "l1_wavefronts_shared",
    "L1 Wavefronts Shared Ideal": "l1_wavefronts_shared_ideal",
}

# A kernel that never touches shared memory has no wavefront columns; they count as 0.
OPTIONAL_COLUMNS = {"L1 Wavefronts Shared", "L1 Wavefronts Shared Ideal"}

NANOSECONDS = {"nsecond": 1.0, "usecond": 1e3, "msecond": 1e6, "second": 1e9}
CYCLES_PER_SECOND = {"cycle/second": 1.0, "cycle/msecond": 1e3, "cycle/usecond": 1e6, "cycle/nsecond": 1e9}


def resolve_ncu(given):
    """The ncu executable to run. ncu.bat is swapped for the ncu.exe behind it:
    cmd.exe would otherwise eat the ^ and | of a --kernel-name regex."""
    found = shutil.which(given or "ncu")
    if found is None:
        sys.exit("ncu not found: put it on PATH or pass --ncu")
    path = Path(found)
    if path.suffix.lower() in (".bat", ".cmd"):
        executables = sorted(path.parent.glob("target/windows-*/ncu.exe"))
        if not executables:
            sys.exit(f"{path} is a batch file and the ncu.exe behind it was not found")
        path = executables[-1]
    return path


def run(command):
    """Runs a command from the repo root and returns its stdout."""
    done = subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if done.returncode != 0:
        tail = "\n".join(done.stdout.splitlines()[-15:])
        sys.exit(f"{Path(str(command[0])).name} exited with {done.returncode}:\n{tail}")
    return done.stdout


def csv_rows(text):
    """The CSV rows of ncu's output, without its ==PROF== lines or the bench's own output."""
    lines = [line for line in text.splitlines() if line.startswith('"')]
    return list(csv.reader(lines))


def number(text):
    """A number from ncu's text (1,179,648 or 127.000000), or the text if it isn't one."""
    cleaned = text.replace(",", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        value = float(cleaned)
    except ValueError:
        return text
    return int(value) if value.is_integer() and abs(value) < 2**53 else value


def dimensions(text):
    """[256, 1, 1] from "(256, 1, 1)"."""
    return [int(part) for part in re.findall(r"\d+", text)]


def launch_sequence(text):
    """The kernel names of a --metrics --csv run, in launch order."""
    rows = csv_rows(text)
    if not rows or "Kernel Name" not in rows[0]:
        raise ValueError("no kernel table in ncu's output")
    column = rows[0].index("Kernel Name")
    return [row[column] for row in rows[1:] if row[0].isdigit()]


def choose_window(sequence):
    """(skip, count) selecting one launch of each distinct kernel in a side's launch sequence.

    The window holds each distinct name once, and none of its launches is the
    first of its name. ncu counts only the launches the side's filter matches,
    so --launch-skip/--launch-count are these numbers directly.
    """
    distinct = list(dict.fromkeys(sequence))
    if not distinct:
        raise ValueError("the side launched no kernels")
    first = {}
    for index, name in enumerate(sequence):
        first.setdefault(name, index)
    count = len(distinct)
    for skip in range(1, len(sequence) - count + 1):
        window = sequence[skip : skip + count]
        if len(set(window)) == count and all(
            skip + i > first[name] for i, name in enumerate(window)
        ):
            return skip, count
    raise ValueError(f"no window of {count} launches covers each of {distinct} once, after a first")


def parse_raw_page(text, metrics=METRICS):
    """Kernels from `ncu --page raw --csv`: name, block, grid, and the metrics with units.

    The last row of a name wins. Private columns (process, host, context,
    stream, device) are never read.
    """
    rows = csv_rows(text)
    if len(rows) < 3:
        raise ValueError("the raw page has no kernel rows")
    header, units = rows[0], rows[1]
    stalls = [name for name in header if STALL_PATTERN.fullmatch(name)]
    wanted = metrics + [name for name in stalls if name not in metrics]
    for name in wanted:
        if name not in header:
            raise ValueError(f"metric {name} is not in the raw page")
    column = {name: header.index(name) for name in wanted}
    kernels = {}
    for row in rows[2:]:
        fields = dict(zip(header, row))
        kernels[fields["Kernel Name"]] = {
            "name": fields["Kernel Name"],
            "block": dimensions(fields["Block Size"]),
            "grid": dimensions(fields["Grid Size"]),
            "metrics": {
                name: {
                    "value": number(row[column[name]]) if row[column[name]] else None,
                    "unit": units[column[name]],
                }
                for name in wanted
            },
        }
    return list(kernels.values())


def raw_kernel_names(text):
    """Every kernel row's name in a raw page, repeats included."""
    rows = csv_rows(text)
    column = rows[0].index("Kernel Name")
    return [row[column] for row in rows[2:]]


def parse_source_page(text):
    """SASS per kernel from `ncu --page source --print-source sass --csv`.

    Returns {kernel name: [instruction, ...]}, the last block of a name winning.
    Offsets are bytes from the kernel's first instruction, not GPU addresses.
    """
    blocks = {}
    name = None
    header = None
    instructions = None
    for row in csv_rows(text):
        if row[0] == "Kernel Name":
            name, header, instructions = row[1], None, []
            blocks[name] = instructions
        elif row[0] == "Address":
            header = row
            for column in SOURCE_COLUMNS:
                if column not in header and column not in OPTIONAL_COLUMNS:
                    raise ValueError(f"column {column} is not in the source page")
        elif header is not None:
            fields = dict(zip(header, row))
            if instructions:
                base = instructions[0]["address"]
            else:
                base = int(fields["Address"], 0)
            instruction = {
                "address": int(fields["Address"], 0),
                "offset": int(fields["Address"], 0) - base,
                "sass": fields["Source"].strip(),
            }
            for column, key in SOURCE_COLUMNS.items():
                instruction[key] = counter(fields.get(column, "-"))
            instruction["stalls"] = {
                column: count
                for column in header
                if column.startswith("stall_")
                and not column.endswith("(Not Issued)")
                and (count := counter(fields[column])) != 0
            }
            instructions.append(instruction)
    for instructions in blocks.values():
        for instruction in instructions:
            del instruction["address"]
    return blocks


def counter(text):
    value = number(text)
    return value if isinstance(value, int) else 0  # "-" marks a column that doesn't apply


def is_idle(instruction):
    numbers = [instruction[key] for key in SOURCE_COLUMNS.values()]
    return not any(numbers) and not instruction["stalls"]


def duration_ns(kernel):
    metric = kernel["metrics"]["gpu__time_duration.sum"]
    return metric["value"] * NANOSECONDS[metric["unit"]]


def gpu_tag(device_name):
    """rtx-4050-laptop from "NVIDIA GeForce RTX 4050 Laptop GPU"."""
    words = re.findall(r"[a-z0-9]+", device_name.lower())
    return "-".join(word for word in words if word not in ("nvidia", "geforce", "gpu"))


def placeholder_command(command, ncu, exe, reports_dir):
    """A command as a list of strings with every path replaced by a placeholder."""
    reports = str(reports_dir)
    shown = []
    for part in command:
        part = str(part)
        if part == str(ncu):
            shown.append("<ncu>")
        elif part == str(exe):
            shown.append("<flint_bench>")
        elif part.startswith(reports):
            shown.append("<reports>/" + part[len(reports) :].lstrip("\\/").replace("\\", "/"))
        else:
            shown.append(part)
    return shown


def scrub_check(text):
    """Raises if text holds a path, the user name, or the host name."""
    patterns = [":\\", "C:/", "/home/", "/Users/", "Users\\", getpass.getuser(), socket.gethostname()]
    lowered = text.lower()
    for pattern in patterns:
        if pattern and pattern.lower() in lowered:
            raise ValueError(f"private text in the output: {pattern!r}")


def dump_json(document):
    """JSON with indent 1, each SASS instruction and each list of plain values on one line."""
    rows = []

    def inline(value):
        rows.append(json.dumps(value))
        return f"@@{len(rows) - 1}@@"

    def lift(node, key=None):
        if isinstance(node, dict):
            return {k: lift(v, k) for k, v in node.items()}
        if isinstance(node, list):
            if key == "source":
                return [inline(item) for item in node]
            if not any(isinstance(item, (dict, list)) for item in node):
                return inline(node)
            return [lift(item) for item in node]
        return node

    text = json.dumps(lift(document), indent=1)
    return re.sub(r'"@@(\d+)@@"', lambda m: rows[int(m.group(1))], text) + "\n"


def fit_document(document):
    """The document as text, without its all-zero SASS rows only if they alone push it past SIZE_LIMIT."""
    text = dump_json(document)
    if len(text.encode("utf-8")) <= SIZE_LIMIT:
        return text
    pruned = copy.deepcopy(document)
    for kernel in pruned["sides"]["flint"]:
        kernel["source"] = [i for i in kernel.get("source", []) if not is_idle(i)]
        kernel["zero_rows_omitted"] = True
    smaller = dump_json(pruned)
    return smaller if len(smaller.encode("utf-8")) <= SIZE_LIMIT else text


def sm_mhz(kernel):
    """The kernel's mean SM clock in MHz; ncu picks the rate's unit per value."""
    metric = kernel["metrics"]["sm__cycles_elapsed.avg.per_second"]
    return metric["value"] * CYCLES_PER_SECOND[metric["unit"]] / 1e6


def side_summary(kernels):
    return {
        "kernels": len(kernels),
        "total_duration_ns": sum(duration_ns(k) for k in kernels),
        "total_sm_cycles": sum(metric_value(k, "sm__cycles_elapsed.max") for k in kernels),
    }


def metric_value(kernel, name):
    return kernel["metrics"][name]["value"]


def format_summary(layout, shape, document):
    """The short human-readable report of one case."""
    lines = [f"{layout} {shape}"]
    for side in ("flint", "baseline"):
        for kernel in document["sides"][side]:
            stalls = sorted(
                (
                    (metric_value(kernel, name) or 0, name[len(STALL_PREFIX) : -len(STALL_SUFFIX)])
                    for name in kernel["metrics"]
                    if STALL_PATTERN.fullmatch(name) and name != SELECTED
                ),
                reverse=True,
            )[:5]
            lines.append(
                f"  {side:8} {kernel['name'][:56]}\n"
                f"           {duration_ns(kernel) / 1e3:.1f} us, SM {sm_mhz(kernel):.0f} MHz, "
                f"tensor {metric_value(kernel, TENSOR_ACTIVE):.1f}%, "
                f"issue {metric_value(kernel, 'smsp__issue_active.avg.pct_of_peak_sustained_active'):.1f}%, "
                f"warp cycles/issue {metric_value(kernel, 'smsp__average_warp_latency_per_inst_issued.ratio'):.1f}\n"
                f"           stalls: " + ", ".join(f"{name} {value:.2f}" for value, name in stalls)
            )
    summary = document["summary"]
    lines.append(
        f"  baseline / flint SM cycles: {summary['baseline_cycles_over_flint_cycles']:.3f}, "
        f"time: {summary['baseline_time_over_flint_time']:.3f} "
        f"(flint {summary['flint']['total_duration_ns'] / 1e3:.1f} us, "
        f"baseline {summary['baseline']['total_duration_ns'] / 1e3:.1f} us)"
    )
    return "\n".join(lines)


def bench_command(exe, layout, shape, out):
    return [
        exe, "gemm", "--allow-busy", "--layout", layout, "--shapes", shape,
        "--warmup", "0", "--rounds", "1", "--sample-ms", "1", "--out", out,
    ]  # fmt: skip


def pass_command(ncu, kernel_filter, selection, sections, report, bench):
    skip, count = selection
    return [
        ncu, "-k", kernel_filter, "--kernel-name-base", "demangled",
        "--clock-control", CLOCK_CONTROL, "--cache-control", CACHE_CONTROL,
        "--launch-skip", str(skip), "--launch-count", str(count),
        *sections, "-f", "--export", report, *bench,
    ]  # fmt: skip


def profile_case(args, ncu, exe, layout, shape):
    reports = Path(args.reports_dir).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    stem = reports / f"{layout}-{shape}"

    def bench_json(label):
        return f"{stem}-{label}.bench.json"

    # A bare run, for the device, the commit, and the bench's own ratio.
    run(bench_command(exe, layout, shape, bench_json("unprofiled")))
    bare = json.loads(Path(bench_json("unprofiled")).read_text(encoding="utf-8"))
    case = bare["cases"][0]

    # Which kernels each side launches, in order.
    discovery = run(
        [ncu, "--kernel-name-base", "demangled", "--metrics", "gpu__time_duration.sum", "--csv"]
        + bench_command(exe, layout, shape, bench_json("discovery"))
    )
    launched = launch_sequence(discovery)
    sequences = {
        "flint": [name for name in launched if name.startswith("flint::")],
        "baseline": [name for name in launched if not name.startswith("flint::")],
    }

    passes = {
        "flint": (FLINT_FILTER, ["--set", "full"]),
        "baseline": (BASELINE_FILTER, [a for s in BASELINE_SECTIONS for a in ("--section", s)]),
    }
    commands = {}
    kernels = {}
    for side, (kernel_filter, sections) in passes.items():
        selection = choose_window(sequences[side])
        report = f"{stem}-{side}.ncu-rep"
        command = pass_command(
            ncu, kernel_filter, selection, sections, report,
            bench_command(exe, layout, shape, bench_json(side)),
        )  # fmt: skip
        run(command)
        commands[side] = placeholder_command(command, ncu, exe, reports)
        raw = run([ncu, "--import", report, "--page", "raw", "--csv", "--print-kernel-base", "demangled"])
        expected = sorted(set(sequences[side]))
        if sorted(raw_kernel_names(raw)) != expected:
            sys.exit(
                f"{side} pass profiled {raw_kernel_names(raw)}, expected one launch of each of "
                f"{expected}; the launch order changed between runs, rerun"
            )
        kernels[side] = parse_raw_page(raw)
        if side == "flint":
            source = run(
                [ncu, "--import", report, "--page", "source", "--print-source", "sass", "--csv",
                 "--print-kernel-base", "demangled"]
            )  # fmt: skip
            blocks = parse_source_page(source)
            for kernel in kernels[side]:
                kernel["source"] = blocks[kernel["name"]]

    version = next(
        line[len("Version ") :] for line in run([ncu, "--version"]).splitlines() if line.startswith("Version ")
    )
    ratio = case["ratio"]
    document = {
        "schema": SCHEMA,
        "device": {key: bare["device"][key] for key in ("name", "compute_capability", "driver_version")},
        "ncu_version": version,
        "git": bare["git"],
        "layout": layout,
        "shape": shape,
        "clock_control": CLOCK_CONTROL,
        "cache_control": CACHE_CONTROL,
        "commands": commands,
        "bench_ratio": {
            "note": "flint_bench's flint/baseline throughput ratio from an unprofiled "
            "--warmup 0 --rounds 1 --sample-ms 1 run; informational only",
            "median": ratio["median"],
            "ok": case["ok"],
        },
        "sides": kernels,
    }
    # The laptop doesn't always hold ncu's base clock (one kernel can run at
    # 825 MHz and the other at 1,140), so elapsed SM cycles compare the sides
    # more fairly than durations do.
    flint, baseline = side_summary(kernels["flint"]), side_summary(kernels["baseline"])
    document["summary"] = {
        "flint": flint,
        "baseline": baseline,
        "baseline_cycles_over_flint_cycles": baseline["total_sm_cycles"] / flint["total_sm_cycles"],
        "baseline_time_over_flint_time": baseline["total_duration_ns"] / flint["total_duration_ns"],
    }

    text = fit_document(document)
    scrub_check(text)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ncu-{gpu_tag(bare['device']['name'])}-{layout}-{shape}.json"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(format_summary(layout, shape, document))
    print(f"  wrote {path.name} ({len(text.encode('utf-8')) / 1000:.0f} kB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layout", required=True, choices=["nt", "nn", "tn"])
    parser.add_argument("--shapes", required=True, help="comma-separated MxNxK, as flint_bench takes them")
    parser.add_argument("--out-dir", required=True, help="where the JSON files go")
    parser.add_argument("--reports-dir", default=str(ROOT / "build" / "profile-reports"))
    parser.add_argument("--exe", default=str(ROOT / "build" / "release" / "bin" / "flint_bench"))
    parser.add_argument("--ncu", default=None)
    args = parser.parse_args()
    shapes = args.shapes.split(",")
    for shape in shapes:
        if not re.fullmatch(r"\d+x\d+x\d+", shape):
            sys.exit(f"bad shape {shape!r}: expected MxNxK")
    exe = Path(args.exe)
    if not exe.exists() and exe.with_suffix(".exe").exists():
        exe = exe.with_suffix(".exe")
    if not exe.exists():
        sys.exit(f"{exe} not found: build the release preset first")
    ncu = resolve_ncu(args.ncu)
    for shape in shapes:
        profile_case(args, ncu, exe.resolve(), args.layout, shape)


if __name__ == "__main__":
    main()
