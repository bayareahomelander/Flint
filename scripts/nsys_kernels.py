"""Summarizes the GPU kernels in Nsight Systems cuda_gpu_trace CSV files.

Usage: python scripts/nsys_kernels.py --case LABEL --out FILE.json [--command CMD] CSV [CSV ...]

Each CSV comes from `nsys stats --report cuda_gpu_trace --format csv --output
PREFIX REPORT.nsys-rep`, for one traced flint_bench gemm case (one shape per
process). Kernel launches are grouped by name, grid, block, registers, and
shared memory, and each group gets its launch count and duration spread. A
group's side is "flint" (the library's kernels), "bench" (flint_bench's own
kernels), or "baseline" (cuBLAS and cuBLASLt). Shared memory is kept as nsys
reports it, in MB to three decimals (Flint's 20,480 bytes show as 0.020), so
it's good to about a kilobyte. With --bench-log, the case's flint_bench log
supplies the commit and the GPU. The JSON holds no ids or timestamps, and is
refused if it holds a path or the user or host name.
"""

import argparse
import csv
import getpass
import io
import json
import socket
import statistics
import sys
from pathlib import Path

SCHEMA = "flint-nsys/1"

# Accepted header spellings per field, lowercase, without the unit suffix
# ("StcSMem (MB)" and "StcSMem (KB)" both match "stcsmem").
ALIASES = {
    "name": ["name", "kernel name", "kernel"],
    "duration": ["duration", "dur"],
    "grid_x": ["grdx", "grid x", "gridx", "grid_x"],
    "grid_y": ["grdy", "grid y", "gridy", "grid_y"],
    "grid_z": ["grdz", "grid z", "gridz", "grid_z"],
    "block_x": ["blkx", "block x", "blockx", "block_x"],
    "block_y": ["blky", "block y", "blocky", "block_y"],
    "block_z": ["blkz", "block z", "blockz", "block_z"],
    "regs": ["reg/trd", "regs/thread", "registers per thread", "regs", "reg"],
    "static_smem": ["stcsmem", "static smem", "static shared memory", "staticsharedmemory"],
    "dynamic_smem": ["dymsmem", "dynamic smem", "dynamic shared memory", "dynamicsharedmemory"],
}
REQUIRED = ["name", "duration", "grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z"]
OPTIONAL = ["regs", "static_smem", "dynamic_smem"]

DURATION_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

# Markers of flint_bench's own kernels. They live in flint::microbench, so this
# is checked before the flint namespace.
BENCH_MARKERS = ("mma_m16n8k8", "flint_bench", "microbench")


def split_header(text):
    """Splits a header like "StcSMem (MB)" into ("stcsmem", "mb")."""
    text = text.strip().lower()
    unit = ""
    if "(" in text and text.endswith(")"):
        text, unit = text[: text.rindex("(")], text[text.rindex("(") + 1 : -1]
    return text.strip(), unit.strip()


def find_columns(headers):
    """Maps each field to (column index, unit), failing with the headers seen."""
    split = [split_header(h) for h in headers]
    found = {}
    for field in REQUIRED + OPTIONAL:
        for index, (base, _) in enumerate(split):
            if base in ALIASES[field]:
                found[field] = (index, split[index][1])
                break
    missing = [f for f in REQUIRED if f not in found]
    if missing:
        raise ValueError(
            f"missing required columns {missing}; headers seen: {[h.strip() for h in headers]}"
        )
    return found


def read_rows(text):
    """The header and data rows of a CSV that may have junk lines before the header."""
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        headers = next(csv.reader([line]))
        try:
            find_columns(headers)
        except ValueError:
            continue
        rows = list(csv.reader(io.StringIO("\n".join(lines[number + 1 :]))))
        return headers, [r for r in rows if r]
    # No line qualified: report the line that looks most like a header.
    candidates = [next(csv.reader([ln])) for ln in lines if ln.strip()]

    def matches(headers):
        names = [split_header(h)[0] for h in headers]
        return sum(any(n in ALIASES[f] for n in names) for f in REQUIRED + OPTIONAL)

    best = max(candidates, key=matches, default=[])
    find_columns(best)  # raises, naming the headers seen
    raise ValueError("no header row found")


def number(cell):
    cell = cell.strip().replace(",", "")
    return float(cell) if cell else None


def side_of(name):
    lowered = name.lower()
    if any(marker in lowered for marker in BENCH_MARKERS):
        return "bench"
    if "flint::" in name or "5flint" in name:
        return "flint"
    return "baseline"


def kernels(paths):
    texts = [Path(p).read_text(encoding="utf-8", errors="replace") for p in paths]
    return kernels_from_text(*texts)


def kernels_from_text(*texts):
    """Kernel launch records from CSV texts, plus the shared-memory unit as reported."""
    records = []
    units = {}
    for text in texts:
        headers, rows = read_rows(text)
        columns = find_columns(headers)
        scale = DURATION_NS.get(columns["duration"][1] or "ns", 1.0)
        for field in ("static_smem", "dynamic_smem"):
            unit = columns[field][1] if field in columns else "absent"
            if units.setdefault(field, unit) != unit:
                raise ValueError(f"{field} is in {units[field]} in one file and {unit} in another")
        for row in rows:

            def cell(field):
                if field not in columns or columns[field][0] >= len(row):
                    return None
                return number(row[columns[field][0]])

            grid = [cell(f) for f in ("grid_x", "grid_y", "grid_z")]
            if any(g is None for g in grid):
                continue  # a memcpy or memset row
            block = [cell(f) for f in ("block_x", "block_y", "block_z")]
            duration = cell("duration")
            if duration is None:
                continue
            records.append(
                {
                    "name": row[columns["name"][0]].strip(),
                    "us": duration * scale / 1e3,
                    "grid": [int(g) for g in grid],
                    "block": [int(b or 0) for b in block],
                    "regs": int(cell("regs") or 0),
                    "static_smem": cell("static_smem") or 0.0,
                    "dynamic_smem": cell("dynamic_smem") or 0.0,
                }
            )
    return records, units


def summarize(records):
    groups = {}
    for r in records:
        key = (r["name"], tuple(r["grid"]), tuple(r["block"]), r["regs"], r["static_smem"], r["dynamic_smem"])
        groups.setdefault(key, []).append(r["us"])
    out = []
    for (name, grid, block, regs, static, dynamic), times in groups.items():
        out.append(
            {
                "side": side_of(name),
                "name": name,
                "count": len(times),
                "median_us": statistics.median(times),
                "min_us": min(times),
                "max_us": max(times),
                "total_us": sum(times),
                "grid": list(grid),
                "block": list(block),
                "blocks": grid[0] * grid[1] * grid[2],
                "threads_per_block": block[0] * block[1] * block[2],
                "registers_per_thread": regs,
                "static_smem": static,
                "dynamic_smem": dynamic,
            }
        )
    order = {"flint": 0, "baseline": 1, "bench": 2}
    out.sort(key=lambda g: (order[g["side"]], -g["total_us"]))
    return out


def scrub_check(text):
    """Raises when text holds a path, the user name, or the host name."""
    lowered = text.lower()
    banned = [":\\", "c:/", "/home/", "/users/", "users\\"]
    for who in (getpass.getuser(), socket.gethostname()):
        if who:
            banned.append(who.lower())
    for item in banned:
        if item in lowered:
            raise ValueError(f"output would leak {item!r}; replace it with a placeholder")


def build(paths, case, command=None, bench_log=None, nsys_version=None):
    records, units = kernels(paths)
    if not records:
        raise ValueError("no kernel rows found in the CSV files")
    document = {
        "schema": SCHEMA,
        "case": case,
        "command": command,
        "nsys_version": nsys_version,
        "csv_files": len(paths),
        "smem_unit": units,
        "groups": summarize(records),
    }
    if bench_log is not None:
        document["git"] = bench_log["git"]
        document["device"] = {
            key: bench_log["device"][key] for key in ("name", "compute_capability", "driver_version")
        }
    return document


def table(result):
    lines = []
    for g in result["groups"]:
        lines.append(
            f"{g['side']:<8} x{g['count']:<4} med {g['median_us']:9.1f} us  "
            f"grid {'x'.join(map(str, g['grid']))} block {'x'.join(map(str, g['block']))} "
            f"regs {g['registers_per_thread']} smem {g['static_smem']}+{g['dynamic_smem']}  "
            f"{g['name'][:90]}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csvs", nargs="+", help="nsys cuda_gpu_trace CSV files")
    parser.add_argument("--case", required=True, help="the traced case's label, such as 1024x768x768")
    parser.add_argument("--out", required=True, help="JSON file to write")
    parser.add_argument("--command", help="the traced command, with paths replaced by placeholders")
    parser.add_argument("--bench-log", help="the traced run's flint_bench JSON, for its commit and GPU")
    parser.add_argument("--nsys-version", help="the version line of `nsys --version`")
    args = parser.parse_args()
    try:
        bench_log = None
        if args.bench_log:
            bench_log = json.loads(Path(args.bench_log).read_text(encoding="utf-8"))
        result = build(args.csvs, args.case, args.command, bench_log, args.nsys_version)
        text = json.dumps(result, indent=2) + "\n"
        scrub_check(text)
    except ValueError as error:
        sys.exit(f"error: {error}")
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(table(result))


if __name__ == "__main__":
    main()
