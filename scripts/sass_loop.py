"""A static profile of Flint's GEMM main loops, from cuobjdump's SASS.

Usage: python scripts/sass_loop.py [--library PATH] [--out FILE]
       python scripts/sass_loop.py --compare OTHER.json [--library PATH] [--out FILE]

For each GEMM kernel and each of sm_75 and sm_89, finds the main loop (the
largest backward branch with HMMA in its range) and counts its instructions by
class, its stall cycles, its scoreboard waits, and how soon each tile load is
waited on (the load-to-use distance). Prints a table and writes the numbers as
JSON (schema flint-sass-loops/1), so the schedule can be committed beside the
benchmark logs and checked against another build's. The sm_75 GPU has no
performance counters, so this static schedule is part of the evidence for why a
loop is slower or faster than cuBLAS's.

--compare reports every difference in the per-kernel numbers between this run's
library and a JSON written earlier (for example, to check that the server's
Linux build compiles to the same SASS as the laptop's).

Control words (Volta and later): cuobjdump prints each instruction's second
64-bit word after it. Bits 41-44 are the stall count, 45 the yield flag, 46-48
the scoreboard barrier the instruction sets (7 for none), 49-51 the barrier it
holds its source registers on (7 for none), 52-57 the mask of barriers it waits
for, and 58-61 the operand-reuse flags.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_NAME = "flint.dll" if sys.platform == "win32" else "libflint.so"
ARCHS = ["sm_75", "sm_89"]
SCHEMA = "flint-sass-loops/1"

# Each kernel with an HMMA main loop, by the name that is length-prefixed in its
# mangled symbol. (gemm_tn_reduce has no HMMA loop.)
KERNELS = [
    "gemm_nt",
    "gemm_nn",
    "gemm_tn",
    "gemm_nt_bias",
    "gemm_nt_bias_gelu",
    "gemm_nn_dgelu",
    "gemm_nt_bias_residual",
]

# An instruction's line in cuobjdump's output ends with its 64-bit encoding in
# a comment, and the next line holds only a second comment: its control word
# (stall counts, barriers and so on, which the scheduler fixes at compile time).
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]{4})\*/\s+(.*?)\s*/\* (0x[0-9a-f]+) \*/\s*$")
CONTROL_WORD = re.compile(r"^\s*/\* (0x[0-9a-f]+) \*/\s*$")
BRANCH = re.compile(r"\bBRA(?:\.\w+)*\s+(0x[0-9a-f]+)")

NO_BARRIER = 7


# -- Parsing cuobjdump's output ----------------------------------------------


def sass_functions(library, arch):
    """Maps each kernel's mangled name to its SASS for one architecture."""
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump not found on PATH; it ships with the CUDA toolkit")
    result = subprocess.run(
        [cuobjdump, "-sass", "-arch", arch, str(library)],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"cuobjdump failed: {result.stderr.strip()}")
    functions = {}
    for section in result.stdout.split("Function : ")[1:]:
        name, _, body = section.partition("\n")
        functions[name.strip()] = body
    return functions


def instructions(body):
    """A kernel's SASS as (address, text, encoding, control word) tuples."""
    lines = body.splitlines()
    result = []
    for line, next_line in zip(lines, lines[1:] + [""]):
        match = INSTRUCTION.match(line)
        if match is None:
            continue
        control = CONTROL_WORD.match(next_line)
        assert control is not None, f"no control word after: {line.strip()}"
        result.append((int(match.group(1), 16), match.group(2), match.group(3), control.group(1)))
    return result


def main_loop(body):
    """The kernel's main loop: the backward branch (to a lower address) whose
    range, from its target to the branch, holds HMMA, and is the largest such
    range. Returns the loop's instructions and its branch's address."""
    code = instructions(body)
    best = None
    for address, text, _, _ in code:
        match = BRANCH.search(text)
        if match is None or int(match.group(1), 16) >= address:
            continue
        target = int(match.group(1), 16)
        loop = [item for item in code if target <= item[0] <= address]
        if any("HMMA" in item[1] for item in loop):
            if best is None or address - target > best[1] - best[0]:
                best = (target, address, loop)
    assert best is not None, "no loop with HMMA in it"
    return best[2], best[1]


# -- Decoding ----------------------------------------------------------------


def decode_control(control):
    """A control word (hex string or int) as its fields. The barriers are None
    when the instruction sets or holds none."""
    word = int(control, 16) if isinstance(control, str) else control
    write = (word >> 46) & 0x7
    read = (word >> 49) & 0x7
    return {
        "stall": (word >> 41) & 0xF,
        "yield": (word >> 45) & 0x1,
        "write_barrier": None if write == NO_BARRIER else write,
        "read_barrier": None if read == NO_BARRIER else read,
        "wait_mask": (word >> 52) & 0x3F,
        "reuse": (word >> 58) & 0xF,
    }


PREDICATE = re.compile(r"^@!?U?P(?:\d|T)\s+")


def opcode(text):
    """The instruction's opcode with its modifiers (HMMA.1688.F32), without the
    guard predicate."""
    return PREDICATE.sub("", text.strip()).split()[0].rstrip(";")


# Classes of the instructions that aren't named after themselves.
INT_ADDR = (
    "IMAD", "IADD3", "IADD", "LEA", "ISETP", "SHF", "LOP3", "LOP", "IABS", "SEL",
    "MOV", "PRMT", "IMNMX", "ICMP", "SHL", "SHR", "XMAD", "P2R", "R2P", "FLO", "POPC",
)  # fmt: skip
BRANCHES = ("BRA", "BRX", "JMP", "CALL", "RET", "EXIT")
SPECIAL = ("S2R", "S2UR", "CS2R")

CLASSES = [
    "HMMA", "LDSM", "LDG", "STS", "LDS", "BAR", "INT_ADDR", "UNIFORM_SPECIAL", "BRANCH", "OTHER",
]  # fmt: skip


def classify(text):
    """The class (one of CLASSES) of an instruction's text."""
    base = opcode(text).split(".")[0]
    if base in ("HMMA", "LDSM", "LDG", "STS", "LDS", "BAR"):
        return base
    if base.startswith("U") or base in SPECIAL:
        return "UNIFORM_SPECIAL"
    if base in BRANCHES:
        return "BRANCH"
    if base in INT_ADDR:
        return "INT_ADDR"
    return "OTHER"


# -- Statistics --------------------------------------------------------------


def load_to_use(loop):
    """For each LDG and LDSM in the loop (as (text, control word) pairs), the
    first later instruction, in program order and without wrapping around the
    loop, whose wait mask includes the barrier the load sets.

    The distance is the waiter's index minus the load's. The static cycles are
    the sum of the stall counts from the load up to, not including, the waiter:
    the cycles the schedule gives the load before its first use (the load's own
    stall count included). When no instruction waits before the loop's end, the
    waiter is None, the distance is the loop's length minus the load's index
    (as if the waiter sat just past the end) and the cycles are the stalls to
    the end. A load that sets no barrier has barrier None and no waiter."""
    fields = [decode_control(control) for _, control in loop]
    result = []
    for index, (text, _) in enumerate(loop):
        cls = classify(text)
        if cls not in ("LDG", "LDSM"):
            continue
        barrier = fields[index]["write_barrier"]
        waiter = None
        if barrier is not None:
            for later in range(index + 1, len(loop)):
                if fields[later]["wait_mask"] >> barrier & 1:
                    waiter = later
                    break
        end = len(loop) if waiter is None else waiter
        result.append({
            "index": index,
            "opcode": opcode(text),
            "barrier": barrier,
            "waiter_index": waiter,
            "waiter_opcode": None if waiter is None else opcode(loop[waiter][0]),
            "distance": end - index,
            "static_cycles": sum(f["stall"] for f in fields[index:end]),
        })
    return result


def phases(loop):
    """The loop split at its BAR instructions: for the stretch before the first,
    between each pair and after the last, the index range and the number of
    LDG, STS, LDSM and HMMA in it, which shows the loop's structure (loads,
    barrier, stores, barrier, math)."""
    cuts = [i for i, (text, _) in enumerate(loop) if classify(text) == "BAR"]
    bounds = [0] + [c + 1 for c in cuts] + [len(loop)]
    result = []
    for start, end in zip(bounds, bounds[1:]):
        if start >= end:
            continue
        classes = [classify(text) for text, _ in loop[start:end]]
        result.append({
            "first": start,
            "last": end - 1,
            **{name.lower(): classes.count(name) for name in ("LDG", "STS", "LDSM", "HMMA")},
        })
    return result


def loop_stats(loop):
    """The statistics of one main loop, given as (text, control word) pairs."""
    fields = [decode_control(control) for _, control in loop]
    classes = [classify(text) for text, _ in loop]
    opcodes = [opcode(text) for text, _ in loop]
    counts = {name: classes.count(name) for name in CLASSES}

    def variants(name):
        found = {}
        for cls, op in zip(classes, opcodes):
            if cls == name:
                found[op] = found.get(op, 0) + 1
        return dict(sorted(found.items()))

    hmma = counts["HMMA"]
    waiting = [i for i, f in enumerate(fields) if f["wait_mask"]]
    positions = {name: [i for i, cls in enumerate(classes) if cls == name]
                 for name in ("BAR", "LDG", "STS", "LDSM")}
    return {
        "length": len(loop),
        "classes": counts,
        "variants": {name: variants(name) for name in ("LDSM", "LDG", "STS", "OTHER")},
        "instr_per_hmma": len(loop) / hmma if hmma else None,
        "non_hmma_per_hmma": (len(loop) - hmma) / hmma if hmma else None,
        "stall_sum": sum(f["stall"] for f in fields),
        "yield_count": sum(f["yield"] for f in fields),
        "waits": len(waiting),
        "hmma_waits": sum(classes[i] == "HMMA" for i in waiting),
        "load_to_use": load_to_use(loop),
        "structure": {
            "bar": positions["BAR"],
            "ldg": positions["LDG"],
            "sts": positions["STS"],
            "ldsm": positions["LDSM"],
            "phases": phases(loop),
        },
    }


def normalize_symbol(symbol):
    """The mangled name with the anonymous namespace's name (length-prefixed
    _GLOBAL__N__ plus a hash, which changes between builds) replaced by <anon>."""
    match = re.match(r"(_ZN5flint)(\d+)(_GLOBAL__N_.*)", symbol)
    if match is None:
        return symbol
    length = int(match.group(2))
    return match.group(1) + "<anon>" + match.group(3)[length:]


def kernel_symbol(functions, kernel):
    """The (mangled name, SASS) of the kernel: the one whose symbol holds the
    length-prefixed name, such as 7gemm_ntE."""
    marker = f"{len(kernel)}{kernel}E"
    found = [(name, body) for name, body in functions.items() if marker in name]
    if len(found) != 1:
        raise RuntimeError(f"{kernel}: expected one symbol, found {sorted(n for n, _ in found)}")
    return found[0]


def arch_stats(library, arch):
    """Maps each kernel's short name to its mangled name and loop statistics."""
    functions = sass_functions(library, arch)
    result = {}
    for kernel in KERNELS:
        name, body = kernel_symbol(functions, kernel)
        loop, _ = main_loop(body)
        result[kernel] = {
            "mangled": normalize_symbol(name),
            "loop": loop_stats([(text, control) for _, text, _, control in loop]),
        }
    return result


# -- The report --------------------------------------------------------------


def run(command, cwd=ROOT):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=cwd)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def cuda_version():
    output = run(["cuobjdump", "--version"])
    if output is None:
        return "unknown"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown"


def git_state(repo=ROOT):
    sha = run(["git", "rev-parse", "HEAD"], repo)
    status = run(["git", "status", "--porcelain"], repo)
    return {
        "sha": sha.strip() if sha else "unknown",
        "dirty": bool(status.strip()) if status is not None else True,
    }


def default_library():
    override = os.environ.get("FLINT_LIB")
    return Path(override) if override else ROOT / "build" / "release" / "bin" / LIBRARY_NAME


def report(library, repo=ROOT):
    library = Path(library)
    if not library.is_file():
        raise RuntimeError(
            f"{library} not found; build it with "
            "`cmake --preset release && cmake --build --preset release`"
        )
    return {
        "schema": SCHEMA,
        "cuda": cuda_version(),
        "git": git_state(repo),
        "library": library.name,
        "archs": {arch: arch_stats(library, arch) for arch in ARCHS},
    }


def min_ldg_cycles(loop):
    cycles = [item["static_cycles"] for item in loop["load_to_use"] if item["opcode"].startswith("LDG")]
    return min(cycles) if cycles else None


def table(data):
    header = ["kernel", "arch", "len", "HMMA", "LDSM", "LDG", "STS", "BAR", "int/addr",
              "instr/HMMA", "stall", "waits (HMMA)", "min LDG->use"]
    rows = []
    for arch, kernels in data["archs"].items():
        for kernel, entry in kernels.items():
            loop = entry["loop"]
            c = loop["classes"]
            cycles = min_ldg_cycles(loop)
            rows.append([
                kernel, arch, str(loop["length"]), str(c["HMMA"]), str(c["LDSM"]), str(c["LDG"]),
                str(c["STS"]), str(c["BAR"]), str(c["INT_ADDR"]), f"{loop['instr_per_hmma']:.2f}",
                str(loop["stall_sum"]), f"{loop['waits']} ({loop['hmma_waits']})",
                "-" if cycles is None else f"{cycles} cyc",
            ])
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    lines = []
    for row in [header] + rows:
        lines.append("  ".join(cell.ljust(width) if i < 2 else cell.rjust(width)
                               for i, (cell, width) in enumerate(zip(row, widths))).rstrip())
    return "\n".join(lines)


# -- Comparing two reports ---------------------------------------------------


def flatten(value, prefix=""):
    """A nested dict/list as {path: leaf}."""
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return {prefix: value}
    flat = {}
    for key, child in items:
        flat.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    if not flat:
        flat[prefix] = value
    return flat


def compare(left, right):
    """The differences between two reports' per-kernel numbers, as lines (empty
    when the same). The git state, library file and CUDA version aren't
    compared; main() prints them."""
    lines = []
    for arch in sorted(set(left["archs"]) | set(right["archs"])):
        a, b = left["archs"].get(arch), right["archs"].get(arch)
        if a is None or b is None:
            lines.append(f"{arch}: only in {'this run' if b is None else 'the other file'}")
            continue
        for kernel in sorted(set(a) | set(b)):
            if kernel not in a or kernel not in b:
                lines.append(f"{arch} {kernel}: only in {'this run' if kernel in a else 'the other file'}")
                continue
            flat_a, flat_b = flatten(a[kernel]), flatten(b[kernel])
            for path in sorted(set(flat_a) | set(flat_b)):
                if flat_a.get(path, "<missing>") != flat_b.get(path, "<missing>"):
                    lines.append(
                        f"{arch} {kernel} {path}: {flat_a.get(path, '<missing>')} "
                        f"!= {flat_b.get(path, '<missing>')}"
                    )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--library", type=Path, default=default_library(),
                        help="flint.dll or libflint.so (default: the release build, or FLINT_LIB)")
    parser.add_argument("--out", type=Path, help="write the JSON here")
    parser.add_argument("--compare", type=Path, metavar="OTHER.json",
                        help="report differences in the per-kernel numbers against this JSON")
    parser.add_argument("--repo", type=Path, default=ROOT,
                        help="the checkout the library was built from, whose commit the JSON "
                             "records (default: this one)")
    args = parser.parse_args()
    try:
        data = report(args.library, args.repo)
    except RuntimeError as error:
        sys.exit(str(error))
    print(table(data))
    if args.out:
        args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.compare:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        if other.get("schema") != SCHEMA:
            sys.exit(f"{args.compare}: not a {SCHEMA} file")
        print()
        for name, mine, theirs in (
            ("CUDA", data["cuda"], other["cuda"]),
            ("commit", data["git"]["sha"], other["git"]["sha"]),
            ("library", data["library"], other["library"]),
        ):
            print(f"{name}: {mine} vs {theirs}")
        differences = compare(data, other)
        if differences:
            print(f"{len(differences)} differences:")
            print("\n".join(differences))
            sys.exit(1)
        print("per-kernel numbers are identical")


if __name__ == "__main__":
    main()
