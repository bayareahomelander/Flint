"""Compares every kernel's SASS between two builds of Flint.

Usage: python scripts/sass_diff.py dump OUT.json [--repo PATH]
       python scripts/sass_diff.py compare BEFORE.json AFTER.json

dump records the SASS of the library and of flint_bench, from a checkout's
release build, for each of sm_75 and sm_89. It refuses a build older than its
sources, which would hide a change. Dump the last commit's build before editing
anything, and the new build afterwards.

compare lists the kernels that are new, gone or changed for each binary and
architecture. It exits with 1 if an existing kernel changed or went away, 0 if
none did, and 2 if a file can't be used.

Two kernels match only if all their instruction lines and control words do. A
control word holds an instruction's stall count and scoreboard barriers, so a
kernel can be rescheduled with every instruction line the same, and on sm_75
small changes to a loop's schedule have cost 10%. Only the anonymous namespace's
name in a kernel's symbol is normalized, since its hash changes between builds.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sass_loop

ROOT = sass_loop.ROOT
SCHEMA = "flint-sass-dump/1"
BENCH_NAME = "flint_bench.exe" if sys.platform == "win32" else "flint_bench"

# A binary's name in a dump, its file, and the folders of its sources.
BINARIES = {
    "library": (sass_loop.LIBRARY_NAME, ("src", "include")),
    "bench": (BENCH_NAME, ("bench",)),
}

# An instruction's line, by its address in hex (five digits past 64 KiB), and
# the control word on the line after it. Everything else in cuobjdump's output
# describes the file, and names local paths.
SASS_LINE = re.compile(r"^\s*(/\*[0-9a-f]+\*/\s|/\* 0x[0-9a-f]+ \*/\s*$)")

# The anonymous namespace's name, after its length in the mangled symbol.
ANONYMOUS = re.compile(r"(\d+)_GLOBAL__N_")


# -- Dumping -----------------------------------------------------------------


def normalize_symbol(symbol):
    """The mangled name with each anonymous namespace's name replaced by <anon>."""
    parts = []
    while (match := ANONYMOUS.search(symbol)) is not None:
        parts.append(symbol[: match.start()] + "<anon>")
        symbol = symbol[match.end(1) + int(match.group(1)) :]
    return "".join(parts) + symbol


def sass_lines(body):
    """A kernel's instruction lines and control words, without their padding."""
    return [line.strip() for line in body.splitlines() if SASS_LINE.match(line)]


def stale_sources(path, repo, folders):
    """The sources under the repo's folders that are newer than the binary."""
    built = path.stat().st_mtime
    return sorted(
        str(source.relative_to(repo))
        for folder in folders
        for source in (repo / folder).rglob("*")
        if source.is_file() and source.stat().st_mtime > built
    )


def binary_kernels(path):
    """Maps each architecture to the binary's kernels, by normalized symbol, as
    lists of SASS lines."""
    result = {}
    for arch in sass_loop.ARCHS:
        kernels = {}
        for name, body in sass_loop.sass_functions(path, arch).items():
            symbol = normalize_symbol(name)
            if symbol in kernels:
                raise RuntimeError(f"{path.name} {arch}: two kernels normalize to {symbol}")
            kernels[symbol] = sass_lines(body)
        if not kernels or not all(kernels.values()):
            raise RuntimeError(f"{path.name} {arch}: found no SASS")
        result[arch] = kernels
    return result


def dump(repo):
    """The dump of a checkout's release build."""
    binaries = {}
    for label, (name, folders) in BINARIES.items():
        path = repo / "build" / "release" / "bin" / name
        if not path.is_file():
            raise RuntimeError(
                f"{path} not found; build it with "
                "`cmake --preset release && cmake --build --preset release`"
            )
        stale = stale_sources(path, repo, folders)
        if stale:
            raise RuntimeError(f"{name} is older than {stale[0]}; build it first")
        binaries[label] = binary_kernels(path)
    return {
        "schema": SCHEMA,
        "cuda": sass_loop.cuda_version(),
        "git": sass_loop.git_state(repo),
        "binaries": binaries,
    }


# -- Comparing ---------------------------------------------------------------


def compare(before, after):
    """For each binary and architecture in either dump, as "binary arch": its
    kernel count after, and the sorted symbols that are new, gone and changed."""
    result = {}
    for binary in sorted(set(before) | set(after)):
        old_arches, new_arches = before.get(binary, {}), after.get(binary, {})
        for arch in sorted(set(old_arches) | set(new_arches)):
            old, new = old_arches.get(arch, {}), new_arches.get(arch, {})
            result[f"{binary} {arch}"] = {
                "kernels": len(new),
                "new": sorted(set(new) - set(old)),
                "gone": sorted(set(old) - set(new)),
                "changed": sorted(k for k in set(old) & set(new) if old[k] != new[k]),
            }
    return result


def describe(data):
    git = data["git"]
    return f"{git['sha']} ({'dirty' if git['dirty'] else 'clean'}), CUDA {data['cuda']}"


def load(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"{path}: {error}") from error
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise RuntimeError(f"{path}: not a {SCHEMA} file")
    return data


def report(before, after):
    """Prints the comparison and returns whether an existing kernel changed or
    went away."""
    print(f"before: {describe(before)}")
    print(f"after:  {describe(after)}")
    if before["git"]["dirty"]:
        print("warning: before was dumped from a dirty tree, so it may not be the last commit's")
    found = False
    for where, entry in compare(before["binaries"], after["binaries"]).items():
        print(
            f"{where}: {entry['kernels']} kernels, {len(entry['new'])} new, "
            f"{len(entry['gone'])} gone, {len(entry['changed'])} changed"
        )
        for kind in ("new", "gone", "changed"):
            for symbol in entry[kind]:
                print(f"  {kind} {symbol}")
        found = found or bool(entry["gone"] or entry["changed"])
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    dumping = commands.add_parser("dump", help="record a build's SASS")
    dumping.add_argument("out", type=Path, help="the JSON to write")
    dumping.add_argument("--repo", type=Path, default=ROOT,
                         help="the checkout whose release build to read (default: this one)")
    comparing = commands.add_parser("compare", help="compare two dumps")
    comparing.add_argument("before", type=Path)
    comparing.add_argument("after", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "dump":
            data = dump(args.repo.resolve())
            args.out.write_text(json.dumps(data) + "\n", encoding="utf-8", newline="\n")
            counts = {binary: len(data["binaries"][binary][sass_loop.ARCHS[0]])
                      for binary in BINARIES}
            print(f"{args.out}: {describe(data)}, kernels per architecture: {counts}")
            return 0
        return 1 if report(load(args.before), load(args.after)) else 0
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
