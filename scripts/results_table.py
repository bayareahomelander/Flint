"""Prints Markdown result tables from flint_bench raw logs.

Usage: python scripts/results_table.py LOG.json [LOG.json ...]

Each log gets a line naming its GPU, driver, commit, and command, then one row
per case. Write-ups paste this output, so their numbers come straight from the
committed raw logs. Links use the paths as given, so run it from the directory
of the document that will hold the table.
"""

import argparse
import json
import sys
from pathlib import Path


def cuda_version(version):
    return f"{version // 1000}.{version % 1000 // 10}"


def percent(fraction):
    return f"{100 * fraction:.1f}%"


def one_sided(case):
    return case.get("one_sided", False)


def flops(case):
    return case.get("flops", 0)


def flags(case):
    found = []
    sides = [case["flint"]] if one_sided(case) else [case["flint"], case["baseline"]]
    for side in sides:
        stats = side["stats"]
        if stats["thermal"] > 0 and "thermal" not in found:
            found.append("thermal")
        if stats["hw_slowdown"] > 0 and "hw-slowdown" not in found:
            found.append("hw-slowdown")
    if not case["ok"]:
        found.append("NOT OK")
    return ", ".join(found) or "-"


def problems(log):
    """Why a log's numbers mustn't be published, if they mustn't."""
    found = []
    if log["quick"]:
        found.append("a --quick run")
    elif not log["standard_protocol"]:
        found.append("options below the standard protocol")
    if log["git"]["dirty"]:
        found.append("built from a tree with uncommitted changes")
    if log["git"]["sha"] == "unknown":
        found.append("built outside a git checkout")
    if not all(case["ok"] for case in log["cases"]):
        found.append("a case that isn't ok")
    return found


def header(log, link, rated):
    device = log["device"]
    command = " ".join(["flint_bench", *log["args"]])
    return (
        f"**{device['name']}**: driver {device['driver_version']}, "
        f"CUDA runtime {cuda_version(device['cuda_runtime_version'])}, "
        f"{rated}, commit `{log['git']['sha'][:7]}`, "
        f"`{command}`, raw log [`{Path(link).name}`]({link})."
    )


def rated_peak_tflops(log):
    """The rated compute peak in TFLOPS, or None when the GPU has no rated clock."""
    device = log["device"]
    clock = device.get("rated_sm_clock_mhz")
    if not clock:
        return None
    peak = log["cases"][0]["peak_flop_per_sm_per_clock"]
    return peak * device["sm_count"] * clock * 1e6 / 1e12


def preamble(log, first):
    lines = [first]
    issues = problems(log)
    if issues:
        lines.append(f"**Not publishable:** {'; '.join(issues)}.")
    if log["gpu_at_start"]["other_processes"]:
        lines.append(
            f"{log['gpu_at_start']['other_processes']} other processes were on the GPU at start."
        )
    return lines


def render_bandwidth(log, link, cases):
    rated = f"rated {log['device']['rated_bandwidth_gbs']:.1f} GB/s"
    lines = preamble(log, header(log, link, rated))
    baseline = cases[0]["baseline_name"]
    lines += [
        "",
        f"| Size | Flint | {baseline} | Flint / {baseline} | Rated | Clock-adj. "
        "| SM clock | Mem clock | Rounds | Flags |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        flint, base, ratio = case["flint"], case["baseline"], case["ratio"]
        lines.append(
            f"| {case['label']} | {flint['gbs']:.1f} GB/s | {base['gbs']:.1f} GB/s "
            f"| {percent(ratio['median'])} [{100 * ratio['min']:.1f}-{100 * ratio['max']:.1f}] "
            f"| {percent(flint['rated_fraction'])} | {percent(flint['clock_adjusted_fraction'])} "
            f"| {flint['stats']['sm_mhz']:.0f} MHz | {flint['stats']['mem_mhz']:.0f} MHz "
            f"| {case['valid_rounds']}/{len(case['rounds'])} | {flags(case)} |"
        )
    return lines


def render_compute(log, link, cases):
    clock = log["device"].get("rated_sm_clock_mhz")
    peak = rated_peak_tflops(log)
    if peak is None:
        rated = "no rated peak"
    else:
        rated = f"rated SM clock {clock} MHz, rated peak {peak:.1f} TFLOPS"
    lines = preamble(log, header(log, link, rated))
    lines += [
        "",
        "| Case | TFLOPS | Rated | Clock-adj. | SM clock | Power | Rounds | Flags |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        flint = case["flint"]
        rated_fraction = flint.get("rated_compute_fraction")
        rated_cell = percent(rated_fraction) if rated_fraction else "-"
        adjusted = flint.get("clock_adjusted_compute_fraction")
        adjusted_cell = percent(adjusted) if adjusted else "-"
        lines.append(
            f"| {case['label']} | {flint['tflops']:.1f} | {rated_cell} | {adjusted_cell} "
            f"| {flint['stats']['sm_mhz']:.0f} MHz | {flint['stats']['power_w']:.0f} W "
            f"| {case['valid_rounds']}/{len(case['rounds'])} | {flags(case)} |"
        )
    return lines


def algorithm_cell(case):
    """The cuBLASLt candidate a case's baseline used, and its place in the heuristic's list."""
    found = case.get("baseline_algorithm")
    if not found:
        return "-"
    used = found["candidates"][found["used"]]
    config = "/".join(
        "?" if used[key] is None else str(used[key]) for key in ("algo_id", "tile_id", "stages_id")
    )
    return f"{config} ({found['used'] + 1} of {len(found['candidates'])})"


def render_compute_baseline(log, link, cases):
    clock = log["device"].get("rated_sm_clock_mhz")
    peak = rated_peak_tflops(log)
    if peak is None:
        rated = "no rated peak"
    else:
        rated = f"rated SM clock {clock} MHz, rated peak {peak:.1f} TFLOPS"
    lines = preamble(log, header(log, link, rated))
    names = {case["baseline_name"] for case in cases}
    baseline = names.pop() if len(names) == 1 else "Baseline"
    with_algorithm = any(case.get("baseline_algorithm") for case in cases)
    columns = [
        ("Shape", "---"),
        ("Flint", "---:"),
        (baseline, "---:"),
        (f"Flint / {baseline}", "---"),
        ("Rated", "---:"),
        ("Clock-adj.", "---:"),
        ("SM clock", "---:"),
        ("Power", "---:"),
        ("Rounds", "---:"),
        ("Flags", "---"),
    ]
    if with_algorithm:
        columns.append(("cuBLASLt algo", "---"))
    lines += [
        "",
        "| " + " | ".join(name for name, _ in columns) + " |",
        "| " + " | ".join(align for _, align in columns) + " |",
    ]
    for case in cases:
        flint, base, ratio = case["flint"], case["baseline"], case["ratio"]
        rated_fraction = flint.get("rated_compute_fraction")
        rated_cell = percent(rated_fraction) if rated_fraction else "-"
        adjusted = flint.get("clock_adjusted_compute_fraction")
        adjusted_cell = percent(adjusted) if adjusted else "-"
        cells = [
            case["label"],
            f"{flint['tflops']:.1f}",
            f"{base['tflops']:.1f}",
            f"{percent(ratio['median'])} [{100 * ratio['min']:.1f}-{100 * ratio['max']:.1f}]",
            rated_cell,
            adjusted_cell,
            f"{flint['stats']['sm_mhz']:.0f} MHz",
            f"{flint['stats']['power_w']:.0f} W",
            f"{case['valid_rounds']}/{len(case['rounds'])}",
            flags(case),
        ]
        if with_algorithm:
            cells.append(algorithm_cell(case))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render(log, link):
    # A log holds one benchmark, so its cases are all bandwidth or all compute.
    cases = log["cases"]
    if any(flops(c) > 0 for c in cases):
        if all(one_sided(c) for c in cases):
            return "\n".join(render_compute(log, link, cases))
        return "\n".join(render_compute_baseline(log, link, cases))
    return "\n".join(render_bandwidth(log, link, cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logs", nargs="+", help="flint_bench JSON raw logs")
    args = parser.parse_args()
    tables = []
    for path in args.logs:
        with open(path, encoding="utf-8") as file:
            log = json.load(file)
        if log.get("schema") != "flint-bench/1":
            sys.exit(f"{path}: not a flint-bench/1 raw log")
        tables.append(render(log, Path(path).as_posix()))
    print("\n\n".join(tables))


if __name__ == "__main__":
    main()
