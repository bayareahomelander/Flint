"""Fits time = a + b*k to flint_bench gemm logs, per side, at fixed m and n.

Usage: python scripts/k_sweep.py LOG [LOG ...] [--out FILE.json]

The logs are flint-bench/1 raw logs of the gemm benchmark whose case labels
("MxNxK") share m and n and differ in k. Cases are grouped by layout,
epilogue, m, and n; each group needs three or more distinct k. A case's time is
the median over its valid rounds of the round's per-launch milliseconds, for
Flint and for the baseline. The intercept a is the side's fixed cost, and the
slope b its main loop's cost per 1,000 of k. Baseline slope / Flint slope is
the speed of Flint's main loop relative to the baseline's. Each fit carries
the standard errors of a and b.

The same fits are made to SM cycles (each round's per-launch time times its
mean SM clock from NVML, in thousands of cycles), which take out the clock
difference between the sides: under a power cap, the side that draws more
power per unit of work runs at a lower clock.

A fit assumes each side runs one kernel at every k. cuBLAS's heuristic can
switch kernels as k grows, so --exclude leaves cases out of their group's fits
(as LAYOUT:MxNxK, such as nt:8192x2304x768). An excluded case is still
reported, beside what the fits predict for it.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

SCHEMA = "flint-k-sweep/1"
SIDES = ("flint", "baseline")
MIN_K_VALUES = 3


def parse_label(label):
    parts = label.split("x")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"case label {label!r} is not MxNxK")
    return tuple(int(p) for p in parts)


def case_times(case, where):
    """Median per-launch milliseconds, mean SM clock (MHz) and thousands of SM
    cycles over the valid rounds, per side."""
    rounds = [r for r in case["rounds"] if r["valid"]]
    if not rounds:
        raise ValueError(f"{where}: case {case['label']} has no valid rounds")
    for r in rounds:
        for side in SIDES:
            if not r[side]["stats"]["sm_mhz"] > 0:
                raise ValueError(f"{where}: case {case['label']} has a round with no SM clock")

    def median(side, value):
        return statistics.median(value(r[side]) for r in rounds)

    return {
        side: {
            "ms": median(side, lambda s: s["ms_per_launch"]),
            "sm_mhz": median(side, lambda s: s["stats"]["sm_mhz"]),
            # ms times MHz is thousands of cycles.
            "kcycles": median(side, lambda s: s["ms_per_launch"] * s["stats"]["sm_mhz"]),
        }
        for side in SIDES
    }, len(rounds)


def collect(logs, exclude=()):
    """Groups the logs' cases by (layout, epilogue, m, n), each as (fitted
    points, excluded points). logs is [(name, log)]; exclude holds
    LAYOUT:MxNxK strings."""
    groups = {}
    for name, log in logs:
        if log.get("schema") != "flint-bench/1":
            raise ValueError(f"{name}: not a flint-bench/1 raw log")
        if log.get("benchmark") != "gemm":
            raise ValueError(f"{name}: benchmark is {log.get('benchmark')!r}, not gemm")
        for case in log["cases"]:
            if case.get("one_sided"):
                raise ValueError(f"{name}: case {case['label']} is one-sided")
            m, n, k = parse_label(case["label"])
            times, valid = case_times(case, name)
            key = (case.get("layout"), case.get("epilogue"), m, n)
            point = {"k": k, "times": times, "valid_rounds": valid, "log": name}
            excluded = f"{case.get('layout')}:{case['label']}" in exclude
            groups.setdefault(key, ([], []))[excluded].append(point)
    return groups


def fit(ks, ys):
    """Ordinary least squares y = a + b*k: a, b per 1,000 of k, their standard
    errors, R^2 and per-point residuals in percent."""
    count = len(ks)
    mean_k = sum(ks) / count
    mean_y = sum(ys) / count
    sxx = sum((k - mean_k) ** 2 for k in ks)
    b = sum((k - mean_k) * (y - mean_y) for k, y in zip(ks, ys)) / sxx
    a = mean_y - b * mean_k
    residuals = [y - (a + b * k) for k, y in zip(ks, ys)]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    variance = ss_res / (count - 2) if count > 2 else 0.0
    return {
        "a": a,
        "a_se": (variance * (1 / count + mean_k**2 / sxx)) ** 0.5,
        "b_per_1000k": b * 1000,
        "b_se_per_1000k": (variance / sxx) ** 0.5 * 1000,
        "r2": r2,
        "residual_pct": [100 * r / y for r, y in zip(residuals, ys)],
    }


def predict(f, k):
    return f["a"] + f["b_per_1000k"] * k / 1000


def describe(point, fits=None):
    """A point's times, clocks and ratios, and what fits (if given) predict for it."""
    t = point["times"]
    described = {
        "k": point["k"],
        "valid_rounds": point["valid_rounds"],
        **{f"{side}_{unit}": t[side][unit] for side in SIDES for unit in ("ms", "sm_mhz", "kcycles")},
        "measured_ratio": t["baseline"]["ms"] / t["flint"]["ms"],
        "cycle_ratio": t["baseline"]["kcycles"] / t["flint"]["kcycles"],
    }
    if fits is not None:
        described["fitted_ratio"] = predict(fits["baseline"], point["k"]) / predict(fits["flint"], point["k"])
        for side in SIDES:
            described[f"{side}_fitted_ms"] = predict(fits[side], point["k"])
    return described


def analyze_group(key, points, excluded=()):
    layout, epilogue, m, n = key
    distinct = sorted({p["k"] for p in points})
    if len(distinct) < MIN_K_VALUES:
        raise ValueError(
            f"group {layout}/{epilogue} {m}x{n} has {len(distinct)} distinct k "
            f"({distinct}), needs at least {MIN_K_VALUES}"
        )
    points = sorted(points, key=lambda p: p["k"])
    ks = [p["k"] for p in points]
    result = {"layout": layout, "epilogue": epilogue, "m": m, "n": n, "k": ks}
    # a is in ms for the time fits and in thousands of cycles for the cycle fits.
    for unit in ("ms", "kcycles"):
        fits = {side: fit(ks, [p["times"][side][unit] for p in points]) for side in SIDES}
        flint, base = fits["flint"], fits["baseline"]
        result[unit] = {
            "fits": fits,
            "slope_ratio": base["b_per_1000k"] / flint["b_per_1000k"],
            "intercept_diff": flint["a"] - base["a"],
        }
    fits = result["ms"]["fits"]
    result["per_k"] = [describe(p, fits) for p in points]
    result["excluded"] = [describe(p, fits) for p in sorted(excluded, key=lambda p: p["k"])]
    return result


def analyze(logs, exclude=()):
    groups = collect(logs, exclude)
    return [analyze_group(key, *groups[key]) for key in sorted(groups, key=str)]


def warnings(logs):
    found = []
    for name, log in logs:
        git = log.get("git", {})
        if git.get("dirty") or git.get("sha", "unknown") == "unknown":
            found.append(f"warning: {name} is not from a clean git checkout")
    return found


def render(results):
    lines = []
    for g in results:
        lines.append(f"{g['layout']} {g['epilogue']} m={g['m']} n={g['n']}, k = {g['k']}")
        for unit, label in (("ms", "ms"), ("kcycles", "k cycles")):
            fits = g[unit]["fits"]
            lines.append(f"  in {label}:")
            lines.append("  side      a (+- se)              b per 1000k (+- se)      R2   residuals (%)")
            for side, f in fits.items():
                residuals = ", ".join(f"{r:+.2f}" for r in f["residual_pct"])
                lines.append(
                    f"  {side:<9} {f['a']:9.4f} +- {f['a_se']:<8.4f}  {f['b_per_1000k']:10.5f} +- "
                    f"{f['b_se_per_1000k']:<8.5f}  {f['r2']:.5f}   {residuals}"
                )
            lines.append(
                f"  slope ratio (baseline b / Flint b): {g[unit]['slope_ratio']:.3f}; "
                f"intercept diff (Flint a - baseline a): {g[unit]['intercept_diff']:+.4f}"
            )
        lines.append("  k        Flint ms  baseline ms  Flint MHz  baseline MHz  ratio  fitted  cycle ratio")
        for p in g["per_k"] + g["excluded"]:
            note = "  (excluded)" if p in g["excluded"] else ""
            lines.append(
                f"  {p['k']:<7} {p['flint_ms']:9.4f}  {p['baseline_ms']:11.4f}  {p['flint_sm_mhz']:9.0f}  "
                f"{p['baseline_sm_mhz']:12.0f}  {p['measured_ratio']:5.3f}  {p['fitted_ratio']:6.3f}  "
                f"{p['cycle_ratio']:11.3f}{note}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("logs", nargs="+", help="flint_bench gemm JSON raw logs")
    parser.add_argument("--out", help="JSON file to write")
    parser.add_argument("--exclude", default="",
                        help="comma-separated LAYOUT:MxNxK cases to leave out of the fits")
    args = parser.parse_args()
    exclude = {item for item in args.exclude.split(",") if item}
    logs = []
    for path in args.logs:
        with open(path, encoding="utf-8") as file:
            logs.append((Path(path).name, json.load(file)))
    for line in warnings(logs):
        print(line, file=sys.stderr)
    try:
        results = analyze(logs, exclude)
        found = {f"{g['layout']}:{g['m']}x{g['n']}x{p['k']}" for g in results for p in g["excluded"]}
        if exclude - found:
            raise ValueError(f"--exclude names cases not in the logs: {sorted(exclude - found)}")
    except ValueError as error:
        sys.exit(f"error: {error}")
    print(render(results))
    if args.out:
        document = {
            "schema": SCHEMA,
            "inputs": [
                {"file": name, "git_sha": log.get("git", {}).get("sha", "unknown")}
                for name, log in logs
            ],
            "excluded": sorted(exclude),
            "groups": results,
        }
        Path(args.out).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
