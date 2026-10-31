"""scripts/k_sweep.py: the least-squares fit over k, on synthetic logs."""

import importlib.util
import json

import pytest

import flint_lib

spec = importlib.util.spec_from_file_location("k_sweep", flint_lib.ROOT / "scripts" / "k_sweep.py")
k_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k_sweep)


def round_(flint_ms, base_ms, valid=True, flint_mhz=1500.0, base_mhz=1500.0):
    return {
        "valid": valid,
        "flint": {"ms_per_launch": flint_ms, "stats": {"sm_mhz": flint_mhz}},
        "baseline": {"ms_per_launch": base_ms, "stats": {"sm_mhz": base_mhz}},
    }


def case(m, n, k, fa, fb, ba, bb, extra_rounds=(), layout="nt", epilogue="none", **clocks):
    """A case whose times are exactly a + b*k (b per 1,000 of k)."""
    flint, base = fa + fb * k / 1000, ba + bb * k / 1000
    rounds = [round_(flint, base, **clocks) for _ in range(3)] + list(extra_rounds)
    return {
        "label": f"{m}x{n}x{k}",
        "layout": layout,
        "epilogue": epilogue,
        "one_sided": False,
        "rounds": rounds,
    }


def log(cases, dirty=False, sha="abc1234"):
    return {
        "schema": "flint-bench/1",
        "benchmark": "gemm",
        "git": {"sha": sha, "dirty": dirty},
        "cases": cases,
    }


def sweep(m=1024, n=768, ks=(512, 1024, 2048), **kwargs):
    return [case(m, n, k, 0.02, 0.10, 0.01, 0.08, **kwargs) for k in ks]


def test_linear_times_give_exact_fits():
    (group,) = k_sweep.analyze([("a.json", log(sweep()))])
    flint, base = group["ms"]["fits"]["flint"], group["ms"]["fits"]["baseline"]
    assert flint["a"] == pytest.approx(0.02)
    assert flint["b_per_1000k"] == pytest.approx(0.10)
    assert base["a"] == pytest.approx(0.01)
    assert base["b_per_1000k"] == pytest.approx(0.08)
    assert flint["r2"] == pytest.approx(1.0)
    assert all(abs(r) < 1e-9 for r in flint["residual_pct"] + base["residual_pct"])
    assert flint["a_se"] == pytest.approx(0, abs=1e-12)
    assert flint["b_se_per_1000k"] == pytest.approx(0, abs=1e-12)
    assert group["ms"]["slope_ratio"] == pytest.approx(0.8)
    assert group["ms"]["intercept_diff"] == pytest.approx(0.01)
    for point in group["per_k"]:
        assert point["measured_ratio"] == pytest.approx(point["fitted_ratio"])


def test_cycle_fits_take_out_the_clock():
    # Flint at 1,450 MHz and the baseline at 1,500: the cycle fits scale each
    # side's line by its own clock, so the cycle slope ratio is the time slope
    # ratio times 1500 / 1450.
    (group,) = k_sweep.analyze([("a.json", log(sweep(flint_mhz=1450.0, base_mhz=1500.0)))])
    flint = group["kcycles"]["fits"]["flint"]
    assert flint["a"] == pytest.approx(0.02 * 1450)
    assert flint["b_per_1000k"] == pytest.approx(0.10 * 1450)
    assert group["kcycles"]["slope_ratio"] == pytest.approx(0.8 * 1500 / 1450)
    assert group["ms"]["slope_ratio"] == pytest.approx(0.8)
    point = group["per_k"][0]
    assert point["flint_sm_mhz"] == 1450.0 and point["baseline_sm_mhz"] == 1500.0
    assert point["cycle_ratio"] == pytest.approx(point["measured_ratio"] * 1500 / 1450)


def test_standard_errors_of_a_noisy_fit():
    # Times 1, 3, 2, 4 at k = 0, 1, 2, 3 thousand: b = 0.8, a = 1.3, and with
    # SSR = 1.8 over 2 degrees of freedom, se(b) = sqrt(0.9 / 5) and
    # se(a) = sqrt(0.9 (1/4 + 1.5^2 / 5)).
    result = k_sweep.fit([0, 1000, 2000, 3000], [1.0, 3.0, 2.0, 4.0])
    assert result["b_per_1000k"] == pytest.approx(0.8)
    assert result["a"] == pytest.approx(1.3)
    assert result["b_se_per_1000k"] == pytest.approx((0.9 / 5) ** 0.5)
    assert result["a_se"] == pytest.approx((0.9 * (0.25 + 2.25 / 5)) ** 0.5)


def test_two_shapes_in_one_log_stay_apart():
    cases = sweep(m=1024, n=768) + sweep(m=2048, n=512)
    groups = k_sweep.analyze([("a.json", log(cases))])
    assert sorted((g["m"], g["n"]) for g in groups) == [(1024, 768), (2048, 512)]
    assert all(len(g["per_k"]) == 3 for g in groups)


def test_layouts_and_epilogues_stay_apart():
    cases = sweep(layout="nt") + sweep(layout="nn")
    groups = k_sweep.analyze([("a.json", log(cases))])
    assert sorted(g["layout"] for g in groups) == ["nn", "nt"]


def test_discarded_rounds_are_ignored():
    junk = [round_(99.0, 99.0, valid=False)]
    cases = sweep(extra_rounds=junk)
    (group,) = k_sweep.analyze([("a.json", log(cases))])
    assert group["ms"]["fits"]["flint"]["b_per_1000k"] == pytest.approx(0.10)
    assert group["ms"]["fits"]["flint"]["r2"] == pytest.approx(1.0)


def test_median_over_valid_rounds():
    c = case(1024, 768, 1024, 0.02, 0.10, 0.01, 0.08)
    c["rounds"] = [round_(1.0, 2.0), round_(3.0, 4.0), round_(100.0, 200.0)]
    times, valid = k_sweep.case_times(c, "a.json")
    assert times["flint"]["ms"] == 3.0 and times["baseline"]["ms"] == 4.0
    assert times["flint"]["kcycles"] == pytest.approx(3.0 * 1500)
    assert valid == 3


def test_round_without_a_clock_refuses():
    c = case(1024, 768, 1024, 0.02, 0.10, 0.01, 0.08)
    c["rounds"][0]["flint"]["stats"]["sm_mhz"] = 0
    with pytest.raises(ValueError, match="no SM clock"):
        k_sweep.case_times(c, "a.json")


def test_case_without_valid_rounds_refuses():
    cases = sweep()
    cases[1]["rounds"] = [round_(1.0, 1.0, valid=False)]
    with pytest.raises(ValueError, match="no valid rounds"):
        k_sweep.analyze([("a.json", log(cases))])


def test_fewer_than_three_k_values_refuses():
    with pytest.raises(ValueError, match="needs at least 3"):
        k_sweep.analyze([("a.json", log(sweep(ks=(512, 1024))))])


def test_other_schema_or_benchmark_refuses():
    bad = log(sweep())
    bad["schema"] = "flint-bench/2"
    with pytest.raises(ValueError, match="flint-bench/1"):
        k_sweep.analyze([("a.json", bad)])
    bad = log(sweep())
    bad["benchmark"] = "copy"
    with pytest.raises(ValueError, match="not gemm"):
        k_sweep.analyze([("a.json", bad)])


def test_cases_from_several_logs_join_one_group():
    logs = [(f"{k}.json", log([case(1024, 768, k, 0.02, 0.10, 0.01, 0.08)])) for k in (512, 1024, 2048)]
    (group,) = k_sweep.analyze(logs)
    assert group["k"] == [512, 1024, 2048]


def test_dirty_or_unknown_logs_warn():
    logs = [("clean.json", log(sweep())), ("dirty.json", log(sweep(), dirty=True)),
            ("none.json", log(sweep(), sha="unknown"))]
    found = k_sweep.warnings(logs)
    assert len(found) == 2
    assert "dirty.json" in found[0] and "none.json" in found[1]
    assert k_sweep.warnings(logs[:1]) == []


def test_command_line_writes_names_only(tmp_path, monkeypatch, capsys):
    path = tmp_path / "gemm.json"
    path.write_text(json.dumps(log(sweep(), dirty=True)), encoding="utf-8")
    out = tmp_path / "sweep.json"
    monkeypatch.setattr("sys.argv", ["k_sweep.py", str(path), "--out", str(out)])
    k_sweep.main()
    captured = capsys.readouterr()
    assert "not from a clean git checkout" in captured.err
    assert "slope ratio" in captured.out
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["schema"] == "flint-k-sweep/1"
    assert document["inputs"] == [{"file": "gemm.json", "git_sha": "abc1234"}]
    assert str(tmp_path) not in out.read_text(encoding="utf-8")


def test_excluded_cases_leave_the_fit_but_are_reported():
    # A fourth point off the line (another kernel, say) would bend the fit;
    # excluded, the fit stays exact and the point is reported with what the
    # fits predict for it.
    cases = sweep(ks=(512, 1024, 2048, 4096))
    cases[3]["rounds"] = [round_(1.0, 0.2) for _ in range(3)]
    (group,) = k_sweep.analyze([("a.json", log(cases))], exclude={"nt:1024x768x4096"})
    assert group["k"] == [512, 1024, 2048]
    assert group["ms"]["fits"]["flint"]["r2"] == pytest.approx(1.0)
    (point,) = group["excluded"]
    assert point["k"] == 4096 and point["baseline_ms"] == pytest.approx(0.2)
    assert point["baseline_fitted_ms"] == pytest.approx(0.01 + 0.08 * 4.096)
    # The layout is part of the name: nn's case of the same label stays in.
    (group,) = k_sweep.analyze([("a.json", log(sweep(layout="nn")))], exclude={"nt:1024x768x512"})
    assert group["k"] == [512, 1024, 2048] and group["excluded"] == []


def test_command_line_refuses_an_unknown_exclusion(tmp_path, monkeypatch):
    path = tmp_path / "gemm.json"
    path.write_text(json.dumps(log(sweep())), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["k_sweep.py", str(path), "--exclude", "nt:1x2x3"])
    with pytest.raises(SystemExit, match="not in the logs"):
        k_sweep.main()
