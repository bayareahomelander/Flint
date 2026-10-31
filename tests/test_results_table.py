"""scripts/results_table.py: tables rendered from the committed raw logs."""

import copy
import importlib.util
import json

import pytest

import flint_lib

RAW_LOGS = sorted((flint_lib.ROOT / "docs" / "results" / "raw").glob("M*/*.json"))

spec = importlib.util.spec_from_file_location(
    "results_table", flint_lib.ROOT / "scripts" / "results_table.py"
)
results_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results_table)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", RAW_LOGS, ids=lambda p: p.name)
def test_rows_match_the_log(path):
    log = load(path)
    lines = results_table.render(log, path.name).splitlines()
    rows = [line for line in lines if line.startswith("| ")][2:]  # past the header and rule
    assert len(rows) == len(log["cases"])
    for row, case in zip(rows, log["cases"]):
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        flint = case["flint"]
        rounds = f"{case['valid_rounds']}/{len(case['rounds'])}"
        assert cells[0] == case["label"]
        if case.get("one_sided"):  # mma: Flint against the GPU's peak
            assert cells[1] == f"{flint['tflops']:.1f}"
            assert cells[3] == f"{100 * flint['clock_adjusted_compute_fraction']:.1f}%"
            assert cells[4] == f"{flint['stats']['sm_mhz']:.0f} MHz"
            assert cells[6] == rounds
            continue
        base, ratio = case["baseline"], case["ratio"]
        ratio_cell = f"{100 * ratio['median']:.1f}% [{100 * ratio['min']:.1f}-{100 * ratio['max']:.1f}]"
        if case.get("flops", 0) > 0:  # gemm: Flint against cuBLAS or cuBLASLt
            assert cells[1] == f"{flint['tflops']:.1f}"
            assert cells[2] == f"{base['tflops']:.1f}"
            assert cells[3] == ratio_cell
            assert cells[6] == f"{flint['stats']['sm_mhz']:.0f} MHz"
            assert cells[8] == rounds
            continue
        assert cells[1] == f"{flint['gbs']:.1f} GB/s"
        assert cells[2] == f"{base['gbs']:.1f} GB/s"
        assert cells[3] == ratio_cell
        assert cells[4] == f"{100 * flint['rated_fraction']:.1f}%"
        assert cells[5] == f"{100 * flint['clock_adjusted_fraction']:.1f}%"
        assert cells[6] == f"{flint['stats']['sm_mhz']:.0f} MHz"
        assert cells[7] == f"{flint['stats']['mem_mhz']:.0f} MHz"
        assert cells[8] == rounds


@pytest.mark.parametrize("path", RAW_LOGS, ids=lambda p: p.name)
def test_write_up_quotes_the_table_verbatim(path):
    # docs/results/M<N>.md must hold each of its logs' tables exactly as the
    # script renders them, so no number in a write-up drifts from its raw log.
    milestone = path.parent.name
    write_up = (flint_lib.ROOT / "docs" / "results" / f"{milestone}.md").read_text(encoding="utf-8")
    link = f"raw/{milestone}/{path.name}"
    assert results_table.render(load(path), link) in write_up


@pytest.mark.parametrize("path", RAW_LOGS, ids=lambda p: p.name)
def test_committed_logs_are_publishable(path):
    log = load(path)
    assert results_table.problems(log) == []
    assert "Not publishable" not in results_table.render(log, path.name)


def test_unpublishable_logs_say_why():
    log = load(RAW_LOGS[0])
    log = copy.deepcopy(log)
    log["quick"] = True
    log["git"]["dirty"] = True
    text = results_table.render(log, "x.json")
    assert "**Not publishable:** a --quick run; built from a tree with uncommitted changes." in text


def synthetic_compute_log(rated_clock):
    def case(chains, tflops, rated, adjusted):
        return {
            "benchmark": "mma",
            "label": f"m16n8k8 f16->f32, {chains} chains",
            "baseline_name": "",
            "bytes_moved": 0,
            "flops": 1.0e12,
            "peak_flop_per_sm_per_clock": 512,
            "one_sided": True,
            "ok": True,
            "valid_rounds": 3,
            "rounds": [{}, {}, {}, {}],
            "ratio": None,
            "baseline": None,
            "flint": {
                "tflops": tflops,
                "rated_compute_fraction": rated,
                "clock_adjusted_compute_fraction": adjusted,
                "stats": {
                    "thermal": 0,
                    "hw_slowdown": 0,
                    "sm_mhz": 1530.4,
                    "power_w": 182.6,
                },
            },
        }

    rated = (lambda f: f) if rated_clock else (lambda f: None)
    return {
        "schema": "flint-bench/1",
        "quick": False,
        "standard_protocol": True,
        "args": ["mma"],
        "git": {"sha": "0123456789abcdef", "dirty": False},
        "device": {
            "name": "Test GPU",
            "driver_version": "1.2",
            "cuda_runtime_version": 12040,
            "sm_count": 68,
            "rated_sm_clock_mhz": rated_clock,
        },
        "gpu_at_start": {"other_processes": 0},
        "cases": [case(4, 50.04, rated(0.5), 0.6), case(8, 80.0, rated(0.8), 0.9)],
    }


def table_cells(text):
    rows = [line for line in text.splitlines() if line.startswith("| ")][2:]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]


def test_compute_table_with_rated_peak():
    log = synthetic_compute_log(1545)
    text = results_table.render(log, "raw/M1/mma.json")
    peak = 512 * 68 * 1545 * 1e6 / 1e12
    assert f"rated SM clock 1545 MHz, rated peak {peak:.1f} TFLOPS" in text
    assert "| Case | TFLOPS | Rated | Clock-adj. | SM clock | Power | Rounds | Flags |" in text
    assert "Not publishable" not in text
    cells = table_cells(text)
    assert cells == [
        ["m16n8k8 f16->f32, 4 chains", "50.0", "50.0%", "60.0%", "1530 MHz", "183 W", "3/4", "-"],
        ["m16n8k8 f16->f32, 8 chains", "80.0", "80.0%", "90.0%", "1530 MHz", "183 W", "3/4", "-"],
    ]


def test_compute_table_without_rated_peak():
    log = synthetic_compute_log(None)
    text = results_table.render(log, "mma.json")
    assert "no rated peak" in text
    cells = table_cells(text)
    assert [row[2] for row in cells] == ["-", "-"]
    assert [row[3] for row in cells] == ["60.0%", "90.0%"]


def test_compute_log_problems_still_reported():
    log = synthetic_compute_log(None)
    log["cases"][0]["ok"] = False
    log["cases"][0]["flint"]["stats"]["thermal"] = 2
    assert results_table.problems(log) == ["a case that isn't ok"]
    text = results_table.render(log, "mma.json")
    assert "**Not publishable:** a case that isn't ok." in text
    assert table_cells(text)[0][-1] == "thermal, NOT OK"


def synthetic_gemm_log(rated_clock, with_algorithm=True):
    def algorithm(used, count, algo_id, tile_id, stages_id):
        candidates = [
            {"index": i, "algo_id": algo_id, "tile_id": tile_id, "stages_id": stages_id}
            for i in range(count)
        ]
        return {"selection": "fastest", "used": used, "candidates": candidates}

    def side(tflops, rated, adjusted, sm_mhz, power_w):
        return {
            "tflops": tflops,
            "rated_compute_fraction": rated,
            "clock_adjusted_compute_fraction": adjusted,
            "stats": {"thermal": 0, "hw_slowdown": 0, "sm_mhz": sm_mhz, "power_w": power_w},
        }

    def case(label, flint_tflops, base_tflops, ratio, algo, rated, adjusted):
        found = {
            "benchmark": "gemm",
            "label": label,
            "baseline_name": "cuBLASLt, FP16 bias",
            "bytes_moved": 0,
            "flops": 1.0e12,
            "peak_flop_per_sm_per_clock": 512,
            "one_sided": False,
            "ok": True,
            "valid_rounds": 10,
            "rounds": [{}] * 10,
            "ratio": {"median": ratio[0], "min": ratio[1], "max": ratio[2]},
            "baseline": side(base_tflops, None, None, 1500.0, 190.0),
            "flint": side(flint_tflops, rated, adjusted, 1706.4, 183.2),
        }
        if with_algorithm:
            found["baseline_algorithm"] = algo
        return found

    rated = (lambda f: f) if rated_clock else (lambda f: None)
    return {
        "schema": "flint-bench/1",
        "quick": False,
        "standard_protocol": True,
        "args": ["gemm", "--epilogue", "bias"],
        "git": {"sha": "0123456789abcdef", "dirty": False},
        "device": {
            "name": "Test GPU",
            "driver_version": "1.2",
            "cuda_runtime_version": 12040,
            "sm_count": 68,
            "rated_sm_clock_mhz": rated_clock,
        },
        "gpu_at_start": {"other_processes": 0},
        "cases": [
            case("8192x2304x768", 44.34, 53.9, (0.8229, 0.8164, 0.8497),
                 algorithm(0, 2, 5, 20, 0), rated(0.5), rated(0.6)),
            case("2048x3072x16384", 50.0, 54.76, (0.9130, 0.8956, 0.9599),
                 algorithm(4, 6, 21, None, 9), rated(0.75), rated(0.8)),
        ],
    }


GEMM_HEADER = (
    "| Shape | Flint | cuBLASLt, FP16 bias | Flint / cuBLASLt, FP16 bias | Rated | Clock-adj. "
    "| SM clock | Power | Rounds | Flags |"
)


def test_gemm_table_with_rated_peak_and_algorithms():
    log = synthetic_gemm_log(1545)
    text = results_table.render(log, "raw/M1/gemm.json")
    assert "rated SM clock 1545 MHz" in text
    assert "Not publishable" not in text
    assert GEMM_HEADER[:-1] + "| cuBLASLt algo |" in text
    assert "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |" in text
    assert table_cells(text) == [
        ["8192x2304x768", "44.3", "53.9", "82.3% [81.6-85.0]", "50.0%", "60.0%",
         "1706 MHz", "183 W", "10/10", "-", "5/20/0 (1 of 2)"],
        ["2048x3072x16384", "50.0", "54.8", "91.3% [89.6-96.0]", "75.0%", "80.0%",
         "1706 MHz", "183 W", "10/10", "-", "21/?/9 (5 of 6)"],
    ]


def test_gemm_table_without_rated_peak():
    text = results_table.render(synthetic_gemm_log(None), "gemm.json")
    assert "no rated peak" in text
    cells = table_cells(text)
    assert [row[4] for row in cells] == ["-", "-"]
    assert [row[5] for row in cells] == ["-", "-"]
    assert [row[1] for row in cells] == ["44.3", "50.0"]


def test_gemm_table_without_algorithms_has_no_algo_column():
    absent = synthetic_gemm_log(1545, with_algorithm=False)
    null = synthetic_gemm_log(1545)
    for case in null["cases"]:
        case["baseline_algorithm"] = None
    for log in (absent, null):
        text = results_table.render(log, "gemm.json")
        assert "cuBLASLt algo" not in text
        assert GEMM_HEADER + "\n| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |" in text
        assert all(len(row) == 10 for row in table_cells(text))


def test_gemm_table_with_mixed_baselines():
    log = synthetic_gemm_log(1545)
    log["cases"][1]["baseline_name"] = "cuBLAS"
    text = results_table.render(log, "gemm.json")
    assert "| Shape | Flint | Baseline | Flint / Baseline | Rated |" in text
    assert "cuBLASLt, FP16 bias" not in text


def test_gemm_log_problems_still_reported():
    log = synthetic_gemm_log(1545)
    log["quick"] = True
    log["cases"][1]["baseline"]["stats"]["hw_slowdown"] = 1
    text = results_table.render(log, "gemm.json")
    assert "**Not publishable:** a --quick run." in text
    assert table_cells(text)[1][9] == "hw-slowdown"
