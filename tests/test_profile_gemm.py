"""scripts/profile_gemm.py: parsing ncu's CSV pages, privacy checks, and launch selection."""

import csv
import importlib.util
import io
import json

import pytest

import flint_lib

spec = importlib.util.spec_from_file_location(
    "profile_gemm", flint_lib.ROOT / "scripts" / "profile_gemm.py"
)
profile_gemm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_gemm)

STALL = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"
PRIVATE = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream"]
SMALL = ["gpu__time_duration.sum", "launch__block_size"]


def csv_text(rows):
    out = io.StringIO()
    csv.writer(out, lineterminator="\n", quoting=csv.QUOTE_ALL).writerows(rows)
    return out.getvalue()


def raw_page(kernels, extra=None):
    """A raw page: noise, a header, a units row, and one row per (name, duration, block) kernel."""
    metrics = SMALL + [STALL.format("wait"), STALL.format("barrier")] + (extra or [])
    header = PRIVATE + ["Block Size", "Grid Size", "Device", "CC"] + metrics
    units = [""] * 11 + ["usecond", "", "inst", "inst"] + [""] * len(extra or [])
    rows = [header, units]
    for i, (name, duration, block) in enumerate(kernels):
        identity = [str(i), "4242", "flint_bench.exe", "somehost", name, "1", "7"]
        launch = ["(256, 1, 1)", "(8, 6, 8)", "0", "8.9"]
        values = [duration, block, "2.50", "0.5"] + ["0"] * len(extra or [])
        rows.append(identity + launch + values)
    return "==PROF== Connected to process 4242\n" + csv_text(rows) + "==PROF== Disconnected\n"


def test_raw_page_keeps_name_dims_metrics_and_units():
    text = raw_page([("kernel_a", "1,179.5", "256")])
    (kernel,) = profile_gemm.parse_raw_page(text, SMALL)
    assert kernel["name"] == "kernel_a"
    assert kernel["block"] == [256, 1, 1] and kernel["grid"] == [8, 6, 8]
    assert kernel["metrics"]["gpu__time_duration.sum"] == {"value": 1179.5, "unit": "usecond"}
    assert kernel["metrics"]["launch__block_size"] == {"value": 256, "unit": ""}
    assert kernel["metrics"][STALL.format("wait")] == {"value": 2.5, "unit": "inst"}
    assert set(kernel) == {"name", "block", "grid", "metrics"}


def test_raw_page_thousands_separators_become_numbers():
    text = raw_page([("kernel_a", "1,179,648", "256")])
    (kernel,) = profile_gemm.parse_raw_page(text, SMALL)
    assert kernel["metrics"]["gpu__time_duration.sum"]["value"] == 1179648


def test_raw_page_last_row_of_a_name_wins():
    text = raw_page([("kernel_a", "10", "64"), ("kernel_b", "20", "64"), ("kernel_a", "30", "64")])
    kernels = profile_gemm.parse_raw_page(text, SMALL)
    assert [k["name"] for k in kernels] == ["kernel_a", "kernel_b"]
    assert kernels[0]["metrics"]["gpu__time_duration.sum"]["value"] == 30
    assert profile_gemm.raw_kernel_names(text) == ["kernel_a", "kernel_b", "kernel_a"]


def test_raw_page_never_carries_private_columns():
    text = raw_page([("kernel_a", "10", "64")])
    kernels = profile_gemm.parse_raw_page(text, SMALL)
    dumped = json.dumps(kernels)
    assert "4242" not in dumped and "flint_bench.exe" not in dumped and "somehost" not in dumped


def test_missing_metric_is_named():
    text = raw_page([("kernel_a", "10", "64")])
    with pytest.raises(ValueError, match="no_such__metric.sum"):
        profile_gemm.parse_raw_page(text, SMALL + ["no_such__metric.sum"])
    with pytest.raises(ValueError, match="sm__cycles_elapsed.avg.per_second"):
        profile_gemm.parse_raw_page(text)  # the real METRICS, absent from this page


def test_stall_columns_are_included_by_pattern():
    text = raw_page([("kernel_a", "10", "64")], extra=[STALL.format("mio_throttle")])
    (kernel,) = profile_gemm.parse_raw_page(text, SMALL)
    stalls = [name for name in kernel["metrics"] if profile_gemm.STALL_PATTERN.fullmatch(name)]
    assert stalls == [STALL.format(x) for x in ("wait", "barrier", "mio_throttle")]
    assert list(kernel["metrics"]) == SMALL + stalls


def source_page():
    columns = (
        ["Address", "Source"]
        + list(profile_gemm.SOURCE_COLUMNS)
        + ["stall_barrier", "stall_wait", "stall_barrier (Not Issued)", "stall_wait (Not Issued)"]
    )
    rows = []
    # a kernel with two blocks of the same name, the second being the last block per name
    for name, first, executed in (("kernel_a", "0x700e9e300", 8), ("kernel_a", "0x700e9e300", 16)):
        rows += [["Kernel Name", name, ""], columns]
        for i, sass in enumerate(["      MOV R1, R2 ", "      HMMA.1688 R4, R8 ", "      EXIT "]):
            address = f"0x{int(first, 16) + 16 * i:x}"
            counts = ["5,000", "3", str(executed), "7", "6", "2", "0", "1", "0"]
            rows.append([address, sass] + counts)
    rows += [["Kernel Name", "kernel_b", ""], columns[:2] + columns[2:5] + columns[7:9]]
    rows.append(["0x9000", "      EXIT ", "0", "0", "4", "0", "0"])
    return "==PROF== noise\n" + csv_text(rows)


def test_source_page_offsets_ints_and_stalls():
    blocks = profile_gemm.parse_source_page(source_page())
    assert list(blocks) == ["kernel_a", "kernel_b"]
    first = blocks["kernel_a"][0]
    assert [i["offset"] for i in blocks["kernel_a"]] == [0, 16, 32]
    assert first["sass"] == "MOV R1, R2"
    assert first["samples"] == 5000 and first["samples_not_issued"] == 3
    assert first["executed"] == 16  # the last block of the name
    assert first["l1_wavefronts_shared"] == 7 and first["l1_wavefronts_shared_ideal"] == 6
    assert first["stalls"] == {"stall_barrier": 2}  # zero counts are left out
    assert all(isinstance(v, int) for i in blocks["kernel_a"] for v in i["stalls"].values())
    assert not any("Not Issued" in key for i in blocks["kernel_a"] for key in i["stalls"])
    assert "address" not in first


def test_source_page_kernel_without_shared_memory_columns():
    (only,) = profile_gemm.parse_source_page(source_page())["kernel_b"]
    assert only["offset"] == 0
    assert only["l1_wavefronts_shared"] == 0 and only["l1_wavefronts_shared_ideal"] == 0
    assert only["executed"] == 4


def test_idle_instructions_are_the_all_zero_ones():
    first = profile_gemm.parse_source_page(source_page())["kernel_a"][0]
    assert not profile_gemm.is_idle(first)
    zero = {key: 0 for key in profile_gemm.SOURCE_COLUMNS.values()} | {"stalls": {}}
    assert profile_gemm.is_idle(zero)


def test_scrub_check_passes_a_clean_document(monkeypatch):
    monkeypatch.setattr(profile_gemm.getpass, "getuser", lambda: "alice")
    monkeypatch.setattr(profile_gemm.socket, "gethostname", lambda: "box7")
    profile_gemm.scrub_check(json.dumps({"cmd": ["<ncu>", "<reports>/x.ncu-rep"], "n": "gemm_nt"}))


@pytest.mark.parametrize(
    "text",
    [
        "C:\\Program Files\\x",
        "d:\\data",
        "see C:/Program Files",
        "/home/someone/flint",
        "/Users/someone",
        "Users\\someone",
        "run by ALICE today",
        "on Box7",
    ],
)
def test_scrub_check_rejects_private_text(monkeypatch, text):
    monkeypatch.setattr(profile_gemm.getpass, "getuser", lambda: "alice")
    monkeypatch.setattr(profile_gemm.socket, "gethostname", lambda: "box7")
    with pytest.raises(ValueError):
        profile_gemm.scrub_check(json.dumps({"x": text}))


def test_commands_lose_every_path():
    ncu = r"C:\Program Files\NVIDIA Corporation\Nsight Compute\ncu.exe"
    exe = r"C:\Users\someone\Flint\build\release\bin\flint_bench.exe"
    reports = r"C:\Users\someone\Flint\build\profile-reports"
    command = [
        ncu, "-k", "regex:^flint::", "--export", reports + r"\nt-1x2x3-flint.ncu-rep",
        exe, "gemm", "--out", reports + r"\nt-1x2x3-flint.bench.json",
    ]  # fmt: skip
    shown = profile_gemm.placeholder_command(command, ncu, exe, reports)
    assert shown == [
        "<ncu>", "-k", "regex:^flint::", "--export", "<reports>/nt-1x2x3-flint.ncu-rep",
        "<flint_bench>", "gemm", "--out", "<reports>/nt-1x2x3-flint.bench.json",
    ]  # fmt: skip
    profile_gemm.scrub_check(json.dumps(shown))


def test_gpu_tag():
    assert profile_gemm.gpu_tag("NVIDIA GeForce RTX 4050 Laptop GPU") == "rtx-4050-laptop"
    assert profile_gemm.gpu_tag("NVIDIA GeForce RTX 2080 Ti") == "rtx-2080-ti"


def test_window_skips_each_kernels_first_launch():
    assert profile_gemm.choose_window(["a"] * 5) == (1, 1)
    assert profile_gemm.choose_window(["a", "b", "a", "b", "a", "b"]) == (2, 2)
    assert profile_gemm.choose_window(["a", "a", "b", "a", "b"]) == (3, 2)
    with pytest.raises(ValueError):
        profile_gemm.choose_window(["a", "b"])
    with pytest.raises(ValueError):
        profile_gemm.choose_window([])


def test_launch_sequence_reads_a_metrics_run():
    rows = [["ID", "Process ID", "Kernel Name", "Metric Value"], ["0", "1", "flint::k", "5"]]
    rows += [["1", "1", "ampere_x", "6"]]
    text = "==PROF== hello\ngemm nt 1x2x3 done\n" + csv_text(rows)
    assert profile_gemm.launch_sequence(text) == ["flint::k", "ampere_x"]


def test_dump_json_puts_instructions_and_plain_lists_on_one_line():
    document = {"block": [1, 2], "sides": {"flint": [{"name": "k", "source": [{"offset": 0}]}]}}
    text = profile_gemm.dump_json(document)
    assert json.loads(text) == document
    assert ' "block": [1, 2],' in text and '{"offset": 0}' in text
    assert text.endswith("}\n") and "\r" not in text


def test_fit_document_drops_idle_rows_only_when_that_saves_the_size_limit(monkeypatch):
    busy = {"offset": 0, "sass": "A", "samples": 1, "executed": 1, "stalls": {}}
    idle = {"offset": 16, "sass": "B", "samples": 0, "executed": 0, "stalls": {}}
    monkeypatch.setattr(profile_gemm, "SOURCE_COLUMNS", {"a": "samples", "b": "executed"})
    document = {"sides": {"flint": [{"name": "k", "source": [busy] + [idle] * 50}]}}
    full = profile_gemm.dump_json(document)
    monkeypatch.setattr(profile_gemm, "SIZE_LIMIT", len(full) - 1)
    pruned = json.loads(profile_gemm.fit_document(document))
    assert pruned["sides"]["flint"][0]["source"] == [busy]
    assert pruned["sides"]["flint"][0]["zero_rows_omitted"]
    monkeypatch.setattr(profile_gemm, "SIZE_LIMIT", 10)  # even pruned it won't fit: keep all
    assert profile_gemm.fit_document(document) == full
    monkeypatch.setattr(profile_gemm, "SIZE_LIMIT", len(full))
    assert profile_gemm.fit_document(document) == full


def test_clock_and_cycles_read_ncus_units():
    def kernel(rate, unit, cycles, ns):
        return {"metrics": {
            "sm__cycles_elapsed.avg.per_second": {"value": rate, "unit": unit},
            "sm__cycles_elapsed.max": {"value": cycles, "unit": "cycle"},
            "gpu__time_duration.sum": {"value": ns, "unit": "nsecond"},
        }}  # fmt: skip

    # ncu reports 1.14 cycle/nsecond for one kernel and 824.7 cycle/usecond for another.
    assert profile_gemm.sm_mhz(kernel(1.14, "cycle/nsecond", 1, 1)) == pytest.approx(1140)
    assert profile_gemm.sm_mhz(kernel(824.7, "cycle/usecond", 1, 1)) == pytest.approx(824.7)
    summary = profile_gemm.side_summary([kernel(1.14, "cycle/nsecond", 1000, 900),
                                         kernel(1.14, "cycle/nsecond", 500, 450)])
    assert summary == {"kernels": 2, "total_duration_ns": 1350, "total_sm_cycles": 1500}
