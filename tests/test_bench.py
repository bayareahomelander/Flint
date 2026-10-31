"""flint_bench: the command line, and a quick end-to-end run with its JSON log."""

import json
import os
import re
import statistics
import subprocess
import sys

import pytest
import torch

import flint_lib

# scripts/sanitize.py runs the whole suite, flint_bench subprocesses included,
# under compute-sanitizer. The GPU then idles between launches and its clock
# drops, so clock-adjusted fractions can pass 100%.
UNDER_SANITIZER = os.environ.get("FLINT_SANITIZER") == "1"

EXE_NAME = "flint_bench.exe" if sys.platform == "win32" else "flint_bench"


@pytest.fixture(scope="module")
def exe():
    path = flint_lib.library_path().parent / EXE_NAME
    if not path.is_file():
        pytest.fail(
            f"{path} not found; build it with "
            "`cmake --preset release && cmake --build --preset release`"
        )
    return str(path)


def run(exe, *args, timeout=300):
    return subprocess.run(
        [exe, *args], capture_output=True, text=True, timeout=timeout, encoding="utf-8"
    )


def expected_git_sha():
    try:
        result = subprocess.run(
            ["git", "-C", str(flint_lib.ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def spellings(path):
    """A path as typed, with forward slashes, and as it appears escaped in JSON."""
    text = str(path)
    return {text, text.replace("\\", "/"), text.replace("\\", "\\\\")}


def assert_stats_follow_from_rounds(stats, windows):
    """A side's summary stats merge its valid rounds' sample windows: counts
    add up, means and shares are weighted by each window's seconds, and the
    maxima and the new-process flag carry over."""
    seconds = sum(w["seconds"] for w in windows)
    assert stats["seconds"] == pytest.approx(seconds)
    assert stats["samples"] == sum(w["samples"] for w in windows)
    assert stats["failed_reads"] == sum(w["failed_reads"] for w in windows)

    def mean(get):
        return pytest.approx(sum(get(w) * w["seconds"] for w in windows) / seconds)

    for name in ("sm_mhz", "mem_mhz", "power_w", "thermal", "hw_slowdown", "sw_power_cap"):
        assert stats[name] == mean(lambda w: w[name]), name
    assert stats["reason_shares"].keys() == windows[0]["reason_shares"].keys()
    for reason in stats["reason_shares"]:
        assert stats["reason_shares"][reason] == mean(lambda w: w["reason_shares"][reason]), reason
    assert stats["temp_c_max"] == max(w["temp_c_max"] for w in windows)
    assert stats["used_bytes_max"] == max(w["used_bytes_max"] for w in windows)
    assert stats["new_process"] == any(w["new_process"] for w in windows)


def assert_baseline_algorithm(algorithm, selection):
    """A cuBLASLt case's candidates: 1 to 8 of them in the heuristic's order,
    with their configurations, at least one that launched and was timed, and
    the one used: the fastest of those with fastest, the first with first."""
    assert algorithm["selection"] == selection
    candidates = algorithm["candidates"]
    assert 1 <= len(candidates) <= 8
    assert [c["index"] for c in candidates] == list(range(len(candidates)))
    config = ("algo_id", "tile_id", "stages_id", "splitk_num", "reduction_scheme",
              "cta_swizzling", "custom_option")
    for c in candidates:
        assert set(c) == {"index", *config, "workspace_bytes", "waves_count", "status",
                          "ms_per_launch"}
        for name in config:
            assert isinstance(c[name], int) and c[name] >= 0, name
        assert c["splitk_num"] >= 1
        assert c["workspace_bytes"] >= 0
        assert c["waves_count"] > 0
        if c["status"] == "CUBLAS_STATUS_SUCCESS":
            assert c["ms_per_launch"] > 0
        else:
            assert c["status"].startswith("CUBLAS_STATUS_")
            assert c["ms_per_launch"] is None
    timed = [c for c in candidates if c["ms_per_launch"] is not None]
    assert timed
    used = algorithm["used"]
    assert 0 <= used < len(candidates)
    if selection == "first":
        assert used == 0
    else:
        assert candidates[used]["ms_per_launch"] == min(c["ms_per_launch"] for c in timed)


def test_help(exe):
    result = run(exe, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage: flint_bench" in result.stdout
    for text in (
        "copy", "mma", "gemm", "--sizes", "--chains", "--shapes", "--layout", "tn", "--epilogue", "bias",
        "bias-gelu", "dgelu", "bias-residual", "--lt-algo", "fastest", "first"
    ):
        assert text in result.stdout


def test_unknown_option(exe):
    result = run(exe, "copy", "--bogus")
    assert result.returncode == 2
    assert "unknown option: --bogus" in result.stderr


def test_unknown_benchmark(exe):
    result = run(exe, "nope")
    assert result.returncode == 2
    assert "unknown benchmark: nope" in result.stderr


def test_busy_gpu_is_refused_without_allow_busy(exe, tmp_path):
    # Whether the GPU counts as busy depends on the machine (the laptop's
    # desktop always uses it), so either outcome passes, but a refusal must
    # say why and how to override it.
    result = run(exe, "copy", "--quick", "--sizes", "4K", "--out", str(tmp_path / "b.json"))
    if result.returncode != 0:
        assert result.returncode == 1
        assert "the GPU is busy" in result.stderr
        assert "--allow-busy" in result.stderr


def test_quick_copy(exe, tmp_path):
    out = tmp_path / "bench.json"
    args = ["copy", "--quick", "--sizes", "1M,4097", "--allow-busy"]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    text = out.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["schema"] == "flint-bench/1"
    assert data["quick"] is True
    assert data["options"]["quick"] is True
    assert data["options"]["lt_algo"] == "fastest"
    assert data["standard_protocol"] is False
    assert isinstance(data["gpu_at_start"]["process_detection"], bool)
    assert 0 <= data["gpu_at_start"]["utilization_percent"] <= 100

    assert data["args"] == args
    for path in (tmp_path, flint_lib.ROOT):
        for spelling in spellings(path):
            assert spelling not in text

    assert data["git"]["sha"] == expected_git_sha()
    assert isinstance(data["git"]["dirty"], bool)

    device = data["device"]
    major, minor = torch.cuda.get_device_capability(0)
    assert device["name"] == torch.cuda.get_device_name(0)
    assert device["compute_capability"] == f"{major}.{minor}"
    assert device["rated_bandwidth_gbs"] > 0

    options = data["options"]
    rated = device["rated_bandwidth_gbs"]
    cases = data["cases"]
    assert [c["label"] for c in cases] == ["1 MiB", "4097 B"]
    for case in cases:
        assert case["layout"] is None
        assert case["epilogue"] is None
        assert case["ok"] and case["outputs_match"]
        assert case["output_check"] is None
        assert case["pre_check"] is None
        assert case["baseline_algorithm"] is None
        assert case["warmup_seconds"] >= options["warmup_seconds"]
        valid = [r for r in case["rounds"] if r["valid"]]
        assert case["valid_rounds"] == len(valid) >= options["rounds"]

        # The summary must follow from the rounds: each ratio is baseline time
        # over Flint time, and the stats cover the valid rounds only.
        for r in valid:
            assert r["ratio"] == pytest.approx(
                r["baseline"]["ms_per_launch"] / r["flint"]["ms_per_launch"]
            )
            for sample in (r["flint"], r["baseline"]):
                assert sample["sample_ms"] >= 0.9 * options["sample_ms"]
        ratios = [r["ratio"] for r in valid]
        assert case["ratio"]["median"] == pytest.approx(statistics.median(ratios))
        assert case["ratio"]["min"] == pytest.approx(min(ratios))
        assert case["ratio"]["max"] == pytest.approx(max(ratios))
        assert [r["flint_first"] for r in valid] == [i % 2 == 0 for i in range(len(valid))]

        for name in ("flint", "baseline"):
            side = case[name]
            assert side["min_sample_ms"] >= 0.9 * options["sample_ms"]
            assert 0 < side["gbs"] <= 1.1 * rated
            assert side["gbs"] == pytest.approx(
                case["bytes_moved"] / (side["ms_per_launch"] * 1e-3) / 1e9
            )
            assert side["rated_fraction"] == pytest.approx(side["gbs"] / rated)
            # The memory clock never exceeds its maximum, so clock-adjusted >= rated.
            assert side["clock_adjusted_fraction"] >= 0.99 * side["rated_fraction"]
            stats = side["stats"]
            assert stats["samples"] > 0 and stats["sm_mhz"] > 0 and stats["mem_mhz"] > 0
            assert len(stats["reason_shares"]) == 9
            assert all(0 <= share <= 1 for share in stats["reason_shares"].values())
            assert_stats_follow_from_rounds(stats, [r[name]["stats"] for r in valid])

        stride = (case["size_bytes"] + 255) // 256 * 256
        assert case["rotation_sets"] * 2 * stride > 4 * device["l2_bytes"]


def test_sizes_rejected_for_mma(exe):
    result = run(exe, "mma", "--sizes", "1M")
    assert result.returncode == 2
    assert "--sizes applies to copy only" in result.stderr


def test_bad_chains_rejected(exe):
    result = run(exe, "mma", "--chains", "5")
    assert result.returncode == 2
    assert "--chains" in result.stderr and "'5'" in result.stderr


def test_chains_rejected_for_copy(exe):
    result = run(exe, "copy", "--chains", "4")
    assert result.returncode == 2
    assert "--chains applies to mma only" in result.stderr


def test_shapes_rejected_for_copy_and_mma(exe):
    for benchmark in ("copy", "mma"):
        result = run(exe, benchmark, "--shapes", "8x8x8")
        assert result.returncode == 2
        assert "--shapes applies to gemm only" in result.stderr


def test_layout_rejected_for_copy_and_mma(exe):
    for benchmark in ("copy", "mma"):
        result = run(exe, benchmark, "--layout", "nt")
        assert result.returncode == 2
        assert result.stderr.splitlines()[0] == "flint_bench: --layout applies to gemm only"


def test_epilogue_rejected_for_copy_and_mma(exe):
    for benchmark in ("copy", "mma"):
        result = run(exe, benchmark, "--epilogue", "bias")
        assert result.returncode == 2
        assert result.stderr.splitlines()[0] == "flint_bench: --epilogue applies to gemm only"


def test_lt_algo_rejected_for_copy_and_mma(exe):
    for benchmark in ("copy", "mma"):
        for value in ("fastest", "first"):
            result = run(exe, benchmark, "--lt-algo", value)
            assert result.returncode == 2
            assert result.stderr.splitlines()[0] == "flint_bench: --lt-algo applies to gemm only"


@pytest.mark.parametrize("value", ["Fastest", "second", "all", "1", ""])
def test_bad_lt_algo_rejected(exe, value):
    result = run(exe, "gemm", "--lt-algo", value)
    assert result.returncode == 2, result.stdout
    first_line = result.stderr.splitlines()[0]
    assert first_line == f"flint_bench: --lt-algo: must be fastest or first: '{value}'"


def test_lt_algo_accepted_for_gemm(exe):
    # --help skips the run, but the parser still reads the option.
    for args in (
        ("--lt-algo", "first"),
        ("--lt-algo=fastest",),
        ("--epilogue", "bias", "--lt-algo", "first"),
    ):
        result = run(exe, "gemm", *args, "--help")
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("epilogue", ["Bias", "relu", "gelu", "bias_gelu", ""])
def test_bad_epilogue_rejected(exe, epilogue):
    result = run(exe, "gemm", "--epilogue", epilogue)
    assert result.returncode == 2, result.stdout
    first_line = result.stderr.splitlines()[0]
    assert first_line.startswith("flint_bench: --epilogue:")
    assert "none, bias, bias-gelu, bias-residual or dgelu" in first_line


@pytest.mark.parametrize("epilogue", ["bias", "bias-gelu", "bias-residual"])
@pytest.mark.parametrize("layout", ["nn", "tn"])
def test_bias_needs_layout_nt(exe, layout, epilogue):
    # The order of --layout and --epilogue doesn't matter.
    for args in (
        ("--layout", layout, "--epilogue", epilogue),
        ("--epilogue", epilogue, "--layout", layout),
    ):
        result = run(exe, "gemm", *args)
        assert result.returncode == 2, result.stdout
        assert result.stderr.splitlines()[0] == f"flint_bench: --epilogue {epilogue} needs --layout nt"
    # nt takes it, and none goes with any layout. --help skips the run.
    for args in (
        ("--layout", "nt", "--epilogue", epilogue),
        ("--epilogue", epilogue),
        ("--layout", layout, "--epilogue", "none"),
    ):
        result = run(exe, "gemm", *args, "--help")
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("layout", ["nt", "tn"])
def test_dgelu_needs_layout_nn(exe, layout):
    # The order of --layout and --epilogue doesn't matter. nt is the default,
    # so it is also tested left out.
    for args in (
        ("--layout", layout, "--epilogue", "dgelu"),
        ("--epilogue", "dgelu", "--layout", layout),
        *((("--epilogue", "dgelu"),) if layout == "nt" else ()),
    ):
        result = run(exe, "gemm", *args)
        assert result.returncode == 2, result.stdout
        assert result.stderr.splitlines()[0] == "flint_bench: --epilogue dgelu needs --layout nn"
    # nn takes it. --help skips the run.
    for args in (
        ("--layout", "nn", "--epilogue", "dgelu"),
        ("--epilogue", "dgelu", "--layout", "nn"),
    ):
        result = run(exe, "gemm", *args, "--help")
        assert result.returncode == 0, result.stderr


@pytest.mark.skipif(UNDER_SANITIZER, reason="full-size shapes, and only the parser's choice is tested")
def test_bias_defaults_to_the_biased_forward_shapes(exe, tmp_path):
    # GPT-2's logits layer has no bias, so its shape is left out.
    out = tmp_path / "bias.json"
    args = ["gemm", "--epilogue", "bias", "--quick", "--allow-busy"]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    cases = json.loads(out.read_text(encoding="utf-8"))["cases"]
    assert [c["label"] for c in cases] == [
        "8192x2304x768", "8192x768x768", "8192x3072x768", "8192x768x3072"
    ]


@pytest.mark.skipif(UNDER_SANITIZER, reason="full-size shapes, and only the parser's choice is tested")
def test_bias_gelu_defaults_to_the_one_gelu_layer(exe, tmp_path):
    # Only the MLP's up projection has a GELU after it.
    out = tmp_path / "bias_gelu.json"
    args = ["gemm", "--epilogue", "bias-gelu", "--quick", "--allow-busy"]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    cases = json.loads(out.read_text(encoding="utf-8"))["cases"]
    assert [c["label"] for c in cases] == ["8192x3072x768"]


@pytest.mark.skipif(UNDER_SANITIZER, reason="full-size shapes, and only the parser's choice is tested")
def test_bias_residual_defaults_to_the_two_layers_feeding_the_residual_stream(exe, tmp_path):
    # The attention projection and the MLP's down projection.
    out = tmp_path / "bias_residual.json"
    args = ["gemm", "--epilogue", "bias-residual", "--quick", "--allow-busy"]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    cases = json.loads(out.read_text(encoding="utf-8"))["cases"]
    assert [c["label"] for c in cases] == ["8192x768x768", "8192x768x3072"]


@pytest.mark.skipif(UNDER_SANITIZER, reason="full-size shapes, and only the parser's choice is tested")
def test_dgelu_defaults_to_the_one_gelu_layer(exe, tmp_path):
    # Only the MLP's up projection has a GELU in front of its input gradient.
    out = tmp_path / "dgelu.json"
    args = ["gemm", "--layout", "nn", "--epilogue", "dgelu", "--quick", "--allow-busy"]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    cases = json.loads(out.read_text(encoding="utf-8"))["cases"]
    assert [c["label"] for c in cases] == ["8192x3072x768"]


@pytest.mark.parametrize("layout", ["TN", "NN", "nx", ""])
def test_bad_layout_rejected(exe, layout):
    result = run(exe, "gemm", "--layout", layout)
    assert result.returncode == 2, result.stdout
    first_line = result.stderr.splitlines()[0]
    assert first_line.startswith("flint_bench: --layout:")
    assert "nt, nn or tn" in first_line


def test_nn_needs_n_a_multiple_of_8(exe):
    # The order of --layout and --shapes doesn't matter.
    for args in (
        ("--layout", "nn", "--shapes", "8x12x8"),
        ("--shapes", "8x12x8", "--layout", "nn"),
    ):
        result = run(exe, "gemm", *args)
        assert result.returncode == 2, result.stdout
        first_line = result.stderr.splitlines()[0]
        assert first_line.startswith("flint_bench: --shapes:")
        assert "N must be a multiple of 8" in first_line
    # nt takes the same shape.
    result = run(exe, "gemm", "--layout", "nt", "--shapes", "8x12x8", "--help")
    assert result.returncode == 0, result.stderr
    result = run(exe, "gemm", "--shapes", "8x12x8", "--layout", "nn", "--help")
    assert result.returncode == 2


def test_bias_gelu_needs_n_a_multiple_of_8(exe):
    # cuBLASLt's GELU_AUX_BIAS needs its aux output's leading dimension, N, to
    # be a multiple of 8; a bias alone doesn't.
    for args in (
        ("--epilogue", "bias-gelu", "--shapes", "8x12x8"),
        ("--shapes", "8x12x8", "--epilogue", "bias-gelu"),
    ):
        result = run(exe, "gemm", *args)
        assert result.returncode == 2, result.stdout
        first_line = result.stderr.splitlines()[0]
        assert first_line == (
            "flint_bench: --shapes: N must be a multiple of 8 with --epilogue bias-gelu: '8x12x8'"
        )
    result = run(exe, "gemm", "--epilogue", "bias", "--shapes", "8x12x8", "--help")
    assert result.returncode == 0, result.stderr


def test_bias_residual_needs_n_even(exe):
    # flint_gemm_nt_bias_residual reads and writes pairs of FP32 values. The
    # order of --epilogue and --shapes doesn't matter.
    for args in (
        ("--epilogue", "bias-residual", "--shapes", "8x13x8"),
        ("--shapes", "8x13x8", "--epilogue", "bias-residual"),
        ("--epilogue", "bias-residual", "--shapes", "8x12x8,8x7x8"),
    ):
        result = run(exe, "gemm", *args)
        assert result.returncode == 2, result.stdout
        first_line = result.stderr.splitlines()[0]
        assert first_line.startswith("flint_bench: --shapes: N must be even with --epilogue bias-residual: '8x")
        assert first_line.endswith("x8'")
    # An even N that is not a multiple of 8 is fine, and so is an odd N with
    # the other epilogues. --help skips the run.
    result = run(exe, "gemm", "--epilogue", "bias-residual", "--shapes", "8x14x8,8x2x8", "--help")
    assert result.returncode == 0, result.stderr
    result = run(exe, "gemm", "--epilogue", "bias", "--shapes", "8x13x8", "--help")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "shape, dimension",
    [("12x8x8", "M"), ("8x12x8", "N"), ("12x12x8", "M")],
)
def test_tn_needs_m_and_n_multiples_of_8(exe, shape, dimension):
    for args in (
        ("--layout", "tn", "--shapes", shape),
        ("--shapes", shape, "--layout", "tn"),
    ):
        result = run(exe, "gemm", *args)
        assert result.returncode == 2, result.stdout
        first_line = result.stderr.splitlines()[0]
        assert first_line.startswith("flint_bench: --shapes:")
        assert f"{dimension} must be a multiple of 8 with --layout tn" in first_line


def test_tn_takes_any_k(exe):
    # K = 1001 and K = 1 are fine for tn and refused for nt and nn; the limits
    # still apply. --help skips the run, but the parser still reads the list.
    result = run(exe, "gemm", "--layout", "tn", "--shapes", "8x8x1001,16x8x1,8x8x65536", "--help")
    assert result.returncode == 0, result.stderr
    for layout in ("nt", "nn"):
        result = run(exe, "gemm", "--layout", layout, "--shapes", "8x8x1001", "--help")
        assert result.returncode == 2
        assert "K must be a multiple of 8" in result.stderr.splitlines()[0]
    result = run(exe, "gemm", "--layout", "tn", "--shapes", "8x8x65537", "--help")
    assert result.returncode == 2
    assert "limits" in result.stderr.splitlines()[0]


def test_sizes_and_chains_rejected_for_gemm(exe):
    result = run(exe, "gemm", "--sizes", "1M")
    assert result.returncode == 2
    assert "--sizes applies to copy only" in result.stderr
    result = run(exe, "gemm", "--chains", "4")
    assert result.returncode == 2
    assert "--chains applies to mma only" in result.stderr


@pytest.mark.parametrize(
    "shapes, reason",
    [
        ("8x8", "not a shape"),
        ("8x8x8x8", "not a shape"),
        ("axbxc", "not a shape"),
        ("8x-8x8", "not a shape"),
        ("8x 8x8", "not a shape"),
        ("8x8x8,", "not a shape"),
        ("0x8x8", "positive"),
        ("8x0x8", "positive"),
        ("8x8x0", "positive"),
        ("8x8x12", "multiple of 8"),
        ("8388481x8x8", "limits"),
        ("8x8388481x8", "limits"),
        ("8x8x65544", "limits"),
        ("99999999999999999999x8x8", "not a shape"),
    ],
)
def test_bad_shapes_rejected(exe, shapes, reason):
    result = run(exe, "gemm", "--shapes", shapes)
    assert result.returncode == 2, result.stdout
    first_line = result.stderr.splitlines()[0]
    assert first_line.startswith("flint_bench: --shapes:")
    assert reason in first_line


def test_shapes_at_the_limits_are_accepted_by_the_parser(exe):
    # --help skips the run, but the parser still reads the list.
    result = run(exe, "gemm", "--shapes", "8388480x8388480x65536,1x8x8", "--help")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "layout, epilogue, shape_list",
    [
        ("nt", "none", "1000x2304x768,129x257x40"),
        # The same shapes against cuBLASLt: an odd N and M, and a small K.
        ("nt", "bias", "1000x2304x768,129x257x40"),
        # And with a GELU after it, which also stores the sums before it.
        # cuBLASLt's GELU_AUX_BIAS needs N a multiple of 8: 264 here.
        ("nt", "bias-gelu", "1000x3072x768,129x264x40"),
        # And with the FP32 residual stream added: N must be even, and 266
        # isn't a multiple of 8.
        ("nt", "bias-residual", "1000x768x3072,129x266x40"),
        # nn's N must be a multiple of 8: 264 here, with an odd M and a small K.
        ("nn", "none", "1000x768x2304,129x264x40"),
        # And with GELU's derivative applied, reading a pre as big as y.
        ("nn", "dgelu", "1000x3072x768,129x264x40"),
        # tn's M and N must be; K needn't be, and 1001 isn't. 768x768x8192
        # splits k on the 2080 Ti only; 136x120x3000 and 1032x8x4096 split on
        # both GPUs, the second with two groups of rows of tiles.
        ("tn", "none", "768x768x8192,136x120x3000,1032x8x4096,72x40x1001"),
    ],
)
def test_quick_gemm(exe, tmp_path, layout, epilogue, shape_list):
    out = tmp_path / "gemm.json"
    # nt without a bias is the default, so its run leaves --layout and
    # --epilogue out.
    args = ["gemm", "--quick", "--shapes", shape_list, "--allow-busy"]
    if layout != "nt":
        args[1:1] = ["--layout", layout]
    if epilogue != "none":
        args[1:1] = ["--epilogue", epilogue]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    # The summary's rows name an epilogue beside the layout, unless it is none.
    first = shape_list.split(",")[0]
    for name in ("none", "bias", "bias-gelu", "dgelu", "bias-residual"):
        label = f"gemm {layout}" + ("" if name == "none" else f"+{name}") + f" {first}"
        assert (label in result.stdout) == (epilogue == name), label

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "flint-bench/1"
    assert data["benchmark"] == "gemm"
    assert data["args"] == args

    device = data["device"]
    options = data["options"]
    assert options["lt_algo"] == "fastest"
    geforce_75_or_89 = "GeForce" in device["name"] and device["compute_capability"] in ("7.5", "8.9")
    cases = data["cases"]
    shapes = [tuple(int(d) for d in item.split("x")) for item in shape_list.split(",")]
    assert [c["label"] for c in cases] == shape_list.split(",")
    for case, (m, n, k) in zip(cases, shapes):
        assert case["benchmark"] == "gemm"
        assert case["layout"] == layout
        assert case["epilogue"] == epilogue
        assert case["baseline_name"] == {
            "none": "cuBLAS", "bias": "cuBLASLt, FP16 bias", "bias-gelu": "cuBLASLt, FP16 bias",
            "dgelu": "cuBLASLt DGELU", "bias-residual": "cuBLASLt, residual in C",
        }[epilogue]
        assert case["one_sided"] is False
        assert case["ok"] and case["outputs_match"]
        assert case["warmup_seconds"] >= options["warmup_seconds"]
        # A plain GEMM's baseline is cublasGemmEx's default algorithm; the
        # cuBLASLt cases log their candidates and use the fastest.
        if epilogue == "none":
            assert case["baseline_algorithm"] is None
        else:
            assert_baseline_algorithm(case["baseline_algorithm"], "fastest")

        # With a GELU, output_check is its output and pre_check the sums
        # before it, the second output; both are compared with cuBLASLt's.
        checks = [case["output_check"]]
        if epilogue == "bias-gelu":
            checks.append(case["pre_check"])
        else:
            assert case["pre_check"] is None
        for check in checks:
            assert check["deterministic"] is True
            assert check["nonfinite"] == 0
            assert 0 <= check["rel_l2"] <= 2**-10
            assert check["max_abs"] >= 0 and check["max_abs_baseline"] > 0

        assert case["flops"] == 2 * m * n * k
        assert case["bytes_moved"] == 0
        if geforce_75_or_89:
            assert case["peak_flop_per_sm_per_clock"] == 512

        valid = [r for r in case["rounds"] if r["valid"]]
        assert case["valid_rounds"] == len(valid) >= options["rounds"]
        for r in valid:
            assert r["ratio"] == pytest.approx(
                r["baseline"]["ms_per_launch"] / r["flint"]["ms_per_launch"]
            )
            for sample in (r["flint"], r["baseline"]):
                assert sample["sample_ms"] >= 0.9 * options["sample_ms"]
        ratios = [r["ratio"] for r in valid]
        assert case["ratio"]["median"] == pytest.approx(statistics.median(ratios))
        assert case["ratio"]["min"] == pytest.approx(min(ratios))
        assert case["ratio"]["max"] == pytest.approx(max(ratios))
        assert [r["flint_first"] for r in valid] == [i % 2 == 0 for i in range(len(valid))]

        for name in ("flint", "baseline"):
            side = case[name]
            assert side["min_sample_ms"] >= 0.9 * options["sample_ms"]
            assert side["tflops"] > 0
            assert side["tflops"] == pytest.approx(
                case["flops"] / (side["ms_per_launch"] * 1e-3) / 1e12
            )
            assert side["gbs"] == 0
            assert side["clock_adjusted_compute_fraction"] > 0
            stats = side["stats"]
            assert stats["samples"] > 0 and stats["sm_mhz"] > 0
            assert_stats_follow_from_rounds(stats, [r[name]["stats"] for r in valid])

        # The sets together must exceed 4x the L2 cache: size_bytes is one
        # set's x, w and y (FP32 for tn and bias-residual), plus pre (as big as
        # y) with a GELU or its derivative, or the FP32 residual (the same size
        # as y) with bias-residual, and the FP32 bias if there is one.
        assert case["size_bytes"] == (
            2 * (m * k + n * k)
            + m * n * (4 if layout == "tn" or epilogue == "bias-residual" else 2)
            + (2 * m * n if epilogue in ("bias-gelu", "dgelu") else 0)
            + (4 * m * n if epilogue == "bias-residual" else 0)
            + (4 * n if epilogue in ("bias", "bias-gelu", "bias-residual") else 0)
        )
        assert case["rotation_sets"] * case["size_bytes"] > 4 * device["l2_bytes"]


@pytest.mark.parametrize("layout, epilogue", [("nt", "bias"), ("nn", "dgelu")])
def test_quick_gemm_lt_algo_first(exe, tmp_path, layout, epilogue):
    # --lt-algo first uses the heuristic's first candidate, and still times the
    # others and logs them.
    out = tmp_path / "first.json"
    args = ["gemm", "--quick", "--layout", layout, "--epilogue", epilogue, "--lt-algo", "first",
            "--shapes", "1000x768x768", "--allow-busy"]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["args"] == args
    assert data["options"]["lt_algo"] == "first"
    (case,) = data["cases"]
    assert case["ok"] and case["outputs_match"]
    algorithm = case["baseline_algorithm"]
    assert_baseline_algorithm(algorithm, "first")
    assert algorithm["used"] == 0
    assert algorithm["candidates"][0]["ms_per_launch"] is not None


def test_lt_algo_first_leaves_a_plain_gemm_alone(exe, tmp_path):
    # Without an epilogue the baseline is cublasGemmEx's default algorithm: the
    # option is recorded and changes nothing else.
    out = tmp_path / "plain.json"
    args = ["gemm", "--quick", "--lt-algo", "first", "--shapes", "1000x768x768", "--allow-busy"]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["options"]["lt_algo"] == "first"
    assert data["cases"][0]["baseline_algorithm"] is None


def test_quick_mma(exe, tmp_path):
    out = tmp_path / "mma.json"
    args = ["mma", "--quick", "--chains", "4,8,16", "--allow-busy"]
    result = run(exe, *args, "--out", str(out))
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "flint-bench/1"
    assert data["args"] == args

    device = data["device"]
    options = data["options"]
    rated_clock = device["rated_sm_clock_mhz"]
    cases = data["cases"]
    labels = [re.fullmatch(r"m16n8k8 f16->f32, (\d+) chains, (\d+) blocks/SM", c["label"]) for c in cases]
    assert all(labels), [c["label"] for c in cases]
    assert [int(m.group(1)) for m in labels] == [4, 8, 16]
    geforce_75_or_89 = "GeForce" in device["name"] and device["compute_capability"] in ("7.5", "8.9")
    for case, label in zip(cases, labels):
        assert case["one_sided"] is True
        assert case["layout"] is None
        assert case["epilogue"] is None
        assert case["baseline"] is None and case["ratio"] is None
        assert case["ok"] and case["outputs_match"]
        assert case["output_check"] is None
        assert case["pre_check"] is None
        assert case["baseline_algorithm"] is None
        assert case["warmup_seconds"] >= options["warmup_seconds"]
        # One full wave: blocks/SM x SMs blocks of 8 warps, each warp running
        # 32768 mma steps of 2048 FLOP.
        blocks_per_sm = int(label.group(2))
        assert blocks_per_sm >= 1
        assert case["flops"] == blocks_per_sm * device["sm_count"] * 8 * 32768 * 2048
        if geforce_75_or_89:
            assert case["peak_flop_per_sm_per_clock"] == 512

        valid = [r for r in case["rounds"] if r["valid"]]
        assert case["valid_rounds"] == len(valid) >= options["rounds"]
        for r in case["rounds"]:
            assert r["baseline"] is None and r["ratio"] is None
        for r in valid:
            assert r["flint"]["sample_ms"] >= 0.9 * options["sample_ms"]

        flint = case["flint"]
        assert flint["min_sample_ms"] >= 0.9 * options["sample_ms"]
        assert flint["ms_per_launch"] == pytest.approx(
            statistics.median(r["flint"]["ms_per_launch"] for r in valid)
        )
        assert flint["tflops"] > 0
        assert flint["tflops"] == pytest.approx(
            case["flops"] / (flint["ms_per_launch"] * 1e-3) / 1e12
        )
        assert flint["clock_adjusted_compute_fraction"] > 0
        if not UNDER_SANITIZER:
            # The 2080 Ti measures 99% of the clock-adjusted peak and the laptop
            # 82-94%, depending on its power state; far below that means a
            # kernel that doesn't saturate the tensor cores.
            assert 0.6 <= flint["clock_adjusted_compute_fraction"] <= 1.05
        if rated_clock is None:
            assert flint["rated_compute_fraction"] is None
        else:
            peak = case["peak_flop_per_sm_per_clock"] * device["sm_count"] * rated_clock * 1e6 / 1e12
            assert flint["rated_compute_fraction"] == pytest.approx(flint["tflops"] / peak)
        stats = flint["stats"]
        assert stats["samples"] > 0 and stats["sm_mhz"] > 0
        assert_stats_follow_from_rounds(stats, [r["flint"]["stats"] for r in valid])
