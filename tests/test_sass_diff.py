"""scripts/sass_diff.py: comparing every kernel's SASS between two builds."""

import copy
import importlib.util
import json
import os
import subprocess
import sys

import pytest

import flint_lib

SCRIPT = flint_lib.ROOT / "scripts" / "sass_diff.py"
spec = importlib.util.spec_from_file_location("sass_diff", SCRIPT)
sass_diff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sass_diff)

# As cuobjdump prints a kernel, with the next file's header after it, since
# sass_loop.sass_functions leaves that header in the body of the file's last
# kernel.
BODY = """
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM75"
        /*0000*/                   MOV R1, c[0x0][0x28] ;                         /* 0x00000a0000017a02 */
                                                                                  /* 0x000fe40000000f00 */
        /*0010*/                   FADD R57, R57, R6 ;                            /* 0x0000000639397221 */
                                                                                  /* 0x000fe20000000000 */
        /*10000*/                  EXIT ;                                         /* 0x000000000000794d */
                                                                                  /* 0x000fea0003800000 */
		..........

Fatbin elf code:
================
arch = sm_75
identifier = C:/Users/someone/Flint/src/kernels/copy.cu
"""
LINES = [
    "/*0000*/                   MOV R1, c[0x0][0x28] ;                         /* 0x00000a0000017a02 */",
    "/* 0x000fe40000000f00 */",
    "/*0010*/                   FADD R57, R57, R6 ;                            /* 0x0000000639397221 */",
    "/* 0x000fe20000000000 */",
    "/*10000*/                  EXIT ;                                         /* 0x000000000000794d */",
    "/* 0x000fea0003800000 */",
]


def test_sass_lines_keep_instructions_and_control_words_only():
    assert sass_diff.sass_lines(BODY) == LINES


@pytest.mark.parametrize(
    "symbol, expected",
    [
        # Each hash's length differs, and what follows starts with digits.
        pytest.param(
            "_ZN5flint43_GLOBAL__N__9253c65b_10_convert_cu_0777972712fp32_to_fp16EP6__halfPKfy",
            "_ZN5flint<anon>12fp32_to_fp16EP6__halfPKfy",
            id="convert",
        ),
        pytest.param(
            "_ZN5flint39_GLOBAL__N__ec217d97_7_copy_cu_315831eb14copy_unalignedEPhPKhy",
            "_ZN5flint<anon>14copy_unalignedEPhPKhy",
            id="copy",
        ),
        # A namespace before the anonymous one.
        pytest.param(
            "_ZN5flint10microbench47_GLOBAL__N__e6b8b030_14_mma_kernels_cu_46d9b14b15mma_m16n8k8_f32ILi16EEEvfiP6float4",
            "_ZN5flint10microbench<anon>15mma_m16n8k8_f32ILi16EEEvfiP6float4",
            id="nested",
        ),
        pytest.param("_ZN5flint7gemm_ntEPv", "_ZN5flint7gemm_ntEPv", id="no_anonymous_namespace"),
    ],
)
def test_normalize_symbol(symbol, expected):
    assert sass_diff.normalize_symbol(symbol) == expected


def test_normalize_symbol_ignores_the_hash_and_keeps_kernels_apart():
    a = "_ZN5flint43_GLOBAL__N__9253c65b_10_convert_cu_0777972712fp32_to_fp16EP6__halfPKfy"
    b = "_ZN5flint43_GLOBAL__N__00000000_10_convert_cu_1234567812fp32_to_fp16EP6__halfPKfy"
    c = "_ZN5flint43_GLOBAL__N__9253c65b_10_convert_cu_0777972712fp32_to_fp17EP6__halfPKfy"
    assert sass_diff.normalize_symbol(a) == sass_diff.normalize_symbol(b)
    assert sass_diff.normalize_symbol(a) != sass_diff.normalize_symbol(c)


# -- Comparing ---------------------------------------------------------------


def dump_data(dirty=False):
    return {
        "schema": sass_diff.SCHEMA,
        "cuda": "12.4",
        "git": {"sha": "abc", "dirty": dirty},
        "binaries": {
            "library": {"sm_75": {"a": list(LINES), "b": list(LINES)}, "sm_89": {"a": list(LINES)}},
            "bench": {"sm_75": {"m": list(LINES)}, "sm_89": {"m": list(LINES)}},
        },
    }


def changed_entry(mutate):
    """Compares a dump with a copy that mutate changes, returning the library's
    sm_75 entry."""
    before = dump_data()
    after = copy.deepcopy(before)
    mutate(after["binaries"]["library"]["sm_75"])
    return sass_diff.compare(before["binaries"], after["binaries"])["library sm_75"]


def test_identical_dumps_have_no_differences():
    data = dump_data()
    for entry in sass_diff.compare(data["binaries"], copy.deepcopy(data["binaries"])).values():
        assert entry["new"] == entry["gone"] == entry["changed"] == []


@pytest.mark.parametrize(
    "line, text",
    [
        # The hand-written comparisons this script replaces missed this one: the
        # same instructions, with another stall count.
        pytest.param(1, "/* 0x000fe80000000f00 */", id="control_word"),
        pytest.param(2, LINES[2].replace("0x0000000639397221", "0x0000000639397222"), id="encoding"),
        pytest.param(2, LINES[2].replace("R6", "R7"), id="text"),
    ],
)
def test_any_changed_line_changes_the_kernel(line, text):
    def mutate(kernels):
        kernels["b"][line] = text

    entry = changed_entry(mutate)
    assert entry["changed"] == ["b"]
    assert entry["new"] == entry["gone"] == []


def test_a_reordered_kernel_changed():
    def mutate(kernels):
        kernels["b"][:2], kernels["b"][2:4] = kernels["b"][2:4], kernels["b"][:2]

    assert changed_entry(mutate)["changed"] == ["b"]


def test_new_and_gone_kernels():
    def mutate(kernels):
        kernels["c"] = kernels.pop("b")

    entry = changed_entry(mutate)
    assert (entry["new"], entry["gone"], entry["changed"]) == (["c"], ["b"], [])
    assert entry["kernels"] == 2


def test_a_change_is_reported_only_where_it_happened():
    before = dump_data()
    after = copy.deepcopy(before)
    after["binaries"]["bench"]["sm_89"]["m"][1] = "/* 0x000fe80000000f00 */"
    result = sass_diff.compare(before["binaries"], after["binaries"])
    assert {where for where, entry in result.items() if entry["changed"]} == {"bench sm_89"}


def test_an_architecture_missing_from_a_dump_counts_as_gone():
    before = dump_data()
    after = copy.deepcopy(before)
    del after["binaries"]["library"]["sm_89"]
    entry = sass_diff.compare(before["binaries"], after["binaries"])["library sm_89"]
    assert entry["gone"] == ["a"]


# -- The command line --------------------------------------------------------


def run_compare(tmp_path, before, after):
    (tmp_path / "before.json").write_text(json.dumps(before), encoding="utf-8")
    (tmp_path / "after.json").write_text(json.dumps(after), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "compare", str(tmp_path / "before.json"),
         str(tmp_path / "after.json")],
        capture_output=True, text=True, timeout=60,
    )


def test_cli_passes_with_only_new_kernels(tmp_path):
    before = dump_data()
    after = copy.deepcopy(before)
    after["binaries"]["library"]["sm_75"]["c"] = list(LINES)
    result = run_compare(tmp_path, before, after)
    assert result.returncode == 0, result.stderr
    assert "library sm_75: 3 kernels, 1 new, 0 gone, 0 changed" in result.stdout
    assert "  new c" in result.stdout
    assert "warning" not in result.stdout


def test_cli_fails_on_a_changed_kernel(tmp_path):
    before = dump_data()
    after = copy.deepcopy(before)
    after["binaries"]["library"]["sm_75"]["a"][1] = "/* 0x000fe80000000f00 */"
    result = run_compare(tmp_path, before, after)
    assert result.returncode == 1
    assert "  changed a" in result.stdout


def test_cli_fails_on_a_gone_kernel(tmp_path):
    before = dump_data()
    after = copy.deepcopy(before)
    del after["binaries"]["library"]["sm_75"]["a"]
    result = run_compare(tmp_path, before, after)
    assert result.returncode == 1
    assert "  gone a" in result.stdout


def test_cli_warns_when_before_is_dirty(tmp_path):
    result = run_compare(tmp_path, dump_data(dirty=True), dump_data(dirty=True))
    assert result.returncode == 0
    assert "warning: before was dumped from a dirty tree" in result.stdout


def test_cli_rejects_another_schema(tmp_path):
    result = run_compare(tmp_path, {"schema": "something/1"}, dump_data())
    assert result.returncode == 2
    assert "not a flint-sass-dump/1 file" in result.stderr


def test_cli_rejects_a_missing_file(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "compare", str(tmp_path / "no.json"), str(tmp_path / "no.json")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2
    assert "no.json" in result.stderr


# -- Dumping -----------------------------------------------------------------


def fake_checkout(tmp_path):
    """A checkout with both binaries built, newer than their sources."""
    for folder in ("src/kernels", "include", "bench", "build/release/bin"):
        (tmp_path / folder).mkdir(parents=True)
    sources = [tmp_path / "src/kernels/a.cu", tmp_path / "include/a.h", tmp_path / "bench/b.cpp"]
    for source in sources:
        source.write_text("x", encoding="utf-8")
        os.utime(source, (1000, 1000))
    for name, _ in sass_diff.BINARIES.values():
        binary = tmp_path / "build/release/bin" / name
        binary.write_text("x", encoding="utf-8")
        os.utime(binary, (2000, 2000))
    return tmp_path


@pytest.fixture
def fake_functions(monkeypatch):
    """cuobjdump's output is the same two kernels for every binary and arch."""
    functions = {
        "_ZN5flint39_GLOBAL__N__ec217d97_7_copy_cu_315831eb4copyEv": BODY,
        "_ZN5flint43_GLOBAL__N__9253c65b_10_convert_cu_077797274convEv": BODY,
    }
    monkeypatch.setattr(sass_diff.sass_loop, "sass_functions", lambda path, arch: dict(functions))
    return functions


def test_dump_records_each_binary_and_architecture(tmp_path, fake_functions):
    data = sass_diff.dump(fake_checkout(tmp_path))
    assert data["schema"] == sass_diff.SCHEMA
    assert set(data["binaries"]) == {"library", "bench"}
    for binary in data["binaries"].values():
        assert set(binary) == {"sm_75", "sm_89"}
        for kernels in binary.values():
            assert kernels == {"_ZN5flint<anon>4copyEv": LINES, "_ZN5flint<anon>4convEv": LINES}


@pytest.mark.parametrize(
    "source, binary",
    [
        pytest.param("src/kernels/a.cu", "library", id="kernel"),
        pytest.param("include/a.h", "library", id="header"),
        pytest.param("bench/b.cpp", "bench", id="bench"),
    ],
)
def test_dump_refuses_a_build_older_than_its_sources(tmp_path, fake_functions, source, binary):
    repo = fake_checkout(tmp_path)
    os.utime(repo / source, (3000, 3000))
    with pytest.raises(RuntimeError, match="build it first") as error:
        sass_diff.dump(repo)
    assert sass_diff.BINARIES[binary][0] in str(error.value)


def test_dump_refuses_a_missing_binary(tmp_path, fake_functions):
    repo = fake_checkout(tmp_path)
    (repo / "build/release/bin" / sass_diff.BENCH_NAME).unlink()
    with pytest.raises(RuntimeError, match="not found"):
        sass_diff.dump(repo)


def test_dump_refuses_a_binary_without_sass(tmp_path, monkeypatch):
    monkeypatch.setattr(sass_diff.sass_loop, "sass_functions", lambda path, arch: {})
    with pytest.raises(RuntimeError, match="found no SASS"):
        sass_diff.dump(fake_checkout(tmp_path))


def test_dump_refuses_a_kernel_without_sass_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(sass_diff.sass_loop, "sass_functions", lambda path, arch: {"k": "no code"})
    with pytest.raises(RuntimeError, match="found no SASS"):
        sass_diff.dump(fake_checkout(tmp_path))


def test_dump_refuses_kernels_that_normalize_alike(tmp_path, monkeypatch):
    functions = {
        "_ZN5flint39_GLOBAL__N__ec217d97_7_copy_cu_315831eb4copyEv": BODY,
        "_ZN5flint39_GLOBAL__N__00000000_7_copy_cu_315831eb4copyEv": BODY,
    }
    monkeypatch.setattr(sass_diff.sass_loop, "sass_functions", lambda path, arch: dict(functions))
    with pytest.raises(RuntimeError, match="normalize to"):
        sass_diff.dump(fake_checkout(tmp_path))


# -- The real build ----------------------------------------------------------


@pytest.mark.parametrize("name", [sass_diff.sass_loop.LIBRARY_NAME, sass_diff.BENCH_NAME])
def test_real_binaries_parse_and_match_themselves(name):
    path = flint_lib.library_path().parent / name
    if not path.is_file():
        pytest.fail(
            f"{path} not found; build it with "
            "`cmake --preset release && cmake --build --preset release`"
        )
    try:
        kernels = sass_diff.binary_kernels(path)
    except RuntimeError as error:
        pytest.fail(str(error))
    assert set(kernels) == {"sm_75", "sm_89"}
    for arch_kernels in kernels.values():
        for symbol, lines in arch_kernels.items():
            assert "GLOBAL__N" not in symbol
            # An instruction's line and its control word, in pairs, without any
            # of cuobjdump's description of the file.
            assert len(lines) % 2 == 0, symbol
            assert all(line.startswith("/*") for line in lines), symbol
            assert all("identifier" not in line for line in lines), symbol
    entries = sass_diff.compare({"x": kernels}, {"x": copy.deepcopy(kernels)})
    assert all(not entry["changed"] for entry in entries.values())
