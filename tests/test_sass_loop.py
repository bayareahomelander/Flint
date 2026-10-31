"""scripts/sass_loop.py: the static profile of the GEMM main loops."""

import copy
import getpass
import importlib.util
import json
import re
import subprocess
import sys

import pytest

import flint_lib

spec = importlib.util.spec_from_file_location(
    "sass_loop", flint_lib.ROOT / "scripts" / "sass_loop.py"
)
sass_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sass_loop)


def control(stall=0, yield_flag=0, write=7, read=7, wait=0, reuse=0):
    """A control word with the given fields set (the barriers 7 for none)."""
    return hex(
        (stall << 41) | (yield_flag << 45) | (write << 46) | (read << 49) | (wait << 52)
        | (reuse << 58)
    )


# -- Decoding and statistics on synthetic SASS --------------------------------


def test_decode_control_reads_each_field():
    # Every field alone, with the others at their "nothing" values.
    assert sass_loop.decode_control(control()) == {
        "stall": 0, "yield": 0, "write_barrier": None, "read_barrier": None,
        "wait_mask": 0, "reuse": 0,
    }
    assert sass_loop.decode_control(control(stall=13))["stall"] == 13
    assert sass_loop.decode_control(control(yield_flag=1))["yield"] == 1
    assert sass_loop.decode_control(control(write=0))["write_barrier"] == 0
    assert sass_loop.decode_control(control(write=5))["write_barrier"] == 5
    assert sass_loop.decode_control(control(read=3))["read_barrier"] == 3
    assert sass_loop.decode_control(control(wait=0b100101))["wait_mask"] == 0b100101
    assert sass_loop.decode_control(control(reuse=0b1010))["reuse"] == 0b1010
    # All together, from a string and from an int, and no field leaks into
    # another.
    word = control(stall=9, yield_flag=1, write=2, read=4, wait=0b011010, reuse=0b0110)
    expected = {"stall": 9, "yield": 1, "write_barrier": 2, "read_barrier": 4,
                "wait_mask": 0b011010, "reuse": 0b0110}
    assert sass_loop.decode_control(word) == expected
    assert sass_loop.decode_control(int(word, 16)) == expected


def test_decode_control_of_a_real_word():
    # Taken from an LDSM in gemm_nt's main loop: stall 8, yield, writes
    # barrier 0, holds none, waits for none.
    assert sass_loop.decode_control("0x000e300000000200") == {
        "stall": 8, "yield": 1, "write_barrier": 0, "read_barrier": None,
        "wait_mask": 0, "reuse": 0,
    }


def test_wait_mask_matches_the_sass_tests_decoding():
    word = control(wait=0b101001, stall=15, write=1)
    assert sass_loop.decode_control(word)["wait_mask"] == (int(word, 16) >> 52) & 0x3F


def test_load_to_use_of_a_synthetic_loop():
    # The LDG (stall 2) sets barrier 2. Two instructions follow with stalls 3
    # and 4, then an STS waits on barrier 2 (mask 0b000100). The waiter is 3
    # instructions after the load, and the schedule gives the load 2 + 3 + 4 =
    # 9 cycles before it: the stalls of the instructions from the load up to,
    # not including, the waiter.
    loop = [
        ("LDG.E.128.CONSTANT.SYS R4, [R2.64] ;", control(stall=2, write=2)),
        ("IMAD.MOV.U32 R6, RZ, RZ, R7 ;", control(stall=3)),
        ("IADD3 R8, R8, 0x20, RZ ;", control(stall=4)),
        ("STS.128 [R9], R4 ;", control(stall=5, wait=0b000100)),
        ("BRA 0x10 ;", control(stall=6)),
    ]
    assert sass_loop.load_to_use(loop) == [{
        "index": 0,
        "opcode": "LDG.E.128.CONSTANT.SYS",
        "barrier": 2,
        "waiter_index": 3,
        "waiter_opcode": "STS.128",
        "distance": 3,
        "static_cycles": 9,
    }]


def test_load_to_use_skips_other_barriers_and_earlier_waiters():
    loop = [
        # Waits on barrier 2 before the load is issued: not its waiter.
        ("IADD3 R1, R1, 0x1, RZ ;", control(stall=1, wait=0b000100)),
        ("LDSM.16.M88.4 R20, [R3] ;", control(stall=1, write=2)),
        # Waits on barriers 0, 1 and 3, not 2.
        ("IMAD R2, R2, R3, RZ ;", control(stall=2, wait=0b001011)),
        ("HMMA.1688.F32 R8, R20, R22, R8 ;", control(stall=8, wait=0b010100)),
        ("HMMA.1688.F32 R12, R20, R22, R12 ;", control(stall=8)),
    ]
    (found,) = sass_loop.load_to_use(loop)
    assert (found["waiter_index"], found["waiter_opcode"]) == (3, "HMMA.1688.F32")
    assert found["distance"] == 2
    assert found["static_cycles"] == 1 + 2


def test_load_to_use_without_a_waiter_runs_to_the_loops_end():
    loop = [
        ("IADD3 R1, R1, 0x1, RZ ;", control(stall=1)),
        ("LDG.E.128.CONSTANT.SYS R4, [R2.64] ;", control(stall=2, write=0)),
        ("IMAD R2, R2, R3, RZ ;", control(stall=3)),
        ("BRA 0x10 ;", control(stall=4)),
    ]
    (found,) = sass_loop.load_to_use(loop)
    assert found["waiter_index"] is None and found["waiter_opcode"] is None
    assert found["distance"] == 3  # to the end, as if a waiter sat past it
    assert found["static_cycles"] == 2 + 3 + 4
    # A load that sets no barrier has none to wait on.
    loop[1] = (loop[1][0], control(stall=2))
    (found,) = sass_loop.load_to_use(loop)
    assert found["barrier"] is None and found["waiter_index"] is None


@pytest.mark.parametrize("text, expected", [
    ("HMMA.1688.F32 R8, R20, R22, R8 ;", "HMMA"),
    ("LDSM.16.M88.4 R20, [R3] ;", "LDSM"),
    ("LDSM.16.MT88.4 R20, [R3] ;", "LDSM"),
    ("LDG.E.128.CONSTANT.SYS R4, [R2.64] ;", "LDG"),
    ("@P0 LDG.E R4, [R2.64] ;", "LDG"),
    ("STS.128 [R9], R4 ;", "STS"),
    ("LDS.U R4, [R9] ;", "LDS"),
    ("BAR.SYNC.DEFER_BLOCKING 0x0 ;", "BAR"),
    ("IMAD.MOV.U32 R6, RZ, RZ, R7 ;", "INT_ADDR"),
    ("IMAD.WIDE.U32 R2, R5, c[0x0][0x0], R2 ;", "INT_ADDR"),
    ("IADD3 R8, R8, 0x20, RZ ;", "INT_ADDR"),
    ("LEA R2, P0, R3, c[0x0][0x160], 0x1 ;", "INT_ADDR"),
    ("LEA.HI.X R2, R3, c[0x0][0x164], R4, 0x1, P0 ;", "INT_ADDR"),
    ("ISETP.GE.AND P0, PT, R2, R3, PT ;", "INT_ADDR"),
    ("SHF.R.S32.HI R3, RZ, 0x1f, R2 ;", "INT_ADDR"),
    ("LOP3.LUT R2, R3, 0x1f, RZ, 0xc0, !PT ;", "INT_ADDR"),
    ("IABS R2, R3 ;", "INT_ADDR"),
    ("SEL R2, R3, R4, P0 ;", "INT_ADDR"),
    ("MOV R2, 0x20 ;", "INT_ADDR"),
    ("@!P1 IADD3 R8, R8, 0x20, RZ ;", "INT_ADDR"),
    ("UIADD3 UR4, UR4, 0x20, URZ ;", "UNIFORM_SPECIAL"),
    ("ULDC UR6, c[0x0][0xc] ;", "UNIFORM_SPECIAL"),
    ("ULDC.64 UR4, c[0x0][0x118] ;", "UNIFORM_SPECIAL"),
    ("S2R R2, SR_TID.X ;", "UNIFORM_SPECIAL"),
    ("S2UR UR4, SR_CTAID.X ;", "UNIFORM_SPECIAL"),
    ("CS2R R2, SRZ ;", "UNIFORM_SPECIAL"),
    ("@P0 BRA 0x230 ;", "BRANCH"),
    ("BRA.U 0x230 ;", "BRANCH"),
    ("EXIT ;", "BRANCH"),
    ("FADD R2, R3, R4 ;", "OTHER"),
    ("NOP ;", "OTHER"),
])
def test_classify(text, expected):
    assert sass_loop.classify(text) == expected


def test_opcode_drops_the_guard_predicate():
    assert sass_loop.opcode("@!P1 LDG.E.128 R4, [R2.64] ;") == "LDG.E.128"
    assert sass_loop.opcode("@UP0 BRA 0x10 ;") == "BRA"
    assert sass_loop.opcode("EXIT ;") == "EXIT"


def test_loop_stats_of_a_synthetic_loop():
    loop = [
        ("LDG.E.128.CONSTANT.SYS R4, [R2.64] ;", control(stall=2, write=2)),
        ("BAR.SYNC.DEFER_BLOCKING 0x0 ;", control(stall=5, yield_flag=1, wait=0b000001)),
        ("STS.128 [R9], R4 ;", control(stall=1)),
        ("BAR.SYNC.DEFER_BLOCKING 0x0 ;", control(stall=5, yield_flag=1)),
        ("LDSM.16.M88.4 R20, [R3] ;", control(stall=1, write=0)),
        ("LDSM.16.MT88.4 R24, [R3] ;", control(stall=1, write=1)),
        ("FADD R1, R2, R3 ;", control(stall=1)),
        ("HMMA.1688.F32 R8, R20, R24, R8 ;", control(stall=8, wait=0b000011)),
        ("HMMA.1688.F32 R12, R20, R24, R12 ;", control(stall=8)),
        ("@P0 BRA 0x10 ;", control(stall=4)),
    ]
    stats = sass_loop.loop_stats(loop)
    assert stats["length"] == 10
    assert stats["classes"] == {
        "HMMA": 2, "LDSM": 2, "LDG": 1, "STS": 1, "LDS": 0, "BAR": 2, "INT_ADDR": 0,
        "UNIFORM_SPECIAL": 0, "BRANCH": 1, "OTHER": 1,
    }
    assert stats["variants"]["LDSM"] == {"LDSM.16.M88.4": 1, "LDSM.16.MT88.4": 1}
    assert stats["variants"]["OTHER"] == {"FADD": 1}
    assert stats["instr_per_hmma"] == 5.0
    assert stats["non_hmma_per_hmma"] == 4.0
    assert stats["stall_sum"] == 2 + 5 + 1 + 5 + 1 + 1 + 1 + 8 + 8 + 4
    assert stats["yield_count"] == 2
    assert (stats["waits"], stats["hmma_waits"]) == (2, 1)
    structure = stats["structure"]
    assert (structure["bar"], structure["ldg"], structure["sts"]) == ([1, 3], [0], [2])
    assert structure["ldsm"] == [4, 5]
    # The BARs end the stretches before them: LDG and BAR, then STS and BAR,
    # then the loads and the math.
    assert [(p["first"], p["last"], p["ldg"], p["sts"], p["ldsm"], p["hmma"])
            for p in structure["phases"]] == [(0, 1, 1, 0, 0, 0), (2, 3, 0, 1, 0, 0),
                                              (4, 9, 0, 0, 2, 2)]
    # The first LDG's barrier 2 is waited on by nothing after it.
    assert stats["load_to_use"][0]["waiter_index"] is None


def test_normalize_symbol_replaces_the_anonymous_namespaces_hash():
    a = "_ZN5flint39_GLOBAL__N__fefee919_7_gemm_cu_c03c3db97gemm_ntEP6__halfPKS1_S4_iii"
    b = "_ZN5flint39_GLOBAL__N__00112233_7_gemm_cu_44556677" "7gemm_ntEP6__halfPKS1_S4_iii"
    assert sass_loop.normalize_symbol(a) == "_ZN5flint<anon>7gemm_ntEP6__halfPKS1_S4_iii"
    assert sass_loop.normalize_symbol(b) == sass_loop.normalize_symbol(a)
    assert sass_loop.normalize_symbol("_Z4mainv") == "_Z4mainv"


def test_compare_reports_differences():
    loop = sass_loop.loop_stats([
        ("LDG.E.128.CONSTANT.SYS R4, [R2.64] ;", control(stall=2, write=2)),
        ("STS.128 [R9], R4 ;", control(stall=1, wait=0b000100)),
        ("HMMA.1688.F32 R8, R20, R24, R8 ;", control(stall=8)),
        ("@P0 BRA 0x10 ;", control(stall=4)),
    ])
    left = {"archs": {"sm_75": {"gemm_nt": {"mangled": "m", "loop": loop}}}}
    right = copy.deepcopy(left)
    assert sass_loop.compare(left, right) == []
    right["archs"]["sm_75"]["gemm_nt"]["loop"]["stall_sum"] += 1
    right["archs"]["sm_75"]["gemm_nt"]["loop"]["load_to_use"][0]["distance"] = 7
    right["archs"]["sm_89"] = {}
    differences = sass_loop.compare(left, right)
    assert "sm_75 gemm_nt loop.stall_sum: 15 != 16" in differences
    assert "sm_75 gemm_nt loop.load_to_use.0.distance: 1 != 7" in differences
    assert "sm_89: only in the other file" in differences
    assert len(differences) == 3


# -- The library's loops ------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    library = flint_lib.library_path()
    if not library.is_file():
        pytest.fail(
            f"{library} not found; build it with "
            "`cmake --preset release && cmake --build --preset release`"
        )
    try:
        return sass_loop.report(library)
    except RuntimeError as error:
        pytest.fail(str(error))


def loop_of(data, arch, kernel):
    return data["archs"][arch][kernel]["loop"]


# Measured with test_sass.py's main_loop(): the stall counts sum to these.
STALL_SUMS = {
    "sm_75": {"gemm_nt": 708, "gemm_nn": 722, "gemm_nn_dgelu": 732,
              "gemm_nt_bias_residual": 705},
    "sm_89": {"gemm_nt": 407, "gemm_nn": 428, "gemm_nn_dgelu": 439,
              "gemm_nt_bias_residual": 405},
}


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_every_gemm_kernel_has_a_profiled_loop(data, arch):
    assert list(data["archs"][arch]) == sass_loop.KERNELS
    for kernel, entry in data["archs"][arch].items():
        loop = entry["loop"]
        assert loop["classes"]["HMMA"] == 64, kernel
        assert loop["classes"]["BAR"] == 2, kernel
        assert loop["length"] == sum(loop["classes"].values()), kernel
        assert f"{len(kernel)}{kernel}E" in entry["mangled"], kernel
        assert "<anon>" in entry["mangled"] and "_GLOBAL__N_" not in entry["mangled"], kernel
        assert "gemm_tn_reduce" not in entry["mangled"], kernel


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_stall_sums_match_the_measured_ones(data, arch):
    for kernel, expected in STALL_SUMS[arch].items():
        assert loop_of(data, arch, kernel)["stall_sum"] == expected, kernel


def test_gemm_nt_waits_on_sm_75(data):
    loop = loop_of(data, "sm_75", "gemm_nt")
    assert (loop["waits"], loop["hmma_waits"]) == (13, 8)


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_epilogue_kernels_keep_their_base_loops_length_and_waits(data, arch):
    def shape(kernel):
        loop = loop_of(data, arch, kernel)
        return loop["length"], loop["classes"]["HMMA"], loop["waits"], loop["hmma_waits"]

    assert shape("gemm_nn_dgelu") == shape("gemm_nn")
    assert shape("gemm_nt_bias_residual") == shape("gemm_nt")


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nt_bias_stats_equal_gemm_nts(data, arch):
    assert loop_of(data, arch, "gemm_nt_bias") == loop_of(data, arch, "gemm_nt")


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_tile_loads_are_waited_on_by_the_stores(data, arch):
    # Each LDG's data goes to shared memory: its first waiter is an STS, after
    # the first barrier (the structure LDG ... BAR ... STS ... BAR ... math).
    for kernel in ("gemm_nt", "gemm_nn", "gemm_tn"):
        loop = loop_of(data, arch, kernel)
        ldgs = [item for item in loop["load_to_use"] if item["opcode"].startswith("LDG")]
        assert len(ldgs) == 4, kernel
        assert all(item["waiter_opcode"] == "STS.128" for item in ldgs), kernel
        assert all(item["static_cycles"] > 0 for item in ldgs), kernel
        structure = loop["structure"]
        assert len(structure["bar"]) == 2, kernel
        assert max(structure["ldg"]) < structure["sts"][0], kernel
        assert structure["sts"][-1] < structure["bar"][1] < structure["ldsm"][0], kernel


def test_the_json_names_no_absolute_path_or_user(data):
    text = json.dumps(data)
    assert data["schema"] == "flint-sass-loops/1"
    assert data["library"] == flint_lib.library_path().name
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", data["git"]["sha"])
    assert isinstance(data["git"]["dirty"], bool)
    assert data["cuda"].startswith("Build cuda_") or data["cuda"] == "unknown"
    assert str(flint_lib.ROOT) not in text
    assert flint_lib.ROOT.as_posix() not in text
    assert getpass.getuser().lower() not in text.lower()
    assert not re.search(r"[A-Za-z]:[\\/]|\\\\|/home/|/Users/", text)


def test_cli_writes_the_json_and_compares_it(data, tmp_path):
    out = tmp_path / "loops.json"
    script = str(flint_lib.ROOT / "scripts" / "sass_loop.py")
    first = subprocess.run([sys.executable, script, "--out", str(out)],
                           capture_output=True, text=True, timeout=300)
    assert first.returncode == 0, first.stderr
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["archs"] == data["archs"]
    assert b"\r\n" not in out.read_bytes()
    assert "gemm_nt_bias_residual" in first.stdout and "sm_89" in first.stdout

    same = subprocess.run([sys.executable, script, "--compare", str(out)],
                          capture_output=True, text=True, timeout=300)
    assert same.returncode == 0, same.stdout + same.stderr
    assert "identical" in same.stdout

    written["archs"]["sm_75"]["gemm_nt"]["loop"]["stall_sum"] += 1
    out.write_text(json.dumps(written), encoding="utf-8", newline="\n")
    different = subprocess.run([sys.executable, script, "--compare", str(out)],
                               capture_output=True, text=True, timeout=300)
    assert different.returncode == 1
    assert "sm_75 gemm_nt loop.stall_sum" in different.stdout
