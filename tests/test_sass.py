"""The mma kernels compile to FP32-accumulate tensor-core instructions."""

import importlib.util
import sys

import pytest

import flint_lib

# The SASS parsing is shared with scripts/sass_loop.py.
spec = importlib.util.spec_from_file_location(
    "sass_loop", flint_lib.ROOT / "scripts" / "sass_loop.py"
)
sass_loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sass_loop)
instructions = sass_loop.instructions
main_loop = sass_loop.main_loop

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


def sass_functions(exe, arch):
    """Maps each kernel's mangled name to its SASS for one architecture."""
    try:
        return sass_loop.sass_functions(exe, arch)
    except RuntimeError as error:
        pytest.fail(str(error))


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_mma_kernels_use_hmma_f32_without_spills(exe, arch):
    # One kernel per chain count; each must issue the FP32-accumulate tensor
    # instruction and keep its accumulators in registers (no local memory).
    kernels = {name: body for name, body in sass_functions(exe, arch).items()
               if "mma_m16n8k8_f32" in name}
    assert len(kernels) == 3, sorted(kernels)
    for name, body in kernels.items():
        assert "HMMA.1688.F32" in body, name
        assert "LDL" not in body and "STL" not in body, name


# Each NT-or-NN kernel, by the length-prefixed name in its mangled symbol.
GEMM_KERNELS = [
    pytest.param("7gemm_ntE", id="nt"),
    pytest.param("7gemm_nnE", id="nn"),
    pytest.param("12gemm_nt_biasE", id="nt_bias"),
    pytest.param("17gemm_nt_bias_geluE", id="nt_bias_gelu"),
    pytest.param("13gemm_nn_dgeluE", id="nn_dgelu"),
    pytest.param("21gemm_nt_bias_residualE", id="nt_bias_residual"),
]

# The kernels of the NN layout, whose w tile (k x n in memory) needs the
# transposing ldmatrix.
NN_KERNELS = ("7gemm_nnE", "13gemm_nn_dgeluE")


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
@pytest.mark.parametrize("symbol", GEMM_KERNELS)
def test_gemm_kernel_uses_hmma_and_ldmatrix_without_spills(arch, symbol):
    library = flint_lib.library_path()
    if not library.is_file():
        pytest.fail(
            f"{library} not found; build it with "
            "`cmake --preset release && cmake --build --preset release`"
        )
    kernels = {name: body for name, body in sass_functions(str(library), arch).items()
               if symbol in name}
    assert len(kernels) == 1, sorted(kernels)
    (name, body), = kernels.items()
    # FP32-accumulate tensor-core math, fed by ldmatrix (LDSM), with every
    # accumulator in registers (no local-memory loads or stores).
    assert "HMMA.1688.F32" in body, name
    assert "LDSM" in body, name
    assert "LDL" not in body and "STL" not in body, name
    # NN reads w's tile (k x n in memory) with the transposing ldmatrix; the NT
    # kernels' w tile is already in the fragment layout, so they have none.
    assert ("LDSM.16.MT88" in body) == (symbol in NN_KERNELS), name


def library_functions(arch):
    library = flint_lib.library_path()
    if not library.is_file():
        pytest.fail(
            f"{library} not found; build it with "
            "`cmake --preset release && cmake --build --preset release`"
        )
    return sass_functions(str(library), arch)


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_tn_kernels_use_hmma_and_transposed_ldmatrix_without_spills(arch):
    functions = library_functions(arch)
    # "7gemm_tnE" is the kernel's own name, length-prefixed in the mangled
    # name; "gemm_tn_reduce" is the reduction.
    kernels = {name: body for name, body in functions.items() if "7gemm_tnE" in name}
    assert len(kernels) == 1, sorted(kernels)
    (name, body), = kernels.items()
    assert "HMMA.1688.F32" in body, name
    # x (k x m) and w (k x n) both have k as their rows, so both fragments come
    # from the transposing ldmatrix, and the plain one is never used.
    assert "LDSM.16.MT88" in body, name
    assert "LDSM.16.M88" not in body, name
    assert "LDL" not in body and "STL" not in body, name

    reductions = {name: body for name, body in functions.items() if "14gemm_tn_reduceE" in name}
    assert len(reductions) == 1, sorted(reductions)
    (name, body), = reductions.items()
    assert "LDL" not in body and "STL" not in body, name


# -- The bias kernel's main loop --------------------------------------------


def encodings(loop):
    return [(encoding, control) for _, _, encoding, control in loop]


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nt_bias_main_loop_is_gemm_nts(arch):
    # The GEMM main loops use 126 to 128 of the 128 registers sm_75 allows at
    # two blocks per SM, and changes that perturbed the loop have cost 10-15%
    # on the 2080 Ti. The bias is the kernel's last argument, so the others keep
    # gemm_nt's constant-bank offsets, and its loop compiles to exactly gemm_nt's.
    # Branches encode their offsets relative to themselves, so the loops compare
    # equal wherever each sits in its kernel.
    functions = library_functions(arch)
    loops = {}
    for symbol in ("7gemm_ntE", "12gemm_nt_biasE", "7gemm_nnE"):
        (body,) = [body for name, body in functions.items() if symbol in name]
        loops[symbol] = main_loop(body)[0]

    for loop in loops.values():
        assert sum("HMMA" in item[1] for item in loop) == 64
    assert encodings(loops["12gemm_nt_biasE"]) == encodings(loops["7gemm_ntE"])
    # The extraction finds something that tells the kernels apart.
    assert encodings(loops["7gemm_nnE"]) != encodings(loops["7gemm_ntE"])


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nt_bias_reads_the_bias_after_the_main_loop(arch):
    (body,) = [body for name, body in library_functions(arch).items() if "12gemm_nt_biasE" in name]
    _, loop_end = main_loop(body)
    # The tiles load in 128-bit LDG.E.128; the bias is read a float at a time:
    # 2 columns for each of the 4 tiles of n a thread holds.
    bias_loads = [address for address, text, _, _ in instructions(body)
                  if "LDG" in text and ".128" not in text]
    assert len(bias_loads) == 8, bias_loads
    assert all(address > loop_end for address in bias_loads), (bias_loads, loop_end)


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nt_bias_gelu_main_loop_keeps_gemm_nts_length_and_math(arch):
    # Unlike gemm_nt_bias's, this kernel's loop can't be required encoding for
    # encoding the same as gemm_nt's. Its epilogue (the GELU and the second
    # output) is more work after the loop, which changes ptxas's register
    # allocation for the whole kernel: the loop keeps its length and its math,
    # but some of its integer instructions and their scheduling differ. So the
    # test pins what the allocation can't change, the loop's length, its HMMA
    # count and the absence of local memory, and the speed is checked where it
    # matters, by timing the kernel on the 2080 Ti.
    functions = library_functions(arch)
    loops = {}
    for symbol in ("7gemm_ntE", "17gemm_nt_bias_geluE"):
        (body,) = [body for name, body in functions.items() if symbol in name]
        loops[symbol] = main_loop(body)[0]

    gelu, plain = loops["17gemm_nt_bias_geluE"], loops["7gemm_ntE"]
    assert sum("HMMA" in item[1] for item in gelu) == 64
    assert sum("HMMA" in item[1] for item in plain) == 64
    assert len(gelu) == len(plain)
    assert not any("LDL" in item[1] or "STL" in item[1] for item in gelu)


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nt_bias_gelu_reads_the_bias_after_the_main_loop(arch):
    (body,) = [body for name, body in library_functions(arch).items() if "17gemm_nt_bias_geluE" in name]
    _, loop_end = main_loop(body)
    # As in gemm_nt_bias: the tiles load in 128-bit LDG.E.128, and the bias is
    # read a float at a time, 2 columns for each of the 4 tiles of n a thread
    # holds.
    bias_loads = [address for address, text, _, _ in instructions(body)
                  if "LDG" in text and ".128" not in text]
    assert len(bias_loads) == 8, bias_loads
    assert all(address > loop_end for address in bias_loads), (bias_loads, loop_end)


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nn_dgelu_main_loop_keeps_gemm_nns_length_and_waits(arch):
    # As for gemm_nt_bias_gelu, the epilogue's extra work after the loop changes
    # ptxas's register allocation, so the loop can't be required encoding for
    # encoding the same as gemm_nn's. The test pins what the allocation can't
    # change: the loop's length, its HMMA count, the absence of local memory,
    # and the scoreboard waits, which is where the loop's tile loads (LDG) are
    # waited on (to store them to shared memory) and the shared-memory reads
    # (LDSM) before the tensor-core math: the same number of instructions
    # wait on a scoreboard barrier, and the same number of those are HMMAs. A
    # wait is a nonzero mask in bits 52-57 of the control word, and its
    # stall count (the cycles the scheduler holds the instruction back) the
    # four bits from bit 41.
    #
    # The stall counts add up to a little more here, not the same: 732 against
    # gemm_nn's 722 cycles on sm_75, and 439 against 428 on sm_89. The
    # difference is three address instructions between the tile loads and their
    # shared-memory stores, which wait on the loads anyway, and the speed is
    # checked where it matters, by timing the kernel on the 2080 Ti.
    functions = library_functions(arch)
    loops = {}
    for symbol in NN_KERNELS:
        (body,) = [body for name, body in functions.items() if symbol in name]
        loops[symbol] = main_loop(body)[0]

    def waits(loop):
        """The loop's instructions that wait on a scoreboard barrier, and the
        HMMAs among them."""
        waiting = [item for item in loop if (int(item[3], 16) >> 52) & 0x3F]
        return len(waiting), sum("HMMA" in item[1] for item in waiting)

    dgelu, plain = loops["13gemm_nn_dgeluE"], loops["7gemm_nnE"]
    assert sum("HMMA" in item[1] for item in dgelu) == 64
    assert sum("HMMA" in item[1] for item in plain) == 64
    assert len(dgelu) == len(plain)
    assert not any("LDL" in item[1] or "STL" in item[1] for item in dgelu)
    # The extraction finds waits, so equal counts mean something.
    assert waits(plain)[0] > 0 and waits(plain)[1] > 0
    assert waits(dgelu) == waits(plain)


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nn_dgelu_reads_pre_after_the_main_loop(arch):
    (body,) = [body for name, body in library_functions(arch).items() if "13gemm_nn_dgeluE" in name]
    _, loop_end = main_loop(body)
    # The tiles load in 128-bit LDG.E.128; pre is read a pair of halves at a
    # time: 2 pairs for each of the 16 tiles of m16n8 a thread holds (the rows
    # 8 apart).
    pre_loads = [address for address, text, _, _ in instructions(body)
                 if "LDG" in text and ".128" not in text]
    assert len(pre_loads) == 32, pre_loads
    assert all(address > loop_end for address in pre_loads), (pre_loads, loop_end)


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nt_bias_residual_main_loop_keeps_gemm_nts_length_and_waits(arch):
    # Like the other epilogue kernels, this one's loop is gemm_nt's in length,
    # HMMA count, absence of local memory and scoreboard waits (how many
    # instructions wait on a scoreboard barrier, and how many of those are
    # HMMAs; see test_gemm_nn_dgelu_main_loop_keeps_gemm_nns_length_and_waits
    # for what the waits are), but not encoding for encoding, and the stall sum
    # isn't pinned either: the loop differs from gemm_nt's only in the order of
    # its address arithmetic, and its stall counts add up to 705 against
    # gemm_nt's 708 cycles on sm_75, and 405 against 407 on sm_89. The order
    # matters on the 2080 Ti, though: an earlier version, which stored each pair
    # right after reading it, compiled to a loop with gemm_nt's waits that ran
    # 10% slower at k = 16,384. So the speed is checked where it matters, by
    # timing the kernel on the 2080 Ti.
    functions = library_functions(arch)
    loops = {}
    for symbol in ("7gemm_ntE", "21gemm_nt_bias_residualE"):
        (body,) = [body for name, body in functions.items() if symbol in name]
        loops[symbol] = main_loop(body)[0]

    def waits(loop):
        """The loop's instructions that wait on a scoreboard barrier, and the
        HMMAs among them."""
        waiting = [item for item in loop if (int(item[3], 16) >> 52) & 0x3F]
        return len(waiting), sum("HMMA" in item[1] for item in waiting)

    residual, plain = loops["21gemm_nt_bias_residualE"], loops["7gemm_ntE"]
    assert sum("HMMA" in item[1] for item in residual) == 64
    assert sum("HMMA" in item[1] for item in plain) == 64
    assert len(residual) == len(plain)
    assert not any("LDL" in item[1] or "STL" in item[1] for item in residual)
    # The extraction finds waits, so equal counts mean something.
    assert waits(plain)[0] > 0 and waits(plain)[1] > 0
    assert waits(residual) == waits(plain)


@pytest.mark.parametrize("arch", ["sm_75", "sm_89"])
def test_gemm_nt_bias_residual_reads_the_bias_and_residual_after_the_main_loop(arch):
    (body,) = [body for name, body in library_functions(arch).items()
               if "21gemm_nt_bias_residualE" in name]
    _, loop_end = main_loop(body)
    # The tiles load in 128-bit LDG.E.128. The bias is read a float at a time:
    # 2 columns for each of the 4 tiles of n a thread holds. The residual is
    # read a pair of floats at a time with 64-bit loads, and y is stored the
    # same way: 2 pairs for each of the 16 tiles of m16n8 a thread holds (the
    # rows 8 apart). (The opcode is the word with LDG or STG in it: the
    # operands of a plain load can say ".64" too, for a 64-bit address.)
    memory = [(address, next(word for word in text.split() if word[:3] in ("LDG", "STG")))
              for address, text, _, _ in instructions(body)
              if "LDG" in text or "STG" in text]
    bias_loads = [a for a, op in memory if op.startswith("LDG") and ".128" not in op and ".64" not in op]
    residual_loads = [a for a, op in memory if op.startswith("LDG") and ".64" in op]
    stores = [a for a, op in memory if op.startswith("STG")]
    assert len(bias_loads) == 8, bias_loads
    assert len(residual_loads) == 32, residual_loads
    assert len(stores) == 32 and all(".64" in op for _, op in memory if op.startswith("STG"))
    for loads in (bias_loads, residual_loads, stores):
        assert all(address > loop_end for address in loads), (loads, loop_end)
