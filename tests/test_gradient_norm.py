"""flint_gradient_norm: the L2 norm of a list of FP32 tensors taken as one vector, and the count of
their values that are inf or NaN (the loss scale's overflow check), in FP64, in two kernels with no
atomics. The tensors are a table of {address, count} entries in device memory (flint_tensor). An FP32
value converts to FP64 exactly and its square is exact there, so the sum of squares is the only
rounding, and it can't overflow: finite gradients always give a finite norm. The count uses isfinite,
never a comparison or a max, which would drop a NaN. The sum is in a fixed order that depends only
on the tensors' counts:

  - Chunks of 4,096 values never span tensors: tensor t has ceil(count_t / 4096) chunks, numbered
    globally in table order. Pass 1 is 1,024 blocks of 256 threads, whatever the GPU, and block b
    takes every chunk whose global number g has g % 1024 == b, in increasing g.
  - In a chunk (its values from c * 4096), thread tid takes, for j = 0 to 3, the group of 4 values
    from c * 4096 + j * 1024 + 4 * tid, in order j then k = 0 to 3. A value past the tensor's end
    is +0 (exact: the sum starts at +0 and squares are at least +0). A thread's sum is
    s = s + d * d over all its values of all its chunks, d the value in FP64 (the same bits as
    fma(d, d, s), the square being exact). Its count is the number of its values that are not
    finite.
  - The block reduction, in both passes: within each warp, for offsets 16, 8, 4, 2, 1, every lane
    adds the value of the lane that many above (a lane past the end its own); lane 0 holds the
    warp's sum. Then thread 0 adds the 8 warps' in order of the warp, one after the other. Block
    b's sum goes to the workspace (the 1,024 sums first, then the 1,024 counts).
  - Pass 2 is one block of 256 threads: thread t starts at +0 and adds the partial sums t, t + 256,
    t + 512 and t + 768 in that order, the same block reduction follows, and thread 0 stores the
    IEEE-rounded square root. The count is summed the same way (as integers, so in any order).

The 16-byte alignment of a tensor only picks the load instruction, so a tensor starting 1, 2 or 3
floats past an aligned address gives the same bits. A numpy float64 emulation of exactly that order
(emulate below, itself checked against a literal scalar transcription of the description) must
equal Flint's norm bit for bit, and its pass 1 partial sums must be the workspace's. The inputs are
built on the CPU from a seeded generator's integers and copied over, so they are the same on every
machine (torch.randn on the CPU isn't: the laptop's and the server's differ, while randint's agree):
matching the emulation on each machine is what shows that the bits are the same on every GPU. A test
also shows which changes of order the comparisons can tell apart.

What the tests cover:
  - the emulation against the literal transcription, and what its bits can tell apart;
  - the bit-exact comparison on counts 0 to 100,003 and tails at 4,096, more chunks than blocks
    (block 0 taking several, a block taking chunks of several tensors), a table of 12,292 small
    tensors (GPT-2 at 1,024 layers), GPT-2 124M's 148 sizes, and mixed magnitudes;
  - the accuracy rule (PLAN.md's D26 rule for reductions): Flint's error against the FP64 norm of
    the same FP32 values at most twice FP32 PyTorch's own (clip_grad_norm_'s computation), with
    FP32 floors (relative 2^-22, 4 FP32 ulps), on N(0, 1), gradient-like N(0, 1e-3), those scaled by
    2^16, per-tensor scales over decades and outliers, on GPT-2's sizes and odd ones; and a plain
    bound of 1e-12 relative;
  - the edges of the values: near FP32's max (every value finite, an FP32 sum of squares overflows),
    FP32 subnormals, -0 and zero tensors;
  - the overflow check: inf, -inf and every kind of NaN at the first and last elements of tensors,
    at chunk and group boundaries, in a scalar tail, in a misaligned tensor, in blocks 1,023 and 0's
    second round, several at once; the count equals (~torch.isfinite(t)).sum() and GradScaler's
    found_inf agrees;
  - what the call reads and writes: guard bands around the results and the workspace, the tensors
    and the table unchanged, garbage in the workspace and the old results changing nothing, a larger
    workspace, misaligned tensors, entries of count 0 reading nothing, a tensor listed twice;
  - determinism and streams, the workspace size, and the argument errors;
  - the model: flint_gpt2_gradient_norm gives flint_gradient_norm's bits on the gradients and the
    emulation's, accumulates, counts an injected inf or NaN, and fails on a model without backward.

Under the sanitizer (scripts/sanitize.py) the inputs shrink to at most a few hundred thousand
values, except one tensor of 1,025 chunks (4,198,401 values), so that a block taking two chunks of
one tensor runs under the tools; the other large cases are skipped, and the model tests run only
the smallest models."""

import gc
import math
import struct

import numpy as np
import pytest
import torch

import flint_lib
import gate
from flint_lib import Gpt2
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, generator
from test_gpt2 import (  # also puts scripts/ on sys.path
    ODD_MODELS,
    TINY,
    all_equal,
    gate_scale,
    gpt2_config,
    make_model,
    random_batch,
    state_of,
)
from test_gpt2_backward import backward_config, forward_backward, poison_gradients, read_gradients

BLOCKS = 1024
THREADS = 256
WARP = 32
WARPS = THREADS // WARP
GROUP = 4
UNROLL = 4
CHUNK = THREADS * GROUP * UNROLL  # 4,096 values
WARP_OFFSETS = (16, 8, 4, 2, 1)
WORKSPACE_BYTES = 16384  # 1,024 FP64 sums and 1,024 uint64 counts

# Cases up to this many values (and entries) have their tensors compared with a copy after the call.
CHECK_VALUES = 1 << 22
CHECK_ENTRIES = 300
# The results and the workspace of a guarded call sit inside a buffer filled with a pattern, which
# must be intact around them afterwards.
GUARD = 64
SENTINEL = 0x5A5A5A5A5A5A5A5A


def as_int64(value):
    return (value + 2**63) % 2**64 - 2**63


FILLS = {
    "nan": as_int64(0x7FF8000000000000),
    "-nan": as_int64(0xFFFFFFFFFFFFFFFF),
    "inf": as_int64(0x7FF0000000000000),
    "-inf": as_int64(0xFFF0000000000000),
    "big": as_int64(0x7FEFFFFFFFFFFFFF),
    "zero": 0,
    "garbage": "garbage",
}
# FP32 bit patterns of the values the overflow check must catch.
NON_FINITE_BITS = {
    "inf": 0x7F800000,
    "-inf": 0xFF800000,
    "nan": 0x7FC00000,
    "snan": 0x7F800001,
    "-nan": 0xFFC00000,
    "nan-payload": 0x7FFFFFFF,
}
NAN_KINDS = ("nan", "snan", "-nan", "nan-payload")


@pytest.fixture(autouse=True)
def free_gpu_memory():
    """The laptop's GPU is 6 GiB and shared, so each test hands back what it cached."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


def bits64(x):
    """The bit pattern of a float64, as the signed integer torch.int64 holds."""
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def float64_of(bits):
    return struct.unpack("<d", struct.pack("<q", bits))[0]


def assert_norm_bits(got_bits, expected, what):
    """The norm's bits (from the GPU) are the expected float64's, bit for bit (any NaN stands for
    any other: a NaN's payload isn't pinned)."""
    if math.isnan(expected):
        assert math.isnan(float64_of(got_bits)), f"{what}: {float64_of(got_bits)!r} is not NaN"
        return
    assert got_bits == bits64(expected), (
        f"{what}: {float64_of(got_bits)!r} ({got_bits & (2**64 - 1):#018x}) against {expected!r} "
        f"({bits64(expected) & (2**64 - 1):#018x})"
    )


# -- The emulation of the kernels' order of additions ---------------------------------------------


def as_arrays(values):
    """The values (CPU torch tensors or numpy arrays, FP32) as flat float32 numpy arrays."""
    out = []
    for v in values:
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        assert a.dtype == np.float32, a.dtype
        out.append(a.reshape(-1))
    return out


def block_reduce(s, warp_offsets=WARP_OFFSETS, warp_order=tuple(range(WARPS))):
    """The kernels' block reduction on s, (rows, 256) float64 (a row per block): every warp's
    shuffle-down tree, then the 8 warps' sums added in order, one after the other. A shuffle past the
    warp's end gives the lane its own value. Returns the totals, (rows,)."""
    w = s.reshape(-1, WARPS, WARP)
    for offset in warp_offsets:
        shifted = w.copy()
        shifted[..., : WARP - offset] = w[..., offset:]
        w = w + shifted
    sums = w[..., 0]
    total = sums[:, warp_order[0]]
    for i in warp_order[1:]:
        total = total + sums[:, i]
    return total


def emulate(values, blocks=BLOCKS, warp_offsets=WARP_OFFSETS, warp_order=tuple(range(WARPS)),
            partial_order=None, with_partials=False):
    """The kernels' norm for the FP32 arrays (a list: one entry per tensor), as a float64: bit for
    bit the module docstring's sums, in numpy float64 throughout, vectorized over the threads and
    blocks (a thread's own sum stays a sequence of adds, one per value). With with_partials, also
    pass 1's block sums, (blocks,) float64, as the workspace holds them.

    A value past a tensor's end is +0 and adds nothing (see the docstring). The other arguments
    change the order, and exist to show what the bit-exact tests can tell apart: the number of
    blocks (a multiple of 256, and the real kernel's 1,024), the offsets of the warp's tree, the
    order of the warps' sums, and the order in which pass 2's thread adds its blocks' sums."""
    arrays = as_arrays(values)
    assert blocks % THREADS == 0
    per_thread = blocks // THREADS
    partial_order = tuple(range(per_thread)) if partial_order is None else partial_order
    sizes = [a.size for a in arrays]
    chunks = [-(-n // CHUNK) for n in sizes]
    first = np.concatenate([[0], np.cumsum(chunks, dtype=np.int64)])
    total_chunks = int(first[-1])
    s = np.zeros((blocks, THREADS), dtype=np.float64)
    with np.errstate(all="ignore"):
        # Round r holds the global chunks r * blocks .. r * blocks + blocks - 1, one to a block:
        # block b takes the chunks g = b, b + blocks, ... in order, so round r gives it its r-th.
        for lo in range(0, total_chunks, blocks):
            hi = min(total_chunks, lo + blocks)
            buf = np.zeros((hi - lo) * CHUNK, dtype=np.float64)
            t = max(int(np.searchsorted(first, lo, side="right")) - 1, 0)
            while t < len(arrays) and first[t] < hi:
                c0, c1 = max(0, lo - int(first[t])), min(chunks[t], hi - int(first[t]))
                if c0 < c1:
                    v0, v1 = c0 * CHUNK, min(sizes[t], c1 * CHUNK)
                    dest = (int(first[t]) + c0 - lo) * CHUNK
                    buf[dest : dest + v1 - v0] = arrays[t][v0:v1]
                t += 1
            x = buf.reshape(hi - lo, UNROLL, THREADS, GROUP)
            for j in range(UNROLL):
                for k in range(GROUP):
                    d = x[:, j, :, k]
                    s[: hi - lo] = s[: hi - lo] + d * d
        partials = block_reduce(s, warp_offsets, warp_order)
        # Pass 2: thread t adds the sums t, t + 256, ... in order (the sums are at least +0, so
        # the first add, to +0, is exact), then the block reduction.
        s2 = np.zeros((1, THREADS), dtype=np.float64)
        for i in partial_order:
            s2 = s2 + partials[i * THREADS : (i + 1) * THREADS][None, :]
        norm = np.sqrt(block_reduce(s2, warp_offsets, warp_order))[0]
    return (norm, partials) if with_partials else norm


def scalar_reference(values, blocks=BLOCKS):
    """The module docstring's order transcribed literally, one Python float (a float64) at a time:
    slow, so for tiny inputs, to check emulate. Returns (norm, the block sums)."""
    arrays = [a.tolist() for a in as_arrays(values)]
    per_thread = blocks // THREADS

    def reduce_block(thread_sums):
        warp_totals = []
        for w in range(WARPS):
            lanes = list(thread_sums[w * WARP : (w + 1) * WARP])
            for offset in (16, 8, 4, 2, 1):
                lanes = [lanes[l] + (lanes[l + offset] if l + offset < WARP else lanes[l])
                         for l in range(WARP)]  # fmt: skip
            warp_totals.append(lanes[0])
        total = warp_totals[0]
        for w in range(1, WARPS):
            total = total + warp_totals[w]
        return total

    partials = []
    for b in range(blocks):
        thread_sums = [0.0] * THREADS
        g = 0
        for values_t in arrays:
            n = len(values_t)
            chunks = -(-n // CHUNK)
            for c in range(chunks):
                if (g + c) % blocks != b:
                    continue
                for tid in range(THREADS):
                    s = thread_sums[tid]
                    for j in range(UNROLL):
                        start = c * CHUNK + j * THREADS * GROUP + GROUP * tid
                        for k in range(GROUP):
                            v = float(values_t[start + k]) if start + k < n else 0.0
                            s = s + v * v
                    thread_sums[tid] = s
            g += chunks
        partials.append(reduce_block(thread_sums))
    thread_sums = []
    for t in range(THREADS):
        s = 0.0
        for i in range(per_thread):
            s = s + partials[t + i * THREADS]
        thread_sums.append(s)
    return math.sqrt(reduce_block(thread_sums)), partials


# -- Inputs ---------------------------------------------------------------------------------------


def cpu_list(counts, seed):
    """FP32 numpy arrays of these counts, built from a seeded CPU generator's integers only, so the
    same on every machine (torch.randn on the CPU gives other values on the laptop and the server):
    a 24-bit integer m and an exponent e from -6 to 6, and the value m * 2^(e - 24), exact in FP32
    and below 2^e in magnitude, so that the FP64 sums round differently in different orders. The
    tensors are drawn one after the other (which bounds the temporaries of GPT-2's 124M values)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in counts:
        m = torch.randint(-(2**24 - 1), 2**24, (n,), generator=g, dtype=torch.int32).float()
        e = torch.randint(-6, 7, (n,), generator=g, dtype=torch.int32).float()
        out.append((m * torch.exp2(e - 24)).numpy())
    return out


def to_gpu(arrays, misalign=0):
    """The arrays in one FP32 buffer on the GPU, each starting on a 16-byte boundary plus `misalign`
    floats, and the gaps and the ends filled with NaN, so a kernel reading past a tensor's end
    would show in the count and the norm. Returns (the buffer, one view per array)."""
    starts, at = [], 0
    for a in arrays:
        at += (-at) % 4
        starts.append(at + misalign)
        at = starts[-1] + a.size
    host = np.full(at + 8, np.nan, dtype=np.float32)
    for start, a in zip(starts, arrays):
        host[start : start + a.size] = a
    flat = torch.from_numpy(host).cuda()
    return flat, [flat[s : s + a.size] for s, a in zip(starts, arrays)]


def gpt2_sizes():
    """GPT-2 124M's 148 parameter sizes, in the order of flint_gpt2_parameter."""
    sizes = [50304 * 768, 1024 * 768]
    for _ in range(12):
        sizes += [768, 768, 2304 * 768, 2304, 768 * 768, 768, 768, 768, 3072 * 768, 3072, 768 * 3072, 768]
    sizes += [768, 768]
    assert len(sizes) == 148
    return sizes


def small_gpt2_counts(layers, width=8, vocab=50, block_size=16):
    """The sizes of a GPT-2 of this many layers, at a tiny width: 2 + 12 * layers + 2 tensors, each
    below a chunk."""
    w = width
    counts = [vocab * w, block_size * w]
    for _ in range(layers):
        counts += [w, w, 3 * w * w, 3 * w, w * w, w, w, w, 4 * w * w, 4 * w, w * 4 * w, w]
    return counts + [w, w]


def random_counts(n, high, seed):
    return torch.randint(1, high + 1, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


# -- Running Flint --------------------------------------------------------------------------------


class Result:
    """What one call left: norm (a float), bits (the norm's bit pattern), nonfinite (the count),
    and with with_workspace the first pass's block sums (a numpy float64 array) and counts."""

    def __init__(self, bits, nonfinite, sums=None, counts=None):
        self.bits, self.nonfinite, self.sums, self.counts = bits, nonfinite, sums, counts
        self.norm = float64_of(bits)


def make_fill(fill, n):
    if isinstance(fill, str):
        return torch.randint(-(2**62), 2**62, (n,), dtype=torch.int64, device="cuda")
    return torch.full((n,), fill, dtype=torch.int64, device="cuda")


def total_values(entries):
    return sum(e.numel() if isinstance(e, torch.Tensor) else e[1] for e in entries)


def run_flint(flint, entries, *, tensor_count=None, offset=2, workspace_fill=FILLS["nan"],
              norm_fill=FILLS["nan"], nonfinite_fill=FILLS["nan"], workspace_extra=0, stream=None,
              watch=None, with_workspace=False, table=None):  # fmt: skip
    """Runs flint_gradient_norm on the entries (FP32 CUDA tensors, or (address, count) pairs) into a
    norm, a count and a workspace that sit `offset` int64s (a multiple of 2 keeps them 16-byte
    aligned) into buffers filled with a sentinel: the workspace's 16 KiB with workspace_fill and
    workspace_extra bytes of room after it, the results with the old fills. Checks that everything
    around them, and the room, the tensors and the table are untouched, and returns a Result. The
    tensors compared are `watch`, or the entries that are tensors when there are few and they are
    small."""
    n = len(entries) if tensor_count is None else tensor_count
    if table is None:
        table = flint_lib.tensor_table(entries)
    if watch is None:
        watch = [e for e in entries if isinstance(e, torch.Tensor)]
        if len(entries) > CHECK_ENTRIES or total_values(entries) > CHECK_VALUES:
            watch = []
    before = [t.clone() for t in watch]
    table_before = table.clone()
    norm_buf = torch.full((offset + 1 + GUARD,), SENTINEL, dtype=torch.int64, device="cuda")
    nonfinite_buf = torch.full((offset + 1 + GUARD,), SENTINEL, dtype=torch.int64, device="cuda")
    words = WORKSPACE_BYTES // 8
    extra = workspace_extra // 8
    workspace_buf = torch.full((offset + words + extra + GUARD,), SENTINEL, dtype=torch.int64, device="cuda")
    norm = norm_buf[offset : offset + 1]
    nonfinite = nonfinite_buf[offset : offset + 1]
    workspace = workspace_buf[offset : offset + words + extra]
    norm.copy_(make_fill(norm_fill, 1))
    nonfinite.copy_(make_fill(nonfinite_fill, 1))
    workspace[:words].copy_(make_fill(workspace_fill, words))
    torch.cuda.synchronize()

    flint.gradient_norm(norm, nonfinite, table, n, workspace, WORKSPACE_BYTES + workspace_extra, stream)
    torch.cuda.synchronize()

    sentinel = torch.full((1,), SENTINEL, dtype=torch.int64, device="cuda")
    for name, buf, used in (("norm", norm_buf, 1), ("nonfinite", nonfinite_buf, 1),
                            ("workspace", workspace_buf, words + extra)):  # fmt: skip
        assert all_equal(buf[:offset], sentinel), f"flint_gradient_norm wrote before {name}"
        assert all_equal(buf[offset + used :], sentinel), f"flint_gradient_norm wrote past {name}"
    if extra:
        assert all_equal(workspace[words:], sentinel), "flint_gradient_norm wrote past its workspace"
    assert torch.equal(table, table_before), "the table changed"
    for t, b in zip(watch, before):
        assert torch.equal(t.view(torch.int32), b.view(torch.int32)), "a tensor changed"
    result = Result(int(norm.item()), int(nonfinite.item()))
    if with_workspace:
        result.sums = workspace[: BLOCKS].view(torch.float64).cpu().numpy().copy()
        result.counts = workspace[BLOCKS : 2 * BLOCKS].cpu().numpy().copy()
    return result


def expected_nonfinite(tensors):
    """How many values are not finite, by torch, over the (CUDA) tensors."""
    return sum(int((~torch.isfinite(t)).sum()) for t in tensors)


def run_arrays(flint, arrays, **kwargs):
    """to_gpu and run_flint on numpy arrays; returns (the Result, the GPU views)."""
    flat, views = to_gpu(arrays)
    kwargs.setdefault("watch", [flat])
    return run_flint(flint, views, **kwargs), views


# -- The emulation itself --------------------------------------------------------------------------

# (counts, blocks): tiny inputs and the real 1,024 blocks, and 256 and 512 blocks with more chunks
# than blocks (the literal transcription is slow, so the wraps use fewer blocks).
SCALAR_CASES = {
    "count-0": ([0], 1024),
    "count-1": ([1], 1024),
    "count-4097": ([4097], 1024),
    "tensors": ([5, 4096, 0, 1, 4099, 17], 1024),
    "two-big": ([2 * CHUNK + 5, 3 * CHUNK], 1024),
    "wrap-256-blocks": ([1] * 300, 256),
    "wrap-256-blocks-big": ([CHUNK + 1] * 130, 256),
    "wrap-512-blocks": ([CHUNK + 3, 2, 0, 6000], 512),
}


@pytest.mark.parametrize("name", SCALAR_CASES)
def test_the_emulation_matches_a_literal_scalar_transcription(name):
    counts, blocks = SCALAR_CASES[name]
    arrays = cpu_list(counts, 100 + len(counts))

    norm, partials = emulate(arrays, blocks=blocks, with_partials=True)

    expected, expected_partials = scalar_reference(arrays, blocks=blocks)
    assert bits64(norm) == bits64(expected), (norm, expected)
    assert [bits64(p) for p in partials] == [bits64(p) for p in expected_partials]


def test_the_emulation_matches_the_scalar_transcription_with_non_finite_values():
    arrays = cpu_list([5000, 3, 9000], 7)
    arrays[0][4097] = np.inf
    arrays[2][100] = -np.inf
    assert math.isinf(emulate(arrays)) and bits64(emulate(arrays)) == bits64(scalar_reference(arrays)[0])
    arrays[1][2] = np.nan
    assert math.isnan(emulate(arrays)) and math.isnan(scalar_reference(arrays)[0])


def test_the_emulation_stays_in_float64():
    arrays = cpu_list([4097, 3], 8)

    assert emulate(arrays).dtype == np.float64
    norm, partials = emulate(arrays, with_partials=True)
    assert partials.dtype == np.float64 and partials.shape == (BLOCKS,)
    # Chunk 0 is block 0's, chunk 1 block 1's, and the tensor of 3 is chunk 2: the rest hold +0.
    assert [bool(p == 0) for p in partials[3:]] == [True] * (BLOCKS - 3)
    assert math.sqrt(sum(float(p) for p in partials)) == pytest.approx(float(norm), rel=1e-12)


def test_an_entry_of_count_0_is_no_chunk():
    # A tensor of count 0 has no chunks, so it doesn't shift the numbering: the same bits with it.
    arrays = cpu_list([5, 4097, 3], 9)

    plain = emulate(arrays)

    with_empties = emulate([np.zeros(0, np.float32), arrays[0], np.zeros(0, np.float32), *arrays[1:],
                            np.zeros(0, np.float32)])
    assert bits64(plain) == bits64(with_empties)


ORDER_CASES = {
    "one-tensor": [1_000_003],
    "tensors-1500": random_counts(1500, 300, 3),
    "chunks-1100": [1100 * CHUNK + 17],
    "one-tensor-300k": [300_007],
    "four-tensors": [20000, 5, 3000, 70000],
    "tensors-600": random_counts(600, 2000, 9),
}


def crafted_pass_2():
    """1,024 tensors of one value each: one of 2^26 (its square, 2^52, has an ulp of 1) and 1,023 of
    0.5 (squares of 0.25, which that ulp absorbs one at a time). Block b holds tensor b, so pass 2's
    thread 0 adds 2^52, then 0.25 three times (the sum stays 2^52); added backwards the three make
    0.75, which rounds the sum up by 1."""
    return [np.array([2.0**26], dtype=np.float32)] + [np.array([0.5], dtype=np.float32) for _ in range(1023)]


def test_the_bit_exact_tests_tell_other_orders_apart():
    # What the comparisons can see. On ordinary values the FP64 norm is nearly the exact one in
    # any tree order (the square root hides a last bit of the sum), so a pairwise sum of the same
    # values gives other bits on only some cases, a sequential one on all. numpy's pairwise sum
    # unrolls by the CPU's vector width, so how often it lands on the kernel's bits depends on the
    # machine (on one draw of inputs, 2 of the 6 cases on the laptop and 0 on the server): it is
    # printed, not asserted, and neither is it on the crafted input. The block sums of pass 1,
    # which the workspace holds and the bit-exact test below compares, are far more sensitive: the
    # warp's tree and the order of the warps change hundreds of them. Pass 2's order of adds is
    # pinned by a crafted input, whose sums the ulp of 2^52 rounds. The counts are printed.
    cases = {name: cpu_list(counts, 11 + i) for i, (name, counts) in enumerate(ORDER_CASES.items())}
    if SANITIZER:
        cases = {"tensors-600": cases["tensors-600"], "four-tensors": cases["four-tensors"]}
    differing, partials_differing = {}, {}
    for name, arrays in cases.items():
        kernel, kernel_partials = emulate(arrays, with_partials=True)
        flat = np.concatenate(arrays).astype(np.float64)
        squares = flat * flat
        for variant, other in (("numpy's pairwise float64 sum", np.sqrt(np.sum(squares))),
                               ("one sequential float64 sum", np.sqrt(np.cumsum(squares)[-1])),
                               ("one sequential sum, backwards", np.sqrt(np.cumsum(squares[::-1])[-1]))):  # fmt: skip
            differing[variant] = differing.get(variant, 0) + (bits64(other) != bits64(kernel))
        for variant, kwargs in (("the warp's offsets 1, 2, 4, 8, 16", dict(warp_offsets=(1, 2, 4, 8, 16))),
                                ("the 8 warps added backwards", dict(warp_order=tuple(reversed(range(WARPS)))))):  # fmt: skip
            _, partials = emulate(arrays, with_partials=True, **kwargs)
            n = int((partials.view(np.int64) != kernel_partials.view(np.int64)).sum())
            partials_differing[variant] = partials_differing.get(variant, 0) + n
    for variant, n in differing.items():
        print(f"ORDER {variant}: other norm bits on {n} of {len(cases)} cases")
    assert differing["one sequential float64 sum"] == differing["one sequential sum, backwards"] == len(cases)
    for variant, n in partials_differing.items():
        print(f"ORDER {variant}: {n} block sums differ (of {len(cases) * BLOCKS})")
        assert n >= 100, variant
    crafted = crafted_pass_2()
    kernel = emulate(crafted)
    pairwise = np.sqrt(np.sum(np.concatenate(crafted).astype(np.float64) ** 2))
    print(f"ORDER crafted, numpy's pairwise float64 sum: {'other' if bits64(pairwise) != bits64(kernel) else 'the same'} bits")
    for variant, other in (("pass 2's 4 sums added backwards", emulate(crafted, partial_order=(3, 2, 1, 0))),
                           ("the exact sum, rounded", math.sqrt(math.fsum(float(a[0]) ** 2 for a in crafted)))):  # fmt: skip
        print(f"ORDER crafted, {variant}: {'other' if bits64(other) != bits64(kernel) else 'the same'} bits")
        assert bits64(other) != bits64(kernel), variant


# -- The bit-exact emulation -----------------------------------------------------------------------

BIG = "big"
EXACT_CASES = {
    "count-0": [0],
    "count-1": [1],
    "count-3": [3],
    "count-4": [4],
    "count-5": [5],
    "count-1023": [1023],
    "count-1024": [1024],
    "count-4095": [4095],
    "count-4096": [4096],
    "count-4097": [4097],
    "count-8192": [8192],
    "count-8193": [8193],
    "count-100003": [100003],
    "tensors": [17, 4096, 1, 5000, 0, 3, 8193, 4095, 4097, 12, 900],
    "tensors-with-empties": [0, 5, 0, 0, 4097, 0, 1, 0],
    # More chunks than blocks: block 0 wraps, and blocks take chunks of several tensors.
    "tensors-1500-small": random_counts(1500, 150, 5),
    "tensors-1025-of-1": [1] * 1025,
    # GPT-2 at 1,024 layers (12,292 tensors) at a tiny width; the sanitizer's has 1,300 tensors.
    "table-small": small_gpt2_counts(108 if SANITIZER else 1024),
    # One tensor of more chunks than blocks, so a block takes chunks c and c + 1,024 of it; kept
    # under the sanitizer (16 MB), the one case there past a few hundred thousand values.
    "chunks-1025": [1024 * CHUNK + 1],
    "chunks-2100-of-3": ([400 * CHUNK + 11, 700 * CHUNK + 1, 1000 * CHUNK + 4095], BIG),
    "tensors-600-random": (random_counts(600, 9000, 6), BIG),
    "gpt2-148": (gpt2_sizes(), BIG),
}


def exact_params():
    params = []
    for name, spec in EXACT_CASES.items():
        counts, big = spec if isinstance(spec, tuple) else (spec, None)
        if SANITIZER and big:
            continue
        params.append(pytest.param(counts, id=name))
    return params


@pytest.mark.parametrize("counts", exact_params())
def test_the_norm_equals_the_emulation_bit_for_bit(flint, counts):
    # The workspace's block sums too: they pin pass 1's order and the workspace's layout apart
    # from pass 2's, which the norm pins.
    arrays = cpu_list(counts, 20 + len(counts))
    expected, partials = emulate(arrays, with_partials=True)

    got, views = run_arrays(flint, arrays, with_workspace=True)

    assert_norm_bits(got.bits, expected, f"norm of {len(counts)} tensors")
    assert got.nonfinite == 0
    assert [bits64(p) for p in got.sums] == [bits64(p) for p in partials], "block sums"
    assert not got.counts.any(), "block counts"
    del views


def test_the_crafted_input_equals_the_emulation(flint):
    # The input above, whose bits depend on pass 2's order of adds: the kernel gives the emulation's.
    arrays = crafted_pass_2()

    got, _ = run_arrays(flint, arrays)

    assert_norm_bits(got.bits, emulate(arrays), "crafted norm")
    assert got.bits != bits64(emulate(arrays, partial_order=(3, 2, 1, 0)))


@pytest.mark.parametrize("seed", [30, 31, 32])
def test_other_inputs_equal_the_emulation_too(flint, seed):
    counts = random_counts(40 if SANITIZER else 300, 3000 if SANITIZER else 20000, seed)
    arrays = cpu_list(counts, seed)

    got, _ = run_arrays(flint, arrays)

    assert_norm_bits(got.bits, emulate(arrays), f"norm seed {seed}")


def test_the_kernel_is_not_another_order_of_the_same_sums(flint):
    # The bits of the real kernel, not only of the emulation, differ from a plain sequential sum of
    # the same values, forwards and backwards: the comparison above has teeth on the GPU too.
    arrays = cpu_list(random_counts(1500, 300, 12), 13)

    got, _ = run_arrays(flint, arrays)

    assert_norm_bits(got.bits, emulate(arrays), "the kernel's order")
    squares = np.concatenate(arrays).astype(np.float64) ** 2
    assert got.bits != bits64(np.sqrt(np.cumsum(squares)[-1]))
    assert got.bits != bits64(np.sqrt(np.cumsum(squares[::-1])[-1]))


# -- The accuracy rule ---------------------------------------------------------------------------------

ACCURACY_KINDS = ["normal", "gradient", "scaled", "spread", "outliers"]
ACCURACY_SIZES = {
    "one": [1],
    "three": [3],
    "tails": [4095, 4096, 4097, 8191, 8193, 1, 5, 12289],
    "odd": [1, 3, 4097, 100003, 7, 4096 * 5 + 1, 2, 65537, 31],
    "many-small": [37] * 3000,
    "single": [200_003 if SANITIZER else (1 << 22) + 3],
}
if not SANITIZER:
    ACCURACY_SIZES["gpt2"] = gpt2_sizes()


def accuracy_tensors(kind, sizes, seed):
    """FP32 GPU tensors of these sizes: N(0, 1), N(0, 1e-3) (gradient-like), those times 2^16 (a
    loss-scaled gradient), N(0, 1) with each tensor's scale 10^u, u uniform from -6 to 2 (several
    decades), or N(0, 1) with one value in a thousand (and the middle one) 10,000 times larger."""
    g = generator(seed)
    scales = 10.0 ** (torch.rand(len(sizes), generator=torch.Generator().manual_seed(seed)) * 8 - 6)
    out = []
    for i, n in enumerate(sizes):
        x = torch.randn(n, generator=g, device="cuda")
        if kind == "gradient":
            x *= 1e-3
        elif kind == "scaled":
            x *= 1e-3 * 65536
        elif kind == "spread":
            x *= float(scales[i])
        elif kind == "outliers":
            x = torch.where(torch.rand(n, generator=g, device="cuda") < 1e-3, x * 1e4, x)
            x[n // 2] *= 1e4
        out.append(x)
    return out


def fp64_norm(tensors):
    """The same operation in FP64 on the FP32 values: the square root of the sum of their squares."""
    total = torch.zeros((), dtype=torch.float64, device="cuda")
    for t in tensors:
        t = t.reshape(-1)
        for lo in range(0, t.numel(), 1 << 24):
            total += t[lo : lo + (1 << 24)].double().square().sum()
    return math.sqrt(total.item())


def torch_norm(tensors):
    """FP32 PyTorch's norm, as clip_grad_norm_ computes it: the FP32 norm of each tensor, then the
    FP32 norm of those norms."""
    norms = torch._foreach_norm(list(tensors), 2.0)
    return torch.linalg.vector_norm(torch.stack(norms), 2.0).item()


def as_scalar(x):
    return torch.tensor([x], dtype=torch.float64)


@pytest.mark.parametrize("kind", ACCURACY_KINDS)
@pytest.mark.parametrize("sizes", ACCURACY_SIZES.values(), ids=list(ACCURACY_SIZES))
def test_the_error_is_at_most_twice_fp32_pytorchs(flint, kind, sizes):
    flint_errors, peer_errors, worst = [], [], 0.0
    for seed in gate.SEEDS:
        tensors = accuracy_tensors(kind, sizes, seed)
        ref = fp64_norm(tensors)
        got = run_flint(flint, tensors, offset=2 * (1 + seed)).norm
        peer = torch_norm(tensors)
        flint_errors.append(gate.errors(as_scalar(got), as_scalar(ref)))
        peer_errors.append(gate.errors(as_scalar(peer), as_scalar(ref)))
        worst = max(worst, abs(got - ref) / ref)
        del tensors
    bounds = gate.check_gate(2.0, flint_errors, peer_errors, output="fp32")
    print(f"NORM {kind} {len(sizes)} tensors of {sum(sizes)} values | flint rel {max(e.rel_l2 for e in flint_errors):.3g} "
          f"| fp32 torch rel {max(e.rel_l2 for e in peer_errors):.3g} | bound rel {bounds.rel_l2:.3g}")  # fmt: skip
    assert worst <= 1e-12, f"flint is {worst:.3g} (relative) from the FP64 norm"


# -- The values' edges --------------------------------------------------------------------------------


def test_values_near_the_fp32_maximum_give_a_finite_norm_where_an_fp32_sum_overflows(flint):
    n = 3000
    big = np.full(n, 3e38, dtype=np.float32)
    big[::3] *= -1
    biggest = np.full(n, np.float32(3.4028235e38), dtype=np.float32)
    biggest[1::2] *= -1
    arrays = [big, biggest, cpu_list([50], 41)[0]]
    assert np.isfinite(np.concatenate(arrays)).all()

    got, views = run_arrays(flint, arrays)

    assert got.nonfinite == 0
    assert math.isfinite(got.norm) and got.norm > 1e40
    assert_norm_bits(got.bits, emulate(arrays), "norm near FP32's maximum")
    assert abs(got.norm - fp64_norm(views)) <= 1e-12 * got.norm
    # The case bites: FP32 PyTorch's norm of the same values overflows.
    assert math.isinf(torch_norm(views)), torch_norm(views)
    assert math.isinf(torch.linalg.vector_norm(views[0]).item())


def test_fp32_subnormals_square_exactly_and_nothing_is_flushed(flint):
    # Down to 2^-149: the squares are normal FP64 values (2^-298 at the least).
    g = torch.Generator().manual_seed(42)
    random_bits = torch.randint(1, 2**23, (5000,), generator=g, dtype=torch.int32)
    random_bits = torch.where(torch.randint(0, 2, (5000,), generator=g) == 1, random_bits, random_bits | -(2**31))
    arrays = [
        random_bits.view(torch.float32).numpy().copy(),
        np.full(100, 2.0**-149, dtype=np.float32),
        np.full(3, 2.0**-126 - 2.0**-149, dtype=np.float32),  # the largest subnormal
    ]
    assert all((np.abs(a) < 2.0**-126).all() and (a != 0).all() for a in arrays)

    got, views = run_arrays(flint, arrays)

    assert got.nonfinite == 0 and got.norm > 0
    assert_norm_bits(got.bits, emulate(arrays), "norm of subnormals")
    assert abs(got.norm - fp64_norm(views)) <= 1e-12 * got.norm
    # 100 values of 2^-149 alone: sqrt(100) * 2^-149 = 10 * 2^-149, exactly.
    alone, _ = run_arrays(flint, [np.full(100, 2.0**-149, dtype=np.float32)])
    assert alone.norm == 10 * 2.0**-149
    one, _ = run_arrays(flint, [np.full(1, -(2.0**-149), dtype=np.float32)])
    assert one.norm == 2.0**-149


@pytest.mark.parametrize("count", [1, 3, 5000, 4097])
def test_negative_zero_gives_exactly_positive_zero(flint, count):
    arrays = [np.full(count, -0.0, dtype=np.float32), np.zeros(5, dtype=np.float32)]
    arrays[1][::2] = -0.0
    assert np.signbit(arrays[0]).all()

    got, _ = run_arrays(flint, arrays)

    assert got.bits == 0, f"the norm is {got.norm!r} ({got.bits & (2**64 - 1):#018x}), not +0"
    assert got.nonfinite == 0


def test_no_tensors_give_a_positive_zero_and_a_count_of_0(flint):
    got = run_flint(flint, [], norm_fill=FILLS["-nan"], nonfinite_fill=FILLS["garbage"])
    assert got.bits == 0 and got.nonfinite == 0
    # Entries that all have a count of 0 give the same, whatever their data.
    entries = [(0, 0), (0xDEADBEEF0000, 0)]
    got = run_flint(flint, entries, norm_fill=FILLS["-nan"], nonfinite_fill=FILLS["garbage"])
    assert got.bits == 0 and got.nonfinite == 0
    empty = torch.empty(0, device="cuda")
    got = run_flint(flint, [empty, empty], norm_fill=FILLS["nan"])
    assert got.bits == 0 and got.nonfinite == 0


# -- The overflow check ------------------------------------------------------------------------------

# Where the injections go: (counts of the tensors, [(tensor, element)], misalign), each place tried
# with every kind of non-finite value.
POSITIONS = {
    "first-element-of-the-first-tensor": ([4099, 1, 9000, 4096, 3, 12345, 4097, 7], [(0, 0)], 0),
    "last-element-of-the-first-tensor": ([4099, 1, 9000, 4096, 3, 12345, 4097, 7], [(0, 4098)], 0),
    "the-tensor-of-1": ([4099, 1, 9000, 4096, 3, 12345, 4097, 7], [(1, 0)], 0),
    "first-element-of-a-middle-tensor": ([4099, 1, 9000, 4096, 3, 12345, 4097, 7], [(4, 0)], 0),
    "last-element-of-a-middle-tensor": ([4099, 1, 9000, 4096, 3, 12345, 4097, 7], [(4, 2)], 0),
    "first-element-of-the-last-tensor": ([4099, 1, 9000, 4096, 3, 12345, 4097, 7], [(7, 0)], 0),
    "last-element-of-the-last-tensor": ([4099, 1, 9000, 4096, 3, 12345, 4097, 7], [(7, 6)], 0),
    "element-4095": ([9000, 12345], [(0, 4095)], 0),
    "element-4096": ([9000, 12345], [(0, 4096)], 0),
    "element-8191": ([9000, 12345], [(1, 8191)], 0),
    "element-8192": ([9000, 12345], [(1, 8192)], 0),
    "in-a-groups-16-byte-load-1": ([9000], [(0, 5)], 0),
    "in-a-groups-16-byte-load-2": ([9000], [(0, 1030)], 0),
    "in-a-groups-16-byte-load-3": ([9000], [(0, 3075)], 0),
    "scalar-tail-of-a-count-not-a-multiple-of-4": ([4099], [(0, 4096)], 0),
    "scalar-tail-of-a-count-of-4097": ([4097], [(0, 4096)], 0),
    "scalar-tail-after-a-vector-group": ([7], [(0, 5)], 0),
    "last-value-of-the-tail": ([4099], [(0, 4098)], 0),
    "misaligned-by-1-first": ([5000, 3000], [(0, 0), (1, 0)], 1),
    "misaligned-by-2-last": ([5000, 3000], [(0, 4999), (1, 2999)], 2),
    "misaligned-by-3-middle": ([5000, 3000], [(0, 2500), (1, 1500)], 3),
    "block-1023": ([1] * 1100, [(1023, 0)], 0),
    "block-0s-second-round": ([1] * 1100, [(1024, 0)], 0),
    "block-0s-first-round": ([1] * 1100, [(0, 0)], 0),
    "the-last-of-1100-chunks": ([1] * 1100, [(1099, 0)], 0),
    "several-in-one-group": ([9000, 2], [(0, 8), (0, 9), (0, 10), (0, 11)], 0),
    "several-in-different-blocks": ([5000] * 6, [(0, 4999), (1, 0), (2, 4096), (3, 100), (4, 4998), (5, 4097)], 0),
    "several-in-one-tensor": ([20000], [(0, 0), (0, 4095), (0, 4096), (0, 12000), (0, 19999)], 0),
}
BIG_POSITIONS = {
    "block-1023-of-a-big-tensor": ([1025 * CHUNK + 3], [(0, 1023 * CHUNK + 5)], 0),
    "block-0s-second-round-of-a-big-tensor": ([1025 * CHUNK + 3], [(0, 1024 * CHUNK + 7)], 0),
    "the-last-element-of-a-big-tensor": ([1025 * CHUNK + 3], [(0, 1025 * CHUNK + 2)], 0),
}
# A big tensor takes 4 MB per 1,000 chunks, so a sanitizer run skips it.
POSITION_PARAMS = [pytest.param(*spec, id=name) for name, spec in POSITIONS.items()]
BIG_POSITION_PARAMS = [] if SANITIZER else [pytest.param(*spec, id=name) for name, spec in BIG_POSITIONS.items()]


def inject(arrays, places, kinds):
    """Writes each kind's FP32 bit pattern at each place ((tensor, element)); a list of kinds goes
    one to a place, a string to every place."""
    kinds = [kinds] * len(places) if isinstance(kinds, str) else kinds
    for (t, i), kind in zip(places, kinds):
        arrays[t].view(np.uint32)[i] = NON_FINITE_BITS[kind]


def found_inf_of(tensors):
    """GradScaler's overflow check on copies of the tensors (it unscales in place)."""
    copies = [t.clone() for t in tensors if t.numel()]
    found = torch.zeros(1, device="cuda")
    torch._amp_foreach_non_finite_check_and_unscale_(copies, found, torch.ones(1, device="cuda"))
    return bool(found.item() > 0)


def expected_class(kinds):
    """What the norm is when values of these kinds are present. flint.h promises only "not
    finite"; this pins the current behaviour: infs alone sum to +inf, any NaN gives NaN."""
    return "nan" if any(k in NAN_KINDS for k in kinds) else "inf"


def check_non_finite(flint, counts, places, kinds, misalign, seed):
    arrays = cpu_list(counts, seed)
    inject(arrays, places, kinds)
    flat, views = to_gpu(arrays, misalign)

    got = run_flint(flint, views, watch=[flat])

    assert got.nonfinite == len(places) == expected_nonfinite(views)
    kinds = [kinds] * len(places) if isinstance(kinds, str) else kinds
    if expected_class(kinds) == "nan":
        assert math.isnan(got.norm), got.norm
    else:
        assert got.norm == math.inf, got.norm
    assert found_inf_of(views) is True


@pytest.mark.parametrize("kind", list(NON_FINITE_BITS))
@pytest.mark.parametrize("counts, places, misalign", POSITION_PARAMS)
def test_the_count_is_every_value_that_is_not_finite(flint, counts, places, misalign, kind):
    check_non_finite(flint, counts, places, kind, misalign, 50)


@pytest.mark.parametrize("kind", ["inf", "nan"])
@pytest.mark.parametrize("counts, places, misalign", BIG_POSITION_PARAMS)
def test_the_count_sees_a_non_finite_value_in_a_big_tensor(flint, counts, places, misalign, kind):
    check_non_finite(flint, counts, places, kind, misalign, 51)


def test_a_mix_of_kinds_counts_each_and_gives_nan_with_any_nan(flint):
    counts = [5000] * 6
    places = [(0, 4999), (1, 0), (2, 4096), (3, 100), (4, 4998), (5, 4097)]
    only_infs = ["inf", "-inf", "inf", "-inf", "-inf", "inf"]
    check_non_finite(flint, counts, places, only_infs, 0, 52)
    for kinds in (["inf", "-inf", "inf", "snan", "-inf", "inf"], ["nan"] * 6,
                  ["-nan", "nan-payload", "snan", "nan", "inf", "-inf"]):  # fmt: skip
        check_non_finite(flint, counts, places, kinds, 0, 52)


def test_infs_of_both_signs_sum_to_positive_infinity_not_nan(flint):
    # inf and -inf square to +inf, so the sum is +inf and the norm +inf: only a NaN gives NaN.
    arrays = cpu_list([100], 53)
    arrays[0][3], arrays[0][4] = np.inf, -np.inf

    got, _ = run_arrays(flint, arrays)

    assert got.norm == math.inf and got.nonfinite == 2


@pytest.mark.parametrize("seed", [60, 61])
def test_clean_values_have_a_count_of_0_and_the_scaler_finds_no_inf(flint, seed):
    arrays = cpu_list([4099, 1, 9000, 4096, 3], seed)

    got, views = run_arrays(flint, arrays)

    assert got.nonfinite == 0 and expected_nonfinite(views) == 0
    assert math.isfinite(got.norm)
    assert found_inf_of(views) is False


def test_the_count_equals_the_scalers_verdict_on_every_case(flint):
    # The scaler's found_inf is set exactly when the count is above 0: with clean tensors, with
    # one bad value, and with the bad value removed again.
    arrays = cpu_list([3000, 4097, 5], 62)
    flat, views = to_gpu(arrays)
    clean = run_flint(flint, views, watch=[flat])
    assert clean.nonfinite == 0 and found_inf_of(views) is False
    original = views[1][4096].clone()
    for kind in NON_FINITE_BITS:
        views[1].view(torch.int32)[4096] = as_int32(NON_FINITE_BITS[kind])
        bad = run_flint(flint, views, watch=[])
        assert bad.nonfinite == 1 and found_inf_of(views) is True, kind
        views[1][4096] = original
        again = run_flint(flint, views, watch=[])
        assert again.nonfinite == 0 and again.bits == clean.bits and found_inf_of(views) is False, kind


def as_int32(value):
    return (value + 2**31) % 2**32 - 2**31


# -- What the call reads and writes -------------------------------------------------------------------


@pytest.mark.parametrize("offset", [2, 4, 8, 62])
def test_nothing_is_written_around_the_results_or_the_workspace(flint, offset):
    # run_flint checks the guard bands, the room after the workspace, the tensors and the table.
    arrays = cpu_list([3000, 1, 4097, 12], 70)

    got, _ = run_arrays(flint, arrays, offset=offset, workspace_extra=4096)

    assert_norm_bits(got.bits, emulate(arrays), "norm")
    assert got.nonfinite == 0


@pytest.mark.parametrize("fill", list(FILLS))
def test_the_workspaces_and_the_old_results_content_changes_nothing(flint, fill):
    arrays = cpu_list([9000, 5, 4097], 71)
    expected = emulate(arrays)

    got, _ = run_arrays(flint, arrays, workspace_fill=FILLS[fill], norm_fill=FILLS[fill],
                        nonfinite_fill=FILLS[fill])  # fmt: skip

    assert_norm_bits(got.bits, expected, f"norm over a workspace and results of {fill}")
    assert got.nonfinite == 0


@pytest.mark.parametrize("extra", [16, 4096])
def test_the_workspace_may_be_larger_than_needed(flint, extra):
    arrays = cpu_list([9000, 5, 4097], 72)

    got, _ = run_arrays(flint, arrays, workspace_extra=extra)

    assert_norm_bits(got.bits, emulate(arrays), "norm")


@pytest.mark.parametrize("misalign", [1, 2, 3])
def test_a_misaligned_tensor_gives_the_bits_of_an_aligned_copy(flint, misalign):
    # Each tensor starts `misalign` floats past a 16-byte boundary: scalar loads throughout, the
    # same sums. The tensors have the padding of NaN around them, so a read past an end would show.
    counts = [5000, 3, 4099, 1, 9000, 17]
    arrays = cpu_list(counts, 73)
    flat, views = to_gpu(arrays, misalign)
    assert all(v.data_ptr() % 16 == 4 * misalign for v in views)
    aligned_flat, aligned = to_gpu(arrays)
    assert all(v.data_ptr() % 16 == 0 for v in aligned)

    got = run_flint(flint, views, watch=[flat])
    expected = run_flint(flint, aligned, watch=[aligned_flat])

    assert got.bits == expected.bits and got.nonfinite == expected.nonfinite == 0
    assert_norm_bits(got.bits, emulate(arrays), "misaligned norm")


def test_one_misaligned_tensor_among_aligned_ones(flint):
    arrays = cpu_list([4096, 5000, 4096], 74)
    flat, views = to_gpu(arrays)
    shifted = torch.zeros(5001, device="cuda")
    shifted[1:] = views[1]
    entries = [views[0], shifted[1:], views[2]]
    assert entries[1].data_ptr() % 16 == 4

    got = run_flint(flint, entries, watch=[flat, shifted])

    assert_norm_bits(got.bits, emulate(arrays), "norm")


def test_an_entry_of_count_0_reads_nothing(flint):
    # A null or garbage data pointer with a count of 0 is never read (memcheck stays clean), and
    # the entry is no chunk: the bits are those of the list without it.
    arrays = cpu_list([5000, 7, 4097], 75)
    flat, views = to_gpu(arrays)
    entries = [(0, 0), views[0], (0xDEADBEEF0000, 0), views[1], (1, 0), (0x7FFFFFFFFFFFFFF0, 0), views[2], (0, 0)]

    got = run_flint(flint, entries, watch=[flat])

    assert_norm_bits(got.bits, emulate(arrays), "norm with empty entries")
    assert got.nonfinite == 0


def test_a_tensor_listed_twice_counts_twice(flint):
    arrays = cpu_list([5000, 7, 4097], 76)
    flat, views = to_gpu(arrays)

    once = run_flint(flint, views, watch=[flat])
    twice = run_flint(flint, [views[0], views[1], views[1], views[2], views[0]], watch=[flat])

    assert_norm_bits(twice.bits, emulate([arrays[0], arrays[1], arrays[1], arrays[2], arrays[0]]), "norm")
    expected_once = math.sqrt(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in arrays))
    assert once.norm == pytest.approx(expected_once, rel=1e-12)
    assert twice.norm > once.norm
    # A non-finite value listed twice is counted twice.
    arrays[1][3] = np.nan
    flat, views = to_gpu(arrays)
    bad = run_flint(flint, [views[0], views[1], views[1]], watch=[flat])
    assert bad.nonfinite == 2 and math.isnan(bad.norm)


def test_the_same_values_in_overlapping_tensors_count_each_time(flint):
    # Tensors may overlap each other: a slice and the tensor it lies in.
    values = cpu_list([3000], 77)[0]
    flat, views = to_gpu([values])
    entries = [views[0], views[0][1000:2000], views[0][2000:]]

    got = run_flint(flint, entries, watch=[flat])

    assert_norm_bits(got.bits, emulate([values, values[1000:2000], values[2000:]]), "norm")


def test_the_workspace_is_16_kib(flint):
    assert flint.gradient_norm_workspace() == WORKSPACE_BYTES == 1024 * 16


# -- Determinism and streams ---------------------------------------------------------------------------


def test_run_twice_is_bitwise_identical(flint):
    for counts, nan in (([20000, 3, 9000], False), (random_counts(60 if SANITIZER else 200, 3000, 80), True)):
        arrays = cpu_list(counts, 81)
        if nan:
            arrays[len(arrays) // 2][1] = np.nan
        flat, views = to_gpu(arrays)

        first = run_flint(flint, views, offset=2, workspace_fill=FILLS["nan"], norm_fill=FILLS["nan"], watch=[flat])
        second = run_flint(flint, views, offset=4, workspace_fill=FILLS["garbage"], norm_fill=FILLS["garbage"],
                           nonfinite_fill=FILLS["garbage"], watch=[flat])  # fmt: skip
        third = run_flint(flint, views, offset=6, workspace_fill=0, norm_fill=0, nonfinite_fill=0, watch=[flat])

        assert (first.nonfinite > 0) == nan
        for name, other in (("second", second), ("third", third)):
            assert other.bits == first.bits and other.nonfinite == first.nonfinite, f"the {name} run differs"


def test_two_different_tables_in_a_row_on_one_workspace(flint):
    # Four calls on one stream, sharing one workspace, with no sync between them: each gives its
    # own table's bits.
    lists = [cpu_list(c, 90 + i) for i, c in enumerate(([9000, 5], [1] * 30, [4097, 4096, 3], [20000]))]
    flat_views = [to_gpu(a) for a in lists]
    tables = [flint_lib.tensor_table(views) for _, views in flat_views]
    workspace = torch.full((WORKSPACE_BYTES // 8,), FILLS["nan"], dtype=torch.int64, device="cuda")
    results = [(torch.zeros(1, dtype=torch.int64, device="cuda"), torch.zeros(1, dtype=torch.int64, device="cuda"))
               for _ in lists]  # fmt: skip

    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]):
        for i in order:
            flint.gradient_norm(*results[i], tables[i], len(lists[i]), workspace, WORKSPACE_BYTES)
        torch.cuda.synchronize()
        for i, arrays in enumerate(lists):
            assert_norm_bits(int(results[i][0].item()), emulate(arrays), f"table {i} in the order {order}")
            assert int(results[i][1].item()) == 0


STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
def test_runs_on_the_given_stream(flint, mode):
    arrays = cpu_list([20000, 9, 4097], 95)
    expected = emulate(arrays)
    sources = [torch.from_numpy(a).cuda() for a in arrays]
    starts, at = [], 0
    for a in arrays:
        starts.append(at)
        at += a.size + (-a.size) % 4
    destination = torch.full((at + 4,), math.nan, device="cuda")
    entries = [(destination.data_ptr() + 4 * s, a.size) for s, a in zip(starts, arrays)]
    table = flint_lib.tensor_table(entries)
    norm = torch.full((1,), FILLS["nan"], dtype=torch.int64, device="cuda")
    nonfinite = torch.full((1,), FILLS["nan"], dtype=torch.int64, device="cuda")
    workspace = torch.full((WORKSPACE_BYTES // 8,), FILLS["nan"], dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    # A first call loads the kernels' module, which takes longer than the sleep below: without this
    # warm-up a call on the wrong stream would start late enough to pass.
    run_flint(flint, entries)
    torch.cuda.synchronize()

    def enqueue(call_stream):
        # The copies into the tensors finish only after the sleep, so the norm is right only if both
        # kernels are queued on the same stream behind them (it is NaN where they ran first). With
        # the current stream the call has no stream argument, and torch's current stream is the side
        # one; with the default stream there is no side one at all.
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        for s, source in zip(starts, sources):
            destination[s : s + source.numel()].copy_(source)
        flint.gradient_norm(norm, nonfinite, table, len(entries), workspace, WORKSPACE_BYTES, call_stream)

    if mode == "default-stream":
        enqueue(None)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            enqueue(stream if mode == "explicit-stream" else None)
        stream.synchronize()

    assert int(nonfinite.item()) == 0
    assert_norm_bits(int(norm.item()), expected, mode)


# -- Argument errors ------------------------------------------------------------------------------------

NAMES = ("norm", "nonfinite", "workspace", "tensors")
TENSOR_COUNT = 2
BYTES = {"norm": 8, "nonfinite": 8, "workspace": WORKSPACE_BYTES, "tensors": 16 * TENSOR_COUNT}
SPACING = 32768
ARENA_SLOT = {name: (i + 2) * SPACING for i, name in enumerate(NAMES)}
FILL = 0x5A


def new_arena():
    return torch.full((8 * SPACING,), FILL, dtype=torch.uint8, device="cuda")


def addresses(arena):
    return {name: arena.data_ptr() + ARENA_SLOT[name] for name in NAMES}


def call(flint, a, tensor_count=TENSOR_COUNT, workspace_bytes=WORKSPACE_BYTES):
    flint.gradient_norm(a["norm"], a["nonfinite"], a["tensors"], tensor_count, a["workspace"], workspace_bytes)


def expect_invalid_argument(flint, reason, arena, a, **kwargs):
    """The call fails with the status and a message naming the function and the problem, and nothing
    in the arena changed (no kernel ran, the results keep their old bits)."""
    before = arena.clone()
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, a, **kwargs)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_gradient_norm", reason):
        assert text in str(error.value)
        assert text in flint.last_error()
    torch.cuda.synchronize()
    assert torch.equal(arena, before), "the call wrote something"


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    arena = new_arena()
    expect_invalid_argument(flint, f"{name} is null", arena, {**addresses(arena), name: None})


@pytest.mark.parametrize("offset", [1, 4, 8])
@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name, offset):
    # 4 bytes is as far off as an FP32 goes, 8 is aligned for 64 bits but not 128.
    arena = new_arena()
    a = addresses(arena)
    a[name] += offset
    expect_invalid_argument(flint, f"{name} is not 16-byte aligned", arena, a)


@pytest.mark.parametrize("name", ["workspace", "tensors"])
def test_a_buffer_may_not_run_past_the_address_space(flint, name):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"{name} runs past the end of the address space", arena, {**addresses(arena), name: 2**64 - 16}
    )


def test_pointers_are_checked_in_order(flint):
    # Per buffer: null, then alignment, then the address space; norm, nonfinite, workspace, tensors;
    # and all of that before any overlap.
    arena = new_arena()
    a = addresses(arena)
    expect_invalid_argument(flint, "norm is not 16-byte aligned", arena, {**a, "norm": a["norm"] + 4, "nonfinite": None})
    expect_invalid_argument(flint, "norm is null", arena, {**a, "norm": None, "nonfinite": a["nonfinite"] + 4})
    expect_invalid_argument(flint, "nonfinite is null", arena, {**a, "nonfinite": None, "workspace": a["workspace"] + 4})
    expect_invalid_argument(flint, "workspace is not 16-byte aligned", arena,
                            {**a, "workspace": a["workspace"] + 4, "tensors": None})  # fmt: skip
    expect_invalid_argument(flint, "workspace runs past the end", arena, {**a, "workspace": 2**64 - 16, "tensors": None})
    expect_invalid_argument(flint, "tensors is null", arena, {**a, "tensors": None, "norm": a["nonfinite"]})
    expect_invalid_argument(flint, "norm and nonfinite overlap", arena, {**a, "norm": a["nonfinite"]})


def test_a_workspace_may_not_be_too_small(flint):
    arena = new_arena()
    for short in (1, 4, 16, 17, 8192, WORKSPACE_BYTES):
        given = WORKSPACE_BYTES - short
        expect_invalid_argument(flint, f"needs {WORKSPACE_BYTES} bytes of workspace, given {given}", arena,
                                addresses(arena), workspace_bytes=given)  # fmt: skip


def test_size_limits_are_checked_in_order(flint):
    # tensor_count, then the workspace's size, then the pointers (null ones included).
    arena = new_arena()
    nothing = {name: None for name in NAMES}
    huge = 2**60
    expect_invalid_argument(flint, f"tensor_count ({huge}) can be up to {huge - 1}", arena, nothing,
                            tensor_count=huge, workspace_bytes=0)  # fmt: skip
    expect_invalid_argument(flint, f"needs {WORKSPACE_BYTES} bytes of workspace, given 0", arena, nothing,
                            workspace_bytes=0)  # fmt: skip
    expect_invalid_argument(flint, "norm is null", arena, nothing)


@pytest.mark.parametrize("count", [2**60, 2**60 + 1, 2**63, 2**64 - 1], ids=str)
def test_a_tensor_count_whose_table_size_wraps_is_rejected(flint, count):
    # tensor_count * 16 wraps for 2^60 and up (SIZE_MAX / 16 + 1 is 2^60).
    arena = new_arena()
    expect_invalid_argument(flint, f"tensor_count ({count}) can be up to {2**60 - 1}", arena,
                            addresses(arena), tensor_count=count)  # fmt: skip


def test_the_largest_table_that_does_not_wrap_runs_past_the_address_space(flint):
    # SIZE_MAX / 16 entries are 2^64 - 16 bytes, which no address leaves room for.
    arena = new_arena()
    expect_invalid_argument(flint, "tensors runs past the end of the address space", arena,
                            addresses(arena), tensor_count=2**60 - 1)  # fmt: skip


def overlapping_offsets(first, second):
    """Where `second` starts relative to `first` (a multiple of 16) so that they overlap: at the same
    address, 16 bytes in, in first's last 16 bytes, and (negative) with second's last 16 bytes
    inside first's start. Those that really overlap, without duplicates."""
    a, b = BYTES[first], BYTES[second]
    candidates = {"same-start": 0, "16-in": 16, "last-16-of-first": (a - 1) // 16 * 16,
                  "ends-in-first": -((b - 1) // 16 * 16)}  # fmt: skip
    out = {}
    for label, d in candidates.items():
        if d % 16 == 0 and d < a and d + b > 0 and d not in out.values():
            out[label] = d
    return out


PAIRS = [(a, b) for i, a in enumerate(NAMES[:3]) for b in NAMES[i + 1 :]]
OVERLAP_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in PAIRS
    for label, offset in overlapping_offsets(a, b).items()
]


@pytest.mark.parametrize("first, second, label, offset", OVERLAP_CASES)
def test_the_results_the_workspace_and_the_table_may_not_overlap(flint, first, second, label, offset):
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no harm
    # to anything else.
    arena = new_arena()
    a = addresses(arena)
    a[second] = a[first] + offset
    expect_invalid_argument(flint, f"{first} and {second} overlap", arena, a)


def test_the_overlap_cases_cover_every_pair(flint):
    assert {(p.values[0], p.values[1]) for p in OVERLAP_CASES} == set(PAIRS)
    assert len(PAIRS) == 6


TOUCHING_CASES = [
    pytest.param(a, b, label, id=f"{a}-{b}-{label}") for a, b in PAIRS for label in ("just-after", "just-before")
]


@pytest.mark.parametrize("first, second, label", TOUCHING_CASES)
def test_buffers_next_to_each_other_are_not_overlapping(flint, first, second, label):
    arena = new_arena()
    start = dict(ARENA_SLOT)
    rounded = lambda n: -(-n // 16) * 16  # noqa: E731
    start[second] = start[first] + (rounded(BYTES[first]) if label == "just-after" else -rounded(BYTES[second]))
    arrays = cpu_list([4097, 5], 120)
    flat, views = to_gpu(arrays)
    rows = torch.tensor([[v.data_ptr(), v.numel()] for v in views], dtype=torch.int64, device="cuda")
    arena[start["tensors"] : start["tensors"] + 32] = rows.view(torch.uint8).flatten()
    used = torch.zeros(arena.numel(), dtype=torch.bool, device="cuda")
    for name in ("norm", "nonfinite", "workspace"):
        used[start[name] : start[name] + BYTES[name]] = True
    outside_before = arena[~used].clone()
    base = arena.data_ptr()
    a = {name: base + start[name] for name in NAMES}

    call(flint, a)
    torch.cuda.synchronize()

    assert torch.equal(arena[~used], outside_before), "something outside the results and the workspace changed"
    bits = int(arena[start["norm"] : start["norm"] + 8].view(torch.int64).item())
    assert_norm_bits(bits, emulate(arrays), label)
    assert int(arena[start["nonfinite"] : start["nonfinite"] + 8].view(torch.int64).item()) == 0


def test_an_error_leaves_nothing_behind_and_a_valid_call_works_after(flint):
    arena = new_arena()
    a = addresses(arena)
    arrays = cpu_list([4097, 5], 121)
    flat, views = to_gpu(arrays)
    rows = torch.tensor([[v.data_ptr(), v.numel()] for v in views], dtype=torch.int64, device="cuda")
    arena[ARENA_SLOT["tensors"] : ARENA_SLOT["tensors"] + 32] = rows.view(torch.uint8).flatten()
    for kwargs in (dict(tensor_count=2**60), dict(workspace_bytes=0), dict(workspace_bytes=WORKSPACE_BYTES - 1)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, a, **kwargs)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "norm": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "workspace": a["workspace"] + 4})
    torch.cuda.synchronize()
    assert all_equal(arena[ARENA_SLOT["norm"] : ARENA_SLOT["norm"] + 8], FILL)
    assert all_equal(arena[ARENA_SLOT["nonfinite"] : ARENA_SLOT["nonfinite"] + 8], FILL)

    call(flint, a)
    torch.cuda.synchronize()

    bits = int(arena[ARENA_SLOT["norm"] : ARENA_SLOT["norm"] + 8].view(torch.int64).item())
    assert_norm_bits(bits, emulate(arrays), "after errors")


def test_the_workspace_function_needs_a_bytes_pointer(flint):
    status = flint.lib.flint_gradient_norm_workspace(None)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_gradient_norm_workspace: bytes is null" in flint.last_error()


def test_the_workspace_function_writes_the_size(flint):
    import ctypes

    size = ctypes.c_size_t(0xDEADBEEF)

    assert flint.lib.flint_gradient_norm_workspace(ctypes.byref(size)) == flint_lib.OK
    assert size.value == WORKSPACE_BYTES


# -- The model ---------------------------------------------------------------------------------------

MODEL_SHAPES = {"tiny": (TINY, 2, 32), **ODD_MODELS}
if SANITIZER:
    MODEL_SHAPES = {name: MODEL_SHAPES[name] for name in ("one-head-of-8", "heads-of-24")}


def gradient_arrays(m):
    """The model's gradients as FP32 numpy arrays, in the order of the parameters."""
    return [g.cpu().numpy().reshape(-1) for g in read_gradients(m).values()]


def gradient_entries(m):
    return [(address, rows * cols) for _, address, rows, cols in m.gradients()]


def check_model_norm(flint, m, what):
    """The model's norm is flint_gradient_norm's, on a table of the same gradients, and the
    emulation's on the gradients read back, bit for bit; its count is 0."""
    m.gradient_norm()
    norm, count = m.read_gradient_norm()
    by_hand = run_flint(flint, gradient_entries(m))
    expected = emulate(gradient_arrays(m))
    assert count == 0 and by_hand.nonfinite == 0, what
    assert bits64(norm) == by_hand.bits, f"{what}: the model's norm {norm!r}, by hand {by_hand.norm!r}"
    assert_norm_bits(bits64(norm), expected, f"{what}: against the emulation")
    assert norm > 0 and math.isfinite(norm)
    return norm


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
@pytest.mark.parametrize("name", MODEL_SHAPES)
def test_the_models_norm_is_the_functions_on_its_gradients_and_the_emulations(flint, name, separate):
    cfg, sequences, length = MODEL_SHAPES[name]
    state = state_of(make_model(cfg, 3, "mixed"))
    a, b = random_batch(cfg, sequences, length, 4), random_batch(cfg, sequences, length, 5)
    scale = gate_scale(a.rows)
    with Gpt2(flint, backward_config(cfg, sequences, length, separate)) as m:
        m.load(state)
        forward_backward(m, a, scale)
        once = check_model_norm(flint, m, "one micro-batch")
        forward_backward(m, b, scale, accumulate=True)
        twice = check_model_norm(flint, m, "two micro-batches")
    assert twice != once
    print(f"MODEL {name} {'separate' if separate else 'one allocation'}: norm {once!r} then {twice!r}")


def injection_places(cfg):
    """(name, parameter, element) of the gradients to poison: wte's first element, a middle layer's
    bias's middle one, and ln_f's last."""
    middle = cfg.n_layer // 2
    return [
        ("wte-first", "wte.weight", 0),
        (f"layer-{middle}-bias", f"blocks.{middle}.attn.c_proj.bias", cfg.n_embd // 2),
        ("ln_f-last", "ln_f.bias", cfg.n_embd - 1),
    ]


@pytest.mark.parametrize("kind", ["inf", "-inf", "nan"])
@pytest.mark.parametrize("name", ["heads-of-24"] if SANITIZER else ["tiny", "heads-of-24"])
def test_an_inf_or_a_nan_in_one_gradient_is_counted_once(flint, name, kind):
    cfg, sequences, length = {"tiny": (TINY, 2, 32), **ODD_MODELS}[name]
    state = state_of(make_model(cfg, 6, "mixed"))
    batch = random_batch(cfg, sequences, length, 7)
    with Gpt2(flint, backward_config(cfg, sequences, length)) as m:
        m.load(state)
        forward_backward(m, batch, gate_scale(batch.rows))
        m.gradient_norm()
        clean_norm, clean_count = m.read_gradient_norm()
        assert clean_count == 0
        by_name = {n: (address, rows * cols) for n, address, rows, cols in m.gradients()}
        for label, parameter, element in injection_places(cfg):
            address, size = by_name[parameter]
            assert element < size
            at = address + 4 * element
            original = m.read(address, (1,), torch.float32, element)
            bad = torch.tensor([as_int32(NON_FINITE_BITS[kind])], dtype=torch.int32, device="cuda")
            m.flint.copy(at, bad, 4)
            torch.cuda.synchronize()

            m.gradient_norm()
            norm, count = m.read_gradient_norm()

            assert count == 1, label
            if kind == "nan":
                assert math.isnan(norm), label
            else:
                assert norm == math.inf, label
            m.flint.copy(at, original, 4)
            torch.cuda.synchronize()
            m.gradient_norm()
            again, count = m.read_gradient_norm()
            assert count == 0 and bits64(again) == bits64(clean_norm), f"{label}: after restoring the value"


def test_a_model_without_backward_and_a_null_model_fail_the_norm_and_its_outputs(flint):
    config = gpt2_config(TINY, 1, 8)
    with Gpt2(flint, config) as m:
        for function, call_ in (("flint_gpt2_gradient_norm", m.gradient_norm),
                                ("flint_gpt2_gradient_norm_outputs", m.gradient_norm_outputs)):  # fmt: skip
            with pytest.raises(flint_lib.FlintError) as error:
                call_()
            assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
            assert function in str(error.value) and "without backward" in str(error.value), str(error.value)
    lib = flint.lib
    assert lib.flint_gpt2_gradient_norm(None, None) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_gpt2_gradient_norm" in flint.last_error() and "model is null" in flint.last_error(), flint.last_error()
    assert lib.flint_gpt2_gradient_norm_outputs(None, None, None) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_gpt2_gradient_norm_outputs" in flint.last_error() and "model is null" in flint.last_error(), flint.last_error()


def test_the_outputs_accept_null_out_pointers(flint):
    import ctypes

    with Gpt2(flint, backward_config(TINY, 1, 8)) as m:
        norm_at, nonfinite_at = m.gradient_norm_outputs()
        assert flint.lib.flint_gpt2_gradient_norm_outputs(m.handle, None, None) == flint_lib.OK
        only = ctypes.c_void_p()
        assert flint.lib.flint_gpt2_gradient_norm_outputs(m.handle, ctypes.byref(only), None) == flint_lib.OK
        assert only.value == norm_at
        only = ctypes.c_void_p()
        assert flint.lib.flint_gpt2_gradient_norm_outputs(m.handle, None, ctypes.byref(only)) == flint_lib.OK
        assert only.value == nonfinite_at
    assert norm_at and nonfinite_at and norm_at != nonfinite_at and norm_at % 16 == 0 and nonfinite_at % 16 == 0


def test_the_models_norm_run_twice_is_bitwise_identical(flint):
    cfg, sequences, length = ODD_MODELS["heads-of-24"]
    state = state_of(make_model(cfg, 8, "mixed"))
    batch = random_batch(cfg, sequences, length, 9)
    with Gpt2(flint, backward_config(cfg, sequences, length)) as m:
        m.load(state)
        forward_backward(m, batch, gate_scale(batch.rows))
        norm_at, nonfinite_at = m.gradient_norm_outputs()
        m.gradient_norm()
        first = m.read_gradient_norm()
        # The results overwritten with garbage in between.
        m.flint.copy(norm_at, torch.full((1,), 0x1234, dtype=torch.int64, device="cuda"), 8)
        m.flint.copy(nonfinite_at, torch.full((1,), 99, dtype=torch.int64, device="cuda"), 8)
        torch.cuda.synchronize()
        m.gradient_norm()
        second = m.read_gradient_norm()
    assert bits64(first[0]) == bits64(second[0]) and first[1] == second[1] == 0


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_the_models_norm_runs_on_the_given_stream(flint, separate):
    cfg, sequences, length = ODD_MODELS["heads-of-24"]
    state = state_of(make_model(cfg, 10, "mixed"))
    batch = random_batch(cfg, sequences, length, 11)
    scale = gate_scale(batch.rows)
    config = backward_config(cfg, sequences, length, separate)

    def run(m, stream=None):
        forward_backward(m, batch, scale, stream=stream)
        m.gradient_norm(stream)
        return m.read_gradient_norm(stream)

    stream = torch.cuda.Stream()
    # The second model is made while the first is alive, so its memory can't be the first's with
    # the expected gradients still in it, and its gradients are poisoned before the stream starts.
    with Gpt2(flint, config) as first, Gpt2(flint, config) as m:
        first.load(state)
        default = run(first)
        poison_gradients(m)
        # The sleep keeps the stream busy: a norm that ran on another stream would race ahead of the
        # weights' copy and the backward pass, read the poisoned gradients and give other bits.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.load(state, stream=stream)
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.sync_weights(stream)
            side = run(m, stream)
        stream.synchronize()
        assert bits64(side[0]) == bits64(default[0]) and side[1] == default[1] == 0
        # Behind a sleep and a write of an inf into a gradient, on the same stream, the norm sees it.
        name, address, rows, cols = m.gradients()[-1]
        bad = torch.tensor([as_int32(NON_FINITE_BITS["inf"])], dtype=torch.int32, device="cuda")
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.flint.copy(address, bad, 4, stream)
            m.gradient_norm(stream)
        stream.synchronize()
        assert m.read_gradient_norm() == (math.inf, 1)
