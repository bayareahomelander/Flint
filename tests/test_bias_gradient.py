"""flint_bias_gradient: a linear layer's bias gradient, the column sums of dy. dy is rows x cols
FP16 (row-major, contiguous), dbias is cols FP32; dbias[j] = the sum over r of dy[r, j], or with
accumulate, dbias[j] += that sum with one FP32 add (micro-batches). It takes a workspace of
ceil(rows / 256) * cols * 8 bytes and runs two kernels with no atomics. The sums are compensated
(TwoSum): plain FP32 sums missed the owner's 2x bound on the 2080 Ti (8453 x 8 with outliers near
3,000), so each sum is a pair (s, e) of a rounded sum and the sum of the rounding errors left behind,
both starting at (+0, +0), with every operation an IEEE FP32 round-to-nearest one:

  - add(a, v): t = s + v; vp = t - s; d = (s - (t - vp)) + (v - vp); e = e + d; s = t.
  - merge(a, b), folding pair b into a: errs = a.e + b.e; a.e = +0; add(a, b.s); a.e = a.e + errs.
  - Pass 1 sums chunks of 256 rows (the last may be shorter). For column j and chunk c, thread
    y = 0..7 has a pair that adds dy[r, j] for r = 256c + y, + 8, ... in order, then thread 0 merges
    threads 1..7 in order of y (a thread without rows holds (0, 0)); the chunk's pair is stored as
    partial_s[c, j] and partial_e[c, j], two FP32 rows per chunk.
  - Pass 2, per column: lane l (of 32) starts at (0, 0) and merges (partial_s, partial_e) of chunks
    l, l + 32, ... in order, then a butterfly (offsets 16, 8, 4, 2, 1, every lane merging its
    xor-partner's pair from before the step, lanes without chunks included) gives lane 0 the total,
    whose value is s + e, or s alone when s is inf or NaN (an inf passing through TwoSum makes e
    NaN). dbias is that value, or with accumulate old + that value.

The order depends only on rows and cols, so the bits are the same on every GPU, and a numpy
float32 emulation of exactly that order (emulate below, itself checked against a literal scalar
transcription of the description) must equal Flint's dbias bit for bit. Its inputs are made on the
CPU with a seeded generator and copied over, so they are the same on every machine: matching the
emulation on each machine is what shows the result is the same on every GPU. A test also shows
which changes of order the bit-exact comparisons can tell apart.

What the tests cover:
  - the correctness gate (PLAN.md's standard) against the FP64 column sums of the FP16 dy, with
    PyTorch's autocast path for a linear layer's bias gradient as the peer (an FP16-rounded
    gradient), on GPT-2's shapes (16,384 rows of 768, 2,304 and 3,072 columns, and 1,024 x 768)
    and odd ones, on N(0, 1), N(0.3, 1) and gradient-like N(0, 1e-3) inputs;
  - a tighter bound the owner chose: Flint's error against the FP64 reference at most twice
    FP32 PyTorch's own (dy.float().sum(0)), on N(0, 1), N(0.3, 1), gradient-like N(0, 1e-3),
    outliers (std 3,000 in 1% of the columns) and magnitudes near FP16's range with subnormals
    mixed in (the compensated sums are what make it hold at every shape);
  - the bit-exact emulation: rows that aren't a multiple of 8 or of 256, one row, more than 32
    chunks (lanes taking several, an uneven number each), a partial last 256-column tile, cols 8,
    and accumulate;
  - accumulate (old + the overwrite result, bit for bit, for olds of +-0, subnormal, inf, NaN and
    large values; micro-batches in a row equal the sequential FP32 adds) and overwrite ignoring
    dbias's old content;
  - inf and NaN in dy (different chunks, y lanes and tiles) reach exactly their column;
  - what the call writes and reads: guard bands around dbias and the workspace, dy unchanged,
    garbage in the workspace changing nothing, a larger workspace, cols = 0 doing nothing;
  - determinism and streams, the workspace size, the limits (the widest cols, rows past 65,535
    chunks), and the argument errors.

The slow-to-reach cases are cheap in memory: the limits take at most a few hundred MB. Under the
sanitizer (scripts/sanitize.py) the shapes shrink to at most 300,000 elements (cols up to 520,
still several chunks, a partial tile, and past 32 chunks at 8 columns), every tensor stays small,
and the tests of the limits are skipped."""

import ctypes
import gc
import math

import numpy as np
import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator, normal

CHUNK_ROWS = 256
MAX_COLS = 65_535 * 256  # 16,776,960
MAX_ROWS = 0x7FFFFFFF * CHUNK_ROWS  # 549,755,813,632

# Elements a sanitizer run may use per tensor.
SANITIZER_ELEMENTS = 300_000
# Tensors up to this many elements are compared with a copy after the call.
CHECK_INPUT_ELEMENTS = 1 << 26

# The outputs of a guarded call sit `offset` elements past the start of a buffer that is filled
# with a pattern, which must be intact around them afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A5A5A


def as_int32(value):
    return (value + 2**31) % 2**32 - 2**31


# FP32 bit patterns.
NAN32 = as_int32(0x7FC00000)
FILLS = {
    "nan": as_int32(0x7FC00000),
    "-nan": as_int32(0xFFFFFFFF),
    "inf": as_int32(0x7F800000),
    "-inf": as_int32(0xFF800000),
    "big": as_int32(0x7F7F7F7F),
    "zero": 0,
    "garbage": "garbage",
}
# FP16 bit patterns.
HALF_INF = 0x7C00
HALF_NEG_INF = 0xFC00
HALF_NAN = 0x7E00


@pytest.fixture(autouse=True)
def free_gpu_memory():
    """The laptop's GPU is 6 GiB and shared, so each test hands back what it cached."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


def shape_id(shape):
    return "x".join(map(str, shape))


def shrunk(shapes):
    """The shapes, with those too big for the sanitizer dropped, and duplicates."""
    if SANITIZER:
        shapes = [s for s in shapes if s[0] * s[1] <= SANITIZER_ELEMENTS]
    return list(dict.fromkeys(shapes))


def shape_params(shapes):
    return [pytest.param(*shape, id=shape_id(shape)) for shape in shapes]


# -- Comparing -------------------------------------------------------------


def canonical(t):
    """An FP32 tensor as int32 bits, with every NaN the same one (a NaN's payload isn't
    pinned: only that the result is a NaN)."""
    t = t.detach().cpu().contiguous()
    bits = t.view(torch.int32).clone()
    return torch.where(torch.isnan(t), torch.full_like(bits, 0x7FC00000), bits)


def assert_same_bits(actual, expected, what):
    """Two FP32 vectors (tensors, or numpy arrays for the expected one) are equal bit for bit,
    NaNs aside. torch.equal, not `==`: pytest hangs explaining a failed tensor assert."""
    if isinstance(expected, np.ndarray):
        expected = torch.from_numpy(expected)
    a, e = canonical(actual), canonical(expected)
    assert a.shape == e.shape, f"{what}: shapes {tuple(a.shape)} and {tuple(e.shape)}"
    if torch.equal(a, e):
        return
    bad = (a != e).nonzero().flatten()
    first = int(bad[0])
    raise AssertionError(
        f"{what}: {bad.numel()} of {a.numel()} elements differ, the first at {first}: "
        f"{actual.reshape(-1)[first].item()!r} against {expected.reshape(-1)[first].item()!r}"
    )


def all_equal(tensor, value):
    """Whether every element equals the value (compared out here, not in an assert)."""
    return bool((tensor == value).all())


# -- The emulation of the kernels' order of additions ------------------------

EMULATION_BATCH_ELEMENTS = 1 << 22
Y_ORDER = tuple(range(1, 8))
BUTTERFLY = (16, 8, 4, 2, 1)


def two_sum_add(s, e, v):
    """add(a, v) of the kernel on numpy float32 arrays (or scalars), in its order: t = s + v;
    vp = t - s; d = (s - (t - vp)) + (v - vp); e = e + d; s = t. Returns (t, e)."""
    t = s + v
    vp = t - s
    d1 = t - vp
    d2 = s - d1
    d3 = v - vp
    d = d2 + d3
    return t, e + d


def two_sum_merge(a, b):
    """merge(a, b) of the kernel: errs = a.e + b.e; a.e = +0; add(a, b.s); a.e = a.e + errs.
    a and b are (sum, error) pairs of float32 arrays (a is not changed)."""
    errs = a[1] + b[1]
    s, e = two_sum_add(a[0], np.zeros_like(a[1]), b[0])
    return s, e + errs


def emulate_plain(dy, old=None):
    """The same order as emulate with plain FP32 sums and no errors (the first version of the
    kernel): used only to show that the compensation changes the bits."""
    x = dy.numpy() if isinstance(dy, torch.Tensor) else dy
    rows, cols = x.shape
    chunks = -(-rows // CHUNK_ROWS)
    full = rows // CHUNK_ROWS
    f32 = np.float32
    with np.errstate(all="ignore"):
        partial = np.empty((chunks, cols), dtype=f32)
        step = max(1, EMULATION_BATCH_ELEMENTS // (CHUNK_ROWS * cols))
        for c0 in range(0, full, step):
            c1 = min(full, c0 + step)
            block = x[c0 * CHUNK_ROWS : c1 * CHUNK_ROWS].astype(f32).reshape(c1 - c0, 32, 8, cols)
            s = np.zeros((c1 - c0, 8, cols), dtype=f32)
            for i in range(32):
                s += block[:, i]
            p = s[:, 0].copy()
            for y in range(1, 8):
                p += s[:, y]
            partial[c0:c1] = p
        if full < chunks:
            n = rows - full * CHUNK_ROWS
            first = full * CHUNK_ROWS
            s = np.zeros((min(8, n), cols), dtype=f32)
            for i in range(-(-n // 8)):
                piece = x[first + 8 * i : first + min(8 * i + 8, n)].astype(f32)
                s[: len(piece)] += piece
            p = s[0].copy()
            for y in range(1, len(s)):
                p += s[y]
            partial[full] = p
        lanes = min(32, chunks)
        groups = chunks // 32
        t = np.zeros((lanes, cols), dtype=f32)
        if groups:
            grid = partial[: groups * 32].reshape(groups, 32, cols)
            for k in range(groups):
                t += grid[k]
        if chunks % 32:
            t[: chunks - groups * 32] += partial[groups * 32 :]
        lane = [t[l] for l in range(lanes)] + [None] * (32 - lanes)
        for offset in BUTTERFLY:
            nxt = []
            for l in range(32):
                a, b = lane[l], lane[l ^ offset]
                nxt.append(b if a is None else a if b is None else a + b)
            lane = nxt
        total = lane[0]
        if old is not None:
            total = np.asarray(old, dtype=f32) + total
    return total


def emulate(dy, old=None, y_order=Y_ORDER, reverse_chunks=False, butterfly=BUTTERFLY, with_partials=False):
    """The kernels' dbias for dy (rows x cols FP16, a CPU torch tensor or numpy array), as a
    float32 numpy array: bit for bit the compensated FP32 additions of the module docstring, in
    float32 throughout. With old (cols FP32), old + the total, with one add. Vectorized over
    columns (and over chunks in pass 1). With with_partials, also pass 1's partial sums and
    errors, (chunks, cols) each, as the workspace holds them.

    A pair that is still (+0, +0) is left out of a merge rather than merged in: for a finite sum
    that changes nothing (the merge adds 0 and errors of 0, and a sum or error is never -0), and
    for an inf or NaN sum it changes only the error, which the result doesn't use. (The literal
    scalar_reference merges every one, and test_the_emulation_matches_a_literal_scalar_
    transcription checks the two against each other, non-finite columns included.) That keeps
    the memory of the widest dy small.

    y_order, reverse_chunks and butterfly change the order of pass 1's merges of the 8 threads,
    of a lane's chunks (backwards), and of the butterfly's offsets: the defaults are the
    kernel's, and the others exist to show what the bit-exact tests can tell apart."""
    x = dy.numpy() if isinstance(dy, torch.Tensor) else dy
    assert x.dtype == np.float16 and x.ndim == 2
    rows, cols = x.shape
    chunks = -(-rows // CHUNK_ROWS)
    full = rows // CHUNK_ROWS
    f32 = np.float32
    with np.errstate(all="ignore"):
        partial_s = np.empty((chunks, cols), dtype=f32)
        partial_e = np.empty((chunks, cols), dtype=f32)

        def thread_merges(s, e, threads):
            """Thread 0's pair merged with the other threads' (those that have rows), in order."""
            a = (s[:, 0].copy(), e[:, 0].copy())
            for y in y_order:
                if y < threads:
                    a = two_sum_merge(a, (s[:, y], e[:, y]))
            return a

        # Pass 1, over batches of whole chunks: thread y adds rows 256c + y + 8i in order of i.
        step = max(1, EMULATION_BATCH_ELEMENTS // (CHUNK_ROWS * cols))
        for c0 in range(0, full, step):
            c1 = min(full, c0 + step)
            block = x[c0 * CHUNK_ROWS : c1 * CHUNK_ROWS].astype(f32).reshape(c1 - c0, 32, 8, cols)
            s = np.zeros((c1 - c0, 8, cols), dtype=f32)
            e = np.zeros((c1 - c0, 8, cols), dtype=f32)
            for i in range(32):
                s, e = two_sum_add(s, e, block[:, i])
            partial_s[c0:c1], partial_e[c0:c1] = thread_merges(s, e, 8)
        # The last chunk, when it has fewer than 256 rows: threads past its rows hold (0, 0).
        if full < chunks:
            n = rows - full * CHUNK_ROWS
            first = full * CHUNK_ROWS
            threads = min(8, n)
            s = np.zeros((1, threads, cols), dtype=f32)
            e = np.zeros((1, threads, cols), dtype=f32)
            for i in range(-(-n // 8)):
                piece = x[first + 8 * i : first + min(8 * i + 8, n)].astype(f32)
                k = len(piece)
                s[0, :k], e[0, :k] = two_sum_add(s[0, :k], e[0, :k], piece)
            partial_s[full], partial_e[full] = (v[0] for v in thread_merges(s, e, threads))
        total = pass2(partial_s, partial_e, old, reverse_chunks, butterfly)
    if with_partials:
        return total, partial_s, partial_e
    return total


def pass2(partial_s, partial_e, old=None, reverse_chunks=False, butterfly=BUTTERFLY):
    """The second pass of the kernels on pass 1's partial sums and errors (float32 numpy
    arrays, chunks x cols each): the total per column as a float32 array, or with old (cols
    FP32), old + the total, with one add. Lane l merges chunks l, l + 32, ... in order (backwards
    with reverse_chunks), then the butterfly merges the lanes (in the order of the offsets
    given); a lane without chunks holds (+0, +0), left out of the merges as in emulate. The value
    is s + e, or s alone when it is inf or NaN. LayerNorm's backward pass reuses it for dgamma."""
    chunks, cols = partial_s.shape
    f32 = np.float32
    with np.errstate(all="ignore"):
        lanes = min(32, chunks)
        groups = chunks // 32
        rem = chunks - groups * 32
        t = (np.zeros((lanes, cols), dtype=f32), np.zeros((lanes, cols), dtype=f32))
        if groups:
            grid_s = partial_s[: groups * 32].reshape(groups, 32, cols)
            grid_e = partial_e[: groups * 32].reshape(groups, 32, cols)

        def merge_remainder(t):
            if not rem:
                return t
            top = (t[0][:rem], t[1][:rem])
            s, e = two_sum_merge(top, (partial_s[groups * 32 :], partial_e[groups * 32 :]))
            t[0][:rem], t[1][:rem] = s, e
            return t

        if reverse_chunks:
            t = merge_remainder(t)
        for k in reversed(range(groups)) if reverse_chunks else range(groups):
            t = two_sum_merge(t, (grid_s[k], grid_e[k]))
        if not reverse_chunks:
            t = merge_remainder(t)
        # The butterfly. A lane with no chunks holds (+0, +0), left out as above.
        lane = [(t[0][l], t[1][l]) for l in range(lanes)] + [None] * (32 - lanes)
        for offset in butterfly:
            nxt = []
            for l in range(32):
                a, b = lane[l], lane[l ^ offset]
                nxt.append(b if a is None else a if b is None else two_sum_merge(a, b))
            lane = nxt
        s, e = lane[0]
        # The value: sum + error, or the sum alone when it is inf or NaN.
        total = np.where(np.isfinite(s), s + e, s).astype(f32)
        if old is not None:
            total = np.asarray(old, dtype=f32) + total
    return total


def scalar_reference(dy, old=None):
    """The module docstring's order transcribed literally, one np.float32 scalar at a time, every
    merge included (those of threads without rows and of lanes without chunks too): slow, so
    for small shapes, to check emulate."""
    f = np.float32
    x = dy.numpy() if isinstance(dy, torch.Tensor) else dy
    rows, cols = x.shape
    chunks = -(-rows // CHUNK_ROWS)
    out = np.empty(cols, dtype=f)

    def add(a, v):
        s, e = a
        t = f(s + v)
        vp = f(t - s)
        d1 = f(t - vp)
        d2 = f(s - d1)
        d3 = f(v - vp)
        d = f(d2 + d3)
        return t, f(e + d)

    def merge(a, b):
        errs = f(a[1] + b[1])
        s, e = add((a[0], f(0.0)), b[0])
        return s, f(e + errs)

    with np.errstate(all="ignore"):
        for j in range(cols):
            partial = []
            for c in range(chunks):
                end = min(CHUNK_ROWS * c + CHUNK_ROWS, rows)
                threads = []
                for y in range(8):
                    a = (f(0.0), f(0.0))
                    r = CHUNK_ROWS * c + y
                    while r < end:
                        a = add(a, f(x[r, j]))
                        r += 8
                    threads.append(a)
                a = threads[0]
                for y in range(1, 8):
                    a = merge(a, threads[y])
                partial.append(a)
            t = [(f(0.0), f(0.0))] * 32
            for l in range(32):
                c = l
                while c < chunks:
                    t[l] = merge(t[l], partial[c])
                    c += 32
            for offset in (16, 8, 4, 2, 1):
                t = [merge(t[l], t[l ^ offset]) for l in range(32)]
            s, e = t[0]
            value = f(s + e) if np.isfinite(s) else s
            out[j] = value if old is None else f(f(old[j]) + value)
    return out


def cpu_dy(rows, cols, seed):
    """dy as FP16 on the CPU from a seeded generator, the same on every machine: N(0, 1) times a
    power of 2 from 2^-6 to 2^6, clamped to +-60,000, so that the FP32 sums round differently in
    different orders. Made in blocks of rows to bound the temporaries."""
    g = torch.Generator().manual_seed(seed)
    dy = torch.empty((rows, cols), dtype=torch.float16)
    step = max(1, (1 << 22) // cols)
    for r0 in range(0, rows, step):
        n = min(step, rows - r0)
        x = torch.randn((n, cols), generator=g)
        e = torch.randint(-6, 7, (n, cols), generator=g).float()
        dy[r0 : r0 + n] = (x * torch.exp2(e)).clamp(-60000, 60000).half()
    return dy


# -- Running Flint -----------------------------------------------------------


def fill_value(fill, n):
    """What `tensor[:n].copy_()` or fill_ takes for a FILLS entry."""
    if isinstance(fill, str):
        return torch.randint(-(2**31), 2**31, (n,), dtype=torch.int32, device="cuda")
    return fill


def set_words(tensor, fill):
    if isinstance(fill, str):
        tensor.copy_(fill_value(fill, tensor.numel()))
    else:
        tensor.fill_(fill)


def run_flint(flint, dy, old=None, accumulate=False, offset=4, workspace_fill=NAN32, workspace_extra=0,
              dbias_fill=NAN32, stream=None, check_input=None, with_workspace=False):
    """Runs the kernels on dy (rows x cols FP16 on the GPU) into a dbias inside a buffer
    pre-filled with a sentinel, `offset` floats in (a multiple of 4 keeps it 16-byte aligned),
    and a workspace likewise (its first workspace_bytes filled with workspace_fill, workspace_extra
    more bytes of room after it). With accumulate, dbias starts as `old`, otherwise as
    dbias_fill (an int32 bit pattern, or "garbage"). Checks that everything around dbias and the
    workspace's used part is untouched and that dy is unchanged, and returns dbias (cols FP32, a
    copy), and with with_workspace also the workspace's used part as FP32 (chunks, 2, cols): each
    chunk's row of partial sums and row of errors."""
    rows, cols = dy.shape
    needed = flint.bias_gradient_workspace(rows, cols)
    assert needed == -(-rows // CHUNK_ROWS) * cols * 8
    if check_input is None:
        check_input = dy.numel() <= CHECK_INPUT_ELEMENTS
    buf = torch.full((offset + cols + GUARD_AFTER,), SENTINEL, dtype=torch.int32, device="cuda")
    dbias = buf[offset : offset + cols]
    if accumulate:
        dbias.view(torch.float32).copy_(old)
    else:
        set_words(dbias, dbias_fill)
    words = needed // 4
    wbuf = torch.full((offset + words + workspace_extra // 4 + GUARD_AFTER,), SENTINEL,
                      dtype=torch.int32, device="cuda")
    workspace = wbuf[offset : offset + words + workspace_extra // 4]
    set_words(workspace[:words], workspace_fill)
    before = dy.clone() if check_input else None
    torch.cuda.synchronize()

    flint.bias_gradient(dbias, dy, rows, cols, accumulate=accumulate, workspace=workspace,
                        workspace_bytes=needed + workspace_extra, stream=stream)
    torch.cuda.synchronize()

    sentinel = torch.full((1,), SENTINEL, dtype=torch.int32, device="cuda")
    assert all_equal(buf[:offset], sentinel), "flint_bias_gradient wrote before dbias"
    assert all_equal(buf[offset + cols :], sentinel), "flint_bias_gradient wrote past dbias"
    assert all_equal(wbuf[:offset], sentinel), "flint_bias_gradient wrote before the workspace"
    assert all_equal(wbuf[offset + words :], sentinel), "flint_bias_gradient wrote past its workspace"
    if check_input:
        assert torch.equal(dy.view(torch.int16), before.view(torch.int16)), "dy changed"
    if with_workspace:
        partials = workspace[:words].view(torch.float32).reshape(-1, 2, cols).clone()
        return dbias.view(torch.float32).clone(), partials
    return dbias.view(torch.float32).clone()


def with_zero_columns(dy, every):
    dy[:, ::every] = 0
    return dy


# -- The emulation itself ------------------------------------------------------

SCALAR_SHAPES = [(1, 8), (7, 16), (300, 16), (513, 8), (8453, 8), (16389, 8)]


@pytest.mark.parametrize("rows, cols", shape_params(SCALAR_SHAPES))
def test_the_emulation_matches_a_literal_scalar_transcription(rows, cols):
    dy = cpu_dy(rows, cols, 100 + rows)
    old = torch.randn(cols, generator=torch.Generator().manual_seed(rows)) * 100

    assert_same_bits(torch.from_numpy(emulate(dy)), torch.from_numpy(scalar_reference(dy)), "overwrite")
    assert_same_bits(
        torch.from_numpy(emulate(dy, old.numpy())), torch.from_numpy(scalar_reference(dy, old.numpy())),
        "accumulate",
    )


def test_the_emulation_matches_the_scalar_transcription_with_non_finite_values():
    # Every kind of inf and NaN, in columns of several chunks and lanes: the emulation leaves
    # (0, 0) pairs out of its merges and the transcription merges them, which may differ only
    # in the errors of non-finite sums, which the value doesn't use.
    rows, cols = 600, 16
    dy = cpu_dy(rows, cols, 5)
    x = dy.numpy().copy()
    x[5, 0] = np.inf
    x[300, 1] = -np.inf
    x[10, 2], x[560, 2] = np.inf, -np.inf
    x[599, 3] = np.nan
    x[0, 4], x[3, 4], x[7, 4] = np.inf, np.inf, np.inf
    x[100, 5], x[270, 5] = -np.inf, -np.inf
    x[511, 6] = np.inf

    got = torch.from_numpy(emulate(x))
    assert_same_bits(got, torch.from_numpy(scalar_reference(x)), "non-finite columns")
    assert got[0].item() == math.inf and got[1].item() == -math.inf and math.isnan(got[2].item())
    assert math.isnan(got[3].item()) and got[4].item() == math.inf and got[5].item() == -math.inf
    assert got[6].item() == math.inf


def test_the_emulation_stays_in_float32():
    dy = cpu_dy(600, 24, 7)

    assert emulate(dy).dtype == np.float32
    assert emulate(dy, np.zeros(24, dtype=np.float32)).dtype == np.float32
    assert emulate_plain(dy).dtype == np.float32


def differing_columns(a, b):
    return int((canonical(torch.from_numpy(a)) != canonical(torch.from_numpy(b))).sum())


def test_the_bit_exact_tests_tell_plain_sums_from_compensated_ones():
    # What the bit-exact comparisons can see on an ordinary input. Compensated sums are nearly
    # correctly rounded, so a different order of the same compensated merges (the threads' merges
    # backwards, a lane's chunks backwards, the butterfly's offsets reversed) changes the rounded
    # sum s but not s + e, and gives the same bits in (nearly) every column: the totals don't pin
    # those orders. Pass 1's partial sums do (test_the_partials_tell_the_order_of_the_thread_
    # merges_apart), and crafted inputs pin pass 2's (test_crafted_inputs_tell_the_orders_of_pass_
    # 2_apart). What the bits pin here is the compensation: plain FP32 sums, in the same order, or
    # torch's own sum, differ in most columns. The counts are of the 768 columns of 16384
    # CPU-generated rows (FP16 magnitudes 2^-6 to 2^6 around N(0, 1)), printed for the record.
    dy = cpu_dy(16384, 768, 3)
    kernel = emulate(dy)
    variants = {
        "plain FP32 sums (the same order)": emulate_plain(dy),
        "torch's CPU FP32 sum": dy.float().sum(0).numpy(),
        "the 7 thread merges in reverse": emulate(dy, y_order=(7, 6, 5, 4, 3, 2, 1)),
        "each lane's chunks in reverse": emulate(dy, reverse_chunks=True),
        "the butterfly's offsets 1, 2, 4, 8, 16": emulate(dy, butterfly=(1, 2, 4, 8, 16)),
        "the exact sums, rounded": dy.double().sum(0).float().numpy(),
    }
    counts = {name: differing_columns(kernel, other) for name, other in variants.items()}
    for name, count in counts.items():
        print(f"ORDER {name}: {count} of 768 columns differ")

    assert counts["plain FP32 sums (the same order)"] > 100
    assert counts["torch's CPU FP32 sum"] > 100
    # The compensated result is the correctly rounded sum, give or take a rare tie.
    assert counts["the exact sums, rounded"] <= 2


# -- The bit-exact emulation ---------------------------------------------------

# One row, rows not a multiple of 8 or 256, whole and partial chunks, more than 32 chunks (8193 has
# 33, 33 * 256 + 5 an uneven number per lane, 16384 has 64), and cols 8, a partial last tile (264,
# 520) and exactly a tile (256).
EXACT_SHAPES = shrunk([
    (1, 8), (1, 264), (7, 8), (8, 24), (9, 16), (255, 8), (256, 24), (257, 264), (300, 520), (511, 256),
    (512, 8), (513, 8), (513, 264), (2049, 8), (8192, 16), (8193, 8), (8193, 520), (33 * 256 + 5, 8),
    (33 * 256 + 5, 264), (40 * 256 + 7, 8), (16384, 8), (16384, 264), (16384, 768),
])  # fmt: skip


@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(7, 8), (257, 264), (513, 520), (8193, 8), (33 * 256 + 5, 264), (16384, 768)])))
def test_the_workspaces_partials_equal_the_emulations_bit_for_bit(flint, rows, cols):
    # The total is nearly always the correctly rounded sum, whatever the order (see
    # test_the_bit_exact_tests_tell_plain_sums_from_compensated_ones), so it can't show the order
    # of the additions. Pass 1's rounded partial sums can: they pin the order of each thread's
    # rows and of the 8 threads' merges, and the workspace's layout. Pass 2's order, a lane's
    # chunks and the butterfly, is pinned by the crafted inputs below.
    dy = cpu_dy(rows, cols, 14)

    _, partials = run_flint(flint, dy.cuda(), with_workspace=True)

    _, sums, errors = emulate(dy, with_partials=True)
    assert_same_bits(partials[:, 0].reshape(-1), sums.reshape(-1), f"partial sums {rows}x{cols}")
    assert_same_bits(partials[:, 1].reshape(-1), errors.reshape(-1), f"partial errors {rows}x{cols}")


def test_the_partials_tell_the_order_of_the_thread_merges_apart():
    # What the test above can see: merging the 8 threads in reverse changes many partial sums.
    dy = cpu_dy(4096, 768, 15)

    _, sums, _ = emulate(dy, with_partials=True)
    _, reversed_sums, _ = emulate(dy, y_order=tuple(reversed(Y_ORDER)), with_partials=True)

    differ = int((canonical(torch.from_numpy(sums)) != canonical(torch.from_numpy(reversed_sums))).sum())
    print(f"ORDER partial sums differing with the thread merges reversed: {differ} of {sums.size}")
    assert differ > 100


# -- Pass 2's order, on crafted inputs -------------------------------------------
# On ordinary inputs the compensated total is the correctly rounded sum whatever the order of pass
# 2's merges, so the bits can't show that order. They can when the errors' own plain FP32 sums
# round: when chunks' sums near 2^24 add to values that round (TwoSum errors of 1 or 2) while other
# errors are near 2^-24, so an error sum spans more than 24 bits, and the chunks' sums cancel, so
# the total is made of the errors. The total is then not always correctly rounded, and which way
# it misses depends on the order. LayerNorm's backward pass reuses these inputs for dgamma, whose
# second pass is this one.

# 128 chunks, 4 to a lane.
CRAFTED_SHAPE = (128 * CHUNK_ROWS, 32)
CRAFTED_SEED = 31


def crafted_cancellation(rows, cols, seed):
    """dy (rows x cols FP16, on the CPU, from a seeded torch generator, so the same on every
    machine) whose column sums show the order of pass 2. Each column pairs up its chunks at
    random, and the two chunks of a pair hold opposite values, so the chunks' sums cancel exactly:
    every row holds one big value from 60,000 to 65,504 (a chunk's sum is near 2^24, and two
    chunks' sums of one sign add to a value that rounds), but for one row of an odd integer below
    1,024 (so the sums are odd). Then every chunk gets one value of FP16's subnormal range in a
    random row, which pass 1 loses into the error (errors near 2^-24). A chunk left without a
    partner holds only that value. Each column's draws follow the last's, so the first columns of
    a wider input are those of a narrower one."""
    g = torch.Generator().manual_seed(seed)
    chunks = -(-rows // CHUNK_ROWS)

    def draw(low, high):
        return int(torch.randint(low, high, (1,), generator=g))

    dy = torch.zeros(rows, cols, dtype=torch.float16)
    for j in range(cols):
        order = torch.randperm(chunks, generator=g).tolist()
        for k in range(0, chunks - 1, 2):
            a, b = order[k], order[k + 1]
            n = min(CHUNK_ROWS, rows - a * CHUNK_ROWS, rows - b * CHUNK_ROWS)
            big = float(60000 + draw(0, 5505))
            odd = float(2 * draw(0, 512) + 1)
            sign = 1.0 if draw(0, 2) else -1.0
            for c, s in ((a, sign), (b, -sign)):
                dy[c * CHUNK_ROWS : c * CHUNK_ROWS + n, j] = s * big
                dy[c * CHUNK_ROWS + draw(0, n), j] = s * odd
        for c in range(chunks):
            n = min(CHUNK_ROWS, rows - c * CHUNK_ROWS)
            dy[c * CHUNK_ROWS + draw(0, n), j] = draw(1, 1024) * 2.0**-24 * (1.0 if draw(0, 2) else -1.0)
    return dy


def pass2_with_lanes(partial_s, partial_e, lanes):
    """pass2 with lane l merging the chunks lanes[l] (a list of 32 lists) in that order, then the
    kernel's butterfly: to show what the bits can tell apart."""
    with np.errstate(all="ignore"):
        lane = []
        for chunks in lanes:
            a = None
            for c in chunks:
                b = (partial_s[c], partial_e[c])
                a = b if a is None else two_sum_merge(a, b)
            lane.append(a)
        for offset in BUTTERFLY:
            nxt = []
            for l in range(32):
                a, b = lane[l], lane[l ^ offset]
                nxt.append(b if a is None else a if b is None else two_sum_merge(a, b))
            lane = nxt
        s, e = lane[0]
        return np.where(np.isfinite(s), s + e, s).astype(np.float32)


def pass2_orders(partial_s, partial_e):
    """The kernel's pass 2 and three other orders of it, by name."""
    chunks = partial_s.shape[0]
    return {
        "the kernel's": pass2(partial_s, partial_e),
        "the kernel's lanes, merged by pass2_with_lanes": pass2_with_lanes(
            partial_s, partial_e, [list(range(l, chunks, 32)) for l in range(32)]),
        "each lane's chunks in reverse": pass2(partial_s, partial_e, reverse_chunks=True),
        "the butterfly's offsets 1, 2, 4, 8, 16": pass2(partial_s, partial_e, butterfly=(1, 2, 4, 8, 16)),
        "lane l merging chunks in a contiguous run": pass2_with_lanes(
            partial_s, partial_e, [[c for c in range(chunks) if c * 32 // chunks == l] for l in range(32)]),
    }


def test_crafted_inputs_tell_the_orders_of_pass_2_apart():
    # On the crafted input, each other order of pass 2 changes the total in several columns, so the
    # bit-exact test below pins pass 2's order: a lane's chunks (l, l + 32, ... in order) and the
    # butterfly's offsets (16 down to 1). The counts are printed for the record.
    dy = crafted_cancellation(*CRAFTED_SHAPE, CRAFTED_SEED)
    _, sums, errors = emulate(dy, with_partials=True)
    orders = pass2_orders(sums, errors)
    kernel = orders.pop("the kernel's")
    rounded = differing_columns(kernel, dy.double().sum(0).float().numpy())
    print(f"ORDER crafted: {rounded} of {CRAFTED_SHAPE[1]} columns not correctly rounded")
    assert differing_columns(kernel, orders.pop("the kernel's lanes, merged by pass2_with_lanes")) == 0
    for name, other in orders.items():
        count = differing_columns(kernel, other)
        print(f"ORDER crafted {name}: {count} of {CRAFTED_SHAPE[1]} columns differ")
        assert count >= 4, name


def test_dbias_equals_the_emulation_on_the_crafted_inputs(flint):
    # Under the sanitizer, the first 8 columns of the same input.
    rows, cols = CRAFTED_SHAPE[0], 8 if SANITIZER else CRAFTED_SHAPE[1]
    dy = crafted_cancellation(rows, cols, CRAFTED_SEED)

    got = run_flint(flint, dy.cuda())

    assert_same_bits(got, emulate(dy), f"dbias crafted {rows}x{cols}")


@pytest.mark.parametrize("rows, cols", shape_params(EXACT_SHAPES))
def test_dbias_equals_the_emulation_bit_for_bit(flint, rows, cols):
    dy = cpu_dy(rows, cols, 11)

    got = run_flint(flint, dy.cuda())

    assert_same_bits(got, emulate(dy), f"dbias {rows}x{cols}")


@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(1, 8), (300, 264), (513, 520), (8193, 8), (33 * 256 + 5, 264), (16384, 264)])))
def test_accumulate_equals_the_emulation_bit_for_bit(flint, rows, cols):
    dy = cpu_dy(rows, cols, 12)
    old = torch.randn(cols, generator=torch.Generator().manual_seed(13)) * 1000

    got = run_flint(flint, dy.cuda(), old=old.cuda(), accumulate=True)

    assert_same_bits(got, emulate(dy, old.numpy()), f"accumulated dbias {rows}x{cols}")


@pytest.mark.parametrize("seed", [20, 21, 22])
def test_other_inputs_equal_the_emulation_too(flint, seed):
    rows, cols = (300, 264) if SANITIZER else (8453, 520)
    dy = cpu_dy(rows, cols, seed)

    got = run_flint(flint, dy.cuda())

    assert_same_bits(got, emulate(dy), f"dbias seed {seed}")


# -- The correctness gate --------------------------------------------------------

GATE_KINDS = ["normal", "shifted", "gradient"]
# GPT-2's micro-batch against its three bias widths (the qkv, mlp-in and attention-out biases and
# so on), a 1,024-row batch, and odd shapes: every rows and every cols of the list appear at
# least once.
GATE_SHAPES = shrunk([
    (16384, 768), (16384, 2304), (16384, 3072), (1024, 768),
    (1, 8), (7, 24), (255, 264), (256, 520), (257, 8), (300, 264), (513, 520), (513, 24), (7, 520),
    (256, 24), (1, 520), (300, 8),
])  # fmt: skip


def make_gate_input(kind, rows, cols, g):
    """N(0, 1), N(0.3, 1), or gradient-like N(0, 1) * 1e-3: column sums that stay well inside
    FP16's range (16,384 rows of mean 0.3 sum to about 4,900), so the peer's FP16 result is
    finite."""
    if kind == "normal":
        return normal((rows, cols), 1.0, g)
    if kind == "shifted":
        return (torch.randn((rows, cols), generator=g, device="cuda") + 0.3).half()
    if kind == "gradient":
        return normal((rows, cols), 1e-3, g)
    raise ValueError(kind)


def autocast_peer(dy):
    """PyTorch's autocast path for a linear layer's bias gradient: nn.Linear(k, cols) under
    torch.autocast to FP16, y.backward(dy), the bias's grad. Autocast casts the bias to FP16, so
    the gradient is FP16 rounded (then FP32, the parameter's type)."""
    rows, cols = dy.shape
    lin = torch.nn.Linear(8, cols, device="cuda")
    x = torch.zeros((rows, 8), device="cuda")
    with torch.enable_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            y = lin(x)
        y.backward(dy)
    grad = lin.bias.grad.clone()
    assert grad.dtype == torch.float32
    return grad


def fp64_sums(dy):
    return dy.sum(0, dtype=torch.float64)


def worst(errors_per_seed):
    return (max(e.rel_l2 for e in errors_per_seed), max(e.max_abs for e in errors_per_seed))


def gate_line(label, flint_errors, peer_errors, bounds):
    fr, fm = worst(flint_errors)
    pr, pm = worst(peer_errors)
    print(
        f"GATE {label} | flint rel_l2 {fr:.3g} max_abs {fm:.3g} | peer rel_l2 {pr:.3g} "
        f"max_abs {pm:.3g} | bound rel_l2 {bounds.rel_l2:.3g} max_abs {bounds.max_abs:.3g}"
    )


@pytest.mark.parametrize("kind", GATE_KINDS)
@pytest.mark.parametrize("rows, cols", shape_params(GATE_SHAPES))
def test_passes_the_gate_against_autocast(flint, kind, rows, cols):
    flint_errors, peer_errors = [], []
    for seed in gate.SEEDS:
        dy = make_gate_input(kind, rows, cols, generator(seed))
        ref = fp64_sums(dy)
        got = run_flint(flint, dy, offset=4 * (1 + seed))
        peer = autocast_peer(dy)
        flint_errors.append(gate.errors(got, ref))
        peer_errors.append(gate.errors(peer, ref))
        del dy, ref, got, peer

    bounds = gate.check_gate(gate.GRADIENT_FACTOR, flint_errors, peer_errors, output="fp32")
    gate_line(f"autocast {kind} {shape_id((rows, cols))}", flint_errors, peer_errors, bounds)


# -- A tighter bound: at most twice FP32 PyTorch's own error -------------------------

TIGHT_KINDS = ["normal", "shifted", "gradient", "outliers", "range"]
TIGHT_SHAPES = shrunk([
    (16384, 768), (16384, 2304), (16384, 3072), (1024, 768),
    (1, 8), (7, 264), (257, 24), (300, 520), (513, 264), (8453, 8), (2049, 520),
])  # fmt: skip


def make_tight_input(kind, rows, cols, g):
    shape = (rows, cols)
    if kind in ("normal", "shifted"):
        return (torch.randn(shape, generator=g, device="cuda") + (0.3 if kind == "shifted" else 0.0)).half()
    if kind == "gradient":
        return normal(shape, 1e-3, g)
    if kind == "outliers":
        # Std 3,000 in 1% of the columns (and in one at least, for the narrow shapes), 1 elsewhere.
        hit = torch.rand(cols, generator=g, device="cuda") < 0.01
        hit[cols // 3] = True
        std = torch.where(hit, 3000.0, 1.0)
        return (torch.randn(shape, generator=g, device="cuda") * std).half()
    if kind == "range":
        # Uniform in +-60,000, with 10% subnormals (+-1 to 1023 times 2^-24), all finite.
        big = (torch.rand(shape, generator=g, device="cuda") * 2 - 1) * 60000
        sign = torch.randint(0, 2, shape, generator=g, device="cuda") * 2.0 - 1.0
        sub = torch.randint(1, 1024, shape, generator=g, device="cuda").float() * 2.0**-24 * sign
        use_sub = torch.rand(shape, generator=g, device="cuda") < 0.1
        out = torch.where(use_sub, sub, big).half()
        assert bool(torch.isfinite(out).all())
        return out
    raise ValueError(kind)


@pytest.mark.parametrize("kind", TIGHT_KINDS)
@pytest.mark.parametrize("rows, cols", shape_params(TIGHT_SHAPES))
def test_error_is_at_most_twice_fp32_pytorchs(flint, kind, rows, cols):
    flint_errors, torch_errors = [], []
    for seed in gate.SEEDS:
        dy = make_tight_input(kind, rows, cols, generator(seed))
        ref = fp64_sums(dy)
        got = run_flint(flint, dy, offset=4 * (1 + seed))
        mine = dy.float().sum(0)
        flint_errors.append(gate.errors(got, ref))
        torch_errors.append(gate.errors(mine, ref))
        del dy, ref, got, mine

    bounds = gate.check_gate(2.0, flint_errors, torch_errors, output="fp32")
    gate_line(f"tight {kind} {shape_id((rows, cols))}", flint_errors, torch_errors, bounds)


# -- accumulate ----------------------------------------------------------------------


def olds(cols, seed):
    """cols FP32 olds: random, with +-0, a subnormal, inf, -inf, NaN and large values among them,
    the first few at the columns that a with_zero_columns(.., 5) dy sums to +0."""
    old = torch.randn(cols, generator=torch.Generator().manual_seed(seed)) * 100
    special = [0.0, -0.0, 1e-40, -1e-40, math.inf, -math.inf, math.nan, 3.0e38, -3.0e38, 1e-30, 1.0]
    for j in range(1, cols, 2):
        old[j] = special[(j // 2) % len(special)]
    for i, value in enumerate(special[:6]):
        if 5 * i < cols:
            old[5 * i] = value
    return old


def overwrite_then_add(flint, dy, old):
    """The overwrite result and old + it, in FP32 on the GPU, as the expected accumulate."""
    total = run_flint(flint, dy)
    return total, old.cuda() + total


ACCUMULATE_SHAPES = shrunk([(1, 8), (7, 264), (300, 264), (513, 520), (8453, 24)])


@pytest.mark.parametrize("rows, cols", shape_params(ACCUMULATE_SHAPES))
def test_accumulate_adds_the_overwrite_result_to_old_with_one_add(flint, rows, cols):
    dy = with_zero_columns(normal((rows, cols), 1.0, generator(30)), 5)
    old = olds(cols, 31)
    total, expected = overwrite_then_add(flint, dy, old)

    got = run_flint(flint, dy, old=old.cuda(), accumulate=True, offset=8)

    assert_same_bits(got, expected, "accumulate against old + overwrite")
    assert_same_bits(got, emulate(dy.cpu(), old.numpy()), "accumulate against the emulation")


def test_accumulate_onto_special_olds_gives_the_fp32_sum_of_each(flint):
    # Columns j % 4 == 0 sum to +0 (their dy is 0), and old goes through unchanged but for -0,
    # which -0 + +0 makes +0; an inf or NaN old stays, and a finite total doesn't disturb it.
    rows, cols = 513, 80
    dy = with_zero_columns(normal((rows, cols), 1.0, generator(32)), 4)
    values = [0.0, -0.0, 1e-40, -1e-40, math.inf, -math.inf, math.nan, 3.0e38, -3.0e38, 3.4028234e38]
    old = torch.tensor([values[j % len(values)] for j in range(cols)])
    total = run_flint(flint, dy)

    got = run_flint(flint, dy, old=old.cuda(), accumulate=True)

    assert_same_bits(got, old.cuda() + total, "old + total")
    kinds = torch.arange(cols, device="cuda") % len(values)
    assert bool(torch.isinf(got)[kinds == 4].all()) and bool((got[kinds == 4] > 0).all())
    assert bool((got[kinds == 5] < 0).all()) and bool(torch.isinf(got)[kinds == 5].all())
    assert bool(torch.isnan(got)[kinds == 6].all())
    zero = (torch.arange(cols, device="cuda") % 4 == 0) & (kinds < 4)
    # old = 0, -0 give +0; the subnormals, with a zero total, come through exactly.
    assert bool((got[zero & (kinds < 2)] == 0).all()) and not bool(torch.signbit(got[zero & (kinds < 2)]).any())
    assert torch.equal(got[zero & (kinds >= 2)], old.cuda()[zero & (kinds >= 2)])


@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(300, 264), (513, 24), (8453, 8)])))
def test_micro_batches_equal_the_sequential_fp32_adds(flint, rows, cols):
    # Five calls in a row on one stream, sharing one workspace, with no sync between them: the
    # first overwrites, the rest accumulate (and the same again from zeros by accumulating all).
    batches = [cpu_dy(rows, cols, 40 + i) for i in range(5)]
    needed = flint.bias_gradient_workspace(rows, cols)
    workspace = torch.full((needed // 4,), NAN32, dtype=torch.int32, device="cuda")
    gpu = [b.cuda() for b in batches]

    from_overwrite = torch.full((cols,), math.nan, device="cuda")
    from_zeros = torch.zeros(cols, device="cuda")
    for i, dy in enumerate(gpu):
        flint.bias_gradient(from_overwrite, dy, rows, cols, accumulate=i > 0, workspace=workspace)
        flint.bias_gradient(from_zeros, dy, rows, cols, accumulate=True, workspace=workspace)
    torch.cuda.synchronize()

    expected = emulate(batches[0])
    expected_zeros = np.zeros(cols, dtype=np.float32) + expected
    for dy in batches[1:]:
        total = emulate(dy)
        expected = expected + total
        expected_zeros = expected_zeros + total
    assert expected.dtype == np.float32
    assert_same_bits(from_overwrite, expected, "overwrite then accumulates")
    assert_same_bits(from_zeros, expected_zeros, "accumulates from zeros")
    assert_same_bits(from_zeros, expected, "zeros + the first total is the total")


@pytest.mark.parametrize("fill", list(FILLS))
def test_overwrite_ignores_dbiass_old_content(flint, fill):
    rows, cols = 513, 264
    dy = cpu_dy(rows, cols, 50)

    got = run_flint(flint, dy.cuda(), dbias_fill=FILLS[fill])

    assert_same_bits(got, emulate(dy), f"dbias over {fill}")
    assert not bool(torch.isnan(got).any())


# -- inf and NaN ------------------------------------------------------------------------

NF_ROWS, NF_COLS = 600, 520
# (column, [(row, FP16 bits)]) in different tiles (256 columns each, the last of 8), chunks (256 rows
# each, the last of 88) and y lanes (row % 8, since chunks start at multiples of 8).
NON_FINITE = [
    (3, [(5, HALF_INF)]),                                            # tile 0, chunk 0, y 5
    (100, [(300, HALF_NEG_INF)]),                                    # chunk 1, y 4
    (129, [(10, HALF_INF), (560, HALF_NEG_INF)]),                    # +inf then -inf: NaN
    (255, [(599, HALF_NAN)]),                                        # last of tile 0, chunk 2, y 7
    (256, [(0, HALF_INF), (3, HALF_NEG_INF)]),                       # first of tile 1, one chunk: NaN
    (300, [(256, HALF_INF), (511, HALF_INF)]),                       # +inf twice: +inf
    (511, [(7, HALF_NEG_INF), (270, HALF_NEG_INF), (580, HALF_NEG_INF)]),  # -inf thrice: -inf
    (512, [(100, HALF_INF), (0, HALF_NAN)]),                         # tile 2: NaN with an inf
    (519, [(598, HALF_INF), (599, HALF_NEG_INF)]),                   # NaN in the last chunk
    (400, [(1, 0xFE01)]),                                            # a signalling NaN with a sign
]


def expected_class(entries):
    bits = {b for _, b in entries}
    nan = bool(bits & {HALF_NAN, 0xFE01}) or (HALF_INF in bits and HALF_NEG_INF in bits)
    if nan:
        return "nan"
    return "inf" if HALF_INF in bits else "-inf"


def non_finite_dy(seed):
    dy = normal((NF_ROWS, NF_COLS), 1.0, generator(seed))
    clean = dy.clone()
    view = dy.view(torch.int16)
    for col, entries in NON_FINITE:
        for row, bits in entries:
            view[row, col] = as_int16(bits)
    return dy, clean


@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
def test_inf_and_nan_reach_their_column_and_no_other(flint, accumulate):
    dy, clean = non_finite_dy(60)
    old = (torch.randn(NF_COLS, generator=torch.Generator().manual_seed(61)) * 100).cuda()
    expected_clean = run_flint(flint, clean, old=old, accumulate=accumulate)

    got = run_flint(flint, dy, old=old, accumulate=accumulate)

    hit = [col for col, _ in NON_FINITE]
    for col, entries in NON_FINITE:
        kind = expected_class(entries)
        value = got[col].item()
        if kind == "nan":
            assert math.isnan(value), (col, value)
        else:
            assert value == (math.inf if kind == "inf" else -math.inf), (col, kind, value)
    other = torch.ones(NF_COLS, dtype=torch.bool)
    other[hit] = False
    assert_same_bits(got.cpu()[other], expected_clean.cpu()[other], "the other columns")
    assert_same_bits(got, emulate(dy.cpu(), old.cpu().numpy() if accumulate else None), "against the emulation")


@pytest.mark.parametrize("bits, expected", [(HALF_INF, math.inf), (HALF_NEG_INF, -math.inf), (HALF_NAN, math.nan)],
                         ids=["inf", "-inf", "nan"])
@pytest.mark.parametrize("row", [0, 7, 8, 255, 256, 299])
def test_one_non_finite_value_makes_its_column_that_value(flint, bits, expected, row):
    rows, cols = 300, 264
    dy = normal((rows, cols), 1.0, generator(62))
    clean = run_flint(flint, dy)
    dy.view(torch.int16)[row, 263] = as_int16(bits)

    got = run_flint(flint, dy)

    value = got[263].item()
    assert math.isnan(value) if math.isnan(expected) else value == expected
    assert_same_bits(got[:263], clean[:263], "the other columns")


# -- What it writes and reads ----------------------------------------------------------------


@pytest.mark.parametrize("offset", [4, 8, 12, 60])
def test_nothing_is_written_around_dbias_or_the_workspace(flint, offset):
    # run_flint checks the guard bands, the workspace's room and dy; the result is right too.
    dy = cpu_dy(513, 264, 70)

    got = run_flint(flint, dy.cuda(), offset=offset, workspace_extra=4096)

    assert_same_bits(got, emulate(dy), "dbias")


@pytest.mark.parametrize("fill", list(FILLS))
def test_the_workspaces_old_content_changes_nothing(flint, fill):
    dy = cpu_dy(1000, 264, 71)
    expected = emulate(dy)

    got = run_flint(flint, dy.cuda(), workspace_fill=FILLS[fill])

    assert_same_bits(got, expected, f"dbias with a workspace of {fill}")


@pytest.mark.parametrize("extra", [16, 4096])
def test_the_workspace_may_be_larger_than_needed(flint, extra):
    dy = cpu_dy(700, 24, 72)

    got = run_flint(flint, dy.cuda(), workspace_extra=extra)

    assert_same_bits(got, emulate(dy), "dbias")


@pytest.mark.parametrize("accumulate", [False, True])
def test_cols_0_does_nothing(flint, accumulate):
    dbias = torch.full((32,), 0.5, device="cuda")
    dy = torch.full((5 * 8,), 0.5, dtype=torch.float16, device="cuda")
    workspace = torch.full((64,), 0.5, device="cuda")

    flint.bias_gradient(dbias, dy, 5, 0, accumulate=accumulate, workspace=workspace)
    flint.bias_gradient(None, None, 5, 0, accumulate=accumulate, workspace=None, workspace_bytes=0)
    flint.bias_gradient(None, None, MAX_ROWS, 0, accumulate=accumulate, workspace=None, workspace_bytes=0)
    torch.cuda.synchronize()

    assert all_equal(dbias, 0.5) and all_equal(dy, 0.5) and all_equal(workspace, 0.5)


# -- Determinism and streams -----------------------------------------------------------------


@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(513, 264), (8453, 24), (16384, 768)])))
def test_run_twice_is_bitwise_identical(flint, rows, cols):
    dy = make_tight_input("range", rows, cols, generator(80))
    dy.view(torch.int16)[rows // 2, cols // 2] = as_int16(HALF_NAN)

    first = run_flint(flint, dy, offset=4, workspace_fill=FILLS["nan"], dbias_fill=FILLS["nan"])
    second = run_flint(flint, dy, offset=8, workspace_fill=FILLS["garbage"], dbias_fill=FILLS["garbage"])
    third = run_flint(flint, dy, offset=12, workspace_fill=0, dbias_fill=0)

    assert bool(torch.isnan(first).any())
    for name, other in (("second", second), ("third", third)):
        assert torch.equal(canonical(other), canonical(first)), f"the {name} run differs"


STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
def test_runs_on_the_given_stream(flint, mode):
    rows, cols = 700, 264
    src = cpu_dy(rows, cols, 90)
    expected = emulate(src)
    source = src.cuda()
    dy = torch.full((rows, cols), math.nan, dtype=torch.float16, device="cuda")
    dbias = torch.full((cols,), math.nan, device="cuda")
    workspace = torch.full((flint.bias_gradient_workspace(rows, cols) // 4,), math.nan, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The copy into dy finishes only after the sleep, so dbias is right only if both kernels are
    # queued on the same stream behind it (it is NaN where they ran first). With the current
    # stream the call has no stream argument, and torch's current stream is the side one; with
    # the default stream there is no side one at all.
    if mode == "default-stream":
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        dy.copy_(source)
        flint.bias_gradient(dbias, dy, rows, cols, workspace=workspace)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            dy.copy_(source)
            flint.bias_gradient(dbias, dy, rows, cols, workspace=workspace,
                                stream=stream if mode == "explicit-stream" else None)
        stream.synchronize()

    assert not bool(torch.isnan(dbias).any())
    assert_same_bits(dbias, expected, mode)


def test_accumulate_on_a_side_stream_after_a_slow_kernel(flint):
    rows, cols = 513, 264
    src = cpu_dy(rows, cols, 91)
    old = torch.randn(cols, generator=torch.Generator().manual_seed(92))
    expected = emulate(src, old.numpy())
    dy = torch.full((rows, cols), math.nan, dtype=torch.float16, device="cuda")
    dbias = torch.full((cols,), math.nan, device="cuda")
    workspace = torch.empty(flint.bias_gradient_workspace(rows, cols) // 4, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        dy.copy_(src.cuda())
        dbias.copy_(old.cuda())
        flint.bias_gradient(dbias, dy, rows, cols, accumulate=True, workspace=workspace, stream=stream)
    stream.synchronize()

    assert_same_bits(dbias, expected, "accumulate on a side stream")


# -- The workspace size --------------------------------------------------------------------


def expected_workspace(rows, cols):
    return -(-rows // CHUNK_ROWS) * cols * 8


WORKSPACE_SHAPES = [
    (1, 8), (1, 768), (7, 24), (255, 8), (256, 8), (257, 8), (300, 264), (511, 520), (512, 520),
    (513, 520), (8192, 8), (8193, 8), (16384, 768), (16384, 2304), (16384, 3072), (1024, 768),
    (33 * 256 + 5, 264), (2**32, 64), (1, MAX_COLS), (2, MAX_COLS), (256, MAX_COLS), (257, MAX_COLS),
    (MAX_ROWS, 8), (MAX_ROWS - 1, 8), (MAX_ROWS, MAX_COLS), (MAX_ROWS - 255, MAX_COLS),
    (1, 0), (5, 0), (256, 0), (257, 0), (MAX_ROWS, 0),
]  # fmt: skip


@pytest.mark.parametrize("rows, cols", shape_params(WORKSPACE_SHAPES))
def test_workspace_is_two_rows_of_partials_per_256_rows(flint, rows, cols):
    assert flint.bias_gradient_workspace(rows, cols) == expected_workspace(rows, cols)


def test_the_largest_workspace_fits_in_a_size_t():
    assert expected_workspace(MAX_ROWS, MAX_COLS) < 2**58


# -- The limits ---------------------------------------------------------------------------------


def check_against_the_fp64_sums(got, dy):
    """A loose sanity check next to the bit-exact one: close to the exact sums."""
    ref = dy.sum(0, dtype=torch.float64)
    scale = max(ref.abs().max().item(), 1.0)
    assert (got.double() - ref).abs().max().item() <= 1e-5 * scale


@pytest.mark.skipif(SANITIZER, reason="the widest dy is too big for the sanitizer")
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
def test_the_widest_dy(flint, rows, accumulate):
    cols = MAX_COLS
    dy = cpu_dy(rows, cols, 100 + rows)
    old = torch.randn(cols, generator=torch.Generator().manual_seed(7)) * 10 if accumulate else None

    got = run_flint(flint, dy.cuda(), old=old.cuda() if accumulate else None, accumulate=accumulate)

    expected = emulate(dy, old.numpy() if accumulate else None)
    assert_same_bits(got, expected, f"dbias {rows}x{cols}")
    if rows == 1 and not accumulate:
        # One row: the sums are the values themselves (-0 as +0).
        assert torch.equal(canonical(got), canonical(dy[0].float() + 0.0))


@pytest.mark.skipif(SANITIZER, reason="268 MB of dy is too much for the sanitizer")
@pytest.mark.parametrize("rows", [65_538 * CHUNK_ROWS, 65_537 * CHUNK_ROWS + 7], ids=["65538-chunks", "65538-chunks-partial"])
def test_more_than_65535_chunks(flint, rows):
    # The first pass has a block per chunk in its grid's x, which holds far more than 65,535.
    cols = 8
    dy = cpu_dy(rows, cols, 110)
    gpu = dy.cuda()

    got = run_flint(flint, gpu)

    assert_same_bits(got, emulate(dy), f"dbias {rows}x{cols}")
    check_against_the_fp64_sums(got, gpu)
    del gpu


# -- Argument errors -----------------------------------------------------------------------------

NAMES = ("dbias", "workspace", "dy")
ROWS, COLS = 300, 24
BYTES = {"dbias": COLS * 4, "workspace": 2 * COLS * 8, "dy": ROWS * COLS * 2}
SPACING = 16384
ARENA_SLOT = {name: (i + 2) * SPACING for i, name in enumerate(NAMES)}
FILL = 0x5A
WORKSPACE_NEEDED = BYTES["workspace"]


def new_arena():
    arena = torch.full((8 * SPACING,), FILL, dtype=torch.uint8, device="cuda")
    return arena


def addresses(arena):
    return {name: arena.data_ptr() + ARENA_SLOT[name] for name in NAMES}


def call(flint, a, rows=ROWS, cols=COLS, workspace_bytes=WORKSPACE_NEEDED, accumulate=False):
    flint.bias_gradient(a["dbias"], a["dy"], rows, cols, accumulate=accumulate, workspace=a["workspace"],
                        workspace_bytes=workspace_bytes)


def expect_invalid_argument(flint, reason, arena, a, **kwargs):
    """The call fails with the status and message naming the problem, and nothing in the arena
    changed."""
    before = arena.clone()
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, a, **kwargs)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_bias_gradient", reason):
        assert text in str(error.value)
        assert text in flint.last_error()
    torch.cuda.synchronize()
    assert torch.equal(arena, before), "the call wrote something"


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    arena = new_arena()
    expect_invalid_argument(flint, f"{name} is null", arena, {**addresses(arena), name: None})


@pytest.mark.parametrize("offset", [2, 4, 8])
@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name, offset):
    # 2 bytes is as far off as an FP16 goes, 4 as far as an FP32, and 8 is aligned for 64 bits
    # but not 128.
    arena = new_arena()
    a = addresses(arena)
    a[name] += offset
    expect_invalid_argument(flint, f"{name} is not 16-byte aligned", arena, a)


@pytest.mark.parametrize("name", NAMES)
def test_a_buffer_may_not_run_past_the_address_space(flint, name):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"{name} runs past the end of the address space", arena, {**addresses(arena), name: 2**64 - 16}
    )


def test_pointers_are_checked_in_order(flint):
    # Per buffer: null, then alignment, then the address space; dbias, workspace, dy; and all of
    # that before any overlap.
    arena = new_arena()
    a = addresses(arena)
    expect_invalid_argument(flint, "dbias is not 16-byte aligned", arena, {**a, "dbias": a["dbias"] + 4, "workspace": None})
    expect_invalid_argument(flint, "dbias is null", arena, {**a, "dbias": None, "workspace": a["workspace"] + 4})
    expect_invalid_argument(flint, "workspace is not 16-byte aligned", arena, {**a, "workspace": a["workspace"] + 4, "dy": None})
    expect_invalid_argument(flint, "workspace is null", arena, {**a, "workspace": None, "dy": a["dy"] + 2})
    expect_invalid_argument(flint, "dy is not 16-byte aligned", arena, {**a, "dy": a["dy"] + 2})
    expect_invalid_argument(flint, "dbias runs past the end", arena, {**a, "dbias": 2**64 - 16, "workspace": None})
    expect_invalid_argument(flint, "dy is null", arena, {**a, "workspace": a["dbias"], "dy": None})


@pytest.mark.parametrize(
    "cols", [1, 4, 7, 9, 12, 20, 25, MAX_COLS + 1, MAX_COLS + 8, 2**32, 2**64 - 8, 2**64 - 1], ids=str
)
def test_cols_must_be_a_multiple_of_8_up_to_16776960(flint, cols):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"cols ({cols}) must be a multiple of 8 up to {MAX_COLS}", arena, addresses(arena), cols=cols
    )


@pytest.mark.parametrize("rows", [0, MAX_ROWS + 1, MAX_ROWS + 256, 2**40, 2**63, 2**64 - 1], ids=str)
def test_rows_must_be_from_1_to_549755813632(flint, rows):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"rows ({rows}) must be from 1 to {MAX_ROWS}", arena, addresses(arena), rows=rows
    )


def test_cols_0_still_checks_rows(flint):
    arena = new_arena()
    for rows in (0, MAX_ROWS + 1, 2**64 - 1):
        expect_invalid_argument(flint, f"rows ({rows}) must be", arena, addresses(arena), rows=rows, cols=0)
    expect_invalid_argument(flint, "cols (4) must be", arena, addresses(arena), rows=0, cols=4)


def test_size_limits_are_checked_in_order(flint):
    # cols, then rows, then the workspace's size, then the pointers (null ones included).
    arena = new_arena()
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(flint, "cols (7) must be", arena, nothing, cols=7, rows=0, workspace_bytes=0)
    expect_invalid_argument(flint, "cols (7) must be", arena, nothing, cols=7, rows=2**64 - 1)
    expect_invalid_argument(flint, "rows (0) must be", arena, nothing, rows=0, workspace_bytes=0)
    expect_invalid_argument(flint, f"needs {WORKSPACE_NEEDED} bytes of workspace, given 0", arena, nothing, workspace_bytes=0)
    expect_invalid_argument(flint, "dbias is null", arena, nothing)


@pytest.mark.parametrize("short", [1, 4, 16, 17, WORKSPACE_NEEDED])
def test_a_workspace_may_not_be_too_small(flint, short):
    arena = new_arena()
    given = WORKSPACE_NEEDED - short
    expect_invalid_argument(
        flint, f"needs {WORKSPACE_NEEDED} bytes of workspace, given {given}", arena, addresses(arena), workspace_bytes=given
    )


def test_a_null_workspace_is_a_null_pointer_with_enough_bytes_and_too_small_with_none(flint):
    arena = new_arena()
    a = {**addresses(arena), "workspace": None}
    expect_invalid_argument(flint, f"needs {WORKSPACE_NEEDED} bytes of workspace, given 0", arena, a, workspace_bytes=0)
    expect_invalid_argument(flint, "workspace is null", arena, a, workspace_bytes=WORKSPACE_NEEDED)


@pytest.mark.parametrize(
    "rows, cols", [(MAX_ROWS, 8), (MAX_ROWS, MAX_COLS), (1, MAX_COLS), (2**32, 64)], ids=lambda v: str(v)
)
def test_the_largest_sizes_pass_the_size_checks(flint, rows, cols):
    # The limits themselves aren't errors of size: the call gets on to the workspace's size
    # (and with enough bytes claimed, on to a null dy), even where rows * cols * 2 is 1e15 bytes.
    arena = new_arena()
    needed = expected_workspace(rows, cols)
    expect_invalid_argument(
        flint, f"needs {needed} bytes of workspace, given {WORKSPACE_NEEDED}", arena, addresses(arena), rows=rows, cols=cols
    )
    expect_invalid_argument(
        flint, "dy is null", arena, {**addresses(arena), "dy": None}, rows=rows, cols=cols, workspace_bytes=needed
    )


def overlapping_starts(first, second):
    """Where `second` starts relative to `first`, 16-byte aligned, so that they overlap: at the
    same address, 16 bytes in, 16 bytes before first's end, and (negative) with second ending 16
    bytes into first."""
    a, b = BYTES[first], BYTES[second]
    return {"same-start": 0, "16-in": 16, "16-before-end": a - 16, "ends-16-in": 16 - b}


def touching_starts(first, second):
    """Where `second` starts so that it touches `first` without overlapping it: just after, and
    just before."""
    return {"just-after": BYTES[first], "just-before": -BYTES[second]}


PAIRS = [("dbias", "workspace"), ("dbias", "dy"), ("workspace", "dy")]
OVERLAP_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in PAIRS
    for label, offset in overlapping_starts(a, b).items()
]


@pytest.mark.parametrize("first, second, label, offset", OVERLAP_CASES)
def test_dbias_the_workspace_and_dy_may_not_overlap(flint, first, second, label, offset):
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no
    # harm to anything else.
    arena = new_arena()
    a = addresses(arena)
    a[second] = a[first] + offset
    expect_invalid_argument(flint, f"{first} and {second} overlap", arena, a)


TOUCHING_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in PAIRS
    for label, offset in touching_starts(a, b).items()
]


@pytest.mark.parametrize("first, second, label, offset", TOUCHING_CASES)
def test_buffers_next_to_each_other_are_not_overlapping(flint, first, second, label, offset):
    arena = new_arena()
    start = dict(ARENA_SLOT)
    start[second] = start[first] + offset
    dy_cpu = cpu_dy(ROWS, COLS, 120)
    arena[start["dy"] : start["dy"] + BYTES["dy"]] = dy_cpu.cuda().view(torch.uint8).flatten()
    used = torch.zeros(arena.numel(), dtype=torch.bool, device="cuda")
    for name in ("dbias", "workspace"):
        used[start[name] : start[name] + BYTES[name]] = True
    outside_before = arena[~used].clone()
    base = arena.data_ptr()
    a = {name: base + start[name] for name in NAMES}

    call(flint, a)
    torch.cuda.synchronize()

    assert torch.equal(arena[~used], outside_before), "something outside dbias and the workspace changed"
    dbias = arena[start["dbias"] : start["dbias"] + BYTES["dbias"]].view(torch.float32)
    assert_same_bits(dbias, emulate(dy_cpu), f"{label}")


def test_an_error_leaves_nothing_behind_and_a_valid_call_works_after(flint):
    arena = new_arena()
    a = addresses(arena)
    dy_cpu = cpu_dy(ROWS, COLS, 121)
    arena[ARENA_SLOT["dy"] : ARENA_SLOT["dy"] + BYTES["dy"]] = dy_cpu.cuda().view(torch.uint8).flatten()
    for kwargs in (dict(cols=7), dict(rows=0), dict(rows=MAX_ROWS + 1), dict(cols=2**64 - 1), dict(workspace_bytes=0)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, a, **kwargs)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "dbias": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "dy": a["dy"] + 2})
    torch.cuda.synchronize()
    assert all_equal(arena[ARENA_SLOT["dbias"] : ARENA_SLOT["dbias"] + BYTES["dbias"]], FILL)

    call(flint, a)
    torch.cuda.synchronize()

    dbias = arena[ARENA_SLOT["dbias"] : ARENA_SLOT["dbias"] + BYTES["dbias"]].view(torch.float32)
    assert_same_bits(dbias, emulate(dy_cpu), "after errors")


# -- The workspace function's arguments --------------------------------------------------------------


def workspace_status(flint, rows, cols, size=None):
    """The raw status of flint_bias_gradient_workspace and the size it left (a sentinel if it
    wrote nothing)."""
    out = ctypes.c_size_t(0xDEADBEEF) if size is None else size
    status = flint.lib.flint_bias_gradient_workspace(rows, cols, ctypes.byref(out))
    return status, out.value


@pytest.mark.parametrize("cols", [1, 4, 7, 9, 25, MAX_COLS + 1, MAX_COLS + 8, 2**32, 2**64 - 8, 2**64 - 1], ids=str)
def test_the_workspace_function_checks_cols(flint, cols):
    status, size = workspace_status(flint, 256, cols)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert size == 0xDEADBEEF, "bytes was written on an error"
    for text in ("flint_bias_gradient_workspace", f"cols ({cols}) must be a multiple of 8 up to {MAX_COLS}"):
        assert text in flint.last_error()
    with pytest.raises(flint_lib.FlintError, match="must be a multiple of 8"):
        flint.bias_gradient_workspace(256, cols)


@pytest.mark.parametrize("rows", [0, MAX_ROWS + 1, MAX_ROWS + 256, 2**40, 2**63, 2**64 - 1], ids=str)
@pytest.mark.parametrize("cols", [0, 8, MAX_COLS], ids=str)
def test_the_workspace_function_checks_rows(flint, rows, cols):
    status, size = workspace_status(flint, rows, cols)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert size == 0xDEADBEEF, "bytes was written on an error"
    for text in ("flint_bias_gradient_workspace", f"rows ({rows}) must be from 1 to {MAX_ROWS}"):
        assert text in flint.last_error()


def test_the_workspace_function_checks_cols_before_rows(flint):
    status, _ = workspace_status(flint, 0, 7)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "cols (7) must be" in flint.last_error()


@pytest.mark.parametrize("rows, cols", [(256, 8), (0, 8), (256, 7), (0, 7), (2**64 - 1, 2**64 - 1)], ids=str)
def test_the_workspace_function_needs_a_bytes_pointer(flint, rows, cols):
    # The null check comes first, before any size is looked at.
    status = flint.lib.flint_bias_gradient_workspace(rows, cols, None)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_bias_gradient_workspace: bytes is null" in flint.last_error()


def test_the_workspace_function_accepts_the_limits(flint):
    for rows, cols in ((MAX_ROWS, MAX_COLS), (1, 8), (1, 0), (MAX_ROWS, 0)):
        status, size = workspace_status(flint, rows, cols)

        assert status == flint_lib.OK
        assert size == expected_workspace(rows, cols)
