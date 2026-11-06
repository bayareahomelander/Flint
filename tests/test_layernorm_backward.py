"""flint_layernorm_backward: LayerNorm's backward pass. dy is rows x cols FP16, x rows x cols FP32,
mean and rstd rows FP32 (the forward pass's), gamma cols FP32; it writes dx (rows x cols FP32) and
dgamma and dbeta (cols FP32), each overwritten, or with accumulate_dx / accumulate_params added to
its old value with one FP32 add. It takes a workspace of ceil(rows / 256) * cols * 8 bytes (the bias
gradient's) and runs five kernels with no atomics. Every operation is an IEEE FP32
round-to-nearest one, and the order depends only on rows and cols, so the bits are the same on
every GPU. With xhat(r, j) = (x[r, j] - mean[r]) * rstd[r] (the subtraction rounded, then the
product) and dy converted exactly from FP16:

  - dx, per row r. The row's cols are pieces of 8 consecutive columns. Lane l (of 32) takes pieces
    l, l + 32, l + 64, ... in order, and within a piece columns 0 to 7 in order. For each value:
    g = dy * gamma (rounded); g_sum = g_sum + g; gx_sum = gx_sum + (g * xhat rounded). Both sums
    start at +0 in every lane (a lane without pieces keeps +0). Then a butterfly over all 32
    lanes, offsets 16, 8, 4, 2, 1: each lane's v = v + (its xor-partner's v from before the step).
    The lanes' totals are identical; g_mean = total_g / float32(cols) and gx_mean = total_gx /
    float32(cols) (IEEE divisions). Then per value dx = rstd * ((g - g_mean) - (xhat * gx_mean)),
    each operation rounded; with accumulate_dx, dx = old + that, with one add.
  - dgamma, compensated (TwoSum, as the bias gradient's sums are; see test_bias_gradient.py for
    add, merge and the second pass). Pass 1 is the bias gradient's pass 1 exactly: chunks of 256
    rows, thread y = 0..7 takes rows 256c + y, + 8, ... and then thread 0 merges threads 1..7 in
    order of y, the chunk's pair stored as partial_s[c, j], partial_e[c, j] in the workspace, except
    that each term is a product added with its exact error: for each (r, j) p = dy * xhat
    (rounded), then `error = error + fmaf(dy, xhat, -p)` (the product's own rounding error, exact),
    then TwoSum-add p into the pair (s, error). Pass 2 is the bias gradient's pass 2 unchanged:
    merge the chunks per lane, a butterfly, the value s + e (s alone when s isn't finite), and with
    accumulate_params old + that value.
  - dbeta is flint_bias_gradient(dy) exactly (the same kernels, run first in the same workspace,
    which then holds dgamma's partial sums), and with accumulate_params likewise.

A numpy float32 emulation of exactly that order (emulate_dx and emulate_dgamma below, with
test_bias_gradient's emulate for dbeta and its second pass for dgamma's; themselves checked against a
literal scalar transcription, one np.float32 number at a time) must equal Flint's outputs and the
workspace's partial sums bit for bit. The inputs are made on the CPU with a seeded generator and
copied over, so they are the same on every machine. fmaf's error is emulated as
float32(float64(dy) * float64(xhat) - float64(p)): the product of an FP16 and an FP32 value is exact
in float64 and so is the difference, so the cast rounds once, like fmaf. Tests also show which
changes of order the bit-exact comparisons can tell apart.

What the tests cover:
  - the emulation against the scalar transcription, non-finite inputs included;
  - Flint against the emulation bit for bit (dx, dgamma, dbeta, and dgamma's partial sums and
    errors in the workspace), for mean and rstd from Flint's forward pass and for arbitrary ones,
    with rows not a multiple of 8 or 256, one row, more than 32 chunks, a partial last 256-column
    tile, cols 8, cols past 256 (lanes taking several pieces, an uneven number each), GPT-2's
    shapes, and inputs of several kinds (outlier channels near 3,000, gradient-scale and near
    FP16's range dy); dbeta equal to a direct flint_bias_gradient call;
  - the correctness gate (PLAN.md) against FP64 PyTorch autograd of F.layer_norm, with FP32
    PyTorch's own backward pass (autocast's path) as the peer, with factor 3 and with the
    owner's tighter factor 2, for dx, dgamma and dbeta;
  - accumulate (old + the overwrite result, bit for bit, for special olds; each flag touching only
    its outputs; micro-batches equal the sequential FP32 adds) and overwrite ignoring old contents;
  - inf and NaN in dy, x and gamma reaching exactly what they reach in FP32 PyTorch;
  - what the call writes and reads: guard bands around dx, dgamma, dbeta and the workspace, inputs
    unchanged, garbage in the workspace changing nothing, a larger workspace, one workspace for
    bias_gradient and layernorm_backward in turn;
  - determinism and streams, the workspace size, the limits (the widest cols, rows past 65,535
    chunks), and the argument errors.

The slow-to-reach cases are cheap in memory (the limits take at most about 2 GB of the GPU). Under
the sanitizer (scripts/sanitize.py) the shapes shrink to at most 300,000 elements and the tests of
the limits are skipped."""

import ctypes
import math
from collections import namedtuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flint_lib
import gate
import test_bias_gradient as bg
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator, normal
from test_bias_gradient import (  # noqa: F401  (free_gpu_memory is an autouse fixture)
    BUTTERFLY,
    CHECK_INPUT_ELEMENTS,
    CHUNK_ROWS,
    FILLS,
    GUARD_AFTER,
    HALF_INF,
    HALF_NAN,
    HALF_NEG_INF,
    MAX_COLS,
    NAN32,
    SENTINEL,
    Y_ORDER,
    all_equal,
    assert_same_bits,
    canonical,
    cpu_dy,
    free_gpu_memory,
    make_tight_input,
    pass2,
    set_words,
    shape_id,
    shape_params,
    shrunk,
    two_sum_add,
    two_sum_merge,
)
from test_layernorm import EPS, float32_value, init, large_mean, outliers, realistic, small_spread

MAX_ROWS = 0x7FFFFFFF * 8  # 17,179,869,176
F32 = np.float32
F64 = np.float64
LANES = np.arange(32)

# Elements per batch of the emulation's pass 1 (small enough to stay in the CPU's cache).
BATCH_ELEMENTS = 1 << 20

Tensors = namedtuple("Tensors", "dy x mean rstd gamma")
Result = namedtuple("Result", "dx dgamma dbeta partials")
Expected = namedtuple("Expected", "dx dgamma dbeta partial_s partial_e")


# -- The emulation of the kernels' arithmetic ---------------------------------------------------


def numpy_of(t):
    return t.numpy() if isinstance(t, torch.Tensor) else t


def lane_sums(v, reverse_lanes=False):
    """v is (n, steps * 256) float32: per row, lane l holds the pieces l, l + 32, ... of 8 columns
    each. Returns (n, 32): each lane's values summed in order with one FP32 add at a time, from
    +0 (np.add.accumulate is sequential, and the +0 in front keeps -0 sums out, as the kernel's
    +0 start does). With reverse_lanes the pieces are taken last first (not the kernel's order)."""
    n, padded = v.shape
    steps = padded // 256
    pieces = v.reshape(n, steps, 32, 8).transpose(0, 2, 1, 3)
    if reverse_lanes:
        pieces = pieces[:, :, ::-1, :]
    seq = np.zeros((n, 32, steps * 8 + 1), dtype=F32)
    seq[:, :, 1:] = pieces.reshape(n, 32, steps * 8)
    np.add.accumulate(seq, axis=2, out=seq)
    return seq[:, :, -1]


def butterfly_total(v, butterfly=BUTTERFLY):
    """Lane 0's total of the butterfly over (n, 32) lanes: each step v = v + (xor-partner's v)."""
    for offset in butterfly:
        v = v + v[:, LANES ^ offset]
    return v[:, 0]


def emulate_dx(dy, x, mean, rstd, gamma, old=None, reverse_lanes=False, butterfly=BUTTERFLY,
               reassociate=False):
    """The kernel's dx for the module docstring's arithmetic, as a float32 numpy array (rows x
    cols), in float32 throughout; with old (rows x cols FP32), old + dx with one add. Vectorized
    over rows. reverse_lanes (a lane's pieces last first), butterfly (the offsets' order) and
    reassociate (dx = rstd * (g - (g_mean + xhat * gx_mean))) are not the kernel's, and exist to
    show what the bit-exact tests can tell apart."""
    dy, x, mean, rstd, gamma = (numpy_of(t) for t in (dy, x, mean, rstd, gamma))
    rows, cols = x.shape
    pieces = cols // 8
    padded = -(-pieces // 32) * 256
    count = F32(cols)
    out = np.empty((rows, cols), dtype=F32)
    batch = max(1, BATCH_ELEMENTS // padded)
    with np.errstate(all="ignore"):
        for r0 in range(0, rows, batch):
            r1 = min(rows, r0 + batch)
            n = r1 - r0
            row_rstd = rstd[r0:r1, None]
            g = dy[r0:r1].astype(F32) * gamma[None, :]
            xhat = (x[r0:r1] - mean[r0:r1, None]) * row_rstd
            padded_g = np.zeros((n, padded), dtype=F32)
            padded_g[:, :cols] = g
            padded_gx = np.zeros((n, padded), dtype=F32)
            padded_gx[:, :cols] = g * xhat
            g_mean = (butterfly_total(lane_sums(padded_g, reverse_lanes), butterfly) / count)[:, None]
            gx_mean = (butterfly_total(lane_sums(padded_gx, reverse_lanes), butterfly) / count)[:, None]
            if reassociate:
                value = row_rstd * (g - (g_mean + xhat * gx_mean))
            else:
                value = row_rstd * ((g - g_mean) - (xhat * gx_mean))
            out[r0:r1] = value if old is None else numpy_of(old)[r0:r1] + value
    return out


def products(dy, x, mean, rstd, r0, r1, j0, j1):
    """Rows r0..r1 and columns j0..j1 of p = dy * xhat (rounded) and of the exact error of the
    product, fmaf(dy, xhat, -p), as float32 arrays."""
    d = dy[r0:r1, j0:j1].astype(F32)
    xhat = (x[r0:r1, j0:j1] - mean[r0:r1, None]) * rstd[r0:r1, None]
    p = d * xhat
    error = (d.astype(F64) * xhat.astype(F64) - p.astype(F64)).astype(F32)
    return p, error


def thread_merges(s, e, threads, y_order):
    """Thread 0's pair merged with those of the other threads (that have rows), in y_order."""
    a = (s[:, 0].copy(), e[:, 0].copy())
    for y in y_order:
        if y < threads:
            a = two_sum_merge(a, (s[:, y], e[:, y]))
    return a


def dgamma_pass1(dy, x, mean, rstd, j0, j1, y_order):
    """Pass 1 for columns j0..j1: the chunks' partial sums and errors, (chunks, j1 - j0) each."""
    rows = x.shape[0]
    cols = j1 - j0
    chunks = -(-rows // CHUNK_ROWS)
    full = rows // CHUNK_ROWS
    partial_s = np.empty((chunks, cols), dtype=F32)
    partial_e = np.empty((chunks, cols), dtype=F32)
    step = max(1, BATCH_ELEMENTS // (CHUNK_ROWS * cols))
    for c0 in range(0, full, step):
        c1 = min(full, c0 + step)
        p, error = products(dy, x, mean, rstd, c0 * CHUNK_ROWS, c1 * CHUNK_ROWS, j0, j1)
        p = p.reshape(c1 - c0, 32, 8, cols)
        error = error.reshape(c1 - c0, 32, 8, cols)
        s = np.zeros((c1 - c0, 8, cols), dtype=F32)
        e = np.zeros((c1 - c0, 8, cols), dtype=F32)
        for i in range(32):
            e = e + error[:, i]
            s, e = two_sum_add(s, e, p[:, i])
        partial_s[c0:c1], partial_e[c0:c1] = thread_merges(s, e, 8, y_order)
    if full < chunks:
        # The last chunk has fewer than 256 rows: threads past its rows hold (0, 0).
        first = full * CHUNK_ROWS
        n = rows - first
        p, error = products(dy, x, mean, rstd, first, rows, j0, j1)
        threads = min(8, n)
        s = np.zeros((1, threads, cols), dtype=F32)
        e = np.zeros((1, threads, cols), dtype=F32)
        for i in range(-(-n // 8)):
            k = min(8, n - 8 * i)
            e[0, :k] = e[0, :k] + error[8 * i : 8 * i + k]
            s[0, :k], e[0, :k] = two_sum_add(s[0, :k], e[0, :k], p[8 * i : 8 * i + k])
        partial_s[full], partial_e[full] = (v[0] for v in thread_merges(s, e, threads, y_order))
    return partial_s, partial_e


def emulate_dgamma(dy, x, mean, rstd, old=None, y_order=Y_ORDER, reverse_chunks=False,
                   butterfly=BUTTERFLY, with_partials=False):
    """The kernels' dgamma for the module docstring's arithmetic, as a float32 numpy array (cols),
    or with old (cols FP32) old + dgamma with one add. With with_partials also pass 1's partial
    sums and errors, (chunks, cols) each, as the workspace holds them. Pass 2 is the bias
    gradient's. y_order, reverse_chunks and butterfly are not the kernel's, and exist to show
    what the bit-exact tests can tell apart."""
    dy, x, mean, rstd = (numpy_of(t) for t in (dy, x, mean, rstd))
    rows, cols = x.shape
    chunks = -(-rows // CHUNK_ROWS)
    # Columns are independent: very wide rows are done in slices of columns, to bound the
    # temporaries.
    width = max(8, BATCH_ELEMENTS // min(rows, CHUNK_ROWS) // 8 * 8)
    partial_s = np.empty((chunks, cols), dtype=F32)
    partial_e = np.empty((chunks, cols), dtype=F32)
    with np.errstate(all="ignore"):
        for j0 in range(0, cols, width):
            j1 = min(cols, j0 + width)
            partial_s[:, j0:j1], partial_e[:, j0:j1] = dgamma_pass1(dy, x, mean, rstd, j0, j1, y_order)
        total = pass2(partial_s, partial_e, old, reverse_chunks, butterfly)
    if with_partials:
        return total, partial_s, partial_e
    return total


def emulate_all(case, old=None, accumulate_dx=False, accumulate_params=False):
    """Expected outputs (and dgamma's partials) for Tensors on the CPU; old is a dict of dx,
    dgamma and dbeta olds (numpy float32), used by the outputs whose flag is set."""
    old = {k: numpy_of(v) for k, v in (old or {}).items()}
    dx = emulate_dx(case.dy, case.x, case.mean, case.rstd, case.gamma, old["dx"] if accumulate_dx else None)
    dgamma, partial_s, partial_e = emulate_dgamma(
        case.dy, case.x, case.mean, case.rstd, old["dgamma"] if accumulate_params else None,
        with_partials=True,
    )
    dbeta = bg.emulate(case.dy, old["dbeta"] if accumulate_params else None)
    return Expected(dx, dgamma, dbeta, partial_s, partial_e)


# The literal scalar transcription of the module docstring.


def ts_add(a, v):
    f = F32
    s, e = a
    t = f(s + v)
    vp = f(t - s)
    d1 = f(t - vp)
    d2 = f(s - d1)
    d3 = f(v - vp)
    return t, f(e + f(d2 + d3))


def ts_merge(a, b):
    f = F32
    errs = f(a[1] + b[1])
    s, e = ts_add((a[0], f(0.0)), b[0])
    return s, f(e + errs)


def scalar_reference_backward(case, old=None, dx_rows=None):
    """dx, dgamma and dbeta by the module docstring, one np.float32 scalar at a time, every merge
    included (those of threads without rows and lanes without chunks too): slow, so for small
    shapes, to check the emulation. old is a dict of the olds to accumulate onto (the keys present
    are accumulated). dx is returned for the rows in dx_rows only (all by default), in that order."""
    f = F32
    old = {k: numpy_of(v) for k, v in (old or {}).items()}
    dy, x, mean, rstd, gamma = (numpy_of(t) for t in case)
    rows, cols = x.shape
    pieces = cols // 8
    chunks = -(-rows // CHUNK_ROWS)

    def xhat_of(r, c):
        return f(f(f(x[r, c]) - f(mean[r])) * f(rstd[r]))

    dx_rows = list(range(rows)) if dx_rows is None else list(dx_rows)
    dx = np.empty((len(dx_rows), cols), dtype=f)
    dgamma = np.empty(cols, dtype=f)
    with np.errstate(all="ignore"):
        for out_row, r in enumerate(dx_rows):
            g_lanes = [f(0.0)] * 32
            gx_lanes = [f(0.0)] * 32
            for lane in range(32):
                for piece in range(lane, pieces, 32):
                    for j in range(8):
                        c = 8 * piece + j
                        g = f(f(dy[r, c]) * f(gamma[c]))
                        g_lanes[lane] = f(g_lanes[lane] + g)
                        gx_lanes[lane] = f(gx_lanes[lane] + f(g * xhat_of(r, c)))
            for offset in (16, 8, 4, 2, 1):
                g_lanes = [f(g_lanes[l] + g_lanes[l ^ offset]) for l in range(32)]
                gx_lanes = [f(gx_lanes[l] + gx_lanes[l ^ offset]) for l in range(32)]
            g_mean = f(g_lanes[0] / f(cols))
            gx_mean = f(gx_lanes[0] / f(cols))
            for c in range(cols):
                g = f(f(dy[r, c]) * f(gamma[c]))
                value = f(f(rstd[r]) * f(f(g - g_mean) - f(xhat_of(r, c) * gx_mean)))
                dx[out_row, c] = value if "dx" not in old else f(f(old["dx"][r, c]) + value)
        for j in range(cols):
            partial = []
            for c in range(chunks):
                end = min(CHUNK_ROWS * c + CHUNK_ROWS, rows)
                threads = []
                for y in range(8):
                    a = (f(0.0), f(0.0))
                    r = CHUNK_ROWS * c + y
                    while r < end:
                        d = f(dy[r, j])
                        xhat = xhat_of(r, j)
                        p = f(d * xhat)
                        error = f(F64(d) * F64(xhat) - F64(p))
                        a = ts_add((a[0], f(a[1] + error)), p)
                        r += 8
                    threads.append(a)
                a = threads[0]
                for y in range(1, 8):
                    a = ts_merge(a, threads[y])
                partial.append(a)
            t = [(f(0.0), f(0.0))] * 32
            for lane in range(32):
                c = lane
                while c < chunks:
                    t[lane] = ts_merge(t[lane], partial[c])
                    c += 32
            for offset in (16, 8, 4, 2, 1):
                t = [ts_merge(t[l], t[l ^ offset]) for l in range(32)]
            s, e = t[0]
            value = f(s + e) if np.isfinite(s) else s
            dgamma[j] = value if "dgamma" not in old else f(f(old["dgamma"][j]) + value)
    dbeta = bg.scalar_reference(dy, old.get("dbeta"))
    return dx, dgamma, dbeta


# -- Inputs ---------------------------------------------------------------------------------


def cpu_x(kind, rows, cols, seed):
    """x on the CPU: N(0, 1), with kind "outliers" two channels near +3,000 and -2,500 in every
    row (as test_layernorm's outliers)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, cols), generator=g)
    if kind == "outliers":
        jitter = 0.9 + 0.2 * torch.rand((rows, 2), generator=g)
        x[:, cols // 3] = 3000.0 * jitter[:, 0]
        x[:, 2 * cols // 3] = -2500.0 * jitter[:, 1]
    elif kind != "normal":
        raise ValueError(kind)
    return x


def cpu_gamma(cols, seed):
    g = torch.Generator().manual_seed(seed)
    return 1.0 + 0.2 * torch.randn(cols, generator=g)


def cpu_dy_kind(kind, rows, cols, seed):
    """dy as FP16 on the CPU from a seeded generator: N(0, 1), gradient-scale N(0, 1e-3), "range"
    (uniform in +-60,000 with 10% subnormals), "outliers" (std 3,000 in 1% of the columns, and in
    one at least), or "spread" (N(0, 1) times a power of 2 from 2^-6 to 2^6)."""
    g = torch.Generator().manual_seed(seed)
    shape = (rows, cols)
    if kind == "normal":
        return torch.randn(shape, generator=g).half()
    if kind == "gradient":
        return (torch.randn(shape, generator=g) * 1e-3).half()
    if kind == "range":
        big = (torch.rand(shape, generator=g) * 2 - 1) * 60000
        sign = torch.randint(0, 2, shape, generator=g) * 2.0 - 1.0
        sub = torch.randint(1, 1024, shape, generator=g).float() * 2.0**-24 * sign
        use_sub = torch.rand(shape, generator=g) < 0.1
        return torch.where(use_sub, sub, big).half()
    if kind == "outliers":
        hit = torch.rand(cols, generator=g) < 0.01
        hit[cols // 3] = True
        std = torch.where(hit, 3000.0, 1.0)
        return (torch.randn(shape, generator=g) * std).half()
    if kind == "spread":
        return cpu_dy(rows, cols, seed)
    raise ValueError(kind)


def cpu_stats(rows, seed):
    """Arbitrary FP32 mean and rstd (not those of any x): mean N(0, 4), rstd positive and lognormal,
    5% of the rows 1,000 times larger."""
    g = torch.Generator().manual_seed(seed)
    mean = 2.0 * torch.randn(rows, generator=g)
    rstd = torch.exp(torch.randn(rows, generator=g))
    big = torch.rand(rows, generator=g) < 0.05
    return mean, torch.where(big, rstd * 1000.0, rstd)


def forward_stats(flint, x, gamma):
    """mean and rstd (GPU tensors) from Flint's forward pass on x (GPU) and gamma."""
    rows, cols = x.shape
    y = torch.empty((rows, cols), dtype=torch.float16, device="cuda")
    mean = torch.empty(rows, device="cuda")
    rstd = torch.empty(rows, device="cuda")
    flint.layernorm_forward(y, mean, rstd, x, gamma, torch.zeros(cols, device="cuda"), rows, cols, EPS)
    torch.cuda.synchronize()
    return mean, rstd


def make_case(flint, rows, cols, seed, x_kind="normal", dy_kind="normal", stats="forward"):
    """Tensors on the CPU. stats "forward" takes mean and rstd from Flint's forward pass on x,
    "random" arbitrary ones."""
    x = cpu_x(x_kind, rows, cols, seed)
    gamma = cpu_gamma(cols, seed + 1)
    dy = cpu_dy_kind(dy_kind, rows, cols, seed + 2)
    if stats == "forward":
        mean, rstd = (t.cpu() for t in forward_stats(flint, x.cuda(), gamma.cuda()))
    else:
        mean, rstd = cpu_stats(rows, seed + 3)
    return Tensors(dy, x, mean, rstd, gamma)


def to_gpu(case):
    return Tensors(*(t.cuda() for t in case))


# -- Running Flint -------------------------------------------------------------------------------


def bits_of(t):
    return t.view(torch.int16 if t.dtype == torch.float16 else torch.int32)


def run_backward(flint, t, old=None, accumulate_dx=False, accumulate_params=False, offset=4,
                 workspace_fill=NAN32, workspace_extra=0, out_fill=NAN32, stream=None,
                 check_inputs=None, with_workspace=False):
    """Runs flint_layernorm_backward on Tensors t (GPU) into dx, dgamma and dbeta inside buffers
    pre-filled with a sentinel, `offset` floats in (a multiple of 4 keeps them 16-byte aligned),
    and a workspace likewise (its first workspace_bytes filled with workspace_fill, workspace_extra
    more bytes of room after it). An output whose flag is set starts as old[name] (a dict of GPU
    tensors), the others as out_fill (an int32 bit pattern, or "garbage"). Checks that everything
    around the outputs and the workspace's used part is untouched and that the inputs are
    unchanged, and returns Result(dx (rows x cols), dgamma, dbeta, partials) as copies; partials
    is the workspace's used part as FP32 (chunks, 2, cols), each chunk's row of sums and row of
    errors, with with_workspace, else None."""
    rows, cols = t.x.shape
    needed = flint.layernorm_backward_workspace(rows, cols)
    assert needed == bg.expected_workspace(rows, cols)
    if check_inputs is None:
        check_inputs = t.x.numel() <= CHECK_INPUT_ELEMENTS
    counts = {"dx": rows * cols, "dgamma": cols, "dbeta": cols}
    buffers, views = {}, {}
    for name, count in counts.items():
        buffers[name] = torch.full((offset + count + GUARD_AFTER,), SENTINEL, dtype=torch.int32, device="cuda")
        views[name] = buffers[name][offset : offset + count]
        if accumulate_dx if name == "dx" else accumulate_params:
            views[name].view(torch.float32).copy_(old[name].reshape(-1))
        else:
            set_words(views[name], out_fill)
    words = needed // 4
    wbuf = torch.full((offset + words + workspace_extra // 4 + GUARD_AFTER,), SENTINEL,
                      dtype=torch.int32, device="cuda")
    workspace = wbuf[offset : offset + words + workspace_extra // 4]
    set_words(workspace[:words], workspace_fill)
    before = [bits_of(v).clone() for v in t] if check_inputs else None
    torch.cuda.synchronize()

    flint.layernorm_backward(
        views["dx"], views["dgamma"], views["dbeta"], t.dy, t.x, t.mean, t.rstd, t.gamma, rows, cols,
        accumulate_dx=accumulate_dx, accumulate_params=accumulate_params, workspace=workspace,
        workspace_bytes=needed + workspace_extra, stream=stream,
    )
    torch.cuda.synchronize()

    sentinel = torch.full((1,), SENTINEL, dtype=torch.int32, device="cuda")
    for name, count in counts.items():
        assert all_equal(buffers[name][:offset], sentinel), f"flint_layernorm_backward wrote before {name}"
        assert all_equal(buffers[name][offset + count :], sentinel), f"flint_layernorm_backward wrote past {name}"
    assert all_equal(wbuf[:offset], sentinel), "flint_layernorm_backward wrote before the workspace"
    assert all_equal(wbuf[offset + words :], sentinel), "flint_layernorm_backward wrote past its workspace"
    if check_inputs:
        for name, v, b in zip(Tensors._fields, t, before):
            assert torch.equal(bits_of(v), b), f"{name} changed"
    partials = None
    if with_workspace:
        partials = workspace[:words].view(torch.float32).reshape(-1, 2, cols).clone()
    return Result(
        views["dx"].view(torch.float32).reshape(rows, cols).clone(),
        views["dgamma"].view(torch.float32).clone(),
        views["dbeta"].view(torch.float32).clone(),
        partials,
    )


def check_result(got, expected, label):
    """Flint's Result equals the Expected bit for bit (and the workspace's partials, when the
    Result has them)."""
    assert_same_bits(got.dx, expected.dx, f"dx {label}")
    assert_same_bits(got.dgamma, expected.dgamma, f"dgamma {label}")
    assert_same_bits(got.dbeta, expected.dbeta, f"dbeta {label}")
    if got.partials is not None:
        assert_same_bits(got.partials[:, 0].reshape(-1), expected.partial_s.reshape(-1), f"partial sums {label}")
        assert_same_bits(got.partials[:, 1].reshape(-1), expected.partial_e.reshape(-1), f"partial errors {label}")


def count_differences(a, b):
    a, b = (torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in (a, b))
    return int((canonical(a) != canonical(b)).sum())


def rows_differing(a, b):
    """The number of rows (of 2-D arrays) in which a and b differ."""
    a, b = (torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in (a, b))
    return int((canonical(a) != canonical(b)).any(dim=1).sum())


# -- The emulation itself ----------------------------------------------------------------------

# (rows, cols): one row; a lane's second piece (264: 33 pieces) and an uneven count per lane (776:
# 97 pieces); more than 32 chunks (8453 rows have 34, so lanes take 2 or 1).
SCALAR_SHAPES = [(1, 8), (7, 264), (300, 24), (3, 776), (513, 8), (8453, 8)]


@pytest.mark.parametrize("rows, cols", shape_params(SCALAR_SHAPES))
def test_the_emulation_matches_a_literal_scalar_transcription(rows, cols):
    case = Tensors(
        cpu_dy_kind("spread", rows, cols, 100 + rows),
        cpu_x("normal", rows, cols, 101 + rows),
        *cpu_stats(rows, 102 + rows),
        cpu_gamma(cols, 103 + rows),
    )
    old = {
        "dx": torch.randn(rows, cols, generator=torch.Generator().manual_seed(rows)) * 100,
        "dgamma": torch.randn(cols, generator=torch.Generator().manual_seed(rows + 1)) * 100,
        "dbeta": torch.randn(cols, generator=torch.Generator().manual_seed(rows + 2)) * 100,
    }

    # dx's rows are independent, so a tall shape checks only its first and last rows.
    dx_rows = list(range(rows)) if rows <= 600 else list(range(20)) + list(range(rows - 20, rows))
    for accumulate in (False, True):
        olds = {k: v.numpy() for k, v in old.items()} if accumulate else None
        dx, dgamma, dbeta = scalar_reference_backward(case, olds, dx_rows)
        expected = emulate_all(case, olds, accumulate, accumulate)
        label = "accumulate" if accumulate else "overwrite"
        assert_same_bits(torch.from_numpy(expected.dx[dx_rows]), torch.from_numpy(dx), f"dx {label}")
        assert_same_bits(torch.from_numpy(expected.dgamma), torch.from_numpy(dgamma), f"dgamma {label}")
        assert_same_bits(torch.from_numpy(expected.dbeta), torch.from_numpy(dbeta), f"dbeta {label}")


def test_the_emulation_matches_the_scalar_transcription_with_non_finite_values():
    # Every kind of inf and NaN in dy, x and gamma, in columns of several chunks and lanes: the
    # emulation leaves (0, 0) pairs out of its merges and the transcription merges them, which may
    # differ only in the errors of non-finite sums, which the values don't use.
    rows, cols = 600, 40
    dy = cpu_dy_kind("normal", rows, cols, 5)
    x = cpu_x("normal", rows, cols, 6)
    mean, rstd = cpu_stats(rows, 7)
    gamma = cpu_gamma(cols, 8)
    dy[5, 0] = math.inf
    dy[300, 1] = -math.inf
    dy[10, 2], dy[560, 2] = math.inf, -math.inf
    dy[599, 3] = math.nan
    x[7, 4] = math.inf
    x[100, 5] = -math.inf
    x[270, 6] = math.nan
    gamma[9] = math.nan
    gamma[10] = math.inf
    gamma[11] = 0.0
    case = Tensors(dy, x, mean, rstd, gamma)

    dx, dgamma, dbeta = scalar_reference_backward(case)
    expected = emulate_all(case)

    assert_same_bits(torch.from_numpy(expected.dx), torch.from_numpy(dx), "dx")
    assert_same_bits(torch.from_numpy(expected.dgamma), torch.from_numpy(dgamma), "dgamma")
    assert_same_bits(torch.from_numpy(expected.dbeta), torch.from_numpy(dbeta), "dbeta")
    assert expected.dbeta[0] == math.inf and expected.dbeta[1] == -math.inf and math.isnan(expected.dbeta[2])
    assert not np.isfinite(expected.dx[5]).any() and np.isnan(expected.dx[:, 9]).all()
    assert not np.isfinite(expected.dgamma[4]) and not np.isfinite(expected.dgamma[5])
    assert math.isnan(expected.dgamma[6])
    # The statistics too: a NaN mean, an infinite and a zero rstd, in a shape of one chunk.
    rows, cols = 40, 16
    mean, rstd = cpu_stats(rows, 9)
    mean[3] = math.nan
    rstd[5] = math.inf
    rstd[6] = 0.0
    rstd[7] = math.nan
    case = Tensors(cpu_dy_kind("normal", rows, cols, 10), cpu_x("normal", rows, cols, 11), mean, rstd,
                   cpu_gamma(cols, 12))

    dx, dgamma, dbeta = scalar_reference_backward(case)
    expected = emulate_all(case)

    assert_same_bits(torch.from_numpy(expected.dx), torch.from_numpy(dx), "dx with statistics")
    assert_same_bits(torch.from_numpy(expected.dgamma), torch.from_numpy(dgamma), "dgamma with statistics")
    assert not np.isfinite(expected.dx[[3, 5, 7]]).any()


def test_the_emulation_stays_in_float32():
    case = make_case(None, 600, 24, 9, stats="random")

    expected = emulate_all(case, {k: np.zeros(s, dtype=F32) for k, s in
                                  (("dx", (600, 24)), ("dgamma", (24,)), ("dbeta", (24,)))}, True, True)

    for name in ("dx", "dgamma", "dbeta", "partial_s", "partial_e"):
        assert getattr(expected, name).dtype == np.float32, name


# -- Flint against the emulation, bit for bit -------------------------------------------------------

# One row, rows not a multiple of 8 or 256, whole and partial chunks, more than 32 chunks (8193 has
# 33, 33 * 256 + 5 an uneven number per lane), cols 8, a partial last tile (264, 520, 776) and cols
# past 256 where lanes take several pieces (776: 97 pieces, 4104: 513, so lane 0 takes one more
# than the others), and GPT-2's shapes.
EXACT_SHAPES = shrunk([
    (1, 8), (7, 24), (9, 16), (255, 8), (257, 264), (300, 520), (513, 8), (2049, 24), (8193, 8),
    (33 * 256 + 5, 264), (40 * 256 + 7, 8), (3, 776), (9, 4104), (1024, 768), (16384, 768),
])  # fmt: skip
STATS = ["forward", "random"]


@pytest.mark.parametrize("stats", STATS)
@pytest.mark.parametrize("rows, cols", shape_params(EXACT_SHAPES))
def test_flint_equals_the_emulation_bit_for_bit(flint, rows, cols, stats):
    case = make_case(flint, rows, cols, 11, stats=stats)

    got = run_backward(flint, to_gpu(case), with_workspace=True)

    check_result(got, emulate_all(case), f"{rows}x{cols} {stats}")


# Input kinds: outlier channels, gradient-scale dy, dy near FP16's range with subnormals, dy
# outliers, and dy of widely spread magnitudes; each on shapes with 2 to 64 chunks and several tiles.
KIND_COMBOS = [
    ("normal", "gradient", "forward"), ("outliers", "normal", "forward"),
    ("outliers", "range", "forward"), ("normal", "outliers", "random"),
    ("outliers", "outliers", "forward"), ("normal", "spread", "random"),
    ("outliers", "spread", "random"),
]  # fmt: skip
KIND_CASES = [
    (shape, combo)
    for shape, combos in (
        ((513, 264), KIND_COMBOS), ((1024, 768), KIND_COMBOS), ((8193, 8), KIND_COMBOS[1:4]),
        ((16384, 768), [KIND_COMBOS[2], KIND_COMBOS[4]]),
    )
    for combo in combos
    if shrunk([shape])
]  # fmt: skip


@pytest.mark.parametrize(
    "shape, combo", [pytest.param(shape, combo, id=f"{shape_id(shape)}-{'-'.join(combo)}") for shape, combo in KIND_CASES]
)
def test_other_inputs_equal_the_emulation_too(flint, shape, combo):
    (rows, cols), (x_kind, dy_kind, stats) = shape, combo
    case = make_case(flint, rows, cols, 21, x_kind, dy_kind, stats)

    got = run_backward(flint, to_gpu(case), with_workspace=True)

    check_result(got, emulate_all(case), f"{rows}x{cols} {x_kind} {dy_kind} {stats}")


FP32_SMALLEST_NORMAL = 1.17549435e-38


def subnormal_count(a):
    a = np.abs(numpy_of(a))
    return int(((a > 0) & (a < FP32_SMALLEST_NORMAL)).sum())


@pytest.mark.parametrize("scale", ["tiny-gamma", "tiny-rstd"])
@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(300, 264), (1024, 768)])))
def test_fp32_subnormals_are_not_flushed(flint, rows, cols, scale):
    # Subnormal FP32 values in the products, the sums, the divisions and the outputs: with gamma
    # near 1e-35 and gradient-scale dy, g = dy * gamma and dx are about 1e-38 (half of them
    # subnormal) and so are the sums' quotients; with rstd near 1e-37 and dy near FP16's range, xhat
    # is about 1e-37 and the products dy * xhat and their errors are subnormal for small dy. The
    # results must still equal the emulation, which keeps subnormals, bit for bit. The kernels are
    # not compiled with flush-to-zero, and dx's division goes through a slow path with FTZ
    # instructions on scaled operands (test_sass.py).
    if scale == "tiny-gamma":
        case = make_case(flint, rows, cols, 25, "normal", "gradient", "random")
        case = case._replace(gamma=case.gamma * 1e-35)
    else:
        case = make_case(flint, rows, cols, 25, "normal", "range", "random")
        case = case._replace(rstd=case.rstd * 1e-37)

    got = run_backward(flint, to_gpu(case), with_workspace=True)

    expected = emulate_all(case)
    check_result(got, expected, f"{rows}x{cols} {scale}")
    with np.errstate(all="ignore"):
        if scale == "tiny-gamma":
            terms = case.dy.numpy().astype(F32) * case.gamma.numpy()[None, :]
            assert subnormal_count(expected.dx) > 100
        else:
            xhat = (case.x.numpy() - case.mean.numpy()[:, None]) * case.rstd.numpy()[:, None]
            terms = case.dy.numpy().astype(F32) * xhat
        assert subnormal_count(terms) > 100
    print(f"SUBNORMAL {scale} {rows}x{cols}: {subnormal_count(terms)} subnormal terms, "
          f"{subnormal_count(expected.dx)} subnormal dx")


@pytest.mark.parametrize("accumulate_dx, accumulate_params", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(1, 8), (300, 264), (8193, 8), (513, 776)])))
def test_accumulate_equals_the_emulation_bit_for_bit(flint, rows, cols, accumulate_dx, accumulate_params):
    case = make_case(flint, rows, cols, 12, "outliers", "spread", "forward")
    g = torch.Generator().manual_seed(13)
    old = {"dx": torch.randn(rows, cols, generator=g) * 1000, "dgamma": torch.randn(cols, generator=g) * 1000,
           "dbeta": torch.randn(cols, generator=g) * 1000}

    got = run_backward(flint, to_gpu(case), old={k: v.cuda() for k, v in old.items()},
                       accumulate_dx=accumulate_dx, accumulate_params=accumulate_params, with_workspace=True)

    check_result(got, emulate_all(case, old, accumulate_dx, accumulate_params),
                 f"{rows}x{cols} dx {accumulate_dx} params {accumulate_params}")


# -- dbeta is the bias gradient -------------------------------------------------------------------------


@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
@pytest.mark.parametrize("dy_kind", ["range", "spread", "outliers"])
@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(300, 264), (8453, 24), (2049, 520)])))
def test_dbeta_equals_a_direct_bias_gradient_call(flint, rows, cols, dy_kind, accumulate):
    case = make_case(flint, rows, cols, 31, "normal", dy_kind, "random")
    old = torch.randn(cols, generator=torch.Generator().manual_seed(32)) * 100
    gpu = to_gpu(case)
    direct = run_flint_bias(flint, gpu.dy, old.cuda() if accumulate else None, accumulate)

    got = run_backward(flint, gpu, old={"dx": None, "dgamma": old.cuda(), "dbeta": old.cuda()},
                       accumulate_params=accumulate)

    assert_same_bits(got.dbeta, direct, "dbeta against flint_bias_gradient")
    assert_same_bits(got.dbeta, bg.emulate(case.dy, old.numpy() if accumulate else None), "dbeta against the emulation")


def run_flint_bias(flint, dy, old=None, accumulate=False):
    return bg.run_flint(flint, dy, old=old, accumulate=accumulate)


# -- The orders the bit-exact comparisons can tell apart ---------------------------------------------


def test_the_bit_exact_tests_tell_the_orders_of_dx_apart(flint):
    # On an ordinary input (N(0, 1) x and dy, gamma near 1, 2048 rows of 768 columns, 3 pieces per
    # lane) Flint's dx equals the emulation of the kernel's order and differs from emulations with
    # the lanes' pieces taken last first, the butterfly's offsets ascending, or the formula
    # reassociated: the rows the sums change, and the elements the formula changes, are counted.
    rows, cols = 2048, 768
    case = make_case(flint, rows, cols, 41)
    got = run_backward(flint, to_gpu(case)).dx

    kernel = emulate_dx(*case)
    assert_same_bits(got, kernel, "the kernel's order")
    variants = {
        "the lanes' pieces last first": emulate_dx(*case, reverse_lanes=True),
        "the butterfly's offsets 1, 2, 4, 8, 16": emulate_dx(*case, butterfly=(1, 2, 4, 8, 16)),
        "the butterfly's offsets 1, 2, 4, 8, 16 and the lanes reversed": emulate_dx(
            *case, reverse_lanes=True, butterfly=(1, 2, 4, 8, 16)),
        "dx = rstd (g - (g_mean + xhat gx_mean))": emulate_dx(*case, reassociate=True),
    }
    for name, other in variants.items():
        elements, differing_rows = count_differences(got, other), rows_differing(got, other)
        print(f"ORDER dx {name}: {elements} of {got.numel()} elements, {differing_rows} of {rows} rows differ")
        assert differing_rows > rows // 10, name
        assert elements > 1000, name


def test_the_partials_tell_the_order_of_the_thread_merges_apart(flint):
    # Merging dgamma's 8 threads in reverse changes many of pass 1's partial sums, which Flint's
    # workspace holds: it equals the emulation of the kernel's order and not this one.
    case = make_case(flint, 4096, 768, 42, "outliers", "normal", "forward")
    got = run_backward(flint, to_gpu(case), with_workspace=True)

    _, sums, errors = emulate_dgamma(*case[:4], with_partials=True)
    _, reversed_sums, _ = emulate_dgamma(*case[:4], y_order=tuple(reversed(Y_ORDER)), with_partials=True)

    assert_same_bits(got.partials[:, 0].reshape(-1), sums.reshape(-1), "partial sums")
    assert_same_bits(got.partials[:, 1].reshape(-1), errors.reshape(-1), "partial errors")
    differ = count_differences(got.partials[:, 0].reshape(-1), reversed_sums.reshape(-1))
    print(f"ORDER dgamma partial sums differing with the thread merges reversed: {differ} of {sums.size}")
    assert differ > 100


def test_the_compensation_changes_dgamma_s_bits():
    # Plain FP32 sums of the products (without the compensation and the products' errors) give a
    # different dgamma in many columns of an outlier input.
    case = make_case(None, 4096, 768, 43, "outliers", "normal", "random")
    dy, x, mean, rstd = (numpy_of(t) for t in case[:4])
    kernel = emulate_dgamma(dy, x, mean, rstd)
    with np.errstate(all="ignore"):
        xhat = (x - mean[:, None]) * rstd[:, None]
        p = dy.astype(F32) * xhat
        plain = np.zeros((p.shape[0] // 256, 8, 768), dtype=F32)
        blocks = p.reshape(-1, 32, 8, 768)
        for i in range(32):
            plain = plain + blocks[:, i]
        plain = plain.sum(1, dtype=F32).sum(0, dtype=F32)
    differ = count_differences(kernel, plain)
    print(f"ORDER dgamma against plain FP32 sums: {differ} of 768 columns differ")
    assert differ > 100


# -- Pass 2's order, on crafted inputs ------------------------------------------------------------------
# dgamma's and dbeta's second pass is the bias gradient's, whose order shows in the bits only on
# crafted inputs (see test_bias_gradient.py's crafted_cancellation). With x = 2, mean 0 and rstd 1,
# xhat is 2 and every product dy xhat is exact, so dgamma's partial sums are those of 2 dy, and the
# crafted dy shows pass 2's order in dgamma as in dbeta.


def crafted_case(cols=bg.CRAFTED_SHAPE[1]):
    """The crafted dy, or its first cols columns, with x = 2, mean 0, rstd 1 and gamma 1."""
    rows = bg.CRAFTED_SHAPE[0]
    dy = bg.crafted_cancellation(rows, cols, bg.CRAFTED_SEED)
    return Tensors(dy, torch.full((rows, cols), 2.0), torch.zeros(rows), torch.ones(rows), torch.ones(cols))


def test_crafted_inputs_tell_the_orders_of_dgamma_s_pass_2_apart():
    case = crafted_case()
    _, sums, errors = emulate_dgamma(*case[:4], with_partials=True)
    orders = bg.pass2_orders(sums, errors)
    kernel = orders.pop("the kernel's")
    assert count_differences(kernel, orders.pop("the kernel's lanes, merged by pass2_with_lanes")) == 0
    for name, other in orders.items():
        count = count_differences(kernel, other)
        print(f"ORDER crafted dgamma {name}: {count} of {kernel.size} columns differ")
        assert count >= 4, name


def test_dgamma_and_dbeta_equal_the_emulation_on_the_crafted_inputs(flint):
    # Under the sanitizer, the first 8 columns of the same input.
    case = crafted_case(8 if SANITIZER else bg.CRAFTED_SHAPE[1])

    got = run_backward(flint, to_gpu(case))

    assert_same_bits(got.dgamma, emulate_dgamma(*case[:4]), "dgamma crafted")
    assert_same_bits(got.dbeta, bg.emulate(case.dy), "dbeta crafted")


# -- The correctness gate -----------------------------------------------------------------------------

GATE_INPUTS = {
    "realistic": realistic,
    "init": init,
    "outliers": outliers,
    "large_mean": large_mean,
    "small_spread": small_spread,
}
GATE_CASES = [
    ("realistic", "normal"), ("init", "gradient"), ("outliers", "normal"), ("outliers", "gradient"),
    ("large_mean", "normal"), ("small_spread", "normal"), ("realistic", "range"), ("outliers", "range"),
    ("realistic", "outliers-dy"), ("outliers", "outliers-dy"),
]  # fmt: skip
GATE_SHAPES = shrunk([
    (16384, 768), (8192, 768), (1024, 768), (8453, 8), (8453, 768), (4096, 3072), (2049, 520),
    (257, 24), (7, 264), (1, 8), (3, 4104),
])  # fmt: skip
OUTPUTS = ("dx", "dgamma", "dbeta")


def gate_dy(kind, rows, cols, seed):
    """dy (FP16 on the GPU) of test_bias_gradient's recipes: "outliers-dy" is its outliers."""
    return make_tight_input("outliers" if kind == "outliers-dy" else kind, rows, cols, generator(1000 + seed))


def torch_backward(x, gamma, beta, dy, dtype):
    """dx, dgamma, dbeta from PyTorch autograd of F.layer_norm in dtype (FP64 for the reference,
    FP32 for the peer, which is what autocast runs), with eps as Flint receives it."""
    cols = x.shape[1]
    xs = x.detach().to(dtype).clone().requires_grad_(True)
    ws = gamma.detach().to(dtype).clone().requires_grad_(True)
    bs = beta.detach().to(dtype).clone().requires_grad_(True)
    with torch.enable_grad():
        y = F.layer_norm(xs, (cols,), ws, bs, float32_value(EPS))
        y.backward(dy.to(dtype))
    return xs.grad, ws.grad, bs.grad


def ratio_line(label, name, flint_errors, peer_errors, factor_bounds):
    fr, fm = bg.worst(flint_errors)
    pr, pm = bg.worst(peer_errors)
    print(
        f"GATE {label} {name} | flint rel_l2 {fr:.3g} max_abs {fm:.3g} | peer rel_l2 {pr:.3g} "
        f"max_abs {pm:.3g} | ratio rel_l2 {fr / pr if pr else float('nan'):.3g} "
        f"max_abs {fm / pm if pm else float('nan'):.3g} | bounds {factor_bounds}"
    )


@pytest.mark.parametrize("x_kind, dy_kind", GATE_CASES, ids=lambda v: str(v).replace("-", "_"))
@pytest.mark.parametrize("rows, cols", shape_params(GATE_SHAPES))
def test_passes_the_gate_against_fp32_pytorch(flint, rows, cols, x_kind, dy_kind):
    flint_errors = {name: [] for name in OUTPUTS}
    peer_errors = {name: [] for name in OUTPUTS}
    for seed in gate.SEEDS:
        x, gamma, beta = GATE_INPUTS[x_kind](rows, cols, seed)
        dy = gate_dy(dy_kind, rows, cols, seed)
        mean, rstd = forward_stats(flint, x, gamma)
        got = run_backward(flint, Tensors(dy, x, mean, rstd, gamma), offset=4 * (1 + seed))
        ref = torch_backward(x, gamma, beta, dy, torch.float64)
        peer = torch_backward(x, gamma, beta, dy.float(), torch.float32)
        for name, f, p, r in zip(OUTPUTS, (got.dx, got.dgamma, got.dbeta), peer, ref):
            flint_errors[name].append(gate.errors(f, r))
            peer_errors[name].append(gate.errors(p, r))
        del x, gamma, beta, dy, mean, rstd, got, ref, peer

    failures = []
    for name in OUTPUTS:
        bounds = []
        for factor in (gate.GRADIENT_FACTOR, 2.0):
            try:
                b = gate.check_gate(factor, flint_errors[name], peer_errors[name], output="fp32")
                bounds.append(f"x{factor:g} ok (rel_l2 {b.rel_l2:.3g}, max_abs {b.max_abs:.3g})")
            except AssertionError as error:
                bounds.append(f"x{factor:g} FAIL")
                failures.append(f"{name} at factor {factor:g}:\n{error}")
        ratio_line(f"{x_kind}/{dy_kind} {shape_id((rows, cols))}", name, flint_errors[name], peer_errors[name],
                   "; ".join(bounds))
    assert not failures, "\n".join(failures)


def test_autocast_runs_layer_norm_in_fp32():
    # The peer of the gate is FP32 PyTorch's layer_norm: under autocast to FP16 it takes FP32 input
    # and returns FP32 (layer_norm is on autocast's FP32 list, so FP16 input is upcast too), and the
    # gradient coming back from the FP16 linear layer after it is upcast to FP32 before the
    # layer_norm backward runs.
    cols = 256
    x = torch.randn(64, cols, device="cuda", requires_grad=True)
    gamma = torch.ones(cols, device="cuda", requires_grad=True)
    beta = torch.zeros(cols, device="cuda", requires_grad=True)
    weight = torch.randn(32, cols, device="cuda", requires_grad=True)
    with torch.autocast("cuda", torch.float16):
        y = F.layer_norm(x, (cols,), gamma, beta)
        y_from_half = F.layer_norm(x.half().detach(), (cols,), gamma, beta)
        y.retain_grad()
        out = F.linear(y, weight)
    assert y.dtype == torch.float32 and y_from_half.dtype == torch.float32
    assert out.dtype == torch.float16
    out.backward(torch.randn(64, 32, device="cuda").half())
    assert y.grad.dtype == torch.float32
    assert x.grad.dtype == torch.float32 and gamma.grad.dtype == torch.float32
    # y's gradient is an FP16-representable value upcast: a round trip through FP16 is exact.
    assert torch.equal(y.grad.half().float(), y.grad)


# -- accumulate --------------------------------------------------------------------------------------


def special_olds(count, seed):
    """FP32 olds, flat: random, with +-0, subnormals, inf, -inf, NaN and large values scattered."""
    return bg.olds(count, seed)


ACCUMULATE_SHAPES = shrunk([(1, 8), (7, 264), (300, 264), (8453, 24)])


@pytest.mark.parametrize("rows, cols", shape_params(ACCUMULATE_SHAPES))
def test_accumulate_adds_the_overwrite_result_to_old_with_one_add(flint, rows, cols):
    # All four combinations of the flags. A flag touches only its outputs: the others are
    # overwritten (they start as NaN) with exactly the overwrite result. Columns j % 5 == 0 of dy
    # are 0.
    case = make_case(flint, rows, cols, 51, "outliers", "normal", "forward")
    case.dy[:, ::5] = 0
    gpu = to_gpu(case)
    total = run_backward(flint, gpu)
    old = {"dx": special_olds(rows * cols, 52).reshape(rows, cols).cuda(),
           "dgamma": special_olds(cols, 53).cuda(), "dbeta": special_olds(cols, 54).cuda()}

    for accumulate_dx in (False, True):
        for accumulate_params in (False, True):
            got = run_backward(flint, gpu, old=old, accumulate_dx=accumulate_dx,
                               accumulate_params=accumulate_params, offset=8)

            label = f"dx {accumulate_dx} params {accumulate_params}"
            assert_same_bits(got.dx, old["dx"] + total.dx if accumulate_dx else total.dx, f"dx {label}")
            assert_same_bits(got.dgamma, old["dgamma"] + total.dgamma if accumulate_params else total.dgamma,
                             f"dgamma {label}")
            assert_same_bits(got.dbeta, old["dbeta"] + total.dbeta if accumulate_params else total.dbeta,
                             f"dbeta {label}")
            expected = emulate_all(case, {k: v.cpu() for k, v in old.items()}, accumulate_dx, accumulate_params)
            check_result(got, expected, label)


def test_accumulate_onto_special_olds_gives_the_fp32_sum_of_each(flint):
    # Every column holds one kind of old: 0, -0, subnormals, inf, -inf, NaN, near FLT_MAX. The result
    # is the FP32 sum of old and the overwrite result, whatever the kind (an inf or NaN old stays).
    rows, cols = 300, 80
    case = make_case(flint, rows, cols, 55, "normal", "normal", "forward")
    gpu = to_gpu(case)
    values = [0.0, -0.0, 1e-40, -1e-40, math.inf, -math.inf, math.nan, 3.0e38, -3.0e38, 3.4028234e38]
    column_old = torch.tensor([values[j % len(values)] for j in range(cols)])
    old = {"dx": column_old[None, :].expand(rows, cols).contiguous().cuda(),
           "dgamma": column_old.cuda(), "dbeta": column_old.cuda()}
    total = run_backward(flint, gpu)

    got = run_backward(flint, gpu, old=old, accumulate_dx=True, accumulate_params=True)

    assert_same_bits(got.dx, old["dx"] + total.dx, "dx")
    assert_same_bits(got.dgamma, old["dgamma"] + total.dgamma, "dgamma")
    assert_same_bits(got.dbeta, old["dbeta"] + total.dbeta, "dbeta")
    kinds = torch.arange(cols, device="cuda") % len(values)
    for name in ("dgamma", "dbeta"):
        out = getattr(got, name)
        assert bool(torch.isposinf(out)[kinds == 4].all()) and bool(torch.isneginf(out)[kinds == 5].all())
        assert bool(torch.isnan(out)[kinds == 6].all())
    assert bool(torch.isnan(got.dx[:, kinds == 6]).all()) and bool(torch.isposinf(got.dx[:, kinds == 4]).all())


@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(300, 264), (513, 24), (8453, 8)])))
def test_micro_batches_equal_the_sequential_fp32_adds(flint, rows, cols):
    # Three calls in a row on one stream, sharing one workspace, with no sync between them: the
    # first overwrites, the rest accumulate (and the same again from zeros by accumulating all).
    cases = [make_case(flint, rows, cols, 60 + 10 * i, "outliers", "spread", "random") for i in range(3)]
    gpu = [to_gpu(c) for c in cases]
    workspace = torch.full((flint.layernorm_backward_workspace(rows, cols) // 4,), math.nan, device="cuda")

    outs = {}
    for label, first in (("overwrite", False), ("zeros", True)):
        outs[label] = (torch.zeros((rows, cols), device="cuda") if first else torch.full((rows, cols), math.nan, device="cuda"),
                       torch.zeros(cols, device="cuda") if first else torch.full((cols,), math.nan, device="cuda"),
                       torch.zeros(cols, device="cuda") if first else torch.full((cols,), math.nan, device="cuda"))
    for i, t in enumerate(gpu):
        for label, first in (("overwrite", False), ("zeros", True)):
            dx, dgamma, dbeta = outs[label]
            accumulate = i > 0 or first
            flint.layernorm_backward(dx, dgamma, dbeta, t.dy, t.x, t.mean, t.rstd, t.gamma, rows, cols,
                                     accumulate_dx=accumulate, accumulate_params=accumulate, workspace=workspace)
    torch.cuda.synchronize()

    totals = [emulate_all(c) for c in cases]
    expected = {}
    for label in outs:
        total = [np.zeros(shape, dtype=F32) for shape in ((rows, cols), (cols,), (cols,))] if label == "zeros" else None
        for e in totals:
            parts = (e.dx, e.dgamma, e.dbeta)
            total = parts if total is None else [a + b for a, b in zip(total, parts)]
        expected[label] = total
    for label in expected:
        for name, got, want in zip(OUTPUTS, outs[label], expected[label]):
            assert want.dtype == np.float32
            assert_same_bits(got, want, f"{name} {label}")


@pytest.mark.parametrize("fill", list(FILLS))
def test_overwrite_ignores_the_old_content_of_the_outputs(flint, fill):
    rows, cols = 513, 264
    case = make_case(flint, rows, cols, 70, "normal", "spread", "forward")

    got = run_backward(flint, to_gpu(case), out_fill=FILLS[fill], with_workspace=True)

    check_result(got, emulate_all(case), f"over {fill}")
    assert not bool(torch.isnan(got.dx).any() or torch.isnan(got.dgamma).any() or torch.isnan(got.dbeta).any())


# -- inf and NaN -------------------------------------------------------------------------------------

NF_ROWS, NF_COLS = 300, 264
# Different rows (chunks of 256, the last of 44, and y lanes) and columns (tiles of 256, the last of
# 8, pieces, and lanes).
NF_POSITIONS = [(0, 0), (5, 3), (255, 100), (256, 263), (299, 131), (130, 8), (77, 255)]
NF_BITS = [(HALF_INF, "inf"), (HALF_NEG_INF, "-inf"), (HALF_NAN, "nan")]


def clean_inputs():
    x, gamma, beta = realistic(NF_ROWS, NF_COLS, 0)
    dy = normal((NF_ROWS, NF_COLS), 1.0, generator(1))
    return x, gamma, beta, dy


def placement(a):
    """What the NaN and inf placement of a tensor compares: its NaN mask and its +inf and -inf masks."""
    return torch.isnan(a), torch.isposinf(a), torch.isneginf(a)


def assert_same_placement(got, peer, what, strict_dx=True):
    nan_got, nan_peer = torch.isnan(got), torch.isnan(peer)
    finite_got, finite_peer = torch.isfinite(got), torch.isfinite(peer)
    assert torch.equal(finite_got, finite_peer), (
        f"{what}: the non-finite elements differ: Flint {int((~finite_got).sum())}, PyTorch {int((~finite_peer).sum())}")
    if strict_dx:
        for mask_got, mask_peer, name in zip(placement(got), placement(peer), ("NaN", "+inf", "-inf")):
            assert torch.equal(mask_got, mask_peer), (
                f"{what}: the {name} elements differ: Flint {int(mask_got.sum())}, PyTorch {int(mask_peer.sum())}")


@pytest.mark.parametrize("bits, kind", NF_BITS, ids=[k for _, k in NF_BITS])
@pytest.mark.parametrize("row, col", NF_POSITIONS, ids=lambda v: str(v))
def test_a_non_finite_dy_reaches_its_row_of_dx_and_its_columns_only(flint, bits, kind, row, col):
    x, gamma, beta, dy = clean_inputs()
    mean, rstd = forward_stats(flint, x, gamma)
    clean = run_backward(flint, Tensors(dy, x, mean, rstd, gamma))
    bad = dy.clone()
    bad.view(torch.int16)[row, col] = as_int16(bits)

    got = run_backward(flint, Tensors(bad, x, mean, rstd, gamma))

    # dx: the whole row is non-finite (g_sum is inf or NaN, so every value of the row is), the
    # other rows are those of the clean run. dgamma and dbeta: the column only.
    assert not bool(torch.isfinite(got.dx[row]).any())
    other_rows = torch.ones(NF_ROWS, dtype=torch.bool)
    other_rows[row] = False
    assert_same_bits(got.dx[other_rows.cuda()], clean.dx[other_rows.cuda()], "the other rows of dx")
    other_cols = torch.ones(NF_COLS, dtype=torch.bool)
    other_cols[col] = False
    for name in ("dgamma", "dbeta"):
        out, clean_out = getattr(got, name), getattr(clean, name)
        assert not bool(torch.isfinite(out[col])), name
        assert_same_bits(out[other_cols.cuda()], clean_out[other_cols.cuda()], f"the other columns of {name}")
    if kind == "nan":
        assert bool(torch.isnan(got.dbeta[col])) and bool(torch.isnan(got.dgamma[col]))
    else:
        sign = math.inf if kind == "inf" else -math.inf
        assert got.dbeta[col].item() == sign
        xhat = (x[row, col] - mean[row]) * rstd[row]
        assert got.dgamma[col].item() == math.copysign(math.inf, sign * xhat.item())
    # Against FP32 PyTorch's placement, and the emulation bit for bit.
    peer = torch_backward(x, gamma, beta, bad.float(), torch.float32)
    assert_same_placement(got.dx, peer[0], "dx")
    assert_same_placement(got.dgamma, peer[1], "dgamma")
    assert_same_placement(got.dbeta, peer[2], "dbeta")
    case = Tensors(*(t.cpu() for t in (bad, x, mean, rstd, gamma)))
    check_result(got, emulate_all(case), f"dy {kind} at {row},{col}")


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan], ids=["inf", "-inf", "nan"])
@pytest.mark.parametrize("row, col", [(0, 0), (130, 100), (255, 255), (256, 3), (299, 263)], ids=lambda v: str(v))
def test_a_non_finite_x_makes_its_row_nan_and_every_dgamma_nan(flint, value, row, col):
    # The forward pass gives the row NaN mean and rstd (NaN xhat), so dx's row is NaN, every column
    # of dgamma sums a NaN, and dbeta (the column sums of dy) is untouched.
    x, gamma, beta, dy = clean_inputs()
    mean, rstd = forward_stats(flint, x, gamma)
    clean = run_backward(flint, Tensors(dy, x, mean, rstd, gamma))
    x_bad = x.clone()
    x_bad[row, col] = value
    mean_bad, rstd_bad = forward_stats(flint, x_bad, gamma)
    assert bool(torch.isnan(mean_bad[row]) and torch.isnan(rstd_bad[row]))

    got = run_backward(flint, Tensors(dy, x_bad, mean_bad, rstd_bad, gamma))

    assert bool(torch.isnan(got.dx[row]).all())
    other_rows = (torch.arange(NF_ROWS, device="cuda") != row)
    assert_same_bits(got.dx[other_rows], clean.dx[other_rows], "the other rows of dx")
    assert bool(torch.isnan(got.dgamma).all())
    assert_same_bits(got.dbeta, clean.dbeta, "dbeta")
    peer = torch_backward(x_bad, gamma, beta, dy.float(), torch.float32)
    assert_same_placement(got.dx, peer[0], "dx")
    assert_same_placement(got.dgamma, peer[1], "dgamma")
    assert_same_placement(got.dbeta, peer[2], "dbeta")
    case = Tensors(*(t.cpu() for t in (dy, x_bad, mean_bad, rstd_bad, gamma)))
    check_result(got, emulate_all(case), f"x {value} at {row},{col}")


@pytest.mark.parametrize("col", [0, 7, 8, 131, 255, 256, 263])
def test_a_nan_in_gamma_makes_dx_nan_and_leaves_dgamma_and_dbeta(flint, col):
    # g = dy * gamma is NaN at the column in every row, so every row's sums and every dx are NaN;
    # gamma doesn't enter dgamma or dbeta.
    x, gamma, beta, dy = clean_inputs()
    mean, rstd = forward_stats(flint, x, gamma)
    clean = run_backward(flint, Tensors(dy, x, mean, rstd, gamma))
    gamma_bad = gamma.clone()
    gamma_bad[col] = math.nan

    got = run_backward(flint, Tensors(dy, x, mean, rstd, gamma_bad))

    assert bool(torch.isnan(got.dx).all())
    assert_same_bits(got.dgamma, clean.dgamma, "dgamma")
    assert_same_bits(got.dbeta, clean.dbeta, "dbeta")
    peer = torch_backward(x, gamma_bad, beta, dy.float(), torch.float32)
    assert_same_placement(got.dx, peer[0], "dx")
    assert_same_placement(got.dgamma, peer[1], "dgamma")
    assert_same_placement(got.dbeta, peer[2], "dbeta")
    case = Tensors(*(t.cpu() for t in (dy, x, mean, rstd, gamma_bad)))
    check_result(got, emulate_all(case), f"gamma NaN at {col}")


# -- What it writes and reads ------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [4, 8, 12, 60])
def test_nothing_is_written_around_the_outputs_or_the_workspace(flint, offset):
    # run_backward checks the guard bands, the workspace's room and the inputs; the result is right too.
    case = make_case(flint, 513, 264, 80)

    got = run_backward(flint, to_gpu(case), offset=offset, workspace_extra=4096, with_workspace=True)

    check_result(got, emulate_all(case), f"offset {offset}")


@pytest.mark.parametrize("fill", list(FILLS))
def test_the_workspaces_old_content_changes_nothing(flint, fill):
    case = make_case(flint, 1000, 264, 81)

    got = run_backward(flint, to_gpu(case), workspace_fill=FILLS[fill], with_workspace=True)

    check_result(got, emulate_all(case), f"workspace of {fill}")


@pytest.mark.parametrize("extra", [16, 4096])
def test_the_workspace_may_be_larger_than_needed(flint, extra):
    case = make_case(flint, 700, 24, 82)

    got = run_backward(flint, to_gpu(case), workspace_extra=extra, with_workspace=True)

    check_result(got, emulate_all(case), f"workspace + {extra}")


def test_one_workspace_serves_bias_gradient_and_layernorm_backward_in_turn(flint):
    rows, cols = 700, 264
    a = make_case(flint, rows, cols, 83, stats="random")
    b = make_case(flint, rows, cols, 84, "outliers", "spread", "random")
    ga, gb = to_gpu(a), to_gpu(b)
    workspace = torch.full((flint.layernorm_backward_workspace(rows, cols) // 4,), math.nan, device="cuda")
    assert flint.layernorm_backward_workspace(rows, cols) == flint.bias_gradient_workspace(rows, cols)
    dbias_1, dbias_2 = torch.empty(cols, device="cuda"), torch.empty(cols, device="cuda")
    dx = torch.empty((rows, cols), device="cuda")
    dgamma, dbeta = torch.empty(cols, device="cuda"), torch.empty(cols, device="cuda")

    flint.bias_gradient(dbias_1, ga.dy, rows, cols, workspace=workspace)
    flint.layernorm_backward(dx, dgamma, dbeta, gb.dy, gb.x, gb.mean, gb.rstd, gb.gamma, rows, cols,
                             workspace=workspace)
    flint.bias_gradient(dbias_2, ga.dy, rows, cols, workspace=workspace)
    torch.cuda.synchronize()

    expected = emulate_all(b)
    assert_same_bits(dbias_1, bg.emulate(a.dy), "bias gradient before")
    assert_same_bits(dbias_2, bg.emulate(a.dy), "bias gradient after")
    assert_same_bits(dx, expected.dx, "dx")
    assert_same_bits(dgamma, expected.dgamma, "dgamma")
    assert_same_bits(dbeta, expected.dbeta, "dbeta")


# -- Determinism and streams -------------------------------------------------------------------------


@pytest.mark.parametrize("rows, cols", shape_params(shrunk([(513, 264), (8453, 24), (16384, 768)])))
def test_run_twice_is_bitwise_identical(flint, rows, cols):
    case = make_case(flint, rows, cols, 90, "outliers", "range", "forward")
    case.dy[rows // 2, cols // 2] = math.nan
    gpu = to_gpu(case)

    first = run_backward(flint, gpu, offset=4, workspace_fill=FILLS["nan"], out_fill=FILLS["nan"], with_workspace=True)
    second = run_backward(flint, gpu, offset=8, workspace_fill=FILLS["garbage"], out_fill=FILLS["garbage"], with_workspace=True)
    third = run_backward(flint, gpu, offset=12, workspace_fill=0, out_fill=0, with_workspace=True)

    assert bool(torch.isnan(first.dgamma).any()) and bool(torch.isnan(first.dx).any())
    for name, other in (("second", second), ("third", third)):
        for field in ("dx", "dgamma", "dbeta"):
            assert torch.equal(canonical(getattr(other, field)), canonical(getattr(first, field))), f"{name} {field}"
        assert torch.equal(canonical(other.partials), canonical(first.partials)), f"{name} partials"


STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
def test_runs_on_the_given_stream(flint, mode, accumulate):
    rows, cols = 700, 264
    case = make_case(flint, rows, cols, 91, "normal", "spread", "random")
    old = {"dx": torch.randn(rows, cols, generator=torch.Generator().manual_seed(92)),
           "dgamma": torch.randn(cols, generator=torch.Generator().manual_seed(93)),
           "dbeta": torch.randn(cols, generator=torch.Generator().manual_seed(94))}
    expected = emulate_all(case, old, accumulate, accumulate)
    source = to_gpu(case)
    t = Tensors(*(torch.full(v.shape, math.nan, dtype=v.dtype, device="cuda") for v in source))
    dx = torch.full((rows, cols), math.nan, device="cuda")
    dgamma = torch.full((cols,), math.nan, device="cuda")
    dbeta = torch.full((cols,), math.nan, device="cuda")
    workspace = torch.full((flint.layernorm_backward_workspace(rows, cols) // 4,), math.nan, device="cuda")
    stream = torch.cuda.Stream()
    # A first call loads the kernels' module, which takes longer than the sleep below: without
    # this warm-up a call on the wrong stream would start late enough to pass.
    run_backward(flint, source)
    torch.cuda.synchronize()

    def enqueue(call_stream):
        # The copies into the inputs finish only after the sleep, so the outputs are right only if
        # all five kernels are queued on the same stream behind them (they are NaN where they ran
        # first). With the current stream the call has no stream argument, and torch's current
        # stream is the side one; with the default stream there is no side one at all.
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        for dst, src in zip(t, source):
            dst.copy_(src)
        if accumulate:
            dx.copy_(old["dx"].cuda())
            dgamma.copy_(old["dgamma"].cuda())
            dbeta.copy_(old["dbeta"].cuda())
        flint.layernorm_backward(dx, dgamma, dbeta, t.dy, t.x, t.mean, t.rstd, t.gamma, rows, cols,
                                 accumulate_dx=accumulate, accumulate_params=accumulate,
                                 workspace=workspace, stream=call_stream)

    if mode == "default-stream":
        enqueue(None)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            enqueue(stream if mode == "explicit-stream" else None)
        stream.synchronize()

    assert not bool(torch.isnan(dx).any() or torch.isnan(dgamma).any() or torch.isnan(dbeta).any())
    assert_same_bits(dx, expected.dx, f"dx {mode}")
    assert_same_bits(dgamma, expected.dgamma, f"dgamma {mode}")
    assert_same_bits(dbeta, expected.dbeta, f"dbeta {mode}")


# -- The workspace size ---------------------------------------------------------------------------------

WORKSPACE_SHAPES = [
    (1, 8), (1, 768), (7, 24), (255, 8), (256, 8), (257, 8), (300, 264), (511, 520), (512, 520),
    (513, 520), (8192, 8), (8193, 8), (16384, 768), (16384, 3072), (1024, 768), (33 * 256 + 5, 264),
    (2**32, 64), (1, MAX_COLS), (2, MAX_COLS), (256, MAX_COLS), (257, MAX_COLS), (MAX_ROWS, 8),
    (MAX_ROWS - 1, 8), (MAX_ROWS, MAX_COLS), (MAX_ROWS - 255, MAX_COLS),
]  # fmt: skip


@pytest.mark.parametrize("rows, cols", shape_params(WORKSPACE_SHAPES))
def test_workspace_is_the_bias_gradients(flint, rows, cols):
    needed = flint.layernorm_backward_workspace(rows, cols)

    assert needed == bg.expected_workspace(rows, cols) == flint.bias_gradient_workspace(rows, cols)


# -- The limits ---------------------------------------------------------------------------------------


@pytest.mark.skipif(SANITIZER, reason="the widest rows are too big for the sanitizer")
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
def test_the_widest_row(flint, accumulate):
    rows, cols = 1, MAX_COLS
    case = make_case(None, rows, cols, 100, "normal", "spread", "random")
    g = torch.Generator().manual_seed(101)
    old = {"dx": torch.randn(rows, cols, generator=g) * 10, "dgamma": torch.randn(cols, generator=g) * 10,
           "dbeta": torch.randn(cols, generator=g) * 10}

    got = run_backward(flint, to_gpu(case), old={k: v.cuda() for k, v in old.items()},
                       accumulate_dx=accumulate, accumulate_params=accumulate, with_workspace=True)

    expected = emulate_all(case, old, accumulate, accumulate)
    check_result(got, expected, f"{rows}x{cols}")
    if not accumulate:
        # One row: the sums are the terms themselves (-0 as +0): dbeta = dy.
        assert torch.equal(canonical(got.dbeta), canonical(case.dy[0].float() + 0.0))


def fp64_parameter_gradients(t, block=1 << 20):
    """dgamma and dbeta in FP64 from the FP32 xhat's inputs, in blocks of rows."""
    rows, cols = t.x.shape
    dgamma = torch.zeros(cols, dtype=torch.float64, device="cuda")
    dbeta = torch.zeros(cols, dtype=torch.float64, device="cuda")
    for r0 in range(0, rows, block):
        r1 = min(rows, r0 + block)
        d = t.dy[r0:r1].double()
        xhat = (t.x[r0:r1].double() - t.mean[r0:r1, None].double()) * t.rstd[r0:r1, None].double()
        dgamma += (d * xhat).sum(0)
        dbeta += d.sum(0)
    return dgamma, dbeta


@pytest.mark.skipif(SANITIZER, reason="the dy and x of 65,538 chunks are too much for the sanitizer")
@pytest.mark.parametrize("rows", [65_538 * CHUNK_ROWS, 65_537 * CHUNK_ROWS + 7], ids=["65538-chunks", "65538-chunks-partial"])
def test_more_than_65535_chunks(flint, rows):
    # The first pass has a block per chunk in its grid's x, which holds far more than 65,535. dgamma
    # and dbeta are checked against FP64 sums, and a sample of dx's rows (the first and last of each
    # kind of block) against the emulation bit for bit.
    cols = 8
    g = generator(110)
    gamma = 1.0 + 0.2 * torch.randn(cols, generator=g, device="cuda")
    t = Tensors(
        normal((rows, cols), 1.0, g),
        torch.randn((rows, cols), generator=g, device="cuda"),
        0.3 * torch.randn(rows, generator=g, device="cuda"),
        torch.exp(0.3 * torch.randn(rows, generator=g, device="cuda")),
        gamma,
    )

    got = run_backward(flint, t)

    dgamma_ref, dbeta_ref = fp64_parameter_gradients(t)
    for name, out, ref in (("dgamma", got.dgamma, dgamma_ref), ("dbeta", got.dbeta, dbeta_ref)):
        scale = max(ref.abs().max().item(), 1.0)
        assert (out.double() - ref).abs().max().item() <= 1e-5 * scale, name
    sample = torch.tensor(sorted({0, 1, 7, 8, 255, 256, 257, 2047, 2048, 8191, rows // 2, rows - 9, rows - 8,
                                  rows - 2, rows - 1}), device="cuda")
    case = Tensors(*(v[sample].cpu() if v.ndim and v.shape[0] == rows else v.cpu() for v in t))
    assert_same_bits(got.dx[sample], emulate_dx(*case), "dx rows")
    assert bool(torch.isfinite(got.dx).all())


# -- Argument errors ----------------------------------------------------------------------------------

NAMES = ("dx", "dgamma", "dbeta", "workspace", "dy", "x", "mean", "rstd", "gamma")
OUTPUT_NAMES = NAMES[:4]
ROWS, COLS = 40, 24
BYTES = {"dx": ROWS * COLS * 4, "dgamma": COLS * 4, "dbeta": COLS * 4, "workspace": COLS * 8,
         "dy": ROWS * COLS * 2, "x": ROWS * COLS * 4, "mean": ROWS * 4, "rstd": ROWS * 4, "gamma": COLS * 4}
SPACING = 16384
ARENA_SLOT = {name: (i + 2) * SPACING for i, name in enumerate(NAMES)}
FILL = 0x5A
WORKSPACE_NEEDED = BYTES["workspace"]
NAME = "flint_layernorm_backward"


def new_arena():
    return torch.full(((len(NAMES) + 4) * SPACING,), FILL, dtype=torch.uint8, device="cuda")


def addresses(arena):
    return {name: arena.data_ptr() + ARENA_SLOT[name] for name in NAMES}


def call(flint, a, rows=ROWS, cols=COLS, workspace_bytes=WORKSPACE_NEEDED, accumulate_dx=False,
         accumulate_params=False):
    flint.layernorm_backward(a["dx"], a["dgamma"], a["dbeta"], a["dy"], a["x"], a["mean"], a["rstd"],
                             a["gamma"], rows, cols, accumulate_dx=accumulate_dx,
                             accumulate_params=accumulate_params, workspace=a["workspace"],
                             workspace_bytes=workspace_bytes)


def expect_invalid_argument(flint, reason, arena, a, **kwargs):
    """The call fails with the status and message naming the problem, and nothing in the arena
    changed."""
    before = arena.clone()
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, a, **kwargs)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in (NAME, reason):
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
    # 2 bytes is as far off as an FP16 goes, 4 as far as an FP32, and 8 is aligned for 64 bits but
    # not 128.
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
    # Per buffer: null, then alignment, then the address space; in the order dx, dgamma, dbeta,
    # workspace, dy, x, mean, rstd, gamma; and all of that before any overlap.
    arena = new_arena()
    a = addresses(arena)
    for i, name in enumerate(NAMES[:-1]):
        later = NAMES[i + 1]
        expect_invalid_argument(flint, f"{name} is not 16-byte aligned", arena, {**a, name: a[name] + 4, later: None})
        expect_invalid_argument(flint, f"{name} is null", arena, {**a, name: None, later: a[later] + 4})
        expect_invalid_argument(flint, f"{name} runs past the end", arena, {**a, name: 2**64 - 16, later: None})
    expect_invalid_argument(flint, "gamma is null", arena, {**a, "dx": a["dgamma"], "gamma": None})


@pytest.mark.parametrize(
    "cols", [0, 1, 4, 7, 9, 12, 20, 25, MAX_COLS + 1, MAX_COLS + 8, 2**32, 2**64 - 8, 2**64 - 1], ids=str
)
def test_cols_must_be_a_multiple_of_8_from_8_to_16776960(flint, cols):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"cols ({cols}) must be a multiple of 8 from 8 to {MAX_COLS}", arena, addresses(arena), cols=cols
    )


@pytest.mark.parametrize("rows", [0, MAX_ROWS + 1, MAX_ROWS + 8, 2**40, 2**63, 2**64 - 1], ids=str)
def test_rows_must_be_from_1_to_17179869176(flint, rows):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"rows ({rows}) must be from 1 to {MAX_ROWS}", arena, addresses(arena), rows=rows
    )


def test_size_limits_are_checked_in_order(flint):
    # cols, then rows, then the workspace's size, then the pointers (null ones included).
    arena = new_arena()
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(flint, "cols (7) must be", arena, nothing, cols=7, rows=0, workspace_bytes=0)
    expect_invalid_argument(flint, "cols (0) must be", arena, nothing, cols=0, rows=2**64 - 1)
    expect_invalid_argument(flint, "rows (0) must be", arena, nothing, rows=0, workspace_bytes=0)
    expect_invalid_argument(flint, f"needs {WORKSPACE_NEEDED} bytes of workspace, given 0", arena, nothing, workspace_bytes=0)
    expect_invalid_argument(flint, "dx is null", arena, nothing)


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
    # The limits themselves aren't errors of size: the call gets on to the workspace's size (and
    # with enough bytes claimed, on to a null dx), even where rows * cols * 4 is 1e18 bytes.
    arena = new_arena()
    needed = bg.expected_workspace(rows, cols)
    expect_invalid_argument(
        flint, f"needs {needed} bytes of workspace, given {WORKSPACE_NEEDED}", arena, addresses(arena), rows=rows, cols=cols
    )
    expect_invalid_argument(
        flint, "dx is null", arena, {**addresses(arena), "dx": None}, rows=rows, cols=cols, workspace_bytes=needed
    )


def overlapping_starts(first, second):
    """Where `second` starts relative to `first`, 16-byte aligned, so that they overlap: at the same
    address, 16 bytes in, 16 bytes before first's end, and (negative) with second ending 16 bytes
    into first."""
    a, b = BYTES[first], BYTES[second]
    return {"same-start": 0, "16-in": 16, "16-before-end": a - 16, "ends-16-in": 16 - b}


def touching_starts(first, second):
    """Where `second` starts so that it touches `first` without overlapping it: just after, and
    just before."""
    return {"just-after": BYTES[first], "just-before": -BYTES[second]}


# Every pair with an output in it, in the order of the argument list (outputs first): the checked
# ones. Pairs of inputs may overlap.
PAIRS = [(a, b) for i, a in enumerate(OUTPUT_NAMES) for b in NAMES[i + 1 :]]
OVERLAP_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in PAIRS
    for label, offset in overlapping_starts(a, b).items()
]
TOUCHING_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in PAIRS
    for label, offset in touching_starts(a, b).items()
]


def test_there_are_26_pairs_with_an_output():
    assert len(PAIRS) == 26 and all(a in OUTPUT_NAMES for a, _ in PAIRS)


@pytest.mark.parametrize("first, second, label, offset", OVERLAP_CASES)
def test_each_output_may_not_overlap_any_other_buffer(flint, first, second, label, offset):
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no harm
    # to anything else.
    arena = new_arena()
    a = addresses(arena)
    a[second] = a[first] + offset
    expect_invalid_argument(flint, f"{first} and {second} overlap", arena, a)


@pytest.mark.parametrize("first, second, label, offset", TOUCHING_CASES)
def test_buffers_next_to_each_other_are_not_overlapping(flint, first, second, label, offset):
    arena = new_arena()
    start = dict(ARENA_SLOT)
    start[second] = start[first] + offset
    case = Tensors(
        cpu_dy_kind("spread", ROWS, COLS, 120), cpu_x("normal", ROWS, COLS, 121), *cpu_stats(ROWS, 122),
        cpu_gamma(COLS, 123),
    )
    for name, tensor in zip(Tensors._fields, case):
        data = tensor.cuda().contiguous().view(torch.uint8).flatten()
        arena[start[name] : start[name] + BYTES[name]] = data
    used = torch.zeros(arena.numel(), dtype=torch.bool, device="cuda")
    for name in OUTPUT_NAMES:
        used[start[name] : start[name] + BYTES[name]] = True
    outside_before = arena[~used].clone()
    base = arena.data_ptr()
    a = {name: base + start[name] for name in NAMES}

    call(flint, a)
    torch.cuda.synchronize()

    assert torch.equal(arena[~used], outside_before), "something outside the outputs and the workspace changed"
    expected = emulate_all(case)
    dx = arena[start["dx"] : start["dx"] + BYTES["dx"]].view(torch.float32).reshape(ROWS, COLS)
    dgamma = arena[start["dgamma"] : start["dgamma"] + BYTES["dgamma"]].view(torch.float32)
    dbeta = arena[start["dbeta"] : start["dbeta"] + BYTES["dbeta"]].view(torch.float32)
    assert_same_bits(dx, expected.dx, f"dx {label}")
    assert_same_bits(dgamma, expected.dgamma, f"dgamma {label}")
    assert_same_bits(dbeta, expected.dbeta, f"dbeta {label}")


def test_the_inputs_may_overlap_each_other(flint):
    # dy, x, mean, rstd and gamma are only read: they may share memory (here all at one address,
    # whatever the values), and the call goes ahead.
    arena = new_arena()
    a = addresses(arena)
    for name in ("x", "mean", "rstd", "gamma"):
        a[name] = a["dy"]
    used = torch.zeros(arena.numel(), dtype=torch.bool, device="cuda")
    for name in OUTPUT_NAMES:
        used[ARENA_SLOT[name] : ARENA_SLOT[name] + BYTES[name]] = True
    outside_before = arena[~used].clone()

    call(flint, a)
    torch.cuda.synchronize()

    assert torch.equal(arena[~used], outside_before), "something outside the outputs changed"


def test_an_error_leaves_nothing_behind_and_a_valid_call_works_after(flint):
    arena = new_arena()
    a = addresses(arena)
    case = Tensors(
        cpu_dy_kind("spread", ROWS, COLS, 124), cpu_x("normal", ROWS, COLS, 125), *cpu_stats(ROWS, 126),
        cpu_gamma(COLS, 127),
    )
    for name, tensor in zip(Tensors._fields, case):
        arena[ARENA_SLOT[name] : ARENA_SLOT[name] + BYTES[name]] = tensor.cuda().contiguous().view(torch.uint8).flatten()
    for kwargs in (dict(cols=7), dict(cols=0), dict(rows=0), dict(rows=MAX_ROWS + 1), dict(cols=2**64 - 1),
                   dict(workspace_bytes=0)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, a, **kwargs)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "dx": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "dy": a["dy"] + 2})
    torch.cuda.synchronize()
    for name in OUTPUT_NAMES:
        assert all_equal(arena[ARENA_SLOT[name] : ARENA_SLOT[name] + BYTES[name]], FILL)

    call(flint, a)
    torch.cuda.synchronize()

    expected = emulate_all(case)
    dx = arena[ARENA_SLOT["dx"] : ARENA_SLOT["dx"] + BYTES["dx"]].view(torch.float32).reshape(ROWS, COLS)
    assert_same_bits(dx, expected.dx, "dx after errors")


# -- The workspace function's arguments -----------------------------------------------------------------


def workspace_status(flint, rows, cols):
    """The raw status of flint_layernorm_backward_workspace and the size it left (a sentinel if it
    wrote nothing)."""
    out = ctypes.c_size_t(0xDEADBEEF)
    status = flint.lib.flint_layernorm_backward_workspace(rows, cols, ctypes.byref(out))
    return status, out.value


@pytest.mark.parametrize("cols", [0, 1, 4, 7, 9, 25, MAX_COLS + 1, MAX_COLS + 8, 2**32, 2**64 - 8, 2**64 - 1], ids=str)
def test_the_workspace_function_checks_cols(flint, cols):
    status, size = workspace_status(flint, 256, cols)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert size == 0xDEADBEEF, "bytes was written on an error"
    for text in ("flint_layernorm_backward_workspace", f"cols ({cols}) must be a multiple of 8 from 8 to {MAX_COLS}"):
        assert text in flint.last_error()
    with pytest.raises(flint_lib.FlintError, match="must be a multiple of 8"):
        flint.layernorm_backward_workspace(256, cols)


@pytest.mark.parametrize("rows", [0, MAX_ROWS + 1, MAX_ROWS + 8, 2**40, 2**63, 2**64 - 1], ids=str)
@pytest.mark.parametrize("cols", [8, 768, MAX_COLS], ids=str)
def test_the_workspace_function_checks_rows(flint, rows, cols):
    status, size = workspace_status(flint, rows, cols)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert size == 0xDEADBEEF, "bytes was written on an error"
    for text in ("flint_layernorm_backward_workspace", f"rows ({rows}) must be from 1 to {MAX_ROWS}"):
        assert text in flint.last_error()


def test_the_workspace_function_checks_cols_before_rows(flint):
    status, _ = workspace_status(flint, 0, 7)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "cols (7) must be" in flint.last_error()


@pytest.mark.parametrize("rows, cols", [(256, 8), (0, 8), (256, 7), (0, 7), (2**64 - 1, 2**64 - 1)], ids=str)
def test_the_workspace_function_needs_a_bytes_pointer(flint, rows, cols):
    # The null check comes first, before any size is looked at.
    status = flint.lib.flint_layernorm_backward_workspace(rows, cols, None)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_layernorm_backward_workspace: bytes is null" in flint.last_error()


def test_the_workspace_function_accepts_the_limits(flint):
    for rows, cols in ((MAX_ROWS, MAX_COLS), (1, 8), (MAX_ROWS, 8), (1, MAX_COLS)):
        status, size = workspace_status(flint, rows, cols)

        assert status == flint_lib.OK
        assert size == bg.expected_workspace(rows, cols)
