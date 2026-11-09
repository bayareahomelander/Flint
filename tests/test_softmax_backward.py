"""flint_causal_softmax_backward: attention's softmax, backward. p (the forward pass's output), dp
(its gradient) and ds are FP16 batches x length x length, row-major and contiguous. Row t of a
batch sees its columns c <= t, and ds[b, t, c] = scale * p * (dp - D) there, with D the sum of
p * dp over the visible columns, and +0 in the rest, whatever p and dp hold there. Every operation
is an IEEE FP32 round-to-nearest one, and the order depends only on t, so the bits are the same on
every GPU:

  - A chunk is 8 consecutive columns. Row t's visible chunks are i = 0 .. t // 8, and chunk i's
    visible columns are 8i + j <= t. Lane l (of 32) takes the visible chunks l, l + 32, ... in
    order, and within a chunk the columns j = 0 .. 7 in order (the visible ones only).
  - Each lane holds a pair (s, e), both starting at +0. For each visible value v = float32(p) *
    float32(dp) (exact in FP32), TwoSum adds it in this order: t_ = s + v; vp = t_ - s;
    e = e + ((s - (t_ - vp)) + (v - vp)); s = t_.
  - Then a butterfly over ALL 32 lanes, at offsets 16, 8, 4, 2, 1: each lane merges its xor-partner's
    pair from before the step: errs = a.e + b.e; a.e = +0; TwoSum adds b.s into a; a.e = a.e + errs.
    (Lanes without chunks hold (+0, +0) and still merge.) All lanes end equal; lane 0's (s, e) is D.
    If s is not finite, e = +0.
  - Then for each visible column: diff = (dp - s) - e (two FP32 subtractions, each rounded); out =
    (p * diff) * scale (two FP32 multiplies, each rounded); ds = out rounded to FP16, to nearest
    even. Masked columns c > t get +0 (bits 0).

A numpy float32 emulation of exactly that (emulate below; numpy keeps subnormals) must equal
Flint's output bit for bit, NaNs compared as NaN. It is checked first against a literal scalar
transcription of the list above (one np.float32 number at a time, every merge included), non-finite
values among the inputs. The inputs of the bit-exact tests are made on the CPU with a seeded
generator and copied over, so they are the same on every machine.

What the tests cover:
  - the emulation against the scalar transcription, non-finite inputs included;
  - Flint against the emulation bit for bit, for lengths 8, 16, 64, 264, 1016 and 1024 (rows with
    fewer chunks than lanes, exactly 32, more than 32 and an uneven number per lane), several
    batches (4 x 1024 x 1024 among them), for P from Flint's forward pass on scores of six kinds
    and for arbitrary FP16 p in [0, 1] (which don't sum to 1), for five kinds of dp (N(0, 1),
    gradient scale, an offset that cancels against D, outliers in 1% of the columns, the whole
    range with subnormals) and scales 1/8, 1, 0.3 and -0.5; and which changes of the arithmetic
    (plain FP32 sums, D added up before the subtraction, the scale applied to p first) the
    comparison can tell apart;
  - the correctness gate (PLAN.md): the reference is the formula in FP64 on the upcast FP16 p and
    dp, the peer PyTorch's FP16 torch._softmax_backward_data times the scale in FP16, with factor
    3, 3 seeds and the FP16 floors, at every length class and GPT-2's whole 192 x 1024 x 1024;
  - a tighter bound than the gate's floors: every visible output within 1 FP16 ulp of the FP64
    result rounded correctly to FP16, and at least 99.9% equal to it;
  - exact cases: +0 by bits in the upper triangle whatever p and dp hold (NaN, infs, random
    patterns), row 0 (one column), in place (ds is dp) and p and dp the same buffer, a row's
    result independent of its batch, the other rows and the matrix's length, scale 0, and overflow
    to inf at FP16's limit;
  - non-finite values in p or dp at columns of different lanes: bit for bit against the emulation,
    the other rows those of a clean run, and NaN and inf places equal to an FP32 PyTorch
    computation of the same formula;
  - guard bands around ds, every element written, inputs unchanged, determinism, streams, the
    limits (65,535 batches of length 8; length 1,024), and every argument error;
  - the chain S = Q K^T / 8, P, dP = dO V^T, dS (this kernel, scale 1/8), dQ = dS K and dK = dS^T Q
    against PyTorch's FP64 autograd of causal attention, with PyTorch's FP16 path as the peer.

Memory: GPT-2's whole matrix is 402 MB for each of p, dp and ds, so the references and the errors
run over chunks of batches. Under the sanitizer (scripts/sanitize.py) the shapes shrink to T at most
128 and few batches, and the tests of the limits are skipped."""

import gc
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator, normal, round_to_half
from test_bias_gradient import two_sum_add, two_sum_merge
from test_softmax import (  # noqa: F401  (free_gpu_memory is an autouse fixture)
    CHUNK_ELEMENTS,
    GUARD_AFTER,
    HEAD_WIDTH,
    LONG,
    MAX_BATCHES,
    MAX_LENGTH,
    MIN_SHARE,
    MIN_SHARE_COUNT,
    NAN_BITS,
    NONFINITE_BITS,
    ODD_LONG,
    OTHER_BITS,
    SENTINEL,
    Accumulator,
    all_equal,
    any_unequal,
    batch_chunks,
    causal_mask,
    fp16_from_bits,
    free_gpu_memory,
    injection_column,
    none_equal,
    pytorch_attention,
    random_bits,
    shape_id,
    shape_params,
)

F32 = np.float32
LANES = np.arange(32)
BUTTERFLY = (16, 8, 4, 2, 1)
SCALES = (0.125, 1.0, 0.3, -0.5)
GRADIENT_SCALE = 0.125  # attention's scale for heads of 64
NAME = "flint_causal_softmax_backward"

# Elements of the emulation's temporaries per batch of rows (small enough to stay in the cache).
EMULATION_ELEMENTS = 1 << 22


# -- The emulation of the kernel's arithmetic ----------------------------------------------------


def numpy_of(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


def emulate(p, dp, scale, *, compensated=True, add_first=False, scale_first=False):
    """The kernel's ds for the module docstring's arithmetic, from FP16 p and dp (batches x t x t,
    numpy or torch on the CPU), as a numpy float16 array. Vectorized over rows. compensated=False
    (plain FP32 sums: no errors), add_first (diff = dp - (s + e)) and scale_first (out = (scale *
    p) * diff) are not the kernel's, and exist to show what the bit-exact tests can tell apart."""
    p, dp = numpy_of(p), numpy_of(dp)
    b, t, _ = p.shape
    rows = b * t
    p, dp = p.reshape(rows, t), dp.reshape(rows, t)
    chunks = t // 8
    steps = -(-chunks // 32)
    padded = steps * 256
    scale = F32(scale)
    out = np.zeros((rows, t), dtype=np.float16)
    batch = max(1, EMULATION_ELEMENTS // padded)
    with np.errstate(all="ignore"):
        for r0 in range(0, rows, batch):
            r1 = min(rows, r0 + batch)
            n = r1 - r0
            row_t = (np.arange(r0, r1) % t)[:, None]
            visible = np.arange(t)[None, :] <= row_t
            p32, dp32 = p[r0:r1].astype(F32), dp[r0:r1].astype(F32)
            # The products of the visible columns (invisible ones are skipped by the kernel,
            # which adding +0 stands in for), padded with +0 to whole rounds of 32 chunks.
            v = np.zeros((n, padded), dtype=F32)
            v[:, :t] = np.where(visible, p32 * dp32, F32(0.0))
            # (n, 32, steps * 8): lane l takes chunks l, l + 32, ... in order, columns 0 to 7 each.
            lanes = np.ascontiguousarray(
                v.reshape(n, steps, 32, 8).transpose(0, 2, 1, 3).reshape(n, 32, steps * 8)
            )
            s = np.zeros((n, 32), dtype=F32)
            e = np.zeros((n, 32), dtype=F32)
            for k in range(steps * 8):
                if compensated:
                    s, e = two_sum_add(s, e, lanes[:, :, k])
                else:
                    s = s + lanes[:, :, k]
            for offset in BUTTERFLY:
                partner = LANES ^ offset
                if compensated:
                    s, e = two_sum_merge((s, e), (s[:, partner], e[:, partner]))
                else:
                    s = s + s[:, partner]
            s0 = s[:, :1]
            e0 = np.where(np.isfinite(s0), e[:, :1], F32(0.0)) if compensated else np.zeros_like(s0)
            diff = dp32 - (s0 + e0) if add_first else (dp32 - s0) - e0
            value = (scale * p32) * diff if scale_first else (p32 * diff) * scale
            out[r0:r1] = np.where(visible, value, F32(0.0)).astype(np.float16)
    return out.reshape(b, t, t)


def scalar_backward(p, dp, scale, rows=None):
    """The module docstring's arithmetic transcribed literally, one np.float32 number at a time,
    every merge included (lanes without chunks too). Slow, so for small shapes or a sample of rows
    (the default is all). Returns ds for those rows as a float16 array (batches, len(rows), t), and
    whether every row with a finite total had all 32 lanes end with the same bits."""
    f = F32
    zero = f(0.0)
    b, t, _ = p.shape
    rows = list(range(t)) if rows is None else list(rows)
    scale = f(scale)
    out = np.zeros((b, len(rows), t), dtype=np.float16)
    lanes_equal = True
    with np.errstate(all="ignore"):
        for bi in range(b):
            for ri, row in enumerate(rows):
                sums, errors = [], []
                for lane in range(32):
                    s, e = zero, zero
                    for chunk in range(lane, row // 8 + 1, 32):
                        for j in range(8):
                            c = 8 * chunk + j
                            if c > row:
                                continue
                            v = f(f(p[bi, row, c]) * f(dp[bi, row, c]))
                            t_ = f(s + v)
                            vp = f(t_ - s)
                            e = f(e + f(f(s - f(t_ - vp)) + f(v - vp)))
                            s = t_
                    sums.append(s)
                    errors.append(e)
                for offset in BUTTERFLY:
                    new_sums, new_errors = [], []
                    for lane in range(32):
                        a_s, a_e = sums[lane], errors[lane]
                        b_s, b_e = sums[lane ^ offset], errors[lane ^ offset]
                        errs = f(a_e + b_e)
                        a_e = zero
                        t_ = f(a_s + b_s)
                        vp = f(t_ - a_s)
                        a_e = f(a_e + f(f(a_s - f(t_ - vp)) + f(b_s - vp)))
                        a_s = t_
                        a_e = f(a_e + errs)
                        new_sums.append(a_s)
                        new_errors.append(a_e)
                    sums, errors = new_sums, new_errors
                s, e = sums[0], errors[0]
                if not np.isfinite(s):
                    e = zero
                elif any(
                    sums[lane].view(np.uint32) != s.view(np.uint32)
                    or errors[lane].view(np.uint32) != e.view(np.uint32)
                    for lane in range(1, 32)
                ):
                    lanes_equal = False
                for c in range(row + 1):
                    diff = f(f(f(dp[bi, row, c]) - s) - e)
                    value = f(f(f(p[bi, row, c]) * diff) * scale)
                    out[bi, ri, c] = np.float16(value)
    return out, lanes_equal


# -- Inputs -------------------------------------------------------------------------------------

S_KINDS = ["s1", "qk", "s4", "s16", "s30", "outliers"]
S_STD = {"s1": 1.0, "s4": 4.0, "s16": 16.0, "s30": 30.0}
P_KINDS = S_KINDS + ["uniform"]
DP_KINDS = ["normal", "gradient", "offset", "outliers", "range"]


def rng(seed, device):
    return torch.Generator(device=device).manual_seed(seed)


def make_scores(flint, kind, b, t, g, device):
    """S as FP16 on `device`, from the generator g (on that device): N(0, std^2) for the s*
    kinds, Q K^T / 8 as the batched NT GEMM makes it ("qk", Q and K about N(0, 1)), and "outliers"
    (N(0, 1) with 1% of the values replaced by +-3,000 (+-10%), which make one-hot rows)."""
    shape = (b, t, t)
    if kind in S_STD:
        return (torch.randn(shape, generator=g, device=device) * S_STD[kind]).half()
    if kind == "qk":
        q = torch.randn((b, t, HEAD_WIDTH), generator=g, device=device).half().cuda()
        k = torch.randn((b, t, HEAD_WIDTH), generator=g, device=device).half().cuda()
        s = torch.empty(shape, dtype=torch.float16, device="cuda")
        flint.gemm_nt_batched(s, q, k, b, t, t, HEAD_WIDTH, scale=0.125)
        return s.to(device)
    if kind == "outliers":
        hit = torch.rand(shape, generator=g, device=device) < 0.01
        sign = torch.randint(0, 2, shape, generator=g, device=device) * 2.0 - 1.0
        jitter = 0.9 + 0.2 * torch.rand(shape, generator=g, device=device)
        return torch.where(hit, 3000.0 * sign * jitter, torch.randn(shape, generator=g, device=device)).half()
    raise ValueError(kind)


def forward_p(flint, s):
    """Flint's forward pass on S (any device): P on the GPU."""
    s = s.cuda()
    b, t, _ = s.shape
    p = torch.empty_like(s)
    flint.causal_softmax_forward(p, s, b, t)
    return p


def subnormals(shape, g, device):
    """Random FP16 subnormals of both signs."""
    bits = torch.randint(1, 0x400, shape, generator=g, device=device)
    sign = torch.randint(0, 2, shape, generator=g, device=device)
    return (bits - sign * 0x8000).short().view(torch.float16)


def make_p(flint, kind, b, t, g, device):
    """P as FP16 on `device`: Flint's forward pass on scores of a kind, or "uniform": arbitrary
    values in [0, 1] (squares of uniform ones, so the exponents vary) that don't sum to 1, with 10%
    zeros and 3% positive subnormals."""
    shape = (b, t, t)
    if kind != "uniform":
        return forward_p(flint, make_scores(flint, kind, b, t, g, device)).to(device)
    u = torch.rand(shape, generator=g, device=device)
    p = (u * u).half()
    zero = torch.zeros((), dtype=torch.float16, device=device)
    p = torch.where(torch.rand(shape, generator=g, device=device) < 0.1, zero, p)
    tiny = torch.randint(1, 0x400, shape, generator=g, device=device).short().view(torch.float16)
    return torch.where(torch.rand(shape, generator=g, device=device) < 0.03, tiny, p)


def make_dp(kind, b, t, g, device):
    """dP as FP16 on `device`: "normal" N(0, 1); "gradient" N(0, 1e-4^2); "offset" 2 + 0.01 N(0, 1)
    (D is about 2 for a P that sums to 1, so dp - D cancels); "outliers" N(0, 1) with std 3,000 in
    1% of the columns (at least one per batch); "range" uniform in +-60,000 with 10% subnormals."""
    shape = (b, t, t)
    r = torch.randn(shape, generator=g, device=device)
    if kind == "normal":
        return r.half()
    if kind == "gradient":
        return (r * 1e-4).half()
    if kind == "offset":
        return (2.0 + 0.01 * r).half()
    if kind == "outliers":
        columns = torch.rand((b, 1, t), generator=g, device=device) < 0.01
        columns[torch.arange(b, device=device), 0, torch.randint(0, t, (b,), generator=g, device=device)] = True
        return (r * torch.where(columns, 3000.0, 1.0)).half()
    if kind == "range":
        values = ((torch.rand(shape, generator=g, device=device) * 2 - 1) * 60000.0).half()
        return torch.where(torch.rand(shape, generator=g, device=device) < 0.1, subnormals(shape, g, device), values)
    raise ValueError(kind)


def make_case(flint, p_kind, dp_kind, b, t, seed, device):
    """(p, dp) on `device`, from a generator seeded with `seed` on that device."""
    g = rng(seed, device)
    p = make_p(flint, p_kind, b, t, g, device)
    return p, make_dp(dp_kind, b, t, g, device)


def inject_non_finite(p, dp, seed):
    """Copies of p and dp (numpy float16) with about one value per row replaced by NaN, +inf or
    -inf, at any column, masked ones included."""
    g = np.random.default_rng(seed)
    t = p.shape[-1]
    p, dp = p.copy(), dp.copy()
    for array in (p, dp):
        hit = g.random(array.shape) < 1.0 / t
        values = np.array([np.nan, np.inf, -np.inf], dtype=np.float16)[g.integers(0, 3, array.shape)]
        array[hit] = values[hit]
    return p, dp


# -- Comparing and running Flint --------------------------------------------------------------


def half_bits(x):
    """An FP16 tensor (any device) as int16 bits on the CPU, with every NaN the same one (a NaN's
    payload isn't pinned: only that the result is a NaN)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.cpu().contiguous()
    bits = x.view(torch.int16).clone()
    bits[torch.isnan(x)] = 0x7E00
    return bits


def assert_same_halves(actual, expected, what):
    """The FP16 tensors (or numpy float16 arrays) are equal bit for bit, NaNs aside. torch.equal,
    not `==`: pytest hangs explaining a failed tensor assert."""
    if isinstance(expected, np.ndarray):
        expected = torch.from_numpy(expected)
    if isinstance(actual, np.ndarray):
        actual = torch.from_numpy(actual)
    a, e = half_bits(actual), half_bits(expected)
    assert a.shape == e.shape, f"{what}: shapes {tuple(a.shape)} and {tuple(e.shape)}"
    if torch.equal(a, e):
        return
    bad = (a != e).nonzero()
    first = tuple(bad[0].tolist())
    raise AssertionError(
        f"{what}: {len(bad)} of {a.numel()} halves differ, the first at {first}: "
        f"{actual.cpu()[first].item()!r} against {expected.cpu()[first].item()!r}"
    )


def count_different(actual, expected):
    return int((half_bits(actual) != half_bits(expected)).sum())


def snapshot(x):
    """A copy of an FP16 tensor, or for a big one (past 2^24 elements) its sum as integers."""
    return x.clone() if x.numel() <= 1 << 24 else x.view(torch.int16).sum(dtype=torch.int64)


def same_as_snapshot(x, before):
    if before.dtype == torch.int64:
        return bool(x.view(torch.int16).sum(dtype=torch.int64) == before)
    return torch.equal(x.view(torch.int16), before.view(torch.int16))


def flint_backward(flint, p, dp, scale, offset=8, fill=SENTINEL, stream=None, in_place=False):
    """Runs the kernel into ds inside a buffer pre-filled with `fill` (a 16-bit pattern), `offset`
    halves in (a multiple of 8 keeps it 16-byte aligned). With in_place, ds is dp's own memory (dp is
    copied into a guarded buffer first). Checks that the buffer around ds is untouched and that p
    (and dp, unless in place) are unchanged, and returns ds as batches x t x t FP16 (a view)."""
    b, t, _ = p.shape
    count = p.numel()
    buf = torch.full((offset + count + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    p_before, dp_before = snapshot(p), snapshot(dp)
    if in_place:
        ds = buf[offset : offset + count].view(torch.float16).view(b, t, t)
        ds.copy_(dp)
        flint.causal_softmax_backward(ds, p, ds, b, t, scale, stream=stream)
    else:
        ds = buf[offset : offset + count]
        flint.causal_softmax_backward(ds, p, dp, b, t, scale, stream=stream)
    torch.cuda.synchronize()
    assert all_equal(buf[:offset], guard), f"{NAME} wrote before ds"
    assert all_equal(buf[offset + count :], guard), f"{NAME} wrote past ds"
    assert same_as_snapshot(p, p_before), "p changed"
    if not in_place:
        assert same_as_snapshot(dp, dp_before), "dp changed"
    return buf[offset : offset + count].view(torch.float16).view(b, t, t)


def to_cuda_run(flint, p, dp, scale, **kwargs):
    """flint_backward on CPU p and dp, ds back on the CPU."""
    return flint_backward(flint, p.cuda(), dp.cuda(), scale, **kwargs).cpu()


# -- The emulation against the scalar transcription -----------------------------------------------

SCALAR_SHAPES = [(2, 8), (2, 16), (1, 24), (2, 72), (1, 264)]
# Rows of the long matrices that the scalar transcription runs: the first, chunk boundaries, the
# row where lane 0 gets its second and third chunk, and the last.
SAMPLED_ROWS = {
    264: [0, 1, 7, 8, 9, 63, 64, 127, 255, 256, 257, 263],
    1016: [0, 5, 8, 100, 255, 256, 257, 511, 512, 1000, 1015],
    1024: [0, 7, 255, 256, 263, 264, 513, 777, 1000, 1022, 1023],
}
SCALAR_CASES = [
    pytest.param(b, t, dk, nonfinite, id=f"{b}x{t}-{dk}-{'non-finite' if nonfinite else 'finite'}")
    for (b, t) in SCALAR_SHAPES
    for dk in DP_KINDS
    for nonfinite in (False, True)
] + [
    pytest.param(1, t, dk, nonfinite, id=f"1x{t}-{dk}-{'non-finite' if nonfinite else 'finite'}")
    for t, dk, nonfinite in [(1016, "normal", False), (1016, "offset", True), (1024, "range", False), (1024, "outliers", True)]
]


@pytest.mark.parametrize("b, t, dp_kind, nonfinite", SCALAR_CASES)
def test_the_emulation_equals_the_scalar_transcription(flint, b, t, dp_kind, nonfinite):
    g = rng(11, "cpu")
    p = make_p(flint, "uniform", b, t, g, "cpu").numpy()
    dp = make_dp(dp_kind, b, t, g, "cpu").numpy()
    if nonfinite:
        p, dp = inject_non_finite(p, dp, seed=t + b)
    scale = SCALES[(t + len(dp_kind)) % 4]
    rows = SAMPLED_ROWS.get(t)

    expected, lanes_equal = scalar_backward(p, dp, scale, rows)

    emulated = emulate(p, dp, scale)
    if rows is not None:
        emulated = emulated[:, rows]
    assert_same_halves(emulated, expected, "the emulation against the scalar transcription")
    if not nonfinite:
        assert lanes_equal, "the 32 lanes do not all end with the same pair"


# -- Flint against the emulation, bit for bit -----------------------------------------------------

BIT_SHAPES = [(3, 8), (3, 16), (3, 64), (3, 264), (2, 1016), (4, 1024)]
BIT_SHAPES = list(dict.fromkeys((min(b, 3) if SANITIZER else b, min(t, LONG)) for b, t in BIT_SHAPES))
BIT_COMBOS = list(itertools.product(P_KINDS, DP_KINDS))
if SANITIZER:
    BIT_COMBOS = BIT_COMBOS[::4]
# Every (kind of p, kind of dp) pair at every shape, the scales in turn, so the 4 x 1024 x 1024
# shape meets every kind of each and every scale.
BIT_CASES = [
    pytest.param(pk, dk, *shape, SCALES[(i + j) % 4], id=f"{pk}-{dk}-{shape_id(shape)}-{SCALES[(i + j) % 4]}")
    for i, (pk, dk) in enumerate(BIT_COMBOS)
    for j, shape in enumerate(BIT_SHAPES)
]


@pytest.mark.parametrize("p_kind, dp_kind, b, t, scale", BIT_CASES)
def test_flint_equals_the_emulation_bit_for_bit(flint, p_kind, dp_kind, b, t, scale):
    p, dp = make_case(flint, p_kind, dp_kind, b, t, seed=3 + t, device="cpu")

    ds = to_cuda_run(flint, p, dp, scale)

    assert_same_halves(ds, emulate(p, dp, scale), f"{p_kind} / {dp_kind} {b}x{t} scale {scale}")


# -- What the bit-exact comparison can tell apart ------------------------------------------------

# A change of the arithmetic, the input and the scale it shows on. The offset input is dp = 2 +
# 0.01 N(0, 1) under P: dp - D is about 0.01, and D rounded to FP32 (a plain sum, or the pair added
# up first) loses the bits that decide the FP16 result. A scale that isn't a power of 2 shows the
# order of the two multiplications.
DISTINGUISHABLE = [
    pytest.param(dict(compensated=False), "qk", "offset", 0.125, id="plain-FP32-sums"),
    pytest.param(dict(add_first=True), "qk", "offset", 0.125, id="D-added-up-first"),
    pytest.param(dict(scale_first=True), "qk", "normal", 0.3, id="scale-applied-to-p-first"),
]


@pytest.mark.parametrize("change, p_kind, dp_kind, scale", DISTINGUISHABLE)
def test_the_bit_exact_comparison_tells_a_changed_arithmetic_apart(flint, change, p_kind, dp_kind, scale):
    b, t = (2, 1024) if not SANITIZER else (2, LONG)
    p, dp = make_case(flint, p_kind, dp_kind, b, t, seed=5, device="cpu")

    ds = to_cuda_run(flint, p, dp, scale)

    assert_same_halves(ds, emulate(p, dp, scale), "the kernel's own arithmetic")
    different = count_different(ds, emulate(p, dp, scale, **change))
    print(f"softmax backward {change}: {different} of {ds.numel()} halves differ")
    assert different > 0, f"{change} gives Flint's bits on {p_kind} / {dp_kind} scale {scale}"


# -- Order of the sums, on crafted rows -----------------------------------------------------------
# On ordinary rows the compensated pair (s, e) holds D so well that another order of the same
# additions gives the same ds, so the bit-exact tests above can't see the order. On these rows
# they can: chunks come in pairs (2k, 2k + 1) with the same p and opposite dp near +-60,000,
# so their products cancel exactly, and even chunks (lane l takes chunks l, l + 32, ..., so even
# lanes) hold the positive ones: the butterfly's offsets 16 to 2 add like signs, to sums near
# 2^25 whose TwoSum errors are near 2, and only offset 1 cancels. Each pair also holds one tiny
# product (p of FP16's subnormal range, dp 1 to 3) in both chunks, which the sums lose into
# errors near 2^-24, so e's own FP32 sums span more than 24 bits and round by the order. And
# each pair has a column with p = 32,768 and dp = 0, whose ds is -4,096 (s + e) at scale 1/8,
# large enough for FP16 to show D's low bits.

CRAFTED_LENGTH = 1024
CRAFTED_ROWS = 4 if SANITIZER else 64
CRAFTED_SEED = 2


def crafted_rows(rows, seed):
    """rows x CRAFTED_LENGTH FP16 p and dp (CPU, from a seeded torch generator, so the same on
    every machine), each row meant as the last row of a matrix, which sees every column."""
    g = torch.Generator().manual_seed(seed)
    n, chunks = CRAFTED_LENGTH, CRAFTED_LENGTH // 8

    def draw(lo, hi):
        return int(torch.randint(lo, hi, (1,), generator=g))

    p = torch.zeros(rows, n, dtype=torch.float16)
    dp = torch.zeros(rows, n, dtype=torch.float16)
    for r in range(rows):
        for k in range(0, chunks, 2):
            big = float(60000 + draw(0, 5505))
            pc = torch.tensor([1.0 - draw(0, 64) * 2.0**-11 for _ in range(8)], dtype=torch.float16)
            dc = torch.full((8,), big, dtype=torch.float16)
            tiny, zero = draw(0, 8), draw(0, 8)
            pc[tiny], dc[tiny] = draw(1, 64) * 2.0**-24, float(draw(1, 4)) * (1.0 if draw(0, 2) else -1.0)
            if zero != tiny:
                pc[zero], dc[zero] = 32768.0, 0.0
            negative = -dc
            negative[tiny] = dc[tiny]  # the same tiny product in both chunks, not cancelled
            p[r, 8 * k : 8 * k + 8], dp[r, 8 * k : 8 * k + 8] = pc, dc
            p[r, 8 * k + 8 : 8 * k + 16], dp[r, 8 * k + 8 : 8 * k + 16] = pc, negative
    return p, dp


def emulate_full_rows(p, dp, scale, lanes, butterfly=BUTTERFLY):
    """ds (float16) of rows that see all their columns, with lane l adding the chunks lanes[l] in
    that order, then the butterfly's offsets in the order given: the kernel's order or another."""
    p32, dp32 = numpy_of(p).astype(F32), numpy_of(dp).astype(F32)
    products = p32 * dp32
    rows = p32.shape[0]
    with np.errstate(all="ignore"):
        pairs = []
        for chunks in lanes:
            s, e = np.zeros(rows, F32), np.zeros(rows, F32)
            for c in chunks:
                for j in range(8):
                    s, e = two_sum_add(s, e, products[:, 8 * c + j])
            pairs.append((s, e))
        for offset in butterfly:
            pairs = [two_sum_merge(pairs[l], pairs[l ^ offset]) for l in range(32)]
        s, e = pairs[0]
        e = np.where(np.isfinite(s), e, F32(0.0))
        diff = (dp32 - s[:, None]) - e[:, None]
        return ((p32 * diff) * F32(scale)).astype(np.float16)


def crafted_orders():
    chunks = CRAFTED_LENGTH // 8
    kernel = [list(range(l, chunks, 32)) for l in range(32)]
    return {
        "the kernel's": (kernel, BUTTERFLY),
        "each lane's chunks last first": ([list(reversed(c)) for c in kernel], BUTTERFLY),
        "the butterfly's offsets 1, 2, 4, 8, 16": (kernel, (1, 2, 4, 8, 16)),
        "lane l taking chunks 4l to 4l + 3": ([[c for c in range(chunks) if c * 32 // chunks == l]
                                               for l in range(32)], BUTTERFLY),
    }


def test_the_crafted_rows_tell_the_orders_of_ds_sum_apart():
    # The kernel's order on the crafted rows is the general emulation's, and each other order of
    # the same additions changes ds in several rows, so the test below pins the order.
    p, dp = crafted_rows(64, CRAFTED_SEED)
    orders = crafted_orders()
    kernel = emulate_full_rows(p, dp, GRADIENT_SCALE, *orders.pop("the kernel's"))
    # The general emulation, on matrices whose last row is each crafted row.
    general = np.stack([
        emulate(torch.cat([torch.zeros(CRAFTED_LENGTH - 1, CRAFTED_LENGTH, dtype=torch.float16), p[r : r + 1]])[None],
                torch.cat([torch.zeros(CRAFTED_LENGTH - 1, CRAFTED_LENGTH, dtype=torch.float16), dp[r : r + 1]])[None],
                GRADIENT_SCALE)[0, -1]
        for r in range(4)
    ])
    assert np.array_equal(general.view(np.int16), kernel[:4].view(np.int16))
    for name, (lanes, butterfly) in orders.items():
        other = emulate_full_rows(p, dp, GRADIENT_SCALE, lanes, butterfly)
        rows = int((other.view(np.int16) != kernel.view(np.int16)).any(1).sum())
        print(f"ORDER crafted {name}: ds differs in {rows} of 64 rows")
        assert rows >= 4, name


def test_flint_equals_the_emulation_on_the_crafted_rows(flint):
    # Each crafted row is the last row of its own matrix; the other rows are zeros.
    p, dp = crafted_rows(CRAFTED_ROWS, CRAFTED_SEED)
    n = CRAFTED_LENGTH
    pm = torch.zeros(CRAFTED_ROWS, n, n, dtype=torch.float16, device="cuda")
    dpm = torch.zeros_like(pm)
    pm[:, n - 1], dpm[:, n - 1] = p.cuda(), dp.cuda()
    ds = torch.empty_like(pm)

    flint.causal_softmax_backward(ds, pm, dpm, CRAFTED_ROWS, n, GRADIENT_SCALE)
    torch.cuda.synchronize()

    lanes, butterfly = crafted_orders()["the kernel's"]
    expected = emulate_full_rows(p, dp, GRADIENT_SCALE, lanes, butterfly)
    assert_same_halves(ds[:, n - 1].cpu(), torch.from_numpy(expected), "ds of the crafted rows")
    assert bool((ds[:, : n - 1].view(torch.int16) == 0).all()), "the zero rows' ds is +0"


# -- The correctness gate -------------------------------------------------------------------------


def reference_ds(p, dp, scale):
    """The formula in FP64 on the upcast FP16 p and dp (batches x t x t, on the GPU), the masked
    columns excluded from D and zero: scale * p * (dp - D). The scale is the FP32 one the kernel
    takes."""
    mask = causal_mask(p.shape[-1])
    pd, dd = p.double(), dp.double()
    d = torch.where(mask, pd * dd, 0.0).sum(-1, keepdim=True)
    return torch.where(mask, float(F32(scale)) * pd * (dd - d), 0.0)


def peer_ds(p, dp, scale):
    """PyTorch's FP16 softmax backward (with dp's masked part set to 0, for the peer only: its
    softmax has no mask) times the scale in FP16."""
    mask = causal_mask(p.shape[-1])
    zero = torch.zeros((), dtype=torch.float16, device="cuda")
    grad = torch._softmax_backward_data(
        torch.where(mask, dp, zero).contiguous(), torch.where(mask, p, zero).contiguous(), -1, torch.float16
    )
    return grad * scale


def ds_errors(ds, p, dp, scale):
    """Flint's and the peer's gate.Errors against the FP64 reference, over chunks of batches."""
    flint_sums, peer_sums = Accumulator(), Accumulator()
    for chunk in batch_chunks(*p.shape[:2]):
        ref = reference_ds(p[chunk], dp[chunk], scale)
        flint_sums.add(ds[chunk], ref)
        peer_sums.add(peer_ds(p[chunk], dp[chunk], scale), ref)
    return flint_sums.errors(), peer_sums.errors()


def check_gate(flint, p_kind, dp_kind, b, t, scale):
    flint_errors, peer_errors = [], []
    for seed in gate.SEEDS:
        p, dp = make_case(flint, p_kind, dp_kind, b, t, seed, "cuda")
        ds = flint_backward(flint, p, dp, scale, offset=8 * (1 + seed))
        f, q = ds_errors(ds, p, dp, scale)
        flint_errors.append(f)
        peer_errors.append(q)
        del p, dp, ds
    bounds = gate.check_gate(gate.GRADIENT_FACTOR, flint_errors, peer_errors)
    flint_rel, peer_rel = max(e.rel_l2 for e in flint_errors), max(e.rel_l2 for e in peer_errors)
    flint_abs, peer_abs = max(e.max_abs for e in flint_errors), max(e.max_abs for e in peer_errors)
    print(
        f"softmax backward gate {p_kind} {dp_kind} {b}x{t} scale {scale}: rel_l2 flint {flint_rel:.3g} "
        f"peer {peer_rel:.3g} ratio {flint_rel / peer_rel if peer_rel else math.nan:.3g}; max_abs flint "
        f"{flint_abs:.3g} peer {peer_abs:.3g} ratio {flint_abs / peer_abs if peer_abs else math.nan:.3g}; "
        f"bound {bounds.rel_l2:.3g}"
    )


# Lengths of 1 chunk, partial lanes' worth, 33 chunks (a lane's second chunk), 127 and 128 chunks
# (4 per lane), and 12 heads of 1,024.
GATE_SHAPES = [(1, 8), (2, 16), (3, 24), (3, 56), (2, 64), (3, 264), (2, 1016), (3, 1024), (12, 1024)]
GATE_SHAPES = list(dict.fromkeys((min(b, 3) if SANITIZER else b, min(t, LONG)) for b, t in GATE_SHAPES))
GATE_COMBOS = list(itertools.product(S_KINDS, DP_KINDS))
if SANITIZER:
    GATE_COMBOS = GATE_COMBOS[::3]


def rotated_cases(offset):
    """Each (S kind, dp kind) pair at one shape of the list, the shape `offset` further along for
    the second rotation, with the scales in turn."""
    cases = []
    for i, (sk, dk) in enumerate(GATE_COMBOS):
        b, t = GATE_SHAPES[(i + offset) % len(GATE_SHAPES)]
        cases.append(pytest.param(sk, dk, b, t, SCALES[i % 4], id=f"{sk}-{dk}-{b}x{t}-{SCALES[i % 4]}"))
    return cases


GATE_CASES = rotated_cases(0)


@pytest.mark.parametrize("p_kind, dp_kind, b, t, scale", GATE_CASES)
def test_passes_the_gate(flint, p_kind, dp_kind, b, t, scale):
    check_gate(flint, p_kind, dp_kind, b, t, scale)


# GPT-2's attention: 16 sequences x 12 heads = 192 batches of T = 1024: 402 MB each for p, dp, ds.
WHOLE_CASES = [("qk", "normal"), ("s4", "offset"), ("s1", "gradient")]
if SANITIZER:
    WHOLE_CASES = WHOLE_CASES[:1]


@pytest.mark.parametrize("p_kind, dp_kind", WHOLE_CASES, ids=lambda v: str(v))
def test_gpt2s_whole_attention_matrix_passes_the_gate(flint, p_kind, dp_kind):
    check_gate(flint, p_kind, dp_kind, 3 if SANITIZER else 192, LONG, GRADIENT_SCALE)


# -- A tighter bound than the gate's floors -----------------------------------------------------


@dataclass
class Ulps:
    """How ds compares with the FP64 result rounded to FP16, over the visible outputs."""

    visible: int = 0
    nonzero: int = 0  # visible outputs whose correct rounding isn't 0
    exact: int = 0  # of those, the ones that equal it
    worst: int = 0  # the most ulps from the correct rounding (also over the masked outputs)
    overflow: int = 0  # correct roundings that are +-inf

    @property
    def share(self):
        return self.exact / self.nonzero if self.nonzero else 1.0


def ordered(half):
    """An FP16 tensor's bits as integers that increase with the value (+0 and -0 are both 0), so
    that the difference of two is their distance in ulps."""
    bits = half.view(torch.int16).int()
    magnitude = bits & 0x7FFF
    return torch.where(bits < 0, -magnitude, magnitude)


def ulp_stats(ds, p, dp, scale):
    b, t, _ = p.shape
    mask = causal_mask(t)
    total = Ulps()
    for chunk in batch_chunks(b, t):
        ref_half = round_to_half(reference_ds(p[chunk], dp[chunk], scale))
        diff = (ordered(ds[chunk]) - ordered(ref_half)).abs()
        nonzero = mask & ((ref_half.view(torch.int16) & 0x7FFF) != 0)
        total.visible += mask.sum().item() * diff.shape[0]
        total.nonzero += nonzero.sum().item()
        total.exact += (nonzero & (diff == 0)).sum().item()
        total.worst = max(total.worst, diff.max().item())
        total.overflow += (mask & torch.isinf(ref_half)).sum().item()
        assert all_equal(ds[chunk].view(torch.int16)[:, ~mask], 0), "the masked outputs are not +0"
    return total


def assert_correctly_rounded(stats, what):
    assert stats.worst <= 1, f"{what}: {stats.worst} ulps from the correct rounding"
    if stats.nonzero >= MIN_SHARE_COUNT:
        assert stats.share >= MIN_SHARE, (
            f"{what}: {stats.share:.6f} of the {stats.nonzero} non-zero outputs are correctly "
            f"rounded, {stats.nonzero - stats.exact} are not"
        )


def check_tight(flint, p_kind, dp_kind, b, t, scale, seed=0):
    p, dp = make_case(flint, p_kind, dp_kind, b, t, seed, "cuda")
    ds = flint_backward(flint, p, dp, scale)
    stats = ulp_stats(ds, p, dp, scale)
    print(
        f"softmax backward ulps {p_kind} {dp_kind} {b}x{t} scale {scale}: worst {stats.worst}, "
        f"share {stats.share:.6f} of {stats.nonzero} (not equal: {stats.nonzero - stats.exact}), "
        f"overflow {stats.overflow}"
    )
    assert_correctly_rounded(stats, f"{p_kind} / {dp_kind} {b}x{t} scale {scale}")


def share_cases():
    """The pairs whose two rotations both land on shapes too small for the share check (fewer
    than twice MIN_SHARE_COUNT visible outputs), once more at 3 x 1024, so every pair gets it."""
    cases = []
    for i, (sk, dk) in enumerate(GATE_COMBOS):
        shapes = [GATE_SHAPES[(i + offset) % len(GATE_SHAPES)] for offset in (0, 4)]
        if all(b * t * (t + 1) // 2 < 2 * MIN_SHARE_COUNT for b, t in shapes):
            scale = SCALES[(i + 1) % 4]
            cases.append(pytest.param(sk, dk, 3, 1024, scale, id=f"{sk}-{dk}-3x1024-{scale}-share"))
    return cases


TIGHT_CASES = GATE_CASES + ([] if SANITIZER else rotated_cases(4) + share_cases())


@pytest.mark.parametrize("p_kind, dp_kind, b, t, scale", TIGHT_CASES)
def test_every_output_is_within_an_ulp_of_the_correctly_rounded_result(flint, p_kind, dp_kind, b, t, scale):
    check_tight(flint, p_kind, dp_kind, b, t, scale)


@pytest.mark.parametrize("p_kind, dp_kind", WHOLE_CASES[:2], ids=lambda v: str(v))
def test_gpt2s_whole_attention_matrix_is_within_an_ulp_of_the_correctly_rounded_result(flint, p_kind, dp_kind):
    check_tight(flint, p_kind, dp_kind, 3 if SANITIZER else 192, LONG, GRADIENT_SCALE)


# -- Exact cases ------------------------------------------------------------------------------------

EXACT_SHAPES = [(2, 8), (3, 24), (2, 72), (2, ODD_LONG), (1, LONG)]
POISONS = ["random", "nan", "-nan", "inf", "-inf"]


def poison_like(poison, like, seed):
    if poison == "random":
        junk = random_bits(like.shape, seed)
        # Random patterns are NaN 3% of the time but inf only twice in 65,536.
        junk.view(torch.int16)[..., 2::13] = as_int16(NONFINITE_BITS["inf"])
        junk.view(torch.int16)[..., 7::13] = as_int16(NONFINITE_BITS["-inf"])
        return junk
    return fp16_from_bits(NONFINITE_BITS[poison], like)


@pytest.mark.parametrize("poison", POISONS)
@pytest.mark.parametrize("b, t", shape_params(EXACT_SHAPES))
def test_the_upper_triangle_is_plus_zero_by_bits_whatever_p_and_dp_hold_there(flint, poison, b, t):
    # Most rows have a chunk that is partly visible, so their visible values share a 16-byte load
    # with junk. The clean run has zeros there.
    p, dp = make_case(flint, "uniform", "normal", b, t, seed=2, device="cuda")
    mask = causal_mask(t)
    zero = torch.zeros_like(p)
    dirty_p = torch.where(mask, p, poison_like(poison, p, 3))
    dirty_dp = torch.where(mask, dp, poison_like(poison, dp, 4))
    scale = 0.3

    clean = flint_backward(flint, torch.where(mask, p, zero), torch.where(mask, dp, zero), scale)
    ds = flint_backward(flint, dirty_p, dirty_dp, scale, fill=NAN_BITS)

    assert all_equal(ds.view(torch.int16)[:, ~mask], 0)
    assert any_unequal(ds.view(torch.int16)[:, mask], 0)
    assert_same_halves(ds, clean, f"{poison} in the masked columns")
    assert_same_halves(ds, emulate(dirty_p.cpu(), dirty_dp.cpu(), scale), "the emulation")
    assert torch.isfinite(ds).all()


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("b, t", shape_params([(8, 8), (8, 72), (8, LONG)]))
def test_row_0_is_one_column(flint, b, t, scale):
    # D is the one product p * dp, exact, so ds = ((p * (dp - p * dp)) * scale) in FP32: computed
    # here directly, with no lanes in it.
    g = rng(6, "cuda")
    p = make_p(flint, "uniform", b, t, g, "cuda")
    dp = make_dp("range", b, t, g, "cuda")
    p[:, 0, 0] = torch.tensor([1.0, 0.5, 0.0, 65504.0, 6e-8, 1.0, 0.25, 3.0], device="cuda").half()
    dp[:, 0, 0] = torch.tensor([0.0, 7.0, 5.0, -3.0, 6e4, -6e4, 6e-8, 0.0], device="cuda").half()

    ds = flint_backward(flint, p, dp, scale)

    p0, d0 = p[:, 0, 0].float(), dp[:, 0, 0].float()
    expected = ((p0 * (d0 - p0 * d0)) * float(F32(scale))).half()
    assert_same_halves(ds[:, 0, 0], expected, "row 0")
    assert all_equal(ds.view(torch.int16)[:, 0, 1:], 0)


@pytest.mark.parametrize("b, t", shape_params(EXACT_SHAPES))
def test_in_place_gives_the_bits_of_out_of_place(flint, b, t):
    p, dp = make_case(flint, "qk", "offset", b, t, seed=9, device="cuda")
    # Junk (random patterns) above the diagonal of dp, which an in-place call overwrites with +0.
    dp = torch.where(causal_mask(t), dp, random_bits(dp.shape, 5))

    out_of_place = flint_backward(flint, p, dp, 0.3)
    in_place = flint_backward(flint, p, dp, 0.3, in_place=True, fill=NAN_BITS, offset=16)

    assert_same_halves(in_place, out_of_place, "in place against out of place")
    assert_same_halves(in_place, emulate(p.cpu(), dp.cpu(), 0.3), "in place against the emulation")


@pytest.mark.parametrize("b, t", shape_params(EXACT_SHAPES))
def test_p_and_dp_may_be_the_same_buffer(flint, b, t):
    p, _ = make_case(flint, "uniform", "normal", b, t, seed=10, device="cuda")
    ds = torch.empty_like(p)

    flint.causal_softmax_backward(ds, p, p, b, t, 0.3)
    torch.cuda.synchronize()

    assert_same_halves(ds, emulate(p.cpu(), p.cpu(), 0.3), "dp is p")


@pytest.mark.parametrize("t", [8, 72, ODD_LONG])
def test_a_rows_bits_do_not_depend_on_its_batch_or_the_other_rows(flint, t):
    # Batch j keeps the base rows r with r % b == j and has other random rows elsewhere (junk
    # above the diagonal too): its kept rows give the base's bits, whichever batch, position and
    # neighbours they have.
    b = min(11, t)
    p0, dp0 = make_case(flint, "qk", "offset", 1, t, seed=0, device="cuda")
    p1, dp1 = make_case(flint, "uniform", "range", b, t, seed=1, device="cuda")
    keep = (torch.arange(t, device="cuda")[None, :, None] % b) == torch.arange(b, device="cuda")[:, None, None]
    p = torch.where(keep, p0, p1)
    dp = torch.where(keep, dp0, dp1)

    ds = flint_backward(flint, p, dp, 0.3)

    alone = flint_backward(flint, p0, dp0, 0.3)
    for j in range(b):
        rows = torch.arange(j, t, b, device="cuda")
        assert_same_halves(ds[j][rows], alone[0][rows], f"rows of batch {j} against the base")


EMBEDDED = [(8, 64), (8, LONG), (64, LONG), (72, LONG), (ODD_LONG, LONG)]


@pytest.mark.parametrize("small, big", EMBEDDED, ids=lambda v: str(v))
def test_a_row_does_not_depend_on_the_length_of_the_matrix(flint, small, big):
    # A small matrix in the top left corner of a bigger one (the rest random): the rows of the
    # corner give the same bits, and the rest of their columns are +0.
    b = 3
    p_small, dp_small = make_case(flint, "qk", "normal", b, small, seed=0, device="cuda")
    p_big, dp_big = make_case(flint, "uniform", "range", b, big, seed=1, device="cuda")
    p_big[:, :small, :small] = p_small
    dp_big[:, :small, :small] = dp_small

    small_ds = flint_backward(flint, p_small, dp_small, 0.3)
    big_ds = flint_backward(flint, p_big, dp_big, 0.3)

    assert_same_halves(big_ds[:, :small, :small], small_ds, f"the corner of a {big} matrix")
    assert all_equal(big_ds.view(torch.int16)[:, :small, small:], 0)


@pytest.mark.parametrize("b, t", shape_params([(5, 72), (4, ODD_LONG), (3, LONG)]))
def test_a_batch_alone_is_the_same_as_in_many(flint, b, t):
    p, dp = make_case(flint, "s4", "outliers", b, t, seed=4, device="cuda")

    many = flint_backward(flint, p, dp, 0.125)

    for j in range(b):
        one = flint_backward(flint, p[j : j + 1].contiguous(), dp[j : j + 1].contiguous(), 0.125)
        assert_same_halves(many[j : j + 1], one, f"batch {j} in {b} against alone")


@pytest.mark.parametrize("scale", [0.0, -0.0])
@pytest.mark.parametrize("b, t", shape_params([(2, 8), (3, 72), (2, LONG)]))
def test_scale_0_gives_zeros_for_finite_inputs(flint, b, t, scale):
    # p * (dp - D) is finite for these inputs (D is at most 1,024 products of 6e4), so
    # the scale makes every output +0 or -0.
    p, dp = make_case(flint, "uniform", "range", b, t, seed=12, device="cuda")

    ds = flint_backward(flint, p, dp, scale)

    assert all_equal(ds, 0.0)
    assert all_equal(ds.view(torch.int16)[:, ~causal_mask(t)], 0)
    assert_same_halves(ds, emulate(p.cpu(), dp.cpu(), scale), "scale 0 against the emulation")


OVERFLOW_PAIRS = [  # (dp[1], dp[2]) under p = 1, 1, 1 and dp[0] = 0: ds[0] = -(dp[1] + dp[2])
    (-32752.0, -32752.0, 65504.0),  # FP16's largest, finite
    (-32752.0, -32768.0, math.inf),  # 65,520: halfway to 65,536, ties to the even one, inf
    (-32768.0, -32768.0, math.inf),
    (32752.0, 32752.0, -65504.0),
    (32752.0, 32768.0, -math.inf),
    (32768.0, 32768.0, -math.inf),
]


@pytest.mark.parametrize("y, z, expected", OVERFLOW_PAIRS, ids=lambda v: str(v))
def test_a_result_past_the_range_of_fp16_is_inf(flint, y, z, expected):
    # Row 2 sees three columns with p = 1: D = dp[0] + dp[1] + dp[2] = y + z exactly, and ds[0] =
    # 1 * (0 - D) = -(y + z), scale 1. The other rows of the batch are zeros.
    b, t = 1, 8
    p = torch.zeros((b, t, t), dtype=torch.float16, device="cuda")
    dp = torch.zeros((b, t, t), dtype=torch.float16, device="cuda")
    p[0, 2, :3] = 1.0
    dp[0, 2, 1], dp[0, 2, 2] = y, z

    ds = flint_backward(flint, p, dp, 1.0)

    assert ds[0, 2, 0].item() == expected
    assert_same_halves(ds, emulate(p.cpu(), dp.cpu(), 1.0), "the emulation")
    # The same row, long: a lane other than lane 0 holds the columns.
    t = 264
    p = torch.zeros((b, t, t), dtype=torch.float16, device="cuda")
    dp = torch.zeros((b, t, t), dtype=torch.float16, device="cuda")
    p[0, 263, [5, 100, 263]] = 1.0
    dp[0, 263, 100], dp[0, 263, 263] = y, z
    ds = flint_backward(flint, p, dp, 1.0)
    assert ds[0, 263, 5].item() == expected
    assert_same_halves(ds, emulate(p.cpu(), dp.cpu(), 1.0), "the emulation (long row)")


# -- Non-finite values ---------------------------------------------------------------------------


def classes(x):
    """0 for a finite value, 1 for NaN, 2 for +inf, 3 for -inf."""
    return torch.where(
        torch.isnan(x), 1, torch.where(torch.isposinf(x), 2, torch.where(torch.isneginf(x), 3, 0))
    )


@pytest.mark.parametrize("target", ["p", "dp"])
@pytest.mark.parametrize("t", [72, ODD_LONG])
def test_non_finite_values_reach_only_their_rows_as_they_do_in_fp32(flint, t, target):
    # Every third row has one NaN, +inf or -inf in p or in dp, at one of five columns (one batch
    # per kind and column; the rows between are controls), the columns reached by different lanes.
    # Flint equals the emulation bit for bit; the controls are bitwise the clean run's; masked
    # columns stay +0; and NaN, +inf and -inf are where an FP32 PyTorch computation of the same
    # formula has them.
    scale = 0.3
    g = rng(21, "cpu")
    p0 = forward_p(flint, make_scores(flint, "s1", 1, t, g, "cpu")).cpu()
    dp0 = make_dp("normal", 1, t, g, "cpu")
    mask = causal_mask(t).cpu()
    rows = torch.arange(0, t, 3)
    is_injected = torch.zeros(t, dtype=torch.bool)
    is_injected[rows] = True
    cases = [(kind, v) for kind in NONFINITE_BITS for v in range(5)]
    p, dp = p0.repeat(len(cases), 1, 1), dp0.repeat(len(cases), 1, 1)
    victim = p if target == "p" else dp
    for i, (kind, v) in enumerate(cases):
        victim.view(torch.int16)[i, rows, injection_column(v, rows)] = as_int16(NONFINITE_BITS[kind])

    ds = to_cuda_run(flint, p, dp, scale)

    assert_same_halves(ds, emulate(p, dp, scale), "the emulation")
    clean = to_cuda_run(flint, p0, dp0, scale)
    assert all_equal(ds.view(torch.int16)[:, ~mask], 0), "the masked columns"
    assert not torch.isnan(clean).any()
    for i, (kind, v) in enumerate(cases):
        assert_same_halves(ds[i][~is_injected], clean[0][~is_injected], f"{kind} at variant {v}: the other rows")
    pf, dpf = p.float().cuda(), dp.float().cuda()
    maskc = mask.cuda()
    zero = torch.zeros((), device="cuda")
    pf, dpf = torch.where(maskc, pf, zero), torch.where(maskc, dpf, zero)
    d = (pf * dpf).sum(-1, keepdim=True)
    torch_ds = ((pf * (dpf - d)) * float(F32(scale))).half()
    flint_classes, torch_classes = classes(ds.cuda()), classes(torch_ds)
    wrong = (flint_classes != torch_classes) & maskc
    assert not wrong.any(), (
        f"{int(wrong.sum())} NaN / inf places differ from FP32 PyTorch, the first at "
        f"{wrong.nonzero()[0].tolist()}"
    )
    # The injected rows are non-finite across their visible columns, with these inputs.
    assert (flint_classes[:, rows][:, :, :1] != 0).all()


# -- Memory -------------------------------------------------------------------------------------------

MEMORY_CASES = [
    pytest.param(b, t, offset, fill, id=f"{b}x{t}-{offset}-{fill:#x}")
    for (b, t), offset, fill in zip(
        [(1, 8), (3, 16), (2, 24), (5, 40), (2, 72), (1, ODD_LONG), (2, LONG)],
        [8, 16, 24, 8, 16, 24, 8],
        [SENTINEL, NAN_BITS, OTHER_BITS, SENTINEL, OTHER_BITS, NAN_BITS, SENTINEL],
    )
]


@pytest.mark.parametrize("b, t, offset, fill", MEMORY_CASES)
def test_every_element_is_written_and_nothing_else(flint, b, t, offset, fill):
    # flint_backward checks the guard bands and that p and dp are unchanged. An element left unwritten
    # would keep the fill, so runs into two different fills (and the emulation) must agree.
    p, dp = make_case(flint, "s1", "normal", b, t, seed=0, device="cuda")

    ds = flint_backward(flint, p, dp, 0.125, offset=offset, fill=fill)

    other = flint_backward(flint, p, dp, 0.125, offset=8, fill=OTHER_BITS if fill != OTHER_BITS else SENTINEL)
    assert_same_halves(ds, other, "against a run into another fill")
    assert_same_halves(ds, emulate(p.cpu(), dp.cpu(), 0.125), "against the emulation")
    assert torch.isfinite(ds).all()


# -- Determinism ------------------------------------------------------------------------------------


@pytest.mark.parametrize("p_kind, dp_kind, b, t", [("outliers", "range", 3, ODD_LONG), ("s30", "outliers", 4, 72), ("uniform", "normal", 2, 24)], ids=str)
def test_run_twice_is_bitwise_identical_including_nan_rows(flint, p_kind, dp_kind, b, t):
    p, dp = make_case(flint, p_kind, dp_kind, b, t, seed=0, device="cuda")
    rows = torch.arange(0, t, 3, device="cuda")
    p.view(torch.int16)[0, rows, rows // 2] = as_int16(NONFINITE_BITS["nan"])
    dp.view(torch.int16)[1, rows, rows] = as_int16(NONFINITE_BITS["inf"])
    dp.view(torch.int16)[-1, rows[1::2], 0] = as_int16(NONFINITE_BITS["-inf"])

    first = flint_backward(flint, p, dp, 0.3, fill=NAN_BITS)
    second = flint_backward(flint, p, dp, 0.3, fill=OTHER_BITS)
    third = flint_backward(flint, p, dp, 0.3, fill=SENTINEL)

    assert torch.isnan(first).any()
    assert torch.equal(second.view(torch.int16), first.view(torch.int16))
    assert torch.equal(third.view(torch.int16), first.view(torch.int16))


# -- The limits ------------------------------------------------------------------------------------


@pytest.mark.skipif(SANITIZER, reason="65,535 blocks are too slow under compute-sanitizer")
def test_65535_batches_of_length_8(flint):
    # Every batch has its own random values, so a block that read another batch's would be far
    # off: the whole result against the FP64 result, and a sample of batches (the first, the last,
    # edges of powers of 2) bit for bit against the emulation.
    b, t = MAX_BATCHES, 8
    p, dp = make_case(flint, "uniform", "normal", b, t, seed=0, device="cuda")

    ds = flint_backward(flint, p, dp, 0.125, fill=NAN_BITS)

    assert_correctly_rounded(ulp_stats(ds, p, dp, 0.125), "65,535 batches of 8")
    for j in [0, 1, 2, 255, 256, 4095, 4096, 32767, 32768, 50000, 65533, MAX_BATCHES - 1]:
        assert_same_halves(ds[j : j + 1], emulate(p[j : j + 1].cpu(), dp[j : j + 1].cpu(), 0.125), f"batch {j}")


@pytest.mark.skipif(SANITIZER, reason="length 1,024 is too slow under compute-sanitizer")
def test_length_1024(flint):
    b, t = 7, MAX_LENGTH
    p, dp = make_case(flint, "qk", "offset", b, t, seed=0, device="cuda")

    ds = flint_backward(flint, p, dp, 0.125)

    assert_correctly_rounded(ulp_stats(ds, p, dp, 0.125), "length 1,024")
    assert_same_halves(ds, emulate(p.cpu(), dp.cpu(), 0.125), "length 1,024 against the emulation")


# -- Streams -------------------------------------------------------------------------------------

STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
def test_runs_on_the_given_stream(flint, mode):
    b, t = 3, 64
    p, good = make_case(flint, "s1", "normal", b, t, seed=0, device="cuda")
    dp = torch.full((b, t, t), math.nan, dtype=torch.float16, device="cuda")
    ds = torch.zeros((b, t, t), dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    # A cold first launch (loading the module) can outlast the side stream's sleep.
    flint.causal_softmax_backward(ds, p, good, b, t, 0.125)
    torch.cuda.synchronize()
    ds.zero_()

    # The copy finishes only after the sleep, so ds is the right result only if the kernel is
    # queued on the same stream behind it (it is NaN where it ran first). With the current
    # stream the call has no stream argument, and torch's current stream is the side one; with
    # the default stream there is no side one at all.
    if mode == "default-stream":
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        dp.copy_(good)
        flint.causal_softmax_backward(ds, p, dp, b, t, 0.125)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            dp.copy_(good)
            flint.causal_softmax_backward(ds, p, dp, b, t, 0.125, stream=stream if mode == "explicit-stream" else None)
        stream.synchronize()

    assert not torch.isnan(ds).any()
    assert_same_halves(ds, emulate(p.cpu(), good.cpu(), 0.125), mode)


# -- Argument errors --------------------------------------------------------------------------------

NAMES = ("ds", "p", "dp")
SMALL = dict(batches=2, length=16)
BYTES = 2 * 16 * 16 * 2
ROW_BYTES = 16 * 2


def small_problem():
    """Valid tensors by name, in buffers with room to spare, filled with a sentinel."""
    return {name: torch.full((2 * 16 * 16 + 64,), 0.5, dtype=torch.float16, device="cuda") for name in NAMES}


def call(flint, buffers, **args):
    args = {**SMALL, "scale": 0.125, **args}
    flint.causal_softmax_backward(
        buffers["ds"], buffers["p"], buffers["dp"], args["batches"], args["length"], args["scale"]
    )


def expect_invalid_argument(flint, reason, buffers, **args):
    """The call fails with the status and message, and no tensor in the buffers changed."""
    before = {n: b.clone() for n, b in buffers.items() if isinstance(b, torch.Tensor)}
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, buffers, **args)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in (NAME, reason):
        assert text in str(error.value)
        assert text in flint.last_error()
    torch.cuda.synchronize()
    for name, tensor in before.items():
        assert torch.equal(buffers[name], tensor), f"{name} changed"


def nothing():
    return {name: None for name in NAMES}


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    expect_invalid_argument(flint, f"{name} is null", {**small_problem(), name: None})


@pytest.mark.parametrize("offset", [2, 8])
@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name, offset):
    # 2 bytes is as far off as an element goes, and 8 is aligned for 64 bits but not 128.
    addresses = {n: t.data_ptr() for n, t in small_problem().items()}
    addresses[name] += offset
    expect_invalid_argument(flint, f"{name} is not 16-byte aligned", addresses)


@pytest.mark.parametrize("name", NAMES)
def test_a_buffer_may_not_run_past_the_address_space(flint, name):
    expect_invalid_argument(
        flint, f"{name} runs past the end of the address space", {**small_problem(), name: 2**64 - 16}
    )


def test_pointers_are_checked_in_order(flint):
    # Per buffer: null, then alignment, then the address space; ds, p, then dp.
    a = {n: t.data_ptr() for n, t in small_problem().items()}
    expect_invalid_argument(flint, "ds is not 16-byte aligned", {**a, "ds": a["ds"] + 2, "p": None, "dp": None})
    expect_invalid_argument(flint, "ds is null", {**a, "ds": None, "p": a["p"] + 2})
    expect_invalid_argument(flint, "p is not 16-byte aligned", {**a, "p": a["p"] + 2, "dp": None})
    expect_invalid_argument(flint, "p is null", {**a, "p": None, "dp": a["dp"] + 2})
    expect_invalid_argument(flint, "dp is not 16-byte aligned", {**a, "dp": a["dp"] + 2})
    expect_invalid_argument(flint, "ds runs past the end", {"ds": 2**64 - 16, "p": None, "dp": None})


@pytest.mark.parametrize(
    "length", [1, 4, 7, 9, 12, 20, 1020, 1032, 1040, 2048, 2**32, 2**64 - 8, 2**64 - 1], ids=str
)
def test_length_must_be_a_multiple_of_8_up_to_1024(flint, length):
    expect_invalid_argument(
        flint, f"length ({length}) must be a multiple of 8 up to 1024", small_problem(), length=length
    )


@pytest.mark.parametrize("batches", [65_536, 65_537, 2**32, 2**64 - 1], ids=str)
def test_batches_can_be_up_to_65535(flint, batches):
    expect_invalid_argument(flint, f"batches ({batches}) can be up to 65535", small_problem(), batches=batches)


@pytest.mark.parametrize("scale", [math.inf, -math.inf, math.nan], ids=str)
def test_scale_must_be_finite(flint, scale):
    expect_invalid_argument(flint, "scale (", small_problem(), scale=scale)
    expect_invalid_argument(flint, "must be finite", small_problem(), scale=scale)


@pytest.mark.parametrize(
    "args",
    [dict(batches=MAX_BATCHES, length=MAX_LENGTH), dict(batches=MAX_BATCHES), dict(length=MAX_LENGTH),
     dict(batches=1, length=MAX_LENGTH), dict(batches=1, length=8), dict(scale=1e38), dict(scale=-3.4e38)],
    ids=lambda args: "-".join(f"{k}={v}" for k, v in args.items()),
)
def test_the_largest_sizes_and_scales_pass_the_checks(flint, args):
    # The limits themselves aren't errors of size, nor a huge finite scale: the call gets on to
    # the pointers (a null ds), even where batches * length^2 * 2 is 1.4e11 bytes.
    expect_invalid_argument(flint, "ds is null", {**small_problem(), "ds": None}, **args)


def test_size_limits_are_checked_in_order(flint):
    # length, then batches, then the scale, then the pointers (null ones included); the
    # empty-problem exit comes after the size and scale checks.
    expect_invalid_argument(flint, "length (7) must be", nothing(), length=7, batches=2**40, scale=math.inf)
    expect_invalid_argument(flint, "length (7) must be", nothing(), length=7, batches=0)
    expect_invalid_argument(flint, "batches (65536) can be up to", nothing(), batches=65_536, scale=math.nan)
    expect_invalid_argument(flint, "batches (65536) can be up to", nothing(), batches=65_536, length=0)
    expect_invalid_argument(flint, "length (1032) must be", nothing(), length=1032, batches=0)
    expect_invalid_argument(flint, "scale (inf) must be finite", nothing(), scale=math.inf)
    expect_invalid_argument(flint, "scale (inf) must be finite", nothing(), scale=math.inf, batches=0)
    expect_invalid_argument(flint, "scale (-inf) must be finite", nothing(), scale=-math.inf, length=0)


SPACING = 4096

# Where the other buffer starts relative to ds, in bytes: inside ds by 16 bytes and by one row, at
# ds's second batch, 16 bytes before its end, and (negative) before ds so that it ends 16 bytes
# into ds, one row into it, or at its second batch. Right after ds's end and right before its start,
# the buffers touch without overlapping.
OVERLAPPING = {
    "same-start": 0,
    "16-bytes-in": 16,
    "one-row-in": ROW_BYTES,
    "second-batch": BYTES // 2,
    "16-bytes-before-the-end": BYTES - 16,
    "ends-16-bytes-into-ds": 16 - BYTES,
    "ends-one-row-into-ds": ROW_BYTES - BYTES,
    "ends-at-ds-second-batch": BYTES // 2 - BYTES + 16,
}
TOUCHING = {"after": BYTES, "before": -BYTES}


def arena_addresses(arena, other, offset):
    base = arena.data_ptr() + 3 * SPACING
    return {"ds": base, other: base + offset, ("dp" if other == "p" else "p"): base + 6 * SPACING}


@pytest.mark.parametrize("label", OVERLAPPING)
@pytest.mark.parametrize("other", ["p", "dp"])
def test_ds_may_not_overlap_p_or_partly_overlap_dp(flint, other, label):
    if other == "dp" and label == "same-start":
        pytest.skip("ds is dp: in place, which is accepted")
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no harm
    # to anything else.
    arena = torch.zeros(12 * SPACING, dtype=torch.uint8, device="cuda")
    before = arena.clone()
    expect_invalid_argument(flint, f"ds and {other} overlap", arena_addresses(arena, other, OVERLAPPING[label]))
    assert torch.equal(arena, before)


def test_ds_may_not_overlap_p_even_when_it_is_dp(flint):
    arena = torch.zeros(12 * SPACING, dtype=torch.uint8, device="cuda")
    base = arena.data_ptr() + 3 * SPACING
    for p_offset in (0, 16, ROW_BYTES, -16):
        expect_invalid_argument(flint, "ds and p overlap", {"ds": base, "dp": base, "p": base + p_offset})


@pytest.mark.parametrize("other", ["p", "dp"])
@pytest.mark.parametrize("label", TOUCHING)
def test_buffers_next_to_each_other_are_not_overlapping(flint, other, label):
    arena = torch.full((12 * SPACING,), 0x5A, dtype=torch.uint8, device="cuda")
    ds_at = 3 * SPACING
    at = {"ds": ds_at, other: ds_at + TOUCHING[label], ("dp" if other == "p" else "p"): ds_at + 6 * SPACING}
    views = {n: arena[a : a + BYTES].view(torch.float16) for n, a in at.items()}
    views["p"].copy_(normal((BYTES // 2,), 1.0, generator(0)).abs().clamp(max=1.0))
    views["dp"].copy_(normal((BYTES // 2,), 1.0, generator(1)))
    p_before, dp_before = views["p"].clone(), views["dp"].clone()
    outside = torch.ones(arena.numel(), dtype=torch.bool, device="cuda")
    outside[ds_at : ds_at + BYTES] = False
    others = arena[outside].clone()

    flint.causal_softmax_backward(views["ds"], views["p"], views["dp"], 2, 16, 0.5)
    torch.cuda.synchronize()

    assert torch.equal(arena[outside], others)
    assert torch.equal(views["p"].view(torch.int16), p_before.view(torch.int16))
    assert torch.equal(views["dp"].view(torch.int16), dp_before.view(torch.int16))
    expected = emulate(p_before.view(2, 16, 16).cpu(), dp_before.view(2, 16, 16).cpu(), 0.5)
    assert_same_halves(views["ds"].view(2, 16, 16), expected, "ds next to " + other)


def test_ds_equal_to_dp_is_accepted(flint):
    arena = torch.zeros(12 * SPACING, dtype=torch.uint8, device="cuda")
    p = arena[3 * SPACING : 3 * SPACING + BYTES].view(torch.float16)
    ds = arena[6 * SPACING : 6 * SPACING + BYTES].view(torch.float16)
    p.copy_(normal((BYTES // 2,), 1.0, generator(0)).abs().clamp(max=1.0))
    ds.copy_(normal((BYTES // 2,), 1.0, generator(1)))
    dp_before = ds.clone()
    flint.causal_softmax_backward(ds, p, ds, 2, 16, 0.5)
    torch.cuda.synchronize()
    assert_same_halves(ds.view(2, 16, 16), emulate(p.view(2, 16, 16).cpu(), dp_before.view(2, 16, 16).cpu(), 0.5), "in place")


@pytest.mark.parametrize(
    "batches, length",
    [(0, 16), (2, 0), (0, 0), (0, 1024), (MAX_BATCHES, 0), (0, 8)],
    ids=lambda v: str(v),
)
def test_empty_problems_do_nothing(flint, batches, length):
    # Not even the null pointers are looked at.
    buffers = small_problem()

    flint.causal_softmax_backward(None, None, None, batches, length)
    call(flint, buffers, batches=batches, length=length)
    torch.cuda.synchronize()

    assert all_equal(buffers["ds"], 0.5)


def test_a_valid_call_works_after_errors(flint):
    buffers = small_problem()
    for args in (dict(length=7), dict(length=1032), dict(batches=65_536), dict(length=2**64 - 1), dict(scale=math.inf)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, buffers, **args)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "ds": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "dp": buffers["dp"].data_ptr() + 2})
    torch.cuda.synchronize()
    assert all_equal(buffers["ds"], 0.5)

    call(flint, buffers)
    torch.cuda.synchronize()

    # p and dp are 0.5 everywhere, so the call is an ordinary one on those values.
    n = 2 * 16 * 16
    ds = buffers["ds"][:n].view(2, 16, 16)
    expected = emulate(buffers["p"][:n].view(2, 16, 16).cpu(), buffers["dp"][:n].view(2, 16, 16).cpu(), 0.125)
    assert_same_halves(ds, expected, "after errors")
    assert all_equal(buffers["ds"][n:], 0.5)


# -- The chain: S = Q K^T / 8, P, dP = dO V^T, dS, dQ = dS K, dK = dS^T Q ------------------------------


def make_qkvd(b, t, seed, std):
    """Q, K and V about N(0, std^2) and dO about N(0, 1), in the heads layout B x T x 64."""
    g = generator(seed)
    q, k, v = (normal((b, t, HEAD_WIDTH), std, g) for _ in range(3))
    return q, k, v, normal((b, t, HEAD_WIDTH), 1.0, g)


def flint_gradients(flint, q, k, v, do, p=None):
    """dQ and dK through Flint's kernels. p (FP16 batches x T x T) replaces the forward pass's
    P, to try another way of making it."""
    b, t, _ = q.shape
    s = torch.empty((b, t, t), dtype=torch.float16, device="cuda")
    dp = torch.empty_like(s)
    ds = torch.empty_like(s)
    dq = torch.empty((b, t, HEAD_WIDTH), dtype=torch.float16, device="cuda")
    dk = torch.empty_like(dq)
    if p is None:
        p = torch.empty_like(s)
        flint.gemm_nt_batched(s, q, k, b, t, t, HEAD_WIDTH, scale=0.125)
        flint.causal_softmax_forward(p, s, b, t)
    flint.gemm_nt_batched(dp, do, v, b, t, t, HEAD_WIDTH, scale=1.0)
    flint.causal_softmax_backward(ds, p, dp, b, t, GRADIENT_SCALE)
    flint.gemm_nn_batched(dq, ds, k, b, t, HEAD_WIDTH, t)
    flint.gemm_tn_batched(dk, ds, q, b, t, HEAD_WIDTH, t)
    torch.cuda.synchronize()
    return dq, dk


def fp32_scores_p(q, k):
    """P from scores kept in FP32 (S, the softmax), rounded to FP16 once."""
    t = q.shape[1]
    with gate.peer_math():
        s = q.float() @ k.float().transpose(1, 2) / 8
        return F.softmax(s.masked_fill(~causal_mask(t), -math.inf), dim=-1).half()


def reference_gradients(q, k, v, do, chunk=4):
    """dQ and dK from PyTorch's FP64 autograd of causal attention on the upcast FP16 inputs, over
    chunks of batches (the batches are independent)."""
    dq, dk = [], []
    for i in range(0, q.shape[0], chunk):
        qd, kd, vd = (x[i : i + chunk].double().requires_grad_() for x in (q, k, v))
        out = pytorch_attention(qd, kd, vd)
        with gate.peer_math():
            out.backward(do[i : i + chunk].double())
        dq.append(qd.grad)
        dk.append(kd.grad)
    return torch.cat(dq), torch.cat(dk)


def peer_gradients(q, k, v, do):
    """PyTorch's FP16 path: causal attention on the FP16 tensors, its backward pass."""
    qh, kh, vh = (x.clone().requires_grad_() for x in (q, k, v))
    out = pytorch_attention(qh, kh, vh)
    with gate.peer_math():
        out.backward(do)
    return qh.grad, kh.grad


CHAIN_CASES = [(3, 64, 1.0), (3, 128, 1.0)] if SANITIZER else [(12, 64, 1.0), (12, 256, 1.0), (24, 1024, 1.0)]


@pytest.mark.parametrize("b, t, std", CHAIN_CASES, ids=lambda v: str(v))
def test_the_chain_passes_the_gate_against_pytorch_autograd(flint, b, t, std):
    # Q, K, V in the heads layout, about N(0, std^2): at std 1 the scores are about N(0, 1).
    errors = {name: ([], []) for name in ("dq", "dk")}
    for seed in gate.SEEDS:
        q, k, v, do = make_qkvd(b, t, seed, std)
        ref = dict(zip(("dq", "dk"), reference_gradients(q, k, v, do)))
        got = dict(zip(("dq", "dk"), flint_gradients(flint, q, k, v, do)))
        peer = dict(zip(("dq", "dk"), peer_gradients(q, k, v, do)))
        for name in errors:
            errors[name][0].append(gate.errors(got[name], ref[name]))
            errors[name][1].append(gate.errors(peer[name], ref[name]))
    for name, (flint_errors, peer_errors) in errors.items():
        print(
            f"softmax backward chain {name} {b}x{t} std {std}: flint rel_l2 "
            f"{max(e.rel_l2 for e in flint_errors):.3g}, max_abs {max(e.max_abs for e in flint_errors):.3g}; "
            f"peer rel_l2 {max(e.rel_l2 for e in peer_errors):.3g}, max_abs "
            f"{max(e.max_abs for e in peer_errors):.3g}"
        )
    for name, (flint_errors, peer_errors) in errors.items():
        bounds = gate.check_gate(gate.GRADIENT_FACTOR, flint_errors, peer_errors)
        print(f"  {name} bound rel_l2 {bounds.rel_l2:.3g}")


@pytest.mark.skipif(SANITIZER, reason="the chain above runs these kernels under the sanitizer")
@pytest.mark.parametrize(
    "std, rel_l2_limit, max_abs_limit, over_gate",
    [(2.0, 2e-3, 0.06, False), (4.0, 8e-3, 0.7, True)],
    ids=lambda v: str(v),
)
def test_fp16_scores_put_the_chain_over_the_gate_at_wide_scores(flint, std, rel_l2_limit, max_abs_limit,
                                                                over_gate):
    # A known cost, pinned so it can't grow unnoticed (the limits are the measured errors with
    # some room, not a gate): Q and K about N(0, std^2) give scores with std about std^2, which
    # the chain stores in FP16 between the GEMM and the softmax, where PyTorch's fused attention
    # (the peer) keeps them in FP32. The same chain with P made from FP32 scores passes the gate
    # (only dS is stored in FP16 then), so S's storage is the cause, as it is for the forward
    # pass (test_softmax.py).
    b, t = 12, 256
    errors = {name: ([], [], []) for name in ("dq", "dk")}
    for seed in gate.SEEDS:
        q, k, v, do = make_qkvd(b, t, seed, std)
        ref = dict(zip(("dq", "dk"), reference_gradients(q, k, v, do)))
        got = dict(zip(("dq", "dk"), flint_gradients(flint, q, k, v, do)))
        fp32 = dict(zip(("dq", "dk"), flint_gradients(flint, q, k, v, do, p=fp32_scores_p(q, k))))
        peer = dict(zip(("dq", "dk"), peer_gradients(q, k, v, do)))
        for name in errors:
            errors[name][0].append(gate.errors(got[name], ref[name]))
            errors[name][1].append(gate.errors(fp32[name], ref[name]))
            errors[name][2].append(gate.errors(peer[name], ref[name]))
    for name, (flint_errors, fp32_errors, peer_errors) in errors.items():
        bounds = gate.check_gate(gate.GRADIENT_FACTOR, fp32_errors, peer_errors)
        # At std 2 the FP16-S chain sits at the gate's edge (over it by max-abs on the laptop,
        # within it on the 2080 Ti), so only its ceilings are pinned there; at std 4 it is
        # clearly over on both.
        if over_gate:
            with pytest.raises(AssertionError):
                gate.check_gate(gate.GRADIENT_FACTOR, flint_errors, peer_errors)
        print(
            f"softmax backward chain {name} {b}x{t} std {std}: FP16 S rel_l2 "
            f"{max(e.rel_l2 for e in flint_errors):.3g}, max_abs {max(e.max_abs for e in flint_errors):.3g}; "
            f"FP32 S rel_l2 {max(e.rel_l2 for e in fp32_errors):.3g}, max_abs "
            f"{max(e.max_abs for e in fp32_errors):.3g}; peer rel_l2 "
            f"{max(e.rel_l2 for e in peer_errors):.3g}, max_abs {max(e.max_abs for e in peer_errors):.3g}; "
            f"bound {bounds.rel_l2:.3g}"
        )
        # Per seed, so that a NaN fails.
        assert all(e.rel_l2 <= rel_l2_limit and e.max_abs <= max_abs_limit for e in flint_errors)
