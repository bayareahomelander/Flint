"""flint_gemm_nt_batched, flint_gemm_nn_batched and flint_gemm_tn_batched: attention's
per-head GEMMs, `batches` matrices at a time in one launch, each batch's matrices
directly after the last's in memory, all FP16 row-major. Every test runs for the
layouts that apply (test ids start with nt-, nn- or tn-).

NT is y[j] = scale x[j] w[j]^T (S = Q K^T / 8, dP = dO V^T), NN is y[j] = x[j] w[j]
(O = P V, dQ = dS K) and TN is y[j] = x[j]^T w[j] (dV = P^T dO, dK = dS^T Q), with x[j]
k x m and w[j] k x n. Unlike the plain flint_gemm_tn, TN's y is FP16: each FP32 sum rounds
once, and k never splits.

The strongest test is bitwise: each batch of a batched result must equal the plain
kernel on that batch (NT with scale 1, NN, and TN rounded by .half() wherever the plain
kernel doesn't split k, which the test asserts), with different random data in every
batch. The correctness gate then runs on attention's realistic inputs (Q and K, the
probabilities P with V, small gradients), on outlier channels, on odd shapes, and on TN
with a k the plain kernel would split. The rest covers NT's scale (a sum past FP16's
range that the scale brings back, powers of 2 and -1 and 0 against the plain result,
1/3 against float64, and the single rounding after the scale), the batches (permuting
them, one alone, a non-finite value in one), every output written and nothing else,
determinism, stream ordering and argument errors.

Memory: the GPT-2 shape (192 batches of T = 1024, d = 64) makes NT's y, and NN's and TN's
x, 402 MB, so the bitwise tests use it whole, and the gate uses 24 batches. Under the
sanitizer (scripts/sanitize.py) the shapes shrink to T = 128 and fewer batches, keeping
every tensor far below 300 MB, and the tests of the limits are skipped."""

import ctypes
import math
import time
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator, normal

# GPT-2's attention: 16 sequences x 12 heads, T = 1024, d = 64.
GPT2_BATCHES = 8 if SANITIZER else 192
GPT2_T = 128 if SANITIZER else 1024
HEAD_WIDTH = 64
# The gate runs on fewer batches, to keep the laptop's memory reasonable.
GATE_BATCHES = 3 if SANITIZER else 24
GATE_T = 128 if SANITIZER else 1024

MAX_BATCHES = 65_535
MAX_M = 1024
MAX_MN = 8_388_480
MAX_K = 65_536

# The output of a guarded call sits `offset` halves past the start of a buffer that is
# filled with a pattern, which must be intact around y afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A
NAN_BITS = 0xFFFF
OTHER_NAN_BITS = 0x7F7F


def shape_id(shape):
    return "x".join(map(str, shape))


def f32(value):
    """value rounded to FP32, as a scale reaches the kernel."""
    return ctypes.c_float(value).value


# -- The three layouts ------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """What differs between the three batched functions. Shapes are (batches, m, n, k)."""

    name: str  # "nt", "nn" or "tn"
    peer: Callable  # PyTorch's FP16 path on one batch
    reference: Callable  # the float reference on one batch
    factor: float  # the gate's factor
    shapes: list  # for the bitwise comparison with the plain kernel
    partial_shapes: list  # partial tiles, odd m and n, the most m
    determinism_shapes: list
    alias_shape: tuple  # x and w the same shape, so one tensor can be both
    exact_k_mn: tuple  # (m, n) of the k = 2^16 integer test

    @property
    def flint_name(self):
        return f"flint_gemm_{self.name}_batched"

    def x_shape(self, b, m, n, k):
        return (b, k, m) if self.name == "tn" else (b, m, k)

    def w_shape(self, b, m, n, k):
        return (b, n, k) if self.name == "nt" else (b, k, n)

    def x_at(self, j, row, kk):
        """The index of the element of x[j] that pairs output row `row` with step kk of k."""
        return (j, kk, row) if self.name == "tn" else (j, row, kk)

    def w_at(self, j, col, kk):
        """The index of the element of w[j] that pairs output column `col` with step kk."""
        return (j, col, kk) if self.name == "nt" else (j, kk, col)

    def call(self, flint, y, x, w, b, m, n, k, scale=1.0, stream=None):
        function = getattr(flint, f"gemm_{self.name}_batched")
        if self.name == "nt":
            function(y, x, w, b, m, n, k, scale=scale, stream=stream)
        else:
            function(y, x, w, b, m, n, k, stream=stream)

    def plain(self, flint, x, w, m, n, k):
        """One batch (2-D x and w) by the plain kernel, as an m x n FP16 tensor. The plain
        kernel wants a 16-byte aligned y, and a batch of the batched y may not be (m n odd),
        so it writes a scratch of its own. TN's FP32 output rounds to FP16 here."""
        if self.name == "tn":
            scratch = torch.full((m * n,), math.nan, dtype=torch.float32, device="cuda")
            flint.gemm_tn(scratch, x, w, m, n, k)
            return scratch.view(m, n).half()
        scratch = torch.full((m * n,), math.nan, dtype=torch.float16, device="cuda")
        getattr(flint, f"gemm_{self.name}")(scratch, x, w, m, n, k)
        return scratch.view(m, n)


def attention_nt_shape(t, b):
    return (b, t, t, HEAD_WIDTH)


def attention_tn_shape(t, b):
    return (b, t, HEAD_WIDTH, t)


NT = Layout(
    name="nt",
    peer=gate.peer_gemm_nt,
    reference=gate.reference,
    factor=gate.FORWARD_FACTOR,
    shapes=[
        (1, 1, 1, 8),
        (2, 7, 9, 8),
        (3, 8, 8, 8),
        (5, 72, 72, 16),
        (7, 128, 128, 64),
        (12, 136, 136, 64),
        (3, 129, 127, 64),
        (2, 16, 1000, 24),
        (2, 64, 64, 1024),
        (24, 256, 256, 64),
        attention_nt_shape(GPT2_T, GPT2_BATCHES),
    ],
    partial_shapes=[(3, 129, 127, 40), (2, 7, 9, 8), (4, 1, 1, 16), (3, 200, 13, 24),
                    (2, 1000, 130, 16), (2, 1024, 100, 16)],
    determinism_shapes=[(3, 129, 127, 64), (6, GATE_T, GATE_T, 64)],
    alias_shape=(4, 72, 72, 40),
    exact_k_mn=(2, 2),
)

NN = Layout(
    name="nn",
    peer=gate.peer_gemm_nn,
    reference=gate.reference_nn,
    factor=gate.FORWARD_FACTOR,
    shapes=[
        (3, 8, 8, 8),
        (5, 72, 16, 72),
        (7, 128, 64, 128),
        (12, 136, 64, 136),
        (3, 100, 64, 128),
        (3, 129, 8, 16),
        (2, 256, 256, 40),
        (3, 72, 1000, 64),
        attention_tn_shape(GPT2_T, GPT2_BATCHES),
    ],
    partial_shapes=[(3, 129, 8, 16), (3, 100, 64, 128), (2, 1, 8, 8), (2, 1000, 24, 16),
                    (2, 1024, 72, 40), (2, 130, 136, 16)],
    determinism_shapes=[(3, 100, 64, 128), (6, GATE_T, HEAD_WIDTH, GATE_T)],
    alias_shape=(4, 72, 72, 72),
    exact_k_mn=(2, 8),
)

TN = Layout(
    name="tn",
    peer=gate.peer_gemm_tn,
    reference=gate.reference_tn,
    factor=gate.GRADIENT_FACTOR,
    shapes=[
        (3, 8, 8, 8),
        (3, 16, 8, 5),
        (2, 128, 64, 33),
        (5, 72, 16, 72),
        (7, 128, 64, 128),
        (12, 136, 64, 136),
        (2, 256, 256, 40),
        (3, 72, 1000, 64),
        attention_tn_shape(GPT2_T, GPT2_BATCHES),
    ],
    partial_shapes=[(3, 16, 8, 5), (2, 136, 72, 33), (3, 8, 8, 1), (2, 1024, 8, 31),
                    (2, 1000, 24, 40), (2, 136, 136, 33)],
    determinism_shapes=[(3, 72, 16, 72), (6, GATE_T, HEAD_WIDTH, GATE_T)],
    alias_shape=(4, 72, 72, 40),
    exact_k_mn=(8, 8),
)

LAYOUTS = [NT, NN, TN]
by_layout = pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.name for layout in LAYOUTS])


# -- Inputs -----------------------------------------------------------------
# Every maker takes (layout, b, m, n, k, seed) and returns x and w in the layout's memory
# form: (b, m, k) and (b, n, k) for NT, (b, m, k) and (b, k, n) for NN, (b, k, m) and
# (b, k, n) for TN.


def random_inputs(layout, b, m, n, k, seed=0, x_std=1.0, w_std=1.0):
    g = generator(seed)
    return (normal(layout.x_shape(b, m, n, k), x_std, g),
            normal(layout.w_shape(b, m, n, k), w_std, g))


def realistic(layout, b, m, n, k, seed):
    """x about N(0, 1) and w about N(0, 0.02)."""
    return random_inputs(layout, b, m, n, k, seed, 1.0, 0.02)


def outliers(layout, b, m, n, k, seed):
    """About 0.5% of the channels (at least 2) near +-3,000 in one operand, as in GPT-2's
    activations: for NT and NN x's channels of k (x about N(0, 1), w N(0, 0.02)), for TN
    w's channels of n (dY about N(0, 0.01), which keeps the sums inside FP16)."""
    if layout is TN:
        x, w = random_inputs(layout, b, m, n, k, seed, 0.01, 1.0)
        target, width = w, n
    else:
        x, w = random_inputs(layout, b, m, n, k, seed, 1.0, 0.02)
        target, width = x, k
    g = generator(10_000 + seed)
    count = max(2, round(0.005 * width))
    channels = torch.randperm(width, generator=g, device="cuda")[:count]
    signs = torch.randint(0, 2, (count,), generator=g, device="cuda") * 2.0 - 1.0
    jitter = 0.9 + 0.2 * torch.rand((b, target.shape[1], count), generator=g, device="cuda")
    target[:, :, channels] = (3000.0 * signs * jitter).half()
    return x, w


def causal_probabilities(b, t, seed):
    """Attention's P: lower-triangular, each row the softmax of a masked N(0, 1) row
    (what Q K^T / 8 looks like), rounded to FP16."""
    g = generator(seed)
    logits = torch.randn((b, t, t), generator=g, device="cuda")
    keep = torch.ones((t, t), dtype=torch.bool, device="cuda").tril()
    return logits.masked_fill(~keep, -math.inf).softmax(dim=-1).half()


def causal_gradients(b, t, seed):
    """dS: lower-triangular, about N(0, 1e-3)."""
    g = generator(seed)
    return (torch.randn((b, t, t), generator=g, device="cuda") * 1e-3).tril().half()


def attention_scores(layout, b, m, n, k, seed):
    """S = Q K^T / 8 with Q and K about N(0, 1)."""
    return random_inputs(layout, b, m, n, k, seed)


def attention_dp(layout, b, m, n, k, seed):
    """dP = dO V^T with dO about N(0, 1e-3) and V about N(0, 1)."""
    return random_inputs(layout, b, m, n, k, seed, 1e-3, 1.0)


def attention_o(layout, b, m, n, k, seed):
    """O = P V: x is P (m = k = T) and w is V, about N(0, 1)."""
    return causal_probabilities(b, m, seed), normal((b, k, n), 1.0, generator(1000 + seed))


def attention_dq(layout, b, m, n, k, seed):
    """dQ = dS K: x is dS and w is K, about N(0, 1)."""
    return causal_gradients(b, m, seed), normal((b, k, n), 1.0, generator(1000 + seed))


def attention_dv(layout, b, m, n, k, seed):
    """dV = P^T dO: x is P (k x m, both T) and w is dO, about N(0, 1e-3)."""
    return causal_probabilities(b, k, seed), normal((b, k, n), 1e-3, generator(1000 + seed))


def attention_dk(layout, b, m, n, k, seed):
    """dK = dS^T Q: x is dS (k x m, both T) and w is Q, about N(0, 1)."""
    return causal_gradients(b, k, seed), normal((b, k, n), 1.0, generator(1000 + seed))


def gradients(layout, b, m, n, k, seed):
    """x and w both about N(0, 1)."""
    return random_inputs(layout, b, m, n, k, seed)


def integers(layout, b, m, n, k, seed, bound):
    """Integers in [-bound, bound], whose sums FP32 holds exactly."""
    g = generator(seed)
    return tuple(
        torch.randint(-bound, bound + 1, shape, generator=g, device="cuda").half()
        for shape in (layout.x_shape(b, m, n, k), layout.w_shape(b, m, n, k))
    )


# -- Running Flint ----------------------------------------------------------


def assert_same_bits(actual, expected, what):
    """The two FP16 tensors are equal bit for bit (NaN payloads included)."""
    a, e = actual.view(torch.int16), expected.view(torch.int16)
    if torch.equal(a, e):
        return
    bad = a != e
    first = bad.nonzero()[0].tolist()
    raise AssertionError(
        f"{what}: {bad.sum().item()} of {bad.numel()} halves differ, the first at {first}: "
        f"{actual[tuple(first)].item()} against {expected[tuple(first)].item()}"
    )


def flint_batched(flint, layout, x, w, b, m, n, k, scale=1.0, offset=8, fill=SENTINEL,
                  stream=None):
    """Runs the layout's batched GEMM into y inside a buffer pre-filled with `fill` (a 16-bit
    pattern), `offset` halves in (a multiple of 8 keeps y 16-byte aligned). Checks that the
    buffer around the whole of y is untouched, and that x and w are unchanged, and returns
    y as b x m x n FP16."""
    count = b * m * n
    buf = torch.full((offset + count + GUARD_AFTER,), as_int16(fill), dtype=torch.int16,
                     device="cuda")
    y = buf[offset : offset + count].view(torch.float16)
    x_before, w_before = x.clone(), w.clone()

    layout.call(flint, y, x, w, b, m, n, k, scale, stream)
    torch.cuda.synchronize()

    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert (buf[:offset] == guard).all(), f"{layout.flint_name} wrote before y"
    assert (buf[offset + count :] == guard).all(), f"{layout.flint_name} wrote past y"
    assert torch.equal(x.view(torch.int16), x_before.view(torch.int16)), "x changed"
    assert torch.equal(w.view(torch.int16), w_before.view(torch.int16)), "w changed"
    return y.view(b, m, n)


def plain_batched(flint, layout, x, w, b, m, n, k):
    """The plain kernel run on each batch in a host loop, as b x m x n FP16."""
    y = torch.empty((b, m, n), dtype=torch.float16, device="cuda")
    for j in range(b):
        y[j] = layout.plain(flint, x[j], w[j], m, n, k)
    torch.cuda.synchronize()
    return y


def stacked(function, x, w):
    """function(x[j], w[j]) for every batch, stacked."""
    first = function(x[0], w[0])
    out = torch.empty((x.shape[0], *first.shape), dtype=first.dtype, device="cuda")
    out[0] = first
    for j in range(1, x.shape[0]):
        out[j] = function(x[j], w[j])
    return out


def gate_triples(flint, layout, make_inputs, b, m, n, k, scale=1.0, offset=8):
    """Per seed, Flint's output, the peer's and the reference's (the peer and the reference
    computed per batch and stacked, and scaled for NT: the peer rounds first and then scales,
    as PyTorch's attention does)."""
    for seed in gate.SEEDS:
        x, w = make_inputs(layout, b, m, n, k, seed)
        y = flint_batched(flint, layout, x, w, b, m, n, k, scale, offset)
        peer = stacked(layout.peer, x, w)
        ref = stacked(layout.reference, x, w)
        if scale != 1.0:
            peer = peer * scale
            ref = ref * f32(scale)
        yield y, peer, ref


def check_gate(flint, layout, make_inputs, shape, scale=1.0, offset=8):
    gate.gate(layout.factor, gate_triples(flint, layout, make_inputs, *shape, scale, offset))


# -- Bitwise against the plain kernels --------------------------------------

BITWISE_CASES = [
    pytest.param(layout, shape, 8 * (1 + i % 3), id=f"{layout.name}-{shape_id(shape)}")
    for layout in LAYOUTS
    for i, shape in enumerate(layout.shapes)
]


@pytest.mark.parametrize("layout, shape, offset", BITWISE_CASES)
def test_batched_equals_the_plain_kernel_per_batch_bitwise(flint, layout, shape, offset):
    b, m, n, k = shape
    if layout is TN:
        # A split of k changes the summation order, so the comparison needs the plain kernel
        # not to split.
        assert flint.gemm_tn_workspace(m, n, k) == 0
    x, w = random_inputs(layout, b, m, n, k, seed=len(layout.name) + b)

    y = flint_batched(flint, layout, x, w, b, m, n, k, scale=1.0, offset=offset)
    expected = plain_batched(flint, layout, x, w, b, m, n, k)

    assert torch.isfinite(expected).all()
    assert_same_bits(y, expected, f"{layout.flint_name} against {layout.name} per batch")


def assert_same_but_nan_bits(actual, expected, what):
    """NaNs in the same places, and every other value equal bit for bit."""
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan), f"{what}: NaNs in different places"
    assert_same_bits(actual[~nan], expected[~nan], what)


# A shape of each layout, with partial tiles and a k that isn't 2^n.
RANGE_SHAPES = [
    pytest.param(NT, (3, 129, 127, 72), id="nt"),
    pytest.param(NN, (3, 100, 64, 136), id="nn"),
    pytest.param(TN, (3, 72, 16, 72), id="tn"),
]


def near_the_top(layout, b, m, n, k, seed):
    """Both operands scaled so that the sums are about N(0, 40,000): many round near FP16's
    largest values, and about 1 in 10 past 65,520, to inf."""
    std = (40_000.0 / math.sqrt(k)) ** 0.5
    return random_inputs(layout, b, m, n, k, seed, std, std)


def near_the_bottom(layout, b, m, n, k, seed):
    """Both operands about N(0, 1e-3): the sums land in FP16's subnormal range."""
    return random_inputs(layout, b, m, n, k, seed, 1e-3, 1e-3)


@pytest.mark.parametrize("layout, shape", RANGE_SHAPES)
def test_sums_near_fp16s_largest_values_match_the_plain_kernel_bitwise(flint, layout, shape):
    b, m, n, k = shape
    x, w = near_the_top(layout, b, m, n, k, seed=1)

    y = flint_batched(flint, layout, x, w, b, m, n, k)
    expected = plain_batched(flint, layout, x, w, b, m, n, k)

    # The inputs did what they were for: overflow to inf of both signs, and values close
    # below it.
    assert torch.isposinf(expected).any() and torch.isneginf(expected).any()
    finite = expected[torch.isfinite(expected)].abs()
    assert (finite > 60_000).any() and (finite > 32_768).sum() > 10
    assert_same_bits(y, expected, f"{layout.flint_name} near FP16's largest values")


@pytest.mark.parametrize("layout, shape", RANGE_SHAPES)
def test_overflow_gives_inf_with_the_references_sign_and_the_rest_passes_the_gate(
    flint, layout, shape
):
    b, m, n, k = shape
    # FP16 rounds to inf from 65,520 up. A thin band, 0.01% either side, is left out, because
    # FP32 error can decide it either way.
    inf_above, finite_below = 65_520.0 * (1 + 1e-4), 65_520.0 * (1 - 1e-4)

    def triples():
        for seed in gate.SEEDS:
            x, w = near_the_top(layout, b, m, n, k, seed)
            y = flint_batched(flint, layout, x, w, b, m, n, k)
            peer = stacked(layout.peer, x, w)
            ref = stacked(layout.reference, x, w)

            too_big = ref.abs() >= inf_above
            fits = ref.abs() <= finite_below
            assert too_big.any() and fits.any()
            expected = torch.where(ref > 0, math.inf, -math.inf).half()
            assert torch.equal(y[too_big], expected[too_big]), "wrong overflow to inf"
            assert torch.isfinite(y[fits]).all(), "a finite output became inf"
            # The peer can overflow on its own where FP16 terms round first, so the gate
            # takes only what both keep finite.
            both = fits & torch.isfinite(peer)
            assert both.float().mean() > 0.5
            yield y[both], peer[both], ref[both]

    gate.gate(layout.factor, triples())


@pytest.mark.parametrize("layout, shape", RANGE_SHAPES)
def test_subnormal_outputs_match_the_plain_kernel_and_pass_the_gate(flint, layout, shape):
    b, m, n, k = shape

    def triples():
        for seed in gate.SEEDS:
            x, w = near_the_bottom(layout, b, m, n, k, seed)
            y = flint_batched(flint, layout, x, w, b, m, n, k)
            expected = plain_batched(flint, layout, x, w, b, m, n, k)
            assert_same_bits(y, expected, f"{layout.flint_name} in the subnormal range")
            assert (y.abs() < 2.0**-14).float().mean() > 0.9
            assert (y != 0).float().mean() > 0.5
            yield y, stacked(layout.peer, x, w), stacked(layout.reference, x, w)

    gate.gate(layout.factor, triples())


@pytest.mark.parametrize("layout, shape", RANGE_SHAPES)
def test_nans_and_infs_in_the_inputs_match_the_plain_kernel(flint, layout, shape):
    # flint.h promises the plain kernel's bits, except a NaN's own bits: NT's scale and TN's
    # rounding each put the sums through an instruction the plain path doesn't, which needn't
    # keep a NaN's payload. A NaN or inf in x reaches every output that reads it, and the other
    # batches and rows must come out the same.
    b, m, n, k = shape
    x, w = random_inputs(layout, b, m, n, k, seed=2)
    x[layout.x_at(0, 0, 0)] = math.nan
    x[layout.x_at(1, 1, 1)] = math.inf
    x[layout.x_at(2, 2, 2)] = -math.inf
    x[layout.x_at(2, 2, 3)] = math.inf

    y = flint_batched(flint, layout, x, w, b, m, n, k)
    expected = plain_batched(flint, layout, x, w, b, m, n, k)

    assert torch.isnan(expected).any() and torch.isinf(expected).any()
    assert_same_but_nan_bits(y, expected, layout.flint_name)


@pytest.mark.skipif(SANITIZER, reason="65,535 blocks of z are too slow under compute-sanitizer")
def test_the_most_batches_nt(flint):
    # 65,535 batches of the smallest tile, a real run: the grid's z at its limit. The whole
    # result is checked against a float bmm (every batch has its own random data, so a
    # block that found another batch's corner would be far off), and a sample of batches
    # (the first, the last, the edges of the powers of 2 and a few in between) bitwise
    # against the plain kernel.
    b, m, n, k = MAX_BATCHES, 8, 8, 8
    start = time.perf_counter()
    x, w = random_inputs(NT, b, m, n, k)

    y = flint_batched(flint, NT, x, w, b, m, n, k, fill=NAN_BITS)

    assert not torch.isnan(y).any()
    ref = torch.bmm(x.float(), w.float().transpose(1, 2))
    magnitude = torch.bmm(x.abs().float(), w.abs().float().transpose(1, 2))
    # FP16 rounding (2^-11 of the value), and the tensor cores' truncating FP32 adds.
    bound = ref.abs() * 2.0**-10 + magnitude * 2.0**-18 + 2.0**-24
    assert ((y.float() - ref).abs() <= bound).all()

    sample = [0, 1, 2, 255, 256, 4095, 4096, 32767, 32768, 50000, 65533, MAX_BATCHES - 1]
    for j in sample:
        assert_same_bits(y[j], NT.plain(flint, x[j], w[j], m, n, k), f"batch {j}")
    print(f"65,535 batches of 8 x 8 x 8 took {time.perf_counter() - start:.2f} s")


# -- The correctness gate ---------------------------------------------------

# Attention at T = GATE_T and GATE_BATCHES of the 192 batches: each (layout, maker, shape,
# scale).
ATTENTION_CASES = [
    pytest.param(NT, attention_scores, attention_nt_shape(GATE_T, GATE_BATCHES), 0.125,
                 id="nt-scores"),
    pytest.param(NT, attention_dp, attention_nt_shape(GATE_T, GATE_BATCHES), 1.0, id="nt-dP"),
    pytest.param(NT, outliers, attention_nt_shape(GATE_T, GATE_BATCHES), 1.0,
                 id="nt-outliers"),
    pytest.param(NN, attention_o, attention_tn_shape(GATE_T, GATE_BATCHES), 1.0, id="nn-PV"),
    pytest.param(NN, attention_dq, attention_tn_shape(GATE_T, GATE_BATCHES), 1.0, id="nn-dQ"),
    pytest.param(NN, outliers, attention_tn_shape(GATE_T, GATE_BATCHES), 1.0,
                 id="nn-outliers"),
    pytest.param(TN, attention_dv, attention_tn_shape(GATE_T, GATE_BATCHES), 1.0, id="tn-dV"),
    pytest.param(TN, attention_dk, attention_tn_shape(GATE_T, GATE_BATCHES), 1.0, id="tn-dK"),
    pytest.param(TN, outliers, attention_tn_shape(GATE_T, GATE_BATCHES), 1.0,
                 id="tn-outliers"),
]


@pytest.mark.parametrize("layout, make_inputs, shape, scale", ATTENTION_CASES)
def test_attention_inputs_pass_the_gate(flint, layout, make_inputs, shape, scale):
    check_gate(flint, layout, make_inputs, shape, scale)


# The shapes of the bitwise list, but not GPT-2's (the gate runs that above), each with
# realistic inputs and with outlier channels.
ODD_GATE_CASES = [
    pytest.param(layout, shape, make_inputs, 8 * (1 + (i + j) % 3),
                 id=f"{layout.name}-{shape_id(shape)}-{make_inputs.__name__}")
    for layout in LAYOUTS
    for i, shape in enumerate(layout.shapes[:-1])
    for j, make_inputs in enumerate((realistic, outliers))
]


@pytest.mark.parametrize("layout, shape, make_inputs, offset", ODD_GATE_CASES)
def test_odd_shapes_pass_the_gate(flint, layout, shape, make_inputs, offset):
    check_gate(flint, layout, make_inputs, shape, offset=offset)


# TN with a k that the plain kernel would split, which has no bitwise comparison: the
# batched kernel never splits. 64 x 64 is a single tile.
if SANITIZER:
    SPLIT_SHAPES = [(1, 8, 8, 4096), (1, 8, 8, MAX_K)]
else:
    SPLIT_SHAPES = [(2, 64, 64, 4096), (2, 64, 64, MAX_K)]
SPLIT_CASES = [
    pytest.param(shape, make_inputs, id=f"{shape_id(shape)}-{make_inputs.__name__}")
    for shape in SPLIT_SHAPES
    for make_inputs in (gradients, realistic)
]


@pytest.mark.parametrize("shape, make_inputs", SPLIT_CASES)
def test_tn_with_a_k_the_plain_kernel_splits_passes_the_gate(flint, shape, make_inputs):
    _, m, n, k = shape
    assert flint.gemm_tn_workspace(m, n, k) > 0
    check_gate(flint, TN, make_inputs, shape)


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@by_layout
def test_largest_allowed_k_is_exact_on_small_integers(flint, layout):
    # Sums of products from {-1, 0, 1} are integers, which FP32 holds exactly at every step
    # (and FP16 below 2,048), so this checks indexing across the whole k with no rounding
    # to blur it.
    m, n = layout.exact_k_mn
    b, k = 3, MAX_K
    x, w = integers(layout, b, m, n, k, seed=0, bound=1)
    expected = stacked(layout.reference, x, w).half()

    assert expected.abs().max().item() < 2048
    assert_same_bits(flint_batched(flint, layout, x, w, b, m, n, k), expected,
                     f"{layout.flint_name} at k = {k}")


# -- The scale --------------------------------------------------------------


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
@pytest.mark.parametrize("k", [8, 16, 64])
def test_a_sum_past_fp16_stays_finite_if_the_scale_brings_it_back(flint, k, sign):
    # x and w all 128: every sum is k * 16,384, which is 131,072 at k = 8. The scale 1 / k
    # makes it exactly 16,384, and scale 1 gives inf, as rounding first does.
    b, m, n = 3, 9, 17
    x = torch.full((b, m, k), 128.0, dtype=torch.float16, device="cuda")
    w = torch.full((b, n, k), 128.0 * sign, dtype=torch.float16, device="cuda")
    sums = stacked(NT.reference, x, w)
    assert (sums == sign * 16384.0 * k).all()

    scaled = flint_batched(flint, NT, x, w, b, m, n, k, scale=1.0 / k)
    unscaled = flint_batched(flint, NT, x, w, b, m, n, k, scale=1.0)
    peer = stacked(NT.peer, x, w) * (1.0 / k)

    assert (scaled == sign * 16384.0).all()
    assert (unscaled == sign * math.inf).all()
    assert (peer == sign * math.inf).all(), "the peer rounds before it scales, so it overflows"


SCALE_INPUTS = {
    # The inputs' std, and the least fraction of outputs that must stay in the comparison.
    # Sums about N(0, 8): almost every output is normal after scaling (the zero rows,
    # 1 in 136, are left out).
    "unit": (1.0, 0.98),
    # Sums about N(0, 8e-4): outputs land around FP16's smallest normal, 2^-14 = 6.1e-5,
    # so the mask below cuts through them, and the subnormals are left out.
    "small": (0.01, 0.4),
}


def scale_inputs(variant, b, m, n, k):
    std, _ = SCALE_INPUTS[variant]
    x, w = random_inputs(NT, b, m, n, k, seed=4, x_std=std, w_std=std)
    # A few exact zeros, so that zero outputs occur.
    x[:, 3, :] = 0
    return x, w


@pytest.mark.parametrize("variant", SCALE_INPUTS)
@pytest.mark.parametrize("scale", [0.5, 0.25, 0.125])
def test_a_power_of_2_scale_gives_the_plain_result_times_it(flint, scale, variant):
    # FP32 sums times a power of 2 are exact, so rounding the scaled sum is the plain
    # result's rounding shifted down: bit for bit wherever the scaled result is normal. (Below
    # 2^-14 FP16's spacing stops shrinking, and the two can differ.)
    b, m, n, k = 4, 136, 72, 64
    x, w = scale_inputs(variant, b, m, n, k)

    plain = flint_batched(flint, NT, x, w, b, m, n, k, scale=1.0)
    scaled = flint_batched(flint, NT, x, w, b, m, n, k, scale=scale)
    product = plain.float() * scale  # exact
    expected = product.half()

    # The mask is on the exact product, not its rounding: a product just below 2^-14 can
    # round up to it.
    keep = plain.isfinite() & (product.abs() >= 2.0**-14)
    assert keep.sum().item() >= SCALE_INPUTS[variant][1] * keep.numel()
    assert_same_bits(scaled[keep], expected[keep], f"scale {scale}")


def test_scale_minus_1_flips_every_sign_bit(flint):
    b, m, n, k = 4, 136, 72, 64
    x, w = scale_inputs("unit", b, m, n, k)

    plain = flint_batched(flint, NT, x, w, b, m, n, k, scale=1.0)
    flipped = flint_batched(flint, NT, x, w, b, m, n, k, scale=-1.0)

    assert (plain[:, 3, :] == 0).all(), "the zero rows are there to give zero outputs"
    assert_same_bits(flipped, (plain.view(torch.int16) ^ -32768).view(torch.float16),
                     "scale -1")


def test_scale_0_gives_only_zeros(flint):
    b, m, n, k = 4, 136, 72, 64
    x, w = scale_inputs("unit", b, m, n, k)

    y = flint_batched(flint, NT, x, w, b, m, n, k, scale=0.0, fill=NAN_BITS)

    assert (y == 0).all()


@pytest.mark.parametrize("shape", [(6, 136, 72, 64), (3, 100, 130, 24)], ids=shape_id)
def test_a_scale_that_is_not_a_power_of_2_passes_the_gate(flint, shape):
    check_gate(flint, NT, gradients, shape, scale=1.0 / 3.0)


@pytest.mark.parametrize("k, bound", [(8, 60), (24, 40), (64, 30)])
def test_rounding_happens_once_after_the_scale(flint, k, bound):
    # Integer inputs, so the FP32 sums are exact integers (below 2^24), many of them past
    # 2,048 where FP16 can't hold every integer. A scale of 1/3 on a sum of 2,049 gives 683
    # when the sum is scaled and then rounded, but 2,048 / 3 = 682.67 rounds to 682.5 when
    # the sum rounds first (to 2,048, a tie to even).
    b, m, n = 3, 40, 24
    x, w = integers(NT, b, m, n, k, seed=1, bound=bound)
    sums = torch.bmm(x.double(), w.double().transpose(1, 2))
    assert sums.abs().max().item() < 2**24
    scale = f32(1.0 / 3.0)

    expected = (sums * scale).half()
    # The same in FP32, as the kernel multiplies: only the double rounding of float64 to
    # FP32 to FP16 could separate them, and it doesn't.
    assert_same_bits(expected, (sums.float() * scale).half(), "the float64 and FP32 references")
    round_first = (sums.half().double() * scale).half()
    different = (expected != round_first).sum().item()
    assert different > 0.05 * expected.numel(), different

    y = flint_batched(flint, NT, x, w, b, m, n, k, scale=1.0 / 3.0)

    assert_same_bits(y, expected, "scale 1/3 on integer sums")


# -- Batches ----------------------------------------------------------------

BATCH_SHAPE = (5, 136, 72, 40)  # valid for all three layouts


@by_layout
def test_permuting_the_batches_permutes_the_results(flint, layout):
    b, m, n, k = BATCH_SHAPE
    x, w = random_inputs(layout, b, m, n, k, seed=5)
    perm = torch.randperm(b, generator=generator(1), device="cuda")
    assert (perm != torch.arange(b, device="cuda")).any()

    y = flint_batched(flint, layout, x, w, b, m, n, k, scale=0.75 if layout is NT else 1.0)
    permuted = flint_batched(flint, layout, x[perm].contiguous(), w[perm].contiguous(), b, m,
                             n, k, scale=0.75 if layout is NT else 1.0)

    assert_same_bits(permuted, y[perm], f"{layout.flint_name} on permuted batches")


SINGLE_CASES = [
    pytest.param(NT, 1.0, id="nt-1"),
    pytest.param(NT, 1.0 / 3.0, id="nt-1/3"),
    pytest.param(NN, 1.0, id="nn"),
    pytest.param(TN, 1.0, id="tn"),
]


@pytest.mark.parametrize("layout, scale", SINGLE_CASES)
def test_a_batch_alone_is_the_same_as_in_many_and_the_plain_kernel(flint, layout, scale):
    b, m, n, k = BATCH_SHAPE
    x, w = random_inputs(layout, b, m, n, k, seed=6)

    many = flint_batched(flint, layout, x, w, b, m, n, k, scale=scale)
    for j in range(b):
        one = flint_batched(flint, layout, x[j : j + 1], w[j : j + 1], 1, m, n, k, scale=scale)
        assert_same_bits(one, many[j : j + 1], f"batch {j} alone against in {b}")
        if scale == 1.0:
            assert_same_bits(one[0], layout.plain(flint, x[j], w[j], m, n, k),
                             f"batch {j} alone against the plain kernel")


@pytest.mark.parametrize("which", ["x", "w"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
@by_layout
def test_a_non_finite_value_reaches_only_its_batch(flint, layout, bad, which):
    # One bad element of x spoils one row of its batch's y, and one of w one column. Every
    # other element, of that batch and of the others, is bitwise the clean run's.
    b, m, n, k = BATCH_SHAPE
    victim, row, col, kk = 3, 77, 29, 13
    x, w = random_inputs(layout, b, m, n, k, seed=7)
    clean = flint_batched(flint, layout, x, w, b, m, n, k)
    if which == "x":
        x = x.clone()
        x[layout.x_at(victim, row, kk)] = bad
    else:
        w = w.clone()
        w[layout.w_at(victim, col, kk)] = bad

    dirty = flint_batched(flint, layout, x, w, b, m, n, k)

    spoiled = dirty[victim, row, :] if which == "x" else dirty[victim, :, col]
    assert not torch.isfinite(spoiled).any()
    assert torch.isfinite(clean).all()
    dirty, clean = dirty.clone(), clean.clone()
    for y in (dirty, clean):
        if which == "x":
            y[victim, row, :] = 0
        else:
            y[victim, :, col] = 0
    assert_same_bits(dirty, clean, "the rest of y")


# -- Everything written, and nothing else ------------------------------------

PARTIAL_CASES = [
    pytest.param(layout, shape, 8 * (1 + i % 3), id=f"{layout.name}-{shape_id(shape)}")
    for layout in LAYOUTS
    for i, shape in enumerate(layout.partial_shapes)
]


@pytest.mark.parametrize("layout, shape, offset", PARTIAL_CASES)
def test_every_output_is_written_and_nothing_else(flint, layout, shape, offset):
    # y starts as NaN: a store that is skipped (a partial tile's, an odd m or n's, a later
    # batch's) leaves one, and flint_batched checks the guard bands around the whole of y.
    b, m, n, k = shape
    x, w = random_inputs(layout, b, m, n, k, seed=3)

    y = flint_batched(flint, layout, x, w, b, m, n, k, offset=offset, fill=NAN_BITS)

    assert not torch.isnan(y).any()
    assert torch.isfinite(y).all()
    if layout is TN:
        assert flint.gemm_tn_workspace(m, n, k) == 0
    assert_same_bits(y, plain_batched(flint, layout, x, w, b, m, n, k), "against the plain kernel")


# -- Determinism ------------------------------------------------------------

DETERMINISM_CASES = [
    pytest.param(layout, shape, id=f"{layout.name}-{shape_id(shape)}")
    for layout in LAYOUTS
    for shape in layout.determinism_shapes
]


@pytest.mark.parametrize("layout, shape", DETERMINISM_CASES)
def test_run_twice_is_bitwise_identical(flint, layout, shape):
    b, m, n, k = shape
    x, w = outliers(layout, b, m, n, k, 0)
    scale = 0.125 if layout is NT else 1.0
    # Different NaN patterns to start from, so a store that is skipped, or one that lands
    # in only one run, shows as a difference or a leftover NaN.
    first = flint_batched(flint, layout, x, w, b, m, n, k, scale, fill=NAN_BITS)
    second = flint_batched(flint, layout, x, w, b, m, n, k, scale, fill=OTHER_NAN_BITS)
    third = flint_batched(flint, layout, x, w, b, m, n, k, scale, fill=0)

    assert not torch.isnan(first).any()
    assert not torch.isnan(second).any()
    assert_same_bits(first, second, "second run")
    assert_same_bits(first, third, "third run")


# -- Aliasing and streams ---------------------------------------------------


@pytest.mark.parametrize("sharing", ["same", "shifted"])
@by_layout
def test_x_and_w_may_share_memory(flint, layout, sharing):
    # "same": one tensor is both x and w (NT: y = a a^T, NN: a a, TN: a^T a). "shifted":
    # w starts partway into x's bytes. Either way the result is the one for separate
    # copies, and the plain kernels'.
    shape = layout.alias_shape
    b, m, n, k = shape
    x_shape, w_shape = layout.x_shape(*shape), layout.w_shape(*shape)
    assert x_shape == w_shape
    count = math.prod(x_shape)
    g = generator(0)
    if sharing == "same":
        buf = normal((count,), 1.0, g)
        x, w = buf.view(x_shape), buf.view(w_shape)
    else:
        shift = count // 2 // 8 * 8
        buf = normal((count + shift,), 1.0, g)
        x, w = buf[:count].view(x_shape), buf[shift:].view(w_shape)

    shared = flint_batched(flint, layout, x, w, b, m, n, k)
    separate = flint_batched(flint, layout, x.clone(), w.clone(), b, m, n, k)

    assert_same_bits(shared, separate, "shared against separate memory")
    assert_same_bits(shared, plain_batched(flint, layout, x, w, b, m, n, k), "against plain")


STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
@by_layout
def test_batched_gemm_runs_on_the_given_stream(flint, layout, mode):
    b, m, n, k = 3, 64, 64, 64
    scale = 0.5 if layout is NT else 1.0
    x = torch.zeros(layout.x_shape(b, m, n, k), dtype=torch.float16, device="cuda")
    w = torch.ones(layout.w_shape(b, m, n, k), dtype=torch.float16, device="cuda")
    y = torch.zeros((b, m, n), dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fill finishes only after the sleep, so the product is k (times the scale) only
    # if the GEMM is queued on the same stream behind it. With the current stream the call
    # has no stream argument, and torch's current stream is the side one; with the default
    # stream there is no side one at all.
    if mode == "default-stream":
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(1)
        layout.call(flint, y, x, w, b, m, n, k, scale)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            x.fill_(1)
            layout.call(flint, y, x, w, b, m, n, k, scale,
                        stream=stream if mode == "explicit-stream" else None)
        stream.synchronize()

    assert (y == k * scale).all()


# -- Argument errors --------------------------------------------------------

NAMES = ("y", "x", "w")


def small_problem(batches=4, m=16, n=16, k=16):
    """Valid tensors by name for any layout, in buffers with room to spare."""
    return {
        "y": torch.zeros(batches * m * n + 64, dtype=torch.float16, device="cuda"),
        "x": torch.ones(batches * m * k + 64, dtype=torch.float16, device="cuda"),
        "w": torch.ones(batches * n * k + 64, dtype=torch.float16, device="cuda"),
    }


def call(flint, layout, pointers, batches=4, m=16, n=16, k=16, scale=1.0):
    layout.call(flint, pointers["y"], pointers["x"], pointers["w"], batches, m, n, k, scale)


def expect_invalid_argument(flint, layout, reason, pointers, **sizes):
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, layout, pointers, **sizes)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in (layout.flint_name, reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("name", NAMES)
@by_layout
def test_null_pointer(flint, layout, name):
    expect_invalid_argument(flint, layout, f"{name} is null", {**small_problem(), name: None})


@pytest.mark.parametrize("name", NAMES)
@by_layout
def test_misaligned_pointer(flint, layout, name):
    addresses = {n: t.data_ptr() for n, t in small_problem().items()}
    addresses[name] += 2
    expect_invalid_argument(flint, layout, f"{name} is not 16-byte aligned", addresses)


@pytest.mark.parametrize("name", NAMES)
@by_layout
def test_a_buffer_may_not_run_past_the_address_space(flint, layout, name):
    # The check runs before any memory is touched, so a made-up address will do.
    expect_invalid_argument(flint, layout, f"{name} runs past the end of the address space",
                            {**small_problem(), name: 2**64 - 16})


@by_layout
def test_pointers_are_checked_in_order(flint, layout):
    # Per buffer: null, then alignment, then the address space; y, x, then w.
    a = {n: t.data_ptr() for n, t in small_problem().items()}
    expect_invalid_argument(flint, layout, "y is not 16-byte aligned",
                            {**a, "y": a["y"] + 2, "x": None})
    expect_invalid_argument(flint, layout, "y is null", {**a, "y": None, "x": a["x"] + 2})
    expect_invalid_argument(flint, layout, "x is not 16-byte aligned",
                            {**a, "x": a["x"] + 2, "w": None})
    expect_invalid_argument(flint, layout, "y runs past the end", {"y": 2**64 - 16, "x": None,
                                                                   "w": None})


@pytest.mark.parametrize("k", [0, 4, 12, 20])
@pytest.mark.parametrize("layout", [NT, NN], ids=["nt", "nn"])
def test_k_must_be_a_positive_multiple_of_8(flint, layout, k):
    expect_invalid_argument(flint, layout, f"k ({k}) must be a positive multiple of 8",
                            small_problem(), k=k)


def test_tn_k_must_be_positive_and_can_be_any(flint):
    expect_invalid_argument(flint, TN, "k must be positive", small_problem(), k=0)
    # A k of 5 is fine: x and w are 5 rows of 16, and the other rows read as zeros.
    pointers = small_problem()
    for tensor in pointers.values():
        tensor[:] = 1
    call(flint, TN, pointers, k=5)
    torch.cuda.synchronize()
    assert (pointers["y"][: 4 * 256] == 5).all()
    assert (pointers["y"][4 * 256 :] == 1).all()


@pytest.mark.parametrize("n", [1, 4, 12])
def test_nn_n_must_be_a_multiple_of_8(flint, n):
    # NT takes any n, since its w rows are k long. NN's are n long.
    expect_invalid_argument(flint, NN, f"n ({n}) must be a multiple of 8", small_problem(), n=n)
    pointers = small_problem(n=n)
    NT.call(flint, pointers["y"], pointers["x"], pointers["w"], 4, 16, n, 16)
    torch.cuda.synchronize()
    assert (pointers["y"][: 4 * 16 * n] == 16).all()
    assert (pointers["y"][4 * 16 * n :] == 0).all()


@pytest.mark.parametrize("which, m, n", [("m", 12, 16), ("n", 16, 12), ("m", 4, 16), ("n", 16, 1)])
def test_tn_m_and_n_must_be_multiples_of_8(flint, which, m, n):
    value = m if which == "m" else n
    expect_invalid_argument(flint, TN, f"{which} ({value}) must be a multiple of 8",
                            small_problem(), m=m, n=n)


def too_large_cases():
    """(layout, id, sizes, reason) just past each limit: batches 65,536; m 1,025 (1,032 for
    TN, which needs a multiple of 8); n 8,388,481 for NT and the next multiple of 8 above
    for NN and TN; k 65,544 for NT and NN and 65,537 for TN. And values near 2^64."""
    huge = 2**64 - 8
    cases = []
    for layout in (NT, NN):
        past_n = MAX_MN + 1 if layout is NT else MAX_MN + 8
        cases += [
            (layout, "batches", dict(batches=MAX_BATCHES + 1),
             f"batches ({MAX_BATCHES + 1}) can be up to {MAX_BATCHES}"),
            (layout, "m", dict(m=MAX_M + 1), f"m ({MAX_M + 1}) can be up to {MAX_M}"),
            (layout, "n", dict(n=past_n),
             f"n ({past_n}) can be up to {MAX_MN}, and k (16) up to {MAX_K}"),
            (layout, "k", dict(k=MAX_K + 8),
             f"n (16) can be up to {MAX_MN}, and k ({MAX_K + 8}) up to {MAX_K}"),
            (layout, "m-past-the-plain-limit", dict(m=MAX_MN + 1),
             f"m ({MAX_MN + 1}) can be up to {MAX_M}"),
            (layout, "huge-batches", dict(batches=2**64 - 1),
             f"batches ({2**64 - 1}) can be up to {MAX_BATCHES}"),
            (layout, "huge-m", dict(m=2**64 - 1), f"m ({2**64 - 1}) can be up to {MAX_M}"),
            (layout, "huge-n", dict(n=huge),
             f"n ({huge}) can be up to {MAX_MN}, and k (16) up to {MAX_K}"),
            (layout, "huge-k", dict(k=huge),
             f"n (16) can be up to {MAX_MN}, and k ({huge}) up to {MAX_K}"),
        ]
    cases += [
        (TN, "batches", dict(batches=MAX_BATCHES + 1),
         f"batches ({MAX_BATCHES + 1}) can be up to {MAX_BATCHES}"),
        (TN, "m", dict(m=MAX_M + 8), f"m ({MAX_M + 8}) can be up to {MAX_M}"),
        (TN, "m-past-the-plain-limit", dict(m=MAX_MN + 8), f"{MAX_MN + 8} x 16 x 16 is too large"),
        (TN, "n", dict(n=MAX_MN + 8), f"16 x {MAX_MN + 8} x 16 is too large"),
        (TN, "k", dict(k=MAX_K + 1), f"16 x 16 x {MAX_K + 1} is too large"),
        (TN, "huge-batches", dict(batches=2**64 - 1),
         f"batches ({2**64 - 1}) can be up to {MAX_BATCHES}"),
        (TN, "huge-m", dict(m=huge), f"{huge} x 16 x 16 is too large"),
        (TN, "huge-n", dict(n=huge), f"16 x {huge} x 16 is too large"),
        (TN, "huge-k", dict(k=2**64 - 1), f"16 x 16 x {2**64 - 1} is too large"),
    ]
    return [pytest.param(layout, sizes, reason, id=f"{layout.name}-{name}")
            for layout, name, sizes, reason in cases]


@pytest.mark.parametrize("layout, sizes, reason", too_large_cases())
def test_too_large_dimensions(flint, layout, sizes, reason):
    # The size checks run before any memory is touched, so small tensors do.
    expect_invalid_argument(flint, layout, reason, small_problem(), **sizes)


MOST_CASES = [
    pytest.param(dict(batches=MAX_BATCHES), id="batches"),
    pytest.param(dict(m=MAX_M), id="m"),
    pytest.param(dict(n=MAX_MN), id="n"),
    pytest.param(dict(k=MAX_K), id="k"),
    pytest.param(dict(batches=MAX_BATCHES, m=MAX_M, n=MAX_MN, k=MAX_K), id="all"),
]


@pytest.mark.parametrize("sizes", MOST_CASES)
@by_layout
def test_the_largest_sizes_pass_the_size_checks(flint, layout, sizes):
    # The limits themselves aren't errors of size: the call gets on to the pointers (a null
    # y), even where batches * m * n * 2 is 1.1e15 bytes.
    expect_invalid_argument(flint, layout, "y is null", {**small_problem(), "y": None}, **sizes)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_nt_scale_must_be_finite(flint, bad):
    expect_invalid_argument(flint, NT, "must be finite", small_problem(), scale=bad)
    assert "scale" in flint.last_error()
    if math.isinf(bad):
        assert f"scale ({'-' if bad < 0 else ''}inf) must be finite" in flint.last_error()
    else:
        assert "nan" in flint.last_error()


def test_nt_scale_is_checked_after_k_and_before_the_rest(flint):
    # k and n first, then scale, then batches and m, then the pointers.
    nan = math.nan
    expect_invalid_argument(flint, NT, "k (12) must be a positive multiple of 8", small_problem(),
                            k=12, scale=nan)
    expect_invalid_argument(flint, NT, f"n ({MAX_MN + 1}) can be up to", small_problem(),
                            n=MAX_MN + 1, scale=nan)
    expect_invalid_argument(flint, NT, "must be finite", small_problem(), batches=2**40, scale=nan)
    expect_invalid_argument(flint, NT, "must be finite", small_problem(), m=2**40, scale=nan)
    expect_invalid_argument(flint, NT, "must be finite", {"y": None, "x": None, "w": None},
                            scale=nan)


@pytest.mark.parametrize("scale", [0.0, -1.0, -0.0, 1e-45, 3.4e38, -3.4e38], ids=str)
def test_nt_scales_that_are_finite_are_accepted(flint, scale):
    pointers = small_problem()
    call(flint, NT, pointers, scale=scale)
    torch.cuda.synchronize()
    y = pointers["y"]
    # Every sum is 16. The tail of the buffer isn't y.
    assert (y[4 * 256 :] == 0).all()
    if scale == 0.0:
        assert (y[: 4 * 256] == 0).all()
    elif scale == -1.0:
        assert (y[: 4 * 256] == -16).all()


# For the overlap tests: with 4 batches of 16 x 16 x 16, each of y, x and w covers 2,048
# bytes, and a batch 512 of them.
EXTENT, SPACING = 2048, 4096

# Where x (or w) starts relative to y, in bytes. In the first three, y's first batch
# doesn't touch x's first batch, and the overlap shows only in the extents as a whole: x
# starts at y's third batch, at y's last batch, and 16 bytes before y's end. Then the same
# address, 16 bytes inside y, and x before y, ending 16 bytes or 1,536 bytes into y.
OVERLAPPING = {
    "third-batch": 1024,
    "last-batch": 1536,
    "last-16-bytes": EXTENT - 16,
    "same-start": 0,
    "inside-y": 16,
    "ends-16-bytes-into-y": 16 - EXTENT,
    "ends-in-the-third-batch": 1024 - EXTENT + 512,
}


@pytest.mark.parametrize("label", OVERLAPPING)
@pytest.mark.parametrize("name", ["x", "w"])
@by_layout
def test_y_may_not_overlap_x_or_w(flint, layout, name, label):
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no
    # harm. Overlaps that appear only at a later batch are overlaps of the whole extent.
    arena = torch.zeros(8 * SPACING, dtype=torch.uint8, device="cuda")
    base = arena.data_ptr() + 2 * SPACING
    far = base + 4 * SPACING
    pointers = {"y": base, "x": far, "w": far}
    pointers[name] = base + OVERLAPPING[label]
    expect_invalid_argument(flint, layout, f"y and {name} overlap", pointers)


@pytest.mark.parametrize("order", ["y-x-w", "x-y-w", "x-w-y", "w-y-x"])
@by_layout
def test_buffers_next_to_each_other_are_not_overlapping(flint, layout, order):
    # y, x and w laid end to end in one buffer with room around it, with all-ones inputs:
    # every sum is 16 (8 with NT's scale of 1/2).
    names = order.split("-")
    buf = torch.zeros(5 * EXTENT, dtype=torch.uint8, device="cuda")
    views = {name: buf[(i + 1) * EXTENT : (i + 2) * EXTENT] for i, name in enumerate(names)}
    views["x"].view(torch.float16).fill_(1)
    views["w"].view(torch.float16).fill_(1)
    before = buf.clone()

    layout.call(flint, views["y"], views["x"], views["w"], 4, 16, 16, 16, 0.5)
    torch.cuda.synchronize()

    assert (views["y"].view(torch.float16) == (8.0 if layout is NT else 16.0)).all()
    # Every byte outside y (x and w included) is as it was.
    start = (names.index("y") + 1) * EXTENT
    outside = torch.ones_like(buf, dtype=torch.bool)
    outside[start : start + EXTENT] = False
    assert torch.equal(buf[outside], before[outside])


@pytest.mark.parametrize("batches, m, n", [
    (0, 16, 16), (4, 0, 16), (4, 16, 0), (0, 0, 0),
    (0, 2**40, 2**40), (2**64 - 1, 0, 16), (2**64 - 1, 16, 0), (0, 2**64 - 1, 5),
], ids=str)
@by_layout
def test_empty_problems_do_nothing(flint, layout, batches, m, n):
    # Not even the null pointers, a k of 0 or a NaN scale are looked at, and sizes past
    # the limits are no error when there is nothing to do.
    pointers = small_problem()
    pointers["y"].fill_(3)

    layout.call(flint, pointers["y"], pointers["x"], pointers["w"], batches, m, n, 0)
    layout.call(flint, None, None, None, batches, m, n, 0, scale=math.nan)
    layout.call(flint, None, None, None, batches, m, n, 12)
    torch.cuda.synchronize()

    assert (pointers["y"] == 3).all()


@by_layout
def test_a_valid_call_works_after_errors(flint, layout):
    pointers = small_problem()
    y = pointers["y"]
    y.fill_(7)
    bad_sizes = [dict(k=0), dict(batches=MAX_BATCHES + 1), dict(m=MAX_M + 8)]
    bad_sizes += [dict(k=12)] if layout is not TN else [dict(m=12), dict(n=12)]
    if layout is NN:
        bad_sizes.append(dict(n=12))
    for sizes in bad_sizes:
        with pytest.raises(flint_lib.FlintError):
            call(flint, layout, pointers, **sizes)
    with pytest.raises(flint_lib.FlintError):
        call(flint, layout, {**pointers, "y": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, layout, {**pointers, "w": pointers["w"].data_ptr() + 2})
    if layout is NT:
        with pytest.raises(flint_lib.FlintError):
            call(flint, layout, pointers, scale=math.nan)
    # The failed calls wrote nothing.
    torch.cuda.synchronize()
    assert (y == 7).all()

    call(flint, layout, pointers)
    torch.cuda.synchronize()

    assert (y[: 4 * 256] == 16).all()
    assert (y[4 * 256 :] == 7).all()
