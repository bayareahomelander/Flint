"""flint_gemm_nn_dgelu: GELU's backward pass in GPT-2's MLP, the up
projection's input gradient. It computes the FP32 sums dh = x w exactly as
flint_gemm_nn does, multiplies each by the derivative of GPT-2's tanh GELU at
pre (the forward's stored input, from flint_gemm_nt_bias_gelu), in FP32, and
rounds the product to FP16 y once.

The tests cover the correctness gate (the peer is PyTorch's FP16 matmul
followed by its tanh GELU backward, which rounds dh to FP16 before the
derivative, so it is the coarser of the two); the derivative itself, with
exactly known sums so that a dense sweep over every finite FP16 pre holds y to
one FP16 ulp (or, near the derivative's root, an absolute bound) and to the
correctly rounded value wherever that is well defined; the exact links to
flint_gemm_nn at pre = +inf, 0 and -inf; and
non-finite pre, FP16's range, determinism, stream ordering, and argument
errors.

The kernel's main loop is nearly flint_gemm_nn's; test_sass.py checks what it
keeps."""

import math
import types

import numpy as np
import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, clamp_half, generator, normal

# Under the sanitizer the GPT-2 shapes shrink.
GPT2_M = 64 if SANITIZER else 1024

# The input gradient of the MLP's up projection, dh = d_out W_proj, is the one
# with a GELU in front of it: m tokens, n = 3072 hidden units, k = 768 outputs.
GPT2_N, GPT2_K = 3072, 768

# The outputs of a guarded call sit a few halves past the start of a buffer
# that is filled with a pattern, which must be intact around them afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A


def shape_id(shape):
    return "x".join(map(str, shape))


# -- Inputs -----------------------------------------------------------------
# Every maker returns x (m x k), w (k x n) and pre (m x n).


def realistic(m, n, k, seed):
    """The gradient of the loss with respect to the layer's output about
    N(0, 1), weights about N(0, 0.02), and pre-activations about N(0, 2)."""
    g = generator(seed)
    return normal((m, k), 1.0, g), normal((k, n), 0.02, g), normal((m, n), 2.0, g)


def outliers(m, n, k, seed):
    """Realistic inputs, with about 0.5% of x's channels and of pre's columns
    (at least 2 of each) near +-3,000 in every row, as in GPT-2's activations:
    the derivative there is 0 or 1."""
    x, w, pre = realistic(m, n, k, seed)
    g = generator(10_000 + seed)
    for tensor, size in ((x, k), (pre, n)):
        count = max(2, round(0.005 * size))
        channels = torch.randperm(size, generator=g, device="cuda")[:count]
        signs = torch.randint(0, 2, (count,), generator=g, device="cuda") * 2.0 - 1.0
        jitter = 0.9 + 0.2 * torch.rand((m, count), generator=g, device="cuda")
        tensor[:, channels] = (3000.0 * signs * jitter).half()
    return x, w, pre


INPUTS = {"realistic": realistic, "outliers": outliers}


# -- Running Flint ----------------------------------------------------------


def bits(y):
    """y's FP16 bit patterns, which compare equal for equal bits, NaNs and -0.0
    included."""
    return y.contiguous().view(torch.int16)


def flint_gemm_dgelu(flint, x, w, pre, offset=8, fill=SENTINEL):
    """Runs flint_gemm_nn_dgelu into y inside a buffer pre-filled with `fill`
    (a 16-bit pattern), offset halves in; pre is copied into a pre-filled
    buffer of its own, 32 - offset halves in, so offsets of 8, 16 and 24 (which
    keep both 16-byte aligned) give both. Checks the buffer around y is
    untouched and so is all of pre's, and returns y as an m x n FP16 view."""
    m, k = x.shape
    n = w.shape[1]
    pre_offset = 32 - offset
    y_buf = torch.full((offset + m * n + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    pre_buf = torch.full((pre_offset + m * n + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    y = y_buf[offset : offset + m * n].view(torch.float16)
    pre_in = pre_buf[pre_offset : pre_offset + m * n]
    pre_in.copy_(bits(pre).reshape(-1))
    pre_before = pre_buf.clone()
    flint.gemm_nn_dgelu(y, x, w, pre_in.view(torch.float16), m, n, k)
    torch.cuda.synchronize()
    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert (y_buf[:offset] == guard).all(), "flint_gemm_nn_dgelu wrote before y"
    assert (y_buf[offset + m * n :] == guard).all(), "flint_gemm_nn_dgelu wrote past y"
    assert torch.equal(pre_buf, pre_before), "pre changed"
    return y.view(m, n)


def flint_gemm_plain(flint, x, w):
    """flint_gemm_nn's output, which pre = +-inf and 0 relate y to."""
    m, k = x.shape
    n = w.shape[1]
    y = torch.empty((m, n), dtype=torch.float16, device="cuda")
    flint.gemm_nn(y, x, w, m, n, k)
    torch.cuda.synchronize()
    return y


def gate_triples(flint, make_inputs, m, n, k, offset=8):
    """Per seed, Flint's y, the peer's and the reference's."""
    for seed in gate.SEEDS:
        x, w, pre = make_inputs(m, n, k, seed)
        yield (
            flint_gemm_dgelu(flint, x, w, pre, offset),
            gate.peer_gemm_nn_dgelu(x, w, pre),
            gate.reference_dgelu(x, w, pre),
        )


# -- The gate ---------------------------------------------------------------


@pytest.mark.parametrize("variant", INPUTS)
def test_gpt2_shape_passes_the_gate(flint, variant):
    # A gradient, so the standard allows 3x the peer's error.
    gate.gate(gate.GRADIENT_FACTOR, gate_triples(flint, INPUTS[variant], GPT2_M, GPT2_N, GPT2_K))


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@pytest.mark.parametrize("variant", INPUTS)
def test_gpt2_shape_at_8192_tokens_passes_the_gate(flint, variant):
    gate.gate(gate.GRADIENT_FACTOR, gate_triples(flint, INPUTS[variant], 8192, GPT2_N, GPT2_K))


# n and k multiples of 8 but not of 128 or 32, odd m, and single rows.
ODD_SHAPES = [
    (1, 8, 8),
    (7, 16, 24),
    (127, 136, 40),
    (129, 120, 136),
    (200, 776, 776),
    (333, 8, 8),
    (1, 768, 50304),
    (130, 16, 16),
    # Tiles go out in groups of up to 8 rows of tiles. 5 rows of tiles make
    # a single short group. 9, 13, 17 and 33 end in a short group of 1 or 5
    # rows after full ones, whose spare blocks exit, with one column of
    # tiles or several.
    (637, 296, 24),
    (1152, 8, 8),
    (1614, 200, 32),
    (2116, 136, 40),
    (4224, 8, 16),
]

# Offsets rotate through 8, 16 and 24 halves.
ODD_CASES = [
    pytest.param(shape, variant, 8 * (1 + (i + j) % 3), id=f"{shape_id(shape)}-{variant}")
    for i, shape in enumerate(ODD_SHAPES)
    for j, variant in enumerate(INPUTS)
]


@pytest.mark.parametrize("shape, variant, offset", ODD_CASES)
def test_odd_shapes_pass_the_gate_and_stay_inside_y(flint, shape, variant, offset):
    # flint_gemm_dgelu also checks that y's guard regions and pre are intact.
    m, n, k = shape
    gate.gate(gate.GRADIENT_FACTOR, gate_triples(flint, INPUTS[variant], m, n, k, offset))


# -- FP16's range -----------------------------------------------------------


def large_x(m, n, k, seed):
    g = generator(seed)
    return (
        clamp_half(torch.randn((m, k), generator=g, device="cuda") * 20000, 65504),
        normal((k, n), 0.02, g),
        normal((m, n), 2.0, g),
    )


def large_w(m, n, k, seed):
    g = generator(seed)
    return (
        normal((m, k), 0.02, g),
        clamp_half(torch.randn((k, n), generator=g, device="cuda") * 20000, 65504),
        normal((m, n), 2.0, g),
    )


def small_both(m, n, k, seed):
    """Magnitudes 1e-3 to 1e-2 in both x and w, so the sums sit around FP16's
    smallest normal, 6.1e-5, and the products below it."""
    g = generator(seed)
    x, w = (
        (10 ** (-3 + torch.rand(shape, generator=g, device="cuda"))
         * (torch.randint(0, 2, shape, generator=g, device="cuda") * 2.0 - 1.0)).half()
        for shape in ((m, k), (k, n))
    )
    return x, w, normal((m, n), 2.0, g)


def wide_pre(m, n, k, seed):
    """Realistic x and w, with pre from 1e-3 to FP16's largest finite value,
    either sign, spread evenly in the exponent: most of it where the derivative
    is 0 or 1, and the rest around it."""
    x, w, _ = realistic(m, n, k, seed)
    g = generator(20_000 + seed)
    scale = 10 ** (-3 + torch.rand((m, n), generator=g, device="cuda") * (math.log10(65504) + 3))
    signs = torch.randint(0, 2, (m, n), generator=g, device="cuda") * 2.0 - 1.0
    return x, w, clamp_half(scale * signs, 65504)


# n is a multiple of 8 (72 for the layers' 70 and 65).
MAGNITUDE_SHAPES = [
    ("large_x", large_x, (130, 72, 104)),
    ("large_w", large_w, (130, 72, 104)),
    ("small_k8", small_both, (97, 72, 8)),
    ("small_k64", small_both, (97, 72, 64)),
    ("wide_pre", wide_pre, (130, 72, 104)),
]


@pytest.mark.parametrize(
    "make_inputs, shape",
    [pytest.param(make_inputs, shape, id=name) for name, make_inputs, shape in MAGNITUDE_SHAPES],
)
def test_fp16_range_inputs_pass_the_gate(flint, make_inputs, shape):
    m, n, k = shape
    if make_inputs in (large_x, large_w):
        # The point of these inputs is large finite outputs, so check they are.
        top = gate.reference_dgelu(*make_inputs(m, n, k, 0)).abs().max().item()
        assert 1000 < top < 65504, top
    gate.gate(gate.GRADIENT_FACTOR, gate_triples(flint, make_inputs, m, n, k))


# -- The derivative ---------------------------------------------------------


def dgelu64(values):
    """GELU's derivative, the tanh approximation, in float64, and the exponent
    a of its sigmoid. With c = sqrt(2/pi), a = 2c (s + 0.044715 s^3) and sigma =
    1 / (1 + exp(-a)), GELU(s) = s sigma, so GELU'(s) = sigma + s sigma (1 -
    sigma) 2c (1 + 3 * 0.044715 s^2). The usual 0.5 (1 + tanh) form of the
    first term cancels to 0 in float64 by s = -8, where the derivative is
    still about -1e-14, and then to the end of FP16's range, where this form
    keeps its relative accuracy."""
    s = values.double()
    c = math.sqrt(2.0 / math.pi)
    a = 2.0 * c * (s + 0.044715 * s**3)
    sigma = torch.sigmoid(a)
    return sigma + s * sigma * (1 - sigma) * 2.0 * c * (1 + 3 * 0.044715 * s * s), a


def ordered(h):
    """FP16 values as integers that rise with the value, -0.0 and +0.0 both 0,
    so that neighbours differ by 1."""
    b = bits(h).int()
    return torch.where(b < 0, -(b & 0x7FFF), b)


def rounded_to_fp16(v):
    """float64 values rounded once to FP16, to nearest even: numpy converts
    directly, where torch's double to half conversion goes through float and so
    rounds twice."""
    with np.errstate(over="ignore"):
        return torch.from_numpy(v.cpu().numpy().astype(np.float16)).to(v.device)


def rounding_margin(v):
    """For float64 values v, how far each is from the nearest boundary between
    FP16 roundings, as a fraction of v. Boundaries are the midpoints between
    neighbouring FP16 values; the overflow threshold 65520 is one. Values from
    65536 up round to inf whatever their error, so their margin is infinite."""
    a = v.abs()
    exponent = torch.frexp(a)[1].double()
    # The spacing is 2^(floor(log2 a) - 10) in the normal range, and 2^-24 in
    # the subnormal range below 2^-14.
    ulp = torch.where(a < 2.0**-14, 2.0**-24, torch.exp2(exponent - 11))
    fraction = (a / ulp) % 1.0
    margin = (fraction - 0.5).abs() * ulp / a
    margin = torch.where(a == 0, math.inf, margin)
    return torch.where(a >= 65536, math.inf, margin)


# Flint's FP32 derivative has two kinds of error.
#
# Relative error. The result is sigma plus a second term, one fma apart, so it
# errs by a few FP32 ulps of itself, and by the rounding of the exponent
# a = -2 sqrt(2/pi) (s + 0.044715 s^3) that exp turns into |a| 2^-22 of exp(a),
# which is most of sigma, and so of the derivative, on the negative side (as in
# test_gemm_bias_gelu.py's analysis of the forward pass: 2^-22 of a). On the
# positive side sigma approaches 1 and exp(-a) vanishes into it, so the
# exponent's error does too. Measured over every finite FP16 pre with dh from
# 1 to 30,000, the largest error beyond half an FP16 ulp of the exact product,
# away from the root below, is 3.2e-7 of it (5.4 FP32 ulps); the model's
# 2^-21 (2 + max(-a, 0)) is at least 9.5e-7.
#
# Absolute error near the root. The derivative crosses zero at s0 = -0.7525,
# GELU's minimum. There sigma (0.23) and the second term cancel, and each is
# rounded to FP32: the error is a few ulps of 0.23 (2^-26 = 1.5e-8), not of the
# tiny result. Over the FP16 values within 0.1 of s0, with dh large enough that
# FP16's rounding of y does not hide it (up to 30,000), the largest |y - dh g|
# beyond half an FP16 ulp of dh g is 2.5e-8 |dh|. ROOT_E is four times that, and
# applies within ROOT_WINDOW of s0, which is wider than where it matters (the
# derivative is under 10% of its terms within 0.1).
ROOT = -0.7525
ROOT_WINDOW = 0.25
ROOT_E = 1e-7

# dh values the sweep runs, all exact in FP16: 1, a large one, a small one,
# negative ones, and the FP16 value nearest 0.1.
SWEEP_DH = [1.0, 1000.0, 2.0**-10, -1.0, -3.0, float(np.float16(0.1))]

# Of the sweep's points, the share whose exact product is far enough from an
# FP16 rounding boundary to be checked bit for bit: 99.8% when measured.
SAFE_FRACTION = 0.95


@pytest.fixture(scope="module")
def sweep(flint):
    """Runs every finite FP16 value as pre (63,488 columns, a multiple of 8 by
    luck) against each SWEEP_DH, one per row. The sums are exactly dh: x holds
    dh in its first column and zeros, w ones in its first row and zeros. Returns
    a namespace with the outputs y (rows x columns), pre's values s, the exact
    float64 products v = dh g, g and a."""
    all_bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).cuda()
    values = all_bits.view(torch.float16)
    s = values[torch.isfinite(values)]
    n = s.numel()
    assert n == 63488 and n % 8 == 0
    m, k = len(SWEEP_DH), 8
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    x[:, 0] = torch.tensor(SWEEP_DH, dtype=torch.float16)
    w = torch.zeros((k, n), dtype=torch.float16, device="cuda")
    w[0] = 1
    y = flint_gemm_dgelu(flint, x, w, s[None, :].expand(m, n))

    g, a = dgelu64(s)
    dh = x[:, 0].double()[:, None]
    # The reference itself, against PyTorch's float64 backward where its tanh
    # form still has digits: s in [-5, 5], to 1e-12 of dh.
    inside = s.abs() <= 5
    torch_g = torch.ops.aten.gelu_backward(torch.ones_like(s[inside], dtype=torch.float64), s[inside].double(), approximate="tanh")
    assert (torch_g - g[inside]).abs().max().item() < 1e-12

    result = types.SimpleNamespace()
    result.y, result.s, result.g, result.a, result.dh = y, s.double(), g, a, dh
    result.v = dh * g[None, :]
    result.exact = rounded_to_fp16(result.v)
    result.near_root = ((result.s - ROOT).abs() < ROOT_WINDOW)[None, :].expand(m, n)
    return result


def describe(sweep, mask):
    """The first failing point of mask (rows x columns), for a message."""
    row, column = mask.nonzero()[0].tolist()
    return (
        f"dh {sweep.dh[row, 0].item()!r}, pre {sweep.s[column].item()!r}: y "
        f"{sweep.y[row, column].item()!r}, correctly rounded {sweep.exact[row, column].item()!r}, "
        f"exact {sweep.v[row, column].item()!r}"
    )


def test_derivative_is_within_one_ulp_or_the_root_bound(sweep):
    # Within one FP16 ulp of the correctly rounded product everywhere, except
    # near the root, where an absolute error of ROOT_E |dh| is allowed too.
    distance = (ordered(sweep.y) - ordered(sweep.exact)).abs()
    close = (sweep.y.double() - sweep.v).abs() <= sweep.dh.abs() * ROOT_E
    wrong = (distance > 1) & ~(sweep.near_root & close)
    assert not wrong.any(), (
        f"{wrong.sum().item()} points are neither within one ulp nor within the root bound, "
        f"the first at {describe(sweep, wrong)}"
    )


def test_derivative_is_the_correctly_rounded_value_where_that_is_well_defined(sweep):
    # Where the exact product is farther from an FP16 rounding boundary than
    # the model's error (relative, plus the absolute root bound near the
    # root), every GPU must round it the same way, so y equals the correctly
    # rounded value. -0.0 and +0.0 count as the same.
    relative = 2.0**-21 * (2 + (-sweep.a).clamp(min=0))
    absolute = torch.where(sweep.near_root[0], ROOT_E / sweep.g.abs(), 0.0)
    safe = rounding_margin(sweep.v) > (relative + absolute)[None, :]
    assert safe.sum() > SAFE_FRACTION * safe.numel(), "almost every point should be safe"
    wrong = safe & (ordered(sweep.y) != ordered(sweep.exact))
    assert not wrong.any(), (
        f"{wrong.sum().item()} safe points are not the correctly rounded value, the first at "
        f"{describe(sweep, wrong)}"
    )


# -- Exact links to flint_gemm_nn -------------------------------------------


def small_sums(m, n, k, seed):
    """x and w about N(0, 3e-3), so the sums are around 1e-4: about half of
    them under 2^-13, where halving is no longer exact."""
    g = generator(seed)
    return normal((m, k), 3e-3, g), normal((k, n), 3e-3, g), None


LINK_INPUTS = {"realistic": realistic, "outliers": outliers, "small": small_sums}


def constant_pre(m, n, value):
    return torch.full((m, n), value, dtype=torch.float16, device="cuda")


@pytest.mark.parametrize("variant", LINK_INPUTS)
def test_pre_at_plus_inf_is_flint_gemm_nn_bit_for_bit(flint, variant):
    # GELU'(+inf) = 1 (the limit), so the sums go through unchanged.
    m, n, k = 129, 264, 136
    x, w, _ = LINK_INPUTS[variant](m, n, k, 0)
    plain = flint_gemm_plain(flint, x, w)
    y = flint_gemm_dgelu(flint, x, w, constant_pre(m, n, math.inf))
    assert torch.equal(bits(y), bits(plain))


@pytest.mark.parametrize("variant", LINK_INPUTS)
def test_pre_at_zero_is_half_flint_gemm_nn_bit_for_bit(flint, variant):
    # GELU'(0) = 0.5 exactly, so each product is the sum halved: exact in FP32,
    # and exact in FP16 wherever the halved output is a normal number. Below
    # that the halved FP32 sum rounds once where halving the rounded sum
    # rounds twice, so they can differ by an ulp.
    m, n, k = 129, 264, 136
    x, w, _ = LINK_INPUTS[variant](m, n, k, 0)
    plain = flint_gemm_plain(flint, x, w)
    y = flint_gemm_dgelu(flint, x, w, constant_pre(m, n, 0.0))
    halved = plain / 2

    normal_range = plain.abs() >= 2.0**-13
    assert torch.equal(bits(y)[normal_range], bits(halved)[normal_range])
    assert normal_range.sum() > 1000, "too few outputs in the normal range"
    small = ~normal_range
    if small.any():
        assert (ordered(y)[small] - ordered(halved)[small]).abs().max().item() <= 1
    if variant == "small":
        # A good share below the normal range, and not zero, so the check is
        # not vacuous.
        assert (plain.abs()[small] > 0).sum() > 1000


@pytest.mark.parametrize("variant", LINK_INPUTS)
def test_pre_at_minus_inf_gives_zeros(flint, variant):
    # GELU'(-inf) = 0 (the limit), where the tanh formula gives NaN.
    m, n, k = 129, 264, 136
    x, w, _ = LINK_INPUTS[variant](m, n, k, 0)
    y = flint_gemm_dgelu(flint, x, w, constant_pre(m, n, -math.inf))
    assert (y == 0).all()


# -- Non-finite pre ---------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_a_nonfinite_pre_reaches_only_its_element(flint, bad):
    m, n, k = 150, 136, 104
    x, w, pre = realistic(m, n, k, 0)
    clean = flint_gemm_dgelu(flint, x, w, pre)
    plain = flint_gemm_plain(flint, x, w)
    assert torch.isfinite(clean).all()

    # The corners and a place in the middle of a tile.
    for row, column in [(0, 0), (m - 1, n - 1), (77, 13)]:
        bad_pre = pre.clone()
        bad_pre[row, column] = bad
        y = flint_gemm_dgelu(flint, x, w, bad_pre)

        others = torch.ones((m, n), dtype=torch.bool, device="cuda")
        others[row, column] = False
        assert torch.equal(bits(y)[others], bits(clean)[others]), (row, column)
        # NaN gives NaN; +inf gives dh (the sums, as flint_gemm_nn rounds
        # them); -inf gives zero.
        if math.isnan(bad):
            assert torch.isnan(y[row, column])
        elif bad > 0:
            assert bits(y)[row, column] == bits(plain)[row, column]
        else:
            assert y[row, column] == 0


# -- Overflow ---------------------------------------------------------------


def test_overflow_gives_inf_where_the_product_rounds_past_fp16(flint):
    # dh near FP16's largest value times a derivative that crosses 1 (it is up
    # to 1.13, at s = 1.5 or so) across the pre values from 0.25 to 8, 5,120
    # of them: the products cross the overflow threshold 65520 and come back.
    all_bits = torch.arange(0x3400, 0x4800, dtype=torch.int32).to(torch.int16).cuda()
    s = all_bits.view(torch.float16)
    n = s.numel()
    assert n == 5120
    dhs = [60000.0, 65504.0, -60000.0, -65504.0]
    m, k = len(dhs), 8
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    x[:, 0] = torch.tensor(dhs, dtype=torch.float16)
    w = torch.zeros((k, n), dtype=torch.float16, device="cuda")
    w[0] = 1
    y = flint_gemm_dgelu(flint, x, w, s[None, :].expand(m, n))

    g, _ = dgelu64(s)
    v = x[:, 0].double()[:, None] * g[None, :]
    # The FP32 product errs by a few ulps (about 1e-6 of itself): outside this
    # band around the threshold, the answer is certain.
    gap = 2.0**-16
    overflows = v.abs() > 65520 * (1 + gap)
    fits = v.abs() < 65520 * (1 - gap)
    assert overflows.sum() > 500 and fits.sum() > 5000, (overflows.sum().item(), fits.sum().item())
    assert (overflows & (v > 0)).any() and (overflows & (v < 0)).any()
    assert not torch.isnan(y).any()
    expected_inf = torch.where(v > 0, math.inf, -math.inf).half()
    assert torch.equal(bits(y)[overflows], bits(expected_inf)[overflows]), "wrong overflow to inf"
    assert torch.isfinite(y[fits]).all(), "finite output became inf"
    # And the finite ones are the correctly rounded values, within an ulp.
    distance = (ordered(y) - ordered(rounded_to_fp16(v))).abs()
    assert distance[fits].max().item() <= 1


# -- Determinism ------------------------------------------------------------

DETERMINISM_SHAPES = [(GPT2_M, GPT2_N, GPT2_K), (129, 120, 40)]


@pytest.mark.parametrize("shape", DETERMINISM_SHAPES, ids=shape_id)
def test_run_twice_is_bitwise_identical(flint, shape):
    m, n, k = shape
    x, w, pre = outliers(m, n, k, 0)
    # Different NaN patterns to start from, so a store that is skipped, or one
    # that lands in only one run, shows as a difference or a leftover NaN.
    first = flint_gemm_dgelu(flint, x, w, pre, fill=0xFFFF)
    second = flint_gemm_dgelu(flint, x, w, pre, fill=0x7F7F)

    assert not torch.isnan(first).any()
    assert not torch.isnan(second).any()
    assert torch.equal(bits(first), bits(second))


# -- Streams ----------------------------------------------------------------


def test_gemm_runs_on_the_given_stream(flint):
    m = n = k = 64
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((k, n), dtype=torch.float16, device="cuda")
    pre = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    y = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fills finish only after the sleep, so the result is k only if the
    # GEMM is queued on the same stream behind them: with x zero or pre zero it
    # would be 0 or k / 2. GELU' of +inf is 1.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(1)
        pre.fill_(math.inf)
        flint.gemm_nn_dgelu(y, x, w, pre, m, n, k, stream=stream)
    stream.synchronize()

    assert (y == k).all()


# -- The size limits --------------------------------------------------------

# Each limit at its edge, as flint_gemm_nn's tests do: m and n at 65,535 tiles
# of 128, n's being the grid's limit (with 2 rows; a full row of tiles at that
# n would need a 2 GiB y), and k where the error is closest to the gate.
# Skipped under the sanitizer, where 65,535 tiles or a 2^16-step k loop would
# take too long.
LIMIT_CASES = [
    pytest.param(shape, id=name)
    for name, shape in [
        ("max_m", (8_388_480, 8, 8)),
        ("max_n", (2, 8_388_480, 8)),
        ("max_k", (64, 64, 2**16)),
    ]
]


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@pytest.mark.parametrize("shape", LIMIT_CASES)
def test_largest_allowed_dimensions_pass_the_gate(flint, shape):
    m, n, k = shape
    gate.gate(gate.GRADIENT_FACTOR, gate_triples(flint, realistic, m, n, k))


# -- Argument errors --------------------------------------------------------


def small_problem(m=16, n=16, k=16):
    """Valid tensors, in buffers with room to spare."""
    y = torch.zeros(m * n + 64, dtype=torch.float16, device="cuda")
    x = torch.ones(m * k + 64, dtype=torch.float16, device="cuda")
    w = torch.ones(k * n + 64, dtype=torch.float16, device="cuda")
    pre = torch.zeros(m * n + 64, dtype=torch.float16, device="cuda")
    return y, x, w, pre


def expect_invalid_argument(flint, reason, y, x, w, pre, m, n, k):
    with pytest.raises(flint_lib.FlintError) as error:
        flint.gemm_nn_dgelu(y, x, w, pre, m, n, k)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_gemm_nn_dgelu", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("name", ["y", "x", "w", "pre"])
def test_null_pointer(flint, name):
    y, x, w, pre = small_problem()
    pointers = {"y": y, "x": x, "w": w, "pre": pre}
    pointers[name] = None
    expect_invalid_argument(
        flint, f"{name} is null", pointers["y"], pointers["x"], pointers["w"], pointers["pre"], 16, 16, 16
    )


@pytest.mark.parametrize("k", [0, 12])
def test_k_must_be_a_positive_multiple_of_8(flint, k):
    y, x, w, pre = small_problem()
    expect_invalid_argument(flint, f"k ({k}) must be a positive multiple of 8", y, x, w, pre, 16, 16, k)


def test_n_must_be_a_multiple_of_8(flint):
    # w's rows are n long, and load in 16-byte chunks.
    y, x, w, pre = small_problem(n=12)
    expect_invalid_argument(flint, "n (12) must be a multiple of 8", y, x, w, pre, 16, 12, 16)


@pytest.mark.parametrize("name", ["y", "x", "w", "pre"])
def test_misaligned_pointer(flint, name):
    y, x, w, pre = small_problem()
    pointers = {"y": y.data_ptr(), "x": x.data_ptr(), "w": w.data_ptr(), "pre": pre.data_ptr()}
    pointers[name] += 2
    expect_invalid_argument(
        flint, f"{name} is not 16-byte aligned", pointers["y"], pointers["x"], pointers["w"],
        pointers["pre"], 16, 16, 16
    )


def test_pre_may_not_run_past_the_address_space(flint):
    # The check runs before any memory is touched, so a made-up address will do.
    y, x, w, _ = small_problem()
    expect_invalid_argument(
        flint, "pre runs past the end of the address space", y, x, w, 2**64 - 16, 16, 16, 16
    )


# The size checks run before any memory is touched, so small tensors do.
TOO_LARGE_CASES = [
    pytest.param(shape, id=name)
    for name, shape in [
        ("m", (8_388_481, 8, 8)),
        ("n", (8, 8_388_488, 8)),
        ("k", (8, 8, 2**16 + 8)),
    ]
]


@pytest.mark.parametrize("shape", TOO_LARGE_CASES)
def test_too_large_dimensions(flint, shape):
    m, n, k = shape
    y, x, w, pre = small_problem()
    expect_invalid_argument(flint, "too large", y, x, w, pre, m, n, k)


def test_y_overlapping_x(flint):
    _, _, _, pre = small_problem()
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # x starts 256 bytes into y, and, in the second call, 16 bytes before y's end.
    expect_invalid_argument(flint, "y and x overlap", buf, buf[128:], buf[2048:], pre, 16, 16, 16)
    expect_invalid_argument(flint, "y and x overlap", buf, buf[248:], buf[2048:], pre, 16, 16, 16)


def test_y_overlapping_w(flint):
    _, _, _, pre = small_problem()
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    expect_invalid_argument(flint, "y and w overlap", buf, buf[2048:], buf[128:], pre, 16, 16, 16)
    expect_invalid_argument(flint, "y and w overlap", buf, buf[2048:], buf[248:], pre, 16, 16, 16)


def test_y_overlapping_pre(flint):
    _, x, w, _ = small_problem()
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # Both are 512 bytes. pre starts 256 bytes into y, then 16 bytes before y's
    # end, and then is y itself; and the other way round, y starts inside pre.
    expect_invalid_argument(flint, "y and pre overlap", buf, x, w, buf[128:], 16, 16, 16)
    expect_invalid_argument(flint, "y and pre overlap", buf, x, w, buf[248:], 16, 16, 16)
    expect_invalid_argument(flint, "y and pre overlap", buf, x, w, buf, 16, 16, 16)
    expect_invalid_argument(flint, "y and pre overlap", buf[128:], x, w, buf, 16, 16, 16)
    expect_invalid_argument(flint, "y and pre overlap", buf[248:], x, w, buf, 16, 16, 16)


@pytest.mark.parametrize("pre_first", [True, False], ids=["pre-then-y", "y-then-pre"])
def test_pre_next_to_y_is_not_overlapping(flint, pre_first):
    # y and pre (512 bytes each) side by side in one buffer, with 16 bytes of
    # pattern around them that must stay intact. pre is +inf, so y is the sums,
    # 16.
    m = n = k = 16
    buf = torch.full((16 + 1024 + 16,), 0x5A, dtype=torch.uint8, device="cuda")
    pre_at, y_at = (16, 16 + 512) if pre_first else (16 + 512, 16)
    pre = buf[pre_at : pre_at + 2 * m * n].view(torch.float16)
    y = buf[y_at : y_at + 2 * m * n].view(torch.float16)
    pre.fill_(math.inf)
    x = torch.ones((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((k, n), dtype=torch.float16, device="cuda")

    flint.gemm_nn_dgelu(y, x, w, pre, m, n, k)
    torch.cuda.synchronize()

    assert (y == 16).all()
    assert (pre == math.inf).all()
    assert (buf[:16] == 0x5A).all() and (buf[16 + 1024 :] == 0x5A).all()


@pytest.mark.parametrize("which", ["x", "w", "both"])
def test_pre_may_overlap_the_inputs(flint, which):
    # pre is read like x and w, so it may share their bytes: x and pre (or w and
    # pre, or all three) are one tensor, which needs m = n = k = 16.
    m = n = k = 16

    def triples():
        for seed in gate.SEEDS:
            g = generator(seed)
            shared = normal((m, n), 1.0, g)
            other = normal((m, n), 0.5, g)
            x, w = (other, shared) if which == "w" else (shared, other)
            if which == "both":
                x = w = shared
            pre = shared
            x_copy, w_copy, pre_copy = x.clone(), w.clone(), pre.clone()
            y = torch.empty((m, n), dtype=torch.float16, device="cuda")
            flint.gemm_nn_dgelu(y, x, w, pre, m, n, k)
            torch.cuda.synchronize()
            assert torch.equal(bits(shared), bits(pre_copy)), "the shared input changed"
            yield (
                y,
                gate.peer_gemm_nn_dgelu(x_copy, w_copy, pre_copy),
                gate.reference_dgelu(x_copy, w_copy, pre_copy),
            )

    gate.gate(gate.GRADIENT_FACTOR, triples())


@pytest.mark.parametrize("m, n", [(0, 16), (16, 0), (0, 0)])
def test_empty_output_does_nothing(flint, m, n):
    # Not even the null pre is looked at.
    y, x, w, pre = small_problem()
    y.fill_(3)

    flint.gemm_nn_dgelu(y, x, w, pre, m, n, 16)
    flint.gemm_nn_dgelu(y, x, w, None, m, n, 16)
    flint.gemm_nn_dgelu(None, None, None, None, m, n, 16)
    torch.cuda.synchronize()

    assert (y == 3).all()


def test_a_valid_call_works_after_errors(flint):
    y, x, w, pre = small_problem()
    pre[:256] = math.inf
    for k in (0, 12):
        with pytest.raises(flint_lib.FlintError):
            flint.gemm_nn_dgelu(y, x, w, pre, 16, 16, k)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nn_dgelu(y, x, w, pre, 16, 12, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nn_dgelu(None, x, w, pre, 16, 16, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nn_dgelu(y, x, w, None, 16, 16, 16)

    flint.gemm_nn_dgelu(y, x, w, pre, 16, 16, 16)
    torch.cuda.synchronize()

    assert (y[:256] == 16).all()
    assert (y[256:] == 0).all()
