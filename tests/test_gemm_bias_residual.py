"""flint_gemm_nt_bias_residual: y = residual + (x w^T + bias), the projections
that feed GPT-2's FP32 residual stream (the attention projection and the MLP's
down projection). x (m x k) and w (n x k) are FP16, bias holds n FP32 values,
and residual and y are m x n FP32. The FP32 sums s of x w^T get the bias first,
exactly as flint_gemm_nt_bias computes s + bias, and then the residual:
y = (s + b) + r, one FP32 add each, with nothing rounded to FP16. n must be
even.

The residual stream's values are in the thousands in the late layers, where an
FP16-ulp floor on y itself (which PLAN's gate applies to outputs) would hide any
error, so the correctness gate applies to the increment y - residual, in float64,
against x w^T + bias in float32 or float64. The peer is PyTorch's FP16 path as
autocast runs it: the linear layer's FP16 output, added to the FP32 residual in
FP32, and its increment is that sum minus the residual (so it includes the
rounding of the residual add, as Flint's does). Exact tests then pin y itself:
a residual of -0.0 gives flint_gemm_nt_bias's output when y rounds to FP16, any
residual adds to that sum in a single FP32 add, and a zero input or integer
sums leave the two FP32 adds as the only rounding. After those come
non-finite values, determinism, stream ordering, and argument errors.

The kernel's main loop is flint_gemm_nt's in length and waits, which
test_sass.py checks; the bias and the residual are read after the loop."""

import math

import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, clamp_half, generator, normal

# Under the sanitizer the GPT-2 shapes shrink.
GPT2_M = 64 if SANITIZER else 1024

# The two layers that feed the residual stream, as (n, k): the forward pass's w
# is n x k.
GPT2_SHAPES = {
    "attention_projection": (768, 768),
    "mlp_down": (768, 3072),
}

# The output of a guarded call sits `offset` floats past the start of a buffer
# that is filled with a pattern, which must be intact around y afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A5A5A
NAN_BITS = -1  # 0xFFFFFFFF, a NaN as float32
OTHER_BITS = 0x7F7F7F7F  # a finite 3.4e38 as float32


def shape_id(shape):
    return "x".join(map(str, shape))


# -- Inputs -----------------------------------------------------------------
# Every maker returns x (m x k), w (n x k), an FP32 bias (n) and an FP32
# residual (m x n).


def make_bias(n, seed, std=0.5):
    """FP32 values about N(0, 0.5): full 24-bit mantissas, which FP16 can't
    hold."""
    g = generator(30_000 + seed)
    return torch.randn(n, generator=g, device="cuda") * std


def make_residual(m, n, seed, kind="stream"):
    """An FP32 residual stream. "stream" is about N(0, 1); "outliers" has about
    0.5% of its columns (at least 2) near +-3,000 in every row, as in GPT-2's
    residual stream; "late" has values from 2,048 to 4,096 in magnitude, either
    sign, as in its late layers, where FP16's step is 2 and FP32's 2^-12."""
    g = generator(20_000 + seed)
    residual = torch.randn((m, n), generator=g, device="cuda")
    if kind == "outliers":
        count = max(2, round(0.005 * n))
        channels = torch.randperm(n, generator=g, device="cuda")[:count]
        signs = torch.randint(0, 2, (count,), generator=g, device="cuda") * 2.0 - 1.0
        jitter = 0.9 + 0.2 * torch.rand((m, count), generator=g, device="cuda")
        residual[:, channels] = 3000.0 * signs * jitter
    elif kind == "late":
        magnitude = 2048.0 + 2048.0 * torch.rand((m, n), generator=g, device="cuda")
        signs = torch.randint(0, 2, (m, n), generator=g, device="cuda") * 2.0 - 1.0
        residual = signs * magnitude
    return residual


def realistic(m, n, k, seed):
    """Activations about N(0, 1), weights about N(0, 0.02), a bias about
    N(0, 0.5), and a residual about N(0, 1)."""
    g = generator(seed)
    return normal((m, k), 1.0, g), normal((n, k), 0.02, g), make_bias(n, seed), make_residual(m, n, seed)


def outliers(m, n, k, seed):
    """Realistic inputs, with about 0.5% of x's channels (at least 2) near
    +-3,000 in every row, as in GPT-2's activations, and the outlier channels in
    the residual too."""
    x, w, bias, _ = realistic(m, n, k, seed)
    g = generator(10_000 + seed)
    count = max(2, round(0.005 * k))
    channels = torch.randperm(k, generator=g, device="cuda")[:count]
    signs = torch.randint(0, 2, (count,), generator=g, device="cuda") * 2.0 - 1.0
    jitter = 0.9 + 0.2 * torch.rand((m, count), generator=g, device="cuda")
    x[:, channels] = (3000.0 * signs * jitter).half()
    return x, w, bias, make_residual(m, n, seed, "outliers")


def late(m, n, k, seed):
    """Realistic inputs with a residual at the late layers' magnitudes."""
    x, w, bias, _ = realistic(m, n, k, seed)
    return x, w, bias, make_residual(m, n, seed, "late")


INPUTS = {"realistic": realistic, "outliers": outliers, "late": late}
RESIDUALS = ("stream", "outliers", "late")


# -- Running Flint ----------------------------------------------------------


def bits(t):
    """An FP32 tensor's bit patterns, which compare equal for equal bits, NaNs
    and -0.0 included."""
    return t.contiguous().view(torch.int32)


def half_bits(t):
    """An FP16 tensor's bit patterns, the same way."""
    return t.contiguous().view(torch.int16)


def raw(t):
    """A tensor's bytes, to tell whether a call changed it."""
    return t.contiguous().view(torch.uint8).reshape(-1)


def flint_gemm_residual(flint, x, w, bias, residual, offset=4, fill=SENTINEL):
    """Runs flint_gemm_nt_bias_residual into y inside a buffer pre-filled with
    `fill` (a 32-bit pattern), offset floats in; the residual is copied into a
    pre-filled buffer of its own, 16 - offset floats in, so offsets of 4, 8 and
    12 (which keep both 16-byte aligned) give both. Checks the buffer around y
    is untouched and so are all of the residual's, x, w and bias, and returns y
    as an m x n FP32 view."""
    m, k = x.shape
    n = w.shape[0]
    residual_offset = 16 - offset
    y_buf = torch.full((offset + m * n + GUARD_AFTER,), fill, dtype=torch.int32, device="cuda")
    residual_buf = torch.full(
        (residual_offset + m * n + GUARD_AFTER,), SENTINEL, dtype=torch.int32, device="cuda"
    )
    residual_in = residual_buf[residual_offset : residual_offset + m * n]
    residual_in.copy_(bits(residual).reshape(-1))
    residual_before = residual_buf.clone()
    inputs_before = [raw(t).clone() for t in (x, w, bias)]
    y = y_buf[offset : offset + m * n].view(torch.float32)
    flint.gemm_nt_bias_residual(y, x, w, bias, residual_in.view(torch.float32), m, n, k)
    torch.cuda.synchronize()
    guard = torch.full((1,), fill, dtype=torch.int32, device="cuda")
    assert (y_buf[:offset] == guard).all(), "flint_gemm_nt_bias_residual wrote before y"
    assert (y_buf[offset + m * n :] == guard).all(), "flint_gemm_nt_bias_residual wrote past y"
    assert torch.equal(residual_buf, residual_before), "residual changed"
    for name, tensor, before in zip(("x", "w", "bias"), (x, w, bias), inputs_before):
        assert torch.equal(raw(tensor), before), f"{name} changed"
    return y.view(m, n)


def flint_gemm_plain(flint, x, w, bias):
    """flint_gemm_nt_bias's output, to compare with the residual kernel's."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), dtype=torch.float16, device="cuda")
    flint.gemm_nt_bias(y, x, w, bias, m, n, k)
    torch.cuda.synchronize()
    return y


def negative_zeros(m, n):
    return torch.full((m, n), -0.0, dtype=torch.float32, device="cuda")


def peer_out(x, w, bias, residual):
    """The peer's y: autocast's FP16 linear layer, added to the FP32 residual
    in FP32."""
    return residual + gate.peer_gemm_nt_bias(x, w, bias).float()


def increment(out, residual):
    """out - residual in float64: what the layer added to the stream."""
    return out.double() - residual.double()


def gate_triples(flint, make_inputs, m, n, k, offset=4):
    """Per seed, Flint's increment, the peer's and the reference's."""
    for seed in gate.SEEDS:
        x, w, bias, residual = make_inputs(m, n, k, seed)
        y = flint_gemm_residual(flint, x, w, bias, residual, offset)
        yield (
            increment(y, residual),
            increment(peer_out(x, w, bias, residual), residual),
            gate.reference_bias(x, w, bias),
        )


# -- The gate ---------------------------------------------------------------

GPT2_CASES = [
    pytest.param(name, variant, id=f"{name}-{variant}") for name in GPT2_SHAPES for variant in INPUTS
]


@pytest.mark.parametrize("name, variant", GPT2_CASES)
def test_gpt2_shapes_pass_the_gate(flint, name, variant):
    n, k = GPT2_SHAPES[name]
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, INPUTS[variant], GPT2_M, n, k))


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@pytest.mark.parametrize("name, variant", GPT2_CASES)
def test_gpt2_shapes_at_8192_tokens_pass_the_gate(flint, name, variant):
    n, k = GPT2_SHAPES[name]
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, INPUTS[variant], 8192, n, k))


# m off the 128 tile, k off the 32 step, n even but off the 8-column group (2,
# 4 and 6 more than a multiple of 8, whose last pair of columns is a partial
# tile's), and single rows.
ODD_SHAPES = [
    (1, 2, 8),
    (7, 14, 24),
    (127, 130, 136),
    (129, 126, 40),
    (200, 300, 776),
    (333, 6, 8),
    (1, 50304, 768),
    (130, 2, 16),
    # Tiles go out in groups of up to 8 rows of tiles. 5 rows of tiles make
    # a single short group. 9, 13, 17 and 33 end in a short group of 1 or 5
    # rows after full ones, whose spare blocks exit, with one column of
    # tiles or several.
    (637, 300, 24),
    (1152, 2, 8),
    (1614, 200, 32),
    (2116, 130, 40),
    (4224, 8, 16),
]

# The inputs rotate through the three variants, and the offsets through 4, 8
# and 12 floats.
ODD_CASES = [
    pytest.param(shape, list(INPUTS)[i % 3], 4 * (1 + i % 3), id=shape_id(shape))
    for i, shape in enumerate(ODD_SHAPES)
]


@pytest.mark.parametrize("shape, variant, offset", ODD_CASES)
def test_odd_shapes_pass_the_gate_and_stay_inside_y(flint, shape, variant, offset):
    # flint_gemm_residual also checks that y's guard regions, the residual, x,
    # w and bias are intact.
    m, n, k = shape
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, INPUTS[variant], m, n, k, offset))


# -- FP16's range -----------------------------------------------------------


def large_x(m, n, k, seed):
    g = generator(seed)
    return (
        clamp_half(torch.randn((m, k), generator=g, device="cuda") * 20000, 65504),
        normal((n, k), 0.02, g),
        make_bias(n, seed),
        make_residual(m, n, seed),
    )


def large_w(m, n, k, seed):
    g = generator(seed)
    return (
        normal((m, k), 0.02, g),
        clamp_half(torch.randn((n, k), generator=g, device="cuda") * 20000, 65504),
        make_bias(n, seed),
        make_residual(m, n, seed),
    )


@pytest.mark.parametrize("make_inputs", [large_x, large_w], ids=["large_x", "large_w"])
def test_fp16_range_inputs_pass_the_gate(flint, make_inputs):
    m, n, k = 130, 72, 104
    # The point of these inputs is large finite sums, so check they are.
    x, w, bias, _ = make_inputs(m, n, k, 0)
    top = gate.reference_bias(x, w, bias).abs().max().item()
    assert 1000 < top < 65504, top
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, make_inputs, m, n, k))


def test_sums_past_fp16_stay_in_fp32(flint):
    # One nonzero product per sum, from 1.9e9 to 3.6e9, each exact in FP32 and
    # far past FP16's 65,504. y keeps them (a store through FP16 would give
    # inf), and the bias and the residual add to them in FP32, each rounding to
    # a multiple of 128 or 256.
    m, n, k = 130, 72, 104
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = torch.zeros((n, k), dtype=torch.float16, device="cuda")
    x[:, 0] = torch.tensor([60000.0, -48000.0], device="cuda").repeat(m // 2).half()
    w[:, 0] = torch.tensor([40000.0, 60000.0, -40000.0], device="cuda").repeat(n // 3).half()
    bias = make_bias(n, 0)
    residual = make_residual(m, n, 0)

    with gate.peer_math():
        sums = x.float() @ w.float().T
    expected = (sums + bias) + residual
    assert sums.abs().min() > 1e9 and torch.isfinite(expected).all()
    y = flint_gemm_residual(flint, x, w, bias, residual)

    assert torch.equal(bits(y), bits(expected))


# -- Exact links ------------------------------------------------------------

LINK_SHAPES = [
    (GPT2_M, *GPT2_SHAPES["attention_projection"]),
    (7, 14, 24),
    (129, 126, 40),
    (200, 300, 776),
    (1, 50304, 768),
]


@pytest.mark.parametrize("shape", LINK_SHAPES, ids=shape_id)
def test_a_residual_of_negative_zero_rounds_to_flint_gemm_nt_biass_output(flint, shape):
    # s + b is the same FP32 value in both kernels, and -0.0 changes no bit of
    # it when added: (-0.0) + v is v for every v, -0.0 included (+0.0 would turn
    # a -0.0 sum into +0.0, which rounds to a different FP16). So y is exactly
    # the value flint_gemm_nt_bias rounds, and y rounded to FP16 (to nearest
    # even, as flint_gemm_nt_bias does) is its output bit for bit.
    m, n, k = shape
    for seed in gate.SEEDS:
        for make_inputs in (realistic, outliers):
            x, w, bias, _ = make_inputs(m, n, k, seed)
            y = flint_gemm_residual(flint, x, w, bias, negative_zeros(m, n))
            plain = flint_gemm_plain(flint, x, w, bias)
            assert torch.equal(half_bits(y.half()), half_bits(plain))
            # y is not already FP16, so the check says something about the
            # rounding.
            assert (y.half().float() != y).any()


def test_outputs_past_fp16_are_inf_when_rounded_and_finite_in_y(flint):
    # The sums' std is about 40,000, so roughly one output in ten is past FP16's
    # range: inf in flint_gemm_nt_bias, which y rounded to FP16 must match, and
    # an FP32 value in y.
    m, n, k = 150, 130, 64
    for seed in gate.SEEDS:
        g = generator(seed)
        x = normal((m, k), 150.0, g)
        w = normal((n, k), 33.0, g)
        bias = make_bias(n, seed)
        y = flint_gemm_residual(flint, x, w, bias, negative_zeros(m, n))
        plain = flint_gemm_plain(flint, x, w, bias)

        # Each class and sign has to occur, or the test checks nothing.
        assert plain.isposinf().any() and plain.isneginf().any() and torch.isfinite(plain).any()
        assert torch.isfinite(y).all() and (y.abs() > 65520).any()
        assert torch.equal(half_bits(y.half()), half_bits(plain))


@pytest.mark.parametrize("shape", LINK_SHAPES[:4], ids=shape_id)
def test_any_residual_adds_to_the_biased_sum_in_one_fp32_add(flint, shape):
    # With a residual of -0.0, y is t = s + b (the test above). Any other
    # residual r gives y = (s + b) + r, which is fp32(r + t) bit for bit: the
    # same operands, one FP32 add, whatever r is.
    m, n, k = shape
    for seed in gate.SEEDS[:2]:
        x, w, bias, _ = realistic(m, n, k, seed)
        t = flint_gemm_residual(flint, x, w, bias, negative_zeros(m, n))
        for kind in RESIDUALS:
            residual = make_residual(m, n, seed, kind)
            y = flint_gemm_residual(flint, x, w, bias, residual)
            assert torch.equal(bits(y), bits(residual + t)), (seed, kind)


COLUMN_SHAPES = [
    (1, 2, 8),
    (7, 14, 24),
    (129, 126, 40),
    (200, 300, 776),
    (130, 2, 16),
    (1152, 2, 8),
    (333, 6, 8),
]


@pytest.mark.parametrize("shape", COLUMN_SHAPES, ids=shape_id)
def test_with_x_zero_y_is_the_residual_plus_each_columns_bias(flint, shape):
    # With x all +0.0 the sums are zero, so y is r + b in FP32, exactly: every
    # column gets its own bias and every row its own residual. The bias values
    # are distinct, nonzero and not FP16-representable, so a column that gets
    # another's bias, or none, or one rounded to FP16 shows as a difference.
    m, n, k = shape
    g = generator(0)
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = normal((n, k), 1.0, g)
    bias = 0.123 + 0.37 * torch.arange(n, dtype=torch.float32, device="cuda")
    assert (bias.half().float() != bias).any()
    for kind in RESIDUALS:
        residual = make_residual(m, n, 0, kind)
        y = flint_gemm_residual(flint, x, w, bias, residual)
        assert torch.equal(bits(y), bits(residual + bias)), kind


# x and w are small integers and k is 64, so every sum is an exact integer
# (at most 2^8 in magnitude), which the tensor cores' truncating accumulation
# can't lose anything from. The bias and the residual have full mantissas.
EXACT_SUM_SHAPES = [(64, 64), (7, 14), (129, 126), (200, 300), (130, 2), (1, 50304), (333, 6)]


@pytest.mark.parametrize("shape", EXACT_SUM_SHAPES, ids=shape_id)
def test_the_bias_then_the_residual_add_in_fp32_with_no_other_rounding(flint, shape):
    m, n = shape
    k = 64
    for seed in gate.SEEDS:
        g = generator(seed)
        x = torch.randint(-2, 3, (m, k), generator=g, device="cuda").half()
        w = torch.randint(-2, 3, (n, k), generator=g, device="cuda").half()
        bias = make_bias(n, seed, std=4.0)
        residual = make_residual(m, n, seed, RESIDUALS[seed % 3])

        with gate.peer_math():
            sums = x.float() @ w.float().T
        expected = (sums + bias) + residual
        y = flint_gemm_residual(flint, x, w, bias, residual)

        assert torch.equal(bits(y), bits(expected))
        if m * n >= 10_000:
            # The check can tell the orders apart: adding the residual to the
            # sum first, or rounding the bias to FP16, would change some
            # outputs.
            assert ((sums + residual) + bias != expected).any()
            assert ((sums + bias.half().float()) + residual != expected).any()


# -- Non-finite values ------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_a_nonfinite_residual_reaches_only_its_element(flint, bad):
    m, n, k = 150, 136, 104
    x, w, bias, residual = realistic(m, n, k, 0)
    clean = flint_gemm_residual(flint, x, w, bias, residual)
    assert torch.isfinite(clean).all()

    # The corners, and places in the middle of a tile: an even column and an
    # odd one, the two of a pair.
    for row, column in [(0, 0), (m - 1, n - 1), (77, 13), (76, 12)]:
        bad_residual = residual.clone()
        bad_residual[row, column] = bad
        y = flint_gemm_residual(flint, x, w, bias, bad_residual)

        others = torch.ones((m, n), dtype=torch.bool, device="cuda")
        others[row, column] = False
        assert torch.equal(bits(y)[others], bits(clean)[others]), (row, column)
        if math.isnan(bad):
            assert torch.isnan(y[row, column])
        else:
            assert y[row, column].item() == bad


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_a_nonfinite_bias_reaches_only_its_column(flint, bad):
    m, n, k = 150, 134, 104

    def triples():
        for seed in gate.SEEDS:
            x, w, bias, residual = realistic(m, n, k, seed)
            column = (seed * 37 + 11) % n
            bias[column] = bad
            y = flint_gemm_residual(flint, x, w, bias, residual)
            peer = peer_out(x, w, bias, residual)
            ref = gate.reference_bias(x, w, bias)

            # The same NaNs and infs, with the same signs, as the reference,
            # which has the bad column and no other.
            flint_inc = increment(y, residual)
            assert torch.equal(torch.isnan(flint_inc), torch.isnan(ref))
            assert torch.equal(torch.isposinf(flint_inc), torch.isposinf(ref))
            assert torch.equal(torch.isneginf(flint_inc), torch.isneginf(ref))
            finite = torch.isfinite(ref)
            assert (~finite).sum() == m
            assert (~finite[:, column]).all()
            yield flint_inc[finite], increment(peer, residual)[finite], ref[finite]

    gate.gate(gate.FORWARD_FACTOR, triples())


def test_a_bias_past_fp16_stays_finite_in_fp32(flint):
    # In flint_gemm_nt_bias these columns are inf. Here y is FP32, so they are
    # 1e6 and so on plus the rest, within FP32's rounding of them.
    m, n, k = 150, 134, 104
    columns = [0, 5, 133]  # the edges and a column in the middle of a tile
    for seed in gate.SEEDS:
        x, w, bias, residual = realistic(m, n, k, seed)
        bias[columns] = torch.tensor([1e6, -1e6, 70000.0], device="cuda")
        y = flint_gemm_residual(flint, x, w, bias, residual)
        ref = gate.reference_bias(x, w, bias)

        assert torch.isfinite(y).all()
        assert gate.errors(increment(y, residual)[:, columns], ref[:, columns]).rel_l2 < 1e-6


# -- Determinism ------------------------------------------------------------

DETERMINISM_SHAPES = [(GPT2_M, *GPT2_SHAPES["attention_projection"]), (129, 126, 40)]


@pytest.mark.parametrize("shape", DETERMINISM_SHAPES, ids=shape_id)
def test_run_twice_is_bitwise_identical(flint, shape):
    m, n, k = shape
    x, w, bias, residual = outliers(m, n, k, 0)
    # Different patterns to start from (a NaN, and a large finite value), so a
    # store that is skipped, or one that lands in only one run, shows as a
    # difference or a leftover.
    first = flint_gemm_residual(flint, x, w, bias, residual, fill=NAN_BITS)
    second = flint_gemm_residual(flint, x, w, bias, residual, fill=OTHER_BITS)

    assert not torch.isnan(first).any()
    assert not torch.isnan(second).any()
    assert (second.abs() < 1e30).all()
    assert torch.equal(bits(first), bits(second))


# -- Streams ----------------------------------------------------------------


def test_gemm_runs_on_the_given_stream(flint):
    m = n = k = 64
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")
    bias = torch.zeros(n, dtype=torch.float32, device="cuda")
    residual = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    y = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fills finish only after the sleep, so the result is k + 0.5 + 2 only
    # if the GEMM is queued on the same stream behind them.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(1)
        bias.fill_(0.5)
        residual.fill_(2)
        flint.gemm_nt_bias_residual(y, x, w, bias, residual, m, n, k, stream=stream)
    stream.synchronize()

    assert (y == k + 2.5).all()


# -- The size limits --------------------------------------------------------

# Each limit at its edge, as flint_gemm_nt_bias_gelu's tests do: m and n at
# 65,535 tiles of 128, n's being the grid's limit (with 2 rows; a full row of
# tiles at that n would need a 2 GiB y), and k where the error is closest to the
# gate. Skipped under the sanitizer, where 65,535 tiles or a 2^16-step k loop
# would take too long.
LIMIT_CASES = [
    pytest.param(shape, id=name)
    for name, shape in [
        ("max_m", (8_388_480, 2, 8)),
        ("max_n", (2, 8_388_480, 8)),
        ("max_k", (64, 64, 2**16)),
    ]
]


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@pytest.mark.parametrize("shape", LIMIT_CASES)
def test_largest_allowed_dimensions_pass_the_gate(flint, shape):
    m, n, k = shape
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, realistic, m, n, k))


# -- Argument errors --------------------------------------------------------


def small_problem(m=16, n=16, k=16):
    """Valid tensors, in buffers with room to spare."""
    y = torch.zeros(m * n + 64, dtype=torch.float32, device="cuda")
    x = torch.ones(m * k + 64, dtype=torch.float16, device="cuda")
    w = torch.ones(n * k + 64, dtype=torch.float16, device="cuda")
    bias = torch.zeros(n + 64, dtype=torch.float32, device="cuda")
    residual = torch.zeros(m * n + 64, dtype=torch.float32, device="cuda")
    return y, x, w, bias, residual


def expect_invalid_argument(flint, reason, y, x, w, bias, residual, m, n, k):
    with pytest.raises(flint_lib.FlintError) as error:
        flint.gemm_nt_bias_residual(y, x, w, bias, residual, m, n, k)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_gemm_nt_bias_residual", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


NAMES = ["y", "x", "w", "bias", "residual"]


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    pointers = dict(zip(NAMES, small_problem()))
    pointers[name] = None
    expect_invalid_argument(flint, f"{name} is null", *pointers.values(), 16, 16, 16)


@pytest.mark.parametrize("k", [0, 12])
def test_k_must_be_a_positive_multiple_of_8(flint, k):
    expect_invalid_argument(flint, f"k ({k}) must be a positive multiple of 8", *small_problem(), 16, 16, k)


@pytest.mark.parametrize("n", [1, 7])
def test_n_must_be_even(flint, n):
    # y and the residual are read and written in pairs of floats.
    expect_invalid_argument(flint, f"n ({n}) must be even", *small_problem(n=n), 16, n, 16)


@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name):
    # y, bias and the residual are FP32, but a multiple of 4 bytes short of
    # 16-byte alignment is still misaligned.
    pointers = {n: t.data_ptr() for n, t in zip(NAMES, small_problem())}
    pointers[name] += 2 if name in ("x", "w") else 4
    expect_invalid_argument(flint, f"{name} is not 16-byte aligned", *pointers.values(), 16, 16, 16)


@pytest.mark.parametrize("name", ["y", "bias", "residual"])
def test_buffer_may_not_run_past_the_address_space(flint, name):
    # The check runs before any memory is touched, so a made-up address will
    # do. y's is 768 bytes from the end: y's 512 bytes as FP16 would fit, but
    # its FP32 extent, 1,024, doesn't.
    pointers = dict(zip(NAMES, small_problem()))
    pointers[name] = 2**64 - (768 if name == "y" else 16)
    expect_invalid_argument(
        flint, f"{name} runs past the end of the address space", *pointers.values(), 16, 16, 16
    )


# The size checks run before any memory is touched, so small tensors do.
TOO_LARGE_CASES = [
    pytest.param(shape, id=name)
    for name, shape in [
        ("m", (8_388_481, 8, 8)),
        ("n", (8, 8_388_482, 8)),
        ("k", (8, 8, 2**16 + 8)),
    ]
]


@pytest.mark.parametrize("shape", TOO_LARGE_CASES)
def test_too_large_dimensions(flint, shape):
    m, n, k = shape
    expect_invalid_argument(flint, "too large", *small_problem(), m, n, k)


# The overlap tests lay the tensors out in one byte buffer. y is 16 x 16 FP32,
# 1,024 bytes, at Y; x and w are 512 bytes each, bias 64 and the residual 1,024.
# Every offset is a multiple of 16, as the alignment checks need.
Y, X, W, BIAS, RESIDUAL = 4096, 8192, 10240, 12288, 14336


def layout(**moved):
    """The addresses of y, x, w, bias and the residual in a fresh buffer, with
    some of them moved to other offsets."""
    buf = torch.zeros(16384, dtype=torch.uint8, device="cuda")
    offsets = {"y": Y, "x": X, "w": W, "bias": BIAS, "residual": RESIDUAL, **moved}
    return buf, [buf.data_ptr() + offsets[name] for name in NAMES]


def test_y_overlapping_x(flint):
    # x starts 256 bytes into y; 768 bytes into it, past the 512 bytes y would
    # have as FP16 and inside its FP32 1,024; 16 bytes before y's end; and it
    # ends 16 bytes into y.
    for at in (Y + 256, Y + 768, Y + 1008, Y - 496):
        buf, pointers = layout(x=at)
        expect_invalid_argument(flint, "y and x overlap", *pointers, 16, 16, 16)


def test_y_overlapping_w(flint):
    for at in (Y + 256, Y + 768, Y + 1008, Y - 496):
        buf, pointers = layout(w=at)
        expect_invalid_argument(flint, "y and w overlap", *pointers, 16, 16, 16)


def test_y_overlapping_bias(flint):
    # bias (64 bytes) starts 256 bytes into y, 768 bytes into it (past y's FP16
    # extent), 16 bytes before y's end, and ends 16 bytes into y.
    for at in (Y + 256, Y + 768, Y + 1008, Y - 48):
        buf, pointers = layout(bias=at)
        expect_invalid_argument(flint, "y and bias overlap", *pointers, 16, 16, 16)


def test_y_overlapping_residual(flint):
    # Both are 1,024 bytes of FP32. The residual starts 256 bytes into y, 512
    # bytes into it (past y's FP16 extent), and 16 bytes before y's end; is y
    # itself (the kernel is out of place); and starts before y, 256 bytes
    # before it and 1,008 bytes before (so that y starts 16 bytes before its
    # end).
    for at in (Y + 256, Y + 512, Y + 1008, Y, Y - 256, Y - 1008):
        buf, pointers = layout(residual=at)
        expect_invalid_argument(flint, "y and residual overlap", *pointers, 16, 16, 16)


@pytest.mark.parametrize("residual_first", [True, False], ids=["residual-then-y", "y-then-residual"])
def test_residual_next_to_y_is_not_overlapping(flint, residual_first):
    # y and the residual (1,024 bytes each) side by side in one buffer, with 16
    # bytes of pattern around them that must stay intact. x and w are ones, so y
    # is the residual plus 16 plus the bias, all exact.
    m = n = k = 16
    buf = torch.full((16 + 2048 + 16,), 0x5A, dtype=torch.uint8, device="cuda")
    residual_at, y_at = (16, 16 + 1024) if residual_first else (16 + 1024, 16)
    residual = buf[residual_at : residual_at + 4 * m * n].view(torch.float32)
    y = buf[y_at : y_at + 4 * m * n].view(torch.float32)
    residual.copy_(100 + 0.25 * torch.arange(m * n, dtype=torch.float32, device="cuda"))
    residual_before = residual.clone()
    bias = 0.25 + 0.5 * torch.arange(n, dtype=torch.float32, device="cuda")
    x = torch.ones((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")

    flint.gemm_nt_bias_residual(y, x, w, bias, residual, m, n, k)
    torch.cuda.synchronize()

    assert torch.equal(y.view(m, n), residual_before.view(m, n) + 16 + bias)
    assert torch.equal(residual, residual_before)
    assert (buf[:16] == 0x5A).all() and (buf[16 + 2048 :] == 0x5A).all()


@pytest.mark.parametrize("which", ["x", "w", "bias", "all"])
def test_residual_may_overlap_the_inputs(flint, which):
    # The residual is read like x, w and bias, so it may share their bytes: the
    # first half of its 1,024 bytes can be x, the second half w, and bias 32
    # bytes in. The floats are bfloat16 values, whose low halves are zero, so
    # that read as FP16 pairs they are finite: 0.0 and a number.
    m = n = k = 16

    def triples():
        for seed in gate.SEEDS:
            g = generator(seed)
            shared = torch.randn(m * n, generator=g, device="cuda").bfloat16().float()
            halves = shared.view(torch.float16)
            assert torch.isfinite(halves).all()
            x = halves[: m * k].view(m, k) if which in ("x", "all") else normal((m, k), 0.5, g)
            w = halves[m * k :].view(n, k) if which in ("w", "all") else normal((n, k), 0.5, g)
            bias = shared[8 : 8 + n] if which in ("bias", "all") else make_bias(n, seed)
            residual = shared.view(m, n)
            x_copy, w_copy, bias_copy, residual_copy = (t.clone() for t in (x, w, bias, residual))
            y = torch.empty((m, n), dtype=torch.float32, device="cuda")
            flint.gemm_nt_bias_residual(y, x, w, bias, residual, m, n, k)
            torch.cuda.synchronize()
            assert torch.equal(raw(shared), raw(residual_copy)), "the shared input changed"
            yield (
                increment(y, residual_copy),
                increment(peer_out(x_copy, w_copy, bias_copy, residual_copy), residual_copy),
                gate.reference_bias(x_copy, w_copy, bias_copy),
            )

    gate.gate(gate.FORWARD_FACTOR, triples())


@pytest.mark.parametrize("m, n", [(0, 16), (16, 0), (0, 0)])
def test_empty_output_does_nothing(flint, m, n):
    # Not even the null bias or residual is looked at.
    y, x, w, bias, residual = small_problem()
    y.fill_(3)

    flint.gemm_nt_bias_residual(y, x, w, None, residual, m, n, 16)
    flint.gemm_nt_bias_residual(y, x, w, bias, None, m, n, 16)
    flint.gemm_nt_bias_residual(None, None, None, None, None, m, n, 16)
    flint.gemm_nt_bias_residual(y, x, w, bias, residual, m, n, 16)
    torch.cuda.synchronize()

    assert (y == 3).all()


def test_a_valid_call_works_after_errors(flint):
    y, x, w, bias, residual = small_problem()
    bias[:16] = 0.5
    residual[:256] = 2.0
    for k in (0, 12):
        with pytest.raises(flint_lib.FlintError):
            flint.gemm_nt_bias_residual(y, x, w, bias, residual, 16, 16, k)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias_residual(y, x, w, bias, residual, 16, 7, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias_residual(None, x, w, bias, residual, 16, 16, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias_residual(y, x, w, None, residual, 16, 16, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias_residual(y, x, w, bias, None, 16, 16, 16)

    flint.gemm_nt_bias_residual(y, x, w, bias, residual, 16, 16, 16)
    torch.cuda.synchronize()

    assert (y[:256] == 18.5).all()
    assert (y[256:] == 0).all()
