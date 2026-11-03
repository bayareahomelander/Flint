"""flint_layernorm_forward: LayerNorm's forward pass over the rows of an FP32
matrix x (rows x cols). For each row it stores the mean and rstd =
1 / sqrt(biased variance + eps) (FP32, one each per row), and writes
y = (x - mean) * rstd * gamma + beta rounded once to FP16, with gamma and beta
holding cols FP32 values. The statistics use compensated sums, so they are
more accurate than PyTorch's.

The kernel has two variants (cols a multiple of 4 reads 128 bits at a time,
any other cols reads a float at a time), and a warp takes each row, 8 rows to
a block, so the tests cover both variants and row counts that leave a partial
block.

The correctness gate has the reference in float64 (torch.native_layer_norm on
the FP32 inputs upcast) and the peer in FP32 (the same function in FP32, which
is what autocast runs, with y rounded to FP16). y goes through the FP16 gate;
mean and rstd are FP32 outputs, so they use the gate's FP32 floors. Exact
tests then need no tolerance: constant rows, a gamma of zero, a single
column, and rows that don't depend on the rest of the batch. After those come
FP16's range (overflow to inf and subnormal outputs), non-finite inputs,
determinism, stream ordering, guard bands, and argument errors. The last
section tests the gate's FP32 floors themselves."""

import math

import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator

EPS = 1e-5

# Under the sanitizer the GPT-2 shapes shrink.
GPT2_ROWS = (64,) if SANITIZER else (8192, 16384)
# GPT-2 medium, large and XL widths run at a few hundred rows, not a multiple of 8.
MID_ROWS = 37 if SANITIZER else 300

# The outputs of a guarded call sit `offset` blocks past the start of buffers
# that are filled with a pattern, which must be intact around them afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A

# What the gate checks, and the floors each output type takes.
OUTPUTS = (("y", "fp16"), ("mean", "fp32"), ("rstd", "fp32"))


def shape_id(shape):
    return "x".join(map(str, shape))


def as_int32(value):
    """A 32-bit pattern as the signed value torch.int32 holds."""
    return (value + 0x80000000) % 0x100000000 - 0x80000000


def bits(t):
    """t's bit patterns (16 bits for FP16, else 32), which compare equal for
    equal bits, NaNs and -0.0 included."""
    return t.contiguous().view(torch.int16 if t.dtype == torch.float16 else torch.int32)


def float32_value(v):
    """v rounded to FP32 as a Python float: the eps the kernel receives."""
    return torch.tensor(v, dtype=torch.float32).item()


# -- Inputs -----------------------------------------------------------------
# Every maker returns x (rows x cols), gamma (cols) and beta (cols), all FP32.


def randn(shape, g):
    return torch.randn(shape, generator=g, device="cuda")


def realistic(rows, cols, seed):
    """x about N(0, 1), gamma about N(1, 0.2) and beta about N(0, 0.1)."""
    g = generator(seed)
    x = randn((rows, cols), g)
    return x, 1.0 + 0.2 * randn(cols, g), 0.1 * randn(cols, g)


def init(rows, cols, seed):
    """As a freshly initialized layer: gamma 1 and beta 0."""
    g = generator(seed)
    x = randn((rows, cols), g)
    return x, torch.ones(cols, device="cuda"), torch.zeros(cols, device="cuda")


def outliers(rows, cols, seed):
    """Realistic inputs with two channels near +3,000 and -2,500 in every row,
    as in GPT-2's residual stream (when cols allows two channels)."""
    x, gamma, beta = realistic(rows, cols, seed)
    if cols >= 2:
        g = generator(10_000 + seed)
        jitter = 0.9 + 0.2 * torch.rand((rows, 2), generator=g, device="cuda")
        x[:, cols // 3] = 3000.0 * jitter[:, 0]
        x[:, 2 * cols // 3] = -2500.0 * jitter[:, 1]
    return x, gamma, beta


def large_mean(rows, cols, seed):
    """x about 1000 + N(0, 1): the mean is large against the spread."""
    x, gamma, beta = realistic(rows, cols, seed)
    return x + 1000.0, gamma, beta


def small_spread(rows, cols, seed):
    """x about 5 + N(0, 0.001): the variance, 1e-6, is below eps."""
    x, gamma, beta = realistic(rows, cols, seed)
    return 5.0 + 1e-3 * x, gamma, beta


INPUTS = {
    "realistic": realistic,
    "init": init,
    "outliers": outliers,
    "large_mean": large_mean,
    "small_spread": small_spread,
}


# -- The reference, the peer, and running Flint -----------------------------


def reference_layernorm(x, gamma, beta, eps=EPS):
    """(y, mean, rstd) in float64 from torch.native_layer_norm on the upcast
    inputs, with eps as the kernel receives it (rounded to FP32). mean and
    rstd are flattened to rows values."""
    rows, cols = x.shape
    y, mean, rstd = torch.native_layer_norm(
        x.double(), [cols], gamma.double(), beta.double(), float32_value(eps)
    )
    return y, mean.reshape(rows), rstd.reshape(rows)


def peer_layernorm(x, gamma, beta, eps=EPS):
    """(y, mean, rstd) from PyTorch's FP32 layer norm, as autocast runs it, with
    y rounded to FP16."""
    rows, cols = x.shape
    y, mean, rstd = torch.native_layer_norm(x, [cols], gamma, beta, float32_value(eps))
    return y.half(), mean.reshape(rows), rstd.reshape(rows)


def flint_layernorm(flint, x, gamma, beta, eps=EPS, offset=1, fill=SENTINEL, stream=None):
    """Runs flint_layernorm_forward with y, mean and rstd each inside a buffer
    pre-filled with `fill` (a 16-bit pattern, repeated for the 32-bit buffers),
    8 * offset halves in for y and 4 * offset floats in for the statistics, which
    keeps them 16-byte aligned. Checks that the buffers around the outputs and
    the inputs are untouched, and returns (y as rows x cols FP16, mean, rstd)."""
    rows, cols = x.shape
    y_at, stat_at = 8 * offset, 4 * offset
    y_buffer = torch.full(
        (y_at + rows * cols + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda"
    )
    stat_buffers = [
        torch.full(
            (stat_at + rows + GUARD_AFTER,),
            as_int32(fill * 0x10001),
            dtype=torch.int32,
            device="cuda",
        )
        for _ in range(2)
    ]
    y = y_buffer[y_at : y_at + rows * cols].view(torch.float16)
    mean = stat_buffers[0][stat_at : stat_at + rows].view(torch.float32)
    rstd = stat_buffers[1][stat_at : stat_at + rows].view(torch.float32)
    inputs_before = [bits(t).clone() for t in (x, gamma, beta)]

    flint.layernorm_forward(y, mean, rstd, x, gamma, beta, rows, cols, eps, stream=stream)
    torch.cuda.synchronize()

    guard16 = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert (y_buffer[:y_at] == guard16).all(), "flint_layernorm_forward wrote before y"
    assert (y_buffer[y_at + rows * cols :] == guard16).all(), "flint_layernorm_forward wrote past y"
    guard32 = torch.full((1,), as_int32(fill * 0x10001), dtype=torch.int32, device="cuda")
    for name, buffer in zip(("mean", "rstd"), stat_buffers):
        assert (buffer[:stat_at] == guard32).all(), f"flint_layernorm_forward wrote before {name}"
        assert (buffer[stat_at + rows :] == guard32).all(), (
            f"flint_layernorm_forward wrote past {name}"
        )
    for name, t, before in zip(("x", "gamma", "beta"), (x, gamma, beta), inputs_before):
        assert torch.equal(bits(t), before), f"{name} changed"
    return y.view(rows, cols), mean, rstd


def gate_cases(flint, make_inputs, rows, cols, eps=EPS, offset=1):
    """Per seed, the outputs (y, mean, rstd) of Flint, of the peer and of the
    reference."""
    for seed in gate.SEEDS:
        x, gamma, beta = make_inputs(rows, cols, seed)
        yield (
            flint_layernorm(flint, x, gamma, beta, eps, offset),
            peer_layernorm(x, gamma, beta, eps),
            reference_layernorm(x, gamma, beta, eps),
        )


def check_gate(cases):
    """Applies the gate to every output of each seed's (flint, peer, reference)
    outputs: y with FP16's floors, mean and rstd with FP32's. Tensors of one
    output share a shape (mask non-finite elements out of all three first)."""
    flint_errors = [[] for _ in OUTPUTS]
    peer_errors = [[] for _ in OUTPUTS]
    for flint_out, peer_out, ref_out in cases:
        for i, (f, p, r) in enumerate(zip(flint_out, peer_out, ref_out)):
            flint_errors[i].append(gate.errors(f, r))
            peer_errors[i].append(gate.errors(p, r))
        del flint_out, peer_out, ref_out
    for i, (name, output) in enumerate(OUTPUTS):
        try:
            gate.check_gate(gate.FORWARD_FACTOR, flint_errors[i], peer_errors[i], output)
        except AssertionError as error:
            raise AssertionError(f"{name}:\n{error}") from None


# -- The gate ---------------------------------------------------------------

# GPT-2 small's width at its training batch's row counts (the 8192 rows of a
# micro-batch, and twice that), then the wider models' widths.
GPT2_SHAPES = [(rows, 768) for rows in GPT2_ROWS] + [
    (MID_ROWS, cols) for cols in (1024, 1280, 1600)
]

GPT2_CASES = [
    pytest.param(shape, variant, id=f"{shape_id(shape)}-{variant}")
    for shape in GPT2_SHAPES
    for variant in INPUTS
]


@pytest.mark.parametrize("shape, variant", GPT2_CASES)
def test_gpt2_shapes_pass_the_gate(flint, shape, variant):
    rows, cols = shape
    check_gate(gate_cases(flint, INPUTS[variant], rows, cols))


# Both variants (cols a multiple of 4 or not), cols below a warp's 32 lanes and
# above them, a single row, and rows that leave a partial block of 8.
ODD_SHAPES = [
    (1, 1),
    (1, 4),
    (3, 7),
    (5, 3),
    (7, 8),
    (9, 33),
    (37, 1001),
    (13, 2047),
    (8, 132),
    (17, 999),
]

# Offsets rotate through 1, 2 and 3 blocks (8, 16 and 24 halves for y).
ODD_CASES = [
    pytest.param(shape, variant, 1 + (i + j) % 3, id=f"{shape_id(shape)}-{variant}")
    for i, shape in enumerate(ODD_SHAPES)
    for j, variant in enumerate(INPUTS)
]


@pytest.mark.parametrize("shape, variant, offset", ODD_CASES)
def test_odd_shapes_pass_the_gate_and_stay_inside_their_outputs(flint, shape, variant, offset):
    # flint_layernorm also checks that the guard regions and the inputs are
    # intact.
    rows, cols = shape
    check_gate(gate_cases(flint, INPUTS[variant], rows, cols, offset=offset))


@pytest.mark.parametrize("eps", [0.0, 1e-3, 1.0], ids=str)
@pytest.mark.parametrize("shape", [(64, 768), (37, 1001)], ids=shape_id)
def test_other_eps_values_pass_the_gate(flint, shape, eps):
    # eps = 0 is accepted, and a large eps changes the output much more than
    # the default does.
    rows, cols = shape
    check_gate(gate_cases(flint, realistic, rows, cols, eps=eps))


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@pytest.mark.parametrize("shape", [(1, 2**24), (3, 100_003)], ids=shape_id)
def test_wide_rows_pass_the_gate(flint, shape):
    # cols at the limit (the divisor 2^24 is the last count FP32 holds
    # exactly), and an odd width with more than 3,000 values per lane.
    rows, cols = shape
    check_gate(gate_cases(flint, outliers, rows, cols))


@pytest.mark.parametrize("variant", ["realistic", "outliers", "large_mean"])
@pytest.mark.parametrize("shape", [(64, 768), (37, 1001)], ids=shape_id)
def test_statistics_are_accurate_to_an_ulp_or_two(flint, shape, variant):
    # The compensated sums make mean and rstd about as accurate as FP32 allows:
    # the mean's two roundings (the sum, then the division) leave it within
    # about 1 ulp of the float64 reference (1.004 at worst over 20,000 rows),
    # and rstd, after a variance, a sqrt and a reciprocal, within 3 (2.4 at
    # worst). PyTorch's mean is off by up to thousands of ulps on the same kind of rows, since
    # its error is relative to the values it adds, not to their mean, so the
    # gate alone (which only asks for twice as good as the peer) would let a
    # regression to plain sums through.
    rows, cols = shape
    flint_worst = {"mean": 0.0, "rstd": 0.0}
    peer_worst = {"mean": 0.0, "rstd": 0.0}
    for seed in gate.SEEDS:
        x, gamma, beta = INPUTS[variant](rows, cols, seed)
        _, mean, rstd = flint_layernorm(flint, x, gamma, beta)
        _, peer_mean, peer_rstd = peer_layernorm(x, gamma, beta)
        _, ref_mean, ref_rstd = reference_layernorm(x, gamma, beta)
        for name, out, peer, ref in (
            ("mean", mean, peer_mean, ref_mean),
            ("rstd", rstd, peer_rstd, ref_rstd),
        ):
            # The spacing of FP32 at each reference value, in float64.
            ref32 = ref.float()
            spacing = (torch.nextafter(ref32, torch.full_like(ref32, math.inf)) - ref32).double()
            flint_worst[name] = max(flint_worst[name], ((out.double() - ref).abs() / spacing).max().item())
            peer_worst[name] = max(peer_worst[name], ((peer.double() - ref).abs() / spacing).max().item())
    assert flint_worst["mean"] <= 1.5, (flint_worst, peer_worst)
    assert flint_worst["rstd"] <= 3.0, (flint_worst, peer_worst)
    assert flint_worst["mean"] < peer_worst["mean"], (flint_worst, peer_worst)


# -- Exact checks -----------------------------------------------------------

# Exactly representable constants, whose sums and means are exact in FP32 for
# these widths. A row of one of them has no deviation from its mean.
CONSTANTS = [0.75, -3.0, 0.0, 1000.5]
CONSTANT_SHAPES = [(8, 768), (9, 33), (13, 2047), (37, 1001), (3, 7), (5, 3), (7, 8)]


def constant_rows(rows, cols):
    values = torch.tensor([CONSTANTS[r % len(CONSTANTS)] for r in range(rows)], device="cuda")
    return values, values[:, None].expand(rows, cols).contiguous()


def expected_rstd(eps):
    """1 / sqrt(eps) computed in float32 by torch: what a constant row's rstd
    must be bit for bit, since the kernel's sqrt and divide are correctly
    rounded."""
    return torch.tensor(eps, dtype=torch.float32, device="cuda").sqrt().reciprocal()


@pytest.mark.parametrize("eps", [EPS, 0.1, 1.0, 1e-30, 1e30], ids=str)
@pytest.mark.parametrize("shape", CONSTANT_SHAPES, ids=shape_id)
def test_constant_rows_give_the_constant_and_beta(flint, shape, eps):
    # The variance is exactly 0, so rstd is 1/sqrt(eps) and the normalized
    # value is exactly 0: y is beta rounded to FP16 whatever gamma is, and the
    # mean is the constant. (beta is never zero, so the sign of a zero product
    # can't change the result's bits.)
    rows, cols = shape
    values, x = constant_rows(rows, cols)
    _, gamma, beta = realistic(rows, cols, 0)

    y, mean, rstd = flint_layernorm(flint, x, gamma, beta, eps)

    assert torch.equal(bits(mean), bits(values))
    assert torch.equal(bits(rstd), bits(expected_rstd(float32_value(eps)).expand(rows)))
    assert torch.equal(bits(y), bits(beta.half().expand(rows, cols)))


@pytest.mark.parametrize("shape", [(8, 768), (9, 33), (3, 7)], ids=shape_id)
def test_eps_zero_gives_inf_rstd_and_nan_for_constant_rows(flint, shape):
    # With eps = 0 a constant row has variance 0 and rstd = 1/sqrt(0) = inf, and
    # the normalized value is 0 * inf = NaN, as in PyTorch. The row that varies
    # is unaffected.
    rows, cols = shape
    values, x = constant_rows(rows, cols)
    x[-1] = randn(cols, generator(0))
    _, gamma, beta = realistic(rows, cols, 0)
    peer_y, _, peer_rstd = peer_layernorm(x, gamma, beta, 0.0)
    assert torch.isinf(peer_rstd[:-1]).all() and torch.isnan(peer_y[:-1]).all()

    y, mean, rstd = flint_layernorm(flint, x, gamma, beta, 0.0)

    assert torch.isposinf(rstd[:-1]).all()
    assert torch.isnan(y[:-1]).all()
    assert torch.equal(bits(mean[:-1]), bits(values[:-1]))
    assert torch.isfinite(rstd[-1]) and torch.isfinite(y[-1]).all()


@pytest.mark.parametrize("variant", ["realistic", "outliers"])
@pytest.mark.parametrize("shape", [(8, 768), (9, 33), (37, 1001), (13, 2047), (1, 1)], ids=shape_id)
def test_a_gamma_of_zero_gives_beta(flint, shape, variant):
    # (x - mean) * rstd is finite, and times 0 it adds nothing to beta.
    rows, cols = shape
    for seed in gate.SEEDS:
        x, _, beta = INPUTS[variant](rows, cols, seed)
        gamma = torch.zeros(cols, device="cuda")

        y, _, _ = flint_layernorm(flint, x, gamma, beta)

        assert torch.equal(bits(y), bits(beta.half().expand(rows, cols)))


@pytest.mark.parametrize("eps", [EPS, 0.0], ids=str)
def test_a_single_column_is_its_own_mean(flint, eps):
    # Every row is a constant: the mean is x itself, the variance 0, and y is
    # beta (rstd is inf and y NaN for eps = 0).
    rows, cols = 37, 1
    x, gamma, beta = realistic(rows, cols, 0)

    y, mean, rstd = flint_layernorm(flint, x, gamma, beta, eps)

    assert torch.equal(bits(mean), bits(x[:, 0]))
    if eps > 0:
        assert torch.equal(bits(rstd), bits(expected_rstd(float32_value(eps)).expand(rows)))
        assert torch.equal(bits(y), bits(beta.half().expand(rows, cols)))
    else:
        assert torch.isposinf(rstd).all()
        assert torch.isnan(y).all()


@pytest.mark.parametrize("cols", [768, 1001], ids=["vector", "scalar"])
def test_a_row_does_not_depend_on_its_batch(flint, cols):
    # Each row, run alone in fresh tensors (inside a batch a row's address may
    # not be 16-byte aligned), gives the same bits as inside the batch.
    rows = 19
    x, gamma, beta = outliers(rows, cols, 0)
    y, mean, rstd = flint_layernorm(flint, x, gamma, beta)

    for r in range(rows):
        single_y, single_mean, single_rstd = flint_layernorm(flint, x[r : r + 1].clone(), gamma, beta)
        assert torch.equal(bits(single_y[0]), bits(y[r])), r
        assert torch.equal(bits(single_mean), bits(mean[r : r + 1])), r
        assert torch.equal(bits(single_rstd), bits(rstd[r : r + 1])), r


# -- FP16's range -----------------------------------------------------------


@pytest.mark.parametrize("shape", [(150, 133), (64, 768)], ids=shape_id)
def test_overflow_gives_inf_with_the_references_sign(flint, shape):
    rows, cols = shape
    # gamma about N(0, 30,000) and a normalized value about N(0, 1) put about
    # one output in ten past FP16's range. The reference rounds to inf from
    # 65,520 up (65,504 is the largest FP16). A thin band, 0.01% either side of
    # that, is left out because FP32 error can decide it either way.
    inf_threshold = 65520.0 * (1 + 1e-4)
    finite_threshold = 65520.0 * (1 - 1e-4)

    def cases():
        for seed in gate.SEEDS:
            x, _, beta = realistic(rows, cols, seed)
            gamma = 30_000.0 * randn(cols, generator(20_000 + seed))
            y, mean, rstd = flint_layernorm(flint, x, gamma, beta)
            peer_y, peer_mean, peer_rstd = peer_layernorm(x, gamma, beta)
            ref_y, ref_mean, ref_rstd = reference_layernorm(x, gamma, beta)

            assert not torch.isnan(y).any(), "overflow produced NaN"
            too_big = ref_y.abs() >= inf_threshold
            fits = ref_y.abs() <= finite_threshold
            # Each class and sign has to occur, or the test checks nothing.
            assert (too_big & (ref_y > 0)).any() and (too_big & (ref_y < 0)).any()
            assert fits.sum() > 0.5 * ref_y.numel()
            expected = torch.where(ref_y > 0, math.inf, -math.inf).half()
            assert torch.equal(y[too_big], expected[too_big]), "wrong overflow to inf"
            assert torch.isfinite(y[fits]).all(), "finite output became inf"
            yield (
                (y[fits], mean, rstd),
                (peer_y[fits], peer_mean, peer_rstd),
                (ref_y[fits], ref_mean, ref_rstd),
            )

    check_gate(cases())


@pytest.mark.parametrize("shape", [(150, 133), (64, 768)], ids=shape_id)
def test_subnormal_outputs_pass_the_gate(flint, shape):
    # gamma about 1e-6 and beta about 1e-7 put y around 1e-6, in FP16's
    # subnormal range (below 2^-14 = 6.1e-5, in steps of 2^-24 = 6e-8), where
    # the rounding is coarse relative to the value.
    rows, cols = shape

    def cases():
        for seed in gate.SEEDS:
            x, gamma, beta = realistic(rows, cols, seed)
            gamma, beta = 1e-6 * gamma, 1e-6 * beta
            y, mean, rstd = flint_layernorm(flint, x, gamma, beta)
            assert (y.abs() < 2.0**-14).all()
            assert (y != 0).float().mean() > 0.9, "the output is mostly zero, not subnormal"
            yield (
                (y, mean, rstd),
                peer_layernorm(x, gamma, beta),
                reference_layernorm(x, gamma, beta),
            )

    check_gate(cases())


# -- Non-finite inputs ------------------------------------------------------


@pytest.mark.parametrize("cols", [768, 1001], ids=["vector", "scalar"])
def test_a_nonfinite_input_makes_only_its_row_nan(flint, cols):
    # An inf or NaN anywhere in a row gives NaN for the row's mean and rstd and
    # all of its y, as in PyTorch. The other rows, in the same blocks and
    # others, come out bit for bit as they do without the bad rows.
    rows = 21
    bad = {
        0: [(0, math.inf)],
        5: [(cols // 2, -math.inf)],
        9: [(1, math.inf), (2, -math.inf)],
        14: [(cols - 1, math.nan)],
        20: [(3, math.nan)],
    }
    x, gamma, beta = realistic(rows, cols, 0)
    x_bad = x.clone()
    for row, entries in bad.items():
        for column, value in entries:
            x_bad[row, column] = value
    bad_rows = torch.tensor(sorted(bad), device="cuda")
    good = torch.ones(rows, dtype=torch.bool, device="cuda")
    good[bad_rows] = False

    clean = flint_layernorm(flint, x, gamma, beta)
    dirty = flint_layernorm(flint, x_bad, gamma, beta)
    peer = peer_layernorm(x_bad, gamma, beta)

    for name, out, peer_out in zip(("y", "mean", "rstd"), dirty, peer):
        assert torch.isnan(out[bad_rows]).all(), name
        assert torch.isnan(peer_out[bad_rows]).all(), f"PyTorch's {name} differs"
        assert torch.isfinite(out[good]).all(), name
    for out, clean_out in zip(dirty, clean):
        assert torch.equal(bits(out[good]), bits(clean_out[good]))


def test_a_row_whose_sums_overflow_is_nan(flint):
    # Once a compensated sum overflows, TwoSum's error is inf - inf. A row of
    # two 2e38s overflows the mean's sum, so all three outputs are NaN. A row
    # of 1e20 and about 0, with deviations of 5e19 from its mean, overflows
    # only the sum of their squares (a square overflows from about 1.8e19), so
    # its mean stays finite and rstd and y are NaN. flint.h documents both;
    # PyTorch's Welford keeps both rows finite. The other rows are unaffected.
    rows, cols = 8, 2
    x, gamma, beta = realistic(rows, cols, 0)
    clean = flint_layernorm(flint, x, gamma, beta)
    x[2] = 2e38
    x[5, 0] = 1e20

    y, mean, rstd = dirty = flint_layernorm(flint, x, gamma, beta)

    assert torch.isnan(y[2]).all() and torch.isnan(mean[2]) and torch.isnan(rstd[2])
    assert torch.isnan(y[5]).all() and torch.isnan(rstd[5])
    assert mean[5] == torch.tensor([1e20, x[5, 1].item()], dtype=torch.float64).mean().float()
    good = torch.ones(rows, dtype=torch.bool, device="cuda")
    good[[2, 5]] = False
    for out, clean_out in zip(dirty, clean):
        assert torch.equal(bits(out[good]), bits(clean_out[good]))


# -- Determinism ------------------------------------------------------------

DETERMINISM_SHAPES = [(GPT2_ROWS[0], 768), (37, 1001)]


@pytest.mark.parametrize("shape", DETERMINISM_SHAPES, ids=shape_id)
def test_run_twice_is_bitwise_identical(flint, shape):
    rows, cols = shape
    x, gamma, beta = outliers(rows, cols, 0)
    # Different fills to start from (all NaN, then a pattern that is NaN as FP16
    # and 3.4e38 as FP32), so a store that is skipped, or one that lands in
    # only one run, shows as a difference or a leftover NaN.
    first = flint_layernorm(flint, x, gamma, beta, fill=0xFFFF)
    second = flint_layernorm(flint, x, gamma, beta, fill=0x7F7F)

    for a, b in zip(first, second):
        assert not torch.isnan(a).any()
        assert not torch.isnan(b).any()
        assert torch.equal(bits(a), bits(b))


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 15, 16, 17, 33])
@pytest.mark.parametrize("cols", [33, 100], ids=["scalar", "vector"])
def test_every_output_is_written_and_nothing_else(flint, rows, cols):
    # The buffers start as NaN, so a row (a partial block's last one, say) that
    # is skipped shows; flint_layernorm checks the guards around them.
    x, gamma, beta = realistic(rows, cols, 0)

    y, mean, rstd = flint_layernorm(flint, x, gamma, beta, fill=0xFFFF)

    for out in (y, mean, rstd):
        assert torch.isfinite(out).all()


# -- Streams ----------------------------------------------------------------


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit-stream", "current-stream"])
def test_layernorm_runs_on_the_given_stream(flint, explicit):
    rows, cols = 16, 64
    x = torch.zeros((rows, cols), dtype=torch.float32, device="cuda")
    gamma = torch.ones(cols, dtype=torch.float32, device="cuda")
    beta = torch.zeros(cols, dtype=torch.float32, device="cuda")
    y = torch.zeros((rows, cols), dtype=torch.float16, device="cuda")
    mean = torch.zeros(rows, dtype=torch.float32, device="cuda")
    rstd = torch.zeros(rows, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fills finish only after the sleep, so y is 0.5 (every row is the
    # constant 2.5, and y = beta) and the mean 2.5 only if the kernel is queued
    # on the same stream behind them. With the current stream the call has no
    # stream argument, and torch's current stream is the side one.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(2.5)
        beta.fill_(0.5)
        flint.layernorm_forward(
            y, mean, rstd, x, gamma, beta, rows, cols, stream=stream if explicit else None
        )
    stream.synchronize()

    assert (y == 0.5).all()
    assert (mean == 2.5).all()
    assert torch.isfinite(rstd).all() and (rstd > 0).all()


# -- Argument errors --------------------------------------------------------

NAMES = ("y", "mean", "rstd", "x", "gamma", "beta")


def small_problem(rows=8, cols=16):
    """Valid tensors by name, in buffers with room to spare."""
    return {
        "y": torch.zeros(rows * cols + 64, dtype=torch.float16, device="cuda"),
        "mean": torch.zeros(rows + 64, dtype=torch.float32, device="cuda"),
        "rstd": torch.zeros(rows + 64, dtype=torch.float32, device="cuda"),
        "x": torch.ones(rows * cols + 64, dtype=torch.float32, device="cuda"),
        "gamma": torch.ones(cols + 64, dtype=torch.float32, device="cuda"),
        "beta": torch.zeros(cols + 64, dtype=torch.float32, device="cuda"),
    }


def call(flint, buffers, rows=8, cols=16, eps=EPS):
    flint.layernorm_forward(*(buffers[name] for name in NAMES), rows, cols, eps)


def expect_invalid_argument(flint, reason, buffers, rows=8, cols=16, eps=EPS):
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, buffers, rows, cols, eps)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_layernorm_forward", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    expect_invalid_argument(flint, f"{name} is null", {**small_problem(), name: None})


@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name):
    # y is FP16, but a multiple of 4 bytes short of 16-byte alignment is still
    # misaligned for the FP32 buffers; y goes 2 bytes off.
    addresses = {name: t.data_ptr() for name, t in small_problem().items()}
    addresses[name] += 2 if name == "y" else 4
    expect_invalid_argument(flint, f"{name} is not 16-byte aligned", addresses)


@pytest.mark.parametrize("name", NAMES)
def test_a_buffer_may_not_run_past_the_address_space(flint, name):
    # The check runs before any memory is touched, so a made-up address will do.
    expect_invalid_argument(
        flint,
        f"{name} runs past the end of the address space",
        {**small_problem(), name: 2**64 - 16},
    )


def test_pointers_are_checked_in_order(flint):
    # Per buffer: null, then alignment, then the address space; and the buffers
    # in the order y, mean, rstd, x, gamma, beta.
    addresses = {name: t.data_ptr() for name, t in small_problem().items()}
    expect_invalid_argument(
        flint, "y is not 16-byte aligned", {**addresses, "y": addresses["y"] + 2, "x": None}
    )
    expect_invalid_argument(
        flint, "mean is null", {**addresses, "mean": None, "rstd": addresses["rstd"] + 4}
    )
    expect_invalid_argument(
        flint, "x is not 16-byte aligned", {**addresses, "x": addresses["x"] + 4, "gamma": None}
    )
    expect_invalid_argument(
        flint, "gamma runs past the end", {**addresses, "gamma": 2**64 - 16, "beta": None}
    )


@pytest.mark.parametrize("cols", [0, 2**24 + 1, 2**64 - 1], ids=str)
def test_cols_must_be_from_1_to_2_to_the_24(flint, cols):
    expect_invalid_argument(
        flint, f"cols ({cols}) must be from 1 to 16777216", small_problem(), cols=cols
    )


@pytest.mark.parametrize("rows", [17_179_869_177, 2**64 - 1], ids=str)
def test_too_many_rows(flint, rows):
    expect_invalid_argument(
        flint, f"rows ({rows}) can be up to 17179869176", small_problem(), rows=rows
    )


def test_the_most_rows_pass_the_size_checks(flint):
    # The limit itself isn't an error of size: the call gets on to the pointers
    # (a null y), even at the widest cols, where rows * cols * 4 is nearly 2^64
    # / 16.
    for cols in (16, 2**24):
        expect_invalid_argument(
            flint, "y is null", {**small_problem(), "y": None}, rows=17_179_869_176, cols=cols
        )


def test_size_limits_are_checked_in_order(flint):
    # cols, then rows, then eps; each before any pointer, null ones included.
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(
        flint, "cols (0) must be from 1 to", nothing, rows=2**40, cols=0, eps=math.nan
    )
    expect_invalid_argument(
        flint, "rows (1099511627776) can be up to", nothing, rows=2**40, eps=math.nan
    )
    expect_invalid_argument(flint, "must be finite and non-negative", nothing, eps=math.nan)


@pytest.mark.parametrize(
    "eps, text",
    [
        (math.nan, "nan"),
        (math.inf, "inf"),
        (-math.inf, "-inf"),
        (-1e-5, "-1e-05"),
        (-1.0, "-1"),
    ],
    ids=str,
)
def test_eps_must_be_finite_and_not_negative(flint, eps, text):
    expect_invalid_argument(
        flint, f"eps ({text}) must be finite and non-negative", small_problem(), eps=eps
    )


# The bytes each buffer covers in the overlap tests, for 8 rows of 16 columns.
SIZES = {"y": 256, "mean": 32, "rstd": 32, "x": 512, "gamma": 64, "beta": 64}
ARENA_BYTES = 1 << 17
SPACING = 4096


def overlap_addresses(arena, first, second, shift):
    """Every buffer in the arena, SPACING bytes apart (far more than any size),
    except that `second` starts `shift` bytes after `first`."""
    base = arena.data_ptr() + 16 * SPACING
    addresses = {name: base + SPACING * i for i, name in enumerate(NAMES)}
    addresses[second] = addresses[first] + shift
    return addresses


# How far the second buffer starts after the first: at the same address, inside
# the first, 16 bytes before the first's end, and (negative) with the second
# ending 16 bytes into the first.
SHIFTS = {
    "same-start": lambda first, second: 0,
    "second-inside": lambda first, second: 16,
    "second-at-end": lambda first, second: SIZES[first] - 16,
    "first-inside": lambda first, second: 16 - SIZES[second],
}

# Each output with every buffer after it in the argument order, at each shift.
OVERLAPS = [
    pytest.param(first, second, shift(first, second), id=f"{first}-{second}-{label}")
    for i, first in enumerate(NAMES[:3])
    for second in NAMES[i + 1 :]
    for label, shift in SHIFTS.items()
]


@pytest.mark.parametrize("first, second, shift", OVERLAPS)
def test_outputs_may_not_overlap_later_buffers(flint, first, second, shift):
    # The buffers are real memory in an arena, so a call that wrongly went
    # ahead would do no harm.
    arena = torch.zeros(ARENA_BYTES, dtype=torch.uint8, device="cuda")
    addresses = overlap_addresses(arena, first, second, shift)
    expect_invalid_argument(flint, f"{first} and {second} overlap", addresses)


def test_the_first_overlap_in_the_argument_order_is_reported(flint):
    arena = torch.zeros(ARENA_BYTES, dtype=torch.uint8, device="cuda")
    addresses = overlap_addresses(arena, "y", "x", 0)
    addresses["mean"] = addresses["y"] + 16
    addresses["rstd"] = addresses["mean"]
    expect_invalid_argument(flint, "y and mean overlap", addresses)
    # Without y involved, mean against rstd comes before mean against x.
    addresses = overlap_addresses(arena, "mean", "x", 0)
    addresses["rstd"] = addresses["mean"] + 16
    expect_invalid_argument(flint, "mean and rstd overlap", addresses)


@pytest.mark.parametrize("order", [("y", "mean", "rstd"), ("rstd", "y", "mean"), ("mean", "rstd", "y")], ids="-".join)
def test_outputs_next_to_each_other_are_not_overlapping(flint, order):
    # y (512 bytes), mean and rstd (64 each) side by side with no gap in one
    # buffer, with a pattern around them that must stay intact.
    rows = cols = 16
    sizes = {"y": 2 * rows * cols, "mean": 4 * rows, "rstd": 4 * rows}
    offsets = {}
    at = 16
    for name in order:
        offsets[name] = at
        at += sizes[name]
    arena = torch.full((at + 16,), 0x5A, dtype=torch.uint8, device="cuda")
    y = arena[offsets["y"] : offsets["y"] + sizes["y"]].view(torch.float16)
    mean = arena[offsets["mean"] : offsets["mean"] + sizes["mean"]].view(torch.float32)
    rstd = arena[offsets["rstd"] : offsets["rstd"] + sizes["rstd"]].view(torch.float32)
    x, gamma, beta = realistic(rows, cols, 0)

    flint.layernorm_forward(y, mean, rstd, x, gamma, beta, rows, cols)
    torch.cuda.synchronize()

    expected = flint_layernorm(flint, x, gamma, beta)
    assert torch.equal(bits(y.view(rows, cols)), bits(expected[0]))
    assert torch.equal(bits(mean), bits(expected[1]))
    assert torch.equal(bits(rstd), bits(expected[2]))
    used = torch.zeros(arena.numel(), dtype=torch.bool, device="cuda")
    used[16:at] = True
    assert (arena[~used] == 0x5A).all()


@pytest.mark.parametrize("sharing", ["gamma-in-x", "gamma-is-beta"])
def test_inputs_may_overlap_each_other(flint, sharing):
    # Inputs are only read, so x, gamma and beta may share memory: gamma and
    # beta are two rows of x (16-byte aligned), or the same buffer.
    rows, cols = 16, 64

    def cases():
        for seed in gate.SEEDS:
            x = randn((rows, cols), generator(seed))
            if sharing == "gamma-in-x":
                gamma, beta = x[0], x[1]
            else:
                gamma = beta = x[2].clone()
            x_copy, gamma_copy, beta_copy = x.clone(), gamma.clone(), beta.clone()
            yield (
                flint_layernorm(flint, x, gamma, beta),
                peer_layernorm(x_copy, gamma_copy, beta_copy),
                reference_layernorm(x_copy, gamma_copy, beta_copy),
            )

    check_gate(cases())


@pytest.mark.parametrize("cols", [16, 2**24])
def test_zero_rows_do_nothing(flint, cols):
    # Not even the null pointers are looked at.
    buffers = small_problem()
    buffers["y"].fill_(3)

    flint.layernorm_forward(None, None, None, None, None, None, 0, cols)
    flint.layernorm_forward(
        buffers["y"], None, buffers["rstd"], buffers["x"], None, buffers["beta"], 0, cols
    )
    call(flint, buffers, rows=0, cols=cols)
    torch.cuda.synchronize()

    assert (buffers["y"] == 3).all()


def test_zero_rows_still_check_cols_and_eps(flint):
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(flint, "cols (0) must be from 1 to", nothing, rows=0, cols=0)
    expect_invalid_argument(flint, "cols (16777217) must be from 1 to", nothing, rows=0, cols=2**24 + 1)
    for eps in (math.nan, math.inf, -1.0):
        expect_invalid_argument(flint, "must be finite and non-negative", nothing, rows=0, eps=eps)


def test_eps_zero_is_accepted(flint):
    buffers = small_problem()
    call(flint, buffers, eps=0.0)
    torch.cuda.synchronize()


def test_a_valid_call_works_after_errors(flint):
    buffers = small_problem()
    buffers["x"][:128] = torch.arange(128, device="cuda") % 16
    for cols in (0, 2**24 + 1):
        with pytest.raises(flint_lib.FlintError):
            call(flint, buffers, cols=cols)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "y": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, buffers, eps=math.nan)

    call(flint, buffers)
    torch.cuda.synchronize()

    # Each row is 0, 1, ..., 15: mean 7.5, and y = (x - 7.5) rstd with gamma 1.
    assert (buffers["mean"][:8] == 7.5).all()
    assert (buffers["mean"][8:] == 0).all()
    variance = 21.25
    expected = (torch.arange(16, device="cuda") - 7.5) / math.sqrt(variance + EPS)
    assert torch.allclose(buffers["y"][:128].view(8, 16).float(), expected.expand(8, 16), atol=2e-3)
    assert (buffers["y"][128:] == 0).all()


# -- gate.py's FP32 floors --------------------------------------------------


def test_fp32_ulp_matches_the_spacing_of_float32():
    # Cross-checked against torch.nextafter at powers of two and at 1.5 times
    # them, from the smallest subnormal to the largest binade, and at 0.
    values = (
        [2.0**e for e in range(-149, 128)]
        + [1.5 * 2.0**e for e in range(-148, 127)]
        + [0.0]
    )
    v = torch.tensor(values, dtype=torch.float32)
    spacing = (torch.nextafter(v, torch.full_like(v, math.inf)).double() - v.double()).tolist()
    for value, expected in zip(v.tolist(), spacing):
        assert gate.fp32_ulp(value) == expected, value


@pytest.mark.parametrize(
    "value, expected",
    [
        (1.0, 2.0**-23),
        (1.9999999, 2.0**-23),
        (2.0, 2.0**-22),
        (0.99999994, 2.0**-24),
        (65504.0, 2.0**-8),
        (2.0**-126, 2.0**-149),
        (2.0**-127, 2.0**-149),
        (0.0, 2.0**-149),
        (3.4028234663852886e38, 2.0**104),
    ],
    ids=str,
)
def test_fp32_ulp_values(value, expected):
    assert gate.fp32_ulp(value) == expected


@pytest.mark.parametrize("value", [-1.0, math.inf, -math.inf, math.nan], ids=str)
def test_fp32_ulp_rejects_values_that_are_not_finite_and_non_negative(value):
    with pytest.raises(ValueError, match="fp32_ulp"):
        gate.fp32_ulp(value)


def test_fp32_floors_are_the_gates_bounds_for_fp32_outputs():
    # A peer with no error leaves only the floors: the FP32 relative L2 floor,
    # and a max-abs of 4 ulps of max|ref| (2^-21 for a reference up to 1.5).
    exact = gate.Errors(0.0, 0.0, 1.5)
    bounds = gate.check_gate(2.0, [exact], [exact], output="fp32")
    assert bounds == (gate.REL_L2_FLOOR_FP32, 2.0**-21)
    # FP16's are what the default gives, and what "fp16" gives.
    expected = (gate.REL_L2_FLOOR, 2.0 * 2.0**-10)
    assert gate.check_gate(2.0, [exact], [exact]) == expected
    assert gate.check_gate(2.0, [exact], [exact], output="fp16") == expected
    # The FP32 floor is the tighter one, but not tighter than one rounding's
    # half ulp.
    assert 2.0**-24 <= gate.REL_L2_FLOOR_FP32 < gate.REL_L2_FLOOR


def test_fp32_floors_tell_fp32_errors_apart():
    # Errors that FP16's floors allow and FP32's don't.
    floor = gate.REL_L2_FLOOR_FP32
    ok = gate.Errors(0.5 * floor, 2.0**-22, 1.5)
    too_big_rel = gate.Errors(2 * floor, 0.0, 1.5)
    too_big_abs = gate.Errors(0.0, 5 * 2.0**-23, 1.5)
    peer = gate.Errors(0.0, 0.0, 1.5)
    gate.check_gate(2.0, [ok], [peer], output="fp32")
    for bad in (too_big_rel, too_big_abs):
        gate.check_gate(2.0, [bad], [peer], output="fp16")
        with pytest.raises(AssertionError, match="FAIL"):
            gate.check_gate(2.0, [bad], [peer], output="fp32")


def test_the_peer_loosens_the_fp32_gate_past_its_floors():
    # Twice the peer's error is the bound when that is above the floors.
    peer = gate.Errors(1e-6, 1e-6, 1.5)
    bounds = gate.check_gate(2.0, [gate.Errors(2e-6, 2e-6, 1.5)], [peer], output="fp32")
    assert bounds == (2e-6, 2e-6)
    with pytest.raises(AssertionError, match="FAIL"):
        gate.check_gate(2.0, [gate.Errors(2.1e-6, 0.0, 1.5)], [peer], output="fp32")


def test_gate_takes_the_output_type_for_tensors():
    # A reference in [1, 2) and outputs 8 FP32 ulps (2^-20) or a quarter ulp
    # above it: the first has a relative L2 near 5e-7, which FP16 outputs are
    # allowed and FP32 ones not, and the second passes both.
    ref = 1.0 + torch.arange(1000, dtype=torch.float64) / 1000.0
    far = ref + 2.0**-20
    near = ref + 2.0**-25
    gate.gate(2.0, [(far, ref, ref)], output="fp16")
    gate.gate(2.0, [(far, ref, ref)])
    with pytest.raises(AssertionError, match="FAIL"):
        gate.gate(2.0, [(far, ref, ref)], output="fp32")
    gate.gate(2.0, [(near, ref, ref)], output="fp32")


@pytest.mark.parametrize("output", ["fp64", "FP16", "", None], ids=str)
def test_an_unknown_output_type_is_a_value_error(output):
    exact = gate.Errors(0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="output"):
        gate.check_gate(2.0, [exact], [exact], output=output)
    with pytest.raises(ValueError, match="output"):
        gate.gate(2.0, [], output=output)
