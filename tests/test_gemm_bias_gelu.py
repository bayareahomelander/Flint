"""flint_gemm_nt_bias_gelu: a linear layer's forward pass with its bias and a
GELU after it, as the MLP's up projection runs them. It computes the FP32 sums
s = x w^T + bias exactly as flint_gemm_nt_bias does, stores pre = s rounded to
FP16 (which GELU's backward pass needs), and stores y = GELU(s) rounded to FP16,
with GELU, GPT-2's tanh approximation, applied in FP32 to the unrounded s.

The tests cover the correctness gate on y (the peer is PyTorch's FP16 linear
layer followed by its tanh GELU, which rounds the bias and the sums to FP16
before the GELU, so it is the coarser of the two); pre, which must equal
flint_gemm_nt_bias's output bit for bit, on every gated case; and the GELU
function itself, with a zero input so that the sums are exactly the bias, which
a dense sweep holds to one FP16 ulp everywhere and to the correctly rounded
value wherever that is well defined. After those come FP16's range, non-finite
biases, determinism, stream ordering, and argument errors.

The kernel's main loop is nearly flint_gemm_nt's; test_sass.py checks what it
keeps."""

import math

import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator, normal

# Under the sanitizer the GPT-2 shapes shrink.
GPT2_M = 64 if SANITIZER else 1024

# GPT-2 small's linear layers as (n, k): the forward pass's w is n x k. Only the
# MLP's up projection, mlp_up, has a GELU after it; the others are coverage of
# the shapes the kernel will see, and the logits layer is the widest n.
GPT2_SHAPES = {
    "qkv": (2304, 768),
    "attention_projection": (768, 768),
    "mlp_up": (3072, 768),
    "mlp_down": (768, 3072),
    "logits": (50304, 768),
}

# The outputs of a guarded call sit a few halves past the start of buffers that
# are filled with a pattern, which must be intact around them afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A


def shape_id(shape):
    return "x".join(map(str, shape))


# -- Inputs -----------------------------------------------------------------
# Every maker returns x (m x k), w (n x k) and an FP32 bias (n).


def make_bias(n, seed, std=0.5):
    """FP32 values about N(0, 0.5): full 24-bit mantissas, which FP16 can't
    hold."""
    g = generator(30_000 + seed)
    return torch.randn(n, generator=g, device="cuda") * std


def realistic(m, n, k, seed):
    """Activations about N(0, 1), weights about N(0, 0.02), and a bias about
    N(0, 0.5)."""
    g = generator(seed)
    return normal((m, k), 1.0, g), normal((n, k), 0.02, g), make_bias(n, seed)


def outliers(m, n, k, seed):
    """Realistic inputs, with about 0.5% of x's channels (at least 2) near
    +-3,000 in every row, as in GPT-2's activations."""
    x, w, bias = realistic(m, n, k, seed)
    g = generator(10_000 + seed)
    count = max(2, round(0.005 * k))
    channels = torch.randperm(k, generator=g, device="cuda")[:count]
    signs = torch.randint(0, 2, (count,), generator=g, device="cuda") * 2.0 - 1.0
    jitter = 0.9 + 0.2 * torch.rand((m, count), generator=g, device="cuda")
    x[:, channels] = (3000.0 * signs * jitter).half()
    return x, w, bias


INPUTS = {"realistic": realistic, "outliers": outliers}


# -- Running Flint ----------------------------------------------------------


def flint_gemm_bias_gelu(flint, x, w, bias, offset=8, fill=SENTINEL):
    """Runs flint_gemm_nt_bias_gelu into y and pre, each inside a buffer
    pre-filled with `fill` (a 16-bit pattern). y sits `offset` halves into its
    buffer and pre 32 - offset into its own, so offsets of 8, 16 and 24 (which
    keep both 16-byte aligned) give both. Checks the buffers around the outputs
    are untouched, and so is bias, and returns y and pre as m x n FP16 views."""
    m, k = x.shape
    n = w.shape[0]
    pre_offset = 32 - offset
    y_buf = torch.full((offset + m * n + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    pre_buf = torch.full((pre_offset + m * n + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    y = y_buf[offset : offset + m * n].view(torch.float16)
    pre = pre_buf[pre_offset : pre_offset + m * n].view(torch.float16)
    bias_before = bias.clone()
    flint.gemm_nt_bias_gelu(y, pre, x, w, bias, m, n, k)
    torch.cuda.synchronize()
    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    for name, buf, start in (("y", y_buf, offset), ("pre", pre_buf, pre_offset)):
        assert (buf[:start] == guard).all(), f"flint_gemm_nt_bias_gelu wrote before {name}"
        assert (buf[start + m * n :] == guard).all(), f"flint_gemm_nt_bias_gelu wrote past {name}"
    assert torch.equal(bias.view(torch.int32), bias_before.view(torch.int32)), "bias changed"
    return y.view(m, n), pre.view(m, n)


def flint_gemm_plain_bias(flint, x, w, bias):
    """flint_gemm_nt_bias's output, which pre must equal."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), dtype=torch.float16, device="cuda")
    flint.gemm_nt_bias(y, x, w, bias, m, n, k)
    torch.cuda.synchronize()
    return y


def bits(y):
    """y's FP16 bit patterns, which compare equal for equal bits, NaNs and -0.0
    included."""
    return y.contiguous().view(torch.int16)


def gate_triples(flint, make_inputs, m, n, k, offset=8):
    """Per seed, Flint's y, the peer's and the reference's. Also requires pre to
    equal flint_gemm_nt_bias's output, bit for bit: the sums round once, the
    same way."""
    for seed in gate.SEEDS:
        x, w, bias = make_inputs(m, n, k, seed)
        y, pre = flint_gemm_bias_gelu(flint, x, w, bias, offset)
        assert torch.equal(bits(pre), bits(flint_gemm_plain_bias(flint, x, w, bias))), "pre differs"
        yield y, gate.peer_gemm_nt_bias_gelu(x, w, bias), gate.reference_bias_gelu(x, w, bias)


# -- The gate ---------------------------------------------------------------

GPT2_CASES = [
    pytest.param(name, variant, id=f"{name}-{variant}") for name in GPT2_SHAPES for variant in INPUTS
]


@pytest.mark.parametrize("name, variant", GPT2_CASES)
def test_gpt2_shapes_pass_the_gate(flint, name, variant):
    n, k = GPT2_SHAPES[name]
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, INPUTS[variant], GPT2_M, n, k))


# m and n off the 128 tile, k off the 32 step, odd n (rows stored one half at a
# time), and single rows.
ODD_SHAPES = [
    (1, 1, 8),
    (7, 13, 24),
    (127, 129, 136),
    (129, 127, 40),
    (200, 300, 776),
    (333, 7, 8),
    (1, 50304, 768),
    (130, 2, 16),
    # Tiles go out in groups of up to 8 rows of tiles. 5 rows of tiles make
    # a single short group. 9, 13, 17 and 33 end in a short group of 1 or 5
    # rows after full ones, whose spare blocks exit, with one column of
    # tiles or several.
    (637, 300, 24),
    (1152, 1, 8),
    (1614, 200, 32),
    (2116, 129, 40),
    (4224, 7, 16),
]

# Offsets rotate through 8, 16 and 24 halves.
ODD_CASES = [
    pytest.param(shape, variant, 8 * (1 + (i + j) % 3), id=f"{shape_id(shape)}-{variant}")
    for i, shape in enumerate(ODD_SHAPES)
    for j, variant in enumerate(INPUTS)
]


@pytest.mark.parametrize("shape, variant, offset", ODD_CASES)
def test_odd_shapes_pass_the_gate_and_stay_inside_their_outputs(flint, shape, variant, offset):
    # flint_gemm_bias_gelu also checks that y's and pre's guard regions and bias
    # are intact.
    m, n, k = shape
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, INPUTS[variant], m, n, k, offset))


# -- The GELU function ------------------------------------------------------


def gelu64(values):
    """GELU, the tanh approximation, in float64."""
    return torch.nn.functional.gelu(values.double(), approximate="tanh")


def gelu_exponent(values):
    """a in GELU = s / (1 + exp(a)), the form Flint computes: -2 sqrt(2/pi) (s +
    0.044715 s^3), in float64."""
    s = values.double()
    return -2.0 * math.sqrt(2.0 / math.pi) * s * (1.0 + 0.044715 * s * s)


def ordered(h):
    """FP16 values as integers that rise with the value, -0.0 and +0.0 both 0,
    so that neighbours differ by 1."""
    b = bits(h).int()
    return torch.where(b < 0, -(b & 0x7FFF), b)


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


# Of gelu_sweep's points, the share whose exact GELU is far enough from an FP16
# rounding boundary to be checked bit for bit: 96.7% when measured.
SAFE_FRACTION = 0.95


def gelu_sweep():
    """FP32 bias values to run through the GELU: a dense grid over [-12, 12]
    (GELU is 0.5 s (1 + tanh(...)), which is s from about 4 up and nothing from
    about -6 down, so the interesting part is well inside), random values with
    full mantissas over the same range, and extremes: tiny values, values
    around FP16's range, the largest floats, and the large negative values
    where a naive exp overflows. A bias of -0.0 is left out, because 0 + -0.0
    is +0.0."""
    g = generator(7)
    grid = torch.linspace(-12.0, 12.0, 32768, dtype=torch.float64, device="cuda")
    random = (torch.rand(16384, generator=g, device="cuda", dtype=torch.float64) * 24 - 12)
    extremes = []
    for magnitude in (
        1e-40, 1e-30, 2.0**-25, 6e-8, 1e-5, 65504.0, 65519.0, 65520.0, 65536.0, 1e5, 1e6, 1e30,
        3.4028234e38, 20.0, 30.0, 100.0, 1000.0, 1e4, 65000.0,
    ):
        extremes += [magnitude, -magnitude]
    extremes.append(0.0)
    return torch.cat([grid, random, torch.tensor(extremes, dtype=torch.float64, device="cuda")]).float()


def test_gelu_matches_the_correctly_rounded_function(flint):
    # With x = 0 the sums are exactly zero, so s is exactly the bias, and y must
    # be GELU(bias) rounded once to FP16.
    bias = gelu_sweep()
    n = bias.numel()
    m, k = 2, 8
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")

    y, pre = flint_gemm_bias_gelu(flint, x, w, bias)

    # pre is the sums rounded: the bias, rounded.
    assert torch.equal(bits(pre), bits(bias.half().expand(m, n))), "pre is not the rounded bias"
    assert torch.equal(bits(y[0]), bits(y[1])), "the rows differ"

    exact = gelu64(bias).half()
    # Flint's FP32 GELU is s / (1 + exp(a)), with a = -2 sqrt(2/pi) (s +
    # 0.044715 s^3). Computing a in FP32 (a few roundings, and the rounded
    # constants) errs by about 2^-22 of a, which exp turns into an error of
    # about |a| 2^-22 relative in exp(a), and so in y wherever exp(a) dominates
    # the denominator. That peaks at about 4.5e-6 near s = -5.29, where |a| is
    # 19 and y drops below half of FP16's smallest subnormal, so it rounds to -0;
    # in FP16's normal range (s above about -4.03, |a| under 11.1) it stays
    # under 2.7e-6. Elsewhere it is a few FP32 ulps.
    # One ulp of FP16 is 2^-11 relative or more, so y is within one ulp
    # everywhere.
    distance = (ordered(y[0]) - ordered(exact)).abs()
    assert distance.max().item() <= 1, (
        f"y is {distance.max().item()} ulps from the rounded GELU, at bias "
        f"{bias[distance.argmax()].item()!r}: {y[0][distance.argmax()].item()!r} vs "
        f"{exact[distance.argmax()].item()!r}"
    )

    # Where the exact value is farther from a boundary than 2^-21 (2 + |a|) of
    # itself, two to three times that error, every GPU must round it the same
    # way, so y equals the correctly rounded value bit for bit.
    safe = rounding_margin(gelu64(bias)) > 2.0**-21 * (2 + gelu_exponent(bias).abs())
    assert safe.sum() > SAFE_FRACTION * n, "almost every point should be safe"
    wrong = safe & (bits(y[0]) != bits(exact))
    assert not wrong.any(), (
        f"{wrong.sum().item()} safe points are not the correctly rounded value, the first at "
        f"bias {bias[wrong][0].item()!r}: {y[0][wrong][0].item()!r} vs {exact[wrong][0].item()!r}"
    )

    # The check can tell GELU of the FP32 sum (what Flint computes) from GELU of
    # the sum rounded to FP16 first (pre, which a GELU kernel working from the
    # stored pre would use): at the same safe points, the latter rounds
    # differently, so matching the former is evidence.
    from_pre = gelu64(bias.half()).half()
    differs = safe & (bits(from_pre) != bits(exact))
    assert differs.sum() > 1000
    assert (bits(y[0])[differs] != bits(from_pre)[differs]).all()


# -- FP16's range -----------------------------------------------------------


def test_sums_past_fp16_give_inf_in_pre_and_the_gelu_of_the_fp32_sum_in_y(flint):
    # Every sum is +-128,000 plus the bias, finite in FP32 and far past FP16's
    # range. For a positive sum both outputs are +inf. For a negative one pre
    # is -inf, but y is the GELU of the finite sum, which is -0.0, so it is not
    # -inf or NaN (GELU of pre's -inf would be NaN).
    m, n, k = 70, 133, 64
    x = torch.full((m, k), 100.0, dtype=torch.float16, device="cuda")
    signs = torch.where(torch.arange(n, device="cuda") % 2 == 0, 1.0, -1.0)
    w = (signs[:, None] * 20.0).expand(n, k).half().contiguous()
    bias = make_bias(n, 0, std=5.0)

    y, pre = flint_gemm_bias_gelu(flint, x, w, bias)

    positive = (signs > 0)[None, :].expand(m, n)
    assert (y[positive] == math.inf).all() and (pre[positive] == math.inf).all()
    assert (pre[~positive] == -math.inf).all()
    expected = gate.reference_bias_gelu(x, w, bias).half()
    assert torch.equal(bits(y[~positive]), bits(expected[~positive]))
    assert (bits(y[~positive]) == as_int16(0x8000)).all(), "y should be -0.0"


def test_a_bias_past_fp16_reaches_only_its_column(flint):
    m, n, k = 150, 133, 104
    columns = [0, 5, 132]  # the edges and a column in the middle of a tile
    values = [1e6, -70000.0, -1e6]

    def triples():
        for seed in gate.SEEDS:
            x, w, bias = realistic(m, n, k, seed)
            bias[columns] = torch.tensor(values, device="cuda")
            y, pre = flint_gemm_bias_gelu(flint, x, w, bias)
            peer = gate.peer_gemm_nt_bias_gelu(x, w, bias)
            ref = gate.reference_bias_gelu(x, w, bias)

            # +1e6 overflows both outputs. A bias of -70000 or -1e6 overflows pre
            # to -inf, but the FP32 sum is finite and GELU of it is -0.0, which
            # is what y holds, as the reference does. (The peer rounds the bias
            # to -inf first, and GELU of -inf is NaN there.)
            assert (y[:, 0] == math.inf).all() and (pre[:, 0] == math.inf).all()
            for column in columns[1:]:
                assert (pre[:, column] == -math.inf).all(), column
                assert torch.equal(bits(y[:, column]), bits(ref[:, column].half())), column
                assert (bits(y[:, column]) == as_int16(0x8000)).all(), column
                assert torch.isnan(peer[:, column]).all(), column
            rest = [j for j in range(n) if j not in columns]
            yield y[:, rest], peer[:, rest], ref[:, rest]

    gate.gate(gate.FORWARD_FACTOR, triples())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_a_nonfinite_bias_reaches_only_its_column(flint, bad):
    m, n, k = 150, 133, 104

    def triples():
        for seed in gate.SEEDS:
            x, w, bias = realistic(m, n, k, seed)
            column = (seed * 37 + 11) % n
            bias[column] = bad
            y, pre = flint_gemm_bias_gelu(flint, x, w, bias)
            peer = gate.peer_gemm_nt_bias_gelu(x, w, bias)
            ref = gate.reference_bias_gelu(x, w, bias)
            ref_pre = gate.reference_bias(x, w, bias)

            # The same NaNs and infs, with the same signs, as the references,
            # which have the bad column and no other. GELU of -inf is NaN, as in
            # torch's.
            for out, expected in ((y, ref), (pre, ref_pre)):
                assert torch.equal(torch.isnan(out), torch.isnan(expected))
                assert torch.equal(torch.isposinf(out), torch.isposinf(expected))
                assert torch.equal(torch.isneginf(out), torch.isneginf(expected))
            assert torch.isnan(y[:, column]).all() == (bad != math.inf)
            assert torch.equal(torch.isfinite(pre), torch.isfinite(ref_pre))
            finite = torch.isfinite(ref)
            assert (~finite).sum() == m
            assert (~finite[:, column]).all()
            yield y[finite], peer[finite], ref[finite]

    gate.gate(gate.FORWARD_FACTOR, triples())


# -- Determinism ------------------------------------------------------------

DETERMINISM_SHAPES = [(GPT2_M, *GPT2_SHAPES["mlp_up"]), (129, 127, 40)]


@pytest.mark.parametrize("shape", DETERMINISM_SHAPES, ids=shape_id)
def test_run_twice_is_bitwise_identical(flint, shape):
    m, n, k = shape
    x, w, bias = outliers(m, n, k, 0)
    # Different NaN patterns to start from, so a store that is skipped, or one
    # that lands in only one run, shows as a difference or a leftover NaN.
    first = flint_gemm_bias_gelu(flint, x, w, bias, fill=0xFFFF)
    second = flint_gemm_bias_gelu(flint, x, w, bias, fill=0x7F7F)

    for a, b in zip(first, second):
        assert not torch.isnan(a).any()
        assert not torch.isnan(b).any()
        assert torch.equal(bits(a), bits(b))


# -- Streams ----------------------------------------------------------------


def test_gemm_runs_on_the_given_stream(flint):
    m = n = k = 64
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")
    bias = torch.zeros(n, dtype=torch.float32, device="cuda")
    y = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    pre = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fills finish only after the sleep, so the result is k + 0.5 only if
    # the GEMM is queued on the same stream behind them. GELU of 64.5 is itself:
    # the exponential underflows to zero.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(1)
        bias.fill_(0.5)
        flint.gemm_nt_bias_gelu(y, pre, x, w, bias, m, n, k, stream=stream)
    stream.synchronize()

    assert (y == k + 0.5).all()
    assert (pre == k + 0.5).all()


# -- The size limits --------------------------------------------------------


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
def test_largest_n_passes_the_gate(flint):
    # n at the grid's limit, 65,535 tiles of 128 (with 2 rows; a full row of
    # tiles at that n would need a 2 GiB y), and a 32 MiB bias.
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, realistic, 2, 8_388_480, 8))


# -- Argument errors --------------------------------------------------------


def small_problem(m=16, n=16, k=16):
    """Valid tensors, in buffers with room to spare."""
    y = torch.zeros(m * n + 64, dtype=torch.float16, device="cuda")
    pre = torch.zeros(m * n + 64, dtype=torch.float16, device="cuda")
    x = torch.ones(m * k + 64, dtype=torch.float16, device="cuda")
    w = torch.ones(n * k + 64, dtype=torch.float16, device="cuda")
    bias = torch.zeros(n + 64, dtype=torch.float32, device="cuda")
    return y, pre, x, w, bias


def expect_invalid_argument(flint, reason, y, pre, x, w, bias, m, n, k):
    with pytest.raises(flint_lib.FlintError) as error:
        flint.gemm_nt_bias_gelu(y, pre, x, w, bias, m, n, k)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_gemm_nt_bias_gelu", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("name", ["y", "pre", "x", "w", "bias"])
def test_null_pointer(flint, name):
    y, pre, x, w, bias = small_problem()
    pointers = {"y": y, "pre": pre, "x": x, "w": w, "bias": bias}
    pointers[name] = None
    expect_invalid_argument(
        flint, f"{name} is null", pointers["y"], pointers["pre"], pointers["x"], pointers["w"],
        pointers["bias"], 16, 16, 16
    )


@pytest.mark.parametrize("k", [0, 12])
def test_k_must_be_a_positive_multiple_of_8(flint, k):
    y, pre, x, w, bias = small_problem()
    expect_invalid_argument(
        flint, f"k ({k}) must be a positive multiple of 8", y, pre, x, w, bias, 16, 16, k
    )


@pytest.mark.parametrize("name", ["y", "pre", "x", "w", "bias"])
def test_misaligned_pointer(flint, name):
    # bias is FP32, but a multiple of 4 bytes short of 16-byte alignment is
    # still misaligned.
    y, pre, x, w, bias = small_problem()
    pointers = {
        "y": y.data_ptr(), "pre": pre.data_ptr(), "x": x.data_ptr(), "w": w.data_ptr(),
        "bias": bias.data_ptr(),
    }
    pointers[name] += 2 if name != "bias" else 4
    expect_invalid_argument(
        flint, f"{name} is not 16-byte aligned", pointers["y"], pointers["pre"], pointers["x"],
        pointers["w"], pointers["bias"], 16, 16, 16
    )


@pytest.mark.parametrize("name", ["bias", "pre"])
def test_buffer_may_not_run_past_the_address_space(flint, name):
    # The check runs before any memory is touched, so a made-up address will do.
    y, pre, x, w, bias = small_problem()
    pointers = {"pre": pre, "bias": bias}
    pointers[name] = 2**64 - 16
    expect_invalid_argument(
        flint, f"{name} runs past the end of the address space", y, pointers["pre"], x, w,
        pointers["bias"], 16, 16, 16
    )


# The size checks run before any memory is touched, so small tensors do.
TOO_LARGE_CASES = [
    pytest.param(shape, id=name)
    for name, shape in [
        ("m", (8_388_481, 8, 8)),
        ("n", (8, 8_388_481, 8)),
        ("k", (8, 8, 2**16 + 8)),
    ]
]


@pytest.mark.parametrize("shape", TOO_LARGE_CASES)
def test_too_large_dimensions(flint, shape):
    m, n, k = shape
    y, pre, x, w, bias = small_problem()
    expect_invalid_argument(flint, "too large", y, pre, x, w, bias, m, n, k)


def test_y_overlapping_x(flint):
    _, pre, _, _, bias = small_problem()
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # x starts 256 bytes into y, and, in the second call, 16 bytes before y's end.
    expect_invalid_argument(flint, "y and x overlap", buf, pre, buf[128:], buf[2048:], bias, 16, 16, 16)
    expect_invalid_argument(flint, "y and x overlap", buf, pre, buf[248:], buf[2048:], bias, 16, 16, 16)


def test_y_overlapping_w(flint):
    _, pre, _, _, bias = small_problem()
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    expect_invalid_argument(flint, "y and w overlap", buf, pre, buf[2048:], buf[128:], bias, 16, 16, 16)
    expect_invalid_argument(flint, "y and w overlap", buf, pre, buf[2048:], buf[248:], bias, 16, 16, 16)


def test_y_overlapping_bias(flint):
    pre, x, w = small_problem()[1:4]
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # y is 512 bytes. bias starts 256 bytes into it and, in the second call,
    # 16 bytes before its end; in the third, y starts 16 bytes before the end
    # of bias (64 bytes), which sits earlier in the buffer.
    expect_invalid_argument(flint, "y and bias overlap", buf, pre, x, w, buf[128:].data_ptr(), 16, 16, 16)
    expect_invalid_argument(flint, "y and bias overlap", buf, pre, x, w, buf[248:].data_ptr(), 16, 16, 16)
    expect_invalid_argument(
        flint, "y and bias overlap", buf[2048:], pre, x, w, buf[2048 - 24 :].data_ptr(), 16, 16, 16
    )


def test_y_overlapping_pre(flint):
    x, w, bias = small_problem()[2:]
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # Both are 512 bytes. pre starts 256 bytes into y, then 16 bytes before y's
    # end, and then is y itself; and the other way round, y starts inside pre.
    expect_invalid_argument(flint, "y and pre overlap", buf, buf[128:], x, w, bias, 16, 16, 16)
    expect_invalid_argument(flint, "y and pre overlap", buf, buf[248:], x, w, bias, 16, 16, 16)
    expect_invalid_argument(flint, "y and pre overlap", buf, buf, x, w, bias, 16, 16, 16)
    expect_invalid_argument(flint, "y and pre overlap", buf[128:], buf, x, w, bias, 16, 16, 16)
    expect_invalid_argument(flint, "y and pre overlap", buf[248:], buf, x, w, bias, 16, 16, 16)


def test_pre_overlapping_x(flint):
    y, _, _, w, bias = small_problem()
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # x and pre are 512 bytes each: pre starts 256 bytes into x and, in the
    # second call, 16 bytes before its end; in the other two, x starts inside
    # pre in the same ways.
    expect_invalid_argument(flint, "pre and x overlap", y, buf[128:], buf, w, bias, 16, 16, 16)
    expect_invalid_argument(flint, "pre and x overlap", y, buf[248:], buf, w, bias, 16, 16, 16)
    expect_invalid_argument(flint, "pre and x overlap", y, buf, buf[128:], w, bias, 16, 16, 16)
    expect_invalid_argument(flint, "pre and x overlap", y, buf, buf[248:], w, bias, 16, 16, 16)


def test_pre_overlapping_w(flint):
    y, _, x, _, bias = small_problem()
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    expect_invalid_argument(flint, "pre and w overlap", y, buf[128:], x, buf, bias, 16, 16, 16)
    expect_invalid_argument(flint, "pre and w overlap", y, buf[248:], x, buf, bias, 16, 16, 16)
    expect_invalid_argument(flint, "pre and w overlap", y, buf, x, buf[128:], bias, 16, 16, 16)
    expect_invalid_argument(flint, "pre and w overlap", y, buf, x, buf[248:], bias, 16, 16, 16)


def test_pre_overlapping_bias(flint):
    y, _, x, w, _ = small_problem()
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # pre is 512 bytes, bias 64. bias starts 256 bytes into pre and, in the
    # second call, 16 bytes before its end; in the third, bias starts 48 bytes
    # before pre, so its last 16 bytes are inside it; in the fourth, pre starts
    # 16 bytes before the end of bias.
    expect_invalid_argument(flint, "pre and bias overlap", y, buf, x, w, buf[128:].data_ptr(), 16, 16, 16)
    expect_invalid_argument(flint, "pre and bias overlap", y, buf, x, w, buf[248:].data_ptr(), 16, 16, 16)
    expect_invalid_argument(
        flint, "pre and bias overlap", y, buf[2048:], x, w, buf[2048 - 24 :].data_ptr(), 16, 16, 16
    )
    expect_invalid_argument(
        flint, "pre and bias overlap", y, buf[2048 - 24 :], x, w, buf[2048 - 48 :].data_ptr(), 16, 16, 16
    )


def sequential_bias(n):
    """0.25, 0.75, 1.25, ...: exact in FP16 once added to 16."""
    return 0.25 + 0.5 * torch.arange(n, dtype=torch.float32, device="cuda")


@pytest.mark.parametrize("pre_first", [True, False], ids=["pre-then-y", "y-then-pre"])
def test_pre_next_to_y_is_not_overlapping(flint, pre_first):
    # y and pre (512 bytes each) side by side in one buffer, with 16 bytes of
    # pattern around them that must stay intact. The sums are 16.25, 16.75, ...:
    # exact in FP16, and GELU of them is themselves (the exponential underflows),
    # so both outputs are the sums.
    m = n = k = 16
    buf = torch.full((16 + 1024 + 16,), 0x5A, dtype=torch.uint8, device="cuda")
    pre_at, y_at = (16, 16 + 512) if pre_first else (16 + 512, 16)
    pre = buf[pre_at : pre_at + 2 * m * n].view(torch.float16)
    y = buf[y_at : y_at + 2 * m * n].view(torch.float16)
    bias = sequential_bias(n)
    x = torch.ones((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")

    flint.gemm_nt_bias_gelu(y, pre, x, w, bias, m, n, k)
    torch.cuda.synchronize()

    expected = (16 + sequential_bias(n)).half().expand(m, n)
    assert torch.equal(y.view(m, n), expected)
    assert torch.equal(pre.view(m, n), expected)
    assert torch.equal(bias, sequential_bias(n))
    assert (buf[:16] == 0x5A).all() and (buf[16 + 1024 :] == 0x5A).all()


def test_bias_may_overlap_x(flint):
    # bias (FP32) and x (FP16) are views of the same bytes, as is allowed since
    # both are only read. The floats are bfloat16 values, whose low halves are
    # zero, so that read as FP16 pairs they are finite: 0.0 and a number.
    m = k = n = 16

    def triples():
        for seed in gate.SEEDS:
            g = generator(seed)
            shared = (torch.randn(128, generator=g, device="cuda")).bfloat16().float()
            x = shared.view(torch.float16)[: m * k].view(m, k)
            bias = shared[8 : 8 + n]  # 32 bytes in: 16-byte aligned
            assert torch.isfinite(x).all()
            w = normal((n, k), 0.5, g)
            x_copy, bias_copy = x.clone(), bias.clone()
            y, _ = flint_gemm_bias_gelu(flint, x, w, bias)
            yield (
                y,
                gate.peer_gemm_nt_bias_gelu(x_copy, w, bias_copy),
                gate.reference_bias_gelu(x_copy, w, bias_copy),
            )

    gate.gate(gate.FORWARD_FACTOR, triples())


@pytest.mark.parametrize("m, n", [(0, 16), (16, 0), (0, 0)])
def test_empty_output_does_nothing(flint, m, n):
    # Not even the null bias or pre is looked at.
    y, pre, x, w, bias = small_problem()
    y.fill_(3)
    pre.fill_(4)

    flint.gemm_nt_bias_gelu(y, pre, x, w, None, m, n, 16)
    flint.gemm_nt_bias_gelu(y, None, x, w, bias, m, n, 16)
    flint.gemm_nt_bias_gelu(None, None, None, None, None, m, n, 16)
    flint.gemm_nt_bias_gelu(y, pre, x, w, bias, m, n, 16)
    torch.cuda.synchronize()

    assert (y == 3).all()
    assert (pre == 4).all()


def test_a_valid_call_works_after_errors(flint):
    y, pre, x, w, bias = small_problem()
    bias[:16] = 0.5
    for k in (0, 12):
        with pytest.raises(flint_lib.FlintError):
            flint.gemm_nt_bias_gelu(y, pre, x, w, bias, 16, 16, k)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias_gelu(None, pre, x, w, bias, 16, 16, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias_gelu(y, None, x, w, bias, 16, 16, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias_gelu(y, pre, x, w, None, 16, 16, 16)

    flint.gemm_nt_bias_gelu(y, pre, x, w, bias, 16, 16, 16)
    torch.cuda.synchronize()

    # The sums are 16.5; GELU of that is itself.
    assert (y[:256] == 16.5).all() and (pre[:256] == 16.5).all()
    assert (y[256:] == 0).all() and (pre[256:] == 0).all()
