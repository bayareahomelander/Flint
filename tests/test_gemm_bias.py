"""flint_gemm_nt_bias: y = x w^T + bias, a linear layer's forward pass with its
bias, where y is FP16 and bias holds n FP32 values, one per column of y. The
bias joins the FP32 sums before the single round to FP16.

The tests cover the correctness gate (the peer is PyTorch's FP16 linear layer,
which rounds the bias to FP16 first, so it is the coarser of the two), then
exact checks that need no tolerance: a bias of -0.0 changes nothing, a zero
input gives back the bias column by column, and small integer sums leave the
FP32 add and the one rounding as the only error. After those come FP16's
range, non-finite biases, determinism, stream ordering, and argument errors.

The kernel's main loop is flint_gemm_nt's, encoding for encoding, which
test_sass.py checks; the bias is read after the loop."""

import math

import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, clamp_half, generator, normal

# Under the sanitizer the GPT-2 shapes shrink.
GPT2_M = 64 if SANITIZER else 1024

# GPT-2 small's linear layers as (n, k): the forward pass's w is n x k. The
# logits layer has no bias in the model, but its shape is the widest n.
GPT2_SHAPES = {
    "qkv": (2304, 768),
    "attention_projection": (768, 768),
    "mlp_up": (3072, 768),
    "mlp_down": (768, 3072),
    "logits": (50304, 768),
}

# The output of a guarded call sits `offset` halves past the start of a buffer
# that is filled with a pattern, which must be intact around y afterwards.
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


def flint_gemm_bias(flint, x, w, bias, offset=8, fill=SENTINEL):
    """Runs flint_gemm_nt_bias into y inside a buffer pre-filled with `fill` (a
    16-bit pattern), offset halves in. A multiple of 8 keeps y 16-byte
    aligned. Checks the buffer around y is untouched, and so is bias, and
    returns y as an m x n FP16 view."""
    m, k = x.shape
    n = w.shape[0]
    buf = torch.full((offset + m * n + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    y = buf[offset : offset + m * n].view(torch.float16)
    bias_before = bias.clone()
    flint.gemm_nt_bias(y, x, w, bias, m, n, k)
    torch.cuda.synchronize()
    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert (buf[:offset] == guard).all(), "flint_gemm_nt_bias wrote before y"
    assert (buf[offset + m * n :] == guard).all(), "flint_gemm_nt_bias wrote past y"
    assert torch.equal(bias.view(torch.int32), bias_before.view(torch.int32)), "bias changed"
    return y.view(m, n)


def flint_gemm_plain(flint, x, w):
    """flint_gemm_nt's output, to compare with the biased kernel's."""
    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), dtype=torch.float16, device="cuda")
    flint.gemm_nt(y, x, w, m, n, k)
    torch.cuda.synchronize()
    return y


def gate_triples(flint, make_inputs, m, n, k, offset=8):
    """Per seed, Flint's output, the peer's and the reference's."""
    for seed in gate.SEEDS:
        x, w, bias = make_inputs(m, n, k, seed)
        yield (
            flint_gemm_bias(flint, x, w, bias, offset),
            gate.peer_gemm_nt_bias(x, w, bias),
            gate.reference_bias(x, w, bias),
        )


def bits(y):
    """y's FP16 bit patterns, which compare equal for equal bits, NaNs and -0.0
    included."""
    return y.contiguous().view(torch.int16)


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
def test_odd_shapes_pass_the_gate_and_stay_inside_y(flint, shape, variant, offset):
    # flint_gemm_bias also checks that y's guard regions and bias are intact.
    m, n, k = shape
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, INPUTS[variant], m, n, k, offset))


# -- Exact checks -----------------------------------------------------------

IDENTITY_SHAPES = [
    (GPT2_M, *GPT2_SHAPES["qkv"]),
    (7, 13, 24),
    (129, 127, 40),
    (200, 300, 776),
    (1, 50304, 768),
]


@pytest.mark.parametrize("shape", IDENTITY_SHAPES, ids=shape_id)
def test_a_bias_of_negative_zero_changes_nothing(flint, shape):
    # s + (-0.0) is s for every s, -0.0 included, so the output must equal
    # flint_gemm_nt's bit for bit.
    m, n, k = shape
    for seed in gate.SEEDS:
        x, w, _ = outliers(m, n, k, seed)
        bias = torch.full((n,), -0.0, dtype=torch.float32, device="cuda")
        assert torch.equal(bits(flint_gemm_bias(flint, x, w, bias)), bits(flint_gemm_plain(flint, x, w)))


COLUMN_SHAPES = [
    (1, 1, 8),
    (7, 13, 24),
    (129, 127, 40),
    (200, 300, 776),
    (130, 2, 16),
    (1152, 1, 8),
    (333, 7, 8),
]


@pytest.mark.parametrize("shape", COLUMN_SHAPES, ids=shape_id)
def test_each_column_gets_its_own_bias(flint, shape):
    # With x all +0.0 the sums are zero, so every row of y is the bias rounded
    # to FP16 (to nearest even, as torch's .half() does). The values are
    # distinct, nonzero and not FP16-representable, so a column that gets
    # another's bias, or none, shows as a difference. (Rounding the bias early
    # would give the same output here; the exact-sum tests below catch that.)
    m, n, k = shape
    g = generator(0)
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = normal((n, k), 1.0, g)
    bias = 0.123 + 0.37 * torch.arange(n, dtype=torch.float32, device="cuda")
    assert bias.max() < 65504 and (bias.half().float() != bias).any()

    y = flint_gemm_bias(flint, x, w, bias)

    assert torch.equal(bits(y), bits(bias.half().expand(m, n)))


# x and w are small integers and k is 64, so every sum is an exact integer
# (at most 2^8 in magnitude), which the tensor cores' truncating accumulation can't lose
# anything from. The bias has full mantissas.
EXACT_SUM_SHAPES = [(64, 64), (7, 13), (129, 127), (200, 300), (130, 2), (1, 50304), (333, 7)]


@pytest.mark.parametrize("shape", EXACT_SUM_SHAPES, ids=shape_id)
def test_the_bias_adds_in_fp32_before_the_one_rounding(flint, shape):
    m, n = shape
    k = 64
    for seed in gate.SEEDS:
        g = generator(seed)
        x = torch.randint(-2, 3, (m, k), generator=g, device="cuda").half()
        w = torch.randint(-2, 3, (n, k), generator=g, device="cuda").half()
        bias = make_bias(n, seed, std=4.0)

        with gate.peer_math():
            sums = x.float() @ w.float().T
        expected = (sums + bias).half()
        y = flint_gemm_bias(flint, x, w, bias)

        assert torch.equal(bits(y), bits(expected))
        if m * n >= 10_000:
            # The check can tell the order apart: rounding the bias to FP16
            # first would change some outputs.
            early = (sums + bias.half().float()).half()
            assert (early != expected).any()


def test_the_bias_is_not_rounded_to_fp16_first(flint):
    # The sum is exactly 1.0 and the bias 2^-11 (1 + 2^-12). In FP32 the total,
    # 1 + 2^-11 + 2^-23, is just above the midpoint of 1 and 1 + 2^-10, FP16's
    # neighbours, and rounds up to 1 + 2^-10. The bias in FP16 is 2^-11, where
    # the total is the midpoint itself, which ties to even: 1.0.
    x = torch.zeros((1, 8), dtype=torch.float16, device="cuda")
    w = torch.zeros((1, 8), dtype=torch.float16, device="cuda")
    x[0, 0] = w[0, 0] = 1
    bias = torch.tensor([2.0**-11 * (1 + 2.0**-12)], dtype=torch.float32, device="cuda")
    assert (1.0 + bias.half()).item() == 1.0

    y = flint_gemm_bias(flint, x, w, bias)

    assert y.item() == 1 + 2.0**-10


# -- FP16's range -----------------------------------------------------------


def test_overflow_gives_inf_with_the_references_sign(flint):
    m, n, k = 150, 130, 64
    # The sums' std is about 40,000 and the bias's 20,000 (kept inside FP16, so
    # the peer's FP16 bias is finite), so roughly one output in seven is past
    # FP16's range. Outputs between the two thresholds are within an FP32
    # accumulation error of the rounding boundary 65,520, so either answer is
    # right for them.
    inf_threshold = 65520.0 * (1 + 2.0**-8)
    finite_threshold = 65504.0 * (1 - 2.0**-8)

    def triples():
        for seed in gate.SEEDS:
            g = generator(seed)
            x = normal((m, k), 150.0, g)
            w = normal((n, k), 33.0, g)
            bias = clamp_half(torch.randn(n, generator=g, device="cuda") * 20000, 65504).float()
            ref = gate.reference_bias(x, w, bias)
            y = flint_gemm_bias(flint, x, w, bias)
            peer = gate.peer_gemm_nt_bias(x, w, bias)

            assert not torch.isnan(y).any(), "overflow produced NaN"
            too_big = ref.abs() >= inf_threshold
            fits = ref.abs() <= finite_threshold
            # Each class and sign has to occur, or the test checks nothing.
            assert (too_big & (ref > 0)).any() and (too_big & (ref < 0)).any()
            assert fits.sum() > 0.5 * ref.numel()
            expected = torch.where(ref > 0, math.inf, -math.inf).half()
            assert torch.equal(y[too_big], expected[too_big]), "wrong overflow to inf"
            assert torch.isfinite(y[fits]).all(), "finite output became inf"
            yield y[fits], peer[fits], ref[fits]

    gate.gate(gate.FORWARD_FACTOR, triples())


def test_a_bias_past_fp16_gives_inf_in_its_column(flint):
    m, n, k = 150, 133, 104
    columns = [0, 5, 132]  # the edges and a column in the middle of a tile

    def triples():
        for seed in gate.SEEDS:
            x, w, bias = realistic(m, n, k, seed)
            bias[columns] = torch.tensor([1e6, -1e6, 70000.0], device="cuda")
            y = flint_gemm_bias(flint, x, w, bias)
            peer = gate.peer_gemm_nt_bias(x, w, bias)
            ref = gate.reference_bias(x, w, bias)

            for column in columns:
                expected = math.copysign(math.inf, bias[column].item())
                assert (y[:, column] == expected).all(), column
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
            y = flint_gemm_bias(flint, x, w, bias)
            peer = gate.peer_gemm_nt_bias(x, w, bias)
            ref = gate.reference_bias(x, w, bias)

            # The same NaNs and infs, with the same signs, as the reference,
            # which has the bad column and no other.
            assert torch.equal(torch.isnan(y), torch.isnan(ref))
            assert torch.equal(torch.isposinf(y), torch.isposinf(ref))
            assert torch.equal(torch.isneginf(y), torch.isneginf(ref))
            finite = torch.isfinite(ref)
            assert (~finite).sum() == m
            assert (~finite[:, column]).all()
            yield y[finite], peer[finite], ref[finite]

    gate.gate(gate.FORWARD_FACTOR, triples())


# -- Determinism ------------------------------------------------------------

DETERMINISM_SHAPES = [(GPT2_M, *GPT2_SHAPES["qkv"]), (129, 127, 40)]


@pytest.mark.parametrize("shape", DETERMINISM_SHAPES, ids=shape_id)
def test_run_twice_is_bitwise_identical(flint, shape):
    m, n, k = shape
    x, w, bias = outliers(m, n, k, 0)
    # Different NaN patterns to start from, so a store that is skipped, or one
    # that lands in only one run, shows as a difference or a leftover NaN.
    first = flint_gemm_bias(flint, x, w, bias, fill=0xFFFF)
    second = flint_gemm_bias(flint, x, w, bias, fill=0x7F7F)

    assert not torch.isnan(first).any()
    assert not torch.isnan(second).any()
    assert torch.equal(bits(first), bits(second))


# -- Streams ----------------------------------------------------------------


def test_gemm_runs_on_the_given_stream(flint):
    m = n = k = 64
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")
    bias = torch.zeros(n, dtype=torch.float32, device="cuda")
    y = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fills finish only after the sleep, so the result is k + 0.5 only if
    # the GEMM is queued on the same stream behind them.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(1)
        bias.fill_(0.5)
        flint.gemm_nt_bias(y, x, w, bias, m, n, k, stream=stream)
    stream.synchronize()

    assert (y == k + 0.5).all()


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
    x = torch.ones(m * k + 64, dtype=torch.float16, device="cuda")
    w = torch.ones(n * k + 64, dtype=torch.float16, device="cuda")
    bias = torch.zeros(n + 64, dtype=torch.float32, device="cuda")
    return y, x, w, bias


def expect_invalid_argument(flint, reason, y, x, w, bias, m, n, k):
    with pytest.raises(flint_lib.FlintError) as error:
        flint.gemm_nt_bias(y, x, w, bias, m, n, k)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_gemm_nt_bias", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("name", ["y", "x", "w", "bias"])
def test_null_pointer(flint, name):
    y, x, w, bias = small_problem()
    pointers = {"y": y, "x": x, "w": w, "bias": bias}
    pointers[name] = None
    expect_invalid_argument(
        flint, f"{name} is null", pointers["y"], pointers["x"], pointers["w"], pointers["bias"], 16, 16, 16
    )


@pytest.mark.parametrize("k", [0, 12])
def test_k_must_be_a_positive_multiple_of_8(flint, k):
    y, x, w, bias = small_problem()
    expect_invalid_argument(flint, f"k ({k}) must be a positive multiple of 8", y, x, w, bias, 16, 16, k)


@pytest.mark.parametrize("name", ["y", "x", "w", "bias"])
def test_misaligned_pointer(flint, name):
    # bias is FP32, but a multiple of 4 bytes short of 16-byte alignment is
    # still misaligned.
    y, x, w, bias = small_problem()
    pointers = {"y": y.data_ptr(), "x": x.data_ptr(), "w": w.data_ptr(), "bias": bias.data_ptr()}
    pointers[name] += 2 if name != "bias" else 4
    expect_invalid_argument(
        flint, f"{name} is not 16-byte aligned", pointers["y"], pointers["x"], pointers["w"],
        pointers["bias"], 16, 16, 16
    )


def test_bias_may_not_run_past_the_address_space(flint):
    # The check runs before any memory is touched, so a made-up address will do.
    y, x, w, _ = small_problem()
    expect_invalid_argument(
        flint, "bias runs past the end of the address space", y, x, w, 2**64 - 16, 16, 16, 16
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
    y, x, w, bias = small_problem()
    expect_invalid_argument(flint, "too large", y, x, w, bias, m, n, k)


def test_y_overlapping_x(flint):
    bias = small_problem()[3]
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # x starts 256 bytes into y, and, in the second call, 16 bytes before y's end.
    expect_invalid_argument(flint, "y and x overlap", buf, buf[128:], buf[2048:], bias, 16, 16, 16)
    expect_invalid_argument(flint, "y and x overlap", buf, buf[248:], buf[2048:], bias, 16, 16, 16)


def test_y_overlapping_w(flint):
    bias = small_problem()[3]
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    expect_invalid_argument(flint, "y and w overlap", buf, buf[2048:], buf[128:], bias, 16, 16, 16)
    expect_invalid_argument(flint, "y and w overlap", buf, buf[2048:], buf[248:], bias, 16, 16, 16)


def test_y_overlapping_bias(flint):
    x, w = small_problem()[1:3]
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # y is 512 bytes. bias starts 256 bytes into it and, in the second call,
    # 16 bytes before its end; in the third, y starts 16 bytes before the end
    # of bias (64 bytes), which sits earlier in the buffer.
    expect_invalid_argument(flint, "y and bias overlap", buf, x, w, buf[128:].data_ptr(), 16, 16, 16)
    expect_invalid_argument(flint, "y and bias overlap", buf, x, w, buf[248:].data_ptr(), 16, 16, 16)
    expect_invalid_argument(
        flint, "y and bias overlap", buf[2048:], x, w, buf[2048 - 24 :].data_ptr(), 16, 16, 16
    )


def sequential_bias(n):
    """0.25, 0.75, 1.25, ...: exact in FP16 once added to 16."""
    return 0.25 + 0.5 * torch.arange(n, dtype=torch.float32, device="cuda")


@pytest.mark.parametrize("bias_first", [True, False], ids=["bias-then-y", "y-then-bias"])
def test_bias_next_to_y_is_not_overlapping(flint, bias_first):
    # bias (64 bytes) and y (512 bytes) side by side in one buffer, with a
    # pattern around them that must stay intact.
    m = n = k = 16
    buf = torch.full((1024,), 0x5A, dtype=torch.uint8, device="cuda")
    bias_at, y_at = (0, 64) if bias_first else (512, 0)
    bias = buf[bias_at : bias_at + 4 * n].view(torch.float32)
    bias.copy_(sequential_bias(n))
    y = buf[y_at : y_at + 2 * m * n].view(torch.float16)
    x = torch.ones((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")

    flint.gemm_nt_bias(y, x, w, bias, m, n, k)
    torch.cuda.synchronize()

    assert torch.equal(y.view(m, n), (16 + sequential_bias(n)).half().expand(m, n))
    assert torch.equal(bias, sequential_bias(n))
    used = torch.zeros(buf.numel(), dtype=torch.bool, device="cuda")
    used[bias_at : bias_at + 4 * n] = True
    used[y_at : y_at + 2 * m * n] = True
    assert (buf[~used] == 0x5A).all()


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
            yield (
                flint_gemm_bias(flint, x, w, bias),
                gate.peer_gemm_nt_bias(x_copy, w, bias_copy),
                gate.reference_bias(x_copy, w, bias_copy),
            )

    gate.gate(gate.FORWARD_FACTOR, triples())


@pytest.mark.parametrize("m, n", [(0, 16), (16, 0), (0, 0)])
def test_empty_output_does_nothing(flint, m, n):
    # Not even the null bias is looked at.
    y, x, w, bias = small_problem()
    y.fill_(3)

    flint.gemm_nt_bias(y, x, w, None, m, n, 16)
    flint.gemm_nt_bias(None, None, None, None, m, n, 16)
    flint.gemm_nt_bias(y, x, w, bias, m, n, 16)
    torch.cuda.synchronize()

    assert (y == 3).all()


def test_a_valid_call_works_after_errors(flint):
    y, x, w, bias = small_problem()
    bias[:16] = 0.5
    for k in (0, 12):
        with pytest.raises(flint_lib.FlintError):
            flint.gemm_nt_bias(y, x, w, bias, 16, 16, k)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias(None, x, w, bias, 16, 16, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_nt_bias(y, x, w, None, 16, 16, 16)

    flint.gemm_nt_bias(y, x, w, bias, 16, 16, 16)
    torch.cuda.synchronize()

    assert (y[:256] == 16.5).all()
    assert (y[256:] == 0).all()
