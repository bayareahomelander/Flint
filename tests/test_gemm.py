"""flint_gemm_nt and flint_gemm_nn: the correctness gate on GPT-2 and edge
shapes, FP16 range, non-finite inputs, determinism, stream ordering, and
argument errors. Every test runs for both layouts (test ids start with nt- or
nn-).

NT is y = x w^T with w n x k (a linear layer's forward pass). NN is y = x w with
w k x n (its input gradient, dX = dY W), which also needs n to be a multiple of
8. The input makers below build w in NT's n x k form, and a layout turns it
into the form its kernel takes."""

import math
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, clamp_half, generator, normal

# Under the sanitizer the GPT-2 shapes shrink.
GPT2_M = 64 if SANITIZER else 1024

# GPT-2 small's linear layers as (n, k): the forward pass's w is n x k.
GPT2_SHAPES = {
    "qkv": (2304, 768),
    "attention_projection": (768, 768),
    "mlp_up": (3072, 768),
    "mlp_down": (768, 3072),
    "logits": (50304, 768),
}
# The same layers' input gradients, dX = dY W, as (n, k): w is k x n, and the
# forward pass's n and k swap.
GPT2_GRAD_SHAPES = {
    "qkv": (768, 2304),
    "attention_projection": (768, 768),
    "mlp_up": (768, 3072),
    "mlp_down": (3072, 768),
    "logits": (768, 50304),
}

# The output of a guarded call sits `offset` halves past the start of a buffer
# that is filled with a pattern, which must be intact around y afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A


# -- The two layouts --------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """What differs between flint_gemm_nt and flint_gemm_nn."""

    name: str  # "nt" or "nn"
    peer: Callable  # PyTorch's FP16 path
    reference: Callable  # the float reference
    n_multiple: int  # n must be a multiple of this
    gpt2_shapes: dict  # name -> (n, k)
    odd_shapes: list  # (m, n, k)
    limit_shapes: list  # (id, (m, n, k)), each dimension at its edge
    exact_k_shape: tuple  # (m, n) for the k = 2^16 integer test
    alias_shape: tuple  # (size, k) of a, with x = w = a
    determinism_shape: tuple  # (m, n, k), odd
    too_large: list  # (id, (m, n, k)) just past each limit

    @property
    def flint_name(self):
        return f"flint_gemm_{self.name}"

    def call(self, flint, y, x, w, m, n, k, stream=None):
        getattr(flint, f"gemm_{self.name}")(y, x, w, m, n, k, stream=stream)

    def fit(self, n):
        """n rounded up to a multiple of what the layout needs."""
        return -(-n // self.n_multiple) * self.n_multiple

    def to_w(self, w):
        """An n x k tensor in the layout's memory form: as is, or k x n."""
        return w if self.name == "nt" else w.T.contiguous()

    def n_of(self, w):
        return w.shape[0] if self.name == "nt" else w.shape[1]

    def set_w(self, w, n_index, k_index, value):
        """Sets the element of w that pairs output column n_index with x's
        column k_index."""
        if self.name == "nt":
            w[n_index, k_index] = value
        else:
            w[k_index, n_index] = value


NT = Layout(
    name="nt",
    peer=gate.peer_gemm_nt,
    reference=gate.reference,
    n_multiple=1,
    gpt2_shapes=GPT2_SHAPES,
    # m and n off the 128 tile, k off the 32 step, odd n (rows stored one half
    # at a time), and single rows.
    odd_shapes=[
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
    ],
    # Each limit at its edge: m and n at 65,535 tiles of 128, n's being the
    # grid's limit (with 2 rows; a full row of tiles at that n would need a 2
    # GiB y), and k where the error, which grows with k as the tensor cores'
    # FP32 accumulation truncates, is closest to the gate.
    limit_shapes=[
        ("max_m", (8_388_480, 1, 8)),
        ("max_n", (2, 8_388_480, 8)),
        ("max_k", (64, 64, 2**16)),
    ],
    exact_k_shape=(2, 2),
    alias_shape=(200, 136),
    determinism_shape=(129, 127, 40),
    too_large=[
        ("m", (8_388_481, 8, 8)),
        ("n", (8, 8_388_481, 8)),
        ("k", (8, 8, 2**16 + 8)),
    ],
)

NN = Layout(
    name="nn",
    peer=gate.peer_gemm_nn,
    reference=gate.reference_nn,
    n_multiple=8,
    gpt2_shapes=GPT2_GRAD_SHAPES,
    # n and k multiples of 8 but not of 128 or 32, and odd m.
    odd_shapes=[
        (1, 8, 8),
        (7, 16, 24),
        (127, 136, 40),
        (129, 120, 136),
        (200, 776, 776),
        (333, 8, 8),
        (1, 768, 50304),
        (130, 16, 16),
        # NT's group edges, with n rounded to a multiple of 8.
        (637, 296, 24),
        (1152, 8, 8),
        (1614, 200, 32),
        (2116, 136, 40),
        (4224, 8, 16),
    ],
    limit_shapes=[
        ("max_m", (8_388_480, 8, 8)),
        ("max_n", (2, 8_388_480, 8)),
        ("max_k", (64, 64, 2**16)),
    ],
    exact_k_shape=(2, 8),
    alias_shape=(200, 200),  # a square matrix, so y = a a
    determinism_shape=(129, 120, 40),
    too_large=[
        ("m", (8_388_481, 8, 8)),
        ("n", (8, 8_388_488, 8)),
        ("k", (8, 8, 2**16 + 8)),
    ],
)

LAYOUTS = [NT, NN]
by_layout = pytest.mark.parametrize("layout", LAYOUTS, ids=[layout.name for layout in LAYOUTS])


def shape_id(shape):
    return "x".join(map(str, shape))


# -- Inputs -----------------------------------------------------------------


def realistic(m, n, k, seed):
    """Activations about N(0, 1) and weights about N(0, 0.02). w is n x k."""
    g = generator(seed)
    return normal((m, k), 1.0, g), normal((n, k), 0.02, g)


def outliers(m, n, k, seed):
    """Realistic inputs, with about 0.5% of x's channels (at least 2) near
    +-3,000 in every row, as in GPT-2's activations."""
    x, w = realistic(m, n, k, seed)
    g = generator(10_000 + seed)
    count = max(2, round(0.005 * k))
    channels = torch.randperm(k, generator=g, device="cuda")[:count]
    signs = torch.randint(0, 2, (count,), generator=g, device="cuda") * 2.0 - 1.0
    jitter = 0.9 + 0.2 * torch.rand((m, count), generator=g, device="cuda")
    x[:, channels] = (3000.0 * signs * jitter).half()
    return x, w


INPUTS = {"realistic": realistic, "outliers": outliers}


# -- Running Flint ----------------------------------------------------------


def prepare(layout, make_inputs, m, n, k, seed):
    """x (m x k) and w in the layout's form (n x k or k x n)."""
    x, w = make_inputs(m, n, k, seed)
    return x, layout.to_w(w)


def flint_gemm(flint, layout, x, w, offset=8, fill=SENTINEL):
    """Runs the layout's GEMM into y inside a buffer pre-filled with `fill` (a
    16-bit pattern), offset halves in. A multiple of 8 keeps y 16-byte
    aligned. Checks the buffer around y is untouched and returns y as an m x n
    FP16 view."""
    m, k = x.shape
    n = layout.n_of(w)
    buf = torch.full((offset + m * n + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    y = buf[offset : offset + m * n].view(torch.float16)
    layout.call(flint, y, x, w, m, n, k)
    torch.cuda.synchronize()
    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert (buf[:offset] == guard).all(), f"{layout.flint_name} wrote before y"
    assert (buf[offset + m * n :] == guard).all(), f"{layout.flint_name} wrote past y"
    return y.view(m, n)


def gate_triples(flint, layout, make_inputs, m, n, k, offset=8):
    """Per seed, Flint's output, the peer's and the reference's."""
    for seed in gate.SEEDS:
        x, w = prepare(layout, make_inputs, m, n, k, seed)
        yield flint_gemm(flint, layout, x, w, offset), layout.peer(x, w), layout.reference(x, w)


# -- The gate ---------------------------------------------------------------

GPT2_CASES = [
    pytest.param(layout, name, variant, id=f"{layout.name}-{name}-{variant}")
    for layout in LAYOUTS
    for name in layout.gpt2_shapes
    for variant in INPUTS
]


@pytest.mark.parametrize("layout, name, variant", GPT2_CASES)
def test_gpt2_shapes_pass_the_gate(flint, layout, name, variant):
    # NT is the forward pass and NN the input gradient, which the standard
    # allows 3x the peer's error. Both are held to the forward pass's 2x: it's
    # stricter, and NN passes it.
    n, k = layout.gpt2_shapes[name]
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, layout, INPUTS[variant], GPT2_M, n, k))


# Offsets rotate through 8, 16 and 24 halves.
ODD_CASES = [
    pytest.param(
        layout, shape, variant, 8 * (1 + (i + j) % 3), id=f"{layout.name}-{shape_id(shape)}-{variant}"
    )
    for layout in LAYOUTS
    for i, shape in enumerate(layout.odd_shapes)
    for j, variant in enumerate(INPUTS)
]


@pytest.mark.parametrize("layout, shape, variant, offset", ODD_CASES)
def test_odd_shapes_pass_the_gate_and_stay_inside_y(flint, layout, shape, variant, offset):
    m, n, k = shape
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, layout, INPUTS[variant], m, n, k, offset))


# -- FP16's range -----------------------------------------------------------


def large_x(m, n, k, seed):
    g = generator(seed)
    return clamp_half(torch.randn((m, k), generator=g, device="cuda") * 20000, 65504), normal(
        (n, k), 0.02, g
    )


def large_w(m, n, k, seed):
    g = generator(seed)
    return normal((m, k), 0.02, g), clamp_half(
        torch.randn((n, k), generator=g, device="cuda") * 20000, 65504
    )


def extreme_x(m, n, k, seed):
    """One +-65,504 per row of x, times weights below 1, so each output is a
    single large product."""
    g = generator(seed)
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    columns = torch.randint(0, k, (m,), generator=g, device="cuda")
    signs = torch.randint(0, 2, (m,), generator=g, device="cuda") * 2.0 - 1.0
    x[torch.arange(m, device="cuda"), columns] = (65504.0 * signs).half()
    w = ((torch.rand((n, k), generator=g, device="cuda") * 2 - 1) * 0.9).half()
    return x, w


def small_both(m, n, k, seed):
    """Magnitudes 1e-3 to 1e-2 in both, so outputs sit around FP16's smallest
    normal, 6.1e-5."""
    g = generator(seed)
    return tuple(
        (10 ** (-3 + torch.rand(shape, generator=g, device="cuda"))
         * (torch.randint(0, 2, shape, generator=g, device="cuda") * 2.0 - 1.0)).half()
        for shape in ((m, k), (n, k))
    )


def small_with_subnormal_w(m, n, k, seed):
    """x as above, w from 1e-5 to 1e-4 (subnormal and just-normal inputs), so
    outputs land deep in the subnormal range, where the max-abs floor matters."""
    x, _ = small_both(m, n, k, seed)
    g = generator(20_000 + seed)
    scale = 10 ** (-5 + torch.rand((n, k), generator=g, device="cuda"))
    signs = torch.randint(0, 2, (n, k), generator=g, device="cuda") * 2.0 - 1.0
    return x, (scale * signs).half()


# (name, inputs, (m, n, k)), with n off the 128 tile. NN's n rounds up to a
# multiple of 8 (70 to 72, 65 to 72).
MAGNITUDE_SHAPES = [
    ("large_x", large_x, (130, 70, 104)),
    ("large_w", large_w, (130, 70, 104)),
    ("extreme_x", extreme_x, (130, 70, 104)),
    ("small_k8", small_both, (97, 65, 8)),
    ("small_k64", small_both, (97, 65, 64)),
    ("small_subnormal_w", small_with_subnormal_w, (97, 65, 64)),
]
MAGNITUDE_CASES = [
    pytest.param(layout, make_inputs, (m, layout.fit(n), k), id=f"{layout.name}-{name}")
    for layout in LAYOUTS
    for name, make_inputs, (m, n, k) in MAGNITUDE_SHAPES
]


@pytest.mark.parametrize("layout, make_inputs, shape", MAGNITUDE_CASES)
def test_fp16_range_inputs_pass_the_gate(flint, layout, make_inputs, shape):
    m, n, k = shape
    if make_inputs in (large_x, large_w, extreme_x):
        # The point of these inputs is large finite outputs, so check they are.
        x, w = prepare(layout, make_inputs, m, n, k, 0)
        top = layout.reference(x, w).abs().max().item()
        assert 1000 < top < 65504, top
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, layout, make_inputs, m, n, k))


@by_layout
def test_overflow_gives_inf_with_the_references_sign(flint, layout):
    m, n, k = 150, layout.fit(130), 64
    # The reference's std is about 40,000, so roughly one output in ten is
    # past FP16's range. Outputs between the two thresholds are within an FP32
    # accumulation error of the rounding boundary 65,520, so either answer is
    # right for them.
    inf_threshold = 65520.0 * (1 + 2.0**-8)
    finite_threshold = 65504.0 * (1 - 2.0**-8)

    def triples():
        for seed in gate.SEEDS:
            g = generator(seed)
            x = normal((m, k), 150.0, g)
            w = layout.to_w(normal((n, k), 33.0, g))
            ref = layout.reference(x, w)
            y = flint_gemm(flint, layout, x, w)
            peer = layout.peer(x, w)

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


# -- Non-finite inputs ------------------------------------------------------


@pytest.mark.parametrize("which", ["nan_in_x", "inf_in_w"])
@by_layout
def test_nonfinite_inputs_propagate_like_the_reference(flint, layout, which):
    m, n, k = 150, layout.fit(133), 104

    def triples():
        for seed in gate.SEEDS:
            x, w = prepare(layout, realistic, m, n, k, seed)
            row, column = (seed * 37 + 11) % m, (seed * 29 + 5) % k
            if which == "nan_in_x":
                x[row, column] = math.nan
            else:
                # The element pairing output column row % n with x's `column`.
                layout.set_w(w, row % n, column, math.inf)
            ref = layout.reference(x, w)
            y = flint_gemm(flint, layout, x, w)
            peer = layout.peer(x, w)

            # The same NaNs and infs in the same places, with the same signs.
            assert torch.equal(torch.isnan(y), torch.isnan(ref))
            assert torch.equal(torch.isposinf(y), torch.isposinf(ref))
            assert torch.equal(torch.isneginf(y), torch.isneginf(ref))
            # One bad element spoils exactly one row (NaN) or column (inf).
            finite = torch.isfinite(ref)
            assert (~finite).sum() == (n if which == "nan_in_x" else m)
            assert finite.sum() > 0
            yield y[finite], peer[finite], ref[finite]

    gate.gate(gate.FORWARD_FACTOR, triples())


# -- Determinism ------------------------------------------------------------

DETERMINISM_CASES = [
    pytest.param(layout, (GPT2_M, *layout.gpt2_shapes["qkv"]), id=f"{layout.name}-qkv")
    for layout in LAYOUTS
] + [
    pytest.param(layout, layout.determinism_shape, id=f"{layout.name}-{shape_id(layout.determinism_shape)}")
    for layout in LAYOUTS
]


@pytest.mark.parametrize("layout, shape", DETERMINISM_CASES)
def test_run_twice_is_bitwise_identical(flint, layout, shape):
    m, n, k = shape
    x, w = prepare(layout, outliers, m, n, k, 0)
    # Different NaN patterns to start from, so a store that is skipped, or one
    # that lands in only one run, shows as a difference or a leftover NaN.
    first = flint_gemm(flint, layout, x, w, fill=0xFFFF)
    second = flint_gemm(flint, layout, x, w, fill=0x7F7F)

    assert not torch.isnan(first).any()
    assert not torch.isnan(second).any()
    assert torch.equal(first.view(torch.int16), second.view(torch.int16))


# -- Aliasing and streams ---------------------------------------------------


@by_layout
def test_x_and_w_may_be_the_same_tensor(flint, layout):
    # NT: y = a a^T. NN needs a square a, so that y = a a.
    size, k = layout.alias_shape

    def triples():
        for seed in gate.SEEDS:
            g = generator(seed)
            a = normal((size, k), 1.0, g)
            yield flint_gemm(flint, layout, a, a), layout.peer(a, a), layout.reference(a, a)

    gate.gate(gate.FORWARD_FACTOR, triples())


@by_layout
def test_gemm_runs_on_the_given_stream(flint, layout):
    m = n = k = 64
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")
    y = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fill finishes only after the sleep, so the product is k only if the
    # GEMM is queued on the same stream behind it.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(1)
        layout.call(flint, y, x, w, m, n, k, stream=stream)
    stream.synchronize()

    assert (y == k).all()


@by_layout
def test_gemm_runs_on_the_default_stream(flint, layout):
    m = n = k = 64
    x = torch.zeros((m, k), dtype=torch.float16, device="cuda")
    w = torch.ones((n, k), dtype=torch.float16, device="cuda")
    y = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()

    torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
    x.fill_(1)
    layout.call(flint, y, x, w, m, n, k)
    torch.cuda.synchronize()

    assert (y == k).all()


# -- The size limits --------------------------------------------------------

# Each layout's shapes are listed with it. Skipped under the sanitizer, where
# 65,535 tiles or a 2^16-step k loop would take too long.
LIMIT_CASES = [
    pytest.param(layout, shape, id=f"{layout.name}-{name}")
    for layout in LAYOUTS
    for name, shape in layout.limit_shapes
]


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@pytest.mark.parametrize("layout, shape", LIMIT_CASES)
def test_largest_allowed_dimensions_pass_the_gate(flint, layout, shape):
    m, n, k = shape
    gate.gate(gate.FORWARD_FACTOR, gate_triples(flint, layout, realistic, m, n, k))


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@by_layout
def test_largest_allowed_k_is_exact_on_small_integers(flint, layout):
    # Sums of products from {-1, 0, 1} are integers, which FP32 holds exactly
    # at every step, so this checks indexing across the whole k with no
    # rounding to blur it.
    (m, n), k = layout.exact_k_shape, 2**16
    g = generator(0)
    x = (torch.randint(-1, 2, (m, k), generator=g, device="cuda")).half()
    w = (torch.randint(-1, 2, (n, k), generator=g, device="cuda")).half()
    expected = (x.float() @ w.float().T).half()

    assert torch.equal(flint_gemm(flint, layout, x, layout.to_w(w)), expected)


# -- Argument errors --------------------------------------------------------


def small_problem(m=16, n=16, k=16):
    """Valid tensors, in buffers with room to spare."""
    y = torch.zeros(m * n + 64, dtype=torch.float16, device="cuda")
    x = torch.ones(m * k + 64, dtype=torch.float16, device="cuda")
    w = torch.ones(n * k + 64, dtype=torch.float16, device="cuda")
    return y, x, w


def expect_invalid_argument(flint, layout, reason, y, x, w, m, n, k):
    with pytest.raises(flint_lib.FlintError) as error:
        layout.call(flint, y, x, w, m, n, k)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in (layout.flint_name, reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("name", ["y", "x", "w"])
@by_layout
def test_null_pointer(flint, layout, name):
    y, x, w = small_problem()
    pointers = {"y": y, "x": x, "w": w}
    pointers[name] = None
    expect_invalid_argument(
        flint, layout, f"{name} is null", pointers["y"], pointers["x"], pointers["w"], 16, 16, 16
    )


@pytest.mark.parametrize("k", [0, 12])
@by_layout
def test_k_must_be_a_positive_multiple_of_8(flint, layout, k):
    y, x, w = small_problem()
    expect_invalid_argument(flint, layout, f"k ({k}) must be a positive multiple of 8", y, x, w, 16, 16, k)


def test_nn_n_must_be_a_multiple_of_8(flint):
    # NT takes any n, since its w rows are k long. NN's are n long.
    y, x, w = small_problem(n=12)
    expect_invalid_argument(flint, NN, "n (12) must be a multiple of 8", y, x, w, 16, 12, 16)
    flint.gemm_nt(y, x, w, 16, 12, 16)
    torch.cuda.synchronize()
    assert (y[: 16 * 12] == 16).all()


@pytest.mark.parametrize("name", ["y", "x", "w"])
@by_layout
def test_misaligned_pointer(flint, layout, name):
    y, x, w = small_problem()
    pointers = {"y": y.data_ptr(), "x": x.data_ptr(), "w": w.data_ptr()}
    pointers[name] += 2
    expect_invalid_argument(
        flint, layout, f"{name} is not 16-byte aligned", pointers["y"], pointers["x"], pointers["w"], 16, 16, 16
    )


# The size checks run before any memory is touched, so small tensors do.
TOO_LARGE_CASES = [
    pytest.param(layout, shape, id=f"{layout.name}-{name}")
    for layout in LAYOUTS
    for name, shape in layout.too_large
]


@pytest.mark.parametrize("layout, shape", TOO_LARGE_CASES)
def test_too_large_dimensions(flint, layout, shape):
    m, n, k = shape
    y, x, w = small_problem()
    expect_invalid_argument(flint, layout, "too large", y, x, w, m, n, k)


@by_layout
def test_y_overlapping_x(flint, layout):
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    # x starts 256 bytes into y, and, in the second call, 16 bytes before y's end.
    expect_invalid_argument(flint, layout, "y and x overlap", buf, buf[128:], buf[2048:], 16, 16, 16)
    expect_invalid_argument(flint, layout, "y and x overlap", buf, buf[248:], buf[2048:], 16, 16, 16)


@by_layout
def test_y_overlapping_w(flint, layout):
    buf = torch.zeros(4096, dtype=torch.float16, device="cuda")
    expect_invalid_argument(flint, layout, "y and w overlap", buf, buf[2048:], buf[128:], 16, 16, 16)
    expect_invalid_argument(flint, layout, "y and w overlap", buf, buf[2048:], buf[248:], 16, 16, 16)


@by_layout
def test_adjacent_ranges_are_not_overlapping(flint, layout):
    # y, then x directly after it, in one buffer.
    buf = torch.zeros(1024, dtype=torch.float16, device="cuda")
    buf[256:512] = 1
    w = torch.ones(256, dtype=torch.float16, device="cuda")

    layout.call(flint, buf, buf[256:], w, 16, 16, 16)
    torch.cuda.synchronize()

    assert (buf[:256] == 16).all()
    assert (buf[256:512] == 1).all()
    assert (buf[512:] == 0).all()


@pytest.mark.parametrize("m, n", [(0, 16), (16, 0), (0, 0)])
@by_layout
def test_empty_output_does_nothing(flint, layout, m, n):
    y, x, w = small_problem()
    y.fill_(3)

    layout.call(flint, y, x, w, m, n, 16)
    layout.call(flint, None, None, None, m, n, 16)
    torch.cuda.synchronize()

    assert (y == 3).all()


@by_layout
def test_a_valid_call_works_after_errors(flint, layout):
    y, x, w = small_problem()
    for k in (0, 12):
        with pytest.raises(flint_lib.FlintError):
            layout.call(flint, y, x, w, 16, 16, k)
    with pytest.raises(flint_lib.FlintError):
        layout.call(flint, None, x, w, 16, 16, 16)
    if layout is NN:
        with pytest.raises(flint_lib.FlintError):
            layout.call(flint, y, x, w, 16, 12, 16)

    layout.call(flint, y, x, w, 16, 16, 16)
    torch.cuda.synchronize()

    assert (y[:256] == 16).all()
    assert (y[256:] == 0).all()
