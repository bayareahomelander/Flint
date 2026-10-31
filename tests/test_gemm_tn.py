"""flint_gemm_tn: y = x^T w, a linear layer's weight gradient (dW = dY^T X), with
FP32 output, optional accumulation, and k split across blocks (with a
workspace) when the output has few tiles.

x is k x m and w is k x n, both FP16; y is m x n FP32; m and n are multiples
of 8 and k is anything from 1. The tests cover the correctness gate on GPT-2's
weight-gradient shapes, a tighter FP32 check, the split-k path, accumulation,
odd shapes, FP16's range, non-finite inputs, determinism, stream ordering,
and argument errors.

The gate's peer is PyTorch's FP16 path, whose FP16 output is far coarser than
Flint's FP32 one, so the gate alone is loose. Every gated case is therefore
also held to FP32_REL_L2_BOUND against the reference.

The number of splits depends on the GPU's SM count, so the tests never assume
one: they ask `gemm_tn_workspace` for the workspace, and assert only that it is
nonzero where a few tiles meet a large k, and zero where nothing can split."""

import math

import pytest
import torch

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, clamp_half, generator, normal

GPT2_K = 64 if SANITIZER else 1024  # the tokens summed over: rows of x and w
# GPT-2 small's weight gradients dW = dY^T X as (m, n): m is the layer's
# output width and n its input width.
GPT2_SHAPES = {
    "qkv": (2304, 768),
    "attention_projection": (768, 768),
    "mlp_up": (3072, 768),
    "mlp_down": (768, 3072),
    "logits": (50304, 768),
}

# Flint's relative L2 error against the reference for FP32 output. The tensor
# cores truncate as they accumulate, so it grows with k. Measured worst over
# this file's cases, on the RTX 4050: 1.3e-5 (2304 x 768, k = 8192, unsplit);
# 7e-6 at k = 65,536 (8 x 8, 16 splits); under 2e-6 at k = 1024 and below. The
# bound leaves 7x over the worst, and is 5x tighter than the gate's floor of
# 2^-11 (4.9e-4), which is what FP16 output rounding alone reaches.
FP32_REL_L2_BOUND = 1e-4

# y sits `offset` floats into a buffer filled with a pattern that must be
# intact around it afterwards. A multiple of 4 floats keeps y 16-byte aligned.
GUARD_FLOATS = 64
SENTINEL = 0x5A5A5A5A
NAN_BITS = -1  # 0xFFFFFFFF, a NaN as float32 or two NaN halves
OTHER_BITS = 0x7F7F7F7F  # a finite 3.4e38 as float32
WORKSPACE_GUARD_FLOATS = 64


def shape_id(shape):
    return "x".join(map(str, shape))


# -- Inputs -----------------------------------------------------------------
# Every maker returns x (k x m) and w (k x n).


def gradients(m, n, k, seed):
    """dY and X both about N(0, 1)."""
    g = generator(seed)
    return normal((k, m), 1.0, g), normal((k, n), 1.0, g)


def realistic(m, n, k, seed):
    """x about N(0, 1) and w about N(0, 0.02)."""
    g = generator(seed)
    return normal((k, m), 1.0, g), normal((k, n), 0.02, g)


def outliers(m, n, k, seed):
    """X with about 0.5% of its channels (at least 2) near +-3,000 for every
    token, as in GPT-2's activations. dY is N(0, 0.01), a gradient's scale: at
    N(0, 1), the outlier columns of dY^T X would sum to about 3,000 sqrt(k) and
    overflow the FP16 peer."""
    g = generator(seed)
    x = normal((k, m), 0.01, g)
    w = normal((k, n), 1.0, g)
    g2 = generator(10_000 + seed)
    count = max(2, round(0.005 * n))
    channels = torch.randperm(n, generator=g2, device="cuda")[:count]
    signs = torch.randint(0, 2, (count,), generator=g2, device="cuda") * 2.0 - 1.0
    jitter = 0.9 + 0.2 * torch.rand((k, count), generator=g2, device="cuda")
    w[:, channels] = (3000.0 * signs * jitter).half()
    return x, w


INPUTS = {"gradient": gradients, "outliers": outliers}
SPLIT_INPUTS = {"gradient": gradients, "realistic": realistic, "outliers": outliers}


# -- Running Flint ----------------------------------------------------------


def run_tn(flint, x, w, *, accumulate=False, y_init=None, offset=4, y_fill=NAN_BITS,
           workspace_fill=NAN_BITS, stream=None):
    """Runs flint_gemm_tn into y inside a guarded FP32 buffer, with the
    workspace gemm_tn_workspace asks for (inside a guarded buffer of its own),
    and checks the guards. y starts as y_init if given, and otherwise as the
    bit pattern y_fill (NaN). Returns y as an m x n FP32 view."""
    k, m = x.shape
    n = w.shape[1]
    buf = torch.full((offset + m * n + GUARD_FLOATS,), SENTINEL, dtype=torch.int32, device="cuda")
    y = buf[offset : offset + m * n].view(torch.float32)
    if y_init is None:
        y.view(torch.int32).fill_(y_fill)
    else:
        y.copy_(y_init.reshape(-1))

    needed = flint.gemm_tn_workspace(m, n, k)
    assert needed % 4 == 0
    workspace = None
    if needed:
        workspace = torch.full(
            (needed // 4 + WORKSPACE_GUARD_FLOATS,), SENTINEL, dtype=torch.int32, device="cuda"
        )
        workspace[: needed // 4] = workspace_fill

    flint.gemm_tn(y, x, w, m, n, k, accumulate=accumulate, workspace=workspace,
                  workspace_bytes=needed, stream=stream)
    torch.cuda.synchronize()

    guard = torch.full((1,), SENTINEL, dtype=torch.int32, device="cuda")
    assert (buf[:offset] == guard).all(), "flint_gemm_tn wrote before y"
    assert (buf[offset + m * n :] == guard).all(), "flint_gemm_tn wrote past y"
    if needed:
        assert (workspace[needed // 4 :] == guard).all(), "flint_gemm_tn wrote past its workspace"
    return y.view(m, n)


def check_fp32(y, ref):
    """Flint's relative L2 against the reference is within FP32_REL_L2_BOUND."""
    rel = gate.errors(y, ref).rel_l2
    assert rel <= FP32_REL_L2_BOUND, (
        f"FP32 relative L2 {rel:.4g} is past the bound {FP32_REL_L2_BOUND:.4g} "
        f"(shape {tuple(y.shape)})"
    )


def tn_triples(flint, make_inputs, m, n, k, offset=4):
    """Per seed, Flint's output, the peer's and the reference's; checks the
    FP32 bound on the way."""
    for seed in gate.SEEDS:
        x, w = make_inputs(m, n, k, seed)
        y = run_tn(flint, x, w, offset=offset)
        ref = gate.reference_tn(x, w)
        check_fp32(y, ref)
        yield y, gate.peer_gemm_tn(x, w), ref


def check_gate(flint, make_inputs, m, n, k, offset=4):
    """The gradient gate and the FP32 bound on one shape."""
    gate.gate(gate.GRADIENT_FACTOR, tn_triples(flint, make_inputs, m, n, k, offset))


# -- The gate ---------------------------------------------------------------

GPT2_CASES = [
    pytest.param(name, variant, id=f"{name}-{variant}")
    for name in GPT2_SHAPES
    for variant in INPUTS
]


@pytest.mark.parametrize("name, variant", GPT2_CASES)
def test_gpt2_weight_gradients_pass_the_gate(flint, name, variant):
    m, n = GPT2_SHAPES[name]
    check_gate(flint, INPUTS[variant], m, n, GPT2_K)


# At k = 8192, 768x768 splits on the 2080 Ti but not here, and 2304x768 (108
# tiles) splits on neither. These tests don't assume the split rule.
LONG_K_CASES = [
    pytest.param(shape, variant, id=f"{shape_id(shape)}-{variant}")
    for shape in [(768, 768, 8192), (2304, 768, 8192)]
    for variant in INPUTS
]


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
@pytest.mark.parametrize("shape, variant", LONG_K_CASES)
def test_long_k_passes_the_gate(flint, shape, variant):
    check_gate(flint, INPUTS[variant], *shape)


# -- Split k ----------------------------------------------------------------

# Few tiles and a large k: these split on every GPU Flint targets (one tile and
# k / 1024 = 16 splits, 2 tiles and 2 splits, and 6 tiles and 2 splits, the
# last one short). The last two have 9 and 17 rows of tiles, two and three
# groups of rows, so they check how a block finds its split and its group.
SPLITTING_SHAPES = [(8, 8, 16384), (136, 120, 3000), (296, 200, 2049), (1032, 8, 4096),
                    (2056, 16, 8192)]
# Whether these split depends on the SM count. The largest k is 2^16.
SPLIT_SHAPES = SPLITTING_SHAPES + [(768, 768, 8192), (8, 8, 65536)]
# k below 1024 never splits, and neither do outputs with plenty of tiles.
NON_SPLITTING_SHAPES = [
    (8, 8, 1),
    (8, 8, 1000),
    (136, 120, 1023),
    (8, 8, 1024),
    (50304, 768, 1024),
    (50304, 768, 65536),  # 393 x 6 tiles
    (4096, 4096, 65536),
    (2304, 768, 1024),
]

SPLIT_CASES = [
    pytest.param(shape, variant, id=f"{shape_id(shape)}-{variant}")
    for shape in SPLIT_SHAPES
    for variant in SPLIT_INPUTS
    if not (SANITIZER and shape[2] >= 8192 and shape[0] > 8)
]


@pytest.mark.parametrize("shape, variant", SPLIT_CASES)
def test_split_k_passes_the_gate(flint, shape, variant):
    check_gate(flint, SPLIT_INPUTS[variant], *shape)


@pytest.mark.parametrize("shape", SPLITTING_SHAPES, ids=shape_id)
def test_few_tiles_with_a_large_k_need_a_workspace(flint, shape):
    m, n, k = shape
    needed = flint.gemm_tn_workspace(m, n, k)
    assert needed > 0
    # One FP32 output per split.
    assert needed % (m * n * 4) == 0 and needed // (m * n * 4) <= 16


@pytest.mark.parametrize("shape", NON_SPLITTING_SHAPES, ids=shape_id)
def test_small_k_or_many_tiles_need_no_workspace(flint, shape):
    assert flint.gemm_tn_workspace(*shape) == 0


@pytest.mark.parametrize("shape", [(8, 8, 1), (136, 120, 1023), (50304, 768, 1024)], ids=shape_id)
def test_unsplit_shapes_pass_the_gate(flint, shape):
    check_gate(flint, gradients, *shape)


# Sums of products from {-1, 0, 1} are integers, which FP32 holds exactly at
# every step, so any summing order gives the same result and every split
# boundary and k index is checked with no rounding to blur it.
EXACT_SHAPES = [(8, 8, 1), (136, 120, 3000), (296, 200, 2049), (8, 8, 16384), (8, 8, 65536),
                (136, 120, 1056), (24, 40, 3333), (1032, 8, 4096)]


def integer_inputs(m, n, k, seed=0):
    g = generator(seed)
    x = torch.randint(-1, 2, (k, m), generator=g, device="cuda").half()
    w = torch.randint(-1, 2, (k, n), generator=g, device="cuda").half()
    return x, w


@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
@pytest.mark.parametrize("shape", EXACT_SHAPES, ids=shape_id)
def test_integer_inputs_are_exact(flint, shape, accumulate):
    m, n, k = shape
    x, w = integer_inputs(m, n, k)
    expected = (x.double().T @ w.double()).float()
    if accumulate:
        g = generator(5)
        y0 = torch.randint(-1000, 1000, (m, n), generator=g, device="cuda").float()
        y = run_tn(flint, x, w, accumulate=True, y_init=y0)
        expected = expected + y0
    else:
        y = run_tn(flint, x, w)
    assert torch.equal(y, expected)


# -- Accumulation -----------------------------------------------------------

# Shapes with and without splits (the second and third split on every GPU),
# and odd ones.
ACCUMULATE_SHAPES = [(136, 120, 96), (136, 120, 3000), (296, 200, 2049), (8, 8, 16384),
                     (2056, 8, 31), (8, 8, 1), (768, 768, 8192)]
ACCUMULATE_CASES = [
    pytest.param(shape, id=shape_id(shape))
    for shape in ACCUMULATE_SHAPES
    if not (SANITIZER and shape[2] >= 8192 and shape[0] > 8)
]


def random_y0(m, n, seed, std=8.0):
    g = generator(seed)
    return torch.randn((m, n), generator=g, device="cuda") * std


@pytest.mark.parametrize("shape", ACCUMULATE_CASES)
def test_accumulate_adds_into_y(flint, shape):
    m, n, k = shape
    for seed in gate.SEEDS:
        x, w = realistic(m, n, k, seed) if seed else gradients(m, n, k, seed)
        y0 = random_y0(m, n, seed)
        y = run_tn(flint, x, w, accumulate=True, y_init=y0)
        ref = gate.reference_tn(x, w)
        check_fp32(y, y0.to(ref.dtype) + ref)


@pytest.mark.parametrize("shape", ACCUMULATE_CASES)
def test_accumulating_is_y_plus_the_plain_result_bitwise(flint, shape):
    # Ordinary FP32 adds: old + (the sum), with the sum the same as an
    # overwriting call gives, split or not.
    m, n, k = shape
    x, w = outliers(m, n, k, 0)
    y0 = random_y0(m, n, 1, std=100.0)
    plain = run_tn(flint, x, w)
    added = run_tn(flint, x, w, accumulate=True, y_init=y0)
    assert torch.equal(added.view(torch.int32), (y0 + plain).view(torch.int32))


MICRO_BATCH_SHAPES = [(136, 120, 96), (136, 120, 3000), (296, 200, 2049), (8, 8, 4096)]


@pytest.mark.parametrize("first_overwrites", [False, True], ids=["from-zeros", "first-overwrites"])
@pytest.mark.parametrize("shape", MICRO_BATCH_SHAPES, ids=shape_id)
def test_four_micro_batches_sum(flint, shape, first_overwrites):
    m, n, k = shape
    needed = flint.gemm_tn_workspace(m, n, k)
    workspace = torch.empty(needed // 4, dtype=torch.int32, device="cuda") if needed else None
    y = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    if first_overwrites:
        y.fill_(math.nan)
    total = torch.zeros((m, n), dtype=torch.float64, device="cuda")
    for batch in range(4):
        x, w = gradients(m, n, k, 100 + batch)
        flint.gemm_tn(y, x, w, m, n, k, accumulate=not (first_overwrites and batch == 0),
                      workspace=workspace, workspace_bytes=needed)
        total += gate.reference_tn(x, w).double()
    torch.cuda.synchronize()
    check_fp32(y, total)


@pytest.mark.parametrize("fill", [math.nan, math.inf, -math.inf, 3.0e38, 1.0])
@pytest.mark.parametrize("shape", [(136, 120, 96), (136, 120, 3000), (8, 8, 16384), (2056, 8, 31)],
                         ids=shape_id)
def test_without_accumulate_old_contents_do_not_matter(flint, shape, fill):
    m, n, k = shape
    x, w = gradients(m, n, k, 0)
    y0 = torch.full((m, n), fill, dtype=torch.float32, device="cuda")
    first = run_tn(flint, x, w, y_init=y0)
    reference_run = run_tn(flint, x, w, y_init=torch.zeros_like(y0))
    assert torch.equal(first.view(torch.int32), reference_run.view(torch.int32))
    check_fp32(first, gate.reference_tn(x, w))


# -- Odd shapes -------------------------------------------------------------

# m and n multiples of 8 but not of the 128 tile, and odd k (so a short last
# step of 32 rows of k). Tiles go out in groups of up to 8 rows of tiles: 17
# rows of tiles end in a short group of 1 row (2056), 9 in a group of 1 (1032),
# 34 in a group of 2 (4232).
ODD_SHAPES = [
    (8, 8, 1),
    (16, 8, 7),
    (136, 120, 40),
    (120, 136, 33),
    (776, 200, 777),
    (8, 50304, 129),
    (2056, 8, 31),
    (1032, 136, 33),
    (4232, 8, 17),
]
# Offsets rotate through 4, 8 and 12 floats.
ODD_CASES = [
    pytest.param(shape, variant, 4 * (1 + (i + j) % 3), id=f"{shape_id(shape)}-{variant}")
    for i, shape in enumerate(ODD_SHAPES)
    for j, variant in enumerate(INPUTS)
]


@pytest.mark.parametrize("shape, variant, offset", ODD_CASES)
def test_odd_shapes_pass_the_gate_and_stay_inside_y(flint, shape, variant, offset):
    check_gate(flint, INPUTS[variant], *shape, offset=offset)


# -- FP16's range -----------------------------------------------------------


def large_x(m, n, k, seed):
    g = generator(seed)
    x = clamp_half(torch.randn((k, m), generator=g, device="cuda") * 20000, 65504)
    return x, normal((k, n), 0.02, g)


def large_x_long_k(m, n, k, seed):
    """large_x with weights of 0.005, so a k near 3,000 still gives outputs
    the FP16 peer can hold."""
    g = generator(seed)
    x = clamp_half(torch.randn((k, m), generator=g, device="cuda") * 20000, 65504)
    return x, normal((k, n), 0.005, g)


def large_w(m, n, k, seed):
    g = generator(seed)
    x = normal((k, m), 0.02, g)
    return x, clamp_half(torch.randn((k, n), generator=g, device="cuda") * 20000, 65504)


def extreme_x(m, n, k, seed):
    """One +-65,504 per column of x (per output row), times weights below 1, so
    each output is a single large product."""
    g = generator(seed)
    x = torch.zeros((k, m), dtype=torch.float16, device="cuda")
    rows = torch.randint(0, k, (m,), generator=g, device="cuda")
    signs = torch.randint(0, 2, (m,), generator=g, device="cuda") * 2.0 - 1.0
    x[rows, torch.arange(m, device="cuda")] = (65504.0 * signs).half()
    w = ((torch.rand((k, n), generator=g, device="cuda") * 2 - 1) * 0.9).half()
    return x, w


def small_both(m, n, k, seed):
    """Magnitudes 1e-3 to 1e-2 in both, so outputs sit around FP16's smallest
    normal, 6.1e-5."""
    g = generator(seed)
    return tuple(
        (10 ** (-3 + torch.rand(shape, generator=g, device="cuda"))
         * (torch.randint(0, 2, shape, generator=g, device="cuda") * 2.0 - 1.0)).half()
        for shape in ((k, m), (k, n))
    )


def small_with_subnormal_w(m, n, k, seed):
    """x as above, w from 1e-5 to 1e-4 (subnormal and just-normal inputs), so
    outputs land deep in the subnormal range, where the max-abs floor matters."""
    x, _ = small_both(m, n, k, seed)
    g = generator(20_000 + seed)
    scale = 10 ** (-5 + torch.rand((k, n), generator=g, device="cuda"))
    signs = torch.randint(0, 2, (k, n), generator=g, device="cuda") * 2.0 - 1.0
    return x, (scale * signs).half()


MAGNITUDE_SHAPES = [
    ("large_x", large_x, (136, 72, 104)),
    ("large_w", large_w, (136, 72, 104)),
    ("extreme_x", extreme_x, (136, 72, 104)),
    ("small_k8", small_both, (104, 72, 8)),
    ("small_k64", small_both, (104, 72, 64)),
    ("small_subnormal_w", small_with_subnormal_w, (104, 72, 64)),
    # Large values over many k, which split.
    ("large_x_long_k", large_x_long_k, (136, 120, 3000)),
    ("small_split", small_both, (8, 8, 4096)),
]
MAGNITUDE_CASES = [
    pytest.param(make_inputs, shape, id=name) for name, make_inputs, shape in MAGNITUDE_SHAPES
]


@pytest.mark.parametrize("make_inputs, shape", MAGNITUDE_CASES)
def test_fp16_range_inputs_pass_the_gate(flint, make_inputs, shape):
    m, n, k = shape
    if make_inputs in (large_x, large_x_long_k, large_w, extreme_x):
        # The point of these inputs is large outputs that the FP16 peer can
        # still hold, so check they are.
        x, w = make_inputs(m, n, k, 0)
        top = gate.reference_tn(x, w).abs().max().item()
        assert 1000 < top < 65504, top
    check_gate(flint, make_inputs, m, n, k)


@pytest.mark.parametrize("shape", [(136, 72, 104), (8, 8, 16384), (24, 16, 2049)], ids=shape_id)
def test_outputs_past_fp16_stay_finite_in_fp32(flint, shape):
    # Products of FP16 values reach 4.3e9, so outputs far past FP16's 65,504
    # are fine in FP32. The FP16 peer overflows here, so only the FP32 bound
    # applies.
    m, n, k = shape
    for seed in gate.SEEDS:
        g = generator(seed)
        x = clamp_half(torch.randn((k, m), generator=g, device="cuda") * 40000, 65504)
        w = clamp_half(torch.randn((k, n), generator=g, device="cuda") * 40000, 65504)
        ref = gate.reference_tn(x, w)
        assert ref.abs().max().item() > 1e9
        y = run_tn(flint, x, w)
        assert torch.isfinite(y).all()
        check_fp32(y, ref)


# -- Non-finite inputs ------------------------------------------------------


@pytest.mark.parametrize("shape", [(152, 136, 104), (136, 120, 3000)], ids=shape_id)
@pytest.mark.parametrize("which", ["nan_in_x", "inf_in_w"])
def test_nonfinite_inputs_propagate_like_the_reference(flint, which, shape):
    m, n, k = shape

    def triples():
        for seed in gate.SEEDS:
            x, w = realistic(m, n, k, seed)
            column, t = (seed * 37 + 11) % m, (seed * 29 + 5 + 1400 * seed) % k
            if which == "nan_in_x":
                x[t, column] = math.nan  # spoils row `column` of y
            else:
                w[t, column % n] = math.inf  # spoils column `column % n`
            ref = gate.reference_tn(x, w)
            y = run_tn(flint, x, w)
            peer = gate.peer_gemm_tn(x, w)

            # The same NaNs and infs in the same places, with the same signs.
            assert torch.equal(torch.isnan(y), torch.isnan(ref))
            assert torch.equal(torch.isposinf(y), torch.isposinf(ref))
            assert torch.equal(torch.isneginf(y), torch.isneginf(ref))
            # One bad element spoils exactly one row (NaN) or column (inf).
            finite = torch.isfinite(ref)
            assert (~finite).sum() == (n if which == "nan_in_x" else m)
            assert finite.sum() > 0
            check_fp32(y[finite], ref[finite])
            yield y[finite], peer[finite], ref[finite]

    gate.gate(gate.GRADIENT_FACTOR, triples())


# -- Determinism ------------------------------------------------------------

DETERMINISM_SHAPES = [(2304, 768, GPT2_K), (136, 120, 40), (2056, 8, 31), (136, 120, 3000),
                      (296, 200, 2049), (8, 8, 16384), (1032, 8, 4096)]
if not SANITIZER:
    DETERMINISM_SHAPES.append((768, 768, 8192))


@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
@pytest.mark.parametrize("shape", DETERMINISM_SHAPES, ids=shape_id)
def test_run_twice_is_bitwise_identical(flint, shape, accumulate):
    m, n, k = shape
    x, w = outliers(m, n, k, 0)
    y0 = random_y0(m, n, 0) if accumulate else None
    # Different patterns in the workspace (and, without accumulate, in y) to
    # start from, so a store that is skipped, or a result that depends on old
    # contents, shows as a difference or a leftover NaN.
    first = run_tn(flint, x, w, accumulate=accumulate, y_init=y0, y_fill=NAN_BITS,
                   workspace_fill=NAN_BITS)
    second = run_tn(flint, x, w, accumulate=accumulate, y_init=y0, y_fill=OTHER_BITS,
                    workspace_fill=OTHER_BITS)
    third = run_tn(flint, x, w, accumulate=accumulate, y_init=y0, y_fill=0, workspace_fill=0)

    assert torch.isfinite(first).all()
    assert torch.equal(first.view(torch.int32), second.view(torch.int32))
    assert torch.equal(first.view(torch.int32), third.view(torch.int32))


# -- Streams ----------------------------------------------------------------

STREAM_SHAPES = [(64, 64, 64), (8, 8, 4096)]  # the second splits


def stream_problem(flint, shape):
    m, n, k = shape
    x = torch.zeros((k, m), dtype=torch.float16, device="cuda")
    w = torch.ones((k, n), dtype=torch.float16, device="cuda")
    y = torch.full((m, n), 5.0, dtype=torch.float32, device="cuda")
    needed = flint.gemm_tn_workspace(m, n, k)
    workspace = torch.empty(needed // 4, dtype=torch.int32, device="cuda") if needed else None
    if shape == STREAM_SHAPES[1]:
        assert needed > 0
    return x, w, y, workspace, needed


@pytest.mark.parametrize("shape", STREAM_SHAPES, ids=shape_id)
def test_gemm_tn_runs_on_the_given_stream(flint, shape):
    m, n, k = shape
    x, w, y, workspace, needed = stream_problem(flint, shape)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fill finishes only after the sleep, so y is 5 + k only if the GEMM
    # (and, when split, its reduction) is queued on the same stream behind it.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(1)
        flint.gemm_tn(y, x, w, m, n, k, accumulate=True, workspace=workspace,
                      workspace_bytes=needed, stream=stream)
    stream.synchronize()

    assert (y == 5 + k).all()


@pytest.mark.parametrize("shape", STREAM_SHAPES, ids=shape_id)
def test_gemm_tn_runs_on_the_default_stream(flint, shape):
    m, n, k = shape
    x, w, y, workspace, needed = stream_problem(flint, shape)
    torch.cuda.synchronize()

    torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
    x.fill_(1)
    flint.gemm_tn(y, x, w, m, n, k, accumulate=True, workspace=workspace, workspace_bytes=needed)
    torch.cuda.synchronize()

    assert (y == 5 + k).all()


# -- Argument errors --------------------------------------------------------


def small_problem(m=16, n=16, k=16):
    """Valid tensors, in buffers with room to spare: y FP32, x and w FP16."""
    y = torch.zeros(m * n + 64, dtype=torch.float32, device="cuda")
    x = torch.ones(k * m + 64, dtype=torch.float16, device="cuda")
    w = torch.ones(k * n + 64, dtype=torch.float16, device="cuda")
    return y, x, w


def expect_invalid_argument(flint, reason, y, x, w, m, n, k, **kwargs):
    with pytest.raises(flint_lib.FlintError) as error:
        flint.gemm_tn(y, x, w, m, n, k, **kwargs)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_gemm_tn", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("name", ["y", "x", "w"])
def test_null_pointer(flint, name, accumulate):
    y, x, w = small_problem()
    pointers = {"y": y, "x": x, "w": w}
    pointers[name] = None
    expect_invalid_argument(flint, f"{name} is null", pointers["y"], pointers["x"], pointers["w"],
                            16, 16, 16, accumulate=accumulate)


def test_k_must_be_positive(flint):
    y, x, w = small_problem()
    expect_invalid_argument(flint, "k must be positive", y, x, w, 16, 16, 0)


@pytest.mark.parametrize("which, m, n", [("m", 12, 16), ("n", 16, 12), ("m", 4, 16), ("n", 16, 1)])
def test_m_and_n_must_be_multiples_of_8(flint, which, m, n):
    y, x, w = small_problem()
    value = m if which == "m" else n
    expect_invalid_argument(flint, f"{which} ({value}) must be a multiple of 8", y, x, w, m, n, 16)


# The size checks run before any memory is touched, so small tensors do.
TOO_LARGE_CASES = [
    pytest.param((8_388_488, 8, 8), id="m"),  # a multiple of 8, past 8,388,480
    pytest.param((8, 8_388_488, 8), id="n"),
    pytest.param((8, 8, 2**16 + 1), id="k"),
]


@pytest.mark.parametrize("shape", TOO_LARGE_CASES)
def test_too_large_dimensions(flint, shape):
    m, n, k = shape
    y, x, w = small_problem()
    expect_invalid_argument(flint, "too large", y, x, w, m, n, k)


@pytest.mark.parametrize("offset", [2, 8], ids=lambda o: f"plus{o}bytes")
@pytest.mark.parametrize("name", ["y", "x", "w"])
def test_misaligned_pointer(flint, name, offset):
    y, x, w = small_problem()
    pointers = {"y": y.data_ptr(), "x": x.data_ptr(), "w": w.data_ptr()}
    pointers[name] += offset
    expect_invalid_argument(flint, f"{name} is not 16-byte aligned", pointers["y"], pointers["x"],
                            pointers["w"], 16, 16, 16)


def byte_buffer(nbytes=1 << 14):
    return torch.zeros(nbytes, dtype=torch.uint8, device="cuda")


def test_y_overlapping_x(flint):
    base = byte_buffer().data_ptr()
    # y is 1,024 bytes and x 512. x starts inside y, then 16 bytes before y's
    # end; and, with k = 64 (x of 2,048 bytes), x starts before y and runs into it.
    expect_invalid_argument(flint, "y and x overlap", base, base + 512, base + 4096, 16, 16, 16)
    expect_invalid_argument(flint, "y and x overlap", base, base + 1008, base + 4096, 16, 16, 16)
    expect_invalid_argument(flint, "y and x overlap", base + 1024, base, base + 8192, 16, 16, 64)


def test_y_overlapping_w(flint):
    base = byte_buffer().data_ptr()
    expect_invalid_argument(flint, "y and w overlap", base, base + 4096, base + 512, 16, 16, 16)
    expect_invalid_argument(flint, "y and w overlap", base, base + 4096, base + 1008, 16, 16, 16)
    expect_invalid_argument(flint, "y and w overlap", base + 1024, base + 8192, base, 16, 16, 64)


@pytest.mark.parametrize("order", ["y-x-w", "x-y-w", "x-w-y"])
def test_adjacent_ranges_are_not_overlapping(flint, order):
    # y is 1,024 bytes and x and w 512 each, laid end to end in one buffer.
    sizes = {"y": 1024, "x": 512, "w": 512}
    buf = byte_buffer(4096)
    views = {}
    offset = 0
    for name in order.split("-"):
        views[name] = buf[offset : offset + sizes[name]]
        offset += sizes[name]
    x = views["x"].view(torch.float16)
    w = views["w"].view(torch.float16)
    x.fill_(1)
    w.fill_(1)

    flint.gemm_tn(views["y"], x, w, 16, 16, 16)
    torch.cuda.synchronize()

    assert (views["y"].view(torch.float32) == 16).all()
    assert (x == 1).all() and (w == 1).all()
    assert (buf[offset:] == 0).all()


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("m, n", [(0, 16), (16, 0), (0, 0)])
def test_empty_output_does_nothing(flint, m, n, accumulate):
    y, x, w = small_problem()
    y.fill_(3)

    flint.gemm_tn(y, x, w, m, n, 16, accumulate=accumulate)
    flint.gemm_tn(None, None, None, m, n, 16, accumulate=accumulate)
    # Nothing is checked, not even k or a workspace.
    flint.gemm_tn(None, None, None, m, n, 0, accumulate=accumulate)
    torch.cuda.synchronize()

    assert (y == 3).all()


# The workspace checks: 8 x 8 x 16384 splits on every GPU Flint targets, with
# y of 256 bytes and x and w of 262,144 each.
WS_SHAPE = (8, 8, 16384)
WS_BUFFER = 1 << 20
WS_Y, WS_X, WS_W = 16384, 20480, 282624  # byte offsets in the buffer


def workspace_problem(flint):
    """A buffer holding y, x and w (all ones) at the offsets above, with free
    room before y, and the workspace bytes the shape needs."""
    buf = byte_buffer(WS_BUFFER)
    m, n, k = WS_SHAPE
    x = buf[WS_X : WS_X + k * m * 2].view(torch.float16)
    w = buf[WS_W : WS_W + k * n * 2].view(torch.float16)
    x.fill_(1)
    w.fill_(1)
    needed = flint.gemm_tn_workspace(m, n, k)
    assert 0 < needed <= WS_Y
    return buf, buf.data_ptr(), needed


def call_ws(flint, base, workspace, workspace_bytes):
    m, n, k = WS_SHAPE
    flint.gemm_tn(base + WS_Y, base + WS_X, base + WS_W, m, n, k, workspace=workspace,
                  workspace_bytes=workspace_bytes)


def test_a_needed_workspace_may_not_be_null(flint):
    buf, base, needed = workspace_problem(flint)
    m, n, k = WS_SHAPE
    expect_invalid_argument(flint, f"needs {needed} bytes of workspace, given 0", base + WS_Y,
                            base + WS_X, base + WS_W, m, n, k)
    expect_invalid_argument(flint, f"needs {needed} bytes of workspace, given 0", base + WS_Y,
                            base + WS_X, base + WS_W, m, n, k, workspace_bytes=needed)


@pytest.mark.parametrize("short", [16, 4, None], ids=["16-short", "4-short", "zero"])
def test_a_workspace_may_not_be_too_small(flint, short):
    buf, base, needed = workspace_problem(flint)
    m, n, k = WS_SHAPE
    given = 0 if short is None else needed - short
    # The buffer itself is big enough; it is the size passed that is short.
    expect_invalid_argument(flint, f"needs {needed} bytes of workspace, given {given}",
                            base + WS_Y, base + WS_X, base + WS_W, m, n, k, workspace=base,
                            workspace_bytes=given)


@pytest.mark.parametrize("offset", [4, 8], ids=lambda o: f"plus{o}bytes")
def test_a_workspace_must_be_aligned(flint, offset):
    buf, base, needed = workspace_problem(flint)
    m, n, k = WS_SHAPE
    expect_invalid_argument(flint, "workspace is not 16-byte aligned", base + WS_Y, base + WS_X,
                            base + WS_W, m, n, k, workspace=base + offset, workspace_bytes=needed)


@pytest.mark.parametrize(
    "where, start",
    [
        ("y_start", WS_Y),
        ("inside_y", WS_Y + 16),
        ("before_y", WS_Y - 16),
        ("inside_x", WS_X + 4096),
        ("before_x", WS_X - 16),
        ("inside_w", WS_W + 4096),
        ("before_w", WS_W - 16),
    ],
)
def test_a_workspace_may_not_overlap_y_x_or_w(flint, where, start):
    buf, base, needed = workspace_problem(flint)
    m, n, k = WS_SHAPE
    expect_invalid_argument(flint, "workspace overlaps y, x or w", base + WS_Y, base + WS_X,
                            base + WS_W, m, n, k, workspace=base + start, workspace_bytes=needed)


@pytest.mark.parametrize("start", ["buffer_start", "right_before_y", "after_w"])
def test_a_workspace_next_to_y_x_and_w_is_fine(flint, start):
    buf, base, needed = workspace_problem(flint)
    offset = {"buffer_start": 0, "right_before_y": WS_Y - needed,
              "after_w": WS_W + 16384 * 8 * 2}[start]
    # Every byte outside y and the workspace stays as it was.
    before = buf.clone()
    call_ws(flint, base, base + offset, needed)
    torch.cuda.synchronize()

    m, n, k = WS_SHAPE
    assert (buf[WS_Y : WS_Y + m * n * 4].view(torch.float32) == k).all()
    outside = torch.ones_like(buf, dtype=torch.bool)
    outside[WS_Y : WS_Y + m * n * 4] = False
    outside[offset : offset + needed] = False
    assert torch.equal(buf[outside], before[outside])


def test_a_workspace_is_not_needed_means_it_is_not_checked(flint):
    # 16 x 16 x 16 needs none: a misaligned or tiny workspace is never read.
    y, x, w = small_problem()
    assert flint.gemm_tn_workspace(16, 16, 16) == 0
    flint.gemm_tn(y, x, w, 16, 16, 16, workspace=y.data_ptr() + 2, workspace_bytes=1)
    flint.gemm_tn(y, x, w, 16, 16, 16)
    torch.cuda.synchronize()
    assert (y[:256] == 16).all()


@pytest.mark.parametrize(
    "m, n, k, reason",
    [
        (16, 16, 0, "k must be positive"),
        (12, 16, 16, "m (12) must be a multiple of 8"),
        (16, 12, 16, "n (12) must be a multiple of 8"),
        (8_388_488, 8, 8, "too large"),
        (8, 8_388_488, 8, "too large"),
        (8, 8, 2**16 + 1, "too large"),
    ],
)
def test_workspace_query_rejects_invalid_dimensions(flint, m, n, k, reason):
    with pytest.raises(flint_lib.FlintError) as error:
        flint.gemm_tn_workspace(m, n, k)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_gemm_tn" in str(error.value) and reason in str(error.value)
    assert reason in flint.last_error()


@pytest.mark.parametrize("m, n, k", [(0, 16, 16), (16, 0, 16), (0, 0, 0), (0, 12, 0)])
def test_workspace_query_for_an_empty_output_is_zero(flint, m, n, k):
    assert flint.gemm_tn_workspace(m, n, k) == 0


def test_workspace_query_needs_somewhere_to_write(flint):
    status = flint.lib.flint_gemm_tn_workspace(16, 16, 16, None)
    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_gemm_tn_workspace" in flint.last_error()


def test_a_valid_call_works_after_errors(flint):
    y, x, w = small_problem()
    y.fill_(7)
    for m, n, k in [(16, 16, 0), (12, 16, 16), (16, 12, 16), (8_388_488, 8, 8)]:
        with pytest.raises(flint_lib.FlintError):
            flint.gemm_tn(y, x, w, m, n, k)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_tn(None, x, w, 16, 16, 16)
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_tn(y.data_ptr() + 8, x, w, 16, 16, 16)
    # The failed calls wrote nothing.
    torch.cuda.synchronize()
    assert (y == 7).all()

    flint.gemm_tn(y, x, w, 16, 16, 16)
    flint.gemm_tn(y, x, w, 16, 16, 16, accumulate=True)
    torch.cuda.synchronize()
    assert (y[:256] == 32).all()
    assert (y[256:] == 7).all()

    # A workspace error, then success.
    buf, base, needed = workspace_problem(flint)
    m, n, k = WS_SHAPE
    with pytest.raises(flint_lib.FlintError):
        flint.gemm_tn(base + WS_Y, base + WS_X, base + WS_W, m, n, k)
    call_ws(flint, base, base, needed)
    torch.cuda.synchronize()
    assert (buf[WS_Y : WS_Y + m * n * 4].view(torch.float32) == k).all()
