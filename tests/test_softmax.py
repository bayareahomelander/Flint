"""flint_causal_softmax_forward: attention's softmax, P from S, both FP16 batches x length x
length, row-major and contiguous. Row t of a batch sees its columns 0 to t:
P[b, t, c] = exp(S[b, t, c] - max) / sum for c <= t (max and sum over the visible scores, in
FP32, each output rounded to FP16 once) and +0 for c > t, whatever S holds there. A visible
NaN or +inf, or a row of visible -inf, makes the row's visible outputs NaN; a -inf among
finite scores gives 0. The kernel is one warp per row and each lane takes every 32nd 16-byte
chunk of the row, so a max or a sum in one lane must reach the others through the shuffles.

The correctness gate runs on S ~ N(0, 1), on S from the batched NT GEMM (Q K^T / 8 with Q and
K ~ N(0, 1)), on wide inputs (N(0, 8^2) and N(0, 30^2), whose outputs underflow to subnormals
and 0), on outlier values near +-3,000 and on magnitudes near FP16's range mixed with
subnormals, at every length class (1 chunk, 4 lanes' worth, 1,016 and 1,024) and GPT-2's whole
192 x 1024 x 1024. The reference is the same softmax in PyTorch on the upcast scores (float64
up to 2^24 elements, float32 above), the peer PyTorch's FP16 softmax.

The gate's floors are nearly empty for a softmax (max|P| is about 1, so a P of 1e-3 could be
entirely wrong), so a tighter test follows: every visible output must be within 1 FP16 ulp
of the FP64 softmax rounded correctly to FP16 (P is non-negative, so the bits compared as
integers differ by the ulps), and at least 99.9% must be equal to that rounding. The exact
cases then check what a tolerance can't: the upper triangle is +0 by bits, poison (random
bit patterns, NaN and inf included) in the masked columns changes nothing, row 0 is 1.0,
equal scores give the correctly rounded 1/(t+1), a dominant score at every column (so the
max has to cross lanes) gives a one-hot row, +0 and -0 mix freely, and the same row in other
batches, other positions and longer matrices gives the same bits. Non-finite scores are
placed at columns reached by different lanes, each row's outputs checked against the
reference's NaN places, with the other rows bitwise those of the clean run. The rest covers
the guard bands and every element written, determinism (including rows with NaN), the limits
(65,535 batches, length 1,024), stream ordering, argument errors, and the chain S = Q K^T / 8,
P, O = P V against PyTorch's causal attention, with the wide scores where FP16 S puts it over
the gate.

Memory: GPT-2's whole matrix is 402 MB for S and for P, so the references and the errors run
over chunks of batches. Under the sanitizer (scripts/sanitize.py) the shapes shrink to T at
most 128 and few batches, keeping every tensor far below 300 MB, and the tests of the limits
are skipped."""

import gc
import math
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator, normal, round_to_half

MAX_LENGTH = 1024
MAX_BATCHES = 65_535
HEAD_WIDTH = 64

# The output of a guarded call sits `offset` halves past the start of a buffer that is filled
# with a pattern, which must be intact around P afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A
NAN_BITS = 0xFFFF
OTHER_BITS = 0x7B7B

# Tensors are compared and rounded over chunks of batches of at most this many elements.
CHUNK_ELEMENTS = 1 << 22
# The tighter bound's share of outputs equal to the correct rounding, asserted from this many
# visible non-zero outputs up (a few outputs can't say anything about a 1e-4 rate).
MIN_SHARE = 0.999
MIN_SHARE_COUNT = 20_000

# The longest rows, and the longest that aren't a power of 2 (127 chunks, one lane short of a
# fourth round).
LONG = 128 if SANITIZER else 1024
ODD_LONG = 120 if SANITIZER else 1016

# FP16 bit patterns.
NONFINITE_BITS = {"nan": 0x7E00, "-nan": 0xFFFF, "inf": 0x7C00, "-inf": 0xFC00}
ONE_BITS = 0x3C00


@pytest.fixture(autouse=True)
def free_gpu_memory():
    """The laptop's GPU is 6 GiB and shared, so each test hands back what it cached."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


def shape_id(shape):
    return "x".join(map(str, shape))


def shape_params(shapes):
    """pytest.params of (batches, length) shapes, with ids like 3x72."""
    return [pytest.param(*shape, id=shape_id(shape)) for shape in shapes]


# -- The references ---------------------------------------------------------


def causal_mask(t):
    """t x t, True where the column is visible to the row."""
    return torch.ones((t, t), dtype=torch.bool, device="cuda").tril()


def reference_dtype(numel):
    return torch.float64 if numel <= gate.FP64_REFERENCE_LIMIT else torch.float32


def reference(s, dtype=None):
    """The causal softmax of the upcast scores s (batches x t x t). Rows with a NaN come out
    NaN in the masked columns too, so callers compare the visible ones."""
    dtype = dtype or reference_dtype(s.numel())
    return F.softmax(s.to(dtype).masked_fill(~causal_mask(s.shape[-1]), -math.inf), dim=-1)


def peer(s):
    """PyTorch's FP16 softmax of the masked scores."""
    return F.softmax(s.masked_fill(~causal_mask(s.shape[-1]), -math.inf), dim=-1)


def batch_chunks(b, t):
    step = max(1, CHUNK_ELEMENTS // (t * t))
    return [slice(i, min(i + step, b)) for i in range(0, b, step)]


@dataclass
class Ulps:
    """How P compares with the FP64 softmax rounded to FP16, over the visible outputs
    (`worst` also covers the masked ones, which must be +0)."""

    visible: int = 0
    nonzero: int = 0  # visible outputs whose correct rounding isn't 0
    exact: int = 0  # of those, the ones that equal it
    worst: int = 0  # the most ulps from the correct rounding
    subnormal: int = 0  # correct roundings that are subnormal
    underflow: int = 0  # positive values that round to 0

    @property
    def share(self):
        return self.exact / self.nonzero if self.nonzero else 1.0


def ulp_diff(p, s):
    """Per element, the ulps between P's bits and the correct rounding of the float64
    softmax of s, with that softmax and its rounding. P's bits as integers differ by the
    ulps between them wherever both are non-negative; a NaN or a negative P differs by
    thousands."""
    ref = reference(s, torch.float64)
    ref_half = round_to_half(torch.nan_to_num(ref))
    diff = (p.view(torch.uint16).int() - ref_half.view(torch.uint16).int()).abs()
    return diff, ref, ref_half


def ulp_stats(p, s):
    b, t, _ = s.shape
    mask = causal_mask(t)
    total = Ulps()
    for chunk in batch_chunks(b, t):
        diff, ref, ref_half = ulp_diff(p[chunk], s[chunk])
        bits = ref_half.view(torch.int16)
        nonzero = mask & (bits != 0)
        total.visible += mask.sum().item() * diff.shape[0]
        total.nonzero += nonzero.sum().item()
        total.exact += (nonzero & (diff == 0)).sum().item()
        total.worst = max(total.worst, diff.max().item())
        total.subnormal += (mask & (bits > 0) & (bits < 0x0400)).sum().item()
        total.underflow += (mask & (ref > 0) & (bits == 0)).sum().item()
    return total


def assert_correctly_rounded(stats, what):
    assert stats.worst <= 1, f"{what}: {stats.worst} ulps from the correct rounding"
    if stats.nonzero >= MIN_SHARE_COUNT:
        assert stats.share >= MIN_SHARE, (
            f"{what}: {stats.share:.6f} of the {stats.nonzero} non-zero outputs are correctly "
            f"rounded, {stats.nonzero - stats.exact} are not"
        )


# -- Inputs -----------------------------------------------------------------


def fp16_from_bits(bits, like):
    """A tensor like `like` (FP16) with every element the 16-bit pattern."""
    return torch.full_like(like.view(torch.int16), as_int16(bits)).view(torch.float16)


def random_bits(shape, seed):
    """Random 16-bit patterns as FP16: NaNs, infs, subnormals and -0 among them."""
    values = torch.randint(
        -(2**15), 2**15, shape, dtype=torch.int16, generator=generator(seed), device="cuda"
    )
    return values.view(torch.float16)


def scores_from_qk(flint, b, t, g):
    """S = Q K^T / 8 as the batched NT GEMM makes it, with Q and K about N(0, 1)."""
    q = normal((b, t, HEAD_WIDTH), 1.0, g)
    k = normal((b, t, HEAD_WIDTH), 1.0, g)
    s = torch.empty((b, t, t), dtype=torch.float16, device="cuda")
    flint.gemm_nt_batched(s, q, k, b, t, t, HEAD_WIDTH, scale=0.125)
    return s


def with_outliers(shape, g):
    """N(0, 1) with 1% of the values replaced by +-3,000 (+-10%), as GPT-2's outlier
    channels reach a softmax."""
    s = torch.randn(shape, generator=g, device="cuda")
    hit = torch.rand(shape, generator=g, device="cuda") < 0.01
    sign = torch.randint(0, 2, shape, generator=g, device="cuda") * 2.0 - 1.0
    jitter = 0.9 + 0.2 * torch.rand(shape, generator=g, device="cuda")
    return torch.where(hit, 3000.0 * sign * jitter, s).half()


def near_the_range(shape, g):
    """Magnitudes near FP16's largest and smallest, mixed: 40% near +60,000 (on the 32-wide
    grid around it, so exact ties are common), 20% near -60,000, 20% subnormals of both signs
    and 20% N(0, 1)."""
    u = torch.rand(shape, generator=g, device="cuda")
    steps = torch.randint(-2, 3, shape, generator=g, device="cuda") * 32.0
    bits = torch.randint(1, 0x400, shape, generator=g, device="cuda")
    sign = torch.randint(0, 2, shape, generator=g, device="cuda")
    subnormal = (bits - sign * 0x8000).short().view(torch.float16)
    ordinary = torch.randn(shape, generator=g, device="cuda").half()
    s = torch.where(u < 0.4, (60000.0 + steps).half(), ordinary)
    s = torch.where((u >= 0.4) & (u < 0.6), (-60000.0 - steps).half(), s)
    return torch.where((u >= 0.6) & (u < 0.8), subnormal, s)


KINDS = ["normal", "qk", "wide8", "wide30", "outliers", "range"]


def make_scores(flint, kind, b, t, seed):
    g = generator(seed)
    shape = (b, t, t)
    if kind == "normal":
        return normal(shape, 1.0, g)
    if kind == "qk":
        return scores_from_qk(flint, b, t, g)
    if kind == "wide8":
        return normal(shape, 8.0, g)
    if kind == "wide30":
        return normal(shape, 30.0, g)
    if kind == "outliers":
        return with_outliers(shape, g)
    if kind == "range":
        return near_the_range(shape, g)
    raise ValueError(kind)


# -- Running Flint ----------------------------------------------------------


def assert_same_bits(actual, expected, what):
    """The two FP16 tensors are equal bit for bit (NaN payloads included)."""
    a, e = actual.view(torch.int16), expected.view(torch.int16)
    assert a.shape == e.shape, f"{what}: shapes {tuple(a.shape)} and {tuple(e.shape)}"
    if torch.equal(a, e):
        return
    bad = a != e
    first = bad.nonzero()[0].tolist()
    raise AssertionError(
        f"{what}: {bad.sum().item()} of {bad.numel()} halves differ, the first at {first}: "
        f"{actual[tuple(first)].item()} against {expected[tuple(first)].item()}"
    )


def all_equal(tensor, value):
    """Whether every element equals the value. The comparison is made out here because
    pytest, on a failed `assert tensor == value`, iterates the tensor on the host to explain
    it, which takes minutes and gigabytes for a big one."""
    return bool((tensor == value).all())


def none_equal(tensor, value):
    return not bool((tensor == value).any())


def any_unequal(a, b):
    return bool((a != b).any())


def flint_softmax(flint, s, offset=8, fill=SENTINEL, stream=None):
    """Runs the softmax into P inside a buffer pre-filled with `fill` (a 16-bit pattern),
    `offset` halves in (a multiple of 8 keeps P 16-byte aligned). Checks that the buffer around
    P is untouched, that S is unchanged and that no element of P still holds the fill, and
    returns P as batches x t x t FP16."""
    b, t, _ = s.shape
    count = s.numel()
    buf = torch.full((offset + count + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    p = buf[offset : offset + count]
    s_before = s.clone()

    flint.causal_softmax_forward(p, s, b, t, stream=stream)
    torch.cuda.synchronize()

    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert all_equal(buf[:offset], guard), "flint_causal_softmax_forward wrote before P"
    assert all_equal(buf[offset + count :], guard), "flint_causal_softmax_forward wrote past P"
    assert torch.equal(s.view(torch.int16), s_before.view(torch.int16)), "S changed"
    assert none_equal(p, guard), "an element of P was not written"
    return p.view(torch.float16).view(b, t, t)


def one_batch_at_a_time(flint, s):
    """The softmax of each batch of s on its own, stacked."""
    return torch.cat([flint_softmax(flint, s[j : j + 1].contiguous()) for j in range(s.shape[0])])


# -- The correctness gate ---------------------------------------------------


class Accumulator:
    """gate.errors over chunks: the sums and maxima that make up Errors, on the device."""

    def __init__(self):
        self.sums = torch.zeros(2, dtype=torch.float64, device="cuda")
        self.maxes = torch.zeros(2, dtype=torch.float64, device="cuda")

    def add(self, out, ref):
        d = out.double() - ref.double()
        r = ref.double()
        self.sums += torch.stack([d.square().sum(), r.square().sum()])
        self.maxes = torch.maximum(self.maxes, torch.stack([d.abs().max(), r.abs().max()]))

    def errors(self):
        sq_diff, sq_ref = self.sums.tolist()
        max_diff, max_ref = self.maxes.tolist()
        if sq_ref == 0.0:
            rel = 0.0 if sq_diff == 0.0 else math.inf
        else:
            rel = math.sqrt(sq_diff) / math.sqrt(sq_ref)
        return gate.Errors(rel, max_diff, max_ref)


def softmax_errors(p, s):
    """Flint's and the peer's gate.Errors against the reference, over chunks of batches."""
    dtype = reference_dtype(s.numel())
    flint_sums, peer_sums = Accumulator(), Accumulator()
    for chunk in batch_chunks(*s.shape[:2]):
        ref = reference(s[chunk], dtype)
        flint_sums.add(p[chunk], ref)
        peer_sums.add(peer(s[chunk]), ref)
    return flint_sums.errors(), peer_sums.errors()


def check_gate(flint, kind, b, t):
    flint_errors, peer_errors = [], []
    for seed in gate.SEEDS:
        s = make_scores(flint, kind, b, t, seed)
        p = flint_softmax(flint, s, offset=8 * (1 + seed))
        f, q = softmax_errors(p, s)
        flint_errors.append(f)
        peer_errors.append(q)
        del s, p
    bounds = gate.check_gate(gate.FORWARD_FACTOR, flint_errors, peer_errors)
    print(
        f"softmax {kind} {b}x{t}: flint rel_l2 {max(e.rel_l2 for e in flint_errors):.3g}, "
        f"peer {max(e.rel_l2 for e in peer_errors):.3g}, bound {bounds.rel_l2:.3g}"
    )


def gate_shapes():
    # Lengths of 1 chunk, partial lanes' worth, one more than a lane's, 4 chunks per lane at
    # 1,024, and the batch counts of GPT-2's attention (12 heads, 192 = 16 x 12).
    shapes = [
        (1, 8), (2, 16), (3, 24), (12, 32), (1, 40), (3, 56), (2, 64), (12, 72), (1, 128),
        (3, 256), (12, 512), (2, 1016), (3, 1024), (1, 1024), (12, 1024), (192, 64),
    ]  # fmt: skip
    if SANITIZER:
        shapes = [(min(b, 3), min(t, 128)) for b, t in shapes]
    seen = []
    for shape in shapes:
        if shape not in seen:
            seen.append(shape)
    return seen


# Each shape gets two kinds of input, rotating through the list, so every kind meets
# short rows, long rows and a partial last chunk.
GATE_CASES = [
    pytest.param(KINDS[(i + offset) % len(KINDS)], b, t, id=f"{KINDS[(i + offset) % len(KINDS)]}-{b}x{t}")
    for i, (b, t) in enumerate(gate_shapes())
    for offset in (0, 2)
]


@pytest.mark.parametrize("kind, b, t", GATE_CASES)
def test_passes_the_gate(flint, kind, b, t):
    check_gate(flint, kind, b, t)


@pytest.mark.parametrize("kind", ["qk", "normal"] if not SANITIZER else ["qk"])
def test_gpt2s_whole_attention_matrix_passes_the_gate(flint, kind):
    # 16 sequences x 12 heads = 192 batches of T = 1024: 402 MB each for S and P.
    check_gate(flint, kind, 3 if SANITIZER else 192, LONG)


# -- A tighter bound than the gate ------------------------------------------

TIGHT_SHAPES = [(3, 8), (2, 24), (5, 72), (4, 128), (12, 256), (3, 520), (2, ODD_LONG), (12, LONG)]
TIGHT_SHAPES = list(dict.fromkeys((min(b, 3) if SANITIZER else b, min(t, LONG)) for b, t in TIGHT_SHAPES))
TIGHT_CASES = [
    pytest.param(kind, b, t, id=f"{kind}-{b}x{t}") for kind in KINDS for b, t in TIGHT_SHAPES
]


@pytest.mark.parametrize("kind, b, t", TIGHT_CASES)
def test_every_output_is_within_an_ulp_of_the_correctly_rounded_softmax(flint, kind, b, t):
    s = make_scores(flint, kind, b, t, seed=0)

    p = flint_softmax(flint, s)
    stats = ulp_stats(p, s)

    print(
        f"softmax ulps {kind} {b}x{t}: worst {stats.worst}, share {stats.share:.6f} of "
        f"{stats.nonzero}, subnormal {stats.subnormal}, underflow {stats.underflow}"
    )
    assert_correctly_rounded(stats, f"{kind} {b}x{t}")
    if kind in ("wide8", "wide30") and t >= 128:
        # The wide inputs did what they are for: outputs in the subnormal range and past it.
        assert stats.subnormal > 10 and stats.underflow > 1000, stats


# -- Exact cases ------------------------------------------------------------

EXACT_SHAPES = [(2, 8), (3, 24), (2, 72), (2, ODD_LONG), (1, LONG)]


@pytest.mark.parametrize("b, t", shape_params(EXACT_SHAPES))
def test_the_upper_triangle_is_plus_zero_by_bits(flint, b, t):
    # Over scores that are wide, and over scores whose masked part is all junk.
    s = normal((b, t, t), 8.0, generator(0))
    junk = torch.where(causal_mask(t), s, random_bits((b, t, t), 1))

    for scores in (s, junk):
        p = flint_softmax(flint, scores, fill=NAN_BITS)
        assert all_equal(p.view(torch.int16)[:, ~causal_mask(t)], 0)
        # Not only +0: the visible part isn't zero (a row's max has exp 1).
        assert any_unequal(p.view(torch.int16)[:, causal_mask(t)], 0)


POISONS = ["random", "nan", "-nan", "inf", "-inf"]


@pytest.mark.parametrize("poison", POISONS)
@pytest.mark.parametrize("b, t", shape_params(EXACT_SHAPES))
def test_poison_in_the_masked_columns_changes_nothing(flint, poison, b, t):
    # Every row but the last has a chunk that is partly visible (t % 8 != 7 for most), so its
    # visible values share a 16-byte load with junk. The clean run has zeros there.
    s = normal((b, t, t), 4.0, generator(2))
    mask = causal_mask(t)
    if poison == "random":
        junk = random_bits((b, t, t), 3)
        # Random patterns are NaN 3% of the time but inf only twice in 65,536.
        junk.view(torch.int16)[..., 2::13] = as_int16(NONFINITE_BITS["inf"])
        junk.view(torch.int16)[..., 7::13] = as_int16(NONFINITE_BITS["-inf"])
        if t >= 72:
            assert torch.isnan(junk).any() and torch.isinf(junk).any()
    else:
        junk = fp16_from_bits(NONFINITE_BITS[poison], s)
    clean = torch.where(mask, s, torch.zeros_like(s))
    dirty = torch.where(mask, s, junk)

    expected = flint_softmax(flint, clean)
    p = flint_softmax(flint, dirty)

    assert_same_bits(p, expected, f"{poison} in the masked columns")
    assert torch.isfinite(p).all()


@pytest.mark.parametrize("b, t", shape_params([(8, 8), (8, 72), (8, LONG)]))
def test_row_0_is_exactly_1(flint, b, t):
    # The only visible score is its own max, whatever it is: finite scores of every kind,
    # including FP16's largest, a subnormal and -0.
    s = near_the_range((b, t, t), generator(5))
    special = [65504.0, -65504.0, 0.0, 1.0, -1.0, 6e-8, 6e-5]
    s[:, 0, 0] = torch.tensor(special + [0.0] * (b - len(special)), device="cuda").half()[:b]
    s.view(torch.int16)[7, 0, 0] = as_int16(0x8000)

    p = flint_softmax(flint, s)

    assert all_equal(p.view(torch.int16)[:, 0, 0], ONE_BITS)
    assert all_equal(p.view(torch.int16)[:, 0, 1:], 0)


@pytest.mark.parametrize("value", [0.0, 1.0, -3.5, 1000.0, -60000.0, 65504.0, 6e-8])
def test_equal_scores_give_the_correctly_rounded_1_over_t_plus_1(flint, value):
    t = LONG
    s = torch.full((2, t, t), value, dtype=torch.float16, device="cuda")

    p = flint_softmax(flint, s)

    n = torch.arange(1, t + 1, dtype=torch.float64, device="cuda")
    column = round_to_half(1.0 / n)
    expected = torch.where(causal_mask(t), column[:, None].expand(t, t), torch.zeros((), dtype=torch.float16, device="cuda"))
    wrong = (p.view(torch.int16) != expected.view(torch.int16)).any(dim=-1)
    assert not wrong.any(), f"t = {wrong.nonzero()[:, 1].unique().tolist()} differ from 1 / (t + 1)"


DOMINANT = 20_000.0


@pytest.mark.parametrize("t", [8, 24, 72, ODD_LONG, LONG])
def test_a_dominant_score_gives_a_one_hot_row_wherever_it_is(flint, t):
    # Batch c has a score of 20,000 at column c of every row from c down (the others N(0, 8^2),
    # whose exp against it is 0), so every (row, column) pair of the matrix gets a turn, and
    # the lane that holds the max is every lane in turn, and the other lanes' maxima are
    # always smaller. Rows above c have no dominant score: they must equal the base run.
    base = normal((1, t, t), 8.0, generator(0))
    base_p = flint_softmax(flint, base)
    rows = torch.arange(t, device="cuda")[:, None]
    columns = torch.arange(t, device="cuda")[None, :]
    step = max(1, min(t, CHUNK_ELEMENTS * 16 // (t * t)))
    for first in range(0, t, step):
        c = torch.arange(first, min(first + step, t), device="cuda")[:, None, None]
        here = (columns == c) & (rows >= c)
        s = torch.where(here, DOMINANT, base.expand(len(c), t, t))

        p = flint_softmax(flint, s)

        expected = torch.where(rows >= c, here.half(), base_p.expand(len(c), t, t))
        assert_same_bits(p, expected, f"dominant score at column {first}..{first + len(c) - 1}")


@pytest.mark.parametrize("b, t", shape_params([(4, 72), (3, ODD_LONG)]))
def test_plus_and_minus_zero_scores_mix_without_changing_the_result(flint, b, t):
    g = generator(7)
    s = normal((b, t, t), 1.0, g)
    s = torch.where(torch.rand(s.shape, generator=g, device="cuda") < 0.4, torch.zeros_like(s), s)
    zero = s.view(torch.int16) == 0
    flip = zero & (torch.rand(s.shape, generator=g, device="cuda") < 0.5)
    mixed = s.view(torch.int16).clone()
    mixed[flip] = as_int16(0x8000)
    assert flip.any() and (zero & ~flip).any()

    plain = flint_softmax(flint, s)
    p = flint_softmax(flint, mixed.view(torch.float16))

    assert_same_bits(p, plain, "-0 for +0 in some scores")
    # All -0, all +0 and alternating: equal scores either way.
    ones = torch.zeros((1, t, t), dtype=torch.float16, device="cuda")
    alternating = ones.view(torch.int16).clone()
    alternating[..., ::2] = as_int16(0x8000)
    alternating_p = flint_softmax(flint, alternating.view(torch.float16))
    negative_p = flint_softmax(flint, fp16_from_bits(0x8000, ones))
    assert_same_bits(negative_p, flint_softmax(flint, ones), "all -0")
    assert_same_bits(alternating_p, flint_softmax(flint, ones), "alternating +0 and -0")


@pytest.mark.parametrize("t", [8, 72, ODD_LONG])
def test_the_same_visible_scores_in_other_batches_give_the_same_bits(flint, t):
    # Every batch sees the same visible scores, with different junk in the masked columns:
    # the rows of different batches sit at different places in memory and on different blocks.
    b = 11
    base = normal((1, t, t), 3.0, generator(0))
    junk = random_bits((b, t, t), 1)
    s = torch.where(causal_mask(t), base, junk)

    p = flint_softmax(flint, s)

    alone = flint_softmax(flint, base)
    for j in range(b):
        assert_same_bits(p[j : j + 1], alone, f"batch {j} against the batch alone")


EMBEDDED = [(8, 64), (8, LONG), (64, LONG), (72, 128), (ODD_LONG, LONG)]


@pytest.mark.parametrize("small, big", EMBEDDED, ids=lambda v: str(v))
def test_a_row_does_not_depend_on_the_length_of_the_matrix(flint, small, big):
    # A small matrix sits in the top left corner of a bigger one (the rest random scores): the
    # rows of the corner give the same bits, and the rest of their columns are +0.
    b = 3
    s_small = normal((b, small, small), 3.0, generator(0))
    s_big = normal((b, big, big), 3.0, generator(1))
    s_big[:, :small, :small] = s_small

    p_small = flint_softmax(flint, s_small)
    p_big = flint_softmax(flint, s_big)

    assert_same_bits(p_big[:, :small, :small], p_small, f"the corner of a {big} matrix")
    assert all_equal(p_big.view(torch.int16)[:, :small, small:], 0)


@pytest.mark.parametrize("b, t", shape_params([(5, 72), (4, ODD_LONG), (3, LONG)]))
def test_a_batch_alone_is_the_same_as_in_many(flint, b, t):
    s = normal((b, t, t), 6.0, generator(4))

    many = flint_softmax(flint, s)

    for j in range(b):
        one = flint_softmax(flint, s[j : j + 1].contiguous())
        assert_same_bits(many[j : j + 1], one, f"batch {j} in {b} against alone")


@pytest.mark.parametrize("t", [24, ODD_LONG])
def test_permuting_the_batches_permutes_the_results(flint, t):
    b = 7
    s = normal((b, t, t), 6.0, generator(5))
    perm = torch.randperm(b, generator=generator(1), device="cuda")
    assert any_unequal(perm, torch.arange(b, device="cuda"))

    p = flint_softmax(flint, s)
    permuted = flint_softmax(flint, s[perm].contiguous())

    assert_same_bits(permuted, p[perm], "permuted batches")


# -- Non-finite scores ------------------------------------------------------


def injection_column(variant, t):
    """The visible column of row t (a tensor of rows) that variant v damages: the first, the
    last, a chunk boundary mid-row, a spread of other columns, and a column of the chunk before
    the last. Together with the rows, they land on different lanes."""
    if variant == 0:
        return torch.zeros_like(t)
    if variant == 1:
        return t
    if variant == 2:
        return (t // 2) // 8 * 8
    if variant == 3:
        return (t * 5 + 3) % (t + 1)
    return torch.clamp(8 * (t // 8 - 1) + 5, min=0)


@pytest.mark.parametrize("t", [72, ODD_LONG, LONG])
def test_non_finite_scores_spoil_only_their_rows_visible_outputs(flint, t):
    # Every third row has one NaN or inf of each kind at five different columns (one batch per
    # kind and column); the rows between are controls. A NaN or +inf makes all of its row's
    # visible outputs NaN, like the reference; -inf among finite scores gives 0 there and
    # leaves the rest within an ulp of the softmax of the others; masked columns stay +0; and
    # every other row is bitwise the clean run's.
    base = normal((1, t, t), 2.0, generator(0))
    clean_p = flint_softmax(flint, base)
    mask = causal_mask(t)
    injected = torch.arange(0, t, 3, device="cuda")
    is_injected = torch.zeros(t, dtype=torch.bool, device="cuda")
    is_injected[injected] = True
    cases = [(kind, v) for kind in NONFINITE_BITS for v in range(5)]
    s = base.repeat(len(cases), 1, 1)
    for i, (kind, v) in enumerate(cases):
        s.view(torch.int16)[i, injected, injection_column(v, injected)] = as_int16(NONFINITE_BITS[kind])

    p = flint_softmax(flint, s)

    for i, (kind, v) in enumerate(cases):
        what = f"{kind} at variant {v}"
        pi = p[i]
        ref = reference(s[i : i + 1], torch.float32)[0]
        # Not NaN on the control rows; NaN across the visible part of the injected rows, but
        # for -inf, whose row is all -inf only when it has one visible column.
        if kind == "-inf":
            expected_nan = mask & (torch.arange(t, device="cuda") == 0)[:, None]
        else:
            expected_nan = mask & is_injected[:, None]
        assert torch.equal(torch.isnan(ref) & mask, expected_nan), f"{what}: the reference"
        assert torch.equal(torch.isnan(pi) & mask, expected_nan), f"{what}: NaN places"
        assert all_equal(pi.view(torch.int16)[~mask], 0), f"{what}: the masked columns"
        assert_same_bits(pi[~is_injected], clean_p[0][~is_injected], f"{what}: the other rows")
        if kind == "-inf":
            c = injection_column(v, injected)
            assert all_equal(pi.view(torch.int16)[injected[1:], c[1:]], 0), f"{what}: -inf's output"
            diff, _, _ = ulp_diff(p[i : i + 1], s[i : i + 1])
            valid = mask & ~expected_nan
            assert diff[0][valid].max().item() <= 1, f"{what}: the rest of the row"


@pytest.mark.parametrize("t", [72, ODD_LONG])
def test_rows_of_only_minus_inf_are_nan_and_one_finite_score_takes_everything(flint, t):
    # Rows t % 5 == 2 see only -inf (NaN); rows t % 5 == 3 see -inf but at one column, and the
    # finite one gets exactly 1.0 with +0 elsewhere. The masked columns are finite.
    base = normal((1, t, t), 2.0, generator(1))
    clean_p = flint_softmax(flint, base)
    mask = causal_mask(t)
    s = base.clone()
    rows = torch.arange(t, device="cuda")
    minus_inf = fp16_from_bits(NONFINITE_BITS["-inf"], base)
    all_inf = (rows % 5 == 2)[None, :, None] & mask
    one_finite = (rows % 5 == 3)[None, :, None] & mask & (torch.arange(t, device="cuda")[None, None, :] != (rows // 2)[None, :, None])
    s = torch.where(all_inf | one_finite, minus_inf, s)

    p = flint_softmax(flint, s)[0]

    all_inf_rows = rows % 5 == 2
    one_finite_rows = rows % 5 == 3
    assert torch.isnan(p[all_inf_rows][mask[all_inf_rows]]).all()
    assert all_equal(p.view(torch.int16)[~mask], 0)
    expected = torch.zeros((t, t), dtype=torch.float16, device="cuda")
    expected[rows, rows // 2] = 1.0
    assert_same_bits(p[one_finite_rows], expected[one_finite_rows], "one finite score")
    other = ~(all_inf_rows | one_finite_rows)
    assert_same_bits(p[other], clean_p[0][other], "the other rows")


@pytest.mark.parametrize("first, second", [("inf", "-inf"), ("nan", "inf"), ("nan", "-inf")])
def test_two_kinds_of_non_finite_score_in_a_row_give_nan(flint, first, second):
    # Every third row from row 3 has one score of each kind, at its last column and halfway
    # along, which long rows put on different lanes. All of its visible outputs are NaN, like
    # the reference's, and the rows between are bitwise the clean run's.
    t = ODD_LONG
    base = normal((1, t, t), 2.0, generator(2))
    clean_p = flint_softmax(flint, base)
    mask = causal_mask(t)
    injected = torch.arange(3, t, 3, device="cuda")
    is_injected = torch.zeros(t, dtype=torch.bool, device="cuda")
    is_injected[injected] = True
    s = base.clone()
    s.view(torch.int16)[0, injected, injected // 2] = as_int16(NONFINITE_BITS[first])
    s.view(torch.int16)[0, injected, injected] = as_int16(NONFINITE_BITS[second])

    p = flint_softmax(flint, s)[0]

    expected_nan = mask & is_injected[:, None]
    assert torch.equal(torch.isnan(reference(s, torch.float32)[0]) & mask, expected_nan)
    assert torch.equal(torch.isnan(p) & mask, expected_nan)
    assert all_equal(p.view(torch.int16)[~mask], 0)
    assert_same_bits(p[~is_injected], clean_p[0][~is_injected], "the other rows")


@pytest.mark.parametrize("kind", ["nan", "inf"])
def test_non_finite_scores_in_one_batch_leave_the_others_alone(flint, kind):
    t = ODD_LONG
    s = normal((4, t, t), 3.0, generator(8))
    clean = flint_softmax(flint, s)
    dirty = s.clone()
    dirty.view(torch.int16)[2, t - 1, 700 % t] = as_int16(NONFINITE_BITS[kind])
    dirty.view(torch.int16)[2, 5, 3] = as_int16(NONFINITE_BITS[kind])

    p = flint_softmax(flint, dirty)

    assert torch.isnan(p[2, t - 1, : t]).all() and torch.isnan(p[2, 5, :6]).all()
    keep = torch.ones((4, t), dtype=torch.bool, device="cuda")
    keep[2, [t - 1, 5]] = False
    assert_same_bits(p[keep], clean[keep], "the other rows")


# -- Memory -----------------------------------------------------------------

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
    # flint_softmax checks the guard bands, that S is unchanged and that no element of P is
    # left holding the fill (the upper triangle's +0 included, as no fill is 0).
    s = normal((b, t, t), 3.0, generator(0))

    p = flint_softmax(flint, s, offset=offset, fill=fill)

    assert torch.isfinite(p).all()
    assert_same_bits(p, flint_softmax(flint, s, fill=SENTINEL), "against a run into another fill")


# -- Determinism ------------------------------------------------------------


@pytest.mark.parametrize("kind, b, t", [("outliers", 3, ODD_LONG), ("wide30", 4, 72), ("range", 2, 24)], ids=str)
def test_run_twice_is_bitwise_identical_including_nan_rows(flint, kind, b, t):
    s = make_scores(flint, kind, b, t, seed=0)
    # NaN and inf scores in a third of the rows (some visible, so those rows are NaN).
    rows = torch.arange(0, t, 3, device="cuda")
    s.view(torch.int16)[0, rows, rows // 2] = as_int16(NONFINITE_BITS["nan"])
    s.view(torch.int16)[1, rows, rows] = as_int16(NONFINITE_BITS["inf"])
    s.view(torch.int16)[-1, rows[1::2], 0] = as_int16(NONFINITE_BITS["-inf"])

    first = flint_softmax(flint, s, fill=NAN_BITS)
    second = flint_softmax(flint, s, fill=OTHER_BITS)
    third = flint_softmax(flint, s, fill=SENTINEL)

    assert torch.isnan(first).any()
    assert_same_bits(second, first, "second run")
    assert_same_bits(third, first, "third run")


# -- The limits -------------------------------------------------------------


@pytest.mark.skipif(SANITIZER, reason="65,535 blocks are too slow under compute-sanitizer")
def test_65535_batches_of_length_8(flint):
    # Every batch has its own random scores, so a block that read another batch's would be
    # far off; the whole result is checked against the float64 softmax, and a sample of
    # batches (the first, the last, edges of powers of 2) bitwise against one-batch calls.
    b, t = MAX_BATCHES, 8
    s = normal((b, t, t), 3.0, generator(0))

    p = flint_softmax(flint, s, fill=NAN_BITS)

    assert_correctly_rounded(ulp_stats(p, s), "65,535 batches of 8")
    for j in [0, 1, 2, 255, 256, 4095, 4096, 32767, 32768, 50000, 65533, MAX_BATCHES - 1]:
        one = flint_softmax(flint, s[j : j + 1].contiguous())
        assert_same_bits(p[j : j + 1], one, f"batch {j}")


@pytest.mark.skipif(SANITIZER, reason="length 1,024 is too slow under compute-sanitizer")
def test_length_1024(flint):
    b, t = 7, MAX_LENGTH
    s = normal((b, t, t), 3.0, generator(0))

    p = flint_softmax(flint, s)

    assert_correctly_rounded(ulp_stats(p, s), "length 1,024")
    for j in range(b):
        assert_same_bits(p[j : j + 1], flint_softmax(flint, s[j : j + 1].contiguous()), f"batch {j}")


# -- Streams ----------------------------------------------------------------

STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
def test_runs_on_the_given_stream(flint, mode):
    b, t = 3, 64
    s = torch.full((b, t, t), math.nan, dtype=torch.float16, device="cuda")
    p = torch.zeros((b, t, t), dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fill finishes only after the sleep, so P is the softmax of zeros only if the kernel is
    # queued on the same stream behind it (it is NaN where it ran first). With the current
    # stream the call has no stream argument, and torch's current stream is the side one; with
    # the default stream there is no side one at all.
    if mode == "default-stream":
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        s.fill_(0)
        flint.causal_softmax_forward(p, s, b, t)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            s.fill_(0)
            flint.causal_softmax_forward(p, s, b, t, stream=stream if mode == "explicit-stream" else None)
        stream.synchronize()

    assert not torch.isnan(p).any()
    assert_correctly_rounded(ulp_stats(p, s), mode)


# -- Argument errors --------------------------------------------------------

NAMES = ("p", "s")
SMALL = dict(batches=2, length=16)
BYTES = 2 * 16 * 16 * 2


def small_problem():
    """Valid tensors by name, in buffers with room to spare, p filled with a sentinel."""
    return {
        "p": torch.full((2 * 16 * 16 + 64,), 0.5, dtype=torch.float16, device="cuda"),
        "s": torch.zeros(2 * 16 * 16 + 64, dtype=torch.float16, device="cuda"),
    }


def call(flint, buffers, **sizes):
    sizes = {**SMALL, **sizes}
    flint.causal_softmax_forward(buffers["p"], buffers["s"], sizes["batches"], sizes["length"])


def expect_invalid_argument(flint, reason, buffers, **sizes):
    """The call fails with the status and message, and no tensor in the buffers changed."""
    before = {n: b.clone() for n, b in buffers.items() if isinstance(b, torch.Tensor)}
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, buffers, **sizes)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_causal_softmax_forward", reason):
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
    # Per buffer: null, then alignment, then the address space; p before s.
    a = {n: t.data_ptr() for n, t in small_problem().items()}
    expect_invalid_argument(flint, "p is not 16-byte aligned", {**a, "p": a["p"] + 2, "s": None})
    expect_invalid_argument(flint, "p is null", {**a, "p": None, "s": a["s"] + 2})
    expect_invalid_argument(flint, "s is not 16-byte aligned", {**a, "s": a["s"] + 2})
    expect_invalid_argument(flint, "p runs past the end", {"p": 2**64 - 16, "s": None})


@pytest.mark.parametrize(
    "length", [1, 4, 7, 9, 12, 20, 1020, 1032, 1040, 2048, 2**32, 2**64 - 8, 2**64 - 1], ids=str
)
def test_length_must_be_a_multiple_of_8_up_to_1024(flint, length):
    expect_invalid_argument(
        flint, f"length ({length}) must be a multiple of 8 up to 1024", small_problem(), length=length
    )


@pytest.mark.parametrize("batches", [65_536, 65_537, 2**32, 2**64 - 1], ids=str)
def test_batches_can_be_up_to_65535(flint, batches):
    expect_invalid_argument(
        flint, f"batches ({batches}) can be up to 65535", small_problem(), batches=batches
    )


@pytest.mark.parametrize(
    "sizes",
    [dict(batches=MAX_BATCHES, length=MAX_LENGTH), dict(batches=MAX_BATCHES), dict(length=MAX_LENGTH),
     dict(batches=1, length=MAX_LENGTH), dict(batches=1, length=8)],
    ids=lambda sizes: "-".join(f"{k}={v}" for k, v in sizes.items()),
)
def test_the_largest_sizes_pass_the_size_checks(flint, sizes):
    # The limits themselves aren't errors of size: the call gets on to the pointers (a null
    # p), even where batches * length^2 * 2 is 1.4e11 bytes.
    expect_invalid_argument(flint, "p is null", {**small_problem(), "p": None}, **sizes)


def test_size_limits_are_checked_in_order(flint):
    # length, then batches, then the pointers (null ones included); the empty-problem exit
    # comes after the size checks.
    expect_invalid_argument(flint, "length (7) must be", nothing(), length=7, batches=2**40)
    expect_invalid_argument(flint, "length (7) must be", nothing(), length=7, batches=0)
    expect_invalid_argument(flint, "batches (65536) can be up to", nothing(), batches=65_536)
    expect_invalid_argument(flint, "batches (65536) can be up to", nothing(), batches=65_536, length=0)
    expect_invalid_argument(flint, "length (1032) must be", nothing(), length=1032, batches=0)


SPACING = 4096

# How far s starts after p: at the same address, inside p, 16 bytes before p's end, at p's
# second batch, and (negative) with s ending 16 bytes into p. Just after p's end, and just
# before its start, s touches p but doesn't overlap it.
OVERLAPPING = {
    "same-start": 0,
    "s-inside-p": 16,
    "s-at-second-batch": BYTES // 2,
    "s-at-end-of-p": BYTES - 16,
    "p-inside-s": 16 - BYTES,
    "p-ends-in-s-second-batch": BYTES // 2 - BYTES + 16,
}
TOUCHING = {"s-after-p": BYTES, "s-before-p": -BYTES}


@pytest.mark.parametrize("label", OVERLAPPING)
def test_p_may_not_overlap_s(flint, label):
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no
    # harm to anything else.
    arena = torch.zeros(8 * SPACING, dtype=torch.uint8, device="cuda")
    base = arena.data_ptr() + 2 * SPACING
    addresses = {"p": base, "s": base + OVERLAPPING[label]}
    before = arena.clone()
    expect_invalid_argument(flint, "p and s overlap", addresses)
    assert torch.equal(arena, before)


@pytest.mark.parametrize("label", TOUCHING)
def test_buffers_next_to_each_other_are_not_overlapping(flint, label):
    arena = torch.full((8 * SPACING,), 0x5A, dtype=torch.uint8, device="cuda")
    p_at = 2 * SPACING
    s_at = p_at + TOUCHING[label]
    p = arena[p_at : p_at + BYTES].view(torch.float16)
    s = arena[s_at : s_at + BYTES].view(torch.float16)
    s.copy_(normal((BYTES // 2,), 1.0, generator(0)))
    s_before = s.clone()
    outside = torch.ones(arena.numel(), dtype=torch.bool, device="cuda")
    outside[p_at : p_at + BYTES] = False
    others = arena[outside].clone()

    flint.causal_softmax_forward(p, s, 2, 16)
    torch.cuda.synchronize()

    assert torch.equal(s.view(torch.int16), s_before.view(torch.int16))
    assert torch.equal(arena[outside], others)
    assert none_equal(p.view(torch.int16), as_int16(0x5A5A))
    assert all_equal(p.view(torch.int16).view(2, 16, 16)[:, 0, 0], ONE_BITS)


@pytest.mark.parametrize(
    "batches, length",
    [(0, 16), (2, 0), (0, 0), (0, 1024), (MAX_BATCHES, 0), (0, 8)],
    ids=lambda v: str(v),
)
def test_empty_problems_do_nothing(flint, batches, length):
    # Not even the null pointers are looked at.
    buffers = small_problem()

    flint.causal_softmax_forward(None, None, batches, length)
    call(flint, buffers, batches=batches, length=length)
    torch.cuda.synchronize()

    assert all_equal(buffers["p"], 0.5)


def test_a_valid_call_works_after_errors(flint):
    buffers = small_problem()
    for sizes in (dict(length=7), dict(length=1032), dict(batches=65_536), dict(length=2**64 - 1)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, buffers, **sizes)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "p": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "s": buffers["s"].data_ptr() + 2})
    torch.cuda.synchronize()
    assert all_equal(buffers["p"], 0.5)

    call(flint, buffers)
    torch.cuda.synchronize()

    # All scores are 0: row t is 1 / (t + 1) across its visible columns.
    p = buffers["p"][: 2 * 16 * 16].view(2, 16, 16)
    s = buffers["s"][: 2 * 16 * 16].view(2, 16, 16)
    assert_correctly_rounded(ulp_stats(p, s), "after errors")
    assert all_equal(buffers["p"][2 * 16 * 16 :], 0.5)


# -- The chain: S = Q K^T / 8, P, O = P V -------------------------------------


def make_qkv(b, t, seed, std):
    g = generator(seed)
    return tuple(normal((b, t, HEAD_WIDTH), std, g) for _ in range(3))


def flint_attention(flint, q, k, v):
    b, t, _ = q.shape
    s = torch.empty((b, t, t), dtype=torch.float16, device="cuda")
    o = torch.empty((b, t, HEAD_WIDTH), dtype=torch.float16, device="cuda")
    flint.gemm_nt_batched(s, q, k, b, t, t, HEAD_WIDTH, scale=0.125)
    p = flint_softmax(flint, s)
    flint.gemm_nn_batched(o, p, v, b, t, HEAD_WIDTH, t)
    torch.cuda.synchronize()
    return o


def pytorch_attention(q, k, v, dtype=None):
    """PyTorch's causal attention, on the FP16 tensors or on their upcast copies."""
    if dtype is not None:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    with gate.peer_math():
        out = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
        )
    return out.squeeze(1)


CHAIN_CASES = (
    [(3, 64, 1.0), (3, 128, 1.0)]
    if SANITIZER
    else [(12, 64, 1.0), (12, 256, 1.0), (24, 1024, 1.0)]
)


@pytest.mark.parametrize("b, t, std", CHAIN_CASES, ids=lambda v: str(v))
def test_the_chain_passes_the_gate_against_pytorch_attention(flint, b, t, std):
    # Q, K and V in the heads layout, B x T x 64, about N(0, std^2): at std 1 the scores
    # are about N(0, 1). Wider scores go over the gate, as the next test pins.
    dtype = torch.float64 if b * t * t <= gate.FP64_REFERENCE_LIMIT else torch.float32
    flint_errors, peer_errors = [], []
    for seed in gate.SEEDS:
        q, k, v = make_qkv(b, t, seed, std)
        o = flint_attention(flint, q, k, v)
        ref = pytorch_attention(q, k, v, dtype)
        flint_errors.append(gate.errors(o, ref))
        peer_errors.append(gate.errors(pytorch_attention(q, k, v), ref))
    bounds = gate.check_gate(gate.FORWARD_FACTOR, flint_errors, peer_errors)
    print(
        f"softmax chain {b}x{t} std {std}: flint rel_l2 {max(e.rel_l2 for e in flint_errors):.3g}, "
        f"max_abs {max(e.max_abs for e in flint_errors):.3g}; peer rel_l2 "
        f"{max(e.rel_l2 for e in peer_errors):.3g}, max_abs {max(e.max_abs for e in peer_errors):.3g}; "
        f"bound {bounds.rel_l2:.3g}"
    )


@pytest.mark.skipif(SANITIZER, reason="the chain above runs these kernels under the sanitizer")
@pytest.mark.parametrize(
    "std, rel_l2_limit, max_abs_limit", [(2.0, 1.3e-3, 0.03), (4.0, 2.5e-3, 0.25)], ids=lambda v: str(v)
)
def test_fp16_scores_put_the_chain_over_the_gate_at_wide_scores(flint, std, rel_l2_limit, max_abs_limit):
    # A known cost, pinned so it can't grow unnoticed. M2's attention stores S in FP16 between
    # the GEMM and the softmax, where PyTorch's fused attention (the peer) keeps it in FP32. Half
    # an ulp of S is an absolute error in S and so a relative error in P, up to 0.8% at |S| from
    # 16 to 32. With Q and K about N(0, std^2), S's std is about std^2. The same chain with S and
    # the softmax in FP32 passes, so S's storage is the cause. M2's done rule gates the whole
    # model, not this chain alone.
    b, t = 12, 256
    mask = torch.ones(t, t, device="cuda").tril().bool()
    flint_errors, fp32_errors, peer_errors = [], [], []
    for seed in gate.SEEDS:
        q, k, v = make_qkv(b, t, seed, std)
        ref = pytorch_attention(q, k, v, torch.float64)
        with gate.peer_math():
            s = q.float() @ k.float().transpose(1, 2) / 8
            p = F.softmax(s.masked_fill(~mask, -math.inf), dim=-1).half()
        o = torch.empty((b, t, HEAD_WIDTH), dtype=torch.float16, device="cuda")
        flint.gemm_nn_batched(o, p, v, b, t, HEAD_WIDTH, t)
        torch.cuda.synchronize()
        fp32_errors.append(gate.errors(o, ref))
        flint_errors.append(gate.errors(flint_attention(flint, q, k, v), ref))
        peer_errors.append(gate.errors(pytorch_attention(q, k, v), ref))
    bounds = gate.check_gate(gate.FORWARD_FACTOR, fp32_errors, peer_errors)
    with pytest.raises(AssertionError):
        gate.check_gate(gate.FORWARD_FACTOR, flint_errors, peer_errors)
    print(
        f"FP16 S at std {std}: flint rel_l2 {max(e.rel_l2 for e in flint_errors):.3g}, max_abs "
        f"{max(e.max_abs for e in flint_errors):.3g}; FP32 S rel_l2 "
        f"{max(e.rel_l2 for e in fp32_errors):.3g}; peer rel_l2 "
        f"{max(e.rel_l2 for e in peer_errors):.3g}; bound {bounds.rel_l2:.3g}"
    )
    # Per seed, so that a NaN fails.
    assert all(e.rel_l2 <= rel_l2_limit and e.max_abs <= max_abs_limit for e in flint_errors)
