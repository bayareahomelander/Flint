"""flint_softmax_cross_entropy: the output layer's softmax and cross-entropy in one kernel.
FP16 logits and dlogits are rows x cols, row-major and contiguous, targets are rows uint16 and
row_loss rows FP32. Over a row's first vocab columns, in FP32, with max the largest logit and
sum adding exp(logit - max): row_loss[r] = log(sum) - (logits[r, target] - max), and
dlogits[r, c] = scale * (p[r, c] - (c == target)) with p = exp(logit - max) / sum, each
rounded to FP16 once; dlogits is +0 for c at or above vocab, whose logits are ignored even
when they hold a NaN or an inf. The kernel is one warp per row and each lane takes every 32nd
16-byte chunk of the row, so a max or a sum in one lane must reach the others through the
shuffles, and the lane that holds the target writes the row's loss.

PyTorch's own path is the peer, and it is two peers. Under torch.autocast, F.cross_entropy
with reduction="none" runs its log_softmax in FP16 and only then upcasts (its loss is bitwise
the FP16 log_softmax at the target, upcast and negated, as the first test pins), so autocast's
row loss has FP16-level error (a relative L2 of about 2e-4, against FP32's 3e-8): a gate
against it would be empty for the loss. The loss is therefore gated against the FP32
F.cross_entropy, and the gradient against autograd through the autocast cross-entropy (an FP16
leaf, (loss.sum() * scale).backward(), the leaf's FP16 grad), which is what a model trained
with autocast gets. The reference is the FP64 cross-entropy and softmax on the upcast logits,
over chunks of rows so that it fits in memory.

The correctness gate runs over 3 seeds on shapes from one row of one class to GPT-2's
(1024 rows of 50,257 classes in 50,304 columns), the widest (65,536) and GPT-2's whole
micro-batch of 16,384 rows (1.65 GB each for logits and dlogits), on init-like logits
(N(0, 0.3^2), loss near log(vocab)), real-like ones (what OpenAI's GPT-2 weights give: a
per-row offset around -110, a within-row spread, and a spike in every 8th row that makes its
top p near 1; also with each target on its row's top logit, as real targets often are), wide
logits (N(0, 8^2) and N(0, 30^2)), outliers near +-3,000, and magnitudes near FP16's range
mixed with subnormals. Scales are 1, 1 / rows (a mean loss), 0.125 (M2's starting loss scale
over its tokens) and 2^-19, the last three on the init-like and real-like inputs; row_loss
doesn't depend on scale, so its gate runs once per input.

The gate's floors are nearly empty for this (max|dlogits| is about 1, and most values are
about 1e-5), so a tighter test follows: every dlogits value below vocab must be within 1 FP16
ulp of the FP64 reference correctly rounded to FP16 (ulps counted as a difference of
sign-magnitude ordinals), with at least 99.9% equal to it. The target column is different near
p = 1: p - 1 cancels in FP32, as in PyTorch, so it is checked within 1 ulp while 1 - p >=
1e-3 and, in all rows, by an absolute bound of 2^-23 * |scale| plus half an FP16 ulp, with a
sweep of 1 - p from 1e-1 to 1e-7. row_loss is checked per row against the FP64 reference in
FP32 ulps.

The exact cases check what a tolerance can't: dlogits past vocab are +0 by bits and poison there
(random patterns, NaN and inf) changes no output bit; every target position (each lane,
the chunk boundaries); every element and nothing else written; equal logits; vocab = 1; a
dominant target (loss exactly 0) and a far lower one (the loss stays finite, which -log(p)
wouldn't); a dominant non-target at every column; row independence; scale's effects (none on
row_loss, 0, negative, overflow). Non-finite logits are placed at columns reached by different
lanes: a NaN or +inf makes its row's loss and gradients NaN at any scale and spoils no other
row, a -inf among finite logits gets p = 0 (and a loss of +inf at the target), and a row of
only -inf or two kinds in a row is NaN. A target at or above vocab touches nothing outside the
buffers. The rest covers determinism, guard bands, streams, and the argument errors.

Memory: GPT-2's whole micro-batch is 1.65 GB each for logits and dlogits, so the logits are made
and the errors computed over chunks of rows. Under the sanitizer (scripts/sanitize.py) the
shapes shrink to a few rows, keeping every tensor far below 300 MB, and the tests of the
limits are skipped."""

import gc
import math
from collections import namedtuple
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import flint_lib
import gate
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator, normal, round_to_half

MAX_COLS = 65_536
MAX_ROWS = 0x7FFFFFFF * 8  # 17,179,869,176

# The outputs of a guarded call sit `offset` elements past the start of a buffer that is
# filled with a pattern, which must be intact around them afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A
NAN_BITS = 0xFFFF
OTHER_BITS = 0x7B7B

# Tensors are compared and rounded over chunks of rows of at most this many elements.
CHUNK_ELEMENTS = 1 << 22
# The tight bound's share of outputs equal to the correct rounding, asserted from this many
# non-target non-zero outputs up (a few outputs can't say anything about a 1e-4 rate).
MIN_SHARE = 0.999
MIN_SHARE_COUNT = 20_000
# The target column is held to 1 ulp while 1 - p is at least this.
TARGET_ULP_MIN_ONE_MINUS_P = 1e-3
# The absolute error of the target column, in units of |scale|: p - 1 cancels in FP32, whose
# p is off by a few ulps of 1 at most (2^-24 each).
TARGET_ABS_ERROR = 2.0**-23
# The loss: 4 FP32 ulps of the reference, plus (for the cancellation in log(sum) when the
# loss is small) the error of a sum rounded in FP32. See assert_tight.
LOSS_ULPS = 4
LOSS_ABS_ERROR = 2.0**-23

BIG = 1 << 40  # an "ulp difference" for a NaN against a number

# FP16 bit patterns.
NONFINITE_BITS = {"nan": 0x7E00, "-nan": 0xFFFF, "snan": 0x7C01, "-snan": 0xFC01, "inf": 0x7C00, "-inf": 0xFC00}
POISONS = ["random", "nan", "-nan", "inf", "-inf"]


@pytest.fixture(autouse=True)
def free_gpu_memory():
    """The laptop's GPU is 6 GiB and shared, so each test hands back what it cached."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


def shape_id(shape):
    return "x".join(map(str, shape))


def shape_params(shapes):
    """pytest.params of (rows, vocab, cols) shapes, with ids like 3x9x16."""
    return [pytest.param(*shape, id=shape_id(shape)) for shape in shapes]


def f32(x):
    """x rounded to FP32, as a float: the scale the kernel gets."""
    return torch.tensor(x, dtype=torch.float32).item()


def as_int32(value):
    return (value + 2**31) % 2**32 - 2**31


def tokens(targets):
    """Targets (int64, from 0 to 65,535) as the uint16 bit patterns the kernel reads."""
    return as_int16(targets).to(torch.int16)


def fp16_from_bits(bits, like):
    """A tensor like `like` (FP16) with every element the 16-bit pattern."""
    return torch.full_like(like.view(torch.int16), as_int16(bits)).view(torch.float16)


def random_bits(shape, seed):
    """Random 16-bit patterns as FP16: NaNs, infs, subnormals and -0 among them."""
    values = torch.randint(
        -(2**15), 2**15, shape, dtype=torch.int16, generator=generator(seed), device="cuda"
    )
    return values.view(torch.float16)


def int_view(tensor):
    return tensor.view(torch.int16 if tensor.dtype == torch.float16 else torch.int32)


def row_chunks(rows, cols):
    step = max(1, CHUNK_ELEMENTS // cols)
    return [slice(i, min(i + step, rows)) for i in range(0, rows, step)]


# -- Comparing -------------------------------------------------------------


def assert_same_bits(actual, expected, what):
    """The two tensors (FP16 or FP32) are equal bit for bit (NaN payloads included)."""
    a, e = int_view(actual), int_view(expected)
    assert a.shape == e.shape, f"{what}: shapes {tuple(a.shape)} and {tuple(e.shape)}"
    if torch.equal(a, e):
        return
    bad = a != e
    first = bad.nonzero()[0].tolist()
    raise AssertionError(
        f"{what}: {bad.sum().item()} of {bad.numel()} elements differ, the first at {first}: "
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


def half_ordinal(h):
    """FP16 values as ordinals that count ulps across signs: the sign-magnitude bits with the
    magnitude negated for a negative value, so +0 and -0 are both 0 and neighbours differ by
    1. A NaN's ordinal is past inf's; callers handle NaN separately."""
    bits = h.view(torch.int16).int() & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where(bits >= 0x8000, -magnitude, magnitude).long()


def float_ordinal(f):
    bits = f.view(torch.int32)
    magnitude = bits & 0x7FFFFFFF
    return torch.where(bits < 0, -magnitude, magnitude).long()


def spacing(t):
    """The distance from |t| to the next larger value of its type (FP16 or FP32), as float64;
    t must be finite."""
    if t.dtype == torch.float16:
        m = (t.view(torch.int16).int() & 0x7FFF).clamp(max=0x7BFE)
        lo = m.to(torch.int16).view(torch.float16).double()
        hi = (m + 1).to(torch.int16).view(torch.float16).double()
    else:
        m = (t.view(torch.int32) & 0x7FFFFFFF).clamp(max=0x7F7FFFFE)
        lo = m.view(torch.float32).double()
        hi = (m + 1).view(torch.float32).double()
    return hi - lo


# -- The references and the peers -------------------------------------------


def reference(logits, targets, scale, vocab):
    """The FP64 loss, gradient (before the FP16 rounding) and target probability of the rows
    of `logits` (rows x cols FP16, int64 targets), over the first vocab columns."""
    x = logits[:, :vocab].double()
    loss = F.cross_entropy(x, targets, reduction="none")
    p = F.softmax(x, dim=-1)
    grad = p * scale
    idx = torch.arange(len(targets), device="cuda")
    p_target = p[idx, targets]
    grad[idx, targets] -= scale
    return loss, grad, p_target


def autocast_peer(logits, targets, scale, vocab):
    """Autograd through autocast's own cross-entropy on an FP16 leaf, as a model trained with
    autocast gets it: the loss (FP32) and the leaf's FP16 gradient."""
    x = logits[:, :vocab].contiguous()
    with torch.enable_grad():
        leaf = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(leaf, targets, reduction="none")
        (loss.sum() * scale).backward()
    return loss.detach(), leaf.grad


def fp32_peer_loss(logits, targets, vocab):
    return F.cross_entropy(logits[:, :vocab].float(), targets, reduction="none")


# -- Inputs -----------------------------------------------------------------


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


def real_like(shape, g):
    """What OpenAI's GPT-2 weights give (measured over all rows: mean about -110, std 37 to
    51, min -334, max 170; a range of about 32 within a row; top p averaging 0.35 and up to
    0.9997; 99% of the gradients subnormal at scale 1): a per-row offset N(-110, 40^2) plus
    N(0, 4.5^2) per column, and in every 8th row a spike 15 above the row's largest value,
    which takes the row's top p to within about 1e-5 of 1."""
    rows, vocab = shape
    x = torch.randn((rows, 1), generator=g, device="cuda") * 40.0 - 110.0
    x = x + torch.randn(shape, generator=g, device="cuda") * 4.5
    spiked = torch.arange(0, rows, 8, device="cuda")
    column = torch.randint(0, vocab, (len(spiked),), generator=g, device="cuda")
    x[spiked, column] = x[spiked].max(dim=-1).values + 15.0
    return x.half()


KINDS = ["init", "real", "real_top", "wide8", "wide30", "outliers", "range"]


def vocab_part(kind, rows, vocab, g):
    shape = (rows, vocab)
    if kind == "init":
        return normal(shape, 0.3, g)
    if kind in ("real", "real_top"):
        return real_like(shape, g)
    if kind == "wide8":
        return normal(shape, 8.0, g)
    if kind == "wide30":
        return normal(shape, 30.0, g)
    if kind == "outliers":
        return with_outliers(shape, g)
    if kind == "range":
        return near_the_range(shape, g)
    raise ValueError(kind)


def pad_columns(x, vocab, g):
    """Makes the columns past vocab large (about +20), so that a kernel that read them
    would be far off."""
    cols = x.shape[1]
    if cols > vocab:
        x[:, vocab:] = normal((x.shape[0], cols - vocab), 10.0, g) + 20.0


def make_logits(kind, rows, vocab, cols, g):
    x = torch.empty((rows, cols), dtype=torch.float16, device="cuda")
    x[:, :vocab] = vocab_part(kind, rows, vocab, g)
    pad_columns(x, vocab, g)
    return x


def make_targets(kind, logits, vocab, g):
    """Uniform over the vocabulary, or (real_top) each row's top logit."""
    rows = logits.shape[0]
    if kind == "real_top":
        return logits[:, :vocab].argmax(dim=-1)
    return torch.randint(0, vocab, (rows,), generator=g, device="cuda")


def random_logits(rows, vocab, cols, std, seed):
    """N(0, std^2) logits with large values past vocab."""
    g = generator(seed)
    x = normal((rows, cols), std, g)
    pad_columns(x, vocab, g)
    return x


def random_targets(rows, vocab, seed):
    return torch.randint(0, vocab, (rows,), generator=generator(seed), device="cuda")


# -- Running Flint ----------------------------------------------------------


def flint_ce(flint, logits, targets, scale, vocab, offset=8, fill=SENTINEL, loss_offset=None,
             stream=None, check_input=True):
    """Runs the kernel into dlogits and row_loss inside buffers pre-filled with `fill` (a
    16-bit pattern; row_loss gets it twice), `offset` halves and `loss_offset` floats in
    (multiples of 8 and 4 keep them 16-byte aligned). Checks that the buffers around the
    outputs are untouched, that the logits and targets are unchanged and that no element of
    either output still holds the fill, and returns dlogits (rows x cols FP16) and row_loss
    (FP32). `targets` are the int16 tokens."""
    rows, cols = logits.shape
    count = rows * cols
    if loss_offset is None:
        loss_offset = max(4, offset // 2)
    buf = torch.full((offset + count + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda")
    fill32 = as_int32(((fill & 0xFFFF) << 16) | (fill & 0xFFFF))
    loss_buf = torch.full((loss_offset + rows + GUARD_AFTER,), fill32, dtype=torch.int32, device="cuda")
    d = buf[offset : offset + count]
    loss = loss_buf[loss_offset : loss_offset + rows]
    before = (logits.clone(), targets.clone()) if check_input else None

    flint.softmax_cross_entropy(d, loss, logits, targets, scale, rows, vocab, cols, stream=stream)
    torch.cuda.synchronize()

    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    guard32 = torch.full((1,), fill32, dtype=torch.int32, device="cuda")
    assert all_equal(buf[:offset], guard), "flint_softmax_cross_entropy wrote before dlogits"
    assert all_equal(buf[offset + count :], guard), "flint_softmax_cross_entropy wrote past dlogits"
    assert all_equal(loss_buf[:loss_offset], guard32), "flint_softmax_cross_entropy wrote before row_loss"
    assert all_equal(loss_buf[loss_offset + rows :], guard32), "flint_softmax_cross_entropy wrote past row_loss"
    if check_input:
        assert torch.equal(logits.view(torch.int16), before[0].view(torch.int16)), "logits changed"
        assert torch.equal(targets, before[1]), "targets changed"
    assert none_equal(d, guard), "an element of dlogits was not written"
    assert none_equal(loss, guard32), "an element of row_loss was not written"
    return d.view(torch.float16).view(rows, cols), loss.view(torch.float32)


# -- The tight comparison ---------------------------------------------------

ChunkAnalysis = namedtuple(
    "ChunkAnalysis",
    "nt_count nt_nonzero nt_exact nt_worst t_diff t_big t_ratio one_minus_p loss_diff loss_large_diff "
    "loss_excess loss_abs",
)


def chunk_analysis(d, loss, logits, targets, scale, vocab):
    """Flint's dlogits and row_loss for some rows, d (rows x cols) and loss (rows), against the
    FP64 reference. The non-target columns are measured in FP16 ulps from the reference
    correctly rounded, the target column in ulps and in absolute error (as a multiple of the
    2^-23 * |scale| that p - 1 allows, beyond half an ulp), the loss in FP32 ulps and in
    absolute error beyond LOSS_ULPS ulps. A NaN where the reference has none, or none where it
    has one, counts as BIG ulps."""
    rows = d.shape[0]
    idx = torch.arange(rows, device="cuda")
    ref_loss, ref, p_target = reference(logits, targets, scale, vocab)
    out = d[:, :vocab]
    out_nan, ref_nan = out.isnan(), ref.isnan()
    ref_half = round_to_half(torch.nan_to_num(ref, nan=0.0))
    diff = (half_ordinal(out) - half_ordinal(ref_half)).abs()
    diff = diff.masked_fill(out_nan != ref_nan, BIG).masked_fill(out_nan & ref_nan, 0)

    target = torch.zeros_like(out, dtype=torch.bool)
    target[idx, targets] = True
    other = ~target
    nonzero = other & ((ref_half.view(torch.int16) & 0x7FFF) != 0)
    t_diff = diff[idx, targets]
    one_minus_p = 1.0 - p_target

    t_out, t_ref, t_half = out[idx, targets], ref[idx, targets], ref_half[idx, targets]
    err = (t_out.double() - t_ref).abs()
    half_ulp = 0.5 * torch.maximum(spacing(t_out), spacing(t_half))
    excess = err - half_ulp
    bound = TARGET_ABS_ERROR * abs(scale)
    t_ratio = torch.where(excess <= 0, torch.zeros_like(excess), excess / max(bound, 1e-300))
    t_bad = t_out.isnan() != t_ref.isnan()
    t_ratio = t_ratio.masked_fill(t_bad, math.inf).masked_fill(t_out.isnan() & t_ref.isnan(), 0)

    ref32 = ref_loss.float()
    loss_nan, ref_loss_nan = loss.isnan(), ref32.isnan()
    loss_diff = (float_ordinal(loss) - float_ordinal(ref32)).abs()
    loss_diff = loss_diff.masked_fill(loss_nan != ref_loss_nan, BIG).masked_fill(loss_nan & ref_loss_nan, 0)
    loss_err = torch.where(loss.double() == ref_loss, torch.zeros_like(ref_loss), (loss.double() - ref_loss).abs())
    loss_err = loss_err.masked_fill(loss_nan != ref_loss_nan, math.inf).masked_fill(loss_nan & ref_loss_nan, 0)
    loss_excess = loss_err - LOSS_ULPS * spacing(ref32.nan_to_num(posinf=0.0))
    return ChunkAnalysis(
        nt_count=other.sum().item(),
        nt_nonzero=nonzero.sum().item(),
        nt_exact=(nonzero & (diff == 0)).sum().item(),
        nt_worst=diff.masked_fill(target, 0).max().item(),
        t_diff=t_diff,
        t_big=one_minus_p >= TARGET_ULP_MIN_ONE_MINUS_P,
        t_ratio=t_ratio,
        one_minus_p=one_minus_p,
        loss_diff=loss_diff,
        loss_large_diff=loss_diff.masked_fill(ref_loss < 1.0, 0),
        loss_excess=loss_excess,
        loss_abs=loss_err,
    )


@dataclass
class Stats:
    """chunk_analysis over all the rows."""

    nt_count: int = 0
    nt_nonzero: int = 0
    nt_exact: int = 0
    nt_worst: int = 0  # ulps, non-target columns
    t_worst_big: int = 0  # ulps, targets with 1 - p >= TARGET_ULP_MIN_ONE_MINUS_P
    t_worst: int = 0  # ulps, all targets
    t_ratio: float = 0.0  # the target's absolute error beyond half an ulp, over 2^-23 |scale|
    loss_worst: int = 0  # FP32 ulps, all rows
    loss_worst_large: int = 0  # FP32 ulps, rows whose loss is at least 1
    loss_excess: float = -math.inf  # |loss error| beyond LOSS_ULPS ulps of the reference
    loss_abs: float = 0.0

    @property
    def share(self):
        return self.nt_exact / self.nt_nonzero if self.nt_nonzero else 1.0

    def add(self, c):
        self.nt_count += c.nt_count
        self.nt_nonzero += c.nt_nonzero
        self.nt_exact += c.nt_exact
        self.nt_worst = max(self.nt_worst, c.nt_worst)
        if c.t_diff.numel():
            self.t_worst = max(self.t_worst, c.t_diff.max().item())
            if c.t_big.any():
                self.t_worst_big = max(self.t_worst_big, c.t_diff[c.t_big].max().item())
            self.t_ratio = max(self.t_ratio, c.t_ratio.max().item())
            self.loss_worst = max(self.loss_worst, c.loss_diff.max().item())
            self.loss_worst_large = max(self.loss_worst_large, c.loss_large_diff.max().item())
            self.loss_excess = max(self.loss_excess, c.loss_excess.max().item())
            self.loss_abs = max(self.loss_abs, c.loss_abs.max().item())


def analyze(d, loss, logits, targets, scale, vocab):
    """Stats of Flint's outputs against the FP64 reference, over chunks of rows."""
    stats = Stats()
    for chunk in row_chunks(*logits.shape):
        stats.add(chunk_analysis(d[chunk], loss[chunk], logits[chunk], targets[chunk], scale, vocab))
    return stats


def assert_tight(stats, what, check_loss=True):
    """Every non-target dlogits value within 1 ulp of the correct rounding and 99.9% equal to
    it; the target within 1 ulp while 1 - p is at least 1e-3 and within the absolute bound in
    all rows; and the loss within LOSS_ULPS FP32 ulps of the reference plus LOSS_ABS_ERROR."""
    assert stats.nt_worst <= 1, f"{what}: a non-target value is {stats.nt_worst} ulps from the correct rounding"
    if stats.nt_nonzero >= MIN_SHARE_COUNT:
        assert stats.share >= MIN_SHARE, (
            f"{what}: {stats.share:.6f} of the {stats.nt_nonzero} non-zero non-target values are "
            f"correctly rounded, {stats.nt_nonzero - stats.nt_exact} are not"
        )
    assert stats.t_worst_big <= 1, f"{what}: a target with 1 - p >= 1e-3 is {stats.t_worst_big} ulps off"
    assert stats.t_ratio <= 1.0, (
        f"{what}: a target's error is {stats.t_ratio:.3g} times 2^-23 * |scale| beyond half an ulp"
    )
    if check_loss:
        assert stats.loss_excess <= LOSS_ABS_ERROR, (
            f"{what}: a loss is off by {stats.loss_excess:.3g} beyond {LOSS_ULPS} FP32 ulps "
            f"(worst {stats.loss_worst} ulps, {stats.loss_abs:.3g} absolute)"
        )


def describe(stats):
    return (
        f"nt_ulps {stats.nt_worst} share {stats.share:.6f} of {stats.nt_nonzero}, "
        f"t_ulps {stats.t_worst_big} (all {stats.t_worst}), t_abs_ratio {stats.t_ratio:.3g}, "
        f"loss_ulps {stats.loss_worst} (loss >= 1: {stats.loss_worst_large}) loss_abs {stats.loss_abs:.3g} "
        f"loss_excess {stats.loss_excess:.3g}"
    )


def run_and_check(flint, logits, targets, scale, vocab, what, **kwargs):
    """The kernel, then the tight comparison; returns the outputs and the stats."""
    d, loss = flint_ce(flint, logits, tokens(targets), scale, vocab, **kwargs)
    stats = analyze(d, loss, logits, targets, scale, vocab)
    assert_tight(stats, what)
    assert all_equal(int_view(d)[:, vocab:], 0), f"{what}: dlogits past vocab are not +0"
    return d, loss, stats


# -- PyTorch's behaviour ----------------------------------------------------


def test_autocast_cross_entropy_takes_its_log_softmax_in_fp16():
    # What makes the loss's peer the FP32 cross-entropy and not autocast's: under autocast,
    # F.cross_entropy(reduction="none") on FP16 logits is bitwise the FP16 log_softmax, upcast
    # and negated at the targets. If a PyTorch upgrade changes this, the gate's choice of peer
    # should be revisited.
    rows, vocab = 64, 1000
    g = generator(0)
    for std in (1.0, 4.0, 8.0):
        x = normal((rows, vocab), std, g)
        targets = torch.randint(0, vocab, (rows,), generator=g, device="cuda")
        idx = torch.arange(rows, device="cuda")

        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(x, targets, reduction="none")

        log_p = F.log_softmax(x, dim=-1, dtype=torch.float16)
        expected = -(log_p[idx, targets].float())
        assert loss.dtype == torch.float32
        assert_same_bits(loss, expected, f"autocast's loss at std {std}")


def test_autocast_loss_has_fp16_error_and_the_fp32_peer_does_not():
    # The numbers behind that: against the FP64 loss, autocast's relative L2 is at FP16's
    # level and F.cross_entropy on FP32 logits is some 1e4 times better.
    rows, vocab = 2048, 4096
    g = generator(1)
    x = normal((rows, vocab), 4.0, g)
    targets = torch.randint(0, vocab, (rows,), generator=g, device="cuda")
    ref = F.cross_entropy(x.double(), targets, reduction="none")

    autocast_loss, grad = autocast_peer(x, targets, 1.0, vocab)
    fp32_loss = fp32_peer_loss(x, targets, vocab)

    autocast_error = gate.errors(autocast_loss, ref).rel_l2
    fp32_error = gate.errors(fp32_loss, ref).rel_l2
    print(f"cross_entropy peers: autocast loss rel_l2 {autocast_error:.3g}, fp32 {fp32_error:.3g}")
    assert grad.dtype == torch.float16
    assert autocast_error > 3e-5, "autocast's loss is no longer at FP16's level of error"
    assert autocast_error > 1000 * fp32_error
    assert fp32_error < 1e-6


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


def gate_errors(d, loss, logits, targets, scale, vocab, with_loss=True):
    """Flint's and the peers' gate.Errors against the reference, over chunks of rows: a dict
    with the loss against the FP32 peer and the gradient against autocast's."""
    names = ["grad_flint", "grad_peer"] + (["loss_flint", "loss_peer"] if with_loss else [])
    acc = {name: Accumulator() for name in names}
    rows, cols = logits.shape
    for chunk in row_chunks(rows, cols):
        x, t = logits[chunk], targets[chunk]
        ref_loss, ref_grad, _ = reference(x, t, scale, vocab)
        acc["grad_flint"].add(d[chunk, :vocab], ref_grad)
        acc["grad_peer"].add(autocast_peer(x, t, scale, vocab)[1], ref_grad)
        if with_loss:
            acc["loss_flint"].add(loss[chunk], ref_loss)
            acc["loss_peer"].add(fp32_peer_loss(x, t, vocab), ref_loss)
        assert all_equal(int_view(d[chunk, vocab:]), 0), "dlogits past vocab are not +0"
        del ref_loss, ref_grad
    return {name: a.errors() for name, a in acc.items()}


def check_gates(errors_per_seed, label, with_loss=True):
    """The gradient's gate (autocast as the peer, GRADIENT_FACTOR) and the loss's (the FP32
    cross-entropy, FORWARD_FACTOR, FP32 floors), over the seeds' errors; prints the numbers."""
    pick = lambda name: [e[name] for e in errors_per_seed]  # noqa: E731
    grad_bounds = gate.check_gate(gate.GRADIENT_FACTOR, pick("grad_flint"), pick("grad_peer"))
    line = (
        f"GATE {label} | grad rel_l2 {max(e.rel_l2 for e in pick('grad_flint')):.3g} "
        f"(autocast {max(e.rel_l2 for e in pick('grad_peer')):.3g}, bound {grad_bounds.rel_l2:.3g}) "
        f"max_abs {max(e.max_abs for e in pick('grad_flint')):.3g} "
        f"(autocast {max(e.max_abs for e in pick('grad_peer')):.3g}, bound {grad_bounds.max_abs:.3g})"
    )
    if with_loss:
        loss_bounds = gate.check_gate(gate.FORWARD_FACTOR, pick("loss_flint"), pick("loss_peer"), output="fp32")
        line += (
            f" | loss rel_l2 {max(e.rel_l2 for e in pick('loss_flint')):.3g} "
            f"(fp32 {max(e.rel_l2 for e in pick('loss_peer')):.3g}, bound {loss_bounds.rel_l2:.3g}) "
            f"max_abs {max(e.max_abs for e in pick('loss_flint')):.3g} "
            f"(fp32 {max(e.max_abs for e in pick('loss_peer')):.3g}, bound {loss_bounds.max_abs:.3g})"
        )
    print(line)


def scale_value(name, rows):
    return {"1": 1.0, "1/rows": f32(1.0 / rows), "0.125": 0.125, "2^-19": 2.0**-19}[name]


def run_gate(flint, kind, rows, vocab, cols, scale_name, with_loss):
    scale = scale_value(scale_name, rows)
    per_seed = []
    for seed in gate.SEEDS:
        g = generator(seed)
        logits = make_logits(kind, rows, vocab, cols, g)
        targets = make_targets(kind, logits, vocab, g)
        d, loss = flint_ce(flint, logits, tokens(targets), scale, vocab, offset=8 * (1 + seed))
        per_seed.append(gate_errors(d, loss, logits, targets, scale, vocab, with_loss))
        del logits, targets, d, loss
    check_gates(per_seed, f"{kind} {shape_id((rows, vocab, cols))} scale {scale_name}", with_loss)


def shrunk(shapes):
    """The shapes, with rows cut to a few under the sanitizer (and duplicates dropped)."""
    if SANITIZER:
        shapes = [(min(rows, 4 if vocab > 20_000 else 16), vocab, cols) for rows, vocab, cols in shapes]
    return list(dict.fromkeys(shapes))


# Lengths of 1 chunk, partial chunks, past 32 chunks (so lanes wrap), GPT-2's, and the widest.
GATE_SHAPES = shrunk([
    (1, 1, 8), (1, 8, 8), (3, 9, 16), (7, 37, 40), (8, 256, 256), (13, 257, 264), (9, 1023, 1024),
    (64, 50257, 50304), (1024, 50257, 50304), (9, 65536, 65536),
])  # fmt: skip
GATE_CASES = [
    pytest.param(kind, *shape, id=f"{kind}-{shape_id(shape)}") for shape in GATE_SHAPES for kind in KINDS
]


@pytest.mark.parametrize("kind, rows, vocab, cols", GATE_CASES)
def test_passes_the_gate(flint, kind, rows, vocab, cols):
    run_gate(flint, kind, rows, vocab, cols, "1", with_loss=True)


# The other scales, on the two most realistic inputs, at shapes of three sizes. The loss
# doesn't depend on scale, so only the gradient is gated.
SCALE_SHAPES = shrunk([(13, 257, 264), (64, 50257, 50304), (1024, 50257, 50304)])
SCALE_CASES = [
    pytest.param(kind, *shape, scale, id=f"{kind}-{shape_id(shape)}-{scale}")
    for shape in SCALE_SHAPES
    for kind in ("init", "real", "real_top")
    for scale in ("1/rows", "0.125", "2^-19")
]


@pytest.mark.parametrize("kind, rows, vocab, cols, scale", SCALE_CASES)
def test_gradient_passes_the_gate_at_every_scale(flint, kind, rows, vocab, cols, scale):
    run_gate(flint, kind, rows, vocab, cols, scale, with_loss=False)


@pytest.mark.skipif(SANITIZER, reason="GPT-2's whole micro-batch is far too big for the sanitizer")
def test_gpt2s_whole_micro_batch_passes_the_gate(flint):
    # 16 sequences x 1024 tokens: 1.65 GB each for logits and dlogits. The logits are made
    # and the errors computed over chunks of rows.
    rows, vocab, cols = 16_384, 50257, 50304
    scale = 0.125
    logits = torch.empty((rows, cols), dtype=torch.float16, device="cuda")
    per_seed = []
    for seed in gate.SEEDS:
        g = generator(seed)
        targets = torch.empty(rows, dtype=torch.int64, device="cuda")
        for chunk in row_chunks(rows, cols):
            logits[chunk] = make_logits("real", chunk.stop - chunk.start, vocab, cols, g)
            targets[chunk] = make_targets("real", logits[chunk], vocab, g)
        d, loss = flint_ce(flint, logits, tokens(targets), scale, vocab, check_input=False)
        per_seed.append(gate_errors(d, loss, logits, targets, scale, vocab))
        del d, loss, targets
        gc.collect()
        torch.cuda.empty_cache()
    check_gates(per_seed, f"real {shape_id((rows, vocab, cols))} scale 0.125")


# -- A tighter bound than the gate ------------------------------------------

TIGHT_SHAPES = shrunk([(5, 37, 40), (16, 257, 264), (32, 1023, 1024), (64, 50257, 50304), (128, 50257, 50304)])
TIGHT_CASES = [
    pytest.param(kind, *shape, id=f"{kind}-{shape_id(shape)}") for kind in KINDS for shape in TIGHT_SHAPES
]


@pytest.mark.parametrize("kind, rows, vocab, cols", TIGHT_CASES)
def test_every_output_is_within_an_ulp_of_the_correctly_rounded_reference(flint, kind, rows, vocab, cols):
    g = generator(0)
    logits = make_logits(kind, rows, vocab, cols, g)
    targets = make_targets(kind, logits, vocab, g)

    _, _, stats = run_and_check(flint, logits, targets, 1.0, vocab, f"{kind} {shape_id((rows, vocab, cols))}")

    print(f"TIGHT {kind} {shape_id((rows, vocab, cols))} scale 1: {describe(stats)}")


TIGHT_SCALE_CASES = [
    pytest.param(kind, *shape, scale, id=f"{kind}-{shape_id(shape)}-{scale}")
    for shape in shrunk([(16, 257, 264), (64, 50257, 50304)])
    for kind in ("init", "real", "real_top")
    for scale in ("1/rows", "0.125", "2^-19")
]


@pytest.mark.parametrize("kind, rows, vocab, cols, scale", TIGHT_SCALE_CASES)
def test_the_bounds_hold_at_every_scale(flint, kind, rows, vocab, cols, scale):
    g = generator(1)
    logits = make_logits(kind, rows, vocab, cols, g)
    targets = make_targets(kind, logits, vocab, g)

    _, _, stats = run_and_check(
        flint, logits, targets, scale_value(scale, rows), vocab, f"{kind} {shape_id((rows, vocab, cols))} scale {scale}"
    )

    print(f"TIGHT {kind} {shape_id((rows, vocab, cols))} scale {scale}: {describe(stats)}")


ONE_MINUS_P = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]


@pytest.mark.parametrize("scale", [1.0, 0.125, 1.0 / 3.0, 2.0**-12], ids=lambda v: f"scale-{v:.4g}")
def test_the_target_near_p_1_stays_within_the_absolute_bound(flint, scale):
    # The target's logit is set so that 1 - p runs from about 1e-1 to 1e-7 (FP16 quantizes the
    # logit, so the reference's actual 1 - p is what is reported). There p - 1 cancels in FP32:
    # the error stays absolute (2^-23 * |scale|, plus half an FP16 ulp) while the relative
    # error grows. Rows are 16 of each, with N(0, 1) logits around the target's.
    scale = f32(scale)
    vocab, cols, per = 1000, 1000, 16
    rows = per * len(ONE_MINUS_P)
    g = generator(3)
    x = normal((rows, cols), 1.0, g)
    targets = torch.randint(0, vocab, (rows,), generator=g, device="cuda")
    idx = torch.arange(rows, device="cuda")
    x[idx, targets] = -100.0
    others = torch.exp(x.double()).sum(dim=-1)  # the target's exp is 0
    delta = torch.tensor(ONE_MINUS_P, dtype=torch.float64, device="cuda").repeat_interleave(per)
    x[idx, targets] = torch.log(others * (1 - delta) / delta).half()

    d, loss = flint_ce(flint, x, tokens(targets), scale, vocab)
    c = chunk_analysis(d, loss, x, targets, scale, vocab)

    one_minus_p = c.one_minus_p
    for i, wanted in enumerate(ONE_MINUS_P):
        rows_i = slice(i * per, (i + 1) * per)
        print(
            f"TARGET-SWEEP scale {scale:.4g} 1-p {wanted:.0e} (measured {one_minus_p[rows_i].min().item():.2e} to "
            f"{one_minus_p[rows_i].max().item():.2e}): target ulps {c.t_diff[rows_i].max().item()}, "
            f"abs error beyond half an ulp {c.t_ratio[rows_i].max().item():.3g} x 2^-23 |scale|"
        )
        assert abs(math.log10(one_minus_p[rows_i].mean().item() / wanted)) < 0.3, "the sweep missed its 1 - p"
    assert c.t_ratio.max().item() <= 1.0
    assert c.t_diff[c.t_big].max().item() <= 1
    # The other columns and the loss hold too.
    assert c.nt_worst <= 1
    assert c.loss_excess.max().item() <= LOSS_ABS_ERROR


# -- Exact cases ------------------------------------------------------------


@pytest.mark.parametrize("poison", POISONS)
@pytest.mark.parametrize(
    "rows, vocab, cols",
    shape_params([(5, 37, 64), (5, 40, 64), (3, 1001, 1024), (3, 1008, 1024), (2, 50257, 50304), (2, 50256, 50304)]),
)
def test_poison_past_vocab_changes_no_output_bit(flint, poison, rows, vocab, cols):
    # vocab is mid-chunk (37, 1001, 50257) or at a chunk boundary (40, 1008, 50256). The clean
    # run has zeros past vocab. dlogits there are +0 by bits whatever the logits hold.
    x = normal((rows, cols), 3.0, generator(2))
    clean = x.clone()
    clean[:, vocab:] = 0
    width = cols - vocab
    if poison == "random":
        junk = random_bits((rows, width), 3)
        # Random patterns are NaN 3% of the time but inf only twice in 65,536.
        junk.view(torch.int16)[..., 2::13] = as_int16(NONFINITE_BITS["inf"])
        junk.view(torch.int16)[..., 7::13] = as_int16(NONFINITE_BITS["-inf"])
    else:
        junk = fp16_from_bits(NONFINITE_BITS[poison], x[:, vocab:])
    dirty = x.clone()
    dirty[:, vocab:] = junk
    targets = tokens(random_targets(rows, vocab, 4))

    d0, loss0 = flint_ce(flint, clean, targets, 0.5, vocab)
    d1, loss1 = flint_ce(flint, dirty, targets, 0.5, vocab)

    assert_same_bits(d1, d0, f"{poison} past vocab: dlogits")
    assert_same_bits(loss1, loss0, f"{poison} past vocab: row_loss")
    assert all_equal(int_view(d1)[:, vocab:], 0)
    assert torch.isfinite(d1[:, :vocab]).all() and torch.isfinite(loss1).all()


# One row per target, so rows = vocab: mid-chunk or at a chunk boundary, with more than 32
# chunks (so that lanes wrap) or fewer.
@pytest.mark.parametrize(
    "rows, vocab, cols", shape_params([(300, 300, 304), (296, 296, 296), (257, 257, 264), (1023, 1023, 1024)])
)
def test_every_target_position_gives_the_right_loss_and_gradient(flint, rows, vocab, cols):
    # Targets run from 0 to vocab - 1: the first column, the last, every chunk boundary and
    # both sides of it, and, with 33 or more chunks, every lane (chunk i is in lane i % 32)
    # holding the target in turn, at each of the chunk's 8 values.
    assert rows == vocab
    logits = random_logits(rows, vocab, cols, 2.0, 5)
    targets = torch.arange(rows, device="cuda")

    d, loss, stats = run_and_check(flint, logits, targets, 1.0, vocab, f"target at every position of {vocab}")

    # Every row's loss was written (flint_ce checks) and every target column is in its bound
    # (assert_tight does); and the target's gradient is negative, the others' positive.
    assert (d[torch.arange(rows), targets] < 0).all()
    print(f"TARGETS {vocab}: {describe(stats)}")


@pytest.mark.parametrize("rows", [1, 3, 7, 9, 13, 17, 63, 65])
@pytest.mark.parametrize("offset, fill", [(8, SENTINEL), (16, NAN_BITS), (24, OTHER_BITS)])
def test_every_element_is_written_and_nothing_else(flint, rows, offset, fill):
    # flint_ce checks that the buffers around both outputs are intact, that the logits and
    # targets are unchanged and that no element is left holding the fill, including at odd
    # row counts, where the last block of 8 rows isn't full.
    vocab, cols = 37, 40
    logits = random_logits(rows, vocab, cols, 3.0, 6)
    targets = random_targets(rows, vocab, 7)

    d, loss = flint_ce(flint, logits, tokens(targets), 1.0, vocab, offset=offset, fill=fill)
    d2, loss2 = flint_ce(flint, logits, tokens(targets), 1.0, vocab, fill=SENTINEL if fill != SENTINEL else NAN_BITS)

    assert torch.isfinite(d).all() and torch.isfinite(loss).all()
    assert_same_bits(d, d2, "against a run into another fill")
    assert_same_bits(loss, loss2, "against a run into another fill")


@pytest.mark.parametrize("offset, loss_offset", [(8, 4), (16, 12), (24, 28), (8, 36), (32, 4)])
@pytest.mark.parametrize("rows, vocab, cols", shape_params([(1, 8, 8), (5, 100, 104), (11, 257, 264)]))
def test_nothing_is_written_around_the_outputs(flint, rows, vocab, cols, offset, loss_offset):
    # The two outputs sit at different alignments in their own buffers; the fill around them
    # (and the logits and targets) must be intact.
    logits = random_logits(rows, vocab, cols, 3.0, 8)
    targets = random_targets(rows, vocab, 9)
    run_and_check(flint, logits, targets, 0.25, vocab, "guarded", offset=offset, loss_offset=loss_offset)


EQUAL_VALUES = [0.0, 1.0, -3.5, 1000.0, -60000.0, 65504.0, 6e-8]
EQUAL_VOCABS = [2, 3, 5, 7, 8, 9, 37, 100, 257, 1000, 1023, 4096, 50257, 65536]


@pytest.mark.parametrize("vocab", EQUAL_VOCABS)
def test_equal_logits_give_log_vocab_and_the_correctly_rounded_one_over_vocab(flint, vocab):
    rows, cols = 8, (vocab + 7) // 8 * 8
    targets = (torch.arange(rows, device="cuda") * 37 + 5) % vocab
    targets[0], targets[1] = 0, vocab - 1
    log_vocab = torch.tensor(math.log(vocab), dtype=torch.float64).float().item()
    for value in EQUAL_VALUES:
        logits = torch.full((rows, cols), value, dtype=torch.float16, device="cuda")
        for scale in (1.0, 0.125, 0.7):
            scale = f32(scale)
            d, loss = flint_ce(flint, logits, tokens(targets), scale, vocab)

            # The loss is log(vocab) within 1 FP32 ulp.
            expected_loss = torch.full((rows,), log_vocab, dtype=torch.float32, device="cuda")
            assert (float_ordinal(loss) - float_ordinal(expected_loss)).abs().max().item() <= 1, (value, scale)
            # Non-targets are the correct rounding of scale / vocab, the target that of
            # scale * (1 / vocab - 1).
            other = round_to_half(torch.tensor(scale / vocab, dtype=torch.float64, device="cuda"))
            target = round_to_half(torch.tensor(scale * (1.0 / vocab - 1.0), dtype=torch.float64, device="cuda"))
            expected = other.expand(rows, cols).clone()
            expected[:, vocab:] = 0
            expected[torch.arange(rows), targets] = target
            assert_same_bits(d, expected, f"equal logits {value} vocab {vocab} scale {scale}")


@pytest.mark.parametrize("cols", [8, 64, 1024])
@pytest.mark.parametrize("scale", [1.0, 0.125, 0.0, 1e6])
def test_vocab_1_gives_loss_0_and_gradient_plus_zero_at_any_finite_logit(flint, cols, scale):
    rows = 40
    x = random_logits(rows, 1, cols, 30.0, 10)
    special = [65504.0, -65504.0, 0.0, 6e-8, -6e-8, 1.0, -1.0, 60000.0]
    x[: len(special), 0] = torch.tensor(special, dtype=torch.float16, device="cuda")
    x.view(torch.int16)[len(special), 0] = as_int16(0x8000)  # -0
    targets = torch.zeros(rows, dtype=torch.int64, device="cuda")

    d, loss = flint_ce(flint, x, tokens(targets), scale, 1)

    assert all_equal(int_view(loss), 0), "row_loss is not +0"
    assert all_equal(int_view(d), 0), "dlogits are not all +0"


@pytest.mark.parametrize("vocab, cols", [(300, 304), (1023, 1024), (50257, 50304)])
@pytest.mark.parametrize("scale", [1.0, 0.125])
def test_a_dominant_target_gives_loss_0_and_gradient_plus_zero(flint, vocab, cols, scale):
    # The target's logit is 100 above every other (N(0, 1)), whose exps against it are 0.
    # With e^-100 * vocab far below an FP32 ulp of 1, the sum is exactly 1, so the loss is
    # exactly 0 and the target's p - 1 is 0; the others round to 0. Targets run over the row
    # positions, which put the max in every lane in turn.
    rows = 64
    x = random_logits(rows, vocab, cols, 1.0, 11)
    targets = (torch.arange(rows, device="cuda") * 53 + 3) % vocab
    targets[:2] = torch.tensor([0, vocab - 1], device="cuda")
    x[torch.arange(rows), targets] = 100.0

    d, loss = flint_ce(flint, x, tokens(targets), scale, vocab)

    assert all_equal(int_view(loss), 0), "row_loss is not exactly +0"
    assert all_equal(int_view(d), 0), "dlogits are not all +0"


@pytest.mark.parametrize("vocab, cols", [(300, 304), (1023, 1024)])
@pytest.mark.parametrize("gap", [-100.0, -1000.0, -60000.0])
@pytest.mark.parametrize("scale", [1.0, 0.125, 1e-3])
def test_a_far_lower_target_has_a_finite_loss_and_gradient_minus_scale(flint, vocab, cols, gap, scale):
    # The target's logit is far below the others' (N(0, 1)): its exp underflows to 0, so
    # -log(p) would be inf, but the loss is log(sum) - (logit - max), about the gap, and
    # p - 1 is exactly -1, so the target's gradient is -scale rounded.
    rows = 64
    x = random_logits(rows, vocab, cols, 1.0, 12)
    targets = random_targets(rows, vocab, 13)
    idx = torch.arange(rows, device="cuda")
    x[idx, targets] = gap

    d, loss, stats = run_and_check(flint, x, targets, scale, vocab, f"target {gap} below")

    assert torch.isfinite(loss).all() and (loss > -gap * 0.99).all()
    expected = round_to_half(torch.tensor(-scale, dtype=torch.float64, device="cuda")).expand(rows)
    assert_same_bits(d[idx, targets], expected.contiguous(), "the target's gradient")


@pytest.mark.parametrize("vocab, cols", [(300, 304), (296, 296), (1023, 1024)])
@pytest.mark.parametrize("scale", [1.0, 0.125])
def test_a_dominant_non_target_at_every_column_gives_a_one_hot_row(flint, vocab, cols, scale):
    # Row r has 50 at column r (the others N(0, 1), whose exps against it are 0), so the max
    # is in every lane and every position of a chunk in turn. The sum is exactly 1, so the
    # row is one-hot: scale at the dominant column and 0 elsewhere, but for the target (a
    # fixed other column for most rows, the dominant column itself for every 7th), whose
    # gradient is -scale (or 0 if it is the dominant one). The loss is the exact gap, 50
    # minus the target's logit, or 0.
    rows = vocab
    x = random_logits(rows, vocab, cols, 1.0, 14)
    idx = torch.arange(rows, device="cuda")
    dominant = idx
    targets = (idx * 29 + 7) % vocab
    targets = torch.where(idx % 7 == 0, dominant, torch.where(targets == dominant, (targets + 1) % vocab, targets))
    x[idx, dominant] = 50.0

    d, loss, stats = run_and_check(flint, x, targets, scale, vocab, "one-hot rows")

    expected = torch.zeros((rows, cols), dtype=torch.float16, device="cuda")
    s = round_to_half(torch.tensor(scale, dtype=torch.float64, device="cuda"))
    expected[idx, dominant] = s
    expected[idx, targets] = torch.where(targets == dominant, torch.zeros_like(s), -s)
    assert_same_bits(d, expected, "one-hot rows")
    gap = (50.0 - x[idx, targets].double()).float()
    assert_same_bits(loss, gap, "the loss of one-hot rows")


@pytest.mark.parametrize("vocab, cols", [(37, 40), (300, 304)])
def test_a_row_does_not_depend_on_its_position_or_its_neighbours(flint, vocab, cols):
    # The same row and target at 21 positions (three blocks of 8 rows, each warp of a block):
    # the same bits everywhere, as for the row alone. A row among other rows, alone, and in a
    # permutation of the rows gives the same bits too.
    row = random_logits(1, vocab, cols, 4.0, 15)
    target = random_targets(1, vocab, 16)
    many = row.repeat(21, 1)
    d, loss = flint_ce(flint, many, tokens(target.repeat(21)), 0.5, vocab)
    alone_d, alone_loss = flint_ce(flint, row, tokens(target), 0.5, vocab)
    for r in range(21):
        assert_same_bits(d[r : r + 1], alone_d, f"row {r} against the row alone")
        assert_same_bits(loss[r : r + 1], alone_loss, f"row {r}'s loss against the row alone")

    rows = 19
    x = random_logits(rows, vocab, cols, 6.0, 17)
    targets = random_targets(rows, vocab, 18)
    d, loss = flint_ce(flint, x, tokens(targets), 0.5, vocab)
    for r in range(rows):
        one_d, one_loss = flint_ce(flint, x[r : r + 1].contiguous(), tokens(targets[r : r + 1]), 0.5, vocab)
        assert_same_bits(d[r : r + 1], one_d, f"row {r} in {rows} against alone")
        assert_same_bits(loss[r : r + 1], one_loss, f"row {r}'s loss in {rows} against alone")

    perm = torch.randperm(rows, generator=generator(1), device="cuda")
    assert any_unequal(perm, torch.arange(rows, device="cuda"))
    pd, ploss = flint_ce(flint, x[perm].contiguous(), tokens(targets[perm]), 0.5, vocab)
    assert_same_bits(pd, d[perm], "permuted rows")
    assert_same_bits(ploss, loss[perm], "permuted rows' loss")


@pytest.mark.parametrize("vocab, small, big", [(37, 40, 64), (300, 304, 1024), (300, 304, 304)])
def test_a_row_does_not_depend_on_cols(flint, vocab, small, big):
    # The same logits in a wider row (whatever the extra columns hold): the same bits, and
    # +0 in the extra columns.
    rows = 9
    narrow = random_logits(rows, vocab, small, 5.0, 19)
    wide = random_logits(rows, vocab, big, 5.0, 20)
    wide[:, :small] = narrow
    targets = tokens(random_targets(rows, vocab, 21))

    d_narrow, loss_narrow = flint_ce(flint, narrow, targets, 0.5, vocab)
    d_wide, loss_wide = flint_ce(flint, wide, targets, 0.5, vocab)

    assert_same_bits(d_wide[:, :small], d_narrow, "the narrow columns")
    assert_same_bits(loss_wide, loss_narrow, "row_loss")
    assert all_equal(int_view(d_wide)[:, small:], 0)


@pytest.mark.parametrize("vocab, cols", [(37, 40), (300, 304)])
def test_row_loss_does_not_depend_on_scale_and_scale_0_gives_zeros(flint, vocab, cols):
    rows = 16
    x = random_logits(rows, vocab, cols, 4.0, 22)
    targets = tokens(random_targets(rows, vocab, 23))
    _, first = flint_ce(flint, x, targets, 1.0, vocab, fill=NAN_BITS)
    for scale in (0.125, 2.0**-19, 1e-3, -2.0, 1e6, 3e38, 0.0):
        d, loss = flint_ce(flint, x, targets, scale, vocab, fill=NAN_BITS)
        assert_same_bits(loss, first, f"row_loss at scale {scale:g}")
        if scale == 0.0:
            assert all_equal(d[:, :vocab], 0.0), "scale 0 doesn't give zeros"
            assert all_equal(int_view(d)[:, vocab:], 0), "padding isn't +0 at scale 0"


@pytest.mark.parametrize("vocab, cols", [(37, 40), (300, 304), (1023, 1024)])
@pytest.mark.parametrize("scale", [1.0, 0.125, 2.0**-19, 1000.0])
def test_a_negative_scale_flips_every_sign_below_vocab(flint, vocab, cols, scale):
    rows = 24
    x = random_logits(rows, vocab, cols, 5.0, 24)
    targets = tokens(random_targets(rows, vocab, 25))

    d, loss = flint_ce(flint, x, targets, scale, vocab, fill=NAN_BITS)
    nd, nloss = flint_ce(flint, x, targets, -scale, vocab, fill=NAN_BITS)

    expected = d.view(torch.int16).clone()
    expected[:, :vocab] ^= as_int16(0x8000)
    assert_same_bits(nd, expected.view(torch.float16), "negated scale")
    assert all_equal(int_view(nd)[:, vocab:], 0), "padding isn't +0 at a negative scale"
    assert_same_bits(nloss, loss, "row_loss")


@pytest.mark.parametrize("scale", [1e6, -1e6, 1e5])
@pytest.mark.parametrize("kind", ["wide8", "init"])
def test_overflow_gives_infs_where_the_reference_rounds_to_them_and_no_nan(flint, kind, scale):
    # |scale * (p - 1)| and scale * p pass FP16's 65,520 for most targets and for a wide
    # row's top logit. The result must match the correctly rounded reference (with an inf in
    # both places, or a finite value within an ulp), and no NaN appears.
    rows, vocab, cols = 64, 1000, 1000
    g = generator(26)
    x = make_logits(kind, rows, vocab, cols, g)
    targets = make_targets(kind, x, vocab, g)
    scale = f32(scale)
    d, loss = flint_ce(flint, x, tokens(targets), scale, vocab, fill=NAN_BITS)

    _, ref, _ = reference(x, targets, scale, vocab)
    expected = round_to_half(ref)
    out = d[:, :vocab]
    assert not out.isnan().any() and not loss.isnan().any()
    diff = (half_ordinal(out) - half_ordinal(expected)).abs()
    assert diff.max().item() <= 1
    # Past a hair beyond the threshold the reference is inf, and so is the output; short of
    # it, the output is finite.
    threshold = 65520.0
    clear_over = ref.abs() >= threshold * (1 + 1e-5)
    clear_under = ref.abs() <= threshold * (1 - 1e-5)
    assert all_equal(diff[clear_over], 0) and torch.isinf(out[clear_over]).all()
    assert torch.isfinite(out[clear_under]).all()
    print(
        f"overflow {kind} scale {scale:g}: +inf {(out == math.inf).sum().item()}, -inf "
        f"{(out == -math.inf).sum().item()}, finite {(out.isfinite()).sum().item()}"
    )
    # The overflow happened (at the targets, and for the wide rows at the top logit too), in
    # the sign the reference says.
    assert clear_over.any() and torch.isinf(out).any()
    if kind == "wide8":
        assert (out == math.inf).any() and (out == -math.inf).any()
    assert torch.equal(torch.sign(out[clear_over].double()), torch.sign(ref[clear_over]))
    assert all_equal(int_view(d)[:, vocab:], 0)


@pytest.mark.skipif(SANITIZER, reason="a million rows are too slow under compute-sanitizer")
def test_many_rows_use_many_blocks(flint):
    # 1,000,003 rows is 125,001 blocks, far past 65,535, and the last block is nearly empty.
    rows, vocab, cols = 1_000_003, 37, 40
    logits = random_logits(rows, vocab, cols, 3.0, 27)
    targets = random_targets(rows, vocab, 28)

    d, loss, stats = run_and_check(flint, logits, targets, 0.125, vocab, "a million rows")

    print(f"MANY-ROWS {describe(stats)}")


# -- Non-finite logits ------------------------------------------------------

NF_ROWS, NF_VOCAB, NF_COLS = 60, 300, 304


def injection_column(variant, r, vocab=NF_VOCAB):
    """The column below vocab that variant v damages in row r (a tensor of rows): the first,
    the last below vocab, a chunk boundary mid-row, a spread of other columns that changes
    with the row, and a column of the chunk before the last. Together they land on different
    lanes (there are 38 chunks, so the first lanes hold two)."""
    if variant == 0:
        return torch.zeros_like(r)
    if variant == 1:
        return torch.full_like(r, vocab - 1)
    if variant == 2:
        return torch.full_like(r, (vocab // 2) // 8 * 8)
    if variant == 3:
        return (r * 37 + 11) % vocab
    return torch.full_like(r, 8 * (vocab // 8 - 1) + 5)


def non_finite_rows(variant, kind):
    """A problem of NF_ROWS rows, every third of which has one `kind` of non-finite logit at
    the variant's column, and the clean version of it. Every other injected row has its
    target on the damaged column. Returns (damaged, clean, targets, injected, columns)."""
    clean = random_logits(NF_ROWS, NF_VOCAB, NF_COLS, 2.0, 30)
    targets = random_targets(NF_ROWS, NF_VOCAB, 31)
    injected = torch.arange(0, NF_ROWS, 3, device="cuda")
    column = injection_column(variant, injected)
    targets[injected[1::2]] = column[1::2]
    targets[injected[0::2]] = torch.where(
        targets[injected[0::2]] == column[0::2], (column[0::2] + 1) % NF_VOCAB, targets[injected[0::2]]
    )
    damaged = clean.clone()
    damaged.view(torch.int16)[injected, column] = as_int16(NONFINITE_BITS[kind])
    return damaged, clean, targets, injected, column


@pytest.mark.parametrize("variant", range(5))
@pytest.mark.parametrize("kind", ["nan", "-nan", "snan", "-snan", "inf"])
def test_a_nan_or_plus_inf_spoils_only_its_row(flint, kind, variant):
    damaged, clean, targets, injected, column = non_finite_rows(variant, kind)
    is_injected = torch.zeros(NF_ROWS, dtype=torch.bool, device="cuda")
    is_injected[injected] = True
    clean_d, clean_loss = flint_ce(flint, clean, tokens(targets), 0.5, NF_VOCAB)

    d, loss = flint_ce(flint, damaged, tokens(targets), 0.5, NF_VOCAB)

    # The FP32 reference has NaN in the same places: the loss of the injected rows, and all of
    # their softmax.
    ref_loss = fp32_peer_loss(damaged, targets, NF_VOCAB)
    ref_p = F.softmax(damaged[:, :NF_VOCAB].float(), dim=-1)
    assert torch.equal(ref_loss.isnan(), is_injected)
    assert torch.equal(ref_p.isnan().all(dim=-1), is_injected) and torch.equal(ref_p.isnan().any(dim=-1), is_injected)
    assert torch.equal(loss.isnan(), is_injected), "row_loss NaN places"
    assert torch.equal(d[:, :NF_VOCAB].isnan().all(dim=-1), is_injected), "dlogits NaN rows"
    assert torch.equal(d[:, :NF_VOCAB].isnan().any(dim=-1), is_injected), "dlogits NaN rows"
    assert all_equal(int_view(d)[:, NF_VOCAB:], 0), "padding isn't +0"
    assert_same_bits(d[~is_injected], clean_d[~is_injected], "the other rows' dlogits")
    assert_same_bits(loss[~is_injected], clean_loss[~is_injected], "the other rows' loss")


@pytest.mark.parametrize("scale", [0.0, -0.5])
@pytest.mark.parametrize("kind", ["nan", "inf"])
def test_a_nan_row_stays_nan_at_a_scale_of_0_or_below(flint, kind, scale):
    # The training overflow check has to see a NaN row at any scale, so a scale of 0 can't turn
    # it into zeros.
    damaged, _, targets, injected, _ = non_finite_rows(0, kind)
    is_injected = torch.zeros(NF_ROWS, dtype=torch.bool, device="cuda")
    is_injected[injected] = True

    d, loss = flint_ce(flint, damaged, tokens(targets), scale, NF_VOCAB)

    assert torch.equal(loss.isnan(), is_injected), "row_loss NaN places"
    assert torch.equal(d[:, :NF_VOCAB].isnan().all(dim=-1), is_injected), "dlogits NaN rows"
    assert torch.equal(d[:, :NF_VOCAB].isnan().any(dim=-1), is_injected), "dlogits NaN rows"
    assert all_equal(int_view(d)[:, NF_VOCAB:], 0), "padding isn't +0"


@pytest.mark.parametrize("variant", range(5))
def test_a_minus_inf_among_finite_logits_gets_p_0(flint, variant):
    # Injected rows alternate between a -inf that isn't the target (its gradient is +0 by
    # bits and the rest of the row within an ulp of the softmax of the other logits) and one
    # that is (the loss is +inf and the gradient there is -scale).
    scale = 0.5
    damaged, clean, targets, injected, column = non_finite_rows(variant, "-inf")
    is_injected = torch.zeros(NF_ROWS, dtype=torch.bool, device="cuda")
    is_injected[injected] = True
    clean_d, clean_loss = flint_ce(flint, clean, tokens(targets), scale, NF_VOCAB)

    d, loss = flint_ce(flint, damaged, tokens(targets), scale, NF_VOCAB)

    assert_same_bits(d[~is_injected], clean_d[~is_injected], "the other rows' dlogits")
    assert_same_bits(loss[~is_injected], clean_loss[~is_injected], "the other rows' loss")
    assert all_equal(int_view(d)[:, NF_VOCAB:], 0), "padding isn't +0"
    is_target = targets[injected] == column
    assert is_target.any() and (~is_target).any()
    # Not the target: +0 at the -inf, and p of the rest as without it.
    plain = injected[~is_target]
    assert all_equal(int_view(d)[plain, column[~is_target]], 0), "-inf's gradient isn't +0"
    # The target: loss +inf and gradient -scale.
    hit = injected[is_target]
    assert all_equal(loss[hit], math.inf), "the loss at a -inf target isn't +inf"
    minus_scale = round_to_half(torch.tensor(-scale, dtype=torch.float64, device="cuda"))
    assert all_equal(d[hit, column[is_target]], minus_scale.item())
    stats = analyze(d[injected], loss[injected], damaged[injected], targets[injected], scale, NF_VOCAB)
    assert_tight(stats, f"-inf at variant {variant}")
    assert torch.isfinite(d[:, :NF_VOCAB]).all()


@pytest.mark.parametrize("vocab, cols", [(1, 8), (1, 64), (37, 40), (300, 304), (1023, 1024)])
def test_a_row_of_only_minus_inf_is_nan(flint, vocab, cols):
    # Every other row is all -inf below vocab (the padding finite); the rows between are
    # finite and bitwise the clean run's.
    rows = 10
    clean = random_logits(rows, vocab, cols, 2.0, 32)
    targets = tokens(random_targets(rows, vocab, 33))
    damaged = clean.clone()
    damaged[::2, :vocab] = -math.inf
    nan_rows = torch.zeros(rows, dtype=torch.bool, device="cuda")
    nan_rows[::2] = True
    clean_d, clean_loss = flint_ce(flint, clean, targets, 1.0, vocab)

    d, loss = flint_ce(flint, damaged, targets, 1.0, vocab)

    ref = F.cross_entropy(damaged[:, :vocab].float(), targets.long(), reduction="none")
    assert torch.equal(ref.isnan(), nan_rows)
    assert torch.equal(loss.isnan(), nan_rows)
    assert torch.equal(d[:, :vocab].isnan().all(dim=-1), nan_rows)
    assert torch.equal(d[:, :vocab].isnan().any(dim=-1), nan_rows)
    assert all_equal(int_view(d)[:, vocab:], 0)
    assert_same_bits(d[~nan_rows], clean_d[~nan_rows], "the other rows' dlogits")
    assert_same_bits(loss[~nan_rows], clean_loss[~nan_rows], "the other rows' loss")


@pytest.mark.parametrize("first, second", [("inf", "-inf"), ("nan", "inf"), ("nan", "-inf")])
def test_two_kinds_of_non_finite_logit_in_a_row_give_nan(flint, first, second):
    # Every third row from row 3 has one of each kind, at columns that long rows put on
    # different lanes. The whole row is NaN, like the reference's, and the rows between are
    # bitwise the clean run's.
    vocab, cols, rows = NF_VOCAB, NF_COLS, NF_ROWS
    clean = random_logits(rows, vocab, cols, 2.0, 34)
    targets = tokens(random_targets(rows, vocab, 35))
    injected = torch.arange(3, rows, 3, device="cuda")
    damaged = clean.clone()
    damaged.view(torch.int16)[injected, (injected * 7) % vocab] = as_int16(NONFINITE_BITS[first])
    damaged.view(torch.int16)[injected, (injected * 7 + 101) % vocab] = as_int16(NONFINITE_BITS[second])
    is_injected = torch.zeros(rows, dtype=torch.bool, device="cuda")
    is_injected[injected] = True
    clean_d, clean_loss = flint_ce(flint, clean, targets, 1.0, vocab)

    d, loss = flint_ce(flint, damaged, targets, 1.0, vocab)

    assert torch.equal(fp32_peer_loss(damaged, targets.long(), vocab).isnan(), is_injected)
    assert torch.equal(loss.isnan(), is_injected)
    assert torch.equal(d[:, :vocab].isnan().all(dim=-1), is_injected)
    assert torch.equal(d[:, :vocab].isnan().any(dim=-1), is_injected)
    assert all_equal(int_view(d)[:, vocab:], 0)
    assert_same_bits(d[~is_injected], clean_d[~is_injected], "the other rows' dlogits")
    assert_same_bits(loss[~is_injected], clean_loss[~is_injected], "the other rows' loss")


# -- A target out of range ----------------------------------------------------


@pytest.mark.parametrize("target", [37, 39, 40, 65535])
def test_a_target_at_or_above_vocab_touches_nothing_outside_the_buffers(flint, target):
    # flint.h leaves such a row's results unspecified, and its row_loss may stay unwritten, so
    # this can't use flint_ce, which checks that every element is written. Nothing outside the
    # buffers may be read or written: the last row's target would be far past the end of logits
    # if the kernel loaded it, which the sanitizer's memcheck sees. The other rows keep their
    # results. 37 and 39 are past vocab in its last chunk, 40 and 65535 past cols.
    rows, vocab, cols = 8, 37, 40
    logits = random_logits(rows, vocab, cols, 1.0, 40)
    targets = random_targets(rows, vocab, 41)
    clean_d, clean_loss = flint_ce(flint, logits, tokens(targets), 1.0, vocab)
    targets[-1] = target
    count = rows * cols
    buf = torch.full((GUARD_AFTER + count + GUARD_AFTER,), as_int16(SENTINEL), dtype=torch.int16, device="cuda")
    fill32 = as_int32((SENTINEL << 16) | SENTINEL)
    loss_buf = torch.full((GUARD_AFTER + rows + GUARD_AFTER,), fill32, dtype=torch.int32, device="cuda")
    d = buf[GUARD_AFTER : GUARD_AFTER + count]
    loss = loss_buf[GUARD_AFTER : GUARD_AFTER + rows]

    flint.softmax_cross_entropy(d, loss, logits, tokens(targets), 1.0, rows, vocab, cols)
    torch.cuda.synchronize()

    assert all_equal(buf[:GUARD_AFTER], as_int16(SENTINEL)), "wrote before dlogits"
    assert all_equal(buf[GUARD_AFTER + count :], as_int16(SENTINEL)), "wrote past dlogits"
    assert all_equal(loss_buf[:GUARD_AFTER], fill32), "wrote before row_loss"
    assert all_equal(loss_buf[GUARD_AFTER + rows :], fill32), "wrote past row_loss"
    assert_same_bits(d.view(torch.float16).view(rows, cols)[:-1], clean_d[:-1], "the other rows' dlogits")
    assert_same_bits(loss.view(torch.float32)[:-1], clean_loss[:-1], "the other rows' loss")


# -- Determinism ------------------------------------------------------------


@pytest.mark.parametrize("kind, rows, vocab, cols", [("outliers", 33, 1023, 1024), ("wide30", 20, 300, 304), ("range", 9, 37, 40)], ids=str)
def test_run_twice_is_bitwise_identical_including_nan_rows(flint, kind, rows, vocab, cols):
    g = generator(0)
    x = make_logits(kind, rows, vocab, cols, g)
    targets = tokens(make_targets(kind, x, vocab, g))
    # NaN, +inf and -inf logits in a third of the rows.
    r = torch.arange(0, rows, 3, device="cuda")
    x.view(torch.int16)[r, r % vocab] = as_int16(NONFINITE_BITS["nan"])
    x.view(torch.int16)[r[1::2] + 1, (r[1::2] * 5) % vocab] = as_int16(NONFINITE_BITS["inf"])
    x.view(torch.int16)[-1, 0] = as_int16(NONFINITE_BITS["-inf"])

    first = flint_ce(flint, x, targets, 0.3, vocab, fill=NAN_BITS)
    second = flint_ce(flint, x, targets, 0.3, vocab, fill=OTHER_BITS)
    third = flint_ce(flint, x, targets, 0.3, vocab, fill=SENTINEL)

    assert first[1].isnan().any() and first[0].isnan().any()
    for name, other in (("second", second), ("third", third)):
        assert_same_bits(other[0], first[0], f"{name} run's dlogits")
        assert_same_bits(other[1], first[1], f"{name} run's row_loss")


# -- The limits -------------------------------------------------------------


@pytest.mark.parametrize("vocab", [65536, 65535, 65529, 40000])
def test_the_widest_rows(flint, vocab):
    rows, cols = (5 if SANITIZER else 17), 65536
    logits = random_logits(rows, vocab, cols, 3.0, 36)
    targets = random_targets(rows, vocab, 37)
    targets[:3] = torch.tensor([0, vocab - 1, vocab // 2], device="cuda")

    _, _, stats = run_and_check(flint, logits, targets, 1.0, vocab, f"cols 65,536, vocab {vocab}")

    print(f"WIDEST vocab {vocab}: {describe(stats)}")


# -- Streams ----------------------------------------------------------------

STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
def test_runs_on_the_given_stream(flint, mode):
    rows, vocab, cols = 16, 100, 104
    x = torch.full((rows, cols), math.nan, dtype=torch.float16, device="cuda")
    targets = torch.zeros(rows, dtype=torch.int16, device="cuda")
    d = torch.full((rows, cols), math.nan, dtype=torch.float16, device="cuda")
    loss = torch.full((rows,), math.nan, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fill finishes only after the sleep, so the loss is log(100) only if the kernel is
    # queued on the same stream behind it (it is NaN where it ran first). With the current
    # stream the call has no stream argument, and torch's current stream is the side one;
    # with the default stream there is no side one at all.
    if mode == "default-stream":
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(0)
        flint.softmax_cross_entropy(d, loss, x, targets, 1.0, rows, vocab, cols)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            x.fill_(0)
            flint.softmax_cross_entropy(
                d, loss, x, targets, 1.0, rows, vocab, cols, stream=stream if mode == "explicit-stream" else None
            )
        stream.synchronize()

    assert not torch.isnan(d[:, :vocab]).any() and not torch.isnan(loss).any()
    assert_tight(analyze(d, loss, x, targets.long(), 1.0, vocab), mode)


# -- Argument errors --------------------------------------------------------

NAMES = ("dlogits", "row_loss", "logits", "targets")
SMALL = dict(rows=8, vocab=20, cols=24, scale=1.0)
BYTES = {"dlogits": 8 * 24 * 2, "row_loss": 8 * 4, "logits": 8 * 24 * 2, "targets": 8 * 2}


def small_problem():
    """Valid tensors by name, in buffers with room to spare, the outputs filled with 0.5."""
    return {
        "dlogits": torch.full((8 * 24 + 64,), 0.5, dtype=torch.float16, device="cuda"),
        "row_loss": torch.full((8 + 64,), 0.5, dtype=torch.float32, device="cuda"),
        "logits": torch.zeros(8 * 24 + 64, dtype=torch.float16, device="cuda"),
        "targets": torch.zeros(8 + 64, dtype=torch.int16, device="cuda"),
    }


def call(flint, buffers, **sizes):
    sizes = {**SMALL, **sizes}
    flint.softmax_cross_entropy(
        buffers["dlogits"], buffers["row_loss"], buffers["logits"], buffers["targets"],
        sizes["scale"], sizes["rows"], sizes["vocab"], sizes["cols"],
    )


def expect_invalid_argument(flint, reason, buffers, **sizes):
    """The call fails with the status and message, and no tensor in the buffers changed."""
    before = {n: b.clone() for n, b in buffers.items() if isinstance(b, torch.Tensor)}
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, buffers, **sizes)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_softmax_cross_entropy", reason):
        assert text in str(error.value)
        assert text in flint.last_error()
    torch.cuda.synchronize()
    for name, tensor in before.items():
        assert torch.equal(buffers[name], tensor), f"{name} changed"


def nothing():
    return {name: None for name in NAMES}


def addresses():
    """The small problem's buffers as plain addresses (the tensors stay alive in the
    returned dict)."""
    tensors = small_problem()
    return {n: t.data_ptr() for n, t in tensors.items()}, tensors


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    expect_invalid_argument(flint, f"{name} is null", {**small_problem(), name: None})


@pytest.mark.parametrize("offset", [2, 8])
@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name, offset):
    # 2 bytes is as far off as an element goes, and 8 is aligned for 64 bits but not 128.
    a, tensors = addresses()
    a[name] += offset
    expect_invalid_argument(flint, f"{name} is not 16-byte aligned", a)


@pytest.mark.parametrize("name", NAMES)
def test_a_buffer_may_not_run_past_the_address_space(flint, name):
    expect_invalid_argument(
        flint, f"{name} runs past the end of the address space", {**small_problem(), name: 2**64 - 16}
    )


def test_pointers_are_checked_in_order(flint):
    # Per buffer: null, then alignment, then the address space; dlogits, row_loss, logits,
    # targets; and all of that before any overlap.
    a, tensors = addresses()
    expect_invalid_argument(flint, "dlogits is not 16-byte aligned", {**a, "dlogits": a["dlogits"] + 2, "row_loss": None})
    expect_invalid_argument(flint, "dlogits is null", {**a, "dlogits": None, "row_loss": a["row_loss"] + 2})
    expect_invalid_argument(flint, "row_loss is not 16-byte aligned", {**a, "row_loss": a["row_loss"] + 2, "logits": None})
    expect_invalid_argument(flint, "logits is null", {**a, "logits": None, "targets": a["targets"] + 2})
    expect_invalid_argument(flint, "targets is not 16-byte aligned", {**a, "targets": a["targets"] + 2})
    expect_invalid_argument(flint, "dlogits runs past the end", {**a, "dlogits": 2**64 - 16, "row_loss": None})
    expect_invalid_argument(flint, "logits is null", {**a, "row_loss": a["dlogits"], "logits": None})


@pytest.mark.parametrize(
    "cols", [0, 1, 4, 7, 9, 12, 20, 65_537, 65_544, 2**32, 2**64 - 8, 2**64 - 1], ids=str
)
def test_cols_must_be_a_multiple_of_8_up_to_65536(flint, cols):
    expect_invalid_argument(
        flint, f"cols ({cols}) must be a multiple of 8 from 8 to 65536", small_problem(), cols=cols, vocab=8
    )


@pytest.mark.parametrize("vocab", [0, 25, 2**32, 2**64 - 1], ids=str)
def test_vocab_must_be_from_1_to_cols(flint, vocab):
    expect_invalid_argument(flint, f"vocab ({vocab}) must be from 1 to cols (24)", small_problem(), vocab=vocab)


@pytest.mark.parametrize("rows", [MAX_ROWS + 1, MAX_ROWS + 8, 2**40, 2**63, 2**64 - 1], ids=str)
def test_rows_can_be_up_to_17179869176(flint, rows):
    expect_invalid_argument(flint, f"rows ({rows}) can be up to {MAX_ROWS}", small_problem(), rows=rows)


@pytest.mark.parametrize("scale", [math.inf, -math.inf, math.nan], ids=str)
def test_scale_must_be_finite(flint, scale):
    expect_invalid_argument(flint, ") must be finite", small_problem(), scale=scale)
    expect_invalid_argument(flint, "scale (", small_problem(), scale=scale)


@pytest.mark.parametrize(
    "sizes",
    [dict(rows=MAX_ROWS, cols=MAX_COLS, vocab=MAX_COLS), dict(rows=MAX_ROWS, cols=MAX_COLS, vocab=50257),
     dict(rows=MAX_ROWS), dict(cols=MAX_COLS, vocab=1), dict(rows=1, cols=8, vocab=1), dict(vocab=24),
     dict(scale=3.0e38), dict(scale=-3.0e38), dict(scale=0.0)],
    ids=lambda sizes: "-".join(f"{k}={v}" for k, v in sizes.items()),
)
def test_the_largest_sizes_pass_the_size_checks(flint, sizes):
    # The limits themselves aren't errors of size: the call gets on to the pointers (a null
    # dlogits), even where rows * cols * 2 is 2.3e15 bytes.
    expect_invalid_argument(flint, "dlogits is null", {**small_problem(), "dlogits": None}, **sizes)


def test_size_limits_are_checked_in_order(flint):
    # cols, then vocab, then rows, then scale, with the pointers (null ones included) after
    # them; the empty-problem exit comes after the size checks.
    expect_invalid_argument(flint, "cols (7) must be", nothing(), cols=7, vocab=0, rows=2**64 - 1, scale=math.inf)
    expect_invalid_argument(flint, "cols (0) must be", nothing(), cols=0, vocab=0, rows=0)
    expect_invalid_argument(flint, "cols (65544) must be", nothing(), cols=65_544, vocab=65_544, rows=0)
    expect_invalid_argument(flint, "vocab (0) must be", nothing(), vocab=0, rows=2**64 - 1, scale=math.nan)
    expect_invalid_argument(flint, "vocab (25) must be", nothing(), vocab=25, rows=0)
    expect_invalid_argument(flint, f"rows ({MAX_ROWS + 1}) can be", nothing(), rows=MAX_ROWS + 1, scale=math.inf)
    expect_invalid_argument(flint, "scale (", nothing(), scale=math.inf, rows=0)
    expect_invalid_argument(flint, "scale (", nothing(), scale=math.nan)


def test_a_size_error_takes_priority_over_a_pointer_error_at_the_largest_rows(flint):
    # Where rows * cols * 2 would be the biggest allowed, and past it.
    expect_invalid_argument(
        flint, f"rows ({MAX_ROWS + 1}) can be up to", nothing(), rows=MAX_ROWS + 1, cols=MAX_COLS, vocab=MAX_COLS
    )
    expect_invalid_argument(
        flint, f"rows ({2**64 - 1}) can be up to", {**nothing(), "dlogits": 2**64 - 16}, rows=2**64 - 1, cols=MAX_COLS, vocab=1
    )
    expect_invalid_argument(
        flint, f"cols ({MAX_COLS + 8}) must be", nothing(), rows=MAX_ROWS, cols=MAX_COLS + 8, vocab=MAX_COLS
    )
    expect_invalid_argument(flint, "dlogits is null", nothing(), rows=MAX_ROWS, cols=MAX_COLS, vocab=MAX_COLS)
    expect_invalid_argument(
        flint, "row_loss runs past the end of the address space",
        {**small_problem(), "row_loss": 2**64 - 16}, rows=MAX_ROWS, cols=MAX_COLS, vocab=MAX_COLS,
    )


SPACING = 4096
ARENA_SLOT = {name: (i + 2) * SPACING for i, name in enumerate(NAMES)}
OUTPUT_PAIRS = [("dlogits", "row_loss"), ("dlogits", "logits"), ("dlogits", "targets"),
                ("row_loss", "logits"), ("row_loss", "targets")]  # fmt: skip


def overlapping_starts(first, second):
    """Where `second` starts relative to `first`, 16-byte aligned, so that they overlap: at
    the same address, 16 bytes in, 16 bytes before first's end, and (negative) with second
    ending 16 bytes into first."""
    a, b = BYTES[first], BYTES[second]
    return {"same-start": 0, "16-in": 16, "16-before-end": a - 16, "ends-16-in": 16 - b}


def touching_starts(first, second):
    """Where `second` starts so that it touches `first` without overlapping it: just after,
    and just before."""
    return {"just-after": BYTES[first], "just-before": -BYTES[second]}


OVERLAP_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in OUTPUT_PAIRS
    for label, offset in overlapping_starts(a, b).items()
]


@pytest.mark.parametrize("first, second, label, offset", OVERLAP_CASES)
def test_an_output_may_not_overlap_another_buffer(flint, first, second, label, offset):
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no
    # harm to anything else.
    arena = torch.zeros(10 * SPACING, dtype=torch.uint8, device="cuda")
    a = {name: arena.data_ptr() + ARENA_SLOT[name] for name in NAMES}
    a[second] = a[first] + offset
    before = arena.clone()
    expect_invalid_argument(flint, f"{first} and {second} overlap", a)
    assert torch.equal(arena, before)


TOUCHING_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in OUTPUT_PAIRS
    for label, offset in touching_starts(a, b).items()
]


@pytest.mark.parametrize("first, second, label, offset", TOUCHING_CASES)
def test_buffers_next_to_each_other_are_not_overlapping(flint, first, second, label, offset):
    arena = torch.zeros(10 * SPACING, dtype=torch.uint8, device="cuda")
    start = {name: ARENA_SLOT[name] for name in NAMES}
    start[second] = start[first] + offset
    fill = 0x5A
    for name in ("dlogits", "row_loss"):
        arena[start[name] : start[name] + BYTES[name]] = fill
    d = arena[start["dlogits"] : start["dlogits"] + BYTES["dlogits"]].view(torch.float16)
    loss = arena[start["row_loss"] : start["row_loss"] + BYTES["row_loss"]].view(torch.float32)
    logits = arena[start["logits"] : start["logits"] + BYTES["logits"]].view(torch.float16)
    targets = arena[start["targets"] : start["targets"] + BYTES["targets"]].view(torch.int16)
    outside = torch.ones(arena.numel(), dtype=torch.bool, device="cuda")
    for name in ("dlogits", "row_loss"):
        outside[start[name] : start[name] + BYTES[name]] = False
    others = arena[outside].clone()

    flint.softmax_cross_entropy(d, loss, logits, targets, 1.0, 8, 20, 24)
    torch.cuda.synchronize()

    assert torch.equal(arena[outside], others), "something outside the outputs changed"
    assert none_equal(d.view(torch.int16), as_int16(0x5A5A))
    assert none_equal(loss.view(torch.int32), as_int32(0x5A5A5A5A))
    # All logits are 0 and every target 0: the loss is log(20), and the gradients are 1/20
    # (and -19/20 at column 0).
    assert_tight(analyze(d.view(8, 24), loss, logits.view(8, 24), targets.long(), 1.0, 20), label)


SHARED_STARTS = {"same-start": 0, "16-in": 16, "in-the-middle": 192, "at-the-end": BYTES["logits"] - BYTES["targets"]}


@pytest.mark.parametrize("label", SHARED_STARTS)
def test_logits_and_targets_may_share_memory(flint, label):
    # They are only read. The shared bytes are 0 (every logit is 0 and every target column 0).
    arena = torch.zeros(10 * SPACING, dtype=torch.uint8, device="cuda")
    arena[ARENA_SLOT["dlogits"] : ARENA_SLOT["dlogits"] + BYTES["dlogits"]] = 0x5A
    arena[ARENA_SLOT["row_loss"] : ARENA_SLOT["row_loss"] + BYTES["row_loss"]] = 0x5A
    d = arena[ARENA_SLOT["dlogits"] :][: BYTES["dlogits"]].view(torch.float16)
    loss = arena[ARENA_SLOT["row_loss"] :][: BYTES["row_loss"]].view(torch.float32)
    logits = arena[ARENA_SLOT["logits"] :][: BYTES["logits"]].view(torch.float16)
    targets = arena[ARENA_SLOT["logits"] + SHARED_STARTS[label] :][: BYTES["targets"]].view(torch.int16)

    flint.softmax_cross_entropy(d, loss, logits, targets, 1.0, 8, 20, 24)
    torch.cuda.synchronize()

    assert none_equal(loss.view(torch.int32), as_int32(0x5A5A5A5A))
    assert_tight(analyze(d.view(8, 24), loss, logits.view(8, 24), torch.zeros(8, dtype=torch.int64, device="cuda"), 1.0, 20), label)


@pytest.mark.parametrize(
    "rows, vocab, cols",
    [(0, 20, 24), (0, 1, 8), (0, 50257, 50304), (0, MAX_COLS, MAX_COLS)],
    ids=lambda v: str(v),
)
def test_empty_problems_do_nothing(flint, rows, vocab, cols):
    # Not even the null pointers are looked at (the sizes are).
    buffers = small_problem()

    flint.softmax_cross_entropy(None, None, None, None, 1.0, rows, vocab, cols)
    flint.softmax_cross_entropy(
        buffers["dlogits"], buffers["row_loss"], buffers["logits"], buffers["targets"], 1.0, rows, vocab, cols
    )
    torch.cuda.synchronize()

    assert all_equal(buffers["dlogits"], 0.5) and all_equal(buffers["row_loss"], 0.5)


def test_a_valid_call_works_after_errors(flint):
    buffers = small_problem()
    for sizes in (dict(cols=7), dict(vocab=25), dict(rows=MAX_ROWS + 1), dict(cols=2**64 - 1), dict(scale=math.nan)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, buffers, **sizes)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "dlogits": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "targets": buffers["targets"].data_ptr() + 2})
    torch.cuda.synchronize()
    assert all_equal(buffers["dlogits"], 0.5) and all_equal(buffers["row_loss"], 0.5)

    call(flint, buffers)
    torch.cuda.synchronize()

    # All logits are 0 and every target is column 0.
    d = buffers["dlogits"][: 8 * 24].view(8, 24)
    loss = buffers["row_loss"][:8]
    logits = buffers["logits"][: 8 * 24].view(8, 24)
    assert_tight(analyze(d, loss, logits, torch.zeros(8, dtype=torch.int64, device="cuda"), 1.0, 20), "after errors")
    assert all_equal(buffers["dlogits"][8 * 24 :], 0.5) and all_equal(buffers["row_loss"][8:], 0.5)
