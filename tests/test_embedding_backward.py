"""flint_embedding_backward: the embedding's backward pass. dx is (sequences * length) x width FP32
(row r = s * length + t), tokens are sequences * length uint16 values, and the function computes

  dwte[v] = the sum of dx[r] over the rows r with tokens[r] == v, for v below vocab: vocab x width
            FP32, written to every row (+0 where no token reaches), or with accumulate_wte added to
            the rows some token reaches with one FP32 add (the other rows are left as they were). A
            token at or above vocab reaches no row.
  dwpe[t] = the sum over the sequences s of dx[s * length + t], for t below length: the first length
            rows of dwpe, written, or with accumulate_wpe added with one FP32 add. Rows past length
            are never touched.

Every sum is compensated (TwoSum, as flint_bias_gradient's), with every operation an IEEE FP32
round-to-nearest one and no FMA, in an order that depends only on the shapes and the inputs:

  - dwte[v]'s column j: s, e start at +0; for each row r of token v in ASCENDING order of r, with
    x = dx[r, j]: t = s + x; vp = t - s; e = e + ((s - (t - vp)) + (x - vp)); s = t. The value is
    s + e, or s alone when s is not finite. (A stable sort by token, two radix passes of 8 bits,
    makes the rows of a token ascending.)
  - dwpe is flint_bias_gradient's column sum of dx viewed as sequences x (length * width), in the
    same two passes with the same chunks of 256 rows and thread order (see test_bias_gradient.py),
    on FP32 values instead of FP16.

Rows with only -0 sum to +0 (the sums start at +0), as PyTorch's do.

The workspace holds, from its start: the position partials (the bias gradient's workspace layout for
sequences x (length * width)), the sorted tokens (rows uint16), the sorted rows (rows uint32), the
offsets (vocab + 1 uint32 values: how many tokens are below v), then the sort's first pass and its
per-tile counts, each part 16-byte aligned. The tests read the first four parts.

What the tests cover:
  - a numpy float32 emulation of all of it (the stable sort, the offsets, dwte's sums, dwpe through
    the bias gradient's passes), checked against a literal scalar transcription (non-finite values,
    -0 and tokens past vocab included) and, for dwpe, against test_bias_gradient's emulation and
    its scalar transcription;
  - Flint against the emulation bit for bit: dwte, dwpe and the workspace's sorted tokens, sorted
    rows, offsets and partial sums, over GPT-2's shapes and odd ones (rows not a multiple of the
    sort's tile of 2,048, exactly one tile, one more, 2^20 rows for the scan's carry), five kinds of
    tokens (uniform, Zipf, one token 3,000 times among others, all one token, a few) and five of dx
    (N(0, 1), gradient scale, outlier channels near +3,000 and +-3,000, magnitudes from 2^-30 to
    65,504 with FP32 subnormals), and tokens that share a low byte but not a high one;
  - crafted runs of rows, whose sums show the order (rows descending, plain sums, chunked merges,
    even and odd rows merged, the first row last each change many columns), placed at random rows
    of a larger batch, and the bias gradient's crafted inputs through dwpe;
  - the correctness gate (PLAN.md: at most 3x the error of FP32 PyTorch against FP64, by relative L2
    and max-abs, with FP32 floors) and the owner's tighter bound of 2x, for dwte and dwpe;
  - exact cases, accumulate (olds of every kind, each flag alone, micro-batches), write mode over
    any old content, vocabulary edges (tokens 0 and vocab - 1, a vocab of 1 and of 65,536, GPT-2's
    padded table, tokens at or past vocab), edge structures, inf and NaN, determinism (with garbage
    left in the workspace by another call), streams, the workspace's size, the limits and the
    argument errors.

Inputs are made on the CPU with seeded generators and copied over, so they are the same on every
machine. Under the sanitizer (scripts/sanitize.py) the shapes shrink to at most 300,000 elements of
dx, and the tests of the limits are skipped."""

import ctypes
import math
from collections import namedtuple

import numpy as np
import pytest
import torch

import flint_lib
import gate
import test_bias_gradient as bg
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES
from test_bias_gradient import (  # noqa: F401  (free_gpu_memory is an autouse fixture)
    FILLS,
    GUARD_AFTER,
    NAN32,
    SENTINEL,
    all_equal,
    assert_same_bits,
    free_gpu_memory,
    set_words,
)

F32 = np.float32
CHUNK_ROWS = bg.CHUNK_ROWS
MAX_WIDTH = bg.MAX_COLS  # 16,776,960
MAX_POSITION_VALUES = bg.MAX_COLS
MAX_VOCAB = 65536
MAX_ROWS = 0xFFFFFFFF
TILE_ROWS = 2048
DIGITS = 256
GPT2_VOCAB = 50304
NAME = "flint_embedding_backward"
WORKSPACE_NAME = "flint_embedding_backward_workspace"

# Elements of dx a sanitizer run may use.
SANITIZER_ELEMENTS = 300_000
# Elements per batch of the emulation's pass 1 (small enough to stay in the CPU's cache).
BATCH_ELEMENTS = 1 << 20
CHECK_INPUT_ELEMENTS = bg.CHECK_INPUT_ELEMENTS

Result = namedtuple("Result", "dwte dwpe workspace")
Expected = namedtuple("Expected", "dwte dwpe sorted_tokens sorted_rows offsets partial_s partial_e reached")


def case_id(case):
    return "x".join(str(v) for v in case[:4]) + "".join(f"-{v}" for v in case[4:])


def shrunk_cases(cases):
    """The cases (sequences, length, vocab, width, ...), with those too big for the sanitizer
    dropped, and duplicates."""
    if SANITIZER:
        cases = [c for c in cases if c[0] * c[1] * c[3] <= SANITIZER_ELEMENTS and c[2] * c[3] <= 2 * SANITIZER_ELEMENTS]
    return list(dict.fromkeys(cases))


def case_params(cases):
    return [pytest.param(*case, id=case_id(case)) for case in cases]


# -- The workspace's layout -------------------------------------------------------------------------


def align16(n):
    return (n + 15) // 16 * 16


def layout(sequences, length, vocab, width):
    """The workspace's parts, by name, in bytes from its start (the documented layout), and its
    total size."""
    rows = sequences * length
    tiles = -(-rows // TILE_ROWS)
    parts = {"position_partials": 0}
    at = align16(bg.expected_workspace(sequences, length * width))
    for name, size in (
        ("sorted_tokens", rows * 2),
        ("sorted_rows", rows * 4),
        ("offsets", (vocab + 1) * 4),
        ("pass_tokens", rows * 2),
        ("pass_rows", rows * 4),
        ("counts", tiles * DIGITS * 4),
    ):
        parts[name] = at
        at += align16(size)
    parts["bytes"] = at
    return parts


# -- Inputs --------------------------------------------------------------------------------------------


def make_dx(kind, rows, width, seed):
    """dx (rows x width float32, numpy) from a seeded generator: N(0, 1); gradient scale (x 2^-12);
    outlier channels (2% of the columns, and one at least) near +3,000, all one sign or of random
    sign, the rest N(0, 1); or wide magnitudes, 2^-30 to 65,504 of random sign, with 3% FP32
    subnormals."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, width), dtype=F32)
    if kind == "normal":
        return x
    if kind == "gradient":
        return x * F32(2.0**-12)
    if kind in ("outliers_same", "outliers_mixed"):
        hit = rng.random(width) < 0.02
        hit[width // 3] = True
        n = int(hit.sum())
        big = F32(3000) * (F32(1) + F32(0.01) * rng.standard_normal((rows, n), dtype=F32))
        if kind == "outliers_mixed":
            big = np.where(rng.random((rows, n), dtype=F32) < F32(0.5), -big, big)
        x[:, hit] = big
        return x.astype(F32)
    if kind == "wide":
        e = rng.random((rows, width), dtype=F32) * F32(45) - F32(30)
        mag = np.minimum(np.exp2(e) * (F32(1) + rng.random((rows, width), dtype=F32)), F32(65504))
        subnormal = rng.integers(1, 2**23, (rows, width), dtype=np.int32).astype(F32) * F32(2.0**-149)
        out = np.where(rng.random((rows, width), dtype=F32) < F32(0.03), subnormal, mag)
        return np.where(rng.random((rows, width), dtype=F32) < F32(0.5), -out, out).astype(F32)
    raise ValueError(kind)


# Tokens that share a low byte and differ in the high one, and the other way round.
COLLIDE_TOKENS = [0x0001, 0x0101, 0xFF01, 0x0100, 0x0102, 0x01FF, 0xFF00, 0xFFFF, 0x0000, 0x8001, 0x0181, 0x7F01, 0x0201]


def make_tokens(kind, rows, vocab, seed):
    """Tokens (numpy uint16): uniform; Zipf-like (the token of rank r with weight 1 / r, the ranks
    shuffled over the vocabulary); one token 3,000 times (a third of the rows if fewer) among
    uniform ones; one token for every row; four tokens; the COLLIDE_TOKENS (vocab 65,536); all
    different (rows <= vocab); only tokens 0, vocab - 1 and vocab // 2, so that runs sit at the
    start and the end of the sorted array; or multiples of 100, with empty rows of the table
    between."""
    rng = np.random.default_rng(seed + 7919)
    if kind == "uniform":
        t = rng.integers(0, vocab, rows)
    elif kind == "zipf":
        weights = 1.0 / np.arange(1, vocab + 1)
        ranks = np.searchsorted(np.cumsum(weights) / weights.sum(), rng.random(rows))
        t = rng.permutation(vocab)[np.minimum(ranks, vocab - 1)]
    elif kind == "repeated":
        t = rng.integers(0, vocab, rows)
        count = min(rows, 3000, max(2, rows // 3))
        t[rng.choice(rows, count, replace=False)] = rng.integers(0, vocab)
    elif kind == "one":
        t = np.full(rows, rng.integers(0, vocab))
    elif kind == "few":
        few = rng.choice(vocab, min(vocab, 4), replace=False)
        t = few[rng.integers(0, len(few), rows)]
    elif kind == "collide":
        assert vocab > 0xFF01
        t = np.array(COLLIDE_TOKENS)[rng.integers(0, len(COLLIDE_TOKENS), rows)]
    elif kind == "distinct":
        assert rows <= vocab
        t = rng.permutation(vocab)[:rows]
    elif kind == "ends":
        t = rng.choice([0, vocab - 1, vocab // 2], rows, p=[0.3, 0.3, 0.4])
    elif kind == "gaps":
        t = rng.integers(0, max(1, vocab // 100), rows) * 100
    else:
        raise ValueError(kind)
    return t.astype(np.uint16)


def tokens_tensor(tokens):
    """The uint16 tokens (numpy) as the int16 bit patterns torch holds, on the GPU."""
    return torch.from_numpy(np.ascontiguousarray(tokens).view(np.int16)).cuda()


# -- The emulation of the kernels' arithmetic ------------------------------------------------------------


def emulate_words(dx, tokens, vocab, old=None):
    """The kernels' dwte, as a float32 numpy array (vocab x width), for dx (rows x width float32)
    and tokens (uint16), by the module docstring's arithmetic, in float32 throughout, vectorized
    over columns and over the tokens (step k adds the k-th row of every token that has one). With
    old (vocab x width), the rows some token reaches are old + the sum with one add, and the others
    are old's bits. Also returns the sort's results as the workspace holds them (sorted tokens, sorted
    rows and offsets, uint16 / uint32 / uint32 numpy arrays) and the mask of the reached rows."""
    rows, width = dx.shape
    tok = tokens.astype(np.int64)
    order = np.argsort(tok, kind="stable")
    offsets = np.searchsorted(tok[order], np.arange(vocab + 1), side="left")
    counts = np.diff(offsets)
    present = np.nonzero(counts)[0]
    by_count = present[np.argsort(-counts[present], kind="stable")]
    starts = offsets[by_count]
    n = counts[by_count]
    s = np.zeros((len(by_count), width), dtype=F32)
    e = np.zeros((len(by_count), width), dtype=F32)
    with np.errstate(all="ignore"):
        steps = int(n[0]) if len(n) else 0
        # n is in descending order: the tokens with more than k rows come first.
        active = np.searchsorted(-n, -np.arange(steps), side="left")
        for k in range(steps):
            a = int(active[k])
            x = dx[order[starts[:a] + k]]
            s[:a], e[:a] = bg.two_sum_add(s[:a], e[:a], x)
        value = np.where(np.isfinite(s), s + e, s).astype(F32)
        if old is None:
            out = np.zeros((vocab, width), dtype=F32)
            out[by_count] = value
        else:
            out = np.array(old, dtype=F32, copy=True)
            out[by_count] = old[by_count] + value
    reached = counts > 0
    return out, tokens[order], order.astype(np.uint32), offsets.astype(np.uint32), reached


def position_pass1(x):
    """The bias gradient's pass 1 on x (rows x cols float32): the chunks' partial sums and errors,
    (chunks, cols) each. Thread y of 8 adds rows 256 c + y + 8 i in order, thread 0 merges 1..7."""
    rows, cols = x.shape
    chunks = -(-rows // CHUNK_ROWS)
    full = rows // CHUNK_ROWS
    partial_s = np.empty((chunks, cols), dtype=F32)
    partial_e = np.empty((chunks, cols), dtype=F32)

    def thread_merges(s, e, threads):
        a = (s[:, 0].copy(), e[:, 0].copy())
        for y in bg.Y_ORDER:
            if y < threads:
                a = bg.two_sum_merge(a, (s[:, y], e[:, y]))
        return a

    step = max(1, BATCH_ELEMENTS // (CHUNK_ROWS * cols))
    for c0 in range(0, full, step):
        c1 = min(full, c0 + step)
        block = x[c0 * CHUNK_ROWS : c1 * CHUNK_ROWS].reshape(c1 - c0, 32, 8, cols)
        s = np.zeros((c1 - c0, 8, cols), dtype=F32)
        e = np.zeros((c1 - c0, 8, cols), dtype=F32)
        for i in range(32):
            s, e = bg.two_sum_add(s, e, block[:, i])
        partial_s[c0:c1], partial_e[c0:c1] = thread_merges(s, e, 8)
    if full < chunks:
        # The last chunk has fewer than 256 rows: threads past its rows hold (0, 0).
        first = full * CHUNK_ROWS
        n = rows - first
        threads = min(8, n)
        s = np.zeros((1, threads, cols), dtype=F32)
        e = np.zeros((1, threads, cols), dtype=F32)
        for i in range(-(-n // 8)):
            piece = x[first + 8 * i : first + min(8 * i + 8, n)]
            k = len(piece)
            s[0, :k], e[0, :k] = bg.two_sum_add(s[0, :k], e[0, :k], piece)
        partial_s[full], partial_e[full] = (v[0] for v in thread_merges(s, e, threads))
    return partial_s, partial_e


def emulate_position_sums(v, old=None):
    """The column sums of v (rows x cols float32) as the bias gradient's two passes do them, on
    FP32 values: (total, partial sums, partial errors). Pass 2 is the bias gradient's own
    emulation; with old (cols), old + the total. Done in slices of columns to bound the
    temporaries."""
    rows, cols = v.shape
    chunks = -(-rows // CHUNK_ROWS)
    partial_s = np.empty((chunks, cols), dtype=F32)
    partial_e = np.empty((chunks, cols), dtype=F32)
    width = max(8, BATCH_ELEMENTS // min(rows, CHUNK_ROWS) // 8 * 8)
    with np.errstate(all="ignore"):
        for j0 in range(0, cols, width):
            j1 = min(cols, j0 + width)
            partial_s[:, j0:j1], partial_e[:, j0:j1] = position_pass1(v[:, j0:j1])
        total = bg.pass2(partial_s, partial_e, old)
    return total, partial_s, partial_e


def emulate_all(dx, tokens, sequences, length, vocab, old_wte=None, old_wpe=None):
    rows, width = dx.shape
    assert rows == sequences * length
    dwte, sorted_tokens, sorted_rows, offsets, reached = emulate_words(dx, tokens, vocab, old_wte)
    total, partial_s, partial_e = emulate_position_sums(
        dx.reshape(sequences, length * width), None if old_wpe is None else old_wpe.reshape(-1)
    )
    return Expected(dwte, total.reshape(length, width), sorted_tokens, sorted_rows, offsets, partial_s, partial_e, reached)


# The literal scalar transcription of the module docstring, one np.float32 number at a time.


def scalar_words(dx, tokens, vocab, old=None):
    f = F32
    rows, width = dx.shape
    out = np.zeros((vocab, width), dtype=f) if old is None else np.array(old, dtype=f, copy=True)
    with np.errstate(all="ignore"):
        for v in range(vocab):
            members = [r for r in range(rows) if int(tokens[r]) == v]
            if not members:
                continue
            for j in range(width):
                s, e = f(0.0), f(0.0)
                for r in members:
                    x = f(dx[r, j])
                    t = f(s + x)
                    vp = f(t - s)
                    d1 = f(t - vp)
                    d2 = f(s - d1)
                    d3 = f(x - vp)
                    e = f(e + f(d2 + d3))
                    s = t
                value = f(s + e) if np.isfinite(s) else s
                out[v, j] = value if old is None else f(f(old[v, j]) + value)
    return out


# -- Comparing --------------------------------------------------------------------------------------------


def assert_bits(actual, expected, what):
    """FP32 values (a torch tensor or numpy array each) equal bit for bit, NaNs aside (any NaN
    for any NaN). A raw comparison first, which is fast; the one that explains and forgives a
    NaN's payload only when it fails."""
    a = actual.detach().cpu().contiguous() if isinstance(actual, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(actual))
    e = expected.detach().cpu().contiguous() if isinstance(expected, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(expected))
    if a.shape == e.shape and torch.equal(a.view(torch.int32), e.view(torch.int32)):
        return
    assert_same_bits(a, e, what)


def assert_ints(actual, expected, what):
    a = actual.detach().cpu().contiguous() if isinstance(actual, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(actual))
    e = expected.detach().cpu().contiguous() if isinstance(expected, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(expected))
    assert a.shape == e.shape, f"{what}: shapes {tuple(a.shape)} and {tuple(e.shape)}"
    if torch.equal(a, e):
        return
    bad = (a != e).nonzero().flatten()
    first = int(bad[0])
    raise AssertionError(
        f"{what}: {bad.numel()} of {a.numel()} elements differ, the first at {first}: "
        f"{a.reshape(-1)[first].item()} against {e.reshape(-1)[first].item()}"
    )


def raw_bits(t):
    """The FP32 bits of a CPU tensor or numpy array, as int32 (NaN payloads included)."""
    t = torch.from_numpy(np.ascontiguousarray(t)) if isinstance(t, np.ndarray) else t.detach().cpu().contiguous()
    return t.view(torch.int32)


def count_differences(a, b):
    """How many elements differ (as bits, NaNs as one), for two float32 numpy arrays."""
    return int((bg.canonical(torch.from_numpy(np.ascontiguousarray(a))) != bg.canonical(torch.from_numpy(np.ascontiguousarray(b)))).sum())


# -- Running Flint ------------------------------------------------------------------------------------------


def read_workspace(words, parts, sequences, length, vocab, width):
    """The parts of the workspace the tests read, from its used words (int32 tensor on the GPU), as
    CPU tensors: sorted tokens (int16 bit patterns), sorted rows and offsets (int32) and the
    position partials, (chunks, 2, cols) float32: each chunk's row of sums and row of errors."""
    rows = sequences * length
    cols = length * width
    chunks = -(-sequences // CHUNK_ROWS)
    raw = words.view(torch.uint8)

    def part(name, dtype, count):
        size = torch.empty((), dtype=dtype).element_size()
        return raw[parts[name] : parts[name] + count * size].view(dtype).cpu().clone()

    return {
        "sorted_tokens": part("sorted_tokens", torch.int16, rows),
        "sorted_rows": part("sorted_rows", torch.int32, rows),
        "offsets": part("offsets", torch.int32, vocab + 1),
        "partials": part("position_partials", torch.float32, chunks * 2 * cols).reshape(chunks, 2, cols),
    }


def run_flint(flint, dx, tokens, sequences, length, vocab, width, *, old_wte=None, old_wpe=None,
              accumulate_wte=False, accumulate_wpe=False, extra_rows=2, offset=4, wte_fill=NAN32,
              wpe_fill=NAN32, workspace_fill=NAN32, workspace_extra=0, shared_workspace=None,
              stream=None, with_workspace=False, check_input=None):
    """Runs the function on dx (rows x width float32 on the GPU) and tokens (int16 bit patterns on
    the GPU), into a dwte and a dwpe inside buffers pre-filled with a sentinel, `offset` floats in
    (a multiple of 4 keeps them 16-byte aligned), and a workspace likewise (its needed bytes filled
    with workspace_fill, workspace_extra more bytes of room after it). dwpe's buffer holds
    extra_rows rows past length, which must keep the sentinel. With an accumulate flag the output
    starts as old_wte or old_wpe (numpy or torch), otherwise as its fill (an int32 bit pattern or
    "garbage"). shared_workspace (a tensor of int32 words) is used as it is, with no fill and no
    guard bands. Checks that everything around the outputs and the workspace is untouched and that
    dx and tokens are unchanged. Returns Result(dwte (vocab x width), dwpe (length x width), the
    workspace parts as read_workspace gives them, or None), the outputs as copies."""
    rows = sequences * length
    needed = flint.embedding_backward_workspace(sequences, length, vocab, width)
    parts = layout(sequences, length, vocab, width)
    assert needed == parts["bytes"]
    wte_words = vocab * width
    first = length * width
    wpe_words = first + extra_rows * width
    if check_input is None:
        check_input = dx.numel() <= CHECK_INPUT_ELEMENTS

    wte_buf = torch.full((offset + wte_words + GUARD_AFTER,), SENTINEL, dtype=torch.int32, device="cuda")
    dwte = wte_buf[offset : offset + wte_words]
    if accumulate_wte:
        dwte.view(torch.float32).copy_(torch.as_tensor(old_wte).reshape(-1))
    else:
        set_words(dwte, wte_fill)
    wpe_buf = torch.full((offset + wpe_words + GUARD_AFTER,), SENTINEL, dtype=torch.int32, device="cuda")
    dwpe = wpe_buf[offset : offset + wpe_words]
    if accumulate_wpe:
        dwpe[:first].view(torch.float32).copy_(torch.as_tensor(old_wpe).reshape(-1))
    else:
        set_words(dwpe[:first], wpe_fill)
    words = needed // 4
    if shared_workspace is None:
        ws_buf = torch.full((offset + words + workspace_extra // 4 + GUARD_AFTER,), SENTINEL,
                            dtype=torch.int32, device="cuda")
        workspace = ws_buf[offset : offset + words + workspace_extra // 4]
        set_words(workspace[:words], workspace_fill)
    else:
        workspace = shared_workspace
    before = (dx.clone(), tokens.clone()) if check_input else None
    torch.cuda.synchronize()

    flint.embedding_backward(
        dwte, dwpe, dx, tokens, sequences, length, vocab, width, accumulate_wte=accumulate_wte,
        accumulate_wpe=accumulate_wpe, workspace=workspace, workspace_bytes=needed + workspace_extra,
        stream=stream,
    )
    torch.cuda.synchronize()

    sentinel = torch.full((1,), SENTINEL, dtype=torch.int32, device="cuda")
    assert all_equal(wte_buf[:offset], sentinel), f"{NAME} wrote before dwte"
    assert all_equal(wte_buf[offset + wte_words :], sentinel), f"{NAME} wrote past dwte"
    assert all_equal(wpe_buf[:offset], sentinel), f"{NAME} wrote before dwpe"
    assert all_equal(wpe_buf[offset + first :], sentinel), f"{NAME} wrote past dwpe's first length rows"
    if shared_workspace is None:
        assert all_equal(ws_buf[:offset], sentinel), f"{NAME} wrote before the workspace"
        assert all_equal(ws_buf[offset + words :], sentinel), f"{NAME} wrote past its workspace"
    if check_input:
        assert torch.equal(dx.view(torch.int32), before[0].view(torch.int32)), "dx changed"
        assert torch.equal(tokens, before[1]), "tokens changed"
    read = read_workspace(workspace[:words], parts, sequences, length, vocab, width) if with_workspace else None
    return Result(
        dwte.view(torch.float32).reshape(vocab, width).cpu(),
        dwpe[:first].view(torch.float32).reshape(length, width).cpu(),
        read,
    )


def check_exact(got, exp, what, *, old_wte=None, accumulate_wte=False):
    """Result against Expected, bit for bit; the workspace's parts too when it was read. With
    accumulate_wte, the rows no token reaches also equal old_wte's raw bits (NaN payloads
    included)."""
    assert_bits(got.dwte, exp.dwte, f"dwte {what}")
    if accumulate_wte:
        untouched = torch.from_numpy(~exp.reached)
        assert torch.equal(raw_bits(got.dwte)[untouched], raw_bits(old_wte)[untouched]), (
            f"{what}: a row no token reaches changed"
        )
    assert_bits(got.dwpe, exp.dwpe, f"dwpe {what}")
    ws = got.workspace
    if ws is not None:
        assert_ints(ws["sorted_tokens"], exp.sorted_tokens.view(np.int16), f"sorted tokens {what}")
        assert_ints(ws["sorted_rows"], exp.sorted_rows.astype(np.int32), f"sorted rows {what}")
        assert_ints(ws["offsets"], exp.offsets.astype(np.int32), f"offsets {what}")
        assert_bits(ws["partials"][:, 0].reshape(-1), exp.partial_s.reshape(-1), f"partial sums {what}")
        assert_bits(ws["partials"][:, 1].reshape(-1), exp.partial_e.reshape(-1), f"partial errors {what}")


def expect_exact(flint, dx, tokens, sequences, length, vocab, width, what="", *, old_wte=None, old_wpe=None,
                 accumulate_wte=False, accumulate_wpe=False, **run_kwargs):
    """Runs Flint on dx and tokens (numpy) and checks everything against the emulation. Returns
    (Result, Expected)."""
    got = run_flint(
        flint, torch.from_numpy(dx).cuda(), tokens_tensor(tokens), sequences, length, vocab, width,
        old_wte=old_wte, old_wpe=old_wpe, accumulate_wte=accumulate_wte, accumulate_wpe=accumulate_wpe,
        with_workspace=True, **run_kwargs,
    )
    exp = emulate_all(dx, tokens, sequences, length, vocab, old_wte if accumulate_wte else None,
                      old_wpe if accumulate_wpe else None)
    check_exact(got, exp, what or case_id((sequences, length, vocab, width)), old_wte=old_wte,
                accumulate_wte=accumulate_wte)
    return got, exp


# -- The emulation itself ----------------------------------------------------------------------------------

SCALAR_CASES = [
    # (rows, width, vocab, tokens below)
    (1, 8, 4, 4), (13, 8, 5, 5), (40, 16, 7, 7), (60, 8, 3, 6), (37, 24, 40, 40),
]


@pytest.mark.parametrize("rows, width, vocab, token_range", SCALAR_CASES, ids=str)
def test_the_words_emulation_matches_a_literal_scalar_transcription(rows, width, vocab, token_range):
    # token_range above vocab: those tokens reach no row.
    dx = make_dx("wide", rows, width, 5 + rows)
    tokens = np.random.default_rng(rows).integers(0, token_range, rows).astype(np.uint16)
    old = bg.cpu_dy(vocab, width, 3).float().numpy() * F32(7)

    out = emulate_words(dx, tokens, vocab)[0]
    out_old = emulate_words(dx, tokens, vocab, old)[0]

    assert_bits(out, scalar_words(dx, tokens, vocab), "overwrite")
    assert_bits(out_old, scalar_words(dx, tokens, vocab, old), "accumulate")


def test_the_words_emulation_matches_the_scalar_transcription_with_non_finite_values_and_negative_zero():
    rows, width, vocab = 90, 16, 6
    dx = make_dx("normal", rows, width, 11)
    tokens = (np.arange(rows) % 7).astype(np.uint16)  # token 6 is the last, 0..5 are below vocab 6
    tokens[40] = 65535
    dx[0, 0] = np.inf  # token 0: +inf
    dx[1, 1], dx[8, 1] = np.inf, -np.inf  # token 1: NaN
    dx[2, 2], dx[9, 2], dx[16, 2] = np.inf, np.inf, np.inf  # token 2: +inf
    dx[3, 3] = np.nan  # token 3: NaN
    dx[4, 4], dx[11, 4] = -np.inf, -np.inf  # token 4: -inf
    dx[5, 5] = np.inf  # token 5
    dx[12, 5] = -np.inf  # token 5 again: NaN
    dx[6, 6] = np.inf  # token 6, past vocab: reaches no row
    dx[:, 7] = F32(-0.0)  # every token's column sums -0 values: +0
    dx[1, 8] = F32(-0.0)
    dx[8, 9] = F32(1e-45)  # a subnormal: kept
    old = bg.cpu_dy(vocab, width, 4).float().numpy() * F32(7)

    out = emulate_words(dx, tokens, vocab)[0]

    assert_bits(out, scalar_words(dx, tokens, vocab), "overwrite")
    assert_bits(emulate_words(dx, tokens, vocab, old)[0], scalar_words(dx, tokens, vocab, old), "accumulate")
    assert out[0, 0] == math.inf and math.isnan(out[1, 1]) and out[2, 2] == math.inf
    assert math.isnan(out[3, 3]) and out[4, 4] == -math.inf and math.isnan(out[5, 5])
    assert not bool(np.signbit(out[:, 7]).any()) and bool((out[:, 7] == 0).all())


@pytest.mark.parametrize("rows, cols", [(7, 8), (300, 16), (513, 8), (1200, 8)], ids=str)
def test_the_position_emulation_matches_the_bias_gradients_and_its_scalar_transcription(rows, cols):
    # dy is FP16 values, which FP32 holds exactly: the bias gradient's emulation on the FP16 and ours
    # on the FP32 are the same sums, the partials included.
    dy = bg.cpu_dy(rows, cols, 21 + rows)
    old = (torch.randn(cols, generator=torch.Generator().manual_seed(rows)) * 100).numpy()

    total, partial_s, partial_e = emulate_position_sums(dy.float().numpy(), old)
    expected, expected_s, expected_e = bg.emulate(dy, old, with_partials=True)

    assert_bits(total, expected, "total")
    assert_bits(partial_s, expected_s, "partial sums")
    assert_bits(partial_e, expected_e, "partial errors")
    if rows <= 600:
        assert_bits(total, bg.scalar_reference(dy.float().numpy(), old), "the scalar transcription")


def test_the_emulations_stay_in_float32():
    dx = make_dx("normal", 300, 16, 2)
    tokens = make_tokens("uniform", 300, 20, 2)

    exp = emulate_all(dx, tokens, 3, 100, 20)

    assert exp.dwte.dtype == np.float32 and exp.dwpe.dtype == np.float32
    assert exp.partial_s.dtype == np.float32 and exp.partial_e.dtype == np.float32
    assert emulate_words(dx, tokens, 20, np.zeros((20, 16), dtype=F32))[0].dtype == np.float32


# -- The bit-exact emulation ------------------------------------------------------------------------------------

# (sequences, length, vocab, width, tokens, dx). One row; GPT-2's shapes (16 x 1024 x 768 and the
# smaller batches) over its padded vocabulary; the sort's tile of 2,048 rows (rows 2,047, 2,048,
# 2,049, 2,100, 4,100, 16,384); many tiles (35 and, off the sanitizer, 2^20 rows: the scan's carry
# over 512 x 256 entries); position chunks (8,448 rows: 33 full chunks; 8,453 with a partial last
# one); widths that aren't a multiple of 32 or of 256, and widths past 256 whose last block of 256
# columns is partial (264, 520); and tokens sharing a low byte.
EXACT_CASES = shrunk_cases([
    (1, 1, 64, 8, "uniform", "normal"),
    (3, 5, 64, 8, "uniform", "wide"),
    (7, 300, 300, 24, "repeated", "outliers_mixed"),
    (2, 1024, 1000, 64, "zipf", "normal"),
    (3, 683, 1000, 64, "uniform", "outliers_same"),
    (1, 2047, GPT2_VOCAB, 8, "zipf", "gradient"),
    (4, 1025, 2000, 16, "few", "wide"),
    (5, 1024, GPT2_VOCAB, 64, "one", "normal"),
    (8453, 1, 64, 8, "uniform", "outliers_same"),
    (33 * 256, 1, 64, 8, "repeated", "outliers_mixed"),
    (3, 50, 1000, 264, "zipf", "outliers_same"),
    (2, 40, 300, 520, "repeated", "wide"),
    (4096, 4, 64, 64, "zipf", "wide"),
    (1, 17 * 2048 + 3, 64, 8, "uniform", "normal"),
    (2, 1000, MAX_VOCAB, 8, "collide", "normal"),
    (16, 1024, GPT2_VOCAB, 768, "zipf", "normal"),
    (16, 1024, GPT2_VOCAB, 768, "repeated", "outliers_same"),
    (16, 1024, GPT2_VOCAB, 768, "uniform", "wide"),
    (8, 1024, GPT2_VOCAB, 768, "few", "outliers_mixed"),
    (1, 1024, GPT2_VOCAB, 768, "zipf", "gradient"),
    (4, 2**18, 64, 8, "uniform", "normal"),
    (4, 2**18, 64, 8, "zipf", "wide"),
])


@pytest.mark.parametrize("sequences, length, vocab, width, token_kind, dx_kind", case_params(EXACT_CASES))
def test_flint_equals_the_emulation_bit_for_bit(flint, sequences, length, vocab, width, token_kind, dx_kind):
    rows = sequences * length
    dx = make_dx(dx_kind, rows, width, 100 + rows)
    tokens = make_tokens(token_kind, rows, vocab, 100 + rows)

    expect_exact(flint, dx, tokens, sequences, length, vocab, width)


@pytest.mark.parametrize("tokens", [[0x0001, 0x0101, 0xFF01], [0xFF01, 0x0101, 0x0001, 0x0101, 0xFF01, 0x0001]], ids=["low-byte-1", "reversed"])
def test_tokens_sharing_a_low_byte_are_ordered_by_their_high_byte(flint, tokens):
    # The sort's two passes: the low byte first, then the high one, stably. Several rows of each token
    # in an unsorted order, around tokens of other low bytes.
    rng = np.random.default_rng(3)
    base = np.array(tokens * 40 + [0x0000, 0xFF00, 0x7F7F, 0xFFFF, 0x0100] * 8, dtype=np.uint16)
    tokens = base[rng.permutation(len(base))]
    rows = len(tokens)
    dx = make_dx("normal", rows, 8, 7)

    got, _ = expect_exact(flint, dx, tokens, 1, rows, MAX_VOCAB, 8)

    sorted_tokens = got.workspace["sorted_tokens"].numpy().view(np.uint16)
    assert bool((np.diff(sorted_tokens.astype(np.int64)) >= 0).all())
    sorted_rows = got.workspace["sorted_rows"].numpy().astype(np.int64)
    same = sorted_tokens[1:] == sorted_tokens[:-1]
    assert bool((np.diff(sorted_rows)[same] > 0).all()), "the sort is not stable"


def test_a_token_shared_by_every_tile_keeps_its_rows_in_ascending_order(flint):
    # 35 tiles, every one holding the same few tokens: each (digit, tile) start comes from the scan.
    rows = 17 * 2048 + 3
    tokens = (np.arange(rows) % 3 * 0x101).astype(np.uint16)
    dx = make_dx("normal", rows, 8, 8)

    got, _ = expect_exact(flint, dx, tokens, 1, rows, 1000, 8)

    sorted_rows = got.workspace["sorted_rows"].numpy()
    offsets = got.workspace["offsets"].numpy()
    first = sorted_rows[offsets[0] : offsets[1]]
    assert bool((first == np.arange(0, rows, 3)).all())


# -- Crafted runs: the order of the sums ---------------------------------------------------------------------------


def ordered_sum(rows):
    """The kernel's order for a run of rows (length x width float32): TwoSum row by row, then s + e.
    Returns the value (width) and the pair."""
    with np.errstate(all="ignore"):
        s = np.zeros(rows.shape[1], F32)
        e = np.zeros(rows.shape[1], F32)
        for x in rows:
            s, e = bg.two_sum_add(s, e, x)
        return np.where(np.isfinite(s), s + e, s).astype(F32), (s, e)


def pair_value(pair):
    s, e = pair
    with np.errstate(all="ignore"):
        return np.where(np.isfinite(s), s + e, s).astype(F32)


def chunked_sum(rows, n):
    """Chunks of n rows summed in order, the chunks' pairs merged in order."""
    acc = None
    for c in range(0, len(rows), n):
        _, pair = ordered_sum(rows[c : c + n])
        acc = pair if acc is None else bg.two_sum_merge(acc, pair)
    return pair_value(acc)


def plain_sum(rows):
    s = np.zeros(rows.shape[1], F32)
    with np.errstate(all="ignore"):
        for x in rows:
            s = s + x
    return s


def interleaved_sum(rows):
    _, a = ordered_sum(rows[0::2])
    _, b = ordered_sum(rows[1::2])
    return pair_value(bg.two_sum_merge(a, b))


def other_orders(rows):
    """The sums of a run in orders that are not the kernel's, by name."""
    return {
        "rows descending": ordered_sum(rows[::-1])[0],
        "plain": plain_sum(rows),
        "chunks of 32 merged": chunked_sum(rows, 32),
        "chunks of 10 merged": chunked_sum(rows, 10),
        "even and odd rows merged": interleaved_sum(rows),
        "first row last": ordered_sum(np.concatenate([rows[1:], rows[:1]]))[0],
    }


def crafted_run(length, width, seed):
    """A run of length rows (width columns) of cancelling pairs: three big values from 2^23 to 2^24
    and the medium ones up to 2^10 with full mantissas, each with its negative elsewhere in the run,
    plus per column a small odd multiple of 2^-24, a subnormal and a value in [-1, 1]. The partial
    sums round at every order, so the order shows in the bits of the sum."""
    rng = np.random.default_rng(seed)
    x = np.zeros((length, width), F32)
    for j in range(width):
        perm = rng.permutation(length)
        k = 0
        for p in range(length // 2 - 2):
            a, b = perm[k], perm[k + 1]
            k += 2
            v = F32(rng.integers(2**23, 2**24)) if p < 3 else F32(rng.uniform(-1024, 1024))
            x[a, j] = v
            x[b, j] = -v
        rest = perm[k:]
        x[rest[0], j] = F32(rng.integers(1, 2**10) * 2 + 1) * F32(2.0**-24)
        x[rest[1], j] = F32(rng.integers(1, 2**10)) * F32(2.0**-149)
        x[rest[2], j] = F32(rng.uniform(-1, 1))
    return x


CRAFTED_RUN = (96, 64)
CRAFTED_RUN_SEED = 8


def test_the_crafted_run_shows_the_order_of_the_sums():
    # Each other order differs from the kernel's in a good share of the 64 columns (39% or more for
    # this seed), so the bit-exact test below pins the order: ascending rows, one sequential sum.
    run = crafted_run(*CRAFTED_RUN, CRAFTED_RUN_SEED)
    kernel = ordered_sum(run)[0]

    for name, other in other_orders(run).items():
        share = count_differences(kernel, other) / kernel.size
        print(f"ORDER crafted run {name}: {share:.0%} of the columns differ")
        assert share >= 0.25, name


@pytest.mark.parametrize("seed", [8, 9, 10])
def test_a_crafted_run_at_random_rows_equals_the_emulation(flint, seed):
    # One token's 96 rows, at random rows of a batch of 384 whose other rows hold other tokens and
    # ordinary values: the token's row of dwte is the ascending-order sum, whatever the rows between.
    sequences, length, vocab, width = 3, 128, 50, 64
    rows = sequences * length
    hot = 17
    run = crafted_run(*CRAFTED_RUN, seed)
    rng = np.random.default_rng(seed + 500)
    tokens = rng.integers(0, vocab - 1, rows)
    tokens[tokens >= hot] += 1
    positions = np.sort(rng.choice(rows, CRAFTED_RUN[0], replace=False))
    tokens[positions] = hot
    dx = rng.standard_normal((rows, width), dtype=F32)
    dx[positions] = run
    tokens = tokens.astype(np.uint16)

    got, exp = expect_exact(flint, dx, tokens, sequences, length, vocab, width)

    kernel = ordered_sum(run)[0]
    assert_bits(exp.dwte[hot], kernel, "the emulation's row of the crafted token")
    assert_bits(got.dwte[hot], kernel, "Flint's row of the crafted token")


def test_dwpe_equals_the_emulation_on_the_bias_gradients_crafted_inputs(flint):
    # The bias gradient's crafted dy (cancelling chunks whose errors' sums round, which pin pass 2's
    # order) as dx viewed as sequences x (length * width): one row of dx per position. Under the
    # sanitizer, the first 8 columns of the same input.
    sequences = bg.CRAFTED_SHAPE[0]
    cols = 8 if SANITIZER else bg.CRAFTED_SHAPE[1]
    length, width = (1, 8) if SANITIZER else (cols // 8, 8)
    dy = bg.crafted_cancellation(sequences, cols, bg.CRAFTED_SEED)
    dx = np.ascontiguousarray(dy.float().numpy().reshape(sequences * length, width))
    tokens = make_tokens("uniform", sequences * length, 64, 3)

    got, exp = expect_exact(flint, dx, tokens, sequences, length, 64, width, "crafted dwpe")

    assert_bits(got.dwpe.reshape(-1), bg.emulate(dy), "the bias gradient's emulation of the same input")


# -- The correctness gate -----------------------------------------------------------------------------------------------

GATE_TOKENS = ["uniform", "zipf", "repeated"]
GATE_DX = ["normal", "gradient", "outliers_same", "wide"]
GATE_SMALL = [(4, 256, 1000, 64), (3, 5, 64, 8), (8453, 1, 64, 8), (4096, 4, 64, 64), (1024, 4, 64, 64)]
GATE_CASES = shrunk_cases(
    [
        (16, 1024, GPT2_VOCAB, 768, "uniform", "normal"),
        (16, 1024, GPT2_VOCAB, 768, "zipf", "gradient"),
        (16, 1024, GPT2_VOCAB, 768, "zipf", "outliers_same"),
        (16, 1024, GPT2_VOCAB, 768, "repeated", "wide"),
        (8, 1024, GPT2_VOCAB, 768, "zipf", "normal"),
        (8, 1024, GPT2_VOCAB, 768, "uniform", "outliers_mixed"),
        (1, 1024, GPT2_VOCAB, 768, "zipf", "wide"),
    ]
    + [shape + (t, d) for shape in GATE_SMALL for t in GATE_TOKENS for d in GATE_DX]
)


def peer_dwte(dx, indices, vocab):
    """FP32 PyTorch's word table gradient (what autocast runs: the embedding stays FP32)."""
    return torch.ops.aten.embedding_dense_backward(dx, indices, vocab, -1, False)


def peer_dwpe(dx, sequences, length, width):
    return dx.view(sequences, length, width).sum(0)


def gate_line(label, flint_errors, peer_errors, bounds):
    print(
        f"GATE {label} | flint rel_l2 {max(e.rel_l2 for e in flint_errors):.3g} max_abs "
        f"{max(e.max_abs for e in flint_errors):.3g} | peer rel_l2 {max(e.rel_l2 for e in peer_errors):.3g} "
        f"max_abs {max(e.max_abs for e in peer_errors):.3g} | bound rel_l2 {bounds.rel_l2:.3g} "
        f"max_abs {bounds.max_abs:.3g}"
    )


@pytest.mark.parametrize("sequences, length, vocab, width, token_kind, dx_kind", case_params(GATE_CASES))
def test_passes_the_gate_and_the_tighter_bound_against_fp32_pytorch(
    flint, sequences, length, vocab, width, token_kind, dx_kind
):
    # PLAN.md's gate: at most 3x the peer's error against FP64 (the peer: FP32 PyTorch, the path
    # autocast takes for the embedding), by relative L2 and max-abs, with the FP32 floors; and
    # D26's rule: at most 2x.
    rows = sequences * length
    errors = {name: ([], []) for name in ("dwte", "dwpe")}
    for seed in gate.SEEDS:
        dx_np = make_dx(dx_kind, rows, width, 31 * seed + 1)
        tokens = make_tokens(token_kind, rows, vocab, 31 * seed + 2)
        dx = torch.from_numpy(dx_np).cuda()
        indices = torch.from_numpy(tokens.astype(np.int64)).cuda()
        got = run_flint(flint, dx, tokens_tensor(tokens), sequences, length, vocab, width, offset=4 * (1 + seed))
        ref_wte = torch.zeros((vocab, width), dtype=torch.float64, device="cuda").index_add_(0, indices, dx.double())
        ref_wpe = dx.view(sequences, length, width).sum(0, dtype=torch.float64)
        for name, mine, ref, peer in (
            ("dwte", got.dwte, ref_wte, peer_dwte(dx, indices, vocab)),
            ("dwpe", got.dwpe, ref_wpe, peer_dwpe(dx, sequences, length, width)),
        ):
            errors[name][0].append(gate.errors(mine.cuda(), ref))
            errors[name][1].append(gate.errors(peer, ref))
        del dx, indices, got, ref_wte, ref_wpe

    label = case_id((sequences, length, vocab, width, token_kind, dx_kind))
    for factor in (gate.GRADIENT_FACTOR, 2.0):
        for name, (flint_errors, peer_errors) in errors.items():
            bounds = gate.check_gate(factor, flint_errors, peer_errors, output="fp32")
            gate_line(f"{name} x{factor:g} {label}", flint_errors, peer_errors, bounds)


# -- Exact cases ---------------------------------------------------------------------------------------------------------


def test_a_token_seen_once_gives_its_row_exactly_but_for_negative_zero(flint):
    sequences, length, vocab, width = 3, 200, 1000, 16
    rows = sequences * length
    dx = make_dx("wide", rows, width, 41)
    dx[::7, ::3] = F32(-0.0)
    dx[5, 5], dx[6, 6], dx[7, 7] = np.nan, np.inf, -np.inf
    tokens = np.random.default_rng(41).permutation(vocab)[:rows].astype(np.uint16)

    got = run_flint(flint, torch.from_numpy(dx).cuda(), tokens_tensor(tokens), sequences, length, vocab, width)

    # +0 added to every value: a -0 becomes +0, nothing else changes (subnormals, inf and NaN included).
    expected = dx + F32(0.0)
    assert_bits(got.dwte[torch.from_numpy(tokens.astype(np.int64))], expected, "the rows of the tokens")
    unreached = np.ones(vocab, dtype=bool)
    unreached[tokens] = False
    assert all_equal(raw_bits(got.dwte)[torch.from_numpy(unreached)], 0)
    assert bool((dx == 0).any() and np.signbit(dx[dx == 0]).any()), "the input has a -0"


@pytest.mark.parametrize("dx_kind", ["wide", "normal"])
def test_one_sequence_gives_dx_in_dwpe_but_for_negative_zero(flint, dx_kind):
    sequences, length, vocab, width = 1, 777, 64, 24
    dx = make_dx(dx_kind, length, width, 42)
    dx[::5, ::4] = F32(-0.0)
    dx[3, 3] = np.inf
    tokens = make_tokens("uniform", length, vocab, 42)

    got = run_flint(flint, torch.from_numpy(dx).cuda(), tokens_tensor(tokens), sequences, length, vocab, width)

    assert_bits(got.dwpe, dx + F32(0.0), "dwpe")


def test_negative_zero_inputs_give_positive_zero_like_pytorch(flint):
    sequences, length, vocab, width = 4, 50, 30, 16
    rows = sequences * length
    dx = np.full((rows, width), -0.0, dtype=F32)
    tokens = make_tokens("repeated", rows, vocab, 43)
    dx_gpu = torch.from_numpy(dx).cuda()

    got = run_flint(flint, dx_gpu, tokens_tensor(tokens), sequences, length, vocab, width)

    assert all_equal(raw_bits(got.dwte), 0), "dwte holds a -0"
    assert all_equal(raw_bits(got.dwpe), 0), "dwpe holds a -0"
    indices = torch.from_numpy(tokens.astype(np.int64)).cuda()
    for name, mine, peer in (
        ("dwte", got.dwte, peer_dwte(dx_gpu, indices, vocab)),
        ("dwpe", got.dwpe, peer_dwpe(dx_gpu, sequences, length, width)),
    ):
        assert all_equal(raw_bits(peer.cpu()), 0), f"PyTorch's {name} is not +0"


# -- Accumulate -----------------------------------------------------------------------------------------------------------------

SPECIAL_OLDS = np.array(
    [0.0, -0.0, 1e-40, -1e-40, np.inf, -np.inf, np.nan, 3.0e38, -3.0e38, 1e-30, 16777216.0, -16777216.0, 1.0],
    dtype=F32,
)
NAN_PAYLOADS = np.array([0x7FC12345, 0xFFC00001, 0x7FA00001, 0xFFFFFFFF], dtype=np.uint32).view(F32)


def make_old(shape, seed):
    """Old values: N(0, 1) times 100 (adds that round), 15% random bit patterns (every exponent,
    NaNs with payloads among them), and 15% specials: +-0, subnormals, +-inf, NaN, near FP32's limit,
    2^24 (where adding 1.x rounds) and NaNs with distinctive payloads."""
    rng = np.random.default_rng(seed)
    old = rng.standard_normal(shape, dtype=F32) * F32(100)
    pick = rng.random(shape)
    bits = rng.integers(-(2**31), 2**31, shape, dtype=np.int64).astype(np.int32).view(F32)
    specials = np.concatenate([SPECIAL_OLDS, NAN_PAYLOADS])
    old = np.where(pick < 0.15, bits, old)
    old = np.where(pick > 0.85, specials[rng.integers(0, len(specials), shape)], old)
    return np.ascontiguousarray(old, dtype=F32)


ACCUMULATE_CASES = shrunk_cases([
    (3, 5, 64, 8, "uniform", "wide"),
    (7, 300, 300, 24, "repeated", "outliers_mixed"),
    (4, 1025, 2000, 16, "zipf", "normal"),
    (2, 1000, GPT2_VOCAB, 8, "zipf", "outliers_same"),
    (8453, 1, 64, 8, "uniform", "wide"),
    (1, 1, 5, 8, "one", "normal"),
])
FLAGS = [(True, True), (True, False), (False, True)]


@pytest.mark.parametrize("flags", FLAGS, ids=lambda f: f"wte-{f[0]}-wpe-{f[1]}")
@pytest.mark.parametrize("sequences, length, vocab, width, token_kind, dx_kind", case_params(ACCUMULATE_CASES))
def test_accumulate_adds_the_sums_to_old_with_one_add_and_leaves_other_rows(
    flint, sequences, length, vocab, width, token_kind, dx_kind, flags
):
    # Reached rows are old + the sum, bit for bit (olds of every kind), the rows no token reaches
    # keep old's bits exactly (NaN payloads, -0 and all), and each flag works alone: the other
    # output is overwritten whatever it held. Rows of dwpe past length keep the sentinel
    # (run_flint checks).
    accumulate_wte, accumulate_wpe = flags
    rows = sequences * length
    dx = make_dx(dx_kind, rows, width, 200 + rows)
    tokens = make_tokens(token_kind, rows, vocab, 200 + rows)
    old_wte = make_old((vocab, width), 201)
    old_wpe = make_old((length, width), 202)

    expect_exact(flint, dx, tokens, sequences, length, vocab, width, old_wte=old_wte, old_wpe=old_wpe,
                 accumulate_wte=accumulate_wte, accumulate_wpe=accumulate_wpe)


def test_accumulate_onto_special_olds_gives_the_fp32_sum_of_each(flint):
    # A row of olds of one kind per column: old + the sum for the reached rows, the old bits
    # for the others.
    sequences, length, vocab, width = 2, 30, 40, 16
    rows = sequences * length
    dx = make_dx("normal", rows, width, 51)
    dx[:, :4] = F32(0.0)  # columns that sum to +0, over the olds +0, -0, 1e-40 and -1e-40
    dx[:, 12:] = F32(0.0)  # and over 1.0 and the NaNs with payloads
    tokens = (np.arange(rows) % 10).astype(np.uint16)  # tokens 0..9 reached, 10..39 not
    olds = np.concatenate([SPECIAL_OLDS, NAN_PAYLOADS])[:width]
    old_wte = np.tile(olds, (vocab, 1)).astype(F32)
    old_wpe = np.tile(olds, (length, 1)).astype(F32)

    got, _ = expect_exact(flint, dx, tokens, sequences, length, vocab, width, old_wte=old_wte, old_wpe=old_wpe,
                          accumulate_wte=True, accumulate_wpe=True)

    # A reached row, and a row of dwpe: -0 + +0 is +0 (and +0 + +0), a subnormal old with a zero sum
    # comes through exactly, 1.0 stays 1.0, inf and NaN stay.
    for table in (got.dwte[0], got.dwpe[0]):
        bits = raw_bits(table)
        assert bits[0].item() == 0 and bits[1].item() == 0
        assert bits[2].item() == raw_bits(torch.tensor([1e-40]))[0].item()
        assert bits[3].item() == raw_bits(torch.tensor([-1e-40]))[0].item()
        assert table[12].item() == 1.0 and bool(torch.isnan(table[13:]).all())
    # A row no token reaches keeps every old bit.
    assert torch.equal(raw_bits(got.dwte[20]), raw_bits(old_wte[20]))


@pytest.mark.parametrize("rows_shape", [(3, 100, 300, 16), (1, 300, 64, 8), (1500, 1, 64, 8)], ids=case_id)
def test_micro_batches_equal_the_sequential_fp32_adds(flint, rows_shape):
    # Five calls in a row on one stream sharing one workspace, with no sync between them: the first
    # overwrites, the rest accumulate.
    sequences, length, vocab, width = rows_shape
    rows = sequences * length
    batches = [(make_dx("wide", rows, width, 60 + i), make_tokens("zipf", rows, vocab, 60 + i)) for i in range(5)]
    needed = flint.embedding_backward_workspace(sequences, length, vocab, width)
    workspace = torch.full((needed // 4,), NAN32, dtype=torch.int32, device="cuda")
    dwte = torch.full((vocab, width), math.nan, device="cuda")
    dwpe = torch.full((length, width), math.nan, device="cuda")
    gpu = [(torch.from_numpy(dx).cuda(), tokens_tensor(tokens)) for dx, tokens in batches]

    for i, (dx, tokens) in enumerate(gpu):
        flint.embedding_backward(dwte, dwpe, dx, tokens, sequences, length, vocab, width,
                                 accumulate_wte=i > 0, accumulate_wpe=i > 0, workspace=workspace)
    torch.cuda.synchronize()

    expected_wte = expected_wpe = None
    for dx, tokens in batches:
        exp = emulate_all(dx, tokens, sequences, length, vocab)
        with np.errstate(all="ignore"):
            expected_wte = exp.dwte if expected_wte is None else expected_wte + exp.dwte
            expected_wpe = exp.dwpe if expected_wpe is None else expected_wpe + exp.dwpe
    assert_bits(dwte, expected_wte, "dwte after the micro-batches")
    assert_bits(dwpe, expected_wpe, "dwpe after the micro-batches")


# -- Write mode ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fill", list(FILLS))
def test_write_mode_ignores_the_old_content_and_zeroes_the_rows_no_token_reaches(flint, fill):
    sequences, length, vocab, width = 4, 50, 300, 16
    rows = sequences * length
    dx = make_dx("normal", rows, width, 70)
    tokens = make_tokens("uniform", rows, vocab, 70)  # 200 rows over 300 tokens: many rows unreached

    got, exp = expect_exact(flint, dx, tokens, sequences, length, vocab, width, wte_fill=FILLS[fill], wpe_fill=FILLS[fill])

    assert not bool(exp.reached.all()) and bool(exp.reached.any())
    assert all_equal(raw_bits(got.dwte)[torch.from_numpy(~exp.reached)], 0), "an unreached row is not +0"
    assert not bool(torch.isnan(got.dwte).any()) and not bool(torch.isnan(got.dwpe).any())


@pytest.mark.parametrize("offset", [4, 8, 12, 60])
def test_nothing_is_written_around_the_outputs_or_the_workspace(flint, offset):
    # run_flint checks the guard bands, the workspace's room, dx and tokens; the result is right too.
    sequences, length, vocab, width = 3, 100, 200, 24
    dx = make_dx("normal", 300, width, 71)
    tokens = make_tokens("zipf", 300, vocab, 71)

    expect_exact(flint, dx, tokens, sequences, length, vocab, width, offset=offset, workspace_extra=4096, extra_rows=5)


@pytest.mark.parametrize("fill", list(FILLS))
def test_the_workspaces_old_content_changes_nothing(flint, fill):
    sequences, length, vocab, width = 3, 700, 1000, 16
    dx = make_dx("wide", 2100, width, 72)
    tokens = make_tokens("uniform", 2100, vocab, 72)

    expect_exact(flint, dx, tokens, sequences, length, vocab, width, workspace_fill=FILLS[fill])


@pytest.mark.parametrize("extra", [16, 4096])
def test_the_workspace_may_be_larger_than_needed(flint, extra):
    dx = make_dx("normal", 700, 8, 73)
    tokens = make_tokens("uniform", 700, 64, 73)

    expect_exact(flint, dx, tokens, 7, 100, 64, 8, workspace_extra=extra)


# -- Vocabulary edges -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("vocab", [300, GPT2_VOCAB])
def test_tokens_0_and_the_last_one_are_summed_and_nothing_runs_past_the_table(flint, vocab):
    # run_flint's guard bands around dwte catch a write one row past it (and before it).
    rows, width = 400, 8
    dx = make_dx("normal", rows, width, 80)
    tokens = make_tokens("uniform", rows, vocab, 80)
    tokens[::40] = 0
    tokens[5::40] = vocab - 1

    got, exp = expect_exact(flint, dx, tokens, 4, 100, vocab, width)

    assert bool(exp.reached[0]) and bool(exp.reached[vocab - 1])
    assert not all_equal(raw_bits(got.dwte)[0], 0) and not all_equal(raw_bits(got.dwte)[vocab - 1], 0)


@pytest.mark.parametrize("accumulate", [False, True], ids=["write", "accumulate"])
def test_a_vocab_of_1(flint, accumulate):
    # Every token is 0 or past the table: the only row sums the rows of token 0.
    rows, width = 300, 16
    dx = make_dx("wide", rows, width, 81)
    tokens = make_tokens("uniform", rows, 4, 81)
    old = make_old((1, width), 82)

    got, exp = expect_exact(flint, dx, tokens, 3, 100, 1, width, old_wte=old, accumulate_wte=accumulate)

    assert got.workspace["offsets"].tolist() == [0, int((tokens == 0).sum())]


def test_a_vocab_of_65536_and_the_last_token(flint):
    # offsets[65536] = the number of rows: the last entry of a table of 65,537.
    rows, width = 3000, 8
    dx = make_dx("normal", rows, width, 83)
    tokens = make_tokens("uniform", rows, MAX_VOCAB, 83)
    tokens[[0, 1, 2999]] = 65535
    tokens[[3, 2998]] = 0

    got, exp = expect_exact(flint, dx, tokens, 3, 1000, MAX_VOCAB, width)

    assert got.workspace["offsets"][-1].item() == rows
    assert got.workspace["offsets"][65535].item() == rows - int((tokens == 65535).sum()) <= rows - 3
    assert bool(exp.reached[65535]) and bool(exp.reached[0])


@pytest.mark.parametrize("accumulate", [False, True], ids=["write", "accumulate"])
def test_gpt2s_padded_table_leaves_its_unused_rows(flint, accumulate):
    # Tokens up to 50256 (<|endoftext|>) of the padded 50,304: rows 50,257 to 50,303 are +0 in write
    # mode and unchanged in accumulate mode.
    rows, width = 3000, 8
    dx = make_dx("normal", rows, width, 84)
    tokens = make_tokens("uniform", rows, 50257, 84)
    tokens[[0, 100]] = 50256
    old = make_old((GPT2_VOCAB, width), 85)

    got, exp = expect_exact(flint, dx, tokens, 3, 1000, GPT2_VOCAB, width, old_wte=old, accumulate_wte=accumulate)

    assert bool(exp.reached[50256]) and not bool(exp.reached[50257:].any())
    tail = raw_bits(got.dwte)[50257:]
    if accumulate:
        assert torch.equal(tail, raw_bits(old)[50257:])
    else:
        assert all_equal(tail, 0)


@pytest.mark.parametrize("vocab, width, past", [(GPT2_VOCAB, 8, [50304, 65535, 50305]), (1, 8, [1, 2, 65535]),
                                                (300, 16, [300, 301, 40000])], ids=str)
@pytest.mark.parametrize("accumulate", [False, True], ids=["write", "accumulate"])
def test_tokens_at_or_above_vocab_reach_no_row_of_dwte_but_reach_dwpe(flint, vocab, width, past, accumulate):
    sequences, length = 4, 150
    rows = sequences * length
    dx = make_dx("normal", rows, width, 86)
    tokens = make_tokens("uniform", rows, vocab, 86)
    tokens[::6] = np.array(past, dtype=np.uint16)[np.arange(len(tokens[::6])) % len(past)]
    below = int((tokens.astype(np.int64) < vocab).sum())
    old = make_old((vocab, width), 87)

    got, exp = expect_exact(flint, dx, tokens, sequences, length, vocab, width, old_wte=old, old_wpe=make_old((length, width), 88),
                            accumulate_wte=accumulate, accumulate_wpe=accumulate)

    # (the guard bands around dwte hold: run_flint checks) offsets[vocab] counts the tokens below vocab.
    assert got.workspace["offsets"][-1].item() == below < rows
    assert bool((got.workspace["sorted_tokens"].numpy().view(np.uint16)[below:] >= vocab).all())
    # dwpe doesn't depend on the tokens: the same with every token 0.
    again = run_flint(flint, torch.from_numpy(dx).cuda(), tokens_tensor(np.zeros(rows, dtype=np.uint16)), sequences,
                      length, vocab, width, old_wpe=make_old((length, width), 88), accumulate_wpe=accumulate)
    assert torch.equal(raw_bits(again.dwpe), raw_bits(got.dwpe))


# -- Edge structures ----------------------------------------------------------------------------------------------------------


EDGE_CASES = shrunk_cases([
    (1, 1, 1, 8, "one", "normal"),
    (1, 1, GPT2_VOCAB, 8, "uniform", "wide"),
    (4, 500, GPT2_VOCAB, 8, "distinct", "wide"),
    (2, 8192, 64, 8, "one", "normal"),
    (2, 1500, 1000, 16, "ends", "outliers_same"),
    (3, 1000, 1000, 8, "gaps", "normal"),
    (1, 2048, 65536, 8, "distinct", "wide"),
    (1, 2048, 1000, 8, "ends", "wide"),
])


@pytest.mark.parametrize("sequences, length, vocab, width, token_kind, dx_kind", case_params(EDGE_CASES))
def test_edge_structures_equal_the_emulation(flint, sequences, length, vocab, width, token_kind, dx_kind):
    # One row; all tokens different; every row one token (16,384 rows); runs at the start and the
    # end of the sorted array; empty rows of the table between present ones.
    rows = sequences * length
    dx = make_dx(dx_kind, rows, width, 90 + rows)
    tokens = make_tokens(token_kind, rows, vocab, 90 + rows)

    got, exp = expect_exact(flint, dx, tokens, sequences, length, vocab, width)

    if token_kind == "gaps":
        assert int(exp.reached.sum()) < vocab // 50
        assert all_equal(raw_bits(got.dwte)[torch.from_numpy(~exp.reached)], 0)


# -- inf and NaN --------------------------------------------------------------------------------------------------------------------

NF_SHAPE = (3, 100, 20, 24)  # sequences, length, vocab, width
# (row, column, value): token = row % 20, position = row % 100.
NON_FINITE = [
    (5, 3, math.inf),                 # token 5, position 5
    (27, 4, -math.inf),               # token 7, position 27
    (40, 5, math.nan),                # token 0, position 40
    (61, 6, math.inf), (81, 6, -math.inf),    # token 1 twice: NaN; positions 61 and 81 each alone
    (101, 7, math.inf), (121, 7, math.inf),   # token 1 twice: +inf; positions 1 and 21 each alone
    (299, 8, math.nan),               # token 19's last row, position 99
    (150, 9, math.inf),               # a token past vocab: no row of dwte, but position 50 of dwpe
    (141, 10, -math.inf), (221, 10, -math.inf), (261, 10, -math.inf),  # token 1 three times: -inf; positions 41, 21, 61
]


def non_finite_inputs():
    sequences, length, vocab, width = NF_SHAPE
    rows = sequences * length
    dx = make_dx("normal", rows, width, 95)
    tokens = (np.arange(rows) % vocab).astype(np.uint16)
    tokens[150] = 65535
    clean = dx.copy()
    for row, col, value in NON_FINITE:
        dx[row, col] = value
    return dx, clean, tokens


def test_inf_and_nan_reach_their_tokens_row_and_their_positions_row_only(flint):
    sequences, length, vocab, width = NF_SHAPE
    dx, clean, tokens = non_finite_inputs()

    got, exp = expect_exact(flint, dx, tokens, sequences, length, vocab, width)

    # Exactly the entries that hold a non-finite value: those of the tokens' rows (not for the
    # token past vocab) and of the positions' rows.
    want_wte = {(int(tokens[row]), col) for row, col, _ in NON_FINITE if int(tokens[row]) < vocab}
    want_wpe = {(row % length, col) for row, col, _ in NON_FINITE}
    assert {tuple(int(v) for v in p) for p in zip(*np.nonzero(~np.isfinite(got.dwte.numpy())))} == want_wte
    assert {tuple(int(v) for v in p) for p in zip(*np.nonzero(~np.isfinite(got.dwpe.numpy())))} == want_wpe
    dwte, dwpe = got.dwte.numpy(), got.dwpe.numpy()
    assert dwte[5, 3] == math.inf and dwte[7, 4] == -math.inf and math.isnan(dwte[0, 5])
    assert math.isnan(dwte[1, 6]) and dwte[1, 7] == math.inf and math.isnan(dwte[19, 8]) and dwte[1, 10] == -math.inf
    assert dwpe[50, 9] == math.inf and dwpe[61, 6] == math.inf and dwpe[81, 6] == -math.inf
    # The other entries are those of the clean input's run.
    clean_run = run_flint(flint, torch.from_numpy(clean).cuda(), tokens_tensor(tokens), sequences, length, vocab, width)
    keep_wte = np.ones((vocab, width), dtype=bool)
    keep_wpe = np.ones((length, width), dtype=bool)
    for row, col, _ in NON_FINITE:
        if int(tokens[row]) < vocab:
            keep_wte[int(tokens[row]), col] = False
        keep_wpe[row % length, col] = False
    assert_bits(got.dwte.numpy()[keep_wte], clean_run.dwte.numpy()[keep_wte], "the entries no inf or NaN reaches (dwte)")
    assert_bits(got.dwpe.numpy()[keep_wpe], clean_run.dwpe.numpy()[keep_wpe], "the entries no inf or NaN reaches (dwpe)")


def placement(t):
    return torch.isnan(t), torch.isposinf(t), torch.isneginf(t)


def test_the_non_finite_entries_are_where_fp32_pytorchs_are(flint):
    sequences, length, vocab, width = NF_SHAPE
    dx, _, tokens = non_finite_inputs()
    dx_gpu = torch.from_numpy(dx).cuda()
    keep = torch.from_numpy(tokens.astype(np.int64) < vocab).cuda()
    indices = torch.from_numpy(tokens.astype(np.int64)).cuda()

    got = run_flint(flint, dx_gpu, tokens_tensor(tokens), sequences, length, vocab, width)

    peer_w = peer_dwte(dx_gpu[keep], indices[keep], vocab)
    peer_p = peer_dwpe(dx_gpu, sequences, length, width)
    for name, mine, peer in (("dwte", got.dwte, peer_w), ("dwpe", got.dwpe, peer_p)):
        for kind, a, b in zip(("NaN", "+inf", "-inf"), placement(mine.cuda()), placement(peer)):
            assert torch.equal(a, b), f"{name}: the places of {kind} differ from PyTorch's"
    assert bool(torch.isfinite(got.dwte).logical_not().any()) and bool(torch.isfinite(got.dwpe).logical_not().any())


def test_large_values_overflow_to_inf_where_pytorchs_do_and_cancelling_ones_do_not(flint):
    # Column 3: +3e38 in every row, so every token's sum overflows to +inf after two adds (and every
    # position's, over 48 sequences); column 5 likewise to -inf; column 7 alternates in sign with the
    # row, so a token's rows (which alternate in parity) never overflow and sum to exactly 0, while a
    # position's rows (all one parity) do.
    sequences, length, vocab, width = 48, 2, 8, 16
    rows = sequences * length
    dx = make_dx("normal", rows, width, 96)
    dx[:, 3] = F32(3e38)
    dx[:, 5] = F32(-3e38)
    dx[:, 7] = np.where(np.arange(rows) % 2 == 0, F32(3e38), F32(-3e38))
    tokens = (np.arange(rows) % 3).astype(np.uint16)
    dx_gpu = torch.from_numpy(dx).cuda()
    indices = torch.from_numpy(tokens.astype(np.int64)).cuda()

    got, exp = expect_exact(flint, dx, tokens, sequences, length, vocab, width)

    assert all_equal(got.dwte[:3, 3], math.inf) and all_equal(got.dwte[:3, 5], -math.inf)
    assert all_equal(raw_bits(got.dwte[:3, 7]), 0), "alternating values did not cancel exactly"
    assert got.dwpe[0, 7].item() == math.inf and got.dwpe[1, 7].item() == -math.inf
    peer_w, peer_p = peer_dwte(dx_gpu, indices, vocab), peer_dwpe(dx_gpu, sequences, length, width)
    other = torch.ones(width, dtype=torch.bool)
    other[7] = False
    for kind, a, b in zip(("NaN", "+inf", "-inf"), placement(got.dwte.cuda()), placement(peer_w)):
        assert torch.equal(a[:, other.cuda()], b[:, other.cuda()]), f"dwte: the places of {kind} differ from PyTorch's"
    for kind, a, b in zip(("NaN", "+inf", "-inf"), placement(got.dwpe.cuda()), placement(peer_p)):
        assert torch.equal(a, b), f"dwpe: the places of {kind} differ from PyTorch's"


# -- Determinism and streams ---------------------------------------------------------------------------------------------------------

DETERMINISM_CASES = shrunk_cases([
    (16, 1024, GPT2_VOCAB, 768, "zipf", "wide"),
    (3, 5, 64, 8, "uniform", "wide"),
    (8453, 1, 64, 8, "repeated", "outliers_same"),
    (1, 17 * 2048 + 3, 64, 8, "uniform", "normal"),
])


@pytest.mark.parametrize("sequences, length, vocab, width, token_kind, dx_kind", case_params(DETERMINISM_CASES))
def test_run_twice_is_bitwise_identical(flint, sequences, length, vocab, width, token_kind, dx_kind):
    rows = sequences * length
    dx = make_dx(dx_kind, rows, width, 97)
    dx[rows // 2, width // 2] = np.nan
    tokens = make_tokens(token_kind, rows, vocab, 97)
    dx_gpu, tokens_gpu = torch.from_numpy(dx).cuda(), tokens_tensor(tokens)

    first = run_flint(flint, dx_gpu, tokens_gpu, sequences, length, vocab, width, offset=4, wte_fill=FILLS["nan"],
                      wpe_fill=FILLS["nan"], workspace_fill=FILLS["nan"], with_workspace=True)
    second = run_flint(flint, dx_gpu, tokens_gpu, sequences, length, vocab, width, offset=8, wte_fill="garbage",
                       wpe_fill="garbage", workspace_fill="garbage", with_workspace=True)
    third = run_flint(flint, dx_gpu, tokens_gpu, sequences, length, vocab, width, offset=12, wte_fill=0, wpe_fill=0,
                      workspace_fill=0, with_workspace=True)

    assert bool(torch.isnan(first.dwte).any()) and bool(torch.isnan(first.dwpe).any())
    for name, other in (("second", second), ("third", third)):
        assert torch.equal(raw_bits(other.dwte), raw_bits(first.dwte)), f"the {name} run's dwte differs"
        assert torch.equal(raw_bits(other.dwpe), raw_bits(first.dwpe)), f"the {name} run's dwpe differs"
        for part in ("sorted_tokens", "sorted_rows", "offsets"):
            assert torch.equal(other.workspace[part], first.workspace[part]), f"the {name} run's {part} differs"
        assert torch.equal(raw_bits(other.workspace["partials"]), raw_bits(first.workspace["partials"]))


def test_a_workspace_left_by_a_different_call_changes_nothing(flint):
    # The same workspace serves a big call, then a small one with other tokens and values, with no
    # refill: nothing the first left in it reaches the second's results.
    big = (4, 600, 3000, 24)
    small = (3, 100, 300, 16)
    needed = max(flint.embedding_backward_workspace(*s) for s in (big, small))
    workspace = torch.randint(-(2**31), 2**31, (needed // 4,), dtype=torch.int32, device="cuda")
    for (sequences, length, vocab, width), tokens_kind, seed in ((big, "zipf", 98), (small, "uniform", 99)):
        rows = sequences * length
        dx = make_dx("wide", rows, width, seed)
        tokens = make_tokens(tokens_kind, rows, vocab, seed)
        got = run_flint(flint, torch.from_numpy(dx).cuda(), tokens_tensor(tokens), sequences, length, vocab, width,
                        shared_workspace=workspace)
        exp = emulate_all(dx, tokens, sequences, length, vocab)
        check_exact(got, exp, f"{sequences}x{length}x{vocab}x{width} on a shared workspace")


STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
def test_runs_on_the_given_stream(flint, mode, accumulate):
    sequences, length, vocab, width = 5, 140, 300, 64
    rows = sequences * length
    dx_np = make_dx("normal", rows, width, 91)
    tokens_np = make_tokens("zipf", rows, vocab, 91)
    old_wte, old_wpe = make_old((vocab, width), 92), make_old((length, width), 93)
    exp = emulate_all(dx_np, tokens_np, sequences, length, vocab, old_wte if accumulate else None,
                      old_wpe if accumulate else None)
    source_dx, source_tokens = torch.from_numpy(dx_np).cuda(), tokens_tensor(tokens_np)
    gpu_old_wte, gpu_old_wpe = torch.from_numpy(old_wte).cuda(), torch.from_numpy(old_wpe).cuda()
    dx = torch.full((rows, width), math.nan, device="cuda")
    tokens = torch.zeros(rows, dtype=torch.int16, device="cuda")
    dwte = torch.full((vocab, width), math.nan, device="cuda")
    dwpe = torch.full((length, width), math.nan, device="cuda")
    workspace = torch.full((flint.embedding_backward_workspace(sequences, length, vocab, width) // 4,), math.nan,
                           device="cuda")
    stream = torch.cuda.Stream()
    # A first call loads the kernels' module, which takes longer than the sleep below: without this
    # warm-up a call on the wrong stream would start late enough to pass.
    run_flint(flint, source_dx, source_tokens, sequences, length, vocab, width)
    torch.cuda.synchronize()

    def enqueue(call_stream):
        # The copies into the inputs finish only after the sleep, so the outputs are right only if
        # all the kernels are queued on the same stream behind them (dx is NaN and every token 0
        # where they ran first). With the current stream the call has no stream argument, and
        # torch's current stream is the side one; with the default stream there is no side one.
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        dx.copy_(source_dx)
        tokens.copy_(source_tokens)
        if accumulate:
            dwte.copy_(gpu_old_wte)
            dwpe.copy_(gpu_old_wpe)
        flint.embedding_backward(dwte, dwpe, dx, tokens, sequences, length, vocab, width, accumulate_wte=accumulate,
                                 accumulate_wpe=accumulate, workspace=workspace, stream=call_stream)

    if mode == "default-stream":
        enqueue(None)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            enqueue(stream if mode == "explicit-stream" else None)
        stream.synchronize()

    assert_bits(dwte, exp.dwte, f"dwte {mode}")
    assert_bits(dwpe, exp.dwpe, f"dwpe {mode}")


# -- The workspace size ---------------------------------------------------------------------------------------------------------------

WORKSPACE_SHAPES = [
    (1, 1, 1, 8), (1, 1, 1, 768), (3, 5, 64, 8), (16, 1024, GPT2_VOCAB, 768), (1, 1024, GPT2_VOCAB, 768),
    (8, 1024, GPT2_VOCAB, 768), (4, 256, 1000, 64), (1, 2047, 100, 8), (1, 2048, 100, 8), (1, 2049, 100, 8),
    (8453, 1, 64, 8), (33 * 256 + 5, 1, 64, 8), (4, 2**18, 64, 8), (2, 1000, MAX_VOCAB, 8), (1, 21845, 1, 768),
    (1, 1, 1, MAX_WIDTH), (1, 1, MAX_VOCAB, MAX_WIDTH), (2, 21845, 1, 768), (1, 2097120, 3, 8),
    (MAX_ROWS, 1, 64, 8), (65537, 65535, 64, 8), (MAX_ROWS, 1, MAX_VOCAB, 8), (2**20, 4095, 5, 8),
]


@pytest.mark.parametrize("sequences, length, vocab, width", [pytest.param(*s, id=case_id(s)) for s in WORKSPACE_SHAPES])
def test_workspace_is_the_documented_layouts_size(flint, sequences, length, vocab, width):
    needed = flint.embedding_backward_workspace(sequences, length, vocab, width)

    assert needed == layout(sequences, length, vocab, width)["bytes"]
    # The partials come first, and the parts after them are each 16-byte aligned.
    assert needed >= bg.expected_workspace(sequences, length * width) + sequences * length * 12 + (vocab + 1) * 4
    assert needed % 16 == 0


# -- The limits --------------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(SANITIZER, reason="the widest dx is too big for the sanitizer")
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
def test_the_widest_row(flint, accumulate):
    # One row of 16,776,960 values: dwte (vocab 1) and dwpe are that row (-0 as +0), or old + it.
    width = MAX_WIDTH
    dx = make_dx("wide", 1, width, 100)
    tokens = np.zeros(1, dtype=np.uint16)
    old_wte, old_wpe = make_old((1, width), 101), make_old((1, width), 102)

    got = run_flint(flint, torch.from_numpy(dx).cuda(), tokens_tensor(tokens), 1, 1, 1, width, old_wte=old_wte,
                    old_wpe=old_wpe, accumulate_wte=accumulate, accumulate_wpe=accumulate, extra_rows=1)

    with np.errstate(all="ignore"):
        # One row: the sum is dx + 0 (a -0 as +0), then old + it.
        expected_wte = old_wte + (dx + F32(0.0)) if accumulate else dx + F32(0.0)
        expected_wpe = old_wpe + (dx + F32(0.0)) if accumulate else dx + F32(0.0)
    assert_bits(got.dwte, expected_wte, "dwte")
    assert_bits(got.dwpe, expected_wpe, "dwpe")


@pytest.mark.skipif(SANITIZER, reason="the widest position gradient is too big for the sanitizer")
@pytest.mark.parametrize("accumulate", [False, True], ids=["overwrite", "accumulate"])
def test_the_widest_position_gradient(flint, accumulate):
    # length * width = 16,776,960 exactly (21,845 x 768), over 2 sequences, with vocab 1 and tokens
    # past it.
    sequences, length, vocab, width = 2, 21845, 1, 768
    assert length * width == MAX_POSITION_VALUES
    rows = sequences * length
    dx = make_dx("gradient", rows, width, 103)
    tokens = make_tokens("uniform", rows, 3, 103)
    old_wte, old_wpe = make_old((vocab, width), 104), make_old((length, width), 105)

    expect_exact(flint, dx, tokens, sequences, length, vocab, width, old_wte=old_wte, old_wpe=old_wpe,
                 accumulate_wte=accumulate, accumulate_wpe=accumulate, extra_rows=1)


@pytest.mark.skipif(SANITIZER, reason="the biggest vocabulary's dx is too big for the sanitizer")
def test_every_token_of_a_65536_vocabulary(flint):
    # Each of the 65,536 rows of the table is reached once: dwte is dx, permuted, and the offsets
    # are 0, 1, 2, ... 65,536.
    sequences, length, vocab, width = 64, 1024, MAX_VOCAB, 8
    dx = make_dx("wide", sequences * length, width, 106)
    tokens = make_tokens("distinct", sequences * length, vocab, 106)

    got, exp = expect_exact(flint, dx, tokens, sequences, length, vocab, width)

    assert got.workspace["offsets"].tolist() == list(range(vocab + 1))
    assert_bits(got.dwte[torch.from_numpy(tokens.astype(np.int64))], dx + F32(0.0), "dwte is dx, permuted")


# -- Argument errors ------------------------------------------------------------------------------------------------------------------------

NAMES = ("dwte", "dwpe", "workspace", "dx", "tokens")
OUTPUT_NAMES = NAMES[:3]
SEQ, LEN, VOC, WID = 3, 8, 7, 8  # 24 rows: every buffer's size is a multiple of 16 bytes
ROWS = SEQ * LEN
WORKSPACE_NEEDED = layout(SEQ, LEN, VOC, WID)["bytes"]
BYTES = {"dwte": VOC * WID * 4, "dwpe": LEN * WID * 4, "workspace": WORKSPACE_NEEDED, "dx": ROWS * WID * 4,
         "tokens": ROWS * 2}
SPACING = 16384
ARENA_SLOT = {name: (i + 2) * SPACING for i, name in enumerate(NAMES)}
FILL = 0x5A


def new_arena():
    return torch.full((8 * SPACING,), FILL, dtype=torch.uint8, device="cuda")


def addresses(arena):
    return {name: arena.data_ptr() + ARENA_SLOT[name] for name in NAMES}


def call(flint, a, sequences=SEQ, length=LEN, vocab=VOC, width=WID, workspace_bytes=WORKSPACE_NEEDED,
         accumulate_wte=False, accumulate_wpe=False):
    flint.embedding_backward(a["dwte"], a["dwpe"], a["dx"], a["tokens"], sequences, length, vocab, width,
                             accumulate_wte=accumulate_wte, accumulate_wpe=accumulate_wpe,
                             workspace=a["workspace"], workspace_bytes=workspace_bytes)


def expect_invalid_argument(flint, reason, arena, a, **kwargs):
    """The call fails with the status and message naming the problem, and nothing in the arena
    changed."""
    before = arena.clone()
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, a, **kwargs)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in (NAME, reason):
        assert text in str(error.value)
        assert text in flint.last_error()
    torch.cuda.synchronize()
    assert torch.equal(arena, before), "the call wrote something"


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    arena = new_arena()
    expect_invalid_argument(flint, f"{name} is null", arena, {**addresses(arena), name: None})


@pytest.mark.parametrize("offset", [2, 4, 8])
@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name, offset):
    # 2 bytes is as far off as a token goes, 4 as far as an FP32, and 8 is aligned for 64 bits but
    # not 128.
    arena = new_arena()
    a = addresses(arena)
    a[name] += offset
    expect_invalid_argument(flint, f"{name} is not 16-byte aligned", arena, a)


@pytest.mark.parametrize("name", NAMES)
def test_a_buffer_may_not_run_past_the_address_space(flint, name):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"{name} runs past the end of the address space", arena, {**addresses(arena), name: 2**64 - 16}
    )


def test_pointers_are_checked_in_order(flint):
    # Per buffer: null, then alignment, then the address space; in the order dwte, dwpe, workspace, dx,
    # tokens; and all of that before any overlap.
    arena = new_arena()
    a = addresses(arena)
    for i, name in enumerate(NAMES[:-1]):
        later = NAMES[i + 1]
        expect_invalid_argument(flint, f"{name} is not 16-byte aligned", arena, {**a, name: a[name] + 4, later: None})
        expect_invalid_argument(flint, f"{name} is null", arena, {**a, name: None, later: a[later] + 4})
        expect_invalid_argument(flint, f"{name} runs past the end", arena, {**a, name: 2**64 - 16, later: None})
    expect_invalid_argument(flint, "tokens is null", arena, {**a, "dwte": a["dwpe"], "tokens": None})


@pytest.mark.parametrize(
    "width", [0, 1, 4, 7, 9, 12, 20, 25, MAX_WIDTH + 1, MAX_WIDTH + 8, 2**32, 2**64 - 8, 2**64 - 1], ids=str
)
def test_width_must_be_a_multiple_of_8_from_8_to_16776960(flint, width):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"width ({width}) must be a multiple of 8 from 8 to {MAX_WIDTH}", arena, addresses(arena), width=width
    )


@pytest.mark.parametrize("vocab", [0, 65537, 2**32, 2**64 - 1], ids=str)
def test_vocab_must_be_from_1_to_65536(flint, vocab):
    arena = new_arena()
    expect_invalid_argument(flint, f"vocab ({vocab}) must be from 1 to {MAX_VOCAB}", arena, addresses(arena), vocab=vocab)


LENGTH_CASES = list(dict.fromkeys(
    (width, length)
    for width in (8, 24, 768, MAX_WIDTH)
    for length in (0, MAX_POSITION_VALUES // width + 1, 2**32, 2**64 - 1)
))


@pytest.mark.parametrize("width, length", LENGTH_CASES, ids=str)
def test_length_must_be_from_1_with_length_x_width_at_most_16776960(flint, width, length):
    arena = new_arena()
    expect_invalid_argument(
        flint, f"length ({length}) must be from 1 to {MAX_POSITION_VALUES // width}", arena, addresses(arena),
        width=width, length=length,
    )


SEQUENCE_CASES = list(dict.fromkeys(
    (length, sequences)
    for length in (1, 5, 1024, 65535)
    for sequences in (0, MAX_ROWS // length + 1, 2**32, 2**63, 2**64 - 1)
))


@pytest.mark.parametrize("length, sequences", SEQUENCE_CASES, ids=str)
def test_sequences_must_be_from_1_with_at_most_4294967295_rows(flint, length, sequences):
    # Including the products that wrap around 2^64 (2^63 x 2 and the like), which must not pass.
    arena = new_arena()
    expect_invalid_argument(
        flint, f"sequences ({sequences}) must be from 1 to {MAX_ROWS // length}", arena, addresses(arena),
        sequences=sequences, length=length,
    )


def test_size_limits_are_checked_in_order(flint):
    # width, then vocab, then length, then sequences, then the workspace's size, then the pointers
    # (null ones included).
    arena = new_arena()
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(flint, "width (7) must be", arena, nothing, width=7, vocab=0, length=0, sequences=0,
                            workspace_bytes=0)
    expect_invalid_argument(flint, "vocab (0) must be", arena, nothing, vocab=0, length=0, sequences=0, workspace_bytes=0)
    expect_invalid_argument(flint, "length (0) must be", arena, nothing, length=0, sequences=0, workspace_bytes=0)
    expect_invalid_argument(flint, "sequences (0) must be", arena, nothing, sequences=0, workspace_bytes=0)
    expect_invalid_argument(flint, f"needs {WORKSPACE_NEEDED} bytes of workspace, given 0", arena, nothing,
                            workspace_bytes=0)
    expect_invalid_argument(flint, "dwte is null", arena, nothing)


@pytest.mark.parametrize("short", [1, 4, 16, 17, WORKSPACE_NEEDED])
def test_a_workspace_may_not_be_too_small(flint, short):
    arena = new_arena()
    given = WORKSPACE_NEEDED - short
    expect_invalid_argument(
        flint, f"needs {WORKSPACE_NEEDED} bytes of workspace, given {given}", arena, addresses(arena),
        workspace_bytes=given,
    )


def test_a_null_workspace_is_a_null_pointer_with_enough_bytes_and_too_small_with_none(flint):
    arena = new_arena()
    a = {**addresses(arena), "workspace": None}
    expect_invalid_argument(flint, f"needs {WORKSPACE_NEEDED} bytes of workspace, given 0", arena, a, workspace_bytes=0)
    expect_invalid_argument(flint, "workspace is null", arena, a, workspace_bytes=WORKSPACE_NEEDED)


@pytest.mark.parametrize(
    "sequences, length, vocab, width",
    [(MAX_ROWS, 1, 8, 8), (1, MAX_POSITION_VALUES // 8, 8, 8), (1, 21845, 8, 768), (1, 1, MAX_VOCAB, MAX_WIDTH),
     (65537, 65535, 8, 8), (1, 1, 1, MAX_WIDTH), (2**20, 4095, 65536, 8)],
    ids=lambda v: str(v),
)
def test_the_largest_sizes_pass_the_size_checks(flint, sequences, length, vocab, width):
    # The limits themselves aren't errors of size: the call gets on to the workspace's size (and with
    # enough bytes claimed, on to a null dwte), even where the workspace is tens of gigabytes.
    arena = new_arena()
    needed = layout(sequences, length, vocab, width)["bytes"]
    expect_invalid_argument(
        flint, f"needs {needed} bytes of workspace, given {WORKSPACE_NEEDED}", arena, addresses(arena),
        sequences=sequences, length=length, vocab=vocab, width=width,
    )
    expect_invalid_argument(
        flint, "dwte is null", arena, {**addresses(arena), "dwte": None}, sequences=sequences, length=length,
        vocab=vocab, width=width, workspace_bytes=needed,
    )


def overlapping_starts(first, second):
    """Where `second` starts relative to `first`, 16-byte aligned, so that they overlap: at the same
    address, 16 bytes in, 16 bytes before first's end, and (negative) with second ending 16 bytes
    into first."""
    a, b = BYTES[first], BYTES[second]
    return {"same-start": 0, "16-in": 16, "16-before-end": a - 16, "ends-16-in": 16 - b}


def touching_starts(first, second):
    """Where `second` starts so that it touches `first` without overlapping it: just after, and
    just before."""
    return {"just-after": BYTES[first], "just-before": -BYTES[second]}


# Every pair with an output in it, in the order of the argument list (outputs first): the checked
# ones. dx and tokens may overlap each other.
PAIRS = [(a, b) for i, a in enumerate(OUTPUT_NAMES) for b in NAMES[i + 1 :]]
OVERLAP_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in PAIRS
    for label, offset in overlapping_starts(a, b).items()
]
TOUCHING_CASES = [
    pytest.param(a, b, label, offset, id=f"{a}-{b}-{label}")
    for a, b in PAIRS
    for label, offset in touching_starts(a, b).items()
]


def test_there_are_9_pairs_with_an_output():
    assert len(PAIRS) == 9 and all(a in OUTPUT_NAMES for a, _ in PAIRS)


@pytest.mark.parametrize("first, second, label, offset", OVERLAP_CASES)
def test_each_output_may_not_overlap_any_other_buffer(flint, first, second, label, offset):
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no harm to
    # anything else.
    arena = new_arena()
    a = addresses(arena)
    a[second] = a[first] + offset
    expect_invalid_argument(flint, f"{first} and {second} overlap", arena, a)


@pytest.mark.parametrize("first, second, label, offset", TOUCHING_CASES)
def test_buffers_next_to_each_other_are_not_overlapping(flint, first, second, label, offset):
    arena = new_arena()
    start = dict(ARENA_SLOT)
    start[second] = start[first] + offset
    dx = make_dx("normal", ROWS, WID, 120)
    tokens = make_tokens("uniform", ROWS, VOC, 120)
    arena[start["dx"] : start["dx"] + BYTES["dx"]].view(torch.float32).copy_(torch.from_numpy(dx).reshape(-1))
    arena[start["tokens"] : start["tokens"] + BYTES["tokens"]].view(torch.int16).copy_(
        torch.from_numpy(tokens.view(np.int16)))
    used = torch.zeros(arena.numel(), dtype=torch.bool, device="cuda")
    for name in OUTPUT_NAMES:
        used[start[name] : start[name] + BYTES[name]] = True
    outside_before = arena[~used].clone()
    base = arena.data_ptr()
    a = {name: base + start[name] for name in NAMES}

    call(flint, a)
    torch.cuda.synchronize()

    assert torch.equal(arena[~used], outside_before), "something outside the outputs and the workspace changed"
    exp = emulate_all(dx, tokens, SEQ, LEN, VOC)
    dwte = arena[start["dwte"] : start["dwte"] + BYTES["dwte"]].view(torch.float32).reshape(VOC, WID)
    dwpe = arena[start["dwpe"] : start["dwpe"] + BYTES["dwpe"]].view(torch.float32).reshape(LEN, WID)
    assert_bits(dwte, exp.dwte, f"dwte {label}")
    assert_bits(dwpe, exp.dwpe, f"dwpe {label}")


def test_dx_and_tokens_may_share_memory(flint):
    # Both are only read: the tokens here are the first bytes of dx (which hold small subnormal
    # values, so every token is below vocab), and the call goes ahead.
    arena = new_arena()
    a = addresses(arena)
    a["tokens"] = a["dx"]
    rng = np.random.default_rng(121)
    dx = make_dx("normal", ROWS, WID, 121)
    first = rng.integers(0, VOC, 12 * 2).astype(np.uint32)
    dx.reshape(-1)[:12] = (first[1::2] << 16 | first[0::2]).view(F32)  # 24 uint16 tokens, all below vocab
    arena[ARENA_SLOT["dx"] : ARENA_SLOT["dx"] + BYTES["dx"]].view(torch.float32).copy_(torch.from_numpy(dx).reshape(-1))
    tokens = dx.reshape(-1).view(np.uint16)[:ROWS]
    assert int(tokens.max()) < VOC
    used = torch.zeros(arena.numel(), dtype=torch.bool, device="cuda")
    for name in OUTPUT_NAMES:
        used[ARENA_SLOT[name] : ARENA_SLOT[name] + BYTES[name]] = True
    outside_before = arena[~used].clone()

    call(flint, a)
    torch.cuda.synchronize()

    assert torch.equal(arena[~used], outside_before), "something outside the outputs changed"
    exp = emulate_all(dx, tokens, SEQ, LEN, VOC)
    dwte = arena[ARENA_SLOT["dwte"] : ARENA_SLOT["dwte"] + BYTES["dwte"]].view(torch.float32).reshape(VOC, WID)
    assert_bits(dwte, exp.dwte, "dwte")


def test_an_error_leaves_nothing_behind_and_a_valid_call_works_after(flint):
    arena = new_arena()
    a = addresses(arena)
    dx = make_dx("normal", ROWS, WID, 122)
    tokens = make_tokens("uniform", ROWS, VOC, 122)
    arena[ARENA_SLOT["dx"] : ARENA_SLOT["dx"] + BYTES["dx"]].view(torch.float32).copy_(torch.from_numpy(dx).reshape(-1))
    arena[ARENA_SLOT["tokens"] : ARENA_SLOT["tokens"] + BYTES["tokens"]].view(torch.int16).copy_(
        torch.from_numpy(tokens.view(np.int16)))
    for kwargs in (dict(width=7), dict(width=0), dict(vocab=0), dict(length=0), dict(sequences=0),
                   dict(sequences=2**64 - 1), dict(workspace_bytes=0)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, a, **kwargs)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "dwte": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "dx": a["dx"] + 2})
    torch.cuda.synchronize()
    for name in OUTPUT_NAMES:
        assert all_equal(arena[ARENA_SLOT[name] : ARENA_SLOT[name] + BYTES[name]], FILL)

    call(flint, a)
    torch.cuda.synchronize()

    exp = emulate_all(dx, tokens, SEQ, LEN, VOC)
    dwte = arena[ARENA_SLOT["dwte"] : ARENA_SLOT["dwte"] + BYTES["dwte"]].view(torch.float32).reshape(VOC, WID)
    assert_bits(dwte, exp.dwte, "dwte after errors")


# -- The workspace function's arguments ----------------------------------------------------------------------------------------------------------


def workspace_status(flint, sequences, length, vocab, width):
    """The raw status of flint_embedding_backward_workspace and the size it left (a sentinel if it
    wrote nothing)."""
    out = ctypes.c_size_t(0xDEADBEEF)
    status = flint.lib.flint_embedding_backward_workspace(sequences, length, vocab, width, ctypes.byref(out))
    return status, out.value


def expect_workspace_error(flint, reason, *args):
    status, size = workspace_status(flint, *args)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert size == 0xDEADBEEF, "bytes was written on an error"
    for text in (WORKSPACE_NAME, reason):
        assert text in flint.last_error()


@pytest.mark.parametrize("width", [0, 1, 4, 7, 9, 25, MAX_WIDTH + 1, MAX_WIDTH + 8, 2**32, 2**64 - 8, 2**64 - 1], ids=str)
def test_the_workspace_function_checks_width(flint, width):
    expect_workspace_error(flint, f"width ({width}) must be a multiple of 8 from 8 to {MAX_WIDTH}", 3, 5, 64, width)
    with pytest.raises(flint_lib.FlintError, match="must be a multiple of 8"):
        flint.embedding_backward_workspace(3, 5, 64, width)


@pytest.mark.parametrize("vocab", [0, 65537, 2**64 - 1], ids=str)
def test_the_workspace_function_checks_vocab(flint, vocab):
    expect_workspace_error(flint, f"vocab ({vocab}) must be from 1 to {MAX_VOCAB}", 3, 5, vocab, 8)


@pytest.mark.parametrize("width, length", LENGTH_CASES[::2], ids=str)
def test_the_workspace_function_checks_length(flint, width, length):
    expect_workspace_error(flint, f"length ({length}) must be from 1 to {MAX_POSITION_VALUES // width}", 3, length, 64, width)


@pytest.mark.parametrize("length, sequences", SEQUENCE_CASES[::2], ids=str)
def test_the_workspace_function_checks_sequences(flint, length, sequences):
    expect_workspace_error(flint, f"sequences ({sequences}) must be from 1 to {MAX_ROWS // length}", sequences, length, 64, 8)


def test_the_workspace_function_checks_in_order(flint):
    # width, vocab, length, sequences.
    expect_workspace_error(flint, "width (7) must be", 0, 0, 0, 7)
    expect_workspace_error(flint, "vocab (0) must be", 0, 0, 0, 8)
    expect_workspace_error(flint, "length (0) must be", 0, 0, 5, 8)
    expect_workspace_error(flint, "sequences (0) must be", 0, 5, 5, 8)


@pytest.mark.parametrize("args", [(3, 5, 64, 8), (0, 5, 64, 8), (3, 5, 64, 7), (0, 0, 0, 0), (2**64 - 1,) * 4], ids=str)
def test_the_workspace_function_needs_a_bytes_pointer(flint, args):
    # The null check comes first, before any size is looked at.
    status = flint.lib.flint_embedding_backward_workspace(*args, None)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert f"{WORKSPACE_NAME}: bytes is null" in flint.last_error()


def test_the_workspace_function_accepts_the_limits(flint):
    for shape in ((MAX_ROWS, 1, 64, 8), (1, 21845, 64, 768), (1, 1, MAX_VOCAB, MAX_WIDTH), (1, 1, 1, 8), (65537, 65535, 1, 8)):
        status, size = workspace_status(flint, *shape)

        assert status == flint_lib.OK
        assert size == layout(*shape)["bytes"]
