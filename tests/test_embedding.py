"""flint_embedding_forward: the embedding's forward pass. For row
r = s * length + t of x (sequences * length rows of width FP32), it adds row
tokens[r] of the word table wte (vocab x width) to row t of the position table
wpe (at least length x width): x[r, :] = wte[tokens[r], :] + wpe[t, :], one
FP32 add each. Tokens are uint16, positions restart at 0 in every sequence,
and a warp takes each row, 8 rows to a block.

The kernel is exact, so every comparison is bitwise against PyTorch's
wte[tokens] + wpe[:length] in FP32 on the GPU (the only thing not compared is a
NaN's payload), with no tolerance and no gate. The tests cover GPT-2's shapes
and the wider models' widths, small odd shapes with partial blocks of rows,
wpe tables longer than length (the extra rows are poisoned with NaN), tokens at
the edges of the vocabulary (including those past 2^15, where a signed 16-bit
read would go wrong), positions restarting per sequence, NaN, inf and signed
zeros, determinism, guard bands around x, stream ordering, and argument
errors."""

import math

import pytest
import torch

import flint_lib
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator

GPT2_VOCAB = 50304
MAX_ROWS = 17_179_869_176

# The output of a guarded call sits `offset` blocks past the start of a buffer
# that is filled with a pattern, which must be intact around it afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A

# Under the sanitizer the GPT-2 shapes shrink. Each is (sequences, length,
# width). GPT-2 small's training batch is 16 x 1024 (and 1 x 1024 for one
# sequence), and the wider models' widths run at a few hundred rows that don't
# fill a last block of 8.
GPT2_SHAPES = (
    [(2, 64, 768), (1, 64, 768), (1, 37, 1024), (1, 37, 1280), (1, 37, 1600)]
    if SANITIZER
    else [(16, 1024, 768), (1, 1024, 768), (3, 199, 1024), (3, 199, 1280), (3, 199, 1600)]
)

# Small and odd: single rows, lengths below and above a block of 8 rows, row
# counts that leave a partial block (1, 51, 297) or fill the last one (56), and
# widths below and above what a warp covers in one step (32 lanes x 4 floats =
# 128).
ODD_SHAPES = [(s, t, w) for s, t in [(1, 1), (3, 17), (7, 8), (9, 33)] for w in (8, 16, 128)]


def shape_id(shape):
    return "x".join(map(str, shape))


def as_int32(value):
    """A 32-bit pattern as the signed value torch.int32 holds."""
    return (value + 0x80000000) % 0x100000000 - 0x80000000


def bits(t):
    """t's FP32 bit patterns, which compare equal for equal bits, NaNs and
    -0.0 included."""
    return t.contiguous().view(torch.int32)


def assert_same(actual, expected):
    """Bitwise equal, except that a NaN may carry any payload."""
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(bits(actual)[~nan], bits(expected)[~nan])


# -- Inputs, the reference, and running Flint -------------------------------


def make_inputs(sequences, length, vocab, width, seed=0, wpe_rows=None):
    """(ids, wte, wpe): realistic weights, N(0, 0.02), and random tokens as int32
    values in [0, vocab)."""
    g = generator(seed)
    wte = 0.02 * torch.randn((vocab, width), generator=g, device="cuda")
    wpe = 0.02 * torch.randn(
        (length if wpe_rows is None else wpe_rows, width), generator=g, device="cuda"
    )
    ids = torch.randint(0, vocab, (sequences * length,), generator=g, device="cuda", dtype=torch.int32)
    return ids, wte, wpe


def token_tensor(ids):
    """The uint16 token values in ids (int32) as the int16 bit patterns torch
    can hold."""
    return as_int16(ids).to(torch.int16)


def reference_embedding(ids, wte, wpe, sequences, length):
    """wte[tokens] + wpe[:length] broadcast over the sequences, in FP32 on the
    GPU, as (sequences * length) x width."""
    width = wte.shape[1]
    x = wte[ids.long()].view(sequences, length, width) + wpe[:length]
    return x.view(sequences * length, width)


def flint_embedding(
    flint, ids, wte, wpe, sequences, length, offset=1, fill=SENTINEL, stream=None
):
    """Runs flint_embedding_forward with x inside a buffer pre-filled with
    `fill` (a 16-bit pattern, repeated for the 32-bit buffer), 4 * offset floats
    in, which keeps it 16-byte aligned. Checks that the buffer around x and the
    inputs are untouched, and returns x as (sequences * length) x width."""
    vocab, width = wte.shape
    rows = sequences * length
    at = 4 * offset
    buffer = torch.full(
        (at + rows * width + GUARD_AFTER,), as_int32(fill * 0x10001), dtype=torch.int32, device="cuda"
    )
    x = buffer[at : at + rows * width].view(torch.float32)
    tokens = token_tensor(ids)
    inputs_before = [t.clone() for t in (tokens, bits(wte), bits(wpe))]

    flint.embedding_forward(x, tokens, wte, wpe, sequences, length, vocab, width, stream=stream)
    torch.cuda.synchronize()

    guard = torch.full((1,), as_int32(fill * 0x10001), dtype=torch.int32, device="cuda")
    assert (buffer[:at] == guard).all(), "flint_embedding_forward wrote before x"
    assert (buffer[at + rows * width :] == guard).all(), "flint_embedding_forward wrote past x"
    for name, t, before in zip(("tokens", "wte", "wpe"), (tokens, bits(wte), bits(wpe)), inputs_before):
        assert torch.equal(t, before), f"{name} changed"
    return x.view(rows, width)


# -- Bitwise against PyTorch ------------------------------------------------


@pytest.mark.parametrize("shape", GPT2_SHAPES, ids=shape_id)
def test_gpt2_shapes_match_pytorch_bitwise(flint, shape):
    sequences, length, width = shape
    ids, wte, wpe = make_inputs(sequences, length, GPT2_VOCAB, width)

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, sequences, length)))


@pytest.mark.parametrize("shape", [(3, 17, 128), (16, 64, 768)], ids=shape_id)
def test_outliers_and_extreme_magnitudes_match_pytorch_bitwise(flint, shape):
    # Beyond realistic weights: values near 3,000 added to small ones, so every
    # add rounds; near FP32's limit, so some overflow to inf; and subnormals,
    # whose sums a kernel that flushed denormals would zero.
    sequences, length, width = shape
    vocab = 64
    ids, wte, wpe = make_inputs(sequences, length, vocab, width)
    g = generator(1)
    wte[:, 3] = 3000.0 * (1.0 + 0.1 * torch.randn(vocab, generator=g, device="cuda"))
    wpe[:, 3] = 1e-3 * torch.randn(length, generator=g, device="cuda")
    wte[:, 5] = 1e-40 * (1.0 + torch.rand(vocab, generator=g, device="cuda"))
    wpe[:, 5] = 1e-40 * (1.0 + torch.rand(length, generator=g, device="cuda"))
    wte[:, 7] = 2e38
    wpe[:, 7] = 2e38 * torch.randn(length, generator=g, device="cuda").sign()

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, sequences, length)))
    # What the inputs were for did happen: rounding at the outlier, subnormal
    # sums that aren't flushed, and both a finite and an overflowing add.
    rows = wte[ids.long(), 3] + wpe[torch.arange(sequences * length, device="cuda") % length, 3]
    assert torch.equal(bits(x[:, 3]), bits(rows))
    assert (x[:, 5] != 0).all() and (x[:, 5].abs() < 2.0**-126).all()
    assert torch.isposinf(x[:, 7]).any() and (x[:, 7] == 0).any()


ODD_CASES = [
    pytest.param(shape, 1 + i % 3, id=shape_id(shape)) for i, shape in enumerate(ODD_SHAPES)
]


@pytest.mark.parametrize("shape, offset", ODD_CASES)
def test_odd_shapes_match_pytorch_and_stay_inside_x(flint, shape, offset):
    # flint_embedding also checks that the guard regions and the inputs are
    # intact.
    sequences, length, width = shape
    for seed in range(2):
        ids, wte, wpe = make_inputs(sequences, length, 97, width, seed)

        x = flint_embedding(flint, ids, wte, wpe, sequences, length, offset)

        assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, sequences, length)))


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 15, 16, 17, 33])
@pytest.mark.parametrize("width", [8, 100], ids=str)
def test_every_row_is_written_and_nothing_else(flint, rows, width):
    # x starts as NaN, so a row (a partial block's last one, say) that is
    # skipped shows; flint_embedding checks the guards around it. One
    # sequence, then rows split into sequences where they divide.
    for sequences, length in [(1, rows)] + ([(rows, 1)] if rows > 1 else []):
        ids, wte, wpe = make_inputs(sequences, length, 50, width + width % 8)
        x = flint_embedding(flint, ids, wte, wpe, sequences, length, fill=0xFFFF)
        assert torch.isfinite(x).all()
        assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, sequences, length)))


def test_only_the_first_length_rows_of_wpe_are_read(flint):
    # Rows past length hold NaN, which must not reach x.
    sequences, length, width = 3, 17, 64
    ids, wte, wpe = make_inputs(sequences, length, 97, width, wpe_rows=length + 9)
    wpe[length:] = math.nan

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    assert torch.isfinite(x).all()
    assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, sequences, length)))


def test_wpe_with_more_rows_is_the_same_as_exactly_length(flint):
    sequences, length, width = 2, 33, 32
    ids, wte, wpe = make_inputs(sequences, length, 97, width, wpe_rows=length + 5)

    longer = flint_embedding(flint, ids, wte, wpe, sequences, length)
    exact = flint_embedding(flint, ids, wte, wpe[:length].clone(), sequences, length)

    assert torch.equal(bits(longer), bits(exact))


# -- Tokens -----------------------------------------------------------------

EDGE_TOKENS = [
    pytest.param(GPT2_VOCAB, 16, [0, 50256, 50257, 50303, 50303, 50257, 50256, 0], id="gpt2"),
    # Around 2^15, where an int16 read of a uint16 token goes negative.
    pytest.param(
        65536, 8, [0, 1, 32767, 32768, 32769, 65534, 65535, 65535, 40000, 32768, 1, 0], id="uint16"
    ),
]


@pytest.mark.parametrize("vocab, width, token_values", EDGE_TOKENS)
def test_tokens_at_the_edges_of_the_vocabulary(flint, vocab, width, token_values):
    sequences, length = 2, len(token_values) // 2
    _, wte, wpe = make_inputs(sequences, length, vocab, width)
    ids = torch.tensor(token_values, dtype=torch.int32, device="cuda")

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, sequences, length)))
    # The last row of the table, read by token vocab - 1, is not a neighbor's.
    last = token_values.index(vocab - 1)
    assert torch.equal(bits(x[last]), bits(wte[vocab - 1] + wpe[last % length]))


def test_every_token_of_a_65536_vocabulary(flint):
    # Each of the 65,536 rows of the table is read once, in order.
    vocab, width = 65536, 8
    sequences, length = 64, 1024
    g = generator(0)
    wte = torch.randn((vocab, width), generator=g, device="cuda")
    wpe = torch.randn((length, width), generator=g, device="cuda")
    ids = torch.arange(vocab, dtype=torch.int32, device="cuda")

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, sequences, length)))


# -- Positions --------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 5, 8), (4, 9, 64), (3, 33, 128)], ids=shape_id)
def test_positions_restart_in_every_sequence(flint, shape):
    # With the word table zero, x is exactly wpe repeated for each sequence.
    sequences, length, width = shape
    ids, wte, wpe = make_inputs(sequences, length, 97, width)
    wte.zero_()

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    expected = wpe[:length].repeat(sequences, 1)
    assert torch.equal(bits(x), bits(expected))


def test_with_the_position_table_zero_x_is_the_words(flint):
    sequences, length, width = 4, 9, 64
    ids, wte, wpe = make_inputs(sequences, length, 97, width)
    wpe.zero_()

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    assert torch.equal(bits(x), bits(wte[ids.long()]))


# -- Non-finite and signed values -------------------------------------------


def special_problem():
    """3 sequences of 5 positions over a vocabulary of 10: token 3 appears at
    every position (in sequence 0), token 4 at position 1 only."""
    sequences, length, vocab, width = 3, 5, 10, 16
    ids, wte, wpe = make_inputs(sequences, length, vocab, width)
    ids = ids.view(sequences, length)
    ids[0] = 3
    ids[1] = torch.tensor([0, 4, 0, 0, 0], dtype=torch.int32, device="cuda")
    ids[2] = torch.tensor([9, 8, 7, 6, 5], dtype=torch.int32, device="cuda")
    return sequences, length, ids.view(-1), wte, wpe


def test_nan_in_the_word_table_reaches_only_the_rows_that_read_it(flint):
    sequences, length, ids, wte, wpe = special_problem()
    wte[3, 5] = math.nan

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    expected = reference_embedding(ids, wte, wpe, sequences, length)
    assert_same(x, expected)
    nan = torch.isnan(x)
    # Row 0 to 4 (token 3), column 5, and nothing else.
    assert nan.sum().item() == length
    assert nan[:length, 5].all()


def test_nan_in_the_position_table_reaches_only_its_position(flint):
    sequences, length, ids, wte, wpe = special_problem()
    wpe[2, 7] = math.nan

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    assert_same(x, reference_embedding(ids, wte, wpe, sequences, length))
    nan = torch.isnan(x).view(sequences, length, -1)
    assert nan.sum().item() == sequences
    assert nan[:, 2, 7].all()


def test_inf_and_negative_inf_make_nan_and_alone_stay_inf(flint):
    sequences, length, ids, wte, wpe = special_problem()
    wte[3, 3] = math.inf
    wte[4, 3] = math.inf
    wpe[1, 3] = -math.inf

    x = flint_embedding(flint, ids, wte, wpe, sequences, length).view(sequences, length, -1)

    assert_same(x.view(-1, x.shape[-1]), reference_embedding(ids, wte, wpe, sequences, length))
    # Token 3 at position 1 (sequence 0) and token 4 at position 1 (sequence 1)
    # add inf to -inf; token 3 elsewhere stays +inf; other tokens at position 1
    # stay -inf.
    assert torch.isnan(x[0, 1, 3]) and torch.isnan(x[1, 1, 3])
    assert torch.isposinf(x[0, 0, 3]) and torch.isposinf(x[0, 4, 3])
    assert torch.isneginf(x[2, 1, 3])


def test_negative_zero_plus_negative_zero_stays_negative_zero(flint):
    sequences, length, ids, wte, wpe = special_problem()
    wte[4] = -0.0
    wpe[1] = -0.0
    wte[5] = -0.0
    wte[6] = 0.0

    x = flint_embedding(flint, ids, wte, wpe, sequences, length)

    assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, sequences, length)))
    # Token 4 at position 1: -0 + -0 = -0, every column.
    assert (bits(x.view(sequences, length, -1)[1, 1]) == -(2**31)).all()
    # -0 + a +0 is +0, in round-to-nearest.
    wpe[1] = 0.0
    y = flint_embedding(flint, ids, wte, wpe, sequences, length)
    assert (bits(y.view(sequences, length, -1)[1, 1]) == 0).all()


# -- Determinism ------------------------------------------------------------


@pytest.mark.parametrize("shape", [GPT2_SHAPES[0], (7, 33, 24)], ids=shape_id)
def test_run_twice_is_bitwise_identical(flint, shape):
    sequences, length, width = shape
    ids, wte, wpe = make_inputs(sequences, length, GPT2_VOCAB, width)
    # Different fills to start from (all NaN, then a pattern that is
    # 3.4e38), so a store that is skipped, or one that lands in only one run,
    # shows as a difference or a leftover.
    first = flint_embedding(flint, ids, wte, wpe, sequences, length, fill=0xFFFF)
    second = flint_embedding(flint, ids, wte, wpe, sequences, length, fill=0x7F7F)

    assert not torch.isnan(first).any()
    assert not torch.isnan(second).any()
    assert torch.equal(bits(first), bits(second))
    assert torch.equal(bits(first), bits(reference_embedding(ids, wte, wpe, sequences, length)))


# -- Streams ----------------------------------------------------------------


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit-stream", "current-stream"])
def test_embedding_runs_on_the_given_stream(flint, explicit):
    sequences, length, vocab, width = 4, 8, 16, 64
    rows = sequences * length
    tokens = torch.zeros(rows, dtype=torch.int16, device="cuda")
    wte = torch.zeros((vocab, width), dtype=torch.float32, device="cuda")
    wpe = torch.zeros((length, width), dtype=torch.float32, device="cuda")
    x = torch.zeros((rows, width), dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fills finish only after the sleep, so x is 1.75 only if the kernel is
    # queued on the same stream behind them. With the current stream the call
    # has no stream argument, and torch's current stream is the side one.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        wte.fill_(1.5)
        wpe.fill_(0.25)
        flint.embedding_forward(
            x, tokens, wte, wpe, sequences, length, vocab, width, stream=stream if explicit else None
        )
    stream.synchronize()

    assert (x == 1.75).all()


# -- Argument errors --------------------------------------------------------

NAMES = ("x", "tokens", "wte", "wpe")


def small_problem(sequences=2, length=4, vocab=16, width=16):
    """Valid tensors by name, in buffers with room to spare."""
    rows = sequences * length
    return {
        "x": torch.zeros(rows * width + 64, dtype=torch.float32, device="cuda"),
        "tokens": torch.zeros(rows + 64, dtype=torch.int16, device="cuda"),
        "wte": torch.ones(vocab * width + 64, dtype=torch.float32, device="cuda"),
        "wpe": torch.ones(length * width + 64, dtype=torch.float32, device="cuda"),
    }


def call(flint, buffers, sequences=2, length=4, vocab=16, width=16):
    flint.embedding_forward(*(buffers[name] for name in NAMES), sequences, length, vocab, width)


def expect_invalid_argument(flint, reason, buffers, **sizes):
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, buffers, **sizes)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_embedding_forward", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    expect_invalid_argument(flint, f"{name} is null", {**small_problem(), name: None})


@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name):
    # Tokens are 2 bytes wide, the rest 4: each goes the width of one element
    # off, which is still misaligned for 16 bytes.
    addresses = {name: t.data_ptr() for name, t in small_problem().items()}
    addresses[name] += 2 if name == "tokens" else 4
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
    # in the order x, tokens, wte, wpe.
    addresses = {name: t.data_ptr() for name, t in small_problem().items()}
    expect_invalid_argument(
        flint, "x is not 16-byte aligned", {**addresses, "x": addresses["x"] + 4, "tokens": None}
    )
    expect_invalid_argument(
        flint, "tokens is null", {**addresses, "tokens": None, "wte": addresses["wte"] + 4}
    )
    expect_invalid_argument(
        flint, "wte is not 16-byte aligned", {**addresses, "wte": addresses["wte"] + 4, "wpe": None}
    )
    expect_invalid_argument(
        flint, "wpe runs past the end", {**addresses, "wpe": 2**64 - 16}
    )


@pytest.mark.parametrize("width", [0, 1, 4, 12, 2**24 + 4, 2**24 + 8, 2**64 - 1], ids=str)
def test_width_must_be_a_multiple_of_8_up_to_2_to_the_24(flint, width):
    expect_invalid_argument(
        flint,
        f"width ({width}) must be a multiple of 8 from 8 to 16777216",
        small_problem(),
        width=width,
    )


@pytest.mark.parametrize("vocab", [0, 65537, 2**64 - 1], ids=str)
def test_vocab_must_be_from_1_to_65536(flint, vocab):
    expect_invalid_argument(
        flint, f"vocab ({vocab}) must be from 1 to 65536", small_problem(), vocab=vocab
    )


@pytest.mark.parametrize(
    "sequences, length",
    [
        (MAX_ROWS + 1, 1),
        (1, MAX_ROWS + 1),
        (MAX_ROWS // 2 + 1, 2),
        (2**32, 2**32),  # the product wraps to 0
        (2**40, 2**40),
        (2**63, 2),
        (2**64 - 1, 2**64 - 1),
        (2**64 - 1, 1),
    ],
    ids=lambda v: str(v),
)
def test_too_many_rows(flint, sequences, length):
    # Including the products that wrap around 2^64, which must not pass as
    # small or as empty.
    expect_invalid_argument(
        flint,
        f"sequences ({sequences}) x length ({length}) can be up to 17179869176 rows",
        small_problem(),
        sequences=sequences,
        length=length,
    )


@pytest.mark.parametrize("sequences, length", [(MAX_ROWS, 1), (1, MAX_ROWS), (MAX_ROWS // 2, 2)])
def test_the_most_rows_pass_the_size_checks(flint, sequences, length):
    # The limit itself isn't an error of size: the call gets on to the pointers
    # (a null x), even at the widest width, where rows * width * 4 is 2^60.
    for width in (16, 2**24):
        expect_invalid_argument(
            flint,
            "x is null",
            {**small_problem(), "x": None},
            sequences=sequences,
            length=length,
            width=width,
        )


def test_size_limits_are_checked_in_order(flint):
    # width, then vocab, then rows; each before any pointer, null ones included.
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(
        flint, "width (0) must be", nothing, sequences=2**40, length=2**40, vocab=0, width=0
    )
    expect_invalid_argument(
        flint, "vocab (0) must be from 1 to", nothing, sequences=2**40, length=2**40, vocab=0
    )
    expect_invalid_argument(flint, "can be up to 17179869176 rows", nothing, sequences=2**40, length=2**40)


# The bytes each buffer covers in the overlap tests, for 2 sequences of 4
# positions over a vocabulary of 16, 16 wide.
SIZES = {"x": 512, "tokens": 16, "wte": 1024, "wpe": 256}
SPACING = 4096


def overlap_addresses(arena, second, shift):
    """Every buffer in the arena, SPACING bytes apart (far more than any size),
    except that `second` starts `shift` bytes after x."""
    base = arena.data_ptr() + SPACING
    addresses = {name: base + SPACING * i for i, name in enumerate(NAMES)}
    addresses[second] = addresses["x"] + shift
    return addresses


# How far the input starts after x: at the same address, inside x, 16 bytes
# before x's end, and (negative) with the input ending 16 bytes into x.
SHIFTS = {
    "same-start": lambda second: 0,
    "input-inside": lambda second: 16,
    "input-at-end": lambda second: SIZES["x"] - 16,
    "x-inside": lambda second: 16 - SIZES[second],
}

OVERLAPS = [
    pytest.param(second, shift(second), id=f"x-{second}-{label}")
    for second in NAMES[1:]
    for label, shift in SHIFTS.items()
]


@pytest.mark.parametrize("second, shift", OVERLAPS)
def test_x_may_not_overlap_an_input(flint, second, shift):
    # The buffers are real memory in an arena, so a call that wrongly went
    # ahead would do no harm.
    arena = torch.zeros(8 * SPACING, dtype=torch.uint8, device="cuda")
    expect_invalid_argument(
        flint, f"x and {second} overlap", overlap_addresses(arena, second, shift)
    )


@pytest.mark.parametrize("order", [("x", "tokens", "wte", "wpe"), ("wpe", "wte", "tokens", "x"), ("tokens", "x", "wpe", "wte")], ids="-".join)
def test_buffers_next_to_each_other_are_not_overlapping(flint, order):
    # The four buffers side by side with no gap in one arena, with a pattern
    # around them that must stay intact.
    sequences, length, vocab, width = 2, 4, 16, 16
    offsets = {}
    at = 16
    for name in order:
        offsets[name] = at
        at += SIZES[name]
    arena = torch.full((at + 16,), 0x5A, dtype=torch.uint8, device="cuda")

    def view(name, dtype):
        return arena[offsets[name] : offsets[name] + SIZES[name]].view(dtype)

    ids, wte, wpe = make_inputs(sequences, length, vocab, width)
    view("tokens", torch.int16).copy_(token_tensor(ids))
    view("wte", torch.float32).copy_(wte.view(-1))
    view("wpe", torch.float32).copy_(wpe.view(-1))
    x = view("x", torch.float32)

    flint.embedding_forward(
        x, view("tokens", torch.int16), view("wte", torch.float32), view("wpe", torch.float32),
        sequences, length, vocab, width,
    )
    torch.cuda.synchronize()

    expected = reference_embedding(ids, wte, wpe, sequences, length)
    assert torch.equal(bits(x.view(-1, width)), bits(expected))
    used = torch.zeros(arena.numel(), dtype=torch.bool, device="cuda")
    used[16:at] = True
    assert (arena[~used] == 0x5A).all()


@pytest.mark.parametrize("sharing", ["wpe-is-wte", "tokens-in-wte"])
def test_inputs_may_overlap_each_other(flint, sharing):
    # Inputs are only read, so they may share memory: wpe is the first rows of
    # wte, or the tokens are the first bytes of wte.
    sequences, length, vocab, width = 2, 4, 16, 16
    ids, wte, wpe = make_inputs(sequences, length, vocab, width)
    x = torch.zeros(sequences * length * width, dtype=torch.float32, device="cuda")
    if sharing == "wpe-is-wte":
        tokens, wpe = token_tensor(ids), wte
    else:
        arena = torch.zeros(vocab * width * 4, dtype=torch.uint8, device="cuda")
        wte = arena.view(torch.float32).view(vocab, width)
        wte.copy_(torch.randn((vocab, width), generator=generator(1), device="cuda"))
        tokens = arena[:16].view(torch.int16)
        tokens.copy_(token_tensor(ids))
        wpe = wte
    expected = reference_embedding(ids, wte.clone(), wpe.clone(), sequences, length)

    flint.embedding_forward(x, tokens, wte, wpe, sequences, length, vocab, width)
    torch.cuda.synchronize()

    assert torch.equal(bits(x.view(-1, width)), bits(expected))


@pytest.mark.parametrize(
    "sequences, length",
    [(0, 4), (2, 0), (0, 0), (0, 2**64 - 1), (2**64 - 1, 0), (0, 2**40)],
    ids=lambda v: str(v),
)
def test_empty_problems_do_nothing(flint, sequences, length):
    # Not even the null pointers are looked at, and a length too large for
    # sequences * length is no error when there are no sequences.
    buffers = small_problem()
    buffers["x"].fill_(3)

    flint.embedding_forward(None, None, None, None, sequences, length, 16, 16)
    call(flint, buffers, sequences=sequences, length=length)
    torch.cuda.synchronize()

    assert (buffers["x"] == 3).all()


def test_empty_problems_still_check_width_and_vocab(flint):
    nothing = {name: None for name in NAMES}
    for width in (0, 4, 2**24 + 8):
        expect_invalid_argument(flint, f"width ({width}) must be", nothing, sequences=0, width=width)
        expect_invalid_argument(flint, f"width ({width}) must be", nothing, length=0, width=width)
    for vocab in (0, 65537):
        expect_invalid_argument(flint, f"vocab ({vocab}) must be", nothing, sequences=0, vocab=vocab)
        expect_invalid_argument(flint, f"vocab ({vocab}) must be", nothing, length=0, vocab=vocab)


def test_the_limits_themselves_are_accepted(flint):
    # The widest row and the biggest vocabulary: the call gets on to the pointers.
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(flint, "x is null", nothing, width=2**24, vocab=65536)
    # And they run: 65,536 words of 8 and a width of 2^24.
    ids = torch.tensor([65535, 0], dtype=torch.int32, device="cuda")
    wte = torch.randn((65536, 8), generator=generator(0), device="cuda")
    wpe = torch.randn((2, 8), generator=generator(1), device="cuda")
    x = flint_embedding(flint, ids, wte, wpe, 1, 2)
    assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, 1, 2)))
    if not SANITIZER:
        ids = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
        wte = torch.randn((2, 2**24), generator=generator(2), device="cuda")
        wpe = torch.randn((2, 2**24), generator=generator(3), device="cuda")
        x = flint_embedding(flint, ids, wte, wpe, 1, 2)
        assert torch.equal(bits(x), bits(reference_embedding(ids, wte, wpe, 1, 2)))


def test_a_valid_call_works_after_errors(flint):
    buffers = small_problem()
    buffers["tokens"][:8] = torch.arange(8, dtype=torch.int16, device="cuda")
    buffers["wte"][: 16 * 16] = torch.arange(16 * 16, device="cuda") / 8
    buffers["wpe"][: 4 * 16] = 0.5
    for width in (0, 12):
        with pytest.raises(flint_lib.FlintError):
            call(flint, buffers, width=width)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "x": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, buffers, sequences=2**40, length=2**40)

    call(flint, buffers)
    torch.cuda.synchronize()

    # Row r is word r, (16 r + c) / 8 in column c, plus 0.5.
    expected = (torch.arange(8 * 16, device="cuda") / 8 + 0.5).view(8, 16)
    assert torch.equal(buffers["x"][: 8 * 16].view(8, 16), expected)
    assert (buffers["x"][8 * 16 :] == 0).all()
