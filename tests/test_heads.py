"""flint_heads_split and flint_heads_merge: attention's head layouts, for FP16
values moved as bits. The "rows" layout is rows x row_width (rows = batches *
length), the "heads" layout batches x heads x length x head_width, both
row-major and contiguous. Split copies the columns [first, first + heads *
head_width) of each row into the heads layout,
y[b, h, t, c] = x[b * length + t, first + h * head_width + c], and merge is its
reverse: it writes only that block of columns of each row of y and leaves the
other columns alone, so three merges (Q, K and V of a fused qkv buffer) fill
one row buffer.

The kernels only move data, so every comparison is bitwise, with no tolerance
and no gate. The data is random 16-bit patterns viewed as FP16 (so NaN
payloads, -0, inf and subnormals all occur) and compared as int16, against
PyTorch's view-and-transpose, which is exactly what Attention.forward in
scripts/gpt2_torch.py does. The tests cover GPT-2's shapes (B = 16 and B = 1;
Q, K and V of qkv's 2304 columns, and dO's 768), the wider models' head counts,
small odd shapes with columns on both sides of the block and head widths whose
chunk count (head_width / 8) doesn't divide a warp's 32 lanes, split followed
by merge, merge leaving the other columns alone, determinism, guard bands,
stream ordering and argument errors, including sizes whose products would wrap
around 2^64."""

import pytest
import torch

import flint_lib
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator

MAX_ROWS = 17_179_869_176
MAX_ROW_WIDTH = 2**24

# The output of a guarded call sits `offset` blocks past the start of a buffer
# that is filled with a pattern, which must be intact around it afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A

FUNCTIONS = ("heads_split", "heads_merge")


def shape_id(shape):
    return "x".join(map(str, shape))


# A shape is (batches, length, heads, head_width, row_width, first).


def rows_numel(shape):
    batches, length, _, _, row_width, _ = shape
    return batches * length * row_width


def heads_numel(shape):
    batches, length, heads, head_width, _, _ = shape
    return batches * heads * length * head_width


def random_bits(n, seed=0):
    """n random 16-bit patterns as int16, which are FP16 values of every kind."""
    return torch.randint(-(2**15), 2**15, (n,), dtype=torch.int16, generator=generator(seed), device="cuda")


# -- The references and running Flint ---------------------------------------


def split_reference(x, shape):
    """The heads layout of x's block of columns, as PyTorch views and
    transposes it: batches x heads x length x head_width."""
    batches, length, heads, head_width, row_width, first = shape
    block = x.view(batches, length, row_width)[:, :, first : first + heads * head_width]
    return block.view(batches, length, heads, head_width).transpose(1, 2).contiguous()


def merge_reference(h, shape, rows):
    """rows (batches * length x row_width) with the block of columns replaced by
    the heads layout h merged back."""
    batches, length, heads, head_width, row_width, first = shape
    y = rows.clone().view(batches, length, row_width)
    y[:, :, first : first + heads * head_width] = h.view(batches, heads, length, head_width).transpose(
        1, 2
    ).reshape(batches, length, heads * head_width)
    return y.view(batches * length, row_width)


def flint_split(flint, x, shape, offset=1, fill=SENTINEL, stream=None):
    """Runs flint_heads_split with y inside a buffer pre-filled with `fill`, 8 *
    offset values in, which keeps it 16-byte aligned. Checks that the buffer
    around y and x are untouched, and returns y as batches x heads x length x
    head_width."""
    batches, length, heads, head_width, _, _ = shape
    count = heads_numel(shape)
    at = 8 * offset
    buffer = torch.full(
        (at + count + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda"
    )
    y = buffer[at : at + count]
    x_before = x.clone()

    flint.heads_split(y, x, *shape, stream=stream)
    torch.cuda.synchronize()

    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert (buffer[:at] == guard).all(), "flint_heads_split wrote before y"
    assert (buffer[at + count :] == guard).all(), "flint_heads_split wrote past y"
    assert torch.equal(x, x_before), "x changed"
    return y.view(batches, heads, length, head_width)


def flint_merge(flint, h, shape, offset=1, fill=SENTINEL, stream=None):
    """Runs flint_heads_merge on the heads layout h with y (the rows layout)
    inside a buffer pre-filled with `fill`, as flint_split does. Returns y as
    rows x row_width; the columns outside the block hold the fill."""
    batches, length, _, _, row_width, _ = shape
    count = rows_numel(shape)
    at = 8 * offset
    buffer = torch.full(
        (at + count + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda"
    )
    y = buffer[at : at + count]
    h_before = h.clone()

    flint.heads_merge(y, h, *shape, stream=stream)
    torch.cuda.synchronize()

    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert (buffer[:at] == guard).all(), "flint_heads_merge wrote before y"
    assert (buffer[at + count :] == guard).all(), "flint_heads_merge wrote past y"
    assert torch.equal(h, h_before), "x changed"
    return y.view(batches * length, row_width)


def filled(shape, fill=SENTINEL):
    """The rows layout of the shape with every value the fill."""
    return torch.full(
        (shape[0] * shape[1], shape[4]), as_int16(fill), dtype=torch.int16, device="cuda"
    )


# -- Shapes -----------------------------------------------------------------

# GPT-2 small's training batch is 16 x 1024 (and 1 sequence, 1024 long); the
# sanitizer gets shorter sequences.
BATCHES = [(2, 64), (1, 64)] if SANITIZER else [(16, 1024), (1, 1024)]
# The wider models' widths run at a few hundred rows that don't fill a last
# block of 8.
MID_BATCHES, MID_LENGTH = (2, 37) if SANITIZER else (3, 199)

GPT2_SHAPES = [
    (batches, length, 12, 64, row_width, first)
    for batches, length in BATCHES
    # qkv's three parts, then dO.
    for row_width, first in [(2304, 0), (2304, 768), (2304, 1536), (768, 0)]
]

# 16, 20 and 25 heads of 64 (GPT-2 medium, large and XL): a head's row alone
# (width 1024, 1280, 1600) and the second part (K) of qkv's three times wider.
WIDER_SHAPES = [
    (MID_BATCHES, MID_LENGTH, heads, 64, row_width, first)
    for heads in (16, 20, 25)
    for row_width, first in [(64 * heads, 0), (3 * 64 * heads, 64 * heads)]
]

# Small and odd: single rows, lengths below and above a block of 8 rows (rows
# of 51 and 297 leave a partial block, and 56 fills its last), head counts of 1
# to 3, head widths of 1 to 8 chunks of 16 bytes, and a first and a row_width
# that leave columns on both sides of the block, or on one side, or none.
def odd_shapes():
    cases = [
        (bt, hh)
        for bt in [(1, 1), (3, 17), (7, 8), (9, 33)]
        for hh in [(1, 8), (2, 16), (3, 64), (1, 64), (3, 8), (2, 64)]
    ]
    shapes = []
    for i, ((batches, length), (heads, head_width)) in enumerate(cases):
        first = [8, 0, 24, 16][i % 4]
        after = [16, 8, 0, 0][(i // 2) % 4]
        shapes.append((batches, length, heads, head_width, first + heads * head_width + after, first))
    return shapes


ODD_SHAPES = odd_shapes()

# Head widths whose chunk count (head_width / 8) is not a power of two, or is
# 32 (a warp's lanes), or is above 32, so a lane's chunks cross heads at
# different places and a step can skip a whole head or none.
CHUNK_SHAPES = [
    (batches, length, heads, head_width, 8 + heads * head_width + 8, 8)
    for batches, length in [(2, 5), (1, 9)]
    for heads, head_width in [(3, 24), (2, 40), (4, 72), (2, 136), (3, 256), (2, 264), (2, 520)]
]


def all_shapes():
    return GPT2_SHAPES + WIDER_SHAPES + ODD_SHAPES + CHUNK_SHAPES


SHAPE_CASES = [
    pytest.param(shape, 1 + i % 3, id=shape_id(shape)) for i, shape in enumerate(all_shapes())
]


# -- Bitwise against PyTorch ------------------------------------------------


@pytest.mark.parametrize("shape, offset", SHAPE_CASES)
def test_split_matches_pytorch_bitwise_and_stays_inside_y(flint, shape, offset):
    # flint_split also checks that the guard regions and x are intact.
    x = random_bits(rows_numel(shape))

    y = flint_split(flint, x, shape, offset)

    assert torch.equal(y, split_reference(x, shape))


@pytest.mark.parametrize("shape, offset", SHAPE_CASES)
def test_merge_matches_pytorch_bitwise_and_leaves_other_columns_alone(flint, shape, offset):
    # The reference puts the block into a buffer that starts as the fill, so
    # the columns outside the block must still hold it.
    h = random_bits(heads_numel(shape), seed=1)
    _, _, heads, head_width, _, first = shape

    y = flint_merge(flint, h, shape, offset)

    assert torch.equal(y, merge_reference(h, shape, filled(shape)))
    sentinel = as_int16(SENTINEL)
    assert (y[:, :first] == sentinel).all()
    assert (y[:, first + heads * head_width :] == sentinel).all()


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 15, 16, 17, 33])
def test_every_row_is_moved_and_nothing_else(flint, rows):
    # One sequence of `rows`, then `rows` sequences of 1, in a layout with
    # columns on both sides. The buffers start as the fill, so a row (a partial
    # block's last one, say) that is skipped shows; the helpers check the guards.
    for batches, length in [(1, rows), (rows, 1)]:
        shape = (batches, length, 2, 24, 8 + 48 + 8, 8)
        x = random_bits(rows_numel(shape))
        assert torch.equal(flint_split(flint, x, shape, fill=0xFFFF), split_reference(x, shape))
        h = random_bits(heads_numel(shape), seed=1)
        assert torch.equal(
            flint_merge(flint, h, shape, fill=0xFFFF), merge_reference(h, shape, filled(shape, 0xFFFF))
        )


# Patterns of FP16 values that a conversion through a float would change: -0,
# +-inf, NaNs with payloads (quiet and signalling), a subnormal, the smallest
# normal and the largest finite.
SPECIAL = [0x8000, 0x7C00, 0xFC00, 0x7E01, 0x7C01, 0xFFFF, 0x0001, 0x8001, 0x03FF, 0x0400, 0x0000, 0x7BFF]


def special_bits(n):
    pattern = torch.tensor([as_int16(v) for v in SPECIAL], dtype=torch.int16, device="cuda")
    return pattern.repeat(n // len(pattern) + 1)[:n].clone()


@pytest.mark.parametrize("shape", [(3, 17, 3, 16, 8 + 48 + 8, 8), (2, 9, 4, 8, 32, 0)], ids=shape_id)
def test_special_values_move_unchanged(flint, shape):
    x = special_bits(rows_numel(shape))
    h = special_bits(heads_numel(shape))

    assert torch.equal(flint_split(flint, x, shape), split_reference(x, shape))
    assert torch.equal(flint_merge(flint, h, shape), merge_reference(h, shape, filled(shape)))


# -- Round trips ------------------------------------------------------------

# Shapes whose block is the whole row.
FULL_ROW_SHAPES = [
    (3, 17, 4, 16, 64, 0),
    (7, 8, 3, 64, 192, 0),
    (2, 33, 1, 8, 8, 0),
    (5, 5, 3, 24, 72, 0),
    (2, 9, 2, 264, 528, 0),
]


@pytest.mark.parametrize("shape", FULL_ROW_SHAPES, ids=shape_id)
def test_split_then_merge_returns_the_input_exactly(flint, shape):
    x = random_bits(rows_numel(shape))

    y = flint_merge(flint, flint_split(flint, x, shape).reshape(-1), shape)

    assert torch.equal(y.reshape(-1), x)


@pytest.mark.parametrize("shape", FULL_ROW_SHAPES, ids=shape_id)
def test_merge_then_split_returns_the_input_exactly(flint, shape):
    h = random_bits(heads_numel(shape))

    y = flint_split(flint, flint_merge(flint, h, shape).reshape(-1), shape)

    assert torch.equal(y.reshape(-1), h)


@pytest.mark.parametrize("batches, length", [(3, 41), (2, 64)] if SANITIZER else [(3, 41), (16, 1024)])
def test_three_merges_fill_the_qkv_buffer(flint, batches, length):
    # Q, K and V split from a qkv buffer, then merged one after another into a
    # buffer that starts as the fill, give the original bit for bit; each merge
    # leaves the others' columns alone.
    row_width = 2304
    x = random_bits(batches * length * row_width)
    y = torch.full_like(x, as_int16(SENTINEL))
    parts = []
    for first in (0, 768, 1536):
        shape = (batches, length, 12, 64, row_width, first)
        parts.append((shape, flint_split(flint, x, shape).reshape(-1)))

    for step, (shape, h) in enumerate(parts):
        flint.heads_merge(y, h, *shape)
        torch.cuda.synchronize()
        done = 768 * (step + 1)
        assert torch.equal(y.view(-1, row_width)[:, :done], x.view(-1, row_width)[:, :done])
        assert (y.view(-1, row_width)[:, done:] == as_int16(SENTINEL)).all()

    assert torch.equal(y, x)


# -- Determinism ------------------------------------------------------------


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("shape", [GPT2_SHAPES[1], ODD_SHAPES[5], CHUNK_SHAPES[3]], ids=shape_id)
def test_run_twice_is_bitwise_identical(flint, function, shape):
    # Different fills to start from, so a store that is skipped, or one that
    # lands in only one run, shows as a difference.
    if function == "heads_split":
        x = random_bits(rows_numel(shape))
        first = flint_split(flint, x, shape, fill=0xFFFF)
        second = flint_split(flint, x, shape, fill=0x7B7B)
        assert torch.equal(first, second)
        assert torch.equal(first, split_reference(x, shape))
    else:
        h = random_bits(heads_numel(shape))
        first = flint_merge(flint, h, shape, fill=0xFFFF)
        second = flint_merge(flint, h, shape, fill=0x7B7B)
        # The block is the same in both runs, and the columns around it hold
        # each run's own fill.
        block = slice(shape[5], shape[5] + shape[2] * shape[3])
        assert torch.equal(first[:, block], second[:, block])
        assert torch.equal(first, merge_reference(h, shape, filled(shape, 0xFFFF)))
        assert torch.equal(second, merge_reference(h, shape, filled(shape, 0x7B7B)))


# -- Streams ----------------------------------------------------------------


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit-stream", "current-stream"])
@pytest.mark.parametrize("function", FUNCTIONS)
def test_heads_run_on_the_given_stream(flint, function, explicit):
    shape = (2, 16, 2, 16, 64, 16)
    source = torch.zeros(
        rows_numel(shape) if function == "heads_split" else heads_numel(shape),
        dtype=torch.int16,
        device="cuda",
    )
    target = torch.zeros(
        heads_numel(shape) if function == "heads_split" else rows_numel(shape),
        dtype=torch.int16,
        device="cuda",
    )
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fill finishes only after the sleep, so the target holds 1234 only if
    # the kernel is queued on the same stream behind it. With the current
    # stream the call has no stream argument, and torch's current stream is the
    # side one.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        source.fill_(1234)
        getattr(flint, function)(target, source, *shape, stream=stream if explicit else None)
    stream.synchronize()

    if function == "heads_split":
        assert (target == 1234).all()
    else:
        rows = target.view(-1, shape[4])
        block = slice(shape[5], shape[5] + shape[2] * shape[3])
        assert (rows[:, block] == 1234).all()
        assert (rows[:, : block.start] == 0).all() and (rows[:, block.stop :] == 0).all()


# -- Argument errors --------------------------------------------------------

NAMES = ("y", "x")
SMALL = dict(batches=2, length=4, heads=2, head_width=8, row_width=32, first=8)
# The rows layout holds 8 x 32 values (512 bytes), the heads layout 2 x 2 x 4 x
# 8 (256 bytes).
ROWS_BYTES, HEADS_BYTES = 512, 256


def small_problem(function):
    """Valid tensors by name, in buffers with room to spare."""
    rows = torch.zeros(8 * 32 + 64, dtype=torch.int16, device="cuda")
    heads = torch.zeros(2 * 2 * 4 * 8 + 64, dtype=torch.int16, device="cuda")
    return {"y": heads, "x": rows} if function == "heads_split" else {"y": rows, "x": heads}


def call(flint, function, buffers, **sizes):
    sizes = {**SMALL, **sizes}
    getattr(flint, function)(
        buffers["y"],
        buffers["x"],
        sizes["batches"],
        sizes["length"],
        sizes["heads"],
        sizes["head_width"],
        sizes["row_width"],
        sizes["first"],
    )


def expect_invalid_argument(flint, function, reason, buffers, **sizes):
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, function, buffers, **sizes)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in (f"flint_{function}", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


def nothing():
    return {name: None for name in NAMES}


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, function, name):
    expect_invalid_argument(
        flint, function, f"{name} is null", {**small_problem(function), name: None}
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, function, name):
    # Both buffers are FP16, so 2 bytes off is as far from 16-byte alignment as
    # an element goes.
    addresses = {n: t.data_ptr() for n, t in small_problem(function).items()}
    addresses[name] += 2
    expect_invalid_argument(flint, function, f"{name} is not 16-byte aligned", addresses)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("name", NAMES)
def test_a_buffer_may_not_run_past_the_address_space(flint, function, name):
    # The check runs before any memory is touched, so a made-up address will do.
    expect_invalid_argument(
        flint,
        function,
        f"{name} runs past the end of the address space",
        {**small_problem(function), name: 2**64 - 16},
    )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_pointers_are_checked_in_order(flint, function):
    # Per buffer: null, then alignment, then the address space; y before x.
    addresses = {n: t.data_ptr() for n, t in small_problem(function).items()}
    expect_invalid_argument(
        flint, function, "y is not 16-byte aligned", {**addresses, "y": addresses["y"] + 2, "x": None}
    )
    expect_invalid_argument(
        flint, function, "y is null", {**addresses, "y": None, "x": addresses["x"] + 2}
    )
    expect_invalid_argument(
        flint, function, "x is not 16-byte aligned", {**addresses, "x": addresses["x"] + 2}
    )
    expect_invalid_argument(
        flint, function, "y runs past the end", {"y": 2**64 - 16, "x": None}
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "row_width", [0, 4, 12, 36, MAX_ROW_WIDTH + 8, 2**25, 2**64 - 8, 2**64 - 1], ids=str
)
def test_row_width_must_be_a_multiple_of_8_up_to_2_to_the_24(flint, function, row_width):
    expect_invalid_argument(
        flint,
        function,
        f"row_width ({row_width}) must be a multiple of 8 from 8 to 16777216",
        small_problem(function),
        row_width=row_width,
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("head_width", [0, 4, 12, 36, 2**64 - 1], ids=str)
def test_head_width_must_be_a_positive_multiple_of_8(flint, function, head_width):
    expect_invalid_argument(
        flint,
        function,
        f"head_width ({head_width}) must be a positive multiple of 8",
        small_problem(function),
        head_width=head_width,
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("first", [4, 12, 25, 2**64 - 1], ids=str)
def test_first_must_be_a_multiple_of_8(flint, function, first):
    expect_invalid_argument(
        flint,
        function,
        f"first ({first}) must be a multiple of 8",
        small_problem(function),
        first=first,
    )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_heads_must_be_positive(flint, function):
    expect_invalid_argument(
        flint, function, "heads must be positive", small_problem(function), heads=0
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "sizes",
    [
        dict(heads=4),  # 32 columns from column 8
        dict(heads=3, first=16),  # 24 columns from column 16
        dict(heads=2, head_width=16),  # 32 columns from column 8
        dict(heads=1, first=32),  # nothing left
        dict(heads=1, head_width=40),  # a head wider than the row
        dict(heads=1, head_width=32, first=8),
        dict(heads=5, first=0),  # 40 columns
        # Products that wrap around 2^64: 2^61 * 8 is 0, and 2^63 * 2 is 0.
        dict(heads=2**61),
        dict(heads=2**61 + 1),
        dict(heads=2**64 - 1),
        dict(heads=2, head_width=2**63),
        dict(heads=1, head_width=2**64 - 8),
        dict(heads=2**32, head_width=2**32),
        # first and the block, added, would wrap.
        dict(first=2**64 - 8),
        dict(heads=1, first=2**64 - 8),
    ],
    ids=lambda sizes: "-".join(f"{k}={v}" for k, v in sizes.items()),
)
def test_the_block_of_columns_must_lie_within_the_row(flint, function, sizes):
    heads, head_width, first = (sizes.get(k, SMALL[k]) for k in ("heads", "head_width", "first"))
    expect_invalid_argument(
        flint,
        function,
        f"{heads} heads of {head_width} columns from column {first} run past row_width (32)",
        small_problem(function),
        **sizes,
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "batches, length",
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
def test_too_many_rows(flint, function, batches, length):
    # Including the products that wrap around 2^64, which must not pass as
    # small or as empty.
    expect_invalid_argument(
        flint,
        function,
        f"batches ({batches}) x length ({length}) can be up to 17179869176 rows",
        small_problem(function),
        batches=batches,
        length=length,
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("batches, length", [(MAX_ROWS, 1), (1, MAX_ROWS), (MAX_ROWS // 2, 2)])
def test_the_most_rows_pass_the_size_checks(flint, function, batches, length):
    # The limit itself isn't an error of size: the call gets on to the pointers
    # (a null y), even at the widest rows, where rows * row_width * 2 is 2^59.
    for sizes in (SMALL, dict(heads=1, head_width=8, row_width=MAX_ROW_WIDTH, first=0)):
        expect_invalid_argument(
            flint,
            function,
            "y is null",
            {**small_problem(function), "y": None},
            **{**sizes, "batches": batches, "length": length},
        )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_the_widest_row_and_the_most_heads_pass_the_size_checks(flint, function):
    # All of a 2^24 row as 2^21 heads of 8, and as one head.
    for sizes in (
        dict(heads=2**21, head_width=8, row_width=MAX_ROW_WIDTH, first=0),
        dict(heads=1, head_width=MAX_ROW_WIDTH, row_width=MAX_ROW_WIDTH, first=0),
        dict(heads=1, head_width=8, row_width=MAX_ROW_WIDTH, first=MAX_ROW_WIDTH - 8),
    ):
        expect_invalid_argument(
            flint, function, "y is null", {**small_problem(function), "y": None}, **sizes
        )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_size_limits_are_checked_in_order(flint, function):
    # row_width, head_width, first, heads, the block, then the rows; each before
    # any pointer, null ones included.
    bad_rows = dict(batches=2**40, length=2**40)
    everything_wrong = dict(row_width=0, head_width=0, first=4, heads=0, **bad_rows)
    expect_invalid_argument(flint, function, "row_width (0) must be", nothing(), **everything_wrong)
    expect_invalid_argument(
        flint, function, "head_width (0) must be", nothing(), **{**everything_wrong, "row_width": 32}
    )
    expect_invalid_argument(
        flint, function, "first (4) must be", nothing(), **{**everything_wrong, "row_width": 32, "head_width": 8}
    )
    expect_invalid_argument(
        flint,
        function,
        "heads must be positive",
        nothing(),
        **{**everything_wrong, "row_width": 32, "head_width": 8, "first": 8},
    )
    expect_invalid_argument(
        flint, function, "run past row_width (32)", nothing(), **{**bad_rows, "heads": 9}
    )
    expect_invalid_argument(flint, function, "can be up to 17179869176 rows", nothing(), **bad_rows)


# For the overlap tests: the bytes each buffer covers, the whole extent of
# its layout whatever columns are moved.
def extent(function):
    return (
        {"y": HEADS_BYTES, "x": ROWS_BYTES}
        if function == "heads_split"
        else {"y": ROWS_BYTES, "x": HEADS_BYTES}
    )


SPACING = 4096

# How far x starts after y: at the same address, inside y, 16 bytes before y's
# end, and (negative) with x ending 16 bytes into y. These are overlaps by
# extent: for merge, x-at-end-of-y starts in the last row's trailing columns,
# which the call doesn't write. Just after y's end, and just before its start,
# x touches y but doesn't overlap it.
OVERLAPPING = {
    "same-start": lambda s: 0,
    "x-inside-y": lambda s: 16,
    "x-at-end-of-y": lambda s: s["y"] - 16,
    "y-inside-x": lambda s: 16 - s["x"],
}
TOUCHING = {
    "x-after-y": lambda s: s["y"],
    "x-before-y": lambda s: -s["x"],
}


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("label", OVERLAPPING)
def test_y_may_not_overlap_x(flint, function, label):
    # The buffers are real memory in an arena, so a call that wrongly went
    # ahead would do no harm.
    arena = torch.zeros(8 * SPACING, dtype=torch.uint8, device="cuda")
    base = arena.data_ptr() + 2 * SPACING
    addresses = {"y": base, "x": base + OVERLAPPING[label](extent(function))}
    expect_invalid_argument(flint, function, "y and x overlap", addresses)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("label", TOUCHING)
def test_buffers_next_to_each_other_are_not_overlapping(flint, function, label):
    sizes = extent(function)
    arena = torch.full((8 * SPACING,), 0x5A, dtype=torch.uint8, device="cuda")
    y_at = 2 * SPACING
    x_at = y_at + TOUCHING[label](sizes)
    y = arena[y_at : y_at + sizes["y"]].view(torch.int16)
    x = arena[x_at : x_at + sizes["x"]].view(torch.int16)
    x.copy_(random_bits(x.numel()))
    x_before = x.clone()
    outside = torch.ones(arena.numel(), dtype=torch.bool, device="cuda")
    outside[y_at : y_at + sizes["y"]] = False
    others = arena[outside].clone()

    getattr(flint, function)(y, x, *SMALL.values())
    torch.cuda.synchronize()

    assert torch.equal(x, x_before)
    assert torch.equal(arena[outside], others)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "batches, length",
    [(0, 4), (2, 0), (0, 0), (0, 2**64 - 1), (2**64 - 1, 0), (0, 2**40), (2**40, 0)],
    ids=lambda v: str(v),
)
def test_empty_problems_do_nothing(flint, function, batches, length):
    # Not even the null pointers are looked at, and a length too large for
    # batches * length is no error when there are no batches.
    buffers = small_problem(function)
    buffers["y"].fill_(3)

    getattr(flint, function)(None, None, batches, length, 2, 8, 32, 8)
    call(flint, function, buffers, batches=batches, length=length)
    torch.cuda.synchronize()

    assert (buffers["y"] == 3).all()


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("empty", [dict(batches=0), dict(length=0), dict(batches=0, length=0)], ids=str)
def test_empty_problems_still_check_the_other_sizes(flint, function, empty):
    for reason, sizes in [
        ("row_width (0) must be", dict(row_width=0)),
        ("row_width (12) must be", dict(row_width=12)),
        ("head_width (0) must be", dict(head_width=0)),
        ("head_width (4) must be", dict(head_width=4)),
        ("first (4) must be", dict(first=4)),
        ("heads must be positive", dict(heads=0)),
        ("run past row_width (32)", dict(heads=4)),
        ("run past row_width (32)", dict(heads=2**61)),
        ("run past row_width (32)", dict(first=32)),
    ]:
        expect_invalid_argument(flint, function, reason, nothing(), **empty, **sizes)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_a_valid_call_works_after_errors(flint, function):
    buffers = small_problem(function)
    source = buffers["x"]
    source[:] = torch.arange(source.numel(), dtype=torch.int16, device="cuda")
    for sizes in (dict(row_width=0), dict(heads=2**61), dict(batches=2**40, length=2**40)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, function, buffers, **sizes)
    with pytest.raises(flint_lib.FlintError):
        call(flint, function, {**buffers, "y": None})

    call(flint, function, buffers)
    torch.cuda.synchronize()

    shape = (2, 4, 2, 8, 32, 8)
    if function == "heads_split":
        expected = split_reference(source[: 8 * 32], shape).reshape(-1)
        assert torch.equal(buffers["y"][:128], expected)
        assert (buffers["y"][128:] == 0).all()
    else:
        expected = merge_reference(source[:128], shape, torch.zeros((8, 32), dtype=torch.int16, device="cuda"))
        assert torch.equal(buffers["y"][: 8 * 32].view(8, 32), expected)
        assert (buffers["y"][8 * 32 :] == 0).all()
