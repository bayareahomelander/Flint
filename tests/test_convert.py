"""flint_fp32_to_fp16: converts count FP32 values to FP16, each rounded to
nearest with ties to even, so a value past 65,520 gives inf (nothing
saturates) and a NaN stays a NaN. Any count works, not only multiples of 4: the
kernel converts four values at a time in grid-stride loops, and the last
count % 4 values one per thread.

The conversion is exact work with one right answer per input, so the tests
compare bit patterns with PyTorch's x.half() on the GPU, never with a
tolerance and without the gate. The exhaustive test runs all 2^32 FP32 bit
patterns (a NaN input must give a NaN; its payload isn't compared); under the
sanitizer, where that is too slow, the sampled test stands in for it with a few
hundred thousand random patterns and the edge values. The explicit tests state the expected FP16
bits by hand, for failures that read: ties at several magnitudes, the overflow
threshold 65,520, FP16's subnormal range and its underflow, signed zeros,
infinities, NaNs of several payloads (one that a truncating conversion would
turn into inf), and every FP16 value converted up and back. The rest covers
counts around the loops' edges, determinism, guard bands, stream ordering and
argument errors."""

import time

import pytest
import torch

import flint_lib
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator

# The output of a guarded call sits `offset` blocks past the start of a buffer
# that is filled with a pattern, which must be intact around it afterwards.
GUARD_AFTER = 64
SENTINEL = 0x5A5A

INF = 0x7C00
SIGN = 0x8000

# 2^24 patterns per chunk is 64 MiB of input and 32 MiB of output, which the
# laptop's 6 GiB (shared with the desktop) holds with room to spare.
CHUNK = 1 << 24


def bits16(t):
    """t's FP16 bit patterns as int16."""
    return t.contiguous().view(torch.int16)


def bits32(t):
    return t.contiguous().view(torch.int32)


def patterns(y):
    """FP16 bit patterns as a Python list of unsigned 16-bit ints."""
    return (bits16(y).int() & 0xFFFF).tolist()


def floats_from_bits(values):
    """FP32 tensor on the GPU with the given unsigned 32-bit patterns."""
    signed = [(v + 0x80000000) % 0x100000000 - 0x80000000 for v in values]
    return torch.tensor(signed, dtype=torch.int32, device="cuda").view(torch.float32)


def flint_convert(flint, x, offset=1, fill=SENTINEL, stream=None):
    """Runs flint_fp32_to_fp16 on the FP32 vector x with y inside a buffer
    pre-filled with `fill` (a 16-bit pattern), 8 * offset halves in, which keeps
    it 16-byte aligned. Checks that the buffer around y and x are untouched, and
    returns y as FP16."""
    count = x.numel()
    at = 8 * offset
    buffer = torch.full(
        (at + count + GUARD_AFTER,), as_int16(fill), dtype=torch.int16, device="cuda"
    )
    y = buffer[at : at + count].view(torch.float16)
    x_before = bits32(x).clone()

    flint.fp32_to_fp16(y, x, count, stream=stream)
    torch.cuda.synchronize()

    guard = torch.full((1,), as_int16(fill), dtype=torch.int16, device="cuda")
    assert (buffer[:at] == guard).all(), "flint_fp32_to_fp16 wrote before y"
    assert (buffer[at + count :] == guard).all(), "flint_fp32_to_fp16 wrote past y"
    assert torch.equal(bits32(x), x_before), "x changed"
    return y


def check_equal_to_pytorch(y, x):
    """y equals x.half() bit for bit, except that a NaN may carry any payload;
    on failure, names the first input that differs."""
    nan = torch.isnan(x)
    assert torch.equal(torch.isnan(y), nan), "NaN-ness differs"
    wrong = (bits16(y) != bits16(x.half())) & ~nan
    if wrong.any():
        i = wrong.nonzero()[0].item()
        raise AssertionError(
            f"input {i}: 0x{bits32(x)[i].item() & 0xFFFFFFFF:08x} ({x[i].item()!r}) gave "
            f"0x{bits16(y)[i].item() & 0xFFFF:04x}, PyTorch 0x{bits16(x.half())[i].item() & 0xFFFF:04x}"
        )


def check_patterns(flint, values, expected, offset=1):
    """Converts values (FP32 tensor) and compares the FP16 bit patterns with
    the expected unsigned ints, naming the first input that differs. values may
    be a list of Python floats."""
    if not isinstance(values, torch.Tensor):
        values = float32_tensor(values)
    got = patterns(flint_convert(flint, values, offset))
    bad = [i for i, (g, e) in enumerate(zip(got, expected)) if g != e]
    assert not bad, [
        (values[i].item(), f"got 0x{got[i]:04x}", f"expected 0x{expected[i]:04x}") for i in bad[:5]
    ]
    assert len(got) == len(expected)


def both_signs(values, expected):
    """The values and their negatives, with the expected patterns and the same
    patterns with the sign bit set."""
    return (
        [v for v in values] + [-v for v in values],
        list(expected) + [e | SIGN for e in expected],
    )


def float32_tensor(values):
    return torch.tensor(values, dtype=torch.float32, device="cuda")


def neighbors(values):
    """The FP32 neighbors of each value, toward zero and away from it."""
    v = float32_tensor(values)
    zero, big = torch.zeros_like(v), torch.full_like(v, float("inf"))
    return torch.nextafter(v, zero).tolist(), torch.nextafter(v, big).tolist()


# -- All of FP32 ------------------------------------------------------------


@pytest.mark.skipif(SANITIZER, reason="too slow under compute-sanitizer")
def test_every_fp32_bit_pattern_matches_pytorch(flint):
    # The chunks run in order through all 2^32 patterns, as signed 32-bit
    # integers from -2^31, so the chunk is an int32 tensor shifted in place.
    base = torch.arange(CHUNK, dtype=torch.int32, device="cuda")
    bits = torch.empty_like(base)
    y = torch.empty(CHUNK, dtype=torch.float16, device="cuda")
    start = time.perf_counter()
    nans = 0
    for first in range(-(2**31), 2**31, CHUNK):
        torch.add(base, first, out=bits)
        x = bits.view(torch.float32)
        flint.fp32_to_fp16(y, x, CHUNK)
        check_equal_to_pytorch(y, x)
        nans += torch.isnan(x).sum().item()
    seconds = time.perf_counter() - start
    print(f"\nall 2^32 FP32 patterns: {seconds:.1f} s")
    # 2 * (2^23 - 1) NaN patterns, so the test did see the NaNs.
    assert nans == 2 * (2**23 - 1)


def edge_patterns():
    """FP32 bit patterns at the edges of FP16's range and of the FP32 format."""
    values = {0, 0x80000000, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x00000001, 0x80000001}
    values |= {0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF}
    # Around 65,520 and 65,504, the tie into inf and the largest finite.
    for pattern in (0x477FE000, 0x477FF000, 0x47800000):
        values |= {pattern - 1, pattern, pattern + 1}
    # Around FP16's smallest normal, 2^-14, and subnormal, 2^-24, and the tie
    # between 0 and the smallest subnormal at 2^-25.
    for pattern in (0x38800000, 0x33800000, 0x33000000):
        values |= {pattern - 1, pattern, pattern + 1}
    return sorted(values | {v | 0x80000000 for v in values})


def test_sampled_fp32_bit_patterns_match_pytorch(flint):
    # Random patterns, which are mostly out of FP16's range, then random
    # patterns whose exponent is in or near it, then the edge values.
    n = 100_000 if SANITIZER else 300_000
    g = generator(0)
    anywhere = torch.randint(-(2**31), 2**31, (n,), dtype=torch.int64, generator=g, device="cuda")
    in_range = torch.randint(0, 2**23, (n,), dtype=torch.int64, generator=g, device="cuda")
    exponent = torch.randint(100, 146, (n,), dtype=torch.int64, generator=g, device="cuda")
    sign = torch.randint(0, 2, (n,), dtype=torch.int64, generator=g, device="cuda")
    near = (sign << 31) | (exponent << 23) | in_range
    near = near - ((near >> 31) << 32)  # as a signed 32-bit value
    edges = floats_from_bits(edge_patterns())
    x = torch.cat([anywhere.int().view(torch.float32), near.int().view(torch.float32), edges])

    check_equal_to_pytorch(flint_convert(flint, x), x)


# -- Explicit values --------------------------------------------------------


@pytest.mark.parametrize("exponent", [-14, -10, -1, 0, 5, 14, 15])
def test_ties_round_to_even_at_each_magnitude(flint, exponent):
    # FP16's value with exponent e and mantissa m is 2^e (1024 + m) / 1024 (the
    # pattern (e + 15) << 10 | m), and halfway to the next one is a tie, which
    # goes to the even mantissa (an odd m rounds up, and the carry at 1023 goes
    # into the next exponent, or to inf from e = 15). One FP32 step either
    # side of the tie goes to the nearer.
    values, expected = [], []
    for m in (0, 1, 2, 3, 510, 511, 512, 1021, 1022, 1023):
        base = ((exponent + 15) << 10) | m
        tie = 2.0**exponent * (1024 + m + 0.5) / 1024
        below, above = neighbors([tie])
        values += [tie, below[0], above[0]]
        expected += [base + (m % 2), base, base + 1]
    check_patterns(flint, *both_signs(float32_tensor(values).tolist(), expected))


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive", "negative"])
def test_the_overflow_threshold_and_the_largest_value(flint, sign):
    below, above = neighbors([65520.0])
    values = [65504.0, 65519.0, below[0], 65520.0, above[0], 65535.0, 65536.0, 1e10, 3.4028235e38]
    expected = [0x7BFF, 0x7BFF, 0x7BFF, INF, INF, INF, INF, INF, INF]
    flags = 0 if sign > 0 else SIGN
    check_patterns(
        flint, float32_tensor([sign * v for v in values]), [e | flags for e in expected]
    )


def test_inf_and_signed_zero(flint):
    values = floats_from_bits([0x7F800000, 0xFF800000, 0, 0x80000000])
    check_patterns(flint, values, [INF, INF | SIGN, 0, SIGN])


def test_subnormals_are_exact(flint):
    # FP16's subnormals are k 2^-24 for k up to 1023 (pattern k), and k
    # 2^-24 is exact in FP32. Both signs.
    k = list(range(1024))
    values, expected = both_signs([v * 2.0**-24 for v in k], k)
    check_patterns(flint, float32_tensor(values), expected)


def test_subnormal_ties_round_to_even_and_the_smallest_normal_is_reached(flint):
    # (k + 1/2) 2^-24 is halfway between subnormals k and k + 1, and goes to
    # the even one: for k = 1023 up to 1024, the smallest normal, 2^-14.
    ks = list(range(1024))
    ties = [(k + 0.5) * 2.0**-24 for k in ks]
    below, above = neighbors(ties)
    values, expected = both_signs(
        [v for triple in zip(ties, below, above) for v in triple],
        [e for k in ks for e in (k + k % 2, k, k + 1)],
    )
    check_patterns(flint, float32_tensor(values), expected)


def test_underflow_to_zero_keeps_the_sign(flint):
    # 2^-25 is the tie between 0 and the smallest subnormal and goes to 0
    # (even); the FP32 neighbors go to either side; and tiny values, down to
    # FP32's smallest subnormal and 2^-126, go to zero.
    below, above = neighbors([2.0**-25])
    values = [2.0**-25, below[0], above[0], 2.0**-26, 1e-10, 1e-30, 2.0**-126, 2.0**-149, 1e-45]
    expected = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    check_patterns(flint, *both_signs(values, expected))


def test_every_fp16_value_survives_a_round_trip(flint):
    # All 65,536 FP16 patterns, widened to FP32 (exact), come back as the same
    # bits; a NaN comes back a NaN.
    halves = as_int16(torch.arange(65536, dtype=torch.int32, device="cuda")).to(torch.int16)
    halves = halves.view(torch.float16)
    x = halves.float()

    y = flint_convert(flint, x)

    nan = torch.isnan(halves)
    assert nan.sum().item() == 2 * 1023
    assert torch.equal(torch.isnan(y), nan)
    assert torch.equal(bits16(y)[~nan], bits16(halves)[~nan])


def test_nan_stays_nan_whatever_its_payload(flint):
    # The payload of 0x7F800001 is in the low bits that a truncating
    # conversion drops, which would turn it into inf.
    values = floats_from_bits(
        [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7F8007FF, 0x7F801FFF, 0x7FFFFFFF,
         0xFFFFFFFF, 0x7FA00000, 0xFFBFFFFF]
    )
    assert torch.isnan(values).all()

    y = flint_convert(flint, values)

    assert torch.isnan(y).all()


# -- Counts -----------------------------------------------------------------

# Around the sizes where the vector loop, its tail and the block grid change:
# less than a quad, the tail's 1 to 3, a block's 256 threads of 4 values.
COUNTS = [1, 2, 3, 4, 5, 7, 8, 255, 256, 257, 1023, 1024, 1025, 4 * 256 * 3 + 1]
# More than 4096 blocks of 256 threads can cover in one pass at 4 values each,
# 4,194,304, so the grid-stride loop makes two passes, and a third for its first
# 257 threads, and a tail of 3.
BIG_COUNT = 4096 * 256 * 4 * 2 + 4 * 257 + 3


def random_values(count, seed=0):
    """FP32 values over many magnitudes, including ones outside FP16's range."""
    g = generator(seed)
    sign = torch.sign(torch.randn(count, generator=g, device="cuda"))
    magnitude = torch.exp2(43.0 * torch.rand(count, generator=g, device="cuda") - 26.0)
    return sign * magnitude * (1.0 + torch.rand(count, generator=g, device="cuda"))


@pytest.mark.parametrize("count", COUNTS + [BIG_COUNT], ids=str)
def test_counts_match_pytorch_and_stay_inside_y(flint, count):
    # flint_convert also checks the guard regions and x. The tail of a count
    # that isn't a multiple of 4 is part of the comparison, and the last
    # values are checked by name.
    x = random_values(count)

    y = flint_convert(flint, x, offset=1 + count % 3)

    assert torch.equal(bits16(y), bits16(x.half()))
    assert torch.equal(bits16(y[-(count % 4 or 4) :]), bits16(x[-(count % 4 or 4) :].half()))


@pytest.mark.parametrize("count", [1, 3, 5, 1025])
def test_every_value_is_written(flint, count):
    # y starts as NaN, so a value that is skipped shows.
    x = random_values(count)

    y = flint_convert(flint, x, fill=0xFFFF)

    assert not torch.isnan(y).any()


# -- Determinism ------------------------------------------------------------


@pytest.mark.parametrize("count", [1025, BIG_COUNT], ids=str)
def test_run_twice_is_bitwise_identical(flint, count):
    x = random_values(count, seed=1)
    # Different fills to start from (NaN, then a pattern that is 61,280), so a
    # store that is skipped, or one that lands in only one run, shows.
    first = flint_convert(flint, x, fill=0xFFFF)
    second = flint_convert(flint, x, fill=0x7B7B)

    assert torch.equal(bits16(first), bits16(second))
    assert torch.equal(bits16(first), bits16(x.half()))


# -- Streams ----------------------------------------------------------------


@pytest.mark.parametrize("explicit", [True, False], ids=["explicit-stream", "current-stream"])
def test_conversion_runs_on_the_given_stream(flint, explicit):
    count = 1000
    x = torch.zeros(count, dtype=torch.float32, device="cuda")
    y = torch.zeros(count, dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fill finishes only after the sleep, so y is 1.5 only if the kernel is
    # queued on the same stream behind it. With the current stream the call has
    # no stream argument, and torch's current stream is the side one.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        x.fill_(1.5)
        flint.fp32_to_fp16(y, x, count, stream=stream if explicit else None)
    stream.synchronize()

    assert (y == 1.5).all()


# -- Argument errors --------------------------------------------------------

NAMES = ("y", "x")
MAX_COUNT = 2**62 - 1


def small_problem(count=64):
    """Valid tensors by name, in buffers with room to spare."""
    return {
        "y": torch.zeros(count + 64, dtype=torch.float16, device="cuda"),
        "x": torch.ones(count + 64, dtype=torch.float32, device="cuda"),
    }


def call(flint, buffers, count=64):
    flint.fp32_to_fp16(buffers["y"], buffers["x"], count)


def expect_invalid_argument(flint, reason, buffers, count=64):
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, buffers, count)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_fp32_to_fp16", reason):
        assert text in str(error.value)
        assert text in flint.last_error()


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    expect_invalid_argument(flint, f"{name} is null", {**small_problem(), name: None})


@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name):
    # y is FP16, but a multiple of 4 bytes short of 16-byte alignment is still
    # misaligned for the FP32 buffer; y goes 2 bytes off.
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
    # Per buffer: null, then alignment, then the address space; y before x.
    addresses = {name: t.data_ptr() for name, t in small_problem().items()}
    expect_invalid_argument(
        flint, "y is not 16-byte aligned", {**addresses, "y": addresses["y"] + 2, "x": None}
    )
    expect_invalid_argument(flint, "y is null", {**addresses, "y": None, "x": addresses["x"] + 4})
    expect_invalid_argument(
        flint, "x is not 16-byte aligned", {**addresses, "x": addresses["x"] + 4}
    )
    expect_invalid_argument(flint, "y runs past the end", {"y": 2**64 - 16, "x": None})


@pytest.mark.parametrize("count", [2**62, 2**62 + 1, 2**63, 2**64 - 1], ids=str)
def test_count_may_not_exceed_a_quarter_of_the_address_space(flint, count):
    expect_invalid_argument(
        flint,
        f"count ({count}) can be up to 4611686018427387903",
        small_problem(),
        count=count,
    )


def test_the_largest_count_passes_the_count_check_and_does_not_wrap(flint):
    # count * 4 is 2^64 - 4 and fits; the call gets on to the pointers, where a
    # null y is the first error, and with real pointers x's extent is what runs
    # past the end of the address space.
    expect_invalid_argument(
        flint, "y is null", {**small_problem(), "y": None}, count=MAX_COUNT
    )
    expect_invalid_argument(flint, "x runs past the end of the address space", small_problem(), count=MAX_COUNT)


def test_the_count_limit_is_checked_before_any_pointer(flint):
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(flint, "count (4611686018427387904) can be up to", nothing, count=2**62)


# The bytes each buffer covers in the overlap tests, for 64 values.
SIZES = {"y": 128, "x": 256}
SPACING = 4096

# How far x starts after y: at the same address, inside y, 16 bytes before y's
# end, just after y's end (adjacent, which is no overlap), and (negative) with
# x ending 16 bytes into y, or just at y's start (adjacent).
SHIFTS = {
    "same-start": (0, True),
    "x-inside-y": (16, True),
    "x-at-end-of-y": (SIZES["y"] - 16, True),
    "y-inside-x": (16 - SIZES["x"], True),
    "x-after-y": (SIZES["y"], False),
    "x-before-y": (-SIZES["x"], False),
}


@pytest.mark.parametrize("label", SHIFTS)
def test_y_may_not_overlap_x_but_may_touch_it(flint, label):
    shift, overlapping = SHIFTS[label]
    # An arena of real memory with a pattern around, so a call that wrongly
    # went ahead would do no harm, and a correct one is seen to write inside y
    # only.
    arena = torch.zeros(8 * SPACING, dtype=torch.uint8, device="cuda")
    y_at = 2 * SPACING
    y = arena[y_at : y_at + SIZES["y"]].view(torch.float16)
    x = arena[y_at + shift : y_at + shift + SIZES["x"]].view(torch.float32)
    if not overlapping:
        x.fill_(2.5)
        arena_before = arena.clone()
        flint.fp32_to_fp16(y, x, 64)
        torch.cuda.synchronize()
        assert (y == 2.5).all()
        outside = torch.ones_like(arena, dtype=torch.bool)
        outside[y_at : y_at + SIZES["y"]] = False
        assert torch.equal(arena[outside], arena_before[outside])
    else:
        with pytest.raises(flint_lib.FlintError) as error:
            flint.fp32_to_fp16(y, x, 64)
        assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
        assert "flint_fp32_to_fp16: y and x overlap" in str(error.value)


def test_zero_count_does_nothing(flint):
    # Not even the null pointers are looked at.
    buffers = small_problem()
    buffers["y"].fill_(3)

    flint.fp32_to_fp16(None, None, 0)
    flint.fp32_to_fp16(buffers["y"], None, 0)
    flint.fp32_to_fp16(None, buffers["x"], 0)
    call(flint, buffers, count=0)
    torch.cuda.synchronize()

    assert (buffers["y"] == 3).all()


def test_a_valid_call_works_after_errors(flint):
    buffers = small_problem()
    buffers["x"][:64] = torch.arange(64, device="cuda") / 4
    for count in (2**62, 2**64 - 1):
        with pytest.raises(flint_lib.FlintError):
            call(flint, buffers, count)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**buffers, "y": None})

    call(flint, buffers)
    torch.cuda.synchronize()

    assert torch.equal(buffers["y"][:64].float(), buffers["x"][:64])
    assert (buffers["y"][64:] == 0).all()
