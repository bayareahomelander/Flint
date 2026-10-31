"""flint_copy: bitwise results, stream ordering, and argument errors."""

import threading

import pytest
import torch

import flint_lib

GUARD = 0xA5
GUARD_BYTES = 64
SIDE_STREAM_SLEEP_CYCLES = 100_000_000  # about 50 ms

# Small sizes cross every source and destination alignment; the big ones run
# aligned plus one unaligned case, to keep the suite quick.
SMALL_SIZES = [1, 15, 16, 17, 255, 4097]
BIG_SIZES = [2**20 + 3, 2**26]
COPY_CASES = [(n, s, d) for n in SMALL_SIZES for s in (0, 1, 8) for d in (0, 1, 8)] + [
    (n, s, d) for n in BIG_SIZES for s, d in ((0, 0), (1, 8))
]


def random_bytes(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=generator).cuda()


def guarded_buffer(offset, n):
    return torch.full((offset + n + GUARD_BYTES,), GUARD, dtype=torch.uint8, device="cuda")


@pytest.mark.parametrize("n, src_offset, dst_offset", COPY_CASES)
def test_bitwise_copy(flint, n, src_offset, dst_offset):
    data = random_bytes(n)
    src = torch.empty(src_offset + n, dtype=torch.uint8, device="cuda")
    src[src_offset:] = data
    dst = guarded_buffer(dst_offset, n)

    flint.copy(dst[dst_offset:], src[src_offset:], n)
    torch.cuda.synchronize()

    assert torch.equal(dst[dst_offset : dst_offset + n], data)
    assert (dst[:dst_offset] == GUARD).all()
    assert (dst[dst_offset + n :] == GUARD).all()


@pytest.mark.parametrize("src_offset, dst_offset", [(0, 0), (1, 8)])
def test_run_twice_is_bitwise_identical(flint, src_offset, dst_offset):
    n = 2**20 + 3
    src = random_bytes(src_offset + n, seed=1)[src_offset:]
    # Different starting contents, so a copy that writes nothing can't pass.
    first = torch.zeros(dst_offset + n, dtype=torch.uint8, device="cuda")[dst_offset:]
    second = torch.full((dst_offset + n,), 0xFF, dtype=torch.uint8, device="cuda")[dst_offset:]

    flint.copy(first, src, n)
    flint.copy(second, src, n)
    torch.cuda.synchronize()

    assert torch.equal(first, second)
    assert torch.equal(first, src)


def test_copy_runs_on_the_given_stream(flint):
    n = 2**20
    src = torch.zeros(n, dtype=torch.uint8, device="cuda")
    dst = torch.zeros(n, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # The fill finishes only after the sleep, so the copy sees 7s only if it is
    # queued on the same stream behind them.
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        src.fill_(7)
        flint.copy(dst, src, n, stream=stream)
    stream.synchronize()

    assert (dst == 7).all()


def test_copy_runs_on_the_default_stream(flint):
    n = 2**20
    src = torch.zeros(n, dtype=torch.uint8, device="cuda")
    dst = torch.zeros(n, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
    src.fill_(7)
    flint.copy(dst, src, n)
    torch.cuda.synchronize()

    assert (dst == 7).all()


def expect_invalid_argument(flint, message, *args):
    with pytest.raises(flint_lib.FlintError) as error:
        flint.copy(*args)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    assert message in flint.last_error()


def test_null_dst(flint):
    src = random_bytes(64)
    expect_invalid_argument(flint, "dst is null", None, src, 64)


def test_null_src(flint):
    dst = torch.empty(64, dtype=torch.uint8, device="cuda")
    expect_invalid_argument(flint, "src is null", dst, None, 64)


def test_overlap(flint):
    buf = random_bytes(128)
    expect_invalid_argument(flint, "overlap", buf[8:], buf, 64)


def test_adjacent_ranges_are_not_overlapping(flint):
    buf = random_bytes(128)
    expected = buf[:64].clone()

    flint.copy(buf[64:], buf[:64], 64)
    torch.cuda.synchronize()

    assert torch.equal(buf[64:], expected)


def test_address_space_wraparound(flint):
    buf = random_bytes(64)
    other = random_bytes(64)
    expect_invalid_argument(flint, "address space", buf, other, 2**64 - 1)


def test_zero_bytes_touches_nothing(flint):
    flint.copy(None, None, 0)


def test_last_error_is_per_thread(flint):
    buf = random_bytes(64)
    with pytest.raises(flint_lib.FlintError):
        flint.copy(None, buf, 64)

    seen = {}

    def worker():
        with pytest.raises(flint_lib.FlintError):
            flint.copy(buf, None, 64)
        seen["error"] = flint.last_error()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert "src is null" in seen["error"]
    assert "dst is null" in flint.last_error()
