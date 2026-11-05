"""Small pieces the GEMM test files share: the sanitizer switch, seeded
generators, and FP16 input makers."""

import os

import torch

# scripts/sanitize.py runs the suite under compute-sanitizer, which is far
# slower, so shapes shrink or are skipped there.
SANITIZER = os.environ.get("FLINT_SANITIZER") == "1"
SIDE_STREAM_SLEEP_CYCLES = 100_000_000  # about 50 ms


def generator(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def normal(shape, std, g):
    return (torch.randn(shape, generator=g, device="cuda") * std).half()


def clamp_half(values, limit):
    return values.clamp(-limit, limit).half()


def as_int16(value):
    """A 16-bit pattern as the signed value torch.int16 holds."""
    return (value + 0x8000) % 0x10000 - 0x8000


# The smallest magnitude that rounds to inf, to nearest: halfway between FP16's largest
# value, 65504, and the next power of 2 it would have, 65536 (a tie goes to the even one,
# which is inf).
FP16_OVERFLOW = 65520.0


def round_to_half(x):
    """x (float64, any sign and magnitude, no NaN) correctly rounded to FP16, ties to even:
    |x| at or above 65520 gives an inf of x's sign, as to-nearest does, and -0.0 gives -0.
    Tensor.half() goes through FP32, whose rounding can differ from a single rounding at a
    tie; this rounds |x| to the nearest of the result and its two neighbours, and restores
    the sign."""
    a = x.abs()
    h = a.half()
    base = h.view(torch.int16)
    best_bits, best_error = base.clone(), (a - h.double()).abs()
    for delta in (-1, 1):
        candidate = (base.int() + delta).clamp(0, 0x7BFF).to(torch.int16)
        error = (a - candidate.view(torch.float16).double()).abs()
        take = (error < best_error) | (
            (error == best_error) & ((candidate & 1) == 0) & ((best_bits & 1) == 1)
        )
        best_bits = torch.where(take, candidate, best_bits)
        best_error = torch.where(take, error, best_error)
    # The neighbour search stops at FP16's largest value, so the overflow is set here.
    best_bits = torch.where(a >= FP16_OVERFLOW, torch.full_like(best_bits, 0x7C00), best_bits)
    sign = torch.signbit(x).to(torch.int16) * as_int16(0x8000)
    return (best_bits | sign).view(torch.float16)
