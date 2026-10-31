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
