"""The correctness gate from PLAN.md, shared by every kernel's tests.

A kernel's output is compared with a float reference (the same operation in
FP32, or FP64 for small cases, on the FP16 inputs upcast) and with a peer (the
PyTorch FP16 path). The gate passes when, for every seed, Flint's error is
within `factor` times the peer's worst error over the seeds, by relative L2 and
by max-abs, but never held tighter than two floors that FP16 output rounding
alone can approach: a relative L2 of 2^-11 and a max-abs of 2 FP16 ulps of the
reference's largest magnitude. Use FORWARD_FACTOR for forward outputs and
GRADIENT_FACTOR for gradients.

Typical use, with one (flint, peer, reference) triple per seed:

    def triples():
        for seed in gate.SEEDS:
            x, w = make_inputs(seed)
            yield flint_gemm(x, w), gate.peer_gemm_nt(x, w), gate.reference(x, w)

    gate.gate(gate.FORWARD_FACTOR, triples())

`gate` consumes the iterable lazily and keeps only the errors, so a generator
holds one seed's tensors in memory at a time.

Errors are computed in float64 and are NaN where the output has a NaN that the
reference lacks, or an inf matching an inf (inf - inf), so the gate fails on
them. Mask non-finite elements out of all three tensors first when a test
expects them.
"""

import contextlib
import math
from collections import namedtuple

import torch

SEEDS = (0, 1, 2)
FORWARD_FACTOR = 2.0
GRADIENT_FACTOR = 3.0
REL_L2_FLOOR = 2.0**-11
MAX_ABS_FLOOR_ULPS = 2
# Up to this many multiply-adds the reference runs in FP64, beyond it in FP32.
FP64_REFERENCE_LIMIT = 2**24

# One output's errors against its reference. ref_max is max|reference|, which
# sets the max-abs floor.
Errors = namedtuple("Errors", "rel_l2 max_abs ref_max")
# The bounds a gate applied, kept for tests that want to print them.
Bounds = namedtuple("Bounds", "rel_l2 max_abs")


def fp16_ulp(v: float) -> float:
    """The spacing of FP16 values at magnitude v: 2^(floor(log2 v) - 10), and
    the subnormal spacing 2^-24 below 2^-14."""
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"fp16_ulp needs a finite, non-negative value, got {v}")
    if v < 2.0**-14:
        return 2.0**-24
    # frexp gives v = mantissa * 2^exponent with mantissa in [0.5, 1), so
    # floor(log2 v) is exponent - 1, with no floating-point log to misround.
    return 2.0 ** (math.frexp(v)[1] - 1 - 10)


def errors(out: torch.Tensor, ref: torch.Tensor, chunk: int = 1 << 22) -> Errors:
    """Relative L2, ||out - ref||_2 / ||ref||_2, and max-abs, max|out - ref|,
    both in float64, plus max|ref|. The tensors must have the same shape; they
    may be any floating dtype on one device. An all-zero reference gives a
    relative L2 of 0 for an exact match and inf otherwise."""
    if out.shape != ref.shape:
        raise ValueError(f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}")
    out = out.reshape(-1)
    ref = ref.reshape(-1)
    zero = torch.zeros((), dtype=torch.float64, device=out.device)
    sq_diff = sq_ref = max_diff = max_ref = zero
    # Chunks bound the float64 temporaries, which matter for the logits shape.
    for start in range(0, out.numel(), chunk):
        o = out[start : start + chunk].double()
        r = ref[start : start + chunk].double()
        d = o - r
        sq_diff = sq_diff + d.square().sum()
        sq_ref = sq_ref + r.square().sum()
        max_diff = torch.maximum(max_diff, d.abs().max())
        max_ref = torch.maximum(max_ref, r.abs().max())
    sq_diff, sq_ref, max_diff, max_ref = torch.stack(
        [sq_diff, sq_ref, max_diff, max_ref]
    ).tolist()
    if sq_ref == 0.0:
        rel = 0.0 if sq_diff == 0.0 else math.inf
    else:
        rel = math.sqrt(sq_diff) / math.sqrt(sq_ref)
    return Errors(rel, max_diff, max_ref)


@contextlib.contextmanager
def peer_math():
    """FP16 matmuls without reduced-precision reduction, and FP32 ones without
    TF32, as the peer and the reference need; restores the previous settings."""
    matmul = torch.backends.cuda.matmul
    saved = (matmul.allow_fp16_reduced_precision_reduction, matmul.allow_tf32)
    matmul.allow_fp16_reduced_precision_reduction = False
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        matmul.allow_fp16_reduced_precision_reduction, matmul.allow_tf32 = saved


def _reference_dtype(m: int, n: int, k: int) -> torch.dtype:
    return torch.float64 if m * n * k <= FP64_REFERENCE_LIMIT else torch.float32


def reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x w^T (w is n x k) on the upcast inputs: float64 when m * n * k is at
    most 2^24, float32 otherwise."""
    m, k = x.shape
    n = w.shape[0]
    dtype = _reference_dtype(m, n, k)
    with peer_math():
        return x.to(dtype) @ w.to(dtype).T


def reference_bias(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """x w^T + bias (w is n x k, bias is n FP32 values, one per column) on the
    upcast inputs: float64 when m * n * k is at most 2^24, float32 otherwise."""
    m, k = x.shape
    n = w.shape[0]
    dtype = _reference_dtype(m, n, k)
    with peer_math():
        return x.to(dtype) @ w.to(dtype).T + bias.to(dtype)


def reference_bias_gelu(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """GELU, the tanh approximation, of reference_bias(x, w, bias), in its
    dtype."""
    return torch.nn.functional.gelu(reference_bias(x, w, bias), approximate="tanh")


def reference_nn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x w (w is k x n) on the upcast inputs: float64 when m * n * k is at most
    2^24, float32 otherwise."""
    m, k = x.shape
    n = w.shape[1]
    dtype = _reference_dtype(m, n, k)
    with peer_math():
        return x.to(dtype) @ w.to(dtype)


def reference_dgelu(x: torch.Tensor, w: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """GELU's backward pass, (x w) GELU'(pre) with the tanh approximation, on
    the upcast inputs (w is k x n, pre is m x n): x w as in reference_nn, then
    PyTorch's gelu_backward, in float64 when m * n * k is at most 2^24 and in
    float32 otherwise."""
    m, k = x.shape
    n = w.shape[1]
    dtype = _reference_dtype(m, n, k)
    with peer_math():
        dh = x.to(dtype) @ w.to(dtype)
    return torch.ops.aten.gelu_backward(dh, pre.to(dtype), approximate="tanh")


def reference_tn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x^T w (x is k x m, w is k x n) on the upcast inputs: float64 when
    m * n * k is at most 2^24, float32 otherwise."""
    k, m = x.shape
    n = w.shape[1]
    dtype = _reference_dtype(m, n, k)
    with peer_math():
        return x.to(dtype).T @ w.to(dtype)


def peer_gemm_nt(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """PyTorch's own FP16 x w^T, the peer, with reduced-precision reduction and
    TF32 off."""
    with peer_math():
        return x @ w.T


def peer_gemm_nt_bias(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """PyTorch's own FP16 x w^T + bias, the peer for the biased NT layout, as
    autocast runs a linear layer (the FP32 bias cast to FP16), with
    reduced-precision reduction and TF32 off."""
    with peer_math():
        return torch.nn.functional.linear(x, w, bias.half())


def peer_gemm_nt_bias_gelu(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """PyTorch's own FP16 GELU(x w^T + bias), the peer for the fused forward
    layer: autocast's linear layer (the FP32 bias cast to FP16) rounding to FP16
    before the tanh GELU, with reduced-precision reduction and TF32 off."""
    with peer_math():
        return torch.nn.functional.gelu(
            torch.nn.functional.linear(x, w, bias.half()), approximate="tanh"
        )


def peer_gemm_nn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """PyTorch's own FP16 x w (w is k x n), the peer for the NN layout, with
    reduced-precision reduction and TF32 off."""
    with peer_math():
        return x @ w


def peer_gemm_nn_dgelu(x: torch.Tensor, w: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """PyTorch's own FP16 GELU backward pass, the peer for the fused input
    gradient: the FP16 x w (rounded to FP16), then the tanh GELU's backward
    pass on it and pre in FP16, with reduced-precision reduction and TF32
    off."""
    with peer_math():
        return torch.ops.aten.gelu_backward(x @ w, pre, approximate="tanh")


def peer_gemm_tn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """PyTorch's own FP16 x^T w (x is k x m, w is k x n), the peer for the TN
    layout, as autocast computes a weight gradient: FP16 out, with
    reduced-precision reduction and TF32 off, upcast to float."""
    with peer_math():
        return (x.T @ w).float()


def check_gate(
    factor: float,
    flint_errors: list,
    peer_errors: list,
) -> Bounds:
    """Applies the gate to per-seed Errors and returns the bounds. Raises
    AssertionError, with every seed's errors beside the bounds, on a failure."""
    if len(flint_errors) != len(peer_errors) or not flint_errors:
        raise ValueError("need one Errors per seed for both Flint and the peer")
    peer_rel = max(e.rel_l2 for e in peer_errors)
    peer_max = max(e.max_abs for e in peer_errors)
    rel_bound = max(factor * peer_rel, REL_L2_FLOOR)
    # The floor depends on each seed's reference, so the max-abs bound does too.
    max_bounds = [
        max(factor * peer_max, MAX_ABS_FLOOR_ULPS * fp16_ulp(e.ref_max)) for e in flint_errors
    ]

    lines = [
        f"gate factor {factor}: peer worst rel_l2 {peer_rel:.4g}, max_abs {peer_max:.4g}; "
        f"rel_l2 bound {rel_bound:.4g} (floor {REL_L2_FLOOR:.4g})"
    ]
    failed = not (math.isfinite(peer_rel) and math.isfinite(peer_max))
    if failed:
        lines.append("the peer's own error is not finite, so the bounds mean nothing")
    for seed, (f, p, max_bound) in enumerate(zip(flint_errors, peer_errors, max_bounds)):
        # A NaN error compares false, so it fails.
        rel_ok = f.rel_l2 <= rel_bound
        max_ok = f.max_abs <= max_bound
        failed = failed or not (rel_ok and max_ok)
        lines.append(
            f"  seed {seed}: flint rel_l2 {f.rel_l2:.4g} (bound {rel_bound:.4g}) "
            f"{'ok' if rel_ok else 'FAIL'}, max_abs {f.max_abs:.4g} (bound {max_bound:.4g}) "
            f"{'ok' if max_ok else 'FAIL'}; peer rel_l2 {p.rel_l2:.4g}, max_abs {p.max_abs:.4g}; "
            f"max|ref| {f.ref_max:.4g}"
        )
    assert not failed, "\n".join(lines)
    return Bounds(rel_bound, max(max_bounds))


def gate(factor: float, seeds) -> Bounds:
    """Runs the gate over an iterable of (flint_out, peer_out, reference)
    triples, one per seed, each with a common shape. Returns the bounds."""
    flint_errors = []
    peer_errors = []
    for flint_out, peer_out, ref in seeds:
        flint_errors.append(errors(flint_out, ref))
        peer_errors.append(errors(peer_out, ref))
        del flint_out, peer_out, ref
    return check_gate(factor, flint_errors, peer_errors)
