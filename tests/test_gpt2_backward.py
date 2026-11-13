"""flint_gpt2_backward: GPT-2's backward pass, from the dlogits that the forward pass leaves to
every master weight's gradient, assembled in src/api/gpt2.cpp from the kernels the other test
files cover. test_gpt2.py covers the forward pass and the model's memory; this file covers what
backward adds, and reuses that file's helpers.

The gate is M2's done rule for the gradients. From identical weights (master weights rounded to
FP16-representable values), the reference is gpt2_torch's FP32 model, the peer is the same model
under autocast FP16 with a GradScaler, and Flint's error against the reference is at most 3x the
peer's for every parameter's gradient, by relative L2 and by max-abs, over 3 seeds (gate.py, with
the FP32 outputs' floors). All three are gradients of the same loss, 2^16 times the mean loss,
which is the forward pass's scale (2^16 / rows) times the sum of the rows' losses, so the
GradScaler's first loss scale is the one the peer trains with and no gradient is unscaled. The
shapes are the tiny config at 4 shapes, three odd models, GPT-2 at (2, 256), (1, 1024) and (8, 64)
on the init and on a mid-training mix, with random tokens and, at (1, 1024), real ones; two
micro-batches accumulated into one gradient (the reference and the peer accumulate the same two
losses' gradients); and GPT-2's whole micro-batch, (16, 1024), which runs when the GPU has 20 GiB
free (the 2080 Ti). The laptop's 6 GiB cannot hold torch's FP32 autograd of GPT-2 beside a Flint
backward model, so every case computes the reference and the peer first, moves their gradients to
the CPU, frees torch's model, and only then creates Flint's. The micro-batch's autograd runs over
chunks of sequences (the loss is a sum over rows, so the gradients add up: the reference's in FP64,
the peer's in FP32 as PyTorch's .grad does) and a small shape runs the same chunked code here. A
failed micro-batch gate is retried at another placement, as the forward pass's is, but with a
stricter rule: a flipped activation changes every gradient a little, so there is no
one-sequence allowance. Every attempt runs Flint twice, and passes only if the first run passes
the gate and the two runs give the same bits; one failure of either kind is retried at another
placement (gradient_gate_with_retry, retry_report.py), which tests on the tiny config check. With
FLINT_REQUIRE_MICRO_BATCH=1, as precheck sets it, too little free memory fails that test where it
otherwise skips. Every case prints its numbers.

The assembly's own tests check what a tolerance can't. compose_backward calls the kernels by hand
from Python, in the order the backward pass's spec gives, and every gradient must equal the
assembly's bit for bit, at tiny shapes, in write mode and after a second batch in accumulate
mode: that pins the arguments' order, the tied embedding's order ((old + output layer) +
embedding) and the accumulate flags. Then the gradients are the parameters' in name, order and
size; flint_gpt2_bytes equals an independent sum with backward; one allocation and
separate_buffers give the same bits; a model with backward gives the forward pass's bits; run
twice, with every gradient poisoned in between, is bitwise identical, wpe's unseen rows are +0
in write mode and untouched in accumulate mode, and wte's padded rows are +0; backward changes
neither the weights, the tokens, the outputs nor a later forward pass, and run twice after one
forward pass gives the same bits; the gradients double with the scale; backward reads the last
forward pass's activations and the FP16 copies of the weights, not the masters; two models, a
create and destroy loop and a side stream behave. The argument errors and the new limits come
last. Under the sanitizer (scripts/sanitize.py) only the test_sanitized_* tests run: the tiny
config with backward at small shapes, with separate_buffers so that memcheck sees each buffer's
bounds, and once as one allocation."""

import ctypes
import gc
import math
import os
import re
import types

import numpy as np
import pytest
import torch

import flint_lib
import gate
import retry_report
from flint_lib import Gpt2, Gpt2Config, Gpt2ParameterInfo
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, generator
from test_gpt2 import (  # also puts scripts/ on sys.path
    ALIGNMENT,
    GOOD,
    GPT2,
    GPT2_SHAPES,
    LEAK_TOLERANCE,
    LIMIT_CONFIGS,
    LOSS_SCALE,
    ODD,
    ODD_MODELS,
    OVERLAPS,
    REQUIRE_MICRO_BATCH,
    RETRY_PADDING_BYTES,
    TINY,
    TINY_SHAPES,
    WEIGHTS,
    Batch,
    GiB,
    MiB,
    Outputs,
    Recorder,
    all_equal,
    any_unequal,
    as_tokens,
    assert_same_bits,
    assert_same_outputs,
    bytes_of,
    call_bytes,
    call_create,
    changed_weight,
    clone_state,
    config_fields,
    expected_bytes,
    free_memory,
    gate_scale,
    gpt2_config,
    int_view,
    make_model,
    named,
    random_batch,
    random_ids,
    read_outputs,
    real_tokens,
    run_model,
    state_of,
    up,
    with_padded_rows,
    write_param,
)

import gpt2_torch as G  # noqa: E402  (test_gpt2 put scripts/ on sys.path)

# GPT-2's micro-batch with backward needs this much free GPU memory: Flint's model takes 14.6 GiB
# there, and the padding, the gradients read back and the comparison's temporaries a few more. It
# must be re-measured on the 2080 Ti, whose 22 GiB it has to fit in.
MICRO_BATCH_FREE_BYTES = 20 * GiB
# The micro-batch's autograd runs this many sequences at a time (GPT-2's FP32 autograd of 2,048
# rows takes about 5 GiB on top of the model and its gradients).
MICRO_BATCH_CHUNK = 2
# What the comparison needs beside a Flint model: a gradient read back, its reference and the
# FP64 temporaries of gate.errors, for the largest parameter (wte, 154 MB).
COMPARISON_BYTES = 1 * GiB
PATTERN32 = 0x5A5A5A5A


@pytest.fixture(autouse=True)
def free_gpu_memory():
    """The laptop's GPU is 6 GiB and shared, so each test hands back what it cached."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def sanitizer_policy(request):
    """Under compute-sanitizer only the sanitized tests run (see the module docstring)."""
    if SANITIZER and not request.node.name.startswith("test_sanitized_"):
        pytest.skip("under the sanitizer only the tiny model's test_sanitized_* tests run")


@pytest.fixture(scope="module")
def tiny_model():
    return make_model(TINY, 0, "mixed")


@pytest.fixture(scope="module")
def tiny_model_b():
    return make_model(TINY, 1, "mixed")


# -- Helpers: models, gradients ------------------------------------------------------------


def backward_config(cfg, sequences, length, separate=False):
    return gpt2_config(cfg, sequences, length, separate, backward=1)


def read_gradients(m, stream=None):
    """Copies of a model's gradients by parameter name: FP32 (rows, cols) device tensors."""
    return {n: m.read(a, (r, c), torch.float32, 0, stream) for n, a, r, c in m.gradients()}


def read_parameters(m, stream=None):
    return {n: m.read(a, (r, c), torch.float32, 0, stream) for n, a, r, c in m.parameters()}


def assert_same_gradients(a, b, what):
    assert list(a) == list(b), f"{what}: parameters {list(a)} and {list(b)}"
    for name in a:
        assert_same_bits(a[name], b[name], f"{what}: gradient of {name}")


def forward_backward(m, batch, scale, accumulate=False, stream=None):
    m.forward(batch.tokens, batch.target_tokens, scale, stream)
    m.backward(batch.tokens, accumulate, stream)


def run_backward_model(flint, config, state, batch, scale):
    """A fresh model with backward, loaded with state, run once; its outputs and gradients."""
    with Gpt2(flint, config) as m:
        m.load(state)
        forward_backward(m, batch, scale)
        return read_outputs(m, config), read_gradients(m)


def write_gradient(m, name, tensor):
    """Overwrites one gradient."""
    for n, address, rows, cols in m.gradients():
        if n == name:
            assert tensor.dtype == torch.float32 and tensor.numel() == rows * cols
            m.flint.copy(address, tensor.contiguous(), rows * cols * 4)
            torch.cuda.synchronize()
            return
    raise KeyError(name)


def poison_gradients(m):
    """Fills every gradient with a finite pattern, so a pass that skipped a write would show."""
    for _, address, rows, cols in m.gradients():
        pattern = torch.full((rows * cols,), PATTERN32, dtype=torch.int32, device="cuda")
        m.flint.copy(address, pattern, rows * cols * 4)
    torch.cuda.synchronize()


def exactly_twice(once, twice, skip=()):
    """The parameters whose gradient in twice is not exactly 2x the one in once, bit for bit.
    Accumulating adds the same values to themselves, one FP32 add each, and x + x is exact; it
    holds for every parameter but wte, whose tied gradient is (old + output layer) + embedding."""
    return [n for n in once if n not in skip and not torch.equal(int_view(2 * once[n]), int_view(twice[n]))]


# -- Helpers: the gradient gate ----------------------------------------------------------------

KINDS = ["wte", "wpe", "LayerNorm weights", "LayerNorm biases", "linear weights", "linear biases"]


def kind_of(name):
    """Which group of parameters a name belongs to (the gate prints the worst error of each)."""
    if name in ("wte.weight", "wpe.weight"):
        return name.split(".")[0]
    layer, leaf = name.split(".")[-2:]
    if layer.startswith("ln_"):
        return "LayerNorm weights" if leaf == "weight" else "LayerNorm biases"
    return "linear weights" if leaf == "weight" else "linear biases"


def torch_gradients(model, micro_batches, chunk, peer):
    """Every parameter's gradient (CPU tensors, by name, torch's shapes) of 2^16 times the mean
    loss of each micro-batch added up, from gpt2_torch's model: the FP32 reference
    (peer=False), or the autocast FP16 peer (peer=True) with a GradScaler at its initial scale
    of 2^16, whose .grad is read without unscaling (a peer gradient that isn't finite would mean
    that the scale is too large for the shape, and the gate fails on it). The model runs chunk
    sequences at a time, each pass's loss weighted by its share of its micro-batch's rows, so
    the sum is the same loss. The peer's .grad accumulates in FP32 over the passes, as PyTorch's
    does in training; the reference's are added up in FP64 when there is more than one pass."""
    scaler = torch.amp.GradScaler("cuda", init_scale=LOSS_SCALE) if peer else None
    passes = [(b, lo) for b in micro_batches for lo in range(0, b.sequences, chunk)]
    total = {}
    with gate.peer_math():
        for b, lo in passes:
            hi = min(lo + chunk, b.sequences)
            weight = (hi - lo) / b.sequences
            idx, targets = b.idx[lo:hi], b.targets[lo * b.length : hi * b.length]
            with torch.autocast("cuda", dtype=torch.float16, enabled=peer):
                loss = model(idx, targets) * weight
            if peer:
                scaler.scale(loss).backward()
                continue
            (loss * LOSS_SCALE).backward()
            if len(passes) > 1:
                for name, p in model.named_parameters():
                    if name in total:
                        total[name] += p.grad.double()
                    else:
                        total[name] = p.grad.double()
                    p.grad = None
    grads = total or {n: p.grad for n, p in model.named_parameters()}
    out = {n: g.detach().cpu() for n, g in grads.items()}
    for p in model.parameters():
        p.grad = None
    return out


def gradient_errors(flint_grads, reference, peer):
    """By parameter, Flint's and the peer's gate.errors against the reference."""
    assert list(flint_grads) == list(reference) == list(peer), "parameter names or order differ"
    errors = {}
    for name, grad in flint_grads.items():
        ref = reference[name].cuda().reshape(grad.shape)
        errors[name] = (gate.errors(grad, ref), gate.errors(peer[name].cuda().reshape(grad.shape), ref))
        del ref
    return errors


def ratio(flint_error, peer_error):
    if not (math.isfinite(flint_error) and math.isfinite(peer_error)):
        return math.inf
    if peer_error == 0:
        return 0.0 if flint_error == 0 else math.inf
    return flint_error / peer_error


def check_gradients(label, by_parameter, seeds, summaries=None):
    """The gradient gate over per-parameter lists of (Flint, peer) Errors, one per seed: each
    parameter against 3x its peer's error, FP32 floors. Prints the worst Flint/peer ratio and,
    per kind of parameter, the worst relative L2 of each; raises one AssertionError listing
    every parameter that fails."""
    failures = []
    worst_rel, worst_abs = (0.0, None), (0.0, None)
    kinds = {k: [0.0, 0.0] for k in KINDS}
    for name, per_seed in by_parameter.items():
        flint_errors, peer_errors = zip(*per_seed, strict=True)
        try:
            gate.check_gate(gate.GRADIENT_FACTOR, list(flint_errors), list(peer_errors), output="fp32")
        except AssertionError as error:
            failures.append(f"{name}: {error}")
        for f, p in per_seed:
            worst_rel = max(worst_rel, (ratio(f.rel_l2, p.rel_l2), name), key=lambda v: v[0])
            worst_abs = max(worst_abs, (ratio(f.max_abs, p.max_abs), name), key=lambda v: v[0])
            kind = kinds[kind_of(name)]
            kind[0] = max(kind[0], f.rel_l2) if math.isfinite(f.rel_l2) else math.inf
            kind[1] = max(kind[1], p.rel_l2) if math.isfinite(p.rel_l2) else math.inf
    lines = [
        f"  worst flint/peer ratio: rel_l2 {worst_rel[0]:.3g} at {worst_rel[1]}, "
        f"max_abs {worst_abs[0]:.3g} at {worst_abs[1]}",
        "  worst rel_l2 against the reference, flint | peer: "
        + "; ".join(f"{k} {kinds[k][0]:.3g} | {kinds[k][1]:.3g}" for k in KINDS),
    ]  # fmt: skip
    summary = f"GRADIENT GATE {label} (worst of {seeds} seeds, {len(by_parameter)} parameters)\n" + "\n".join(lines)
    print(summary + ("\n  FAILS: " + ", ".join(f.split(":")[0] for f in failures) if failures else ""))
    if summaries is not None:
        summaries.append(summary)
    if failures:
        # Raised by hand: assert's rewriting would indent and quote the message.
        raise AssertionError(f"{label}:\n" + "\n".join(failures))


def gradient_case(flint, name, sequences, length, weights, tokens, *, cfg=None, seeds=gate.SEEDS, micro_batches=1,
                  chunk=None, separate=False, runs=1, after_backward=None, summaries=None,
                  skip_if_too_big=True):  # fmt: skip
    """The gradient gate on one shape: per seed, a model (weights "init" or "mixed"), random or
    real tokens, the torch reference and peer (in chunks of sequences, if chunk is given), then
    Flint's forward and backward pass on micro_batches batches (the first written, the others
    accumulated); then the gate over the seeds. name is a config of gpt2_torch's, unless cfg
    gives one. The torch gradients go to the CPU and torch's model is freed before Flint's model
    is created, because the laptop's 6 GiB cannot hold both. A case that does not fit skips with
    the numbers, unless skip_if_too_big is False.

    With runs=2, Flint runs twice per seed on fresh models; only the first is gated, and the
    second must give the same bits: the return value lists the (seed, parameter) pairs where it
    does not. after_backward(seed, m), if given, runs after each run's last backward pass."""
    cfg = cfg or G.CONFIGS[name]
    config = backward_config(cfg, sequences, length, separate)
    rows = sequences * length
    scale = gate_scale(rows)
    chunk = chunk or sequences
    by_parameter, mismatches = {}, []
    for seed in seeds:
        model = make_model(cfg, seed, weights)
        batches = []
        for j in range(micro_batches):
            if tokens == "real":
                ids = real_tokens(rows + 1, seed)
                assert int(ids.max()) < cfg.vocab_size
            else:
                ids = random_ids(cfg, rows + 1, seed + 1000 * j)
            batches.append(Batch(ids, sequences, length))
        torch.cuda.reset_peak_memory_stats()
        try:
            reference = torch_gradients(model, batches, chunk, peer=False)
            peer = torch_gradients(model, batches, chunk, peer=True)
        except torch.cuda.OutOfMemoryError:
            if not skip_if_too_big:
                raise
            pytest.skip(
                f"torch's FP32 autograd of {name} at ({sequences}, {length}) in chunks of {chunk} ran out of memory "
                f"after {torch.cuda.max_memory_allocated() / GiB:.1f} GiB; {free_memory() / GiB:.1f} GiB are free"
            )
        state = {n: p.detach() for n, p in model.named_parameters()}
        del model
        needed = bytes_of(flint, config) + COMPARISON_BYTES
        free = free_memory()
        if skip_if_too_big and free < needed:
            pytest.skip(
                f"Flint's backward model of {name} at ({sequences}, {length}) needs {needed / GiB:.1f} GiB "
                f"(with {COMPARISON_BYTES / GiB:.0f} GiB for the comparison), {free / GiB:.1f} GiB are free"
            )
        first = None
        for run in range(runs):
            with Gpt2(flint, config) as m:
                m.load(state)
                for j, b in enumerate(batches):
                    forward_backward(m, b, scale, accumulate=j > 0)
                if after_backward is not None:
                    after_backward(seed, m)
                grads = read_gradients(m)
            if run == 0:
                first = grads
                continue
            mismatches += [(seed, n) for n in first if not torch.equal(int_view(first[n]), int_view(grads[n]))]
            del grads
        for n, errors in gradient_errors(first, reference, peer).items():
            by_parameter.setdefault(n, []).append(errors)
        del state, first, reference, peer, batches
        free_memory()
    label = f"{name} {sequences}x{length} {weights} weights, {tokens} tokens"
    label += (f", {micro_batches} micro-batches accumulated" if micro_batches > 1 else "")
    label += (f", chunks of {chunk}" if chunk < sequences else "") + (", separate buffers" if separate else "")
    check_gradients(label, by_parameter, len(seeds), summaries)
    return mismatches


def gradient_gate_with_retry(flint, name, sequences, length, weights, tokens, *, record, after_backward=None, **kwargs):
    """gradient_case, retried once if the first attempt fails, with RETRY_PADDING_BYTES allocated
    first. Each attempt runs Flint twice per seed on fresh models, and passes only if its first run
    passes the gate and its two runs give the same bits for every gradient. Either failure, a
    failed gate or two runs that differ, takes the retry, which is the same attempt at another
    placement. If the retry passes, the function calls record (the test's record_property) with
    retry_report.PROPERTY and a text with the first attempt's failure and the retry's numbers,
    which retry_report.py shows even under pytest -q. If the retry fails too, it raises one
    AssertionError. Anything else than a failed gate or differing runs (a skip, running out of
    memory, a CUDA error) isn't retried. after_backward(attempt, seed, m), if given, runs after
    each run's last backward pass, with attempt 0 or 1 (both of an attempt's runs share its number).

    Why: one cell of the 2080 Ti's memory, which has no ECC, sometimes reads a 1 back as 0 (see
    test_gpt2.py's gate_with_retry). When a buffer of Flint's lands on it, one value of one pass is
    wrong, and a flipped activation changes every gradient a little, so the forward gate's rule
    (the outputs may differ within one sequence) has no analogue here. The data are the same in
    both runs and Flint is deterministic, so two runs at one placement must be the same bits,
    wherever the cell lies: that agreement is the determinism check, and a bug that doesn't repeat
    (a race, a read of stale memory) would show as a difference. What this can hide: a fault
    confined to the first attempt's runs, in either of them, passes on a clean retry, and is
    recorded and reported, not silent. What it can't catch: a fault that hits both runs of an
    attempt alike (a flipped weight, or a buffer that both runs share), which the gate sees only if
    it is large; a failure that follows the data or the kernels repeats at the retry and fails."""

    def attempt(number, runs, summaries):
        hook = None if after_backward is None else (lambda seed, m: after_backward(number, seed, m))
        return gradient_case(flint, name, sequences, length, weights, tokens, runs=runs, after_backward=hook,
                             summaries=summaries, **kwargs)  # fmt: skip

    try:
        mismatches = attempt(0, 2, None)
        if not mismatches:
            return
        first = (
            "the first attempt's gate passed, but its two runs at one placement gave different gradient bits:\n"
            + "\n".join(f"  seed {seed}: the gradient of {n} differs" for seed, n in mismatches)
        )
    except AssertionError as error:
        first = str(error)
    # Leaving the except block drops the traceback, and with it the first attempt's frames.
    free_memory()
    padding = torch.empty(RETRY_PADDING_BYTES, dtype=torch.uint8, device="cuda")
    summaries = []
    try:
        mismatches = attempt(1, 2, summaries)
    except AssertionError as error:
        raise AssertionError(
            f"the gate failed twice.\nFirst attempt:\n{first}\n"
            f"Retry, with {RETRY_PADDING_BYTES / GiB:g} GiB allocated first:\n{error}"
        ) from None
    finally:
        del padding
        free_memory()
    if mismatches:
        described = "\n".join(f"  seed {seed}: the gradient of {n} differs" for seed, n in mismatches)
        raise AssertionError(
            "the gate passed on its retry, but Flint may not be deterministic: on the same data, its two runs at "
            f"the retry's placement gave different gradient bits.\n{described}\nFirst attempt:\n{first}"
        )
    record(
        retry_report.PROPERTY,
        f"The first attempt failed:\n{first}\n"
        f"The retry, with {RETRY_PADDING_BYTES / GiB:g} GiB allocated first, passed, and its two runs gave the same "
        "bits for every gradient:\n" + "\n".join(summaries),
    )


# -- The same kernels called by hand --------------------------------------------------------------


def compose_backward(flint, config, state, copy_state, batch, scale, old=None):
    """flint_gpt2_forward and flint_gpt2_backward as the kernels called one by one from Python on
    fresh buffers, written from the backward pass's specification (the order of its steps, not the
    assembly's code): the forward pass keeping each layer's activations, then the backward pass.
    The masters (embedding, biases, LayerNorm parameters) come from state and the FP16 copies of
    wte and the linear layers' weights from copy_state; with the same state for both this is a
    synced model. old, if given, is a dict of the gradients to accumulate into (they are copied);
    otherwise the gradients are written. Returns the outputs and a dict of gradients by name."""
    c = config
    W, H, B, T = c.width, c.heads, c.sequences, c.length
    rows, head_width, batches, padded = B * T, W // H, B * H, c.padded_vocab
    accumulate = old is not None

    def half(*shape):
        return torch.empty(*shape, dtype=torch.float16, device="cuda")

    def single(*shape):
        return torch.empty(*shape, dtype=torch.float32, device="cuda")

    def space(nbytes):
        return torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def fp16_of(name):
        master = copy_state[name].detach().contiguous()
        y = half(master.numel())
        flint.fp32_to_fp16(y, master.reshape(-1), master.numel())
        return y

    def norm(x, prefix):
        y, mean, rstd = half(rows, W), single(rows), single(rows)
        flint.layernorm_forward(y, mean, rstd, x, state[f"{prefix}.weight"], state[f"{prefix}.bias"], rows, W)
        return y, mean, rstd

    # -- the forward pass, keeping each block's activations
    score_scale = float(np.float32(1.0) / np.sqrt(np.float32(head_width)))
    x = single(rows, W)
    flint.embedding_forward(x, batch.tokens, state["wte.weight"], state["wpe.weight"], B, T, padded, W)
    saved = []
    for i in range(c.layers):
        p = f"blocks.{i}."
        a = types.SimpleNamespace(x_in=x)
        a.h1, a.mean1, a.rstd1 = norm(x, p + "ln_1")
        qkv = half(rows, 3 * W)
        flint.gemm_nt_bias(qkv, a.h1, fp16_of(p + "attn.c_attn.weight"), state[p + "attn.c_attn.bias"], rows, 3 * W, W)
        a.q, a.k, a.v = (half(B * H * T * head_width) for _ in range(3))
        for j, part in enumerate((a.q, a.k, a.v)):
            flint.heads_split(part, qkv, B, T, H, head_width, 3 * W, j * W)
        scores, a.p = half(batches, T, T), half(batches, T, T)
        flint.gemm_nt_batched(scores, a.q, a.k, batches, T, T, head_width, score_scale)
        flint.causal_softmax_forward(a.p, scores, batches, T)
        heads_output = half(B * H * T * head_width)
        flint.gemm_nn_batched(heads_output, a.p, a.v, batches, T, head_width, T)
        a.attn_output = half(rows, W)
        flint.heads_merge(a.attn_output, heads_output, B, T, H, head_width, W, 0)
        a.x_mid = single(rows, W)
        flint.gemm_nt_bias_residual(
            a.x_mid, a.attn_output, fp16_of(p + "attn.c_proj.weight"), state[p + "attn.c_proj.bias"], x, rows, W, W
        )
        a.h2, a.mean2, a.rstd2 = norm(a.x_mid, p + "ln_2")
        a.gelu, a.pre = half(rows, 4 * W), half(rows, 4 * W)
        flint.gemm_nt_bias_gelu(
            a.gelu, a.pre, a.h2, fp16_of(p + "mlp.c_fc.weight"), state[p + "mlp.c_fc.bias"], rows, 4 * W, W
        )
        x = single(rows, W)
        flint.gemm_nt_bias_residual(
            x, a.gelu, fp16_of(p + "mlp.c_proj.weight"), state[p + "mlp.c_proj.bias"], a.x_mid, rows, W, 4 * W
        )
        saved.append(a)
    x_final = x
    h_f, mean_f, rstd_f = norm(x_final, "ln_f")
    wte16 = fp16_of("wte.weight")
    logits = half(rows, padded)
    flint.gemm_nt(logits, h_f, wte16, rows, padded, W)
    dlogits, row_loss = half(rows, padded), single(rows)
    flint.softmax_cross_entropy(dlogits, row_loss, logits, batch.target_tokens, scale, rows, c.vocab, padded)

    # -- the gradients: written (filled with NaN, so a missed write shows) or accumulated into
    grads = {}
    for name, master in state.items():
        shape = (1, master.numel()) if master.dim() == 1 else tuple(master.shape)
        grads[name] = old[name].clone() if accumulate else torch.full(shape, math.nan, device="cuda")
    if not accumulate:
        grads["wpe.weight"][T:] = 0  # the rows past length are never seen: +0 in write mode

    def weight_gradient(name, x, w, m, n):
        # y += x^T w over the rows, with a workspace of exactly the size this call asks for
        workspace = space(flint.gemm_tn_workspace(m, n, rows))
        flint.gemm_tn(grads[name], x, w, m, n, rows, accumulate=accumulate, workspace=workspace)

    def bias_gradient(name, dy, cols):
        flint.bias_gradient(grads[name], dy, rows, cols, accumulate=accumulate,
                            workspace=space(flint.bias_gradient_workspace(rows, cols)))  # fmt: skip

    def norm_backward(prefix, dres, dh, x, mean, rstd, accumulate_dx):
        flint.layernorm_backward(
            dres, grads[f"{prefix}.weight"], grads[f"{prefix}.bias"], dh, x, mean, rstd, state[f"{prefix}.weight"],
            rows, W, accumulate_dx=accumulate_dx, accumulate_params=accumulate,
            workspace=space(flint.layernorm_backward_workspace(rows, W)),
        )  # fmt: skip

    # 1. the output layer: logits = h_f wte16^T
    weight_gradient("wte.weight", dlogits, h_f, padded, W)
    dh = half(rows, W)
    flint.gemm_nn(dh, dlogits, wte16, rows, W, padded)
    # 2. ln_f writes the residual stream's gradient
    dres = single(rows, W)
    norm_backward("ln_f", dres, dh, x_final, mean_f, rstd_f, False)
    # 3. the blocks, last to first
    for i in reversed(range(c.layers)):
        p, a = f"blocks.{i}.", saved[i]
        # the MLP
        g = half(rows, W)
        flint.fp32_to_fp16(g, dres, rows * W)
        weight_gradient(p + "mlp.c_proj.weight", g, a.gelu, W, 4 * W)
        bias_gradient(p + "mlp.c_proj.bias", g, W)
        dpre = half(rows, 4 * W)
        flint.gemm_nn_dgelu(dpre, g, fp16_of(p + "mlp.c_proj.weight"), a.pre, rows, 4 * W, W)
        weight_gradient(p + "mlp.c_fc.weight", dpre, a.h2, 4 * W, W)
        bias_gradient(p + "mlp.c_fc.bias", dpre, 4 * W)
        dh = half(rows, W)
        flint.gemm_nn(dh, dpre, fp16_of(p + "mlp.c_fc.weight"), rows, W, 4 * W)
        norm_backward(p + "ln_2", dres, dh, a.x_mid, a.mean2, a.rstd2, True)
        # the attention
        g = half(rows, W)
        flint.fp32_to_fp16(g, dres, rows * W)
        weight_gradient(p + "attn.c_proj.weight", g, a.attn_output, W, W)
        bias_gradient(p + "attn.c_proj.bias", g, W)
        d_attn_output = half(rows, W)
        flint.gemm_nn(d_attn_output, g, fp16_of(p + "attn.c_proj.weight"), rows, W, W)
        d_heads = half(B * H * T * head_width)
        flint.heads_split(d_heads, d_attn_output, B, T, H, head_width, W, 0)
        dv, dq, dk = (half(B * H * T * head_width) for _ in range(3))
        flint.gemm_tn_batched(dv, a.p, d_heads, batches, T, head_width, T)
        ds = half(batches, T, T)  # dP first, then dS in place
        flint.gemm_nt_batched(ds, d_heads, a.v, batches, T, T, head_width, 1.0)
        flint.causal_softmax_backward(ds, a.p, ds, batches, T, score_scale)
        flint.gemm_nn_batched(dq, ds, a.k, batches, T, head_width, T)
        flint.gemm_tn_batched(dk, ds, a.q, batches, T, head_width, T)
        dqkv = half(rows, 3 * W)
        for j, part in enumerate((dq, dk, dv)):
            flint.heads_merge(dqkv, part, B, T, H, head_width, 3 * W, j * W)
        weight_gradient(p + "attn.c_attn.weight", dqkv, a.h1, 3 * W, W)
        bias_gradient(p + "attn.c_attn.bias", dqkv, 3 * W)
        dh = half(rows, W)
        flint.gemm_nn(dh, dqkv, fp16_of(p + "attn.c_attn.weight"), rows, W, 3 * W)
        norm_backward(p + "ln_1", dres, dh, a.x_in, a.mean1, a.rstd1, True)
    # 4. the embedding: wte's gradient always accumulates, onto the output layer's part
    flint.embedding_backward(
        grads["wte.weight"], grads["wpe.weight"], dres, batch.tokens, B, T, padded, W,
        accumulate_wte=True, accumulate_wpe=accumulate,
        workspace=space(flint.embedding_backward_workspace(B, T, padded, W)),
    )  # fmt: skip
    torch.cuda.synchronize()
    return Outputs(logits, dlogits, row_loss, math.nan), grads


# -- 1. The gradient gate -----------------------------------------------------------------------


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("sequences, length", TINY_SHAPES, ids=lambda v: str(v))
def test_tiny_gradients_pass_the_gate(flint, sequences, length, weights):
    gradient_case(flint, "tiny", sequences, length, weights, "random")


# one-head-of-8 has gradients of 8 elements (its LayerNorm parameters and biases) summed over 16 rows
# at ODD_MODELS' (2, 8), and one implementation's own error on those swings 2.5 to 6.9 times between
# equally accurate rounding draws, so the worst-of-3-seeds gate there fails by chance about 6% of the
# time (Flint against a redraw of itself fails as often as Flint against the peer). At (64, 8) with 5
# seeds none failed, for init or mixed: the investigation resampled 2,000 seed sets from 96 seeds per
# case, and 0 of the 2,000 resampled sets of 5 failed. That is one GPU's measurement (the laptop's),
# made, not assumed; the other odd models keep ODD_MODELS' shapes and 3 seeds, and the model's own
# (2, 8) is still covered bit for bit by the by-hand comparison (HAND_CASES).
ODD_GATE_OVERRIDES = {"one-head-of-8": dict(sequences=64, seeds=range(5))}


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("name", ODD_MODELS)
def test_odd_models_gradients_pass_the_gate(flint, name, weights):
    cfg, sequences, length = ODD_MODELS[name]
    override = ODD_GATE_OVERRIDES.get(name, {})
    gradient_case(flint, name, override.get("sequences", sequences), length, weights, "random", cfg=cfg,
                  seeds=tuple(override.get("seeds", gate.SEEDS)))  # fmt: skip


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("sequences, length", GPT2_SHAPES, ids=lambda v: str(v))
def test_gpt2_gradients_pass_the_gate(flint, sequences, length, weights):
    gradient_case(flint, "gpt2", sequences, length, weights, "random")


@pytest.mark.parametrize("weights", WEIGHTS)
def test_gpt2_gradients_pass_the_gate_on_real_tokens(flint, weights):
    gradient_case(flint, "gpt2", 1, 1024, weights, "real")


@pytest.mark.parametrize("name, sequences, length", [("tiny", 2, 32), ("gpt2", 2, 256)], ids=str)
def test_two_micro_batches_accumulate_and_pass_the_gate(flint, name, sequences, length):
    # The first batch's backward pass writes and the second's adds; the reference and the peer add
    # the same two losses' gradients.
    gradient_case(flint, name, sequences, length, "mixed", "random", micro_batches=2)


def test_chunked_references_agree_with_the_unchunked_one():
    # The micro-batch gate's reference runs over chunks of sequences. Whatever the chunking (even,
    # uneven, one sequence at a time), the sum of the chunks' gradients is the whole batch's, to
    # FP32 rounding, which is far below the peers' error (about 1e-3).
    sequences, length = 4, 32
    model = make_model(TINY, 0, "mixed")
    batches = [random_batch(TINY, sequences, length, 80), random_batch(TINY, sequences, length, 81)]
    whole = torch_gradients(model, batches, sequences, peer=False)
    for chunk in (3, 1):
        chunked = torch_gradients(model, batches, chunk, peer=False)
        assert list(chunked) == list(whole)
        worst = max(((chunked[n] - whole[n].double()).norm() / whole[n].double().norm()).item() for n in whole)
        assert worst < 1e-5, (chunk, worst)
    # And the peer's: FP16 matmuls, so the chunks' sums differ from the whole batch's by the
    # peer's own error, not more.
    peer_whole = torch_gradients(model, batches, sequences, peer=True)
    peer_chunked = torch_gradients(model, batches, 3, peer=True)
    errors = {n: (gate.errors(peer_chunked[n], whole[n]).rel_l2, gate.errors(peer_whole[n], whole[n]).rel_l2) for n in whole}
    assert all(chunked_error <= 3 * whole_error + 1e-6 for chunked_error, whole_error in errors.values()), errors


@pytest.mark.parametrize("chunk", [1, 3], ids=lambda v: f"chunks-of-{v}")
def test_tiny_chunked_gradients_pass_the_gate(flint, chunk):
    # The micro-batch gate's code path (torch in chunks, summed; Flint's model created after
    # torch's is freed) at a shape that fits the laptop, with an uneven last chunk for 3.
    gradient_case(flint, "tiny", 4, 32, "mixed", "random", chunk=chunk)
    gradient_case(flint, "tiny", 4, 16, "init", "random", chunk=chunk, micro_batches=2, seeds=(0,))


def require_free_memory():
    """Skips the micro-batch gate, or with FLINT_REQUIRE_MICRO_BATCH=1 fails it, when the GPU
    has less free memory than it needs."""
    free = torch.cuda.mem_get_info()[0]
    if free < MICRO_BATCH_FREE_BYTES:
        reason = (
            f"GPT-2's (16, 1024) micro-batch with backward needs {MICRO_BATCH_FREE_BYTES / GiB:.0f} GiB free, "
            f"{free / GiB:.1f} GiB are"
        )
        if os.environ.get(REQUIRE_MICRO_BATCH) == "1":
            pytest.fail(f"{reason}, and {REQUIRE_MICRO_BATCH}=1 doesn't let this gate skip")
        pytest.skip(reason)


@pytest.mark.parametrize("weights", WEIGHTS)
def test_gpt2_micro_batch_gradients_pass_the_gate(flint, weights, record_property):
    require_free_memory()
    gradient_gate_with_retry(flint, "gpt2", 16, 1024, weights, "random", record=record_property,
                             chunk=MICRO_BATCH_CHUNK, skip_if_too_big=False)  # fmt: skip


@pytest.mark.parametrize("required", [False, True])
def test_too_little_memory_skips_the_micro_batch_gradient_gate_or_fails_it_when_required(flint, monkeypatch, required):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (MICRO_BATCH_FREE_BYTES - 1, 24 * GiB))
    if required:
        monkeypatch.setenv(REQUIRE_MICRO_BATCH, "1")
    else:
        monkeypatch.delenv(REQUIRE_MICRO_BATCH, raising=False)
    # Caught by hand: a skip raised inside pytest.raises would skip this test, not fail it.
    try:
        test_gpt2_micro_batch_gradients_pass_the_gate(flint, "init", Recorder())
        outcome, message = "ran", ""
    except pytest.skip.Exception as skipped:
        outcome, message = "skipped", str(skipped)
    except pytest.fail.Exception as failed:
        outcome, message = "failed", str(failed)
    assert outcome == ("failed" if required else "skipped"), (outcome, message)
    assert "needs 20 GiB free" in message, message


# The retry itself, on the tiny config. A fault is made on purpose: after the backward pass, one
# bit of one gradient value flips. A high one (the lowest exponent bit) halves or doubles the
# value, as a flipped bit of the 2080 Ti's memory would; a low one changes the bits and nothing
# the gate sees. A fault on the first attempt only is one the retry escapes; on both, a bug that
# follows the data. These tests give gradient_gate_with_retry a Recorder in place of
# record_property, so they never show as retried themselves.
RETRY_CASE = ("tiny", 2, 32, "mixed", "random")
FAULTY_SEED = 1
FAULTY_GRADIENT = "ln_f.weight"
HIGH_BIT, LOW_BIT = 0x00800000, 0x00000001
# check_gate numbers the seeds by their position in the list.
FAULTY_LINE = (rf"{FAULTY_GRADIENT}: gate factor[^\n]*\n(?:  seed \d+:[^\n]*\n)*"
               rf"  seed {gate.SEEDS.index(FAULTY_SEED)}: flint[^\n]*FAIL")  # fmt: skip


def flip_a_gradient_bit(m, name, bit):
    """Flips the bit of the gradient's largest value by magnitude, in Flint's memory."""
    values = read_gradients(m)[name]
    flat = int_view(values).reshape(-1).clone()
    at = int(values.abs().argmax())
    flat[at] ^= bit
    write_gradient(m, name, flat.view(torch.float32).reshape(values.shape))


class Faults:
    """An after_backward hook that flips bits of gradient values: faults maps (attempt, run) to
    the bit, for the given seeds. It logs, per call, the attempt, the seed, the run (the
    attempt's calls for that seed so far) and torch's allocated bytes (the padding is torch's)."""

    def __init__(self, faults=(), seeds=(FAULTY_SEED,), name=FAULTY_GRADIENT):
        self.faults, self.seeds, self.name, self.calls = dict(faults), seeds, name, []

    def __call__(self, attempt, seed, m):
        run = sum(1 for a, s, _, _ in self.calls if (a, s) == (attempt, seed))
        self.calls.append((attempt, seed, run, torch.cuda.memory_allocated()))
        if seed in self.seeds and (attempt, run) in self.faults:
            flip_a_gradient_bit(m, self.name, self.faults[attempt, run])

    def attempts(self):
        return [c[0] for c in self.calls]

    def padded(self, attempt):
        """Whether the padding was allocated at each of the attempt's calls."""
        return [allocated >= RETRY_PADDING_BYTES for a, _, _, allocated in self.calls if a == attempt]


def test_the_retry_is_not_taken_when_the_gradient_gate_passes(flint):
    hook, recorded = Faults(), Recorder()
    gradient_gate_with_retry(flint, *RETRY_CASE, record=recorded, after_backward=hook)
    assert recorded == []
    assert hook.attempts() == [0] * (2 * len(gate.SEEDS))  # two runs per seed
    assert not any(hook.padded(0))


def test_a_flipped_bit_in_a_gradient_fails_the_gate(flint):
    # What the retry tests below stand on: the high bit's fault fails the gate, naming the
    # parameter and the seed, and the low bit's does not.
    for bit, fails in ((HIGH_BIT, True), (LOW_BIT, False)):
        faults = Faults({(0, 0): bit})
        run = lambda: gradient_case(flint, *RETRY_CASE, after_backward=lambda seed, m: faults(0, seed, m))  # noqa: E731
        if not fails:
            run()
            continue
        with pytest.raises(AssertionError) as caught:
            run()
        assert re.search(FAULTY_LINE, str(caught.value)), str(caught.value)
        assert str(caught.value).count("gate factor") == 1, "only the one parameter should fail"


def test_a_failure_that_does_not_repeat_passes_on_the_retry(flint):
    hook, recorded = Faults({(0, 0): HIGH_BIT}), Recorder()
    gradient_gate_with_retry(flint, *RETRY_CASE, record=recorded, after_backward=hook)
    # Each attempt runs twice per seed.
    assert hook.attempts() == [0] * (2 * len(gate.SEEDS)) + [1] * (2 * len(gate.SEEDS))
    # The padding is held through the whole retry, and only there.
    assert not any(hook.padded(0)) and all(hook.padded(1))
    assert torch.cuda.memory_allocated() < RETRY_PADDING_BYTES
    assert len(recorded) == 1 and recorded[0][0] == retry_report.PROPERTY, recorded
    first, _, second = recorded[0][1].partition("The retry, with 1 GiB allocated first, passed")
    assert second, recorded[0][1]
    assert first.startswith("The first attempt failed") and "gate factor" in first
    assert re.search(FAULTY_LINE, first), first
    assert "GRADIENT GATE tiny 2x32" in second and "FAILS" not in second
    assert "the same bits for every gradient" in second


def test_a_failure_that_repeats_on_the_retry_fails(flint):
    hook, recorded = Faults({(0, 0): HIGH_BIT, (1, 0): HIGH_BIT}), Recorder()
    with pytest.raises(AssertionError) as caught:
        gradient_gate_with_retry(flint, *RETRY_CASE, record=recorded, after_backward=hook)
    text = str(caught.value)
    # The gate is checked after all the seeds, so every attempt's runs all happened.
    assert hook.attempts() == [0] * (2 * len(gate.SEEDS)) + [1] * (2 * len(gate.SEEDS))
    assert all(hook.padded(1))
    assert torch.cuda.memory_allocated() < RETRY_PADDING_BYTES
    assert recorded == []
    first, _, second = text.partition("Retry, with 1 GiB allocated first:")
    assert text.startswith("the gate failed twice") and second, text
    for attempt in (first, second):
        assert re.search(FAULTY_LINE, attempt), text


def test_a_retry_whose_two_runs_differ_fails_though_its_gate_passes(flint):
    # Flint gave different bits for the same data at the same placement: a bug that doesn't repeat,
    # which the retry must not hide, though the changed bit (the lowest) is invisible to the gate.
    for second_fault in (LOW_BIT, HIGH_BIT):
        hook, recorded = Faults({(0, 0): HIGH_BIT, (1, 1): second_fault}), Recorder()
        with pytest.raises(AssertionError) as caught:
            gradient_gate_with_retry(flint, *RETRY_CASE, record=recorded, after_backward=hook)
        text = str(caught.value)
        assert hook.attempts() == [0] * (2 * len(gate.SEEDS)) + [1] * (2 * len(gate.SEEDS))
        assert torch.cuda.memory_allocated() < RETRY_PADDING_BYTES
        assert recorded == []
        assert text.startswith("the gate passed on its retry, but Flint may not be deterministic"), text
        assert f"seed {FAULTY_SEED}: the gradient of {FAULTY_GRADIENT} differs" in text, text
        assert text.count("differs") == 1, text
        assert re.search(FAULTY_LINE, text), text


def test_a_first_attempt_whose_two_runs_differ_takes_the_retry_though_its_gate_passes(flint):
    # The first attempt's gate is clean (its first run is, and the second isn't gated), but the two
    # runs at one placement gave different bits, which is a failure of the attempt: it takes the
    # retry, and a clean retry passes and is recorded, with the difference.
    for bit in (LOW_BIT, HIGH_BIT):
        hook, recorded = Faults({(0, 1): bit}), Recorder()
        gradient_gate_with_retry(flint, *RETRY_CASE, record=recorded, after_backward=hook)
        assert hook.attempts() == [0] * (2 * len(gate.SEEDS)) + [1] * (2 * len(gate.SEEDS))
        assert not any(hook.padded(0)) and all(hook.padded(1))
        assert len(recorded) == 1 and recorded[0][0] == retry_report.PROPERTY, recorded
        first, _, second = recorded[0][1].partition("The retry, with 1 GiB allocated first, passed")
        assert second and "FAILS" not in second, recorded[0][1]
        assert first.startswith("The first attempt failed:\nthe first attempt's gate passed, but its two runs"), first
        assert f"seed {FAULTY_SEED}: the gradient of {FAULTY_GRADIENT} differs" in first and "gate factor" not in first, first


def test_only_a_failed_gradient_gate_is_retried(flint):
    calls, recorded = [], Recorder()

    def broken(attempt, seed, m):
        calls.append(attempt)
        raise RuntimeError("not a gate failure")

    with pytest.raises(RuntimeError, match="not a gate failure"):
        gradient_gate_with_retry(flint, *RETRY_CASE, record=recorded, after_backward=broken)
    assert calls == [0] and recorded == []


def test_the_retry_works_over_chunks_and_micro_batches(flint):
    # The micro-batch gate's whole path at a shape that fits the laptop: torch in chunks (the last
    # one uneven), two micro-batches accumulated, and Flint run twice at the retry's placement.
    hook, recorded = Faults({(0, 0): HIGH_BIT}), Recorder()
    gradient_gate_with_retry(flint, "tiny", 4, 16, "mixed", "random", record=recorded, after_backward=hook,
                             chunk=3, micro_batches=2)  # fmt: skip
    assert hook.attempts() == [0] * (2 * len(gate.SEEDS)) + [1] * (2 * len(gate.SEEDS))
    assert len(recorded) == 1 and recorded[0][0] == retry_report.PROPERTY, recorded
    assert re.search(FAULTY_LINE, recorded[0][1]), recorded
    assert "GRADIENT GATE tiny 4x16 mixed weights, random tokens, 2 micro-batches accumulated, chunks of 3" in recorded[0][1]


# -- 2. The assembly equals the kernels called by hand ------------------------------------------

# flint_gemm_tn splits k (here the rows) across blocks, and then needs a workspace, only for an
# output with few tiles and from 2,048 rows up; the tests that call the assembly at 192 rows or
# fewer run with a workspace of 0 bytes. This shape of the tiny config has 2,048 rows.
SPLIT_K_SHAPE = (32, 64)
# The smallest legal model has one class, so every row's softmax is 1, dlogits is exactly 0 and so is
# every gradient: the comparison by hand is then a check of the buffers' sizes and nothing more.
# SMALL_VOCAB is the same shape with 8 classes, whose gradients are not zero.
SMALLEST = G.GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=8, vocab_size=1, padded_vocab_size=8)
SMALL_VOCAB = G.GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=8, vocab_size=8, padded_vocab_size=8)
HAND_CASES = {
    "tiny-1x8": (TINY, 1, 8),
    "tiny-3x64": (TINY, 3, 64),
    "tiny-2x32": (TINY, 2, 32),
    # 2,048 rows: from here flint_gemm_tn splits k across blocks and needs a workspace (see
    # test_the_split_k_shape_needs_a_gemm_tn_workspace), which the smaller shapes never touch.
    "tiny-32x64-with-the-split-k-workspace": (TINY, *SPLIT_K_SHAPE),
    "odd-3x24-in-a-block-of-40": ODD_MODELS["heads-of-24"],
    # The odd model whose gradient gate runs at (64, 8): bit-exactness still covers its own shape.
    "odd-one-head-of-8-2x8": ODD_MODELS["one-head-of-8"],
    "smallest": (SMALLEST, 1, 8),
    "smallest-with-8-classes": (SMALL_VOCAB, 1, 8),
}


def hand_batch(cfg, sequences, length, seed):
    batch = random_batch(cfg, sequences, length, seed)
    if cfg is SMALLEST:  # one class, so every target is 0; the tokens may be any of the 8 rows
        batch.target_tokens = as_tokens(torch.zeros(batch.rows, dtype=torch.int64, device="cuda"))
        batch.tokens = as_tokens(torch.randint(0, 8, (batch.rows,), generator=generator(seed + 1), device="cuda"))
    return batch


@pytest.mark.parametrize("case", HAND_CASES)
def test_the_assembly_equals_the_kernels_called_by_hand(flint, case):
    # The same kernels in the same order from Python, bit for bit, in write mode and, after a
    # second batch, in accumulate mode: the assembly passes the right arguments in the right
    # order, the tied embedding's gradient is (old + output layer) + embedding, and accumulate
    # reaches every kernel.
    cfg, sequences, length = HAND_CASES[case]
    config = backward_config(cfg, sequences, length)
    model = make_model(cfg, 0, "mixed")
    state = state_of(model)
    a, b = hand_batch(cfg, sequences, length, 60), hand_batch(cfg, sequences, length, 61)
    scale = gate_scale(a.rows)
    with Gpt2(flint, config) as m:
        m.load(state)
        forward_backward(m, a, scale)
        first_outputs, first = read_outputs(m, config), read_gradients(m)
        forward_backward(m, b, scale, accumulate=True)
        second = read_gradients(m)
    by_hand_outputs, by_hand_first = compose_backward(flint, config, state, state, a, scale)
    _, by_hand_second = compose_backward(flint, config, state, state, b, scale, old=by_hand_first)
    assert_same_outputs(first_outputs, by_hand_outputs, "by hand", loss=False)
    assert_same_gradients(first, by_hand_first, "write mode, by hand")
    assert_same_gradients(second, by_hand_second, "accumulate mode, by hand")
    # The gradients are not trivially equal: finite, and most of them nonzero.
    assert all(torch.isfinite(g).all() for g in first.values())
    if cfg is SMALLEST:
        assert all(all_equal(int_view(g), 0) for g in first.values())
        return
    assert sum(any_unequal(int_view(g), 0) for g in first.values()) >= len(first) - 2, "too many zero gradients"
    assert any_unequal(int_view(first["wte.weight"]), int_view(second["wte.weight"]))


def weight_gradient_shapes(cfg):
    """The (m, n) of the GEMMs of the weight gradients: wte, then a block's c_attn, its attention's
    c_proj, c_fc and the MLP's c_proj (k is the rows)."""
    W = cfg.n_embd
    return [(cfg.padded_vocab_size, W), (3 * W, W), (W, W), (4 * W, W), (W, 4 * W)]


def test_the_split_k_shape_needs_a_gemm_tn_workspace(flint):
    # Pins what the by-hand and poison tests at this shape cover: that k is split, so a workspace of
    # the wrong size or reused wrongly could fail them. The split depends on the GPU's SM count,
    # which has more SMs on the 2080 Ti, so only that the largest is above 0 is asserted.
    sequences, length = SPLIT_K_SHAPE
    rows = sequences * length
    sizes = {shape: flint.gemm_tn_workspace(*shape, rows) for shape in weight_gradient_shapes(TINY)}
    print(f"TN WORKSPACE at {rows} rows, by (m, n): {sizes}")
    assert max(sizes.values()) > 0, f"flint_gemm_tn needs no workspace at {rows} rows on this GPU: {sizes}"


# -- 3. Exact properties ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tiny", "gpt2"])
@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_gradients_are_the_parameters_in_name_order_and_size(flint, name, separate):
    cfg = G.CONFIGS[name]
    with Gpt2(flint, backward_config(cfg, 1, 8, separate)) as m:
        parameters, gradients = m.parameters(), m.gradients()
    assert len(gradients) == len(parameters) == 2 + 12 * cfg.n_layer + 2
    for (p_name, p_at, p_rows, p_cols), (g_name, g_at, g_rows, g_cols) in zip(parameters, gradients, strict=True):
        assert (g_name, g_rows, g_cols) == (p_name, p_rows, p_cols)
        assert g_at is not None and g_at % ALIGNMENT == 0, g_name
    addresses = [a for _, a, _, _ in parameters + gradients]
    assert len(set(addresses)) == len(addresses), "a gradient shares an address with a parameter or another gradient"
    assert gradients[0][0] == "wte.weight" and gradients[0][2] == cfg.padded_vocab_size


def expected_bytes_backward(flint, layers, heads, width, block_size, vocab, padded_vocab, sequences, length):
    """The device bytes of one allocation of a model with backward, summed in Python: every
    buffer rounded up to 256 bytes, listed by hand from the backward pass's spec, with the two
    workspaces from the bindings' own size functions."""
    rows, head_width, batches = sequences * length, width // heads, sequences * heads
    heads_layout = sequences * heads * length * head_width * 2
    scores = batches * length * length * 2
    masters = [padded_vocab * width * 4, block_size * width * 4]  # wte, wpe
    for _ in range(layers):
        masters += [width * 4] * 2  # ln_1
        masters += [3 * width * width * 4, 3 * width * 4]  # attn.c_attn
        masters += [width * width * 4, width * 4]  # attn.c_proj
        masters += [width * 4] * 2  # ln_2
        masters += [4 * width * width * 4, 4 * width * 4]  # mlp.c_fc
        masters += [width * 4 * width * 4, width * 4]  # mlp.c_proj
    masters += [width * 4] * 2  # ln_f
    buffers = masters + masters  # the gradients: one per master, the same size
    buffers += [padded_vocab * width * 2]  # wte's FP16 copy
    for _ in range(layers):
        buffers += [3 * width * width * 2, width * width * 2, 4 * width * width * 2, width * 4 * width * 2]
    # the residual stream: each block's x_in and x_mid, and the last block's output
    buffers += [rows * width * 4] * (2 * layers + 1)
    for _ in range(layers):
        buffers += [rows * width * 2, rows * 4, rows * 4]  # h1, mean, rstd
        buffers += [heads_layout] * 3 + [scores]  # q, k, v, p
        buffers += [rows * width * 2]  # attention's merged output
        buffers += [rows * width * 2, rows * 4, rows * 4]  # h2, mean, rstd
        buffers += [rows * 4 * width * 2] * 2  # gelu, pre
    buffers += [rows * width * 2, rows * 4, rows * 4]  # ln_f's output, mean, rstd
    buffers += [rows * 3 * width * 2, scores, heads_layout]  # the shared scratch: qkv rows, S, the heads' output
    buffers += [rows * width * 4, rows * width * 2, rows * width * 2, rows * 4 * width * 2]  # dres, its FP16 copy, dh, dpre
    buffers += [heads_layout] * 3  # dq, dk, dv
    shared = [flint.bias_gradient_workspace(rows, cols) for cols in (width, 3 * width, 4 * width)]
    shared += [flint.layernorm_backward_workspace(rows, width),
               flint.embedding_backward_workspace(sequences, length, padded_vocab, width)]  # fmt: skip
    tn = [flint.gemm_tn_workspace(m, n, rows) for m, n in
          ((padded_vocab, width), (3 * width, width), (width, width), (4 * width, width), (width, 4 * width))]  # fmt: skip
    buffers += [max(shared), max(tn)]
    buffers += [rows * padded_vocab * 2] * 2 + [rows * 4]  # logits, dlogits, row_loss
    return sum(up(b) for b in buffers)


BACKWARD_BYTES = {
    "tiny-2x32": config_fields(TINY, 2, 32),
    "tiny-5x8": config_fields(TINY, 5, 8),
    "gpt2-1x8": config_fields(GPT2, 1, 8),
    "gpt2-16x1024": config_fields(GPT2, 16, 1024),
    "odd": ODD,
    "layers-1024": LIMIT_CONFIGS["layers-1024"],
    "smallest": LIMIT_CONFIGS["smallest"],
    # The limits: the most rows (64 x 1024 = 65,536), and the widest model the embedding's backward
    # pass allows (length x width up to 16,776,960).
    "gpt2-64x1024": config_fields(GPT2, 64, 1024),
    "rows-65536-in-short-sequences": dict(layers=2, heads=2, width=128, block_size=64, vocab=50257,
                                          padded_vocab=50304, sequences=8192, length=8),
    "width-16384-length-1016": dict(layers=2, heads=128, width=16384, block_size=1024, vocab=65536,
                                    padded_vocab=65536, sequences=2, length=1016),
}  # fmt: skip


@pytest.mark.parametrize("fields", BACKWARD_BYTES.values(), ids=list(BACKWARD_BYTES))
def test_bytes_with_backward_equals_an_independent_sum(flint, fields):
    # Python's ints don't wrap, so a match shows that the C++ sizes don't either; the workspaces
    # are asked of the current GPU, as flint_gpt2_bytes asks them.
    with_backward = bytes_of(flint, Gpt2Config(**fields, backward=1))
    assert with_backward == expected_bytes_backward(flint, **fields)
    without = bytes_of(flint, Gpt2Config(**fields))
    assert without == expected_bytes(**fields), "a model without backward changed size"
    assert with_backward > without
    print(f"BYTES backward {with_backward} = {with_backward / GiB:.3f} GiB, forward only {without / GiB:.3f} GiB")


@pytest.mark.parametrize("fields", [config_fields(TINY, 2, 32), ODD], ids=["tiny-2x32", "odd"])
@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_buffers_with_backward_lie_inside_the_bytes_and_do_not_overlap(flint, fields, separate):
    config = Gpt2Config(**fields, separate_buffers=int(separate), backward=1)
    total = expected_bytes_backward(flint, **fields)
    rows, padded = fields["sequences"] * fields["length"], fields["padded_vocab"]
    with Gpt2(flint, config) as m:
        parameters, gradients = m.parameters(), m.gradients()
        logits_at, dlogits_at, loss_at = m.outputs()
    regions = [(f"{n}", a, r * c * 4) for n, a, r, c in parameters]
    regions += [(f"gradient of {n}", a, r * c * 4) for n, a, r, c in gradients]
    regions += [("logits", logits_at, rows * padded * 2), ("dlogits", dlogits_at, rows * padded * 2),
                ("row_loss", loss_at, rows * 4)]  # fmt: skip
    regions.sort(key=lambda r: r[1])
    for (name_a, start_a, size_a), (name_b, start_b, _) in zip(regions, regions[1:]):
        assert start_a + size_a <= start_b, f"{name_a} and {name_b} overlap"
    if separate:
        return
    base = parameters[0][1]
    assert base % ALIGNMENT == 0 and min(r[1] for r in regions) == base
    # The parameters come first, their gradients right after them, and the outputs last.
    assert gradients[0][1] == base + sum(up(r * c * 4) for _, _, r, c in parameters)
    assert loss_at + up(rows * 4) - base == total == bytes_of(flint, config)
    for name_, start, size in regions:
        assert base <= start and start + size <= base + total, name_
        assert start % ALIGNMENT == 0, name_


@pytest.mark.parametrize("name, sequences, length", [("tiny", 3, 64), ("gpt2", 2, 256)], ids=str)
def test_one_allocation_and_separate_buffers_give_the_same_gradient_bits(flint, name, sequences, length):
    cfg = G.CONFIGS[name]
    state = state_of(make_model(cfg, 3, "mixed"))
    batch = random_batch(cfg, sequences, length, 4)
    scale = gate_scale(batch.rows)
    one_outputs, one = run_backward_model(flint, backward_config(cfg, sequences, length), state, batch, scale)
    separate_config = backward_config(cfg, sequences, length, True)
    separate_outputs, separate = run_backward_model(flint, separate_config, state, batch, scale)
    assert_same_outputs(one_outputs, separate_outputs, "separate_buffers")
    assert_same_gradients(one, separate, "separate_buffers")


@pytest.mark.parametrize("cfg, sequences, length", [(TINY, 3, 64), ODD_MODELS["heads-of-24"]], ids=["tiny-3x64", "odd-3x24"])
def test_a_model_with_backward_gives_the_forward_passes_bits(flint, cfg, sequences, length):
    state = state_of(make_model(cfg, 5, "mixed"))
    batch = random_batch(cfg, sequences, length, 6)
    scale = gate_scale(batch.rows)
    config = backward_config(cfg, sequences, length)
    without = run_model(flint, gpt2_config(cfg, sequences, length), state, batch, scale)
    with Gpt2(flint, config) as m:
        m.load(state)
        m.forward(batch.tokens, batch.target_tokens, scale)
        forward_only = read_outputs(m, config)
        m.backward(batch.tokens)
        after_backward = read_outputs(m, config)
    assert_same_outputs(forward_only, without, "forward of a model with backward")
    assert_same_outputs(after_backward, without, "outputs after backward")


@pytest.mark.parametrize("sequences, length", [(3, 32), SPLIT_K_SHAPE], ids=["3x32", "32x64-with-the-split-k-workspace"])
def test_run_twice_is_bitwise_identical_with_every_gradient_poisoned_in_between(flint, tiny_model, sequences, length):
    # Length 32 of a block of 64: wpe's rows 32..63 are never seen, and write mode must make them
    # +0 over whatever was there, while accumulate mode leaves them be (at 64 there are none). The
    # second shape has the 2,048 rows from which flint_gemm_tn splits k and needs its workspace.
    config = backward_config(TINY, sequences, length)
    batch = random_batch(TINY, sequences, length, 90)
    scale = gate_scale(batch.rows)
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        forward_backward(m, batch, scale)
        first = read_gradients(m)
        assert all_equal(int_view(first["wpe.weight"])[length:], 0), "wpe's unseen rows are not +0 in write mode"
        assert any_unequal(int_view(first["wpe.weight"])[:length], 0)
        poison_gradients(m)
        forward_backward(m, batch, scale)
        second = read_gradients(m)
        assert_same_gradients(first, second, "second pass over poisoned gradients")
        # Accumulate: the unseen rows are left as they were (1.0 here), and every other gradient
        # is the old value plus this pass's, which is exactly twice for all but wte.
        tail = torch.ones((config.block_size - length) * config.width, dtype=torch.float32, device="cuda")
        wpe_at = {n: a for n, a, _, _ in m.gradients()}["wpe.weight"]
        if tail.numel():
            m.flint.copy(wpe_at + length * config.width * 4, tail, tail.numel() * 4)
        torch.cuda.synchronize()
        forward_backward(m, batch, scale, accumulate=True)
        third = read_gradients(m)
        assert all_equal(third["wpe.weight"][length:], 1.0), "accumulate changed wpe's unseen rows"
        third["wpe.weight"][length:] = 0  # the unseen rows were 1.0, not the first pass's +0
        assert exactly_twice(first, third, skip=("wte.weight",)) == [], "accumulate is not old + new"
        wte = first["wte.weight"]
        assert torch.allclose(third["wte.weight"], 2 * wte, rtol=1e-5, atol=1e-6 * float(wte.abs().max()))
        # Write mode over those: back to the first pass's bits, the unseen rows +0 again.
        forward_backward(m, batch, scale)
        assert_same_gradients(first, read_gradients(m), "write mode after accumulating")


def test_accumulate_is_any_nonzero_value(flint, tiny_model):
    config = backward_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 91)
    scale = gate_scale(batch.rows)
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        m.forward(batch.tokens, batch.target_tokens, scale)
        m.backward(batch.tokens)
        m.backward(batch.tokens, accumulate=True)
        by_one = read_gradients(m)
        m.backward(batch.tokens)
        tokens, stream = flint_lib.pointer(batch.tokens), flint_lib.stream_handle(None)
        flint.check(flint.lib.flint_gpt2_backward(m.handle, tokens, 7, stream))
        by_seven = read_gradients(m)
    assert_same_gradients(by_one, by_seven, "accumulate = 7 against 1")


@pytest.mark.parametrize("with_nonzero_padded_rows", [False, True], ids=["zero-padded-rows", "nonzero-padded-rows"])
def test_wtes_padded_rows_have_exactly_zero_gradient(flint, tiny_model, with_nonzero_padded_rows):
    # Tokens are below vocab, so no row of the embedding's gradient reaches the padded rows, and
    # dlogits is +0 there: the output layer's part is +0 too, and the sum stays +0 when
    # accumulated. The padded rows of the weights are nonzero in one of the runs.
    config = backward_config(TINY, 3, 64)
    batch = random_batch(TINY, 3, 64, 92)
    scale = gate_scale(batch.rows)
    state = state_of(tiny_model)
    if with_nonzero_padded_rows:
        state = with_padded_rows(state, 93)
    vocab = TINY.vocab_size
    with Gpt2(flint, config) as m:
        m.load(state)
        wte = m.read(m.parameters()[0][1], (TINY.padded_vocab_size, TINY.n_embd), torch.float32)
        assert any_unequal(int_view(wte)[vocab:], 0) == with_nonzero_padded_rows
        poison_gradients(m)
        forward_backward(m, batch, scale)
        written = read_gradients(m)["wte.weight"]
        forward_backward(m, batch, scale, accumulate=True)
        accumulated = read_gradients(m)["wte.weight"]
    for what, wte in (("written", written), ("accumulated", accumulated)):
        assert all_equal(int_view(wte)[vocab:], 0), f"wte's padded rows' gradient is not +0 when {what}"
        assert any_unequal(int_view(wte)[:vocab], 0), what


def test_backward_changes_neither_the_weights_nor_the_tokens_nor_the_outputs(flint, tiny_model):
    config = backward_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 94)
    scale = gate_scale(batch.rows)
    tokens, targets = batch.tokens.clone(), batch.target_tokens.clone()
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        m.forward(batch.tokens, batch.target_tokens, scale)
        outputs, parameters = read_outputs(m, config), read_parameters(m)
        m.backward(batch.tokens)
        m.backward(batch.tokens, accumulate=True)
        assert_same_outputs(read_outputs(m, config), outputs, "outputs after backward")
        for name, before in parameters.items():
            assert_same_bits(read_parameters(m)[name], before, f"master weight {name} after backward")
        # The FP16 copies, the activations and the rest: a forward pass after the backward passes
        # gives the same bits, as from a model that never ran them.
        m.forward(batch.tokens, batch.target_tokens, scale)
        assert_same_outputs(read_outputs(m, config), outputs, "a forward pass after backward")
    assert_same_bits(batch.tokens.view(torch.float16), tokens.view(torch.float16), "tokens")
    assert_same_bits(batch.target_tokens.view(torch.float16), targets.view(torch.float16), "targets")


def test_backward_twice_after_one_forward_pass_gives_the_same_bits(flint, tiny_model):
    # It reads only the forward pass's buffers and its own scratch, none of which it changes.
    config = backward_config(TINY, 3, 64)
    batch = random_batch(TINY, 3, 64, 95)
    scale = gate_scale(batch.rows)
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        m.forward(batch.tokens, batch.target_tokens, scale)
        m.backward(batch.tokens)
        first = read_gradients(m)
        m.backward(batch.tokens)
        assert_same_gradients(first, read_gradients(m), "second backward pass")
        poison_gradients(m)
        m.backward(batch.tokens)
        assert_same_gradients(first, read_gradients(m), "third backward pass over poisoned gradients")
        m.backward(batch.tokens, accumulate=True)
        assert exactly_twice(first, read_gradients(m), skip=("wte.weight",)) == []


def test_the_gradients_double_with_the_scale(flint):
    # dlogits is scale * (p - one-hot) rounded to FP16, and the backward pass is linear in it, so
    # doubling the scale doubles every FP16 value and every FP32 sum exactly, as long as no value
    # overflows or is subnormal: a subnormal value rounds on a grid that doesn't double. Whether
    # some value is subnormal depends on the data, and on the GPU, whose last bits of a forward
    # pass can differ: at (1, 8) at the init it held for all 8 seeds tried on the laptop and
    # failed for 1 of 8 at the mix and for 1 of 8 at (1, 16), and at larger shapes some value or
    # other is always subnormal. So exactness can't be promised for any shape, and what is asserted
    # everywhere is that the doubling holds to FP16's subnormal rounding, far below the 1e-6 of
    # the largest value that is the bound (the laptop measured 4e-9 to 2e-8 where inexact). How
    # many gradients were exactly doubled is printed.
    cases = [((1, 8), "init"), ((1, 8), "init"), ((2, 32), "mixed"), ((3, 64), "init")]
    for seed, ((sequences, length), weights) in enumerate(cases):
        state = state_of(make_model(TINY, seed, weights))
        batch = random_batch(TINY, sequences, length, 96 + seed)
        config = backward_config(TINY, sequences, length)
        scale = gate_scale(batch.rows)
        with Gpt2(flint, config) as m:
            m.load(state)
            grads = []
            for s in (scale, 2 * scale):
                forward_backward(m, batch, s)
                grads.append(read_gradients(m))
                dlogits = m.read(m.outputs()[1], (batch.rows, config.padded_vocab), torch.float16)
                assert torch.isfinite(dlogits.float()).all(), "dlogits overflowed"
        assert all(torch.isfinite(g).all() for g in grads[1].values())
        worst = max(float((2 * grads[0][n] - grads[1][n]).abs().max() / grads[1][n].abs().max()) for n in grads[0])
        print(f"SCALE x2 at ({sequences}, {length}) {weights}: {len(grads[0]) - len(exactly_twice(grads[0], grads[1]))} of "
              f"{len(grads[0])} gradients exactly doubled, worst max difference {worst:.2e} of the largest value")  # fmt: skip
        assert worst < 1e-6, ((sequences, length), weights, worst)


def test_backward_reads_the_last_forward_passs_activations(flint, tiny_model):
    config = backward_config(TINY, 2, 32)
    a, b = random_batch(TINY, 2, 32, 97), random_batch(TINY, 2, 32, 98)
    scale = gate_scale(a.rows)
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        forward_backward(m, a, scale)
        only_a = read_gradients(m)
        forward_backward(m, b, scale)
        only_b = read_gradients(m)
        # A forward pass on B between A's forward and backward: the activations and dlogits are B's,
        # the tokens A's (which only the embedding's backward pass reads).
        m.forward(a.tokens, a.target_tokens, scale)
        m.forward(b.tokens, b.target_tokens, scale)
        m.backward(a.tokens)
        crossed = read_gradients(m)
        # And with B's tokens it is B's backward pass, whatever came before.
        m.forward(a.tokens, a.target_tokens, scale)
        m.forward(b.tokens, b.target_tokens, scale)
        m.backward(b.tokens)
        assert_same_gradients(read_gradients(m), only_b, "forward A, forward B, backward with B's tokens")
    unchanged = [n for n in only_a if not any_unequal(int_view(crossed[n]), int_view(only_a[n]))]
    assert unchanged == [], f"the gradients of {unchanged} did not change when the forward pass was B's"
    # Only wte's gradient depends on the tokens, through the embedding's backward pass: everything
    # else is B's.
    for n in only_b:
        if n != "wte.weight":
            assert_same_bits(crossed[n], only_b[n], f"gradient of {n} after the crossed passes")
    assert any_unequal(int_view(crossed["wte.weight"]), int_view(only_b["wte.weight"]))


@pytest.mark.parametrize("name", ["blocks.0.attn.c_attn.weight", "blocks.0.attn.c_proj.weight",
                                  "blocks.1.mlp.c_fc.weight", "blocks.1.mlp.c_proj.weight"])  # fmt: skip
def test_a_changed_linear_weight_has_no_effect_on_the_gradients_until_a_sync(flint, tiny_model, name):
    # The GEMMs read the FP16 copies, so the gradients are those of the model as it was when the
    # copies were made, until sync_weights makes new ones.
    config = backward_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 99)
    scale = gate_scale(batch.rows)
    state = state_of(tiny_model)
    changed = clone_state(state)
    changed[name] = changed_weight(state[name], 100)
    _, fresh = run_backward_model(flint, config, changed, batch, scale)
    with Gpt2(flint, config) as m:
        m.load(state)
        forward_backward(m, batch, scale)
        before = read_gradients(m)
        write_param(m, name, changed[name])
        forward_backward(m, batch, scale)
        unsynced = read_gradients(m)
        m.sync_weights()
        forward_backward(m, batch, scale)
        synced = read_gradients(m)
    assert_same_gradients(unsynced, before, "after changing the master without a sync")
    assert_same_gradients(synced, fresh, "after the sync, against a fresh model with the changed weights")
    assert [n for n in before if any_unequal(int_view(synced[n]), int_view(before[n]))], "the change had no effect at all"


def test_wtes_gradient_takes_two_paths_the_embedding_reads_dx_and_the_output_layer_the_copy(flint, tiny_model):
    # wte's FP16 copy is the output layer's: dh = dlogits wte16 is the only place backward reads
    # it. Without a sync the gradients are those of the changed masters (the forward pass's
    # embedding) with the old copy, which the kernels called by hand give.
    config = backward_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 101)
    scale = gate_scale(batch.rows)
    state = state_of(tiny_model)
    changed = clone_state(state)
    new_wte = changed_weight(state["wte.weight"], 102, 0.1)
    new_wte[TINY.vocab_size :] = 0  # the padded rows stay zero
    changed["wte.weight"] = new_wte
    _, fresh = run_backward_model(flint, config, changed, batch, scale)
    with Gpt2(flint, config) as m:
        m.load(state)
        forward_backward(m, batch, scale)
        before = read_gradients(m)
        write_param(m, "wte.weight", new_wte)
        forward_backward(m, batch, scale)
        unsynced = read_gradients(m)
        m.sync_weights()
        forward_backward(m, batch, scale)
        synced = read_gradients(m)
    differ = lambda a, b: [n for n in a if any_unequal(int_view(a[n]), int_view(b[n]))]  # noqa: E731
    assert differ(unsynced, before), "the embedding didn't see the change"
    assert differ(unsynced, fresh), "the output layer saw the change"
    _, mixed = compose_backward(flint, config, changed, state, batch, scale)
    assert_same_gradients(unsynced, mixed, "unsynced wte, against the changed masters with the old copies")
    assert_same_gradients(synced, fresh, "after the sync, against a fresh model with the changed weights")


@pytest.mark.parametrize("name", ["wpe.weight", "blocks.0.ln_1.weight", "blocks.0.ln_1.bias", "blocks.1.attn.c_proj.bias",
                                  "blocks.0.mlp.c_fc.bias", "ln_f.weight", "ln_f.bias"])  # fmt: skip
def test_fp32_parameters_take_effect_on_the_gradients_without_a_sync(flint, tiny_model, name):
    config = backward_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 103)
    scale = gate_scale(batch.rows)
    state = state_of(tiny_model)
    changed = clone_state(state)
    changed[name] = changed_weight(state[name], 104, 0.1)
    _, fresh = run_backward_model(flint, config, changed, batch, scale)
    with Gpt2(flint, config) as m:
        m.load(state)
        forward_backward(m, batch, scale)
        before = read_gradients(m)
        write_param(m, name, changed[name])
        forward_backward(m, batch, scale)
        unsynced = read_gradients(m)
    assert_same_gradients(unsynced, fresh, "without a sync, against a fresh model")
    assert [n for n in before if any_unequal(int_view(unsynced[n]), int_view(before[n]))]


def test_two_live_models_one_with_backward_and_one_without_do_not_disturb_each_other(flint, tiny_model, tiny_model_b):
    config_a, config_b = backward_config(TINY, 2, 32), gpt2_config(TINY, 3, 64, separate=True)
    batch_a, batch_b = random_batch(TINY, 2, 32, 105), random_batch(TINY, 3, 64, 106)
    other_b = random_batch(TINY, 3, 64, 107)
    scale_a, scale_b = gate_scale(batch_a.rows), gate_scale(batch_b.rows)
    solo_a = run_backward_model(flint, config_a, state_of(tiny_model), batch_a, scale_a)
    solo_b = run_model(flint, config_b, state_of(tiny_model_b), batch_b, scale_b)
    solo_other_b = run_model(flint, config_b, state_of(tiny_model_b), other_b, scale_b)
    with Gpt2(flint, config_a) as a, Gpt2(flint, config_b) as b:
        a.load(state_of(tiny_model))
        b.load(state_of(tiny_model_b))
        regions = [(f"A's {n}", at, r * c * 4) for n, at, r, c in a.parameters() + a.gradients()]
        regions += [(f"B's {n}", at, r * c * 4) for n, at, r, c in b.parameters()]
        spans = sorted(regions, key=lambda r: r[1])
        for (n1, s1, z1), (n2, s2, _) in zip(spans, spans[1:]):
            assert s1 + z1 <= s2, f"{n1} and {n2} overlap"
        forward_backward(a, batch_a, scale_a)
        b.forward(batch_b.tokens, batch_b.target_tokens, scale_b)
        # B ran after A's backward pass, and A's gradients and outputs are still its own.
        assert_same_gradients(read_gradients(a), solo_a[1], "A's gradients after B ran")
        assert_same_outputs(read_outputs(a, config_a), solo_a[0], "A's outputs after B ran")
        assert_same_outputs(read_outputs(b, config_b), solo_b, "B")
        b.forward(other_b.tokens, other_b.target_tokens, scale_b)
        forward_backward(a, batch_a, scale_a)
        assert_same_gradients(read_gradients(a), solo_a[1], "A's gradients after B's second pass")
        assert_same_outputs(read_outputs(b, config_b), solo_other_b, "B on other tokens")
    # And destroying one leaves the other working.
    with Gpt2(flint, config_a) as a:
        a.load(state_of(tiny_model))
        with Gpt2(flint, config_b) as b:
            b.load(state_of(tiny_model_b))
            b.forward(batch_b.tokens, batch_b.target_tokens, scale_b)
            torch.cuda.synchronize()  # flint.h: a model's work must finish before it's destroyed
        forward_backward(a, batch_a, scale_a)
        assert_same_gradients(read_gradients(a), solo_a[1], "A after B was destroyed")


def test_creating_and_destroying_a_model_with_backward_many_times_leaks_no_device_memory(flint):
    # GPT-2 at (1, 256) with backward takes about 2 GB. The driver's free memory, on a GPU that
    # desktop apps share, moves by itself, so the baseline's own noise is measured and printed, and
    # the tolerance (LEAK_TOLERANCE) is far above it and far below one model's size.
    cfg = GPT2
    config = backward_config(cfg, 1, 256)
    model = make_model(cfg, 0, "init")
    batch = random_batch(cfg, 1, 256, 108)
    size = bytes_of(flint, config)
    state = state_of(model)

    def cycle():
        with Gpt2(flint, config) as m:
            m.load(state)
            forward_backward(m, batch, gate_scale(batch.rows))
            m.mean_loss()
            torch.cuda.synchronize()
            return torch.cuda.mem_get_info()[0]

    cycle()  # the first pass loads the kernels' modules, which stay
    baseline = [free_memory() for _ in range(5)]
    noise = max(baseline) - min(baseline)
    during = []
    for _ in range(5):
        during.append(cycle())
    after = free_memory()
    drop_during = baseline[0] - min(during)
    leak = baseline[0] - after
    print(f"LEAK backward model {size / MiB:.0f} MiB: free-memory noise {noise / MiB:.1f} MiB, drop while alive "
          f"{drop_during / MiB:.0f} MiB, after 5 create/destroy cycles {leak / MiB:.1f} MiB below the start "
          f"(tolerance {LEAK_TOLERANCE / MiB:.0f} MiB)")  # fmt: skip
    assert drop_during > 0.9 * size, "free memory didn't drop by the model's size while it lived: the measure is blind"
    if noise > size / 4:
        pytest.skip(f"free memory moved {noise / MiB:.0f} MiB by itself: too noisy to measure a leak")
    assert leak < LEAK_TOLERANCE, f"{leak / MiB:.0f} MiB of device memory were not given back"


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_a_forward_and_backward_pass_on_a_side_stream_give_the_same_bits(flint, tiny_model, separate):
    config = backward_config(TINY, 3, 64, separate)
    a, b = random_batch(TINY, 3, 64, 109), random_batch(TINY, 3, 64, 110)
    scale = gate_scale(a.rows)
    state = state_of(tiny_model)

    def two_micro_batches(m, stream=None):
        forward_backward(m, a, scale, stream=stream)
        forward_backward(m, b, scale, accumulate=True, stream=stream)
        return read_gradients(m, stream)

    with Gpt2(flint, config) as m:
        m.load(state)
        default = two_micro_batches(m)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with Gpt2(flint, config) as m:
        # The sleep keeps the stream busy: a step that ran on another stream would race ahead of
        # the weights' copy and give other bits.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.load(state, stream=stream)
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.sync_weights(stream)
            side = two_micro_batches(m, stream)
    stream.synchronize()
    assert_same_gradients(side, default, "side stream")


# -- 4. Argument errors and limits ---------------------------------------------------------------


def test_backward_and_gradient_fail_on_a_model_without_backward(flint, tiny_model):
    config = gpt2_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 111)
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        m.forward(batch.tokens, batch.target_tokens, 0.5)
        before = read_outputs(m, config)
        for call in (lambda: m.backward(batch.tokens), lambda: m.backward(batch.tokens, accumulate=True)):
            with pytest.raises(flint_lib.FlintError) as error:
                call()
            assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
            assert "flint_gpt2_backward" in str(error.value) and "without backward" in str(error.value), str(error.value)
        with pytest.raises(flint_lib.FlintError) as error:
            m.gradients()
        assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
        assert "flint_gpt2_gradient" in str(error.value) and "without backward" in str(error.value), str(error.value)
        # A valid call still works, and nothing changed.
        assert_same_outputs(read_outputs(m, config), before, "outputs after the rejected calls")
        m.forward(batch.tokens, batch.target_tokens, 0.5)
        assert_same_outputs(read_outputs(m, config), before, "a forward pass after the rejected calls")


def test_the_nulls_of_a_null_backward_model(flint):
    lib = flint.lib
    info = Gpt2ParameterInfo()
    assert lib.flint_gpt2_backward(None, None, 0, None) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_gpt2_backward" in flint.last_error() and "model is null" in flint.last_error(), flint.last_error()
    assert lib.flint_gpt2_gradient(None, 0, ctypes.byref(info)) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_gpt2_gradient" in flint.last_error() and "null" in flint.last_error(), flint.last_error()
    assert info.data is None


def test_gradient_checks_the_index_and_the_info_pointer(flint):
    with Gpt2(flint, backward_config(TINY, 1, 8)) as m:
        count = flint.lib.flint_gpt2_parameter_count(m.handle)
        assert count == 2 + 12 * TINY.n_layer + 2
        info = Gpt2ParameterInfo()
        for index in (count, count + 1, 2**32, 2**64 - 1):
            assert flint.lib.flint_gpt2_gradient(m.handle, index, ctypes.byref(info)) == flint_lib.ERROR_INVALID_ARGUMENT
            assert f"index ({index}) must be below {count}" in flint.last_error(), flint.last_error()
            assert "flint_gpt2_gradient" in flint.last_error()
        assert info.data is None, "info was written on an error"
        assert flint.lib.flint_gpt2_gradient(m.handle, count - 1, ctypes.byref(info)) == flint_lib.OK
        assert info.name == b"ln_f.bias" and (info.rows, info.cols) == (1, TINY.n_embd) and info.data
        assert flint.lib.flint_gpt2_gradient(m.handle, 0, None) == flint_lib.ERROR_INVALID_ARGUMENT
        assert "must not be null" in flint.last_error() and "flint_gpt2_gradient" in flint.last_error()
        assert flint.lib.flint_gpt2_gradient(m.handle, 0, ctypes.byref(info)) == flint_lib.OK
        assert info.name == b"wte.weight" and (info.rows, info.cols) == (TINY.padded_vocab_size, TINY.n_embd)


class BadBackward:
    """A tiny model with backward after a forward pass, its gradients filled with a pattern, to
    show that a rejected backward pass changes nothing, and that a valid one works after it."""

    def __init__(self, flint, state):
        self.flint = flint
        self.config = backward_config(TINY, 2, 32)
        self.rows = 64
        self.batch = random_batch(TINY, 2, 32, 112)
        self.scale = gate_scale(self.rows)
        _, self.expected = run_backward_model(flint, self.config, state, self.batch, self.scale)
        self.m = Gpt2(flint, self.config)
        self.m.load(state)
        self.m.forward(self.batch.tokens, self.batch.target_tokens, self.scale)
        self.outputs = read_outputs(self.m, self.config)
        self.logits_at, self.dlogits_at, self.loss_at = self.m.outputs()
        padded = TINY.padded_vocab_size
        self.sizes = {"logits": self.rows * padded * 2, "dlogits": self.rows * padded * 2, "row_loss": self.rows * 4}
        self.addresses = {"logits": self.logits_at, "dlogits": self.dlogits_at, "row_loss": self.loss_at}
        self.gradients = self.m.gradients()
        poison_gradients(self.m)

    def untouched(self):
        torch.cuda.synchronize()
        if not all(all_equal(int_view(g), PATTERN32) for g in read_gradients(self.m).values()):
            return False
        out = read_outputs(self.m, self.config, mean_loss=False)
        return all(
            all_equal(int_view(now), int_view(before))
            for now, before in zip((out.logits, out.dlogits, out.row_loss), self.outputs[:3])
        )

    def expect_rejected(self, reason, tokens, accumulate=False):
        with pytest.raises(flint_lib.FlintError) as error:
            self.m.backward(tokens, accumulate)
        assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
        for text in ("flint_gpt2_backward", reason):
            assert text in str(error.value) and text in self.flint.last_error(), (text, str(error.value))
        assert self.untouched(), f"a rejected backward pass ({reason}) changed the gradients or the outputs"
        # A valid call works after the error, and gives the bits of a clean run.
        self.m.backward(self.batch.tokens)
        what = f"a valid backward pass after the rejected one ({reason})"
        assert_same_gradients(read_gradients(self.m), self.expected, what)
        poison_gradients(self.m)

    def close(self):
        self.m.close()


@pytest.fixture
def bad_backward(flint, tiny_model):
    bad = BadBackward(flint, state_of(tiny_model))
    yield bad
    bad.close()


def test_backward_rejects_a_null_pointer(bad_backward):
    bad_backward.expect_rejected("tokens is null", None)


@pytest.mark.parametrize("offset", [2, 8])
def test_backward_rejects_a_misaligned_pointer(bad_backward, offset):
    # 2 bytes is as far off as an element goes, and 8 is aligned for 64 bits but not 128.
    room = torch.zeros(bad_backward.rows + 64, dtype=torch.int16, device="cuda")
    bad_backward.expect_rejected("tokens is not 16-byte aligned", room.data_ptr() + offset)


def first_overlapped(bad, address):
    """The first buffer, in the order the checks run (logits, dlogits, row_loss, then the
    gradients in order), that a region of rows * 2 bytes at address overlaps, as the message
    names it."""
    end = address + bad.rows * 2
    for name in ("logits", "dlogits", "row_loss"):
        if address < bad.addresses[name] + bad.sizes[name] and bad.addresses[name] < end:
            return f"{name} and tokens overlap"
    for name, at, rows, cols in bad.gradients:
        if address < at + rows * cols * 4 and at < end:
            return f"tokens and the gradient of {name} overlap"
    raise AssertionError("the region overlaps nothing")


def overlap_start(label, size, rows_bytes):
    """Where a region of rows_bytes starts, relative to a buffer of size bytes: on its start, 16
    bytes in, 16 bytes before its end, and ending 16 bytes into it."""
    return {"same-start": 0, "16-in": 16, "starts-16-before-end": size - 16, "ends-16-in": 16 - rows_bytes}[label]


@pytest.mark.parametrize("label", OVERLAPS)
@pytest.mark.parametrize("output", ["logits", "dlogits", "row_loss"])
def test_backward_rejects_tokens_that_overlap_an_output(bad_backward, output, label):
    b = bad_backward
    address = b.addresses[output] + overlap_start(label, b.sizes[output], b.rows * 2)
    b.expect_rejected(first_overlapped(b, address), address)


@pytest.mark.parametrize("label", OVERLAPS)
@pytest.mark.parametrize("which", ["first", "middle", "last"])
def test_backward_rejects_tokens_that_overlap_a_gradient(bad_backward, which, label):
    b = bad_backward
    index = {"first": 0, "middle": len(b.gradients) // 2, "last": len(b.gradients) - 1}[which]
    name, at, rows, cols = b.gradients[index]
    address = at + overlap_start(label, rows * cols * 4, b.rows * 2)
    expected = first_overlapped(b, address)
    assert expected.endswith("overlap") and "gradient of" in expected, expected
    b.expect_rejected(expected, address)
    if label in ("same-start", "16-in"):  # nothing before the gradient can be involved
        assert expected == f"tokens and the gradient of {name} overlap"


REJECTED_WITH_BACKWARD = {
    "rows-65-x-1024": (["sequences", "length"], dict(sequences=65, length=1024, block_size=1024)),
    "rows-8193-x-8": (["sequences", "length"], dict(sequences=8193, length=8)),
    "rows-with-1-head": (["sequences", "length"], dict(sequences=65535, heads=1, length=8)),
    "length-x-width-1024-x-16384": (["length", "width"],
                                    dict(heads=128, width=16384, length=1024, block_size=1024, sequences=1)),
}  # fmt: skip
ACCEPTED_WITH_BACKWARD = {
    "rows-64-x-1024": dict(sequences=64, length=1024, block_size=1024),
    "rows-8192-x-8": dict(sequences=8192, length=8),
    "length-x-width-1016-x-16384": dict(heads=128, width=16384, length=1016, block_size=1024, sequences=1),
}  # fmt: skip


@pytest.mark.parametrize("case", REJECTED_WITH_BACKWARD, ids=str)
def test_a_config_is_rejected_only_with_backward(flint, case):
    names, fields = REJECTED_WITH_BACKWARD[case]
    # Without backward it is a legal config (bytes only: the models are far too large to create).
    status, value = call_bytes(flint, fields)
    assert status == flint_lib.OK and value > 0, flint.last_error()
    fields = {**fields, "backward": 1}
    status, value = call_bytes(flint, fields)
    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert value == 0xDEAD, "bytes was written on an error"
    message = flint.last_error()
    assert "flint_gpt2_bytes" in message and "with backward" in message and named(message, names, fields), message
    status, model = call_create(flint, fields)
    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert model is None, "create left *model set"
    message = flint.last_error()
    assert "flint_gpt2_create" in message and "with backward" in message and named(message, names, fields), message
    # A valid call works after the error.
    status, value = call_bytes(flint, dict(backward=1))
    assert status == flint_lib.OK and value > 0


@pytest.mark.parametrize("case", ACCEPTED_WITH_BACKWARD, ids=str)
def test_a_legal_backward_config_at_its_limits_passes_validation(flint, case):
    fields = {**GOOD, **ACCEPTED_WITH_BACKWARD[case]}
    fields.pop("separate_buffers")
    status, value = call_bytes(flint, {**fields, "backward": 1})
    assert status == flint_lib.OK, flint.last_error()
    assert value == expected_bytes_backward(flint, **fields)
    print(f"BYTES backward limit {case}: {value / GiB:.1f} GiB")


def test_the_configs_that_backward_does_not_narrow_are_still_rejected_with_the_same_messages(flint):
    # Every kernel limit the other calls bring is checked in the same place, ahead of backward's.
    status, _ = call_bytes(flint, dict(backward=1, width=12))
    assert status == flint_lib.ERROR_INVALID_ARGUMENT and "width (12)" in flint.last_error()
    status, _ = call_bytes(flint, dict(backward=1, layers=0))
    assert status == flint_lib.ERROR_INVALID_ARGUMENT and "layers (0)" in flint.last_error()


# -- 5. The sanitizer ------------------------------------------------------------------------------


@pytest.mark.parametrize("seeds", [(0, 1)], ids=str)
@pytest.mark.parametrize("sequences, length", [(1, 8), (2, 16)], ids=str)
def test_sanitized_tiny_backward_with_separate_buffers(flint, sequences, length, seeds):
    # separate_buffers gives each buffer its own allocation, so memcheck sees each one's
    # bounds. Small shapes keep the sanitized run short.
    gradient_case(flint, "tiny", sequences, length, "mixed", "random", seeds=seeds, separate=True)


def test_sanitized_tiny_backward_accumulates_with_separate_buffers(flint):
    # Accumulate mode reads the old gradients, which write mode never does: two micro-batches, the
    # second accumulated, with memcheck seeing each buffer's bounds and initcheck the reads of what
    # the first wrote.
    gradient_case(flint, "tiny", 1, 8, "mixed", "random", seeds=(0,), separate=True, micro_batches=2)


def test_sanitized_tiny_backward_with_one_allocation(flint):
    config = backward_config(TINY, 2, 16)
    state = state_of(make_model(TINY, 0, "mixed"))
    batch = random_batch(TINY, 2, 16, 120)
    scale = gate_scale(batch.rows)
    one_outputs, one = run_backward_model(flint, config, state, batch, scale)
    separate_outputs, separate = run_backward_model(flint, backward_config(TINY, 2, 16, separate=True), state, batch, scale)
    assert_same_outputs(one_outputs, separate_outputs, "one allocation against separate buffers")
    assert_same_gradients(one, separate, "one allocation against separate buffers")
    gradient_case(flint, "tiny", 2, 16, "mixed", "random", seeds=(0,), separate=False)
