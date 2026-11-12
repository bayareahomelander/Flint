"""flint_gpt2: GPT-2's forward pass, from tokens to logits, each row's loss and the logits'
gradient, assembled in src/api/gpt2.cpp from the kernels the other test files cover.

The whole-model gate is M2's done rule for the forward pass: from identical weights (master
weights rounded to FP16-representable values, so every run starts equal), with PyTorch FP32
(scripts/gpt2_torch.py) as the reference and autocast FP16 as the peer, Flint's error
against the reference is at most 2x the peer's (3x for gradients), by relative L2 and by
max-abs, over 3 seeds (gate.py). It covers
  - the logits below vocab, against the FP32 model's;
  - the mean loss, against the FP64 cross-entropy of the FP32 model's logits, with autocast's
    own model(idx, targets) as the peer: its log_softmax runs in FP16, which is what a model
    trained with autocast gets, so the gate is deliberately against that;
  - dlogits below vocab at scale 2^16 / rows, GradScaler's first loss scale over the mean
    loss, against the FP64 softmax gradient of the FP32 logits, with autograd through
    autocast's cross-entropy on an FP16 leaf as the peer (test_cross_entropy.py's
    autocast_peer). At scale 1 / rows, GPT-2's gradients away from the target are below
    FP16's smallest subnormal and round to 0 in Flint and the peer alike, so a gate there
    compares the target's column and nothing else.
The shapes are the tiny config at 4 shapes, three odd models (other head widths, a length below
block_size, a vocabulary ending mid-chunk) and GPT-2 at (2, 256), (1, 1024) and (8, 64), on
the init and on a mid-training mix (nonzero biases, LayerNorm parameters moved), with random
tokens and, at GPT-2 (1, 1024), real FineWeb-Edu tokens. GPT-2's whole micro-batch,
(16, 1024), runs when the GPU has 16 GiB free (the 2080 Ti); a failed gate there is retried
once, at another placement; a pass on the retry counts only if the two attempts' outputs
differ as one flipped bit of memory would make them, within one sequence per seed, and it is
reported (gate_with_retry, retry_report.py), which tests on the tiny config check. With
FLINT_REQUIRE_MICRO_BATCH=1, as precheck sets it, too little free memory fails that test where
it otherwise skips. The logits, dlogits and references are
made and compared over chunks of rows, so GPT-2's 16,384 x 50,304 stays small. Every case
prints its errors.

The assembly's own tests check what a tolerance can't. The parameters' names, order and sizes
equal gpt2_torch's. flint_gpt2_bytes equals an independent sum in Python, and the buffers of
one allocation lie inside it without overlapping. One allocation and separate_buffers give the
same bits. The same kernels called by hand in Python give the same bits, which shows that the
assembly passes the right arguments in the right order. Then run twice, no state leaking from
one pass to the next, mean_loss as the exact FP64 mean of row_loss, the inputs unchanged,
targets and scale touching only what they should, sync_weights' two paths for wte (the
embedding reads the FP32 master, the output GEMM the FP16 copy), the padded vocabulary, two
models alive at once, create and destroy in a loop without a leak, and streams. The argument
errors and the failed create come last. Under the sanitizer (scripts/sanitize.py) only the
test_sanitized_* tests run: the tiny config at small shapes, with separate_buffers so that
memcheck sees each buffer's bounds, and once as one allocation."""

import ctypes
import gc
import hashlib
import math
import os
import re
import sys
import types
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flint_lib
import gate
import retry_report
from flint_lib import Gpt2, Gpt2Config, Gpt2ParameterInfo
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES, as_int16, generator

sys.path.insert(0, str(flint_lib.ROOT / "scripts"))
import gpt2_torch as G  # noqa: E402

MiB = 1 << 20
GiB = 1 << 30
ALIGNMENT = 256
# References and errors are made over chunks of rows of at most this many logits.
CHUNK_ELEMENTS = 1 << 23
FINEWEB_SHARD = "fineweb-edu-10B/shard_001.bin"
FINEWEB_HEADER_BYTES = 64
# Where the windows of real tokens start (the same as the throwaway real-weights script's).
FINEWEB_WINDOW_START = 50_000_000
FINEWEB_WINDOW_STRIDE = 1_000_000
# GPT-2's micro-batch needs this much free GPU memory: it takes 5.0 GiB, and torch's model,
# references and peer the rest. Measured on the 2080 Ti, a gate took 14.0 GiB and its retry,
# with the padding below, 15.0 GiB; this leaves a margin over that.
MICRO_BATCH_FREE_BYTES = 16 * GiB
# A failed micro-batch gate is retried once with this much GPU memory allocated first (see
# gate_with_retry).
RETRY_PADDING_BYTES = 1 * GiB
# GradScaler's first loss scale. The gates run the loss's kernel at LOSS_SCALE / rows, the
# gradient of the scaled mean loss, where GPT-2's gradients away from the target are normal
# FP16 numbers (about 8e-5 at 16 x 1024, against 1.2e-9 at scale 1 / rows, which rounds to 0).
LOSS_SCALE = 2.0**16
# With this set to 1, as precheck sets it, the micro-batch gate fails where it would skip.
REQUIRE_MICRO_BATCH = "FLINT_REQUIRE_MICRO_BATCH"
# The device memory a created and destroyed model may leave behind, as the driver reports
# it: a shared GPU's free memory moves by tens of MiB on its own (see the leak test).
LEAK_TOLERANCE = 128 * MiB

TINY = G.TINY_CONFIG
GPT2 = G.GPT2_CONFIG


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


# -- Helpers: data, comparing -----------------------------------------------------------


def data_file(relative):
    """data/<relative> in the repo, else ~/flint-data/<relative>, else None. The weights and
    tokens are gitignored and sit in different places on the laptop and the server."""
    for base in (flint_lib.ROOT / "data", Path.home() / "flint-data"):
        path = base / relative
        if path.is_file():
            return path
    return None


def real_tokens(count, window):
    """count real FineWeb-Edu tokens (int64, on the GPU) from one of the shard's windows;
    skips the test if the shard isn't there."""
    path = data_file(FINEWEB_SHARD)
    if path is None:
        pytest.skip(f"{FINEWEB_SHARD} not found in data/ or ~/flint-data/")
    shard = np.memmap(path, dtype="<u2", mode="r", offset=FINEWEB_HEADER_BYTES)
    start = FINEWEB_WINDOW_START + window * FINEWEB_WINDOW_STRIDE
    if start + count > len(shard):
        pytest.skip(f"{FINEWEB_SHARD} has no room for window {window}")
    ids = np.array(shard[start : start + count], dtype=np.int64)
    del shard
    return torch.from_numpy(ids).cuda()


def sha256_of(path, block=16 * MiB):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def int_view(tensor):
    return tensor.view(torch.int16 if tensor.dtype == torch.float16 else torch.int32)


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


def any_unequal(a, b):
    return bool((a != b).any())


# -- Helpers: models, batches, outputs ----------------------------------------------------


def gpt2_config(cfg, sequences, length, separate=False, **overrides):
    """A flint_gpt2_config for gpt2_torch's config and a batch shape."""
    fields = dict(
        layers=cfg.n_layer, heads=cfg.n_head, width=cfg.n_embd, block_size=cfg.block_size,
        vocab=cfg.vocab_size, padded_vocab=cfg.padded_vocab_size, sequences=sequences,
        length=length, separate_buffers=int(separate),
    )
    return Gpt2Config(**{**fields, **overrides})


def make_model(cfg, seed, weights):
    """gpt2_torch's model on the GPU with FP16-representable FP32 weights: the init
    (weights="init") or, with "mixed", a mid-training mix (nonzero biases, LayerNorm
    parameters moved), as the done rule starts all three runs from equal weights."""
    model = G.build_model(cfg, seed=seed, device="cuda")
    g = generator(seed + 100)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if weights == "mixed":
                if name.endswith("bias"):
                    p.add_(0.01 * torch.randn(p.shape, generator=g, device="cuda"))
                elif ".ln_" in name or "ln_f" in name:
                    p.add_(0.05 * torch.randn(p.shape, generator=g, device="cuda"))
            p.copy_(p.half().float())
    return model


def state_of(model):
    return dict(model.named_parameters())


def clone_state(state):
    return {name: p.detach().clone() for name, p in state.items()}


def as_tokens(ids):
    """int64 token ids (below 65,536) as the uint16 bit patterns Flint reads (torch.int16)."""
    return as_int16(ids).to(torch.int16)


class Batch:
    """rows = sequences * length ids: idx (sequences, length) and the next token of each as
    targets (rows), as int64 and as the device tokens Flint reads."""

    def __init__(self, ids, sequences, length):
        self.sequences, self.length, self.rows = sequences, length, sequences * length
        self.idx = ids[:-1].reshape(sequences, length)
        self.targets = ids[1:].contiguous()
        self.tokens = as_tokens(ids[:-1])
        self.target_tokens = as_tokens(ids[1:])


def random_ids(cfg, count, seed):
    return torch.randint(0, cfg.vocab_size, (count,), generator=generator(seed), device="cuda")


def random_batch(cfg, sequences, length, seed):
    return Batch(random_ids(cfg, sequences * length + 1, seed), sequences, length)


Outputs = namedtuple("Outputs", "logits dlogits row_loss loss")


def read_outputs(m, config, stream=None, mean_loss=True):
    """Copies of a model's outputs: logits and dlogits (rows, padded_vocab) FP16, row_loss
    (rows) FP32, and its mean loss."""
    rows = config.sequences * config.length
    logits_at, dlogits_at, loss_at = m.outputs()
    shape = (rows, config.padded_vocab)
    return Outputs(
        m.read(logits_at, shape, torch.float16, 0, stream),
        m.read(dlogits_at, shape, torch.float16, 0, stream),
        m.read(loss_at, (rows,), torch.float32, 0, stream),
        m.mean_loss(stream) if mean_loss else math.nan,
    )


def run_model(flint, config, state, batch, scale):
    """A fresh model, loaded with state, run once; its outputs."""
    with Gpt2(flint, config) as m:
        m.load(state)
        m.forward(batch.tokens, batch.target_tokens, scale)
        return read_outputs(m, config)


def assert_same_outputs(a, b, what, loss=True):
    assert_same_bits(a.logits, b.logits, f"{what}: logits")
    assert_same_bits(a.dlogits, b.dlogits, f"{what}: dlogits")
    assert_same_bits(a.row_loss, b.row_loss, f"{what}: row_loss")
    if loss:
        assert a.loss == b.loss, f"{what}: mean loss {a.loss!r} against {b.loss!r}"


def write_param(m, name, tensor):
    """Overwrites one master weight (without a sync)."""
    for n, address, rows, cols in m.parameters():
        if n == name:
            assert tensor.dtype == torch.float32 and tensor.numel() == rows * cols
            m.flint.copy(address, tensor.contiguous(), rows * cols * 4)
            torch.cuda.synchronize()
            return
    raise KeyError(name)


def free_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


# -- The same kernels called by hand ------------------------------------------------------


def compose_forward(flint, config, state, copy_state, batch, scale):
    """flint_gpt2_forward's sequence of kernels, called one by one from Python on fresh
    buffers: the masters (embedding, biases, LayerNorm parameters) from state and the FP16
    copies of wte and the linear layers' weights from copy_state. With the same state for
    both this is a synced model; with a changed one for state, a model whose weights changed
    without a sync."""
    c = config
    W, H, B, T = c.width, c.heads, c.sequences, c.length
    rows, head_width, batches, padded = B * T, W // H, B * H, c.padded_vocab

    def half(*shape):
        return torch.empty(*shape, dtype=torch.float16, device="cuda")

    def single(*shape):
        return torch.empty(*shape, dtype=torch.float32, device="cuda")

    def fp16_of(name):
        master = copy_state[name].detach().contiguous()
        y = half(master.numel())
        flint.fp32_to_fp16(y, master.reshape(-1), master.numel())
        return y

    def norm(x, prefix):
        y, mean, rstd = half(rows, W), single(rows), single(rows)
        flint.layernorm_forward(
            y, mean, rstd, x, state[f"{prefix}.weight"], state[f"{prefix}.bias"], rows, W
        )
        return y

    score_scale = float(np.float32(1.0) / np.sqrt(np.float32(head_width)))
    x = single(rows, W)
    flint.embedding_forward(x, batch.tokens, state["wte.weight"], state["wpe.weight"], B, T, padded, W)
    for i in range(c.layers):
        p = f"blocks.{i}."
        h = norm(x, p + "ln_1")
        qkv = half(rows, 3 * W)
        flint.gemm_nt_bias(qkv, h, fp16_of(p + "attn.c_attn.weight"), state[p + "attn.c_attn.bias"], rows, 3 * W, W)
        q, k, v = (half(B * H * T * head_width) for _ in range(3))
        for j, part in enumerate((q, k, v)):
            flint.heads_split(part, qkv, B, T, H, head_width, 3 * W, j * W)
        s, probs = half(batches, T, T), half(batches, T, T)
        flint.gemm_nt_batched(s, q, k, batches, T, T, head_width, score_scale)
        flint.causal_softmax_forward(probs, s, batches, T)
        heads_output = half(B * H * T * head_width)
        flint.gemm_nn_batched(heads_output, probs, v, batches, T, head_width, T)
        attn_output = half(rows, W)
        flint.heads_merge(attn_output, heads_output, B, T, H, head_width, W, 0)
        y = single(rows, W)
        flint.gemm_nt_bias_residual(
            y, attn_output, fp16_of(p + "attn.c_proj.weight"), state[p + "attn.c_proj.bias"], x, rows, W, W
        )
        h = norm(y, p + "ln_2")
        gelu, pre = half(rows, 4 * W), half(rows, 4 * W)
        flint.gemm_nt_bias_gelu(gelu, pre, h, fp16_of(p + "mlp.c_fc.weight"), state[p + "mlp.c_fc.bias"], rows, 4 * W, W)
        x = single(rows, W)
        flint.gemm_nt_bias_residual(
            x, gelu, fp16_of(p + "mlp.c_proj.weight"), state[p + "mlp.c_proj.bias"], y, rows, W, 4 * W
        )
    h = norm(x, "ln_f")
    logits = half(rows, padded)
    flint.gemm_nt(logits, h, fp16_of("wte.weight"), rows, padded, W)
    dlogits, row_loss = half(rows, padded), single(rows)
    flint.softmax_cross_entropy(dlogits, row_loss, logits, batch.target_tokens, scale, rows, c.vocab, padded)
    torch.cuda.synchronize()
    return Outputs(logits, dlogits, row_loss, math.nan)


# -- The gate ---------------------------------------------------------------------------


class Accumulator:
    """gate.errors over chunks: relative L2, max-abs and max|reference| of tensors compared
    piece by piece."""

    def __init__(self):
        self.sq_diff = self.sq_ref = self.max_diff = self.max_ref = 0.0

    def add(self, out, ref):
        o, r = out.double(), ref.double()
        d = o - r
        sq_diff, sq_ref, max_diff, max_ref = torch.stack(
            [d.square().sum(), r.square().sum(), d.abs().max(), r.abs().max()]
        ).tolist()
        self.sq_diff += sq_diff
        self.sq_ref += sq_ref
        # A NaN anywhere must stay a NaN, whatever order the chunks come in.
        self.max_diff = math.nan if math.isnan(self.max_diff) or math.isnan(max_diff) else max(self.max_diff, max_diff)
        self.max_ref = math.nan if math.isnan(self.max_ref) or math.isnan(max_ref) else max(self.max_ref, max_ref)

    def errors(self):
        if self.sq_ref == 0.0:
            rel = 0.0 if self.sq_diff == 0.0 else math.inf
        else:
            rel = math.sqrt(self.sq_diff) / math.sqrt(self.sq_ref)
        return gate.Errors(rel, self.max_diff, self.max_ref)


def scalar_errors(value, ref):
    return gate.errors(torch.tensor([value], dtype=torch.float64), torch.tensor([ref], dtype=torch.float64))


class TorchReference:
    """gpt2_torch's model on a batch, as the reference (FP32) and the peer (autocast FP16)
    of its logits, made chunk by chunk from the final LayerNorm's output, and the peer's
    mean loss, model(idx, targets) under autocast. with_reference=False skips the FP32
    pass for callers whose reference is another model."""

    def __init__(self, model, idx, targets, with_reference=True):
        self.model, self.vocab = model, model.config.vocab_size
        width = model.config.n_embd
        with torch.no_grad(), gate.peer_math():
            if with_reference:
                self.hidden32 = model.hidden(idx).reshape(-1, width)
            with torch.autocast("cuda", dtype=torch.float16):
                self.hidden16 = model.hidden(idx).reshape(-1, width)
                self.peer_loss = model(idx, targets).item()

    def reference(self, lo, hi):
        with torch.no_grad(), gate.peer_math():
            return F.linear(self.hidden32[lo:hi], self.model.wte.weight)[:, : self.vocab]

    def peer(self, lo, hi):
        with torch.no_grad(), gate.peer_math(), torch.autocast("cuda", dtype=torch.float16):
            return F.linear(self.hidden16[lo:hi], self.model.wte.weight)[:, : self.vocab]


Measured = namedtuple(
    "Measured",
    "logits loss dlogits off_target flint_loss ref_loss peer_loss top1_flint top1_peer padded_ok fingerprint",
)


def gate_scale(rows):
    """The loss's scale in the gates: GradScaler's first loss scale over the rows' mean loss."""
    return LOSS_SCALE / rows


def row_hashes(bits):
    """One int64 per row of an integer matrix: its values times fixed odd weights, added up
    with wraparound. A row with one changed element always gets another hash (an odd weight
    times a difference below 2^32 is never 0 mod 2^64)."""
    g = torch.Generator(device="cuda").manual_seed(bits.shape[1])
    weights = torch.randint(-(2**62), 2**62, (bits.shape[1],), generator=g, device="cuda") | 1
    return (bits.to(torch.int64) * weights).sum(1)


def measure(m, config, reference, peer, targets, scale, peer_loss):
    """Flint's forward pass (already run on m) against a reference and a peer, over chunks of
    rows. reference(lo, hi) gives the reference's FP32 logits of those rows below vocab, and
    peer(lo, hi) the peer's FP16; targets are int64 on the GPU; scale is the loss scale the
    forward ran with. Returns the (Flint, peer) errors of the logits, the loss, dlogits, and
    dlogits away from the targets (off_target: each row's target column set to 0 in all three).
    A row's target holds nearly all of its gradient's norm (at init the rest is about 0.5% of
    it), so the dlogits gate alone passes a 10% error in every other column."""
    vocab, padded, rows = config.vocab, config.padded_vocab, len(targets)
    logits_at, dlogits_at, _ = m.outputs()
    step = max(1, CHUNK_ELEMENTS // padded)
    scale32 = float(np.float32(scale))
    flint_logits, peer_logits, flint_grad, peer_grad, flint_off, peer_off = (Accumulator() for _ in range(6))
    ref_loss_sum, agree_flint, agree_peer, padded_ok = 0.0, 0, 0, True
    logit_hashes, grad_hashes = [], []
    for lo in range(0, rows, step):
        hi = min(lo + step, rows)
        tgt = targets[lo:hi]
        ref, peer_chunk = reference(lo, hi), peer(lo, hi)
        fl = m.read(logits_at, (hi - lo, padded), torch.float16, lo * padded)
        fd = m.read(dlogits_at, (hi - lo, padded), torch.float16, lo * padded)
        padded_ok = padded_ok and all_equal(int_view(fl)[:, vocab:], 0) and all_equal(int_view(fd)[:, vocab:], 0)
        logit_hashes.append(row_hashes(int_view(fl)))
        grad_hashes.append(row_hashes(int_view(fd)))
        flint_logits.add(fl[:, :vocab], ref)
        peer_logits.add(peer_chunk, ref)

        ref64 = ref.double()
        ref_loss_sum += F.cross_entropy(ref64, tgt, reduction="sum").item()
        grad = F.softmax(ref64, dim=-1) * scale32
        grad[torch.arange(hi - lo, device="cuda"), tgt] -= scale32
        flint_grad.add(fd[:, :vocab], grad)
        with torch.enable_grad():
            leaf = peer_chunk.contiguous().clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(leaf, tgt, reduction="none")
            (loss.sum() * scale).backward()
        peer_grad.add(leaf.grad, grad)
        at_targets = (torch.arange(hi - lo, device="cuda"), tgt)
        flint_rest, peer_rest = fd[:, :vocab].clone(), leaf.grad.clone()
        flint_rest[at_targets] = peer_rest[at_targets] = grad[at_targets] = 0
        flint_off.add(flint_rest, grad)
        peer_off.add(peer_rest, grad)
        del flint_rest, peer_rest

        ref_top = ref.argmax(-1)
        agree_flint += int((fl[:, :vocab].argmax(-1) == ref_top).sum())
        agree_peer += int((peer_chunk.argmax(-1) == ref_top).sum())
        del ref, ref64, peer_chunk, fl, fd, grad, leaf
    flint_loss, ref_loss = m.mean_loss(), ref_loss_sum / rows
    # Each row's outputs as three integers, which gate_with_retry compares between attempts.
    row_loss_bits = int_view(m.read(m.outputs()[2], (rows,), torch.float32)).to(torch.int64)
    fingerprint = torch.stack([torch.cat(logit_hashes), torch.cat(grad_hashes), row_loss_bits], 1).cpu()
    return Measured(
        (flint_logits.errors(), peer_logits.errors()),
        (scalar_errors(flint_loss, ref_loss), scalar_errors(peer_loss, ref_loss)),
        (flint_grad.errors(), peer_grad.errors()),
        (flint_off.errors(), peer_off.errors()),
        flint_loss, ref_loss, peer_loss, agree_flint / rows, agree_peer / rows, padded_ok, fingerprint,
    )


GATES = (("logits", gate.FORWARD_FACTOR, "fp16"), ("loss", gate.FORWARD_FACTOR, "fp32"),
         ("dlogits", gate.GRADIENT_FACTOR, "fp16"), ("off_target", gate.GRADIENT_FACTOR, "fp16"))  # fmt: skip


def check_measured(label, results, summaries=None):
    """Runs the four gates over a list of Measured (one per seed), printing the numbers
    first, and appending them to summaries if given; returns the bounds by quantity. Raises
    one AssertionError with every failed gate."""
    failures, bounds, lines = [], {}, []
    for what, factor, output in GATES:
        flint_errors = [getattr(r, what)[0] for r in results]
        peer_errors = [getattr(r, what)[1] for r in results]
        try:
            bounds[what] = gate.check_gate(factor, flint_errors, peer_errors, output)
            verdict = f"bound rel_l2 {bounds[what].rel_l2:.3g} max_abs {bounds[what].max_abs:.3g}"
        except AssertionError as error:
            verdict = "FAILS"
            failures.append(f"{what}: {error}")
        lines.append(
            f"  {what:10s} flint rel_l2 {max(e.rel_l2 for e in flint_errors):.3g} max_abs "
            f"{max(e.max_abs for e in flint_errors):.3g} | peer rel_l2 "
            f"{max(e.rel_l2 for e in peer_errors):.3g} max_abs {max(e.max_abs for e in peer_errors):.3g} | {verdict}"
        )
    loss_line = "  loss     " + ", ".join(
        f"seed {s}: flint {r.flint_loss:.6f} fp32 {r.ref_loss:.6f} autocast {r.peer_loss:.6f}"
        for s, r in enumerate(results)
    )
    top1 = "  top-1 vs fp32: " + ", ".join(
        f"flint {r.top1_flint:.4f} autocast {r.top1_peer:.4f}" for r in results
    )
    summary = f"GATE {label} (worst of {len(results)} seeds)\n" + "\n".join(lines) + "\n" + loss_line + "\n" + top1
    print(summary)
    if summaries is not None:
        summaries.append(summary)
    assert all(r.padded_ok for r in results), f"{label}: logits or dlogits past vocab are not +0"
    assert not failures, f"{label}:\n" + "\n".join(failures)
    return bounds


def gate_case(flint, name, sequences, length, weights, tokens, seeds=gate.SEEDS, separate=False, cfg=None,
              after_forward=None, summaries=None, fingerprints=None):  # fmt: skip
    """The whole-model gate on one shape: per seed, a model (weights "init" or "mixed"),
    random or real tokens, Flint's forward pass and the torch reference and peer; then the
    three gates over the seeds. name is a config of gpt2_torch's, unless cfg gives one.
    after_forward(seed, m, config), if given, runs after each forward pass (the retry's tests
    use it to corrupt an output); summaries goes to check_measured, and fingerprints, if
    given, gets each seed's Measured.fingerprint before the gates are checked."""
    cfg = cfg or G.CONFIGS[name]
    config = gpt2_config(cfg, sequences, length, separate)
    rows = sequences * length
    scale = gate_scale(rows)
    results = []
    for seed in seeds:
        model = make_model(cfg, seed, weights)
        if tokens == "real":
            ids = real_tokens(rows + 1, seed)
            assert int(ids.max()) < cfg.vocab_size
        else:
            ids = random_ids(cfg, rows + 1, seed)
        batch = Batch(ids, sequences, length)
        with torch.no_grad():
            torch_ref = TorchReference(model, batch.idx, batch.targets)
            with Gpt2(flint, config) as m:
                m.load(state_of(model))
                m.forward(batch.tokens, batch.target_tokens, scale)
                if after_forward is not None:
                    after_forward(seed, m, config)
                results.append(
                    measure(m, config, torch_ref.reference, torch_ref.peer, batch.targets, scale, torch_ref.peer_loss)
                )
        del model, batch, torch_ref, ids
        free_memory()
    if fingerprints is not None:
        fingerprints.extend(r.fingerprint for r in results)
    label = f"{name} {sequences}x{length} {weights} weights, {tokens} tokens" + (", separate buffers" if separate else "")
    return check_measured(label, results, summaries)


def attempt_differences(first, second, length):
    """How Flint's outputs differ between two attempts on the same data: per seed, the rows
    whose logits, dlogits or row_loss differ, as (rows differing, sequences they lie in, the
    first and the last such row). first and second hold each seed's Measured.fingerprint."""
    differences = []
    for a, b in zip(first, second, strict=True):
        rows = (a != b).any(1).nonzero().flatten()
        sequences = sorted(set((rows // length).tolist()))
        ends = (int(rows[0]), int(rows[-1])) if len(rows) else (None, None)
        differences.append((len(rows), sequences, *ends))
    return differences


def describe_differences(differences, length):
    lines = []
    for seed, (count, sequences, first, last) in enumerate(differences):
        if count == 0:
            lines.append(f"  seed {seed}: Flint's outputs are the same bits in both attempts")
        else:
            lines.append(
                f"  seed {seed}: {count} rows differ, from row {first} (sequence {first // length}, position "
                f"{first % length}) to row {last} (sequence {last // length}, position {last % length}), in "
                f"{len(sequences)} sequence(s): {sequences[:8]}{'...' if len(sequences) > 8 else ''}"
            )
    return "\n".join(lines)


def gate_with_retry(flint, name, sequences, length, weights, tokens, *, record, after_forward=None, **kwargs):
    """gate_case, retried once if its gate fails, with RETRY_PADDING_BYTES allocated first;
    returns the bounds. When only the retry passed, it calls record (the test's record_property)
    with retry_report.PROPERTY and a text with the first attempt's failure, how the two attempts'
    outputs differ and the retry's numbers, which retry_report.py shows even under pytest -q. If
    the retry fails too, raises one AssertionError with both failures. Anything else than a failed
    gate (a skip, running out of memory, a CUDA error) isn't retried. after_forward(attempt, seed,
    m, config), if given, runs after each forward pass, with attempt 0 or 1.

    Why: one cell of the 2080 Ti's memory, which has no ECC, sometimes reads a 1 back as 0
    (found by a memory test that runs none of Flint's code). When a buffer of Flint's lands on
    it, one value of one pass is wrong: a logit, or an activation, whose error causal attention
    carries to the later positions of its sequence. With 1 GiB allocated before the model, the
    same run was wrong in 0 of 80. The retry's padding moves every later allocation the same
    way. A failure that follows the data or the kernels repeats, so it still fails.

    A pass on the retry alone would also let through a Flint bug that doesn't repeat (a race,
    a read of stale memory). The data are the same in both attempts and Flint is deterministic,
    so the attempts' outputs may differ only as one flipped activation or output makes them:
    per seed, in rows of one sequence (one row, or a row and the later positions of its
    sequence). Each seed's model is its own allocation, so each may meet the cell. Differences
    in two sequences of one seed fail, whatever the retry measured. That is a narrower hole,
    not a closed one: most kernels work within a sequence, so a bug that doesn't repeat can
    stay inside one too and pass here, which is why the record gives the rows' count and span
    for a reader to judge, and why scripts/sanitize_full.py exists. The other way round, the
    cell can also fail this rule without any bug, by flipping a weight, or a value in a buffer
    that every row passes through (the test's own chunk buffers); then the run is repeated."""

    def attempt(number, summaries, fingerprints):
        hook = None if after_forward is None else (lambda *a: after_forward(number, *a))
        return gate_case(flint, name, sequences, length, weights, tokens, after_forward=hook,
                         summaries=summaries, fingerprints=fingerprints, **kwargs)  # fmt: skip

    first_prints = []
    try:
        return attempt(0, None, first_prints)
    except AssertionError as error:
        first = str(error)
    # Leaving the except block drops the traceback, and with it the first attempt's frames.
    free_memory()
    padding = torch.empty(RETRY_PADDING_BYTES, dtype=torch.uint8, device="cuda")
    summaries, retry_prints = [], []
    try:
        bounds = attempt(1, summaries, retry_prints)
    except AssertionError as error:
        raise AssertionError(
            f"the gate failed twice.\nFirst attempt:\n{first}\n"
            f"Retry, with {RETRY_PADDING_BYTES / GiB:g} GiB allocated first:\n{error}"
        ) from None
    finally:
        del padding
        free_memory()
    # A first attempt that failed before every seed was measured has nothing to compare.
    assert len(first_prints) == len(retry_prints), f"the first attempt failed before its gates:\n{first}"
    differences = attempt_differences(first_prints, retry_prints, length)
    described = describe_differences(differences, length)
    if any(len(difference[1]) > 1 for difference in differences):
        raise AssertionError(
            "the gate passed on its retry, but Flint may not be deterministic: on the same data, the two "
            "attempts' outputs differ in more than one sequence of a seed. One flipped activation can't "
            "do that; a bug that doesn't repeat can, and so can a flip in a weight or in a buffer every "
            f"row passes through. Run it again before deciding.\n{described}\nFirst attempt:\n{first}"
        )
    record(
        retry_report.PROPERTY,
        f"The first attempt failed:\n{first}\n"
        f"Flint's outputs, first attempt against retry:\n{described}\n"
        f"The retry, with {RETRY_PADDING_BYTES / GiB:g} GiB allocated first, passed:\n" + "\n".join(summaries),
    )
    return bounds


# -- 1. The whole-model gate ----------------------------------------------------------------

TINY_SHAPES = [(1, 64), (3, 64), (2, 32), (5, 8)]
GPT2_SHAPES = [(2, 256), (1, 1024), (8, 64)]
WEIGHTS = ["init", "mixed"]


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("sequences, length", TINY_SHAPES, ids=lambda v: str(v))
def test_tiny_forward_passes_the_gate(flint, sequences, length, weights):
    gate_case(flint, "tiny", sequences, length, weights, "random")


# Every other shape has heads of width 64, whose scale is exactly 1/8. These have others, a
# length below block_size, and a vocabulary that ends mid-chunk.
ODD_MODELS = {
    "heads-of-24": (G.GPTConfig(n_layer=3, n_head=4, n_embd=96, block_size=40, vocab_size=100, padded_vocab_size=104), 3, 24),
    "one-head-of-8": (G.GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=8, vocab_size=5, padded_vocab_size=8), 2, 8),
    "one-head-of-128": (G.GPTConfig(n_layer=2, n_head=1, n_embd=128, block_size=64, vocab_size=100, padded_vocab_size=128), 2, 32),
}  # fmt: skip


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("name", ODD_MODELS)
def test_odd_models_pass_the_gate(flint, name, weights):
    cfg, sequences, length = ODD_MODELS[name]
    gate_case(flint, name, sequences, length, weights, "random", cfg=cfg)


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("sequences, length", GPT2_SHAPES, ids=lambda v: str(v))
def test_gpt2_forward_passes_the_gate(flint, sequences, length, weights):
    gate_case(flint, "gpt2", sequences, length, weights, "random")


def test_the_gates_scale_keeps_gpt2s_gradients_out_of_fp16s_underflow(flint):
    # Why the gates run at LOSS_SCALE / rows. At init a gradient away from the target is about
    # scale / vocab: at scale 1 / rows and 1,024 rows that is about a third of FP16's smallest
    # subnormal, so every dlogits value but the targets' is 0 or one subnormal step, in Flint
    # and in the peer, and a gate passes whatever those columns should hold. At the gates'
    # scale they are normal numbers.
    sequences, length = 1, 1024
    config = gpt2_config(GPT2, sequences, length)
    state = state_of(make_model(GPT2, 0, "init"))
    batch = random_batch(GPT2, sequences, length, 0)
    shares = {}
    for name, scale in (("gate", gate_scale(batch.rows)), ("1/rows", 1.0 / batch.rows)):
        dlogits = run_model(flint, config, state, batch, scale).dlogits[:, : config.vocab]
        magnitudes = dlogits.float().abs()
        assert torch.isfinite(magnitudes).all(), name
        shares[name] = ((magnitudes > 0).float().mean().item(), (magnitudes >= 2.0**-14).float().mean().item())
    print(f"dlogits nonzero / normal: gate scale {shares['gate']}, 1/rows {shares['1/rows']}")
    assert shares["gate"][1] > 0.99, shares
    assert shares["1/rows"][0] < 0.5 and shares["1/rows"][1] < 0.001, shares


@pytest.mark.parametrize("weights", WEIGHTS)
def test_gpt2_forward_passes_the_gate_on_real_tokens(flint, weights):
    gate_case(flint, "gpt2", 1, 1024, weights, "real")


@pytest.mark.parametrize("weights", WEIGHTS)
def test_gpt2_micro_batch_passes_the_gate(flint, weights, record_property):
    free = torch.cuda.mem_get_info()[0]
    if free < MICRO_BATCH_FREE_BYTES:
        reason = f"GPT-2's (16, 1024) micro-batch needs {MICRO_BATCH_FREE_BYTES / GiB:.0f} GiB free, {free / GiB:.1f} GiB are"
        if os.environ.get(REQUIRE_MICRO_BATCH) == "1":
            pytest.fail(f"{reason}, and {REQUIRE_MICRO_BATCH}=1 doesn't let this gate skip")
        pytest.skip(reason)
    gate_with_retry(flint, "gpt2", 16, 1024, weights, "random", record=record_property)


# The retry itself, on the tiny config. A fault is made on purpose: after the forward pass,
# one logit's lowest exponent bit flips, halving or doubling it, as the 2080 Ti's fault did
# (0x344c read as 0x304c). On the first attempt only, it is a fault that the retry escapes;
# on both, a bug that follows the data. These tests give gate_with_retry a Recorder in place
# of record_property, so they never show as retried themselves.
RETRY_CASE = ("tiny", 2, 32, "mixed", "random")
FAULTY_SEED = 1
# check_gate numbers the seeds by their position in the list.
FAULTY_LINE = rf"seed {gate.SEEDS.index(FAULTY_SEED)}: flint[^\n]*FAIL"


def flip_a_logit(m, config, row=0):
    """Flips bit 0x0400 of the row's largest logit by magnitude, in Flint's output."""
    logits_at, _, _ = m.outputs()
    row_at = logits_at + row * config.padded_vocab * 2
    values = m.read(row_at, (config.vocab,), torch.float16)
    col = int(values.float().abs().argmax())
    bits = values.view(torch.int16)[col : col + 1] ^ 0x0400
    m.flint.copy(row_at + col * 2, bits, 2)
    torch.cuda.synchronize()


class Attempts:
    """An after_forward hook that flips a logit in each of the given rows, for the given seeds,
    on the given attempts and logs, per call, the attempt, the seed and torch's allocated bytes
    (the padding is torch's)."""

    def __init__(self, faulty=(), rows=(0,), seeds=(FAULTY_SEED,)):
        self.faulty, self.rows, self.seeds, self.calls = set(faulty), rows, seeds, []

    def __call__(self, attempt, seed, m, config):
        self.calls.append((attempt, seed, torch.cuda.memory_allocated()))
        if attempt in self.faulty and seed in self.seeds:
            for row in self.rows:
                flip_a_logit(m, config, row)

    def attempts(self):
        return [c[0] for c in self.calls]

    def padded(self, attempt):
        """Whether the padding was allocated at each of the attempt's calls."""
        return [allocated >= RETRY_PADDING_BYTES for a, _, allocated in self.calls if a == attempt]


class Recorder(list):
    """Stands in for record_property: keeps each (name, value)."""

    def __call__(self, name, value):
        self.append((name, value))


def test_retry_is_not_taken_when_the_gate_passes(flint):
    hook, recorded = Attempts(), Recorder()
    gate_with_retry(flint, *RETRY_CASE, record=recorded, after_forward=hook)
    assert recorded == []
    assert hook.attempts() == [0] * len(gate.SEEDS)
    assert not any(hook.padded(0))


def test_a_failure_that_does_not_repeat_passes_on_the_retry(flint):
    hook, recorded = Attempts(faulty={0}), Recorder()
    gate_with_retry(flint, *RETRY_CASE, record=recorded, after_forward=hook)
    assert hook.attempts() == [0] * len(gate.SEEDS) + [1] * len(gate.SEEDS)
    # The padding is held through the whole retry, and only there.
    assert not any(hook.padded(0)) and all(hook.padded(1))
    assert torch.cuda.memory_allocated() < RETRY_PADDING_BYTES
    assert len(recorded) == 1 and recorded[0][0] == retry_report.PROPERTY, recorded
    text = recorded[0][1]
    first, _, second = text.partition("The retry, with 1 GiB allocated first, passed:")
    assert second, text
    assert first.startswith("The first attempt failed") and "logits: gate factor" in first
    assert re.search(FAULTY_LINE, first), first
    assert "GATE tiny 2x32" in second and "FAILS" not in second
    # How the attempts differ: the faulty seed's one row, and nothing else.
    faulty = gate.SEEDS.index(FAULTY_SEED)
    assert f"seed {faulty}: 1 rows differ, from row 0 (sequence 0, position 0) to row 0 (" in first, first
    for seed in range(len(gate.SEEDS)):
        if seed != faulty:
            assert f"seed {seed}: Flint's outputs are the same bits in both attempts" in first, first


# RETRY_CASE has 2 sequences of 32 rows: rows 3 and 20 share a sequence, rows 0 and 32 don't.
def test_differences_within_one_sequence_pass_on_the_retry(flint):
    # One flipped activation reaches the later positions of its sequence, and no other.
    hook, recorded = Attempts(faulty={0}, rows=(3, 20)), Recorder()
    gate_with_retry(flint, *RETRY_CASE, record=recorded, after_forward=hook)
    assert len(recorded) == 1, recorded
    expected = "2 rows differ, from row 3 (sequence 0, position 3) to row 20 (sequence 0, position 20), in 1 sequence(s)"
    assert expected in recorded[0][1], recorded[0][1]


def test_each_seed_may_differ_in_a_sequence_of_its_own(flint):
    # Each seed's model is its own allocation, so two seeds can meet the cell in one attempt.
    hook, recorded = Attempts(faulty={0}, rows=(40,), seeds=gate.SEEDS[:2]), Recorder()
    gate_with_retry(flint, *RETRY_CASE, record=recorded, after_forward=hook)
    assert len(recorded) == 1, recorded
    assert recorded[0][1].count("1 rows differ, from row 40 (sequence 1, position 8) to row 40 (") == 2, recorded[0][1]


def test_differences_in_two_sequences_fail_though_the_retry_passes(flint):
    # The retry's gate is clean, but no single flipped value changes two sequences of a pass:
    # Flint gave different bits for the same data, which the retry must not hide.
    hook, recorded = Attempts(faulty={0}, rows=(0, 32)), Recorder()
    with pytest.raises(AssertionError) as caught:
        gate_with_retry(flint, *RETRY_CASE, record=recorded, after_forward=hook)
    text = str(caught.value)
    assert hook.attempts() == [0] * len(gate.SEEDS) + [1] * len(gate.SEEDS)
    assert torch.cuda.memory_allocated() < RETRY_PADDING_BYTES
    assert recorded == []
    assert text.startswith("the gate passed on its retry, but Flint may not be deterministic"), text
    expected = "2 rows differ, from row 0 (sequence 0, position 0) to row 32 (sequence 1, position 0), in 2 sequence(s): [0, 1]"
    assert expected in text, text
    assert re.search(FAULTY_LINE, text), text


@pytest.mark.parametrize("required", [False, True])
def test_too_little_memory_skips_the_micro_batch_gate_or_fails_it_when_required(flint, monkeypatch, required):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (MICRO_BATCH_FREE_BYTES - 1, 24 * GiB))
    if required:
        monkeypatch.setenv(REQUIRE_MICRO_BATCH, "1")
    else:
        monkeypatch.delenv(REQUIRE_MICRO_BATCH, raising=False)
    # Caught by hand: a skip raised inside pytest.raises would skip this test, not fail it.
    try:
        test_gpt2_micro_batch_passes_the_gate(flint, "init", Recorder())
        outcome, message = "ran", ""
    except pytest.skip.Exception as skipped:
        outcome, message = "skipped", str(skipped)
    except pytest.fail.Exception as failed:
        outcome, message = "failed", str(failed)
    assert outcome == ("failed" if required else "skipped"), (outcome, message)
    assert "needs 16 GiB free" in message, message


def test_a_failure_that_repeats_on_the_retry_fails(flint):
    hook, recorded = Attempts(faulty={0, 1}), Recorder()
    with pytest.raises(AssertionError) as caught:
        gate_with_retry(flint, *RETRY_CASE, record=recorded, after_forward=hook)
    text = str(caught.value)
    assert hook.attempts() == [0] * len(gate.SEEDS) + [1] * len(gate.SEEDS)
    assert all(hook.padded(1))
    assert torch.cuda.memory_allocated() < RETRY_PADDING_BYTES
    assert recorded == []
    first, _, second = text.partition("Retry, with 1 GiB allocated first:")
    assert text.startswith("the gate failed twice") and second, text
    for attempt in (first, second):
        assert re.search(FAULTY_LINE, attempt), text


def test_only_a_failed_gate_is_retried(flint):
    calls, recorded = [], Recorder()

    def broken(attempt, seed, m, config):
        calls.append(attempt)
        raise RuntimeError("not a gate failure")

    with pytest.raises(RuntimeError, match="not a gate failure"):
        gate_with_retry(flint, *RETRY_CASE, record=recorded, after_forward=broken)
    assert calls == [0] and recorded == []


RETRY_REPORT_TESTS = dict(
    test_quiet="""
def test_clean():
    pass

def test_retried(record_property):
    record_property("gate_retry", "The first attempt failed: logits FAIL")
""",
    test_loud="""
def test_retried_but_failed(record_property):
    record_property("gate_retry", "this one failed anyway")
    assert False
""",
)


def test_a_retry_shows_in_quiet_output(pytester, pytestconfig):
    # conftest.py registers the plugin for every run of the suite.
    assert pytestconfig.pluginmanager.has_plugin("retry_report")
    # precheck reads pytest -q. A passed retry counts on the last line, which turns yellow,
    # with its text in a section above it, and the exit status stays a pass.
    pytester.makepyfile(**RETRY_REPORT_TESTS)
    result = pytester.runpytest("-q", "-p", "retry_report", "test_quiet.py")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [f"*{retry_report.TITLE}*", "test_quiet.py::test_retried", "The first attempt failed: logits FAIL",
         "1 passed, 1 retried in *"]
    )  # fmt: skip
    assert result.parseoutcomes() == {"passed": 1, "retried": 1}

    # A test that failed counts as failed, whatever it recorded.
    result = pytester.runpytest("-q", "-p", "retry_report", "test_loud.py")
    assert result.ret == 1
    assert result.parseoutcomes() == {"failed": 1}
    assert retry_report.TITLE not in result.stdout.str()


# -- 2. The assembly ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    return make_model(TINY, 0, "mixed")


@pytest.fixture(scope="module")
def tiny_model_b():
    return make_model(TINY, 1, "mixed")


@pytest.mark.parametrize("name", ["tiny", "gpt2"])
@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_parameters_are_gpt2_torchs_in_order(flint, name, separate):
    cfg = G.CONFIGS[name]
    meta = G.build_model(cfg, device="meta")
    expected = [(n, tuple(p.shape)) for n, p in meta.named_parameters()]
    with Gpt2(flint, gpt2_config(cfg, 1, 8, separate)) as m:
        assert flint.lib.flint_gpt2_parameter_count(m.handle) == len(expected)
        parameters = m.parameters()
    assert [p[0] for p in parameters] == [n for n, _ in expected]
    for (name_, address, rows, cols), (_, shape) in zip(parameters, expected):
        assert address is not None and address % ALIGNMENT == 0, name_
        # A 1-D parameter is 1 row; a linear layer's weight is out x in, as PyTorch's is.
        assert (rows, cols) == ((1, shape[0]) if len(shape) == 1 else shape), name_
        assert rows * cols == math.prod(shape), name_
    assert len({p[1] for p in parameters}) == len(parameters), "two parameters share an address"
    assert parameters[0][0] == "wte.weight" and parameters[0][2] == cfg.padded_vocab_size


def up(n):
    return -(-n // ALIGNMENT) * ALIGNMENT


def expected_bytes(layers, heads, width, block_size, vocab, padded_vocab, sequences, length):
    """The device bytes of one allocation, summed in Python: every buffer rounded up to 256
    bytes, listed by hand."""
    rows, head_width, batches = sequences * length, width // heads, sequences * heads
    heads_layout = sequences * heads * length * head_width * 2
    buffers = [padded_vocab * width * 4, block_size * width * 4]  # wte, wpe (FP32 masters)
    for _ in range(layers):
        buffers += [width * 4, width * 4]  # ln_1
        buffers += [3 * width * width * 4, 3 * width * 4]  # attn.c_attn
        buffers += [width * width * 4, width * 4]  # attn.c_proj
        buffers += [width * 4, width * 4]  # ln_2
        buffers += [4 * width * width * 4, 4 * width * 4]  # mlp.c_fc
        buffers += [width * 4 * width * 4, width * 4]  # mlp.c_proj
    buffers += [width * 4, width * 4]  # ln_f
    buffers += [padded_vocab * width * 2]  # wte's FP16 copy
    for _ in range(layers):  # the linear layers' FP16 copies
        buffers += [3 * width * width * 2, width * width * 2, 4 * width * width * 2, width * 4 * width * 2]
    buffers += [rows * width * 4, rows * width * 4]  # the residual stream, twice
    buffers += [rows * width * 2]  # LayerNorm's output
    buffers += [rows * 4, rows * 4]  # its mean and rstd
    buffers += [rows * 3 * width * 2]  # qkv
    buffers += [heads_layout] * 3  # q, k, v
    buffers += [batches * length * length * 2] * 2  # scores, probabilities
    buffers += [heads_layout]  # the heads' output
    buffers += [rows * width * 2]  # attention's output rows
    buffers += [rows * 4 * width * 2] * 2  # GELU's output, its input
    buffers += [rows * padded_vocab * 2] * 2  # logits, dlogits
    buffers += [rows * 4]  # row_loss
    return sum(up(b) for b in buffers)


def config_fields(cfg, sequences, length):
    return dict(
        layers=cfg.n_layer, heads=cfg.n_head, width=cfg.n_embd, block_size=cfg.block_size,
        vocab=cfg.vocab_size, padded_vocab=cfg.padded_vocab_size, sequences=sequences, length=length,
    )


def bytes_of(flint, config):
    n = ctypes.c_size_t(0)
    flint.check(flint.lib.flint_gpt2_bytes(ctypes.byref(config), ctypes.byref(n)))
    return n.value


ODD = dict(layers=3, heads=4, width=96, block_size=40, vocab=100, padded_vocab=104, sequences=3, length=24)
# Legal configs at their limits (flint.h): the most layers, the widest width, the most
# vocabulary, the most batches (13,107 x 5 = 65,535) and the most rows.
LIMIT_CONFIGS = {
    "layers-1024": dict(layers=1024, heads=2, width=64, block_size=1024, vocab=50257, padded_vocab=50304, sequences=4, length=1024),
    "width-16384": dict(layers=2, heads=128, width=16384, block_size=1024, vocab=65536, padded_vocab=65536, sequences=2, length=1024),
    "batches-65535": dict(layers=2, heads=5, width=4000, block_size=640, vocab=65536, padded_vocab=65536, sequences=13107, length=640),
    "rows-8388480": dict(layers=2, heads=1, width=64, block_size=128, vocab=65536, padded_vocab=65536, sequences=65535, length=128),
    "smallest": dict(layers=1, heads=1, width=8, block_size=8, vocab=1, padded_vocab=8, sequences=1, length=8),
    "largest": dict(layers=1024, heads=8, width=16384, block_size=1024, vocab=65536, padded_vocab=65536, sequences=8191, length=1024),
}  # fmt: skip


@pytest.mark.parametrize(
    "fields",
    [config_fields(TINY, 2, 32), config_fields(TINY, 5, 8), config_fields(GPT2, 1, 8),
     config_fields(GPT2, 16, 1024), ODD, *LIMIT_CONFIGS.values()],
    ids=["tiny-2x32", "tiny-5x8", "gpt2-1x8", "gpt2-16x1024", "odd", *LIMIT_CONFIGS],
)  # fmt: skip
def test_bytes_equals_an_independent_sum(flint, fields):
    # flint_gpt2_bytes needs no GPU memory: the largest legal config is terabytes. Python's
    # ints don't wrap, so a match shows that the C++ sizes don't either.
    total = bytes_of(flint, Gpt2Config(**fields))
    assert total == expected_bytes(**fields)
    if fields == config_fields(GPT2, 16, 1024):
        print(f"BYTES GPT-2 (16, 1024): {total} bytes = {total / GiB:.3f} GiB = {total / 1e9:.2f} GB")


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_only_separate_buffers_leaves_gaps_between_the_large_weights(flint, separate):
    # What separate_buffers is for: memcheck can only see a buffer's bounds if it is its own
    # allocation. One allocation packs the buffers at exact multiples of 256 bytes, while
    # separate ones this large start on the driver's larger pages, so at least one gap differs.
    # GPT-2's weights are the large buffers, and the shortest batch keeps the rest small.
    with Gpt2(flint, gpt2_config(GPT2, 1, 8, separate=separate)) as m:
        parameters = m.parameters()
    packed = all(
        next_at - at == (rows * cols * 4 + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        for (_, at, rows, cols), (_, next_at, _, _) in zip(parameters, parameters[1:])
    )
    assert packed != separate


@pytest.mark.parametrize("fields", [config_fields(TINY, 2, 32), ODD], ids=["tiny-2x32", "odd"])
@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_buffers_lie_inside_the_bytes_and_do_not_overlap(flint, fields, separate):
    config = Gpt2Config(**fields, separate_buffers=int(separate))
    total = expected_bytes(**fields)
    rows, padded = fields["sequences"] * fields["length"], fields["padded_vocab"]
    with Gpt2(flint, config) as m:
        parameters = m.parameters()
        logits_at, dlogits_at, loss_at = m.outputs()
    regions = [(name, address, rows_ * cols * 4) for name, address, rows_, cols in parameters]
    regions += [("logits", logits_at, rows * padded * 2), ("dlogits", dlogits_at, rows * padded * 2),
                ("row_loss", loss_at, rows * 4)]  # fmt: skip
    regions.sort(key=lambda r: r[1])
    for (name_a, start_a, size_a), (name_b, start_b, _) in zip(regions, regions[1:]):
        assert start_a + size_a <= start_b, f"{name_a} and {name_b} overlap"
    if separate:
        return
    base = parameters[0][1]
    assert base % ALIGNMENT == 0 and min(r[1] for r in regions) == base
    # The parameters come first and the outputs last: the span of their addresses is the
    # whole allocation.
    assert max(start + up(size) for _, start, size in regions) - base == total
    assert loss_at + up(rows * 4) - base == total == bytes_of(flint, config)
    for name_, start, size in regions:
        assert base <= start and start + size <= base + total, name_
        assert start % ALIGNMENT == 0, name_


@pytest.mark.parametrize("name, sequences, length", [("tiny", 3, 64), ("gpt2", 2, 256)], ids=str)
def test_one_allocation_and_separate_buffers_give_the_same_bits(flint, name, sequences, length):
    cfg = G.CONFIGS[name]
    model = make_model(cfg, 3, "mixed")
    batch = random_batch(cfg, sequences, length, 4)
    one = run_model(flint, gpt2_config(cfg, sequences, length, False), state_of(model), batch, 1.0 / batch.rows)
    separate = run_model(flint, gpt2_config(cfg, sequences, length, True), state_of(model), batch, 1.0 / batch.rows)
    assert_same_outputs(one, separate, "separate_buffers")


@pytest.mark.parametrize("sequences, length", [(1, 8), (3, 64), (2, 32)], ids=str)
def test_the_assembly_equals_the_kernels_called_by_hand(flint, tiny_model, sequences, length):
    # The same kernels in the same order from Python, bit for bit: the assembly passes the
    # right arguments in the right order.
    batch = random_batch(TINY, sequences, length, 5)
    scale = 1.0 / batch.rows
    state = state_of(tiny_model)
    by_assembly = run_model(flint, gpt2_config(TINY, sequences, length), state, batch, scale)
    by_hand = compose_forward(flint, gpt2_config(TINY, sequences, length), state, state, batch, scale)
    assert_same_outputs(by_assembly, by_hand, "by hand", loss=False)


def test_the_smallest_legal_model_works_and_equals_the_kernels_by_hand(flint):
    fields = LIMIT_CONFIGS["smallest"]
    cfg = G.GPTConfig(n_layer=1, n_head=1, n_embd=8, block_size=8, vocab_size=1, padded_vocab_size=8)
    model = make_model(cfg, 0, "mixed")
    ids = torch.randint(0, 8, (9,), generator=generator(1), device="cuda")
    batch = Batch(ids, 1, 8)
    batch.target_tokens = as_tokens(torch.zeros(8, dtype=torch.int64, device="cuda"))
    config = Gpt2Config(**fields)
    out = run_model(flint, config, state_of(model), batch, 1.0)
    by_hand = compose_forward(flint, config, state_of(model), state_of(model), batch, 1.0)
    assert_same_outputs(out, by_hand, "smallest model", loss=False)
    # One class: p = 1, so every row's loss is exactly log(1) = 0.
    assert all_equal(out.row_loss, 0) and out.loss == 0.0
    assert torch.isfinite(out.logits.float()).all()


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_run_twice_is_bitwise_identical(flint, tiny_model, separate):
    config = gpt2_config(TINY, 3, 64, separate)
    batch = random_batch(TINY, 3, 64, 6)
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        m.forward(batch.tokens, batch.target_tokens, 1.0 / batch.rows)
        first = read_outputs(m, config)
        # Poison every output, so a second pass that skipped a write would show.
        for address, tensor in zip(m.outputs(), (first.logits, first.dlogits, first.row_loss)):
            flint.copy(address, torch.full_like(int_view(tensor), as_int16(0x5A5A) if tensor.dtype == torch.float16 else 0x5A5A5A5A), tensor.numel() * tensor.element_size())
        torch.cuda.synchronize()
        m.forward(batch.tokens, batch.target_tokens, 1.0 / batch.rows)
        second = read_outputs(m, config)
    assert_same_outputs(first, second, "second pass")


def test_no_state_leaks_between_passes(flint, tiny_model):
    config = gpt2_config(TINY, 3, 64)
    a = random_batch(TINY, 3, 64, 7)
    b = random_batch(TINY, 3, 64, 8)
    scale = 1.0 / a.rows
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        m.forward(a.tokens, a.target_tokens, scale)
        first_a = read_outputs(m, config)
        m.forward(b.tokens, b.target_tokens, scale)
        first_b = read_outputs(m, config)
        m.forward(a.tokens, a.target_tokens, scale)
        second_a = read_outputs(m, config)
    assert_same_outputs(first_a, second_a, "A again after B")
    assert any_unequal(int_view(first_a.logits), int_view(first_b.logits)), "B's logits equal A's"
    assert any_unequal(int_view(first_a.row_loss), int_view(first_b.row_loss))
    assert first_a.loss != first_b.loss


@pytest.mark.parametrize("name, sequences, length", [("tiny", 3, 64), ("tiny", 1, 8), ("gpt2", 1, 1024)], ids=str)
def test_mean_loss_is_the_exact_mean_of_row_loss(flint, name, sequences, length):
    cfg = G.CONFIGS[name]
    model = make_model(cfg, 9, "mixed")
    batch = random_batch(cfg, sequences, length, 10)
    config = gpt2_config(cfg, sequences, length)
    with Gpt2(flint, config) as m:
        m.load(state_of(model))
        m.forward(batch.tokens, batch.target_tokens, 1.0 / batch.rows)
        out = read_outputs(m, config)
        again = m.mean_loss()
    values = out.row_loss.cpu().tolist()  # FP32 values, exactly, as Python floats (FP64)
    total = 0.0
    for v in values:
        total += v
    # The sum in the order of the rows, in FP64, divided by the row count: the very value.
    assert out.loss == total / len(values) == again
    assert out.loss == math.fsum(values) / len(values)
    assert 8.0 < out.loss < 16.0, "random weights give a loss near log(50257) = 10.8"


def test_inputs_are_unchanged(flint, tiny_model):
    config = gpt2_config(TINY, 3, 64)
    batch = random_batch(TINY, 3, 64, 11)
    tokens, targets = batch.tokens.clone(), batch.target_tokens.clone()
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        m.forward(batch.tokens, batch.target_tokens, 0.5)
        read_outputs(m, config)
    assert_same_bits(batch.tokens.view(torch.float16), tokens.view(torch.float16), "tokens")
    assert_same_bits(batch.target_tokens.view(torch.float16), targets.view(torch.float16), "targets")


def test_targets_affect_only_the_losses_outputs(flint, tiny_model):
    config = gpt2_config(TINY, 3, 64)
    batch = random_batch(TINY, 3, 64, 12)
    other = random_batch(TINY, 3, 64, 13)
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        m.forward(batch.tokens, batch.target_tokens, 1.0 / batch.rows)
        first = read_outputs(m, config)
        m.forward(batch.tokens, other.target_tokens, 1.0 / batch.rows)
        second = read_outputs(m, config)
    assert_same_bits(second.logits, first.logits, "logits with other targets")
    assert any_unequal(int_view(second.dlogits), int_view(first.dlogits))
    assert any_unequal(int_view(second.row_loss), int_view(first.row_loss))


def test_scale_affects_only_dlogits(flint, tiny_model):
    config = gpt2_config(TINY, 3, 64)
    batch = random_batch(TINY, 3, 64, 14)
    scales = [1.0, 1.0 / batch.rows, 0.125, -4.0, 0.0]
    with Gpt2(flint, config) as m:
        m.load(state_of(tiny_model))
        outputs = []
        for scale in scales:
            m.forward(batch.tokens, batch.target_tokens, scale)
            outputs.append(read_outputs(m, config))
    for scale, out in zip(scales, outputs):
        assert_same_bits(out.logits, outputs[0].logits, f"logits at scale {scale}")
        assert_same_bits(out.row_loss, outputs[0].row_loss, f"row_loss at scale {scale}")
        assert out.loss == outputs[0].loss
    for i, a in enumerate(outputs):
        for j in range(i + 1, len(outputs)):
            assert any_unequal(int_view(a.dlogits), int_view(outputs[j].dlogits)), f"scales {scales[i]} and {scales[j]}"
    assert all_equal(outputs[-1].dlogits, 0), "a scale of 0 gives 0 gradients"


def changed_weight(tensor, seed, std=0.05):
    return tensor.detach() + std * torch.randn(tensor.shape, generator=generator(seed), device="cuda")


@pytest.mark.parametrize("name", ["blocks.0.attn.c_attn.weight", "blocks.0.attn.c_proj.weight",
                                  "blocks.1.mlp.c_fc.weight", "blocks.1.mlp.c_proj.weight"])  # fmt: skip
def test_a_changed_linear_weight_has_no_effect_until_a_sync(flint, tiny_model, name):
    config = gpt2_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 15)
    scale = 1.0 / batch.rows
    state = state_of(tiny_model)
    changed = clone_state(state)
    changed[name] = changed_weight(state[name], 16)
    fresh = run_model(flint, config, changed, batch, scale)
    with Gpt2(flint, config) as m:
        m.load(state)
        m.forward(batch.tokens, batch.target_tokens, scale)
        before = read_outputs(m, config)
        write_param(m, name, changed[name])
        m.forward(batch.tokens, batch.target_tokens, scale)
        unsynced = read_outputs(m, config)
        m.sync_weights()
        m.forward(batch.tokens, batch.target_tokens, scale)
        synced = read_outputs(m, config)
    assert_same_outputs(unsynced, before, "after changing the master without a sync")
    assert_same_outputs(synced, fresh, "after the sync, against a fresh model with the changed weights")
    assert any_unequal(int_view(synced.logits), int_view(before.logits)), "the change had no effect at all"


def test_wte_takes_two_paths_the_embedding_reads_the_master_and_the_output_gemm_the_copy(flint, tiny_model):
    config = gpt2_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 17)
    scale = 1.0 / batch.rows
    state = state_of(tiny_model)
    changed = clone_state(state)
    new_wte = changed_weight(state["wte.weight"], 18, 0.1)
    new_wte[TINY.vocab_size :] = 0  # the padded rows stay zero
    changed["wte.weight"] = new_wte
    fresh = run_model(flint, config, changed, batch, scale)
    with Gpt2(flint, config) as m:
        m.load(state)
        m.forward(batch.tokens, batch.target_tokens, scale)
        before = read_outputs(m, config)
        write_param(m, "wte.weight", new_wte)
        m.forward(batch.tokens, batch.target_tokens, scale)
        unsynced = read_outputs(m, config)
        m.sync_weights()
        m.forward(batch.tokens, batch.target_tokens, scale)
        synced = read_outputs(m, config)
    # Without a sync the embedding already reads the new FP32 master while the output GEMM
    # still reads the old copy: neither the old model nor the new one, but the mix, which
    # the kernels called by hand give from the changed masters and the old copies.
    assert any_unequal(int_view(unsynced.logits), int_view(before.logits)), "the embedding didn't see the change"
    assert any_unequal(int_view(unsynced.logits), int_view(fresh.logits)), "the output GEMM saw the change"
    mixed = compose_forward(flint, config, changed, state, batch, scale)
    assert_same_outputs(unsynced, mixed, "unsynced wte, against the changed masters with the old copies", loss=False)
    assert_same_outputs(synced, fresh, "after the sync, against a fresh model with the changed weights")


@pytest.mark.parametrize("name", ["wpe.weight", "blocks.0.ln_1.weight", "blocks.0.ln_1.bias", "blocks.1.attn.c_proj.bias",
                                  "blocks.0.mlp.c_fc.bias", "ln_f.weight", "ln_f.bias"])  # fmt: skip
def test_fp32_parameters_take_effect_without_a_sync(flint, tiny_model, name):
    # Biases and LayerNorm's parameters stay FP32 and need no copy (and so does wpe).
    config = gpt2_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 19)
    scale = 1.0 / batch.rows
    state = state_of(tiny_model)
    changed = clone_state(state)
    changed[name] = changed_weight(state[name], 20, 0.1)
    fresh = run_model(flint, config, changed, batch, scale)
    with Gpt2(flint, config) as m:
        m.load(state)
        m.forward(batch.tokens, batch.target_tokens, scale)
        before = read_outputs(m, config)
        write_param(m, name, changed[name])
        m.forward(batch.tokens, batch.target_tokens, scale)
        unsynced = read_outputs(m, config)
    assert_same_outputs(unsynced, fresh, "without a sync, against a fresh model")
    assert any_unequal(int_view(unsynced.logits), int_view(before.logits))


def with_padded_rows(state, seed, std=0.02):
    """A copy of state whose wte has nonzero padded rows."""
    padded = clone_state(state)
    wte = padded["wte.weight"]
    rows = torch.arange(wte.shape[0], device="cuda") >= TINY.vocab_size
    noise = std * torch.randn(wte.shape, generator=generator(seed), device="cuda")
    padded["wte.weight"] = torch.where(rows[:, None], noise, wte)
    return padded


SCALES = [1.0 / 192, -1.0, 0.0]


@pytest.mark.parametrize("scale", SCALES, ids=str)
def test_padded_vocabulary_with_zero_rows_gives_zero_logits_and_gradients(flint, tiny_model, scale):
    config = gpt2_config(TINY, 3, 64)
    batch = random_batch(TINY, 3, 64, 21)
    out = run_model(flint, config, state_of(tiny_model), batch, scale)
    assert all_equal(int_view(out.logits)[:, TINY.vocab_size :], 0), "logits past vocab are not +0"
    assert all_equal(int_view(out.dlogits)[:, TINY.vocab_size :], 0), "dlogits past vocab are not +0"
    assert any_unequal(int_view(out.logits)[:, : TINY.vocab_size], 0)


@pytest.mark.parametrize("scale", SCALES, ids=str)
def test_padded_vocabulary_with_nonzero_rows_still_gives_zero_gradients(flint, tiny_model, scale):
    config = gpt2_config(TINY, 3, 64)
    batch = random_batch(TINY, 3, 64, 22)
    state = state_of(tiny_model)
    zero_padded = run_model(flint, config, state, batch, scale)
    nonzero = run_model(flint, config, with_padded_rows(state, 23), batch, scale)
    vocab = TINY.vocab_size
    assert any_unequal(int_view(nonzero.logits)[:, vocab:], 0), "the padded logits should be nonzero here"
    assert all_equal(int_view(nonzero.dlogits)[:, vocab:], 0), "dlogits past vocab are not +0"
    # No token is in the padded range, so only the padded output columns change: the other
    # logits, each row's loss and the gradients are the zero-padded run's bits.
    assert_same_bits(nonzero.logits[:, :vocab], zero_padded.logits[:, :vocab], "logits below vocab")
    assert_same_bits(nonzero.row_loss, zero_padded.row_loss, "row_loss")
    assert_same_bits(nonzero.dlogits, zero_padded.dlogits, "dlogits")
    assert nonzero.loss == zero_padded.loss


def test_a_padded_token_reaches_the_loss_only_through_the_master_rows_it_reads(flint, tiny_model):
    # A token in the padded range (allowed: tokens only need to be below padded_vocab) reads
    # the master row of wte, which is zero or not. Sequence 0 has none; sequence 1 has one
    # at position 20, so its rows before that are untouched too (attention is causal).
    config = gpt2_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 24)
    position = 32 + 20
    tokens = batch.tokens.clone()
    tokens[position] = as_int16(TINY.vocab_size + 3)
    batch.tokens = tokens
    state = state_of(tiny_model)
    scale = 1.0 / batch.rows
    zero_padded = run_model(flint, config, state, batch, scale)
    nonzero = run_model(flint, config, with_padded_rows(state, 25), batch, scale)
    untouched = torch.arange(batch.rows, device="cuda") < position
    assert_same_bits(nonzero.row_loss[untouched], zero_padded.row_loss[untouched], "row_loss before the padded token")
    assert any_unequal(int_view(nonzero.row_loss)[position:], int_view(zero_padded.row_loss)[position:])
    assert all_equal(int_view(nonzero.dlogits)[:, TINY.vocab_size :], 0)
    assert all_equal(int_view(zero_padded.dlogits)[:, TINY.vocab_size :], 0)


def regions_of(m, config):
    """(name, address, bytes) of a model's parameters and outputs."""
    rows, padded = config.sequences * config.length, config.padded_vocab
    logits_at, dlogits_at, loss_at = m.outputs()
    regions = [(n, address, r * c * 4) for n, address, r, c in m.parameters()]
    return regions + [("logits", logits_at, rows * padded * 2), ("dlogits", dlogits_at, rows * padded * 2),
                      ("row_loss", loss_at, rows * 4)]  # fmt: skip


def test_two_live_models_do_not_disturb_each_other(flint, tiny_model, tiny_model_b):
    config_a, config_b = gpt2_config(TINY, 2, 32), gpt2_config(TINY, 3, 64, separate=True)
    batch_a, batch_b = random_batch(TINY, 2, 32, 26), random_batch(TINY, 3, 64, 27)
    other_b = random_batch(TINY, 3, 64, 28)
    solo_a = run_model(flint, config_a, state_of(tiny_model), batch_a, 0.25)
    solo_b = run_model(flint, config_b, state_of(tiny_model_b), batch_b, 0.125)
    solo_other_b = run_model(flint, config_b, state_of(tiny_model_b), other_b, 0.125)
    with Gpt2(flint, config_a) as a, Gpt2(flint, config_b) as b:
        a.load(state_of(tiny_model))
        b.load(state_of(tiny_model_b))
        spans = sorted(regions_of(a, config_a) + regions_of(b, config_b), key=lambda r: r[1])
        for (n1, s1, z1), (n2, s2, _) in zip(spans, spans[1:]):
            assert s1 + z1 <= s2, f"{n1} and {n2} overlap"
        a.forward(batch_a.tokens, batch_a.target_tokens, 0.25)
        b.forward(batch_b.tokens, batch_b.target_tokens, 0.125)
        # B ran after A, and A's outputs and weights are still its own.
        assert_same_outputs(read_outputs(a, config_a), solo_a, "A after B ran")
        assert_same_outputs(read_outputs(b, config_b), solo_b, "B")
        b.forward(other_b.tokens, other_b.target_tokens, 0.125)
        a.forward(batch_a.tokens, batch_a.target_tokens, 0.25)
        assert_same_outputs(read_outputs(a, config_a), solo_a, "A after B's second pass")
        assert_same_outputs(read_outputs(b, config_b), solo_other_b, "B on other tokens")
    # And destroying one leaves the other working.
    with Gpt2(flint, config_a) as a:
        a.load(state_of(tiny_model))
        with Gpt2(flint, config_b) as b:
            b.load(state_of(tiny_model_b))
            b.forward(batch_b.tokens, batch_b.target_tokens, 0.125)
            torch.cuda.synchronize()  # flint.h: a model's work must finish before it's destroyed
        a.forward(batch_a.tokens, batch_a.target_tokens, 0.25)
        assert_same_outputs(read_outputs(a, config_a), solo_a, "A after B was destroyed")


def test_creating_and_destroying_a_model_many_times_leaks_no_device_memory(flint):
    # GPT-2 at (1, 1024) takes about 1 GB. The driver's free memory, on a GPU that desktop apps
    # share, moves by itself, so the baseline's own noise is measured and printed, and the
    # tolerance (LEAK_TOLERANCE) is far above it and far below one model's size.
    cfg = GPT2
    config = gpt2_config(cfg, 1, 1024)
    model = make_model(cfg, 0, "init")
    batch = random_batch(cfg, 1, 1024, 29)
    size = bytes_of(flint, config)
    state = state_of(model)

    def cycle():
        with Gpt2(flint, config) as m:
            m.load(state)
            m.forward(batch.tokens, batch.target_tokens, 1.0 / batch.rows)
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
    print(f"LEAK model {size / MiB:.0f} MiB: free-memory noise {noise / MiB:.1f} MiB, drop while alive "
          f"{drop_during / MiB:.0f} MiB, after 5 create/destroy cycles {leak / MiB:.1f} MiB below the start "
          f"(tolerance {LEAK_TOLERANCE / MiB:.0f} MiB)")  # fmt: skip
    assert drop_during > 0.9 * size, "free memory didn't drop by the model's size while it lived: the measure is blind"
    if noise > size / 4:
        pytest.skip(f"free memory moved {noise / MiB:.0f} MiB by itself: too noisy to measure a leak")
    assert leak < LEAK_TOLERANCE, f"{leak / MiB:.0f} MiB of device memory were not given back"


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_a_forward_pass_on_a_side_stream_gives_the_same_bits(flint, tiny_model, separate):
    config = gpt2_config(TINY, 3, 64, separate)
    batch = random_batch(TINY, 3, 64, 30)
    scale = 1.0 / batch.rows
    state = state_of(tiny_model)
    default = run_model(flint, config, state, batch, scale)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with Gpt2(flint, config) as m:
        # The sleep keeps the stream busy: a step that ran on another stream would race
        # ahead of the weights' copy and give other bits.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.load(state, stream=stream)
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.sync_weights(stream)
            m.forward(batch.tokens, batch.target_tokens, scale, stream)
            side = read_outputs(m, config, stream=stream)
    stream.synchronize()
    assert_same_outputs(side, default, "side stream")


def test_mean_loss_waits_for_its_stream(flint, tiny_model):
    # Nothing but mean_loss itself waits for the side stream's pass, which the sleep holds back.
    # Reading too early, or waiting on another stream, gives the first pass's loss.
    config = gpt2_config(TINY, 3, 64)
    first, second = random_batch(TINY, 3, 64, 31), random_batch(TINY, 3, 64, 32)
    scale = 1.0 / first.rows
    state = state_of(tiny_model)
    expected = run_model(flint, config, state, second, scale).loss
    stream = torch.cuda.Stream()
    with Gpt2(flint, config) as m:
        m.load(state)
        m.forward(first.tokens, first.target_tokens, scale)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.forward(second.tokens, second.target_tokens, scale, stream)
            loss = m.mean_loss(stream)
        stream.synchronize()
    assert loss == expected


# -- 3. Argument errors ---------------------------------------------------------------------

GOOD = dict(layers=2, heads=2, width=128, block_size=64, vocab=50257, padded_vocab=50304,
            sequences=2, length=32, separate_buffers=0)  # fmt: skip
NOT_A_MODEL = 0x1234_5000  # a fake model pointer, which a failing create must replace by null


def call_bytes(flint, fields):
    n = ctypes.c_size_t(0xDEAD)
    config = Gpt2Config(**{**GOOD, **fields})
    status = flint.lib.flint_gpt2_bytes(ctypes.byref(config), ctypes.byref(n))
    return status, n.value


def call_create(flint, fields):
    handle = ctypes.c_void_p(NOT_A_MODEL)
    config = Gpt2Config(**{**GOOD, **fields})
    status = flint.lib.flint_gpt2_create(ctypes.byref(config), ctypes.byref(handle))
    return status, handle.value


# (the arguments the message must name with their values, the fields that break the config)
REJECTED = {
    "layers-0": (["layers"], dict(layers=0)),
    "layers-1025": (["layers"], dict(layers=1025)),
    "layers-2^64-1": (["layers"], dict(layers=2**64 - 1)),
    "width-0": (["width"], dict(width=0)),
    "width-12": (["width"], dict(width=12)),
    "width-16392": (["width"], dict(width=16392)),
    "width-2^64-8": (["width"], dict(width=2**64 - 8)),
    "width-2^64-1": (["width"], dict(width=2**64 - 1)),
    "heads-0": (["heads"], dict(heads=0)),
    "heads-not-a-divisor": (["heads"], dict(heads=5)),
    "heads-width-128-heads-32": (["heads"], dict(width=128, heads=32)),
    "heads-more-than-width": (["heads"], dict(heads=256)),
    "heads-2^64-1": (["heads"], dict(heads=2**64 - 1)),
    "length-0": (["length"], dict(length=0)),
    "length-12": (["length"], dict(length=12)),
    "length-1032": (["length"], dict(length=1032, block_size=1032)),
    "length-2^64-8": (["length"], dict(length=2**64 - 8)),
    "block_size-below-length": (["block_size", "length"], dict(block_size=24)),
    "block_size-0": (["block_size"], dict(block_size=0)),
    "block_size-1025": (["block_size"], dict(block_size=1025)),
    "block_size-2^64-1": (["block_size"], dict(block_size=2**64 - 1)),
    "padded_vocab-0": (["padded_vocab"], dict(padded_vocab=0)),
    "padded_vocab-12": (["padded_vocab"], dict(padded_vocab=12)),
    "padded_vocab-65544": (["padded_vocab"], dict(padded_vocab=65544)),
    "padded_vocab-2^64-8": (["padded_vocab"], dict(padded_vocab=2**64 - 8)),
    "vocab-0": (["vocab", "padded_vocab"], dict(vocab=0)),
    "vocab-past-padded_vocab": (["vocab", "padded_vocab"], dict(vocab=50305)),
    "vocab-2^64-1": (["vocab", "padded_vocab"], dict(vocab=2**64 - 1)),
    "sequences-0": (["sequences", "heads"], dict(sequences=0)),
    "sequences-x-heads-65536": (["sequences", "heads"], dict(sequences=32768)),
    "sequences-x-heads-1-head": (["sequences", "heads"], dict(sequences=65536, heads=1)),
    "sequences-x-heads-wraps-to-0": (["sequences", "heads"], dict(sequences=2**63)),
    "sequences-2^54": (["sequences", "heads"], dict(sequences=2**54)),
    "sequences-2^64-1": (["sequences", "heads"], dict(sequences=2**64 - 1)),
    "sequences-x-length-8388608": (["sequences", "length"], dict(sequences=8192, length=1024, block_size=1024)),
    "sequences-x-length-8388488": (["sequences", "length"], dict(sequences=8193, length=1024, block_size=1024)),
    "sequences-x-length-with-1-head": (["sequences", "length"], dict(sequences=65535, heads=1, length=1024, block_size=1024)),
}  # fmt: skip


def named(message, names, fields):
    """Whether message names each argument with its value, as 'width (12)'."""
    merged = {**GOOD, **fields}
    return all(re.search(rf"(?<![A-Za-z_]){n} \({merged[n]}\)", message) for n in names)


@pytest.mark.parametrize("case", REJECTED, ids=str)
def test_a_rejected_config_fails_bytes_and_create_with_a_message_naming_the_argument(flint, case):
    names, fields = REJECTED[case]
    status, value = call_bytes(flint, fields)
    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert value == 0xDEAD, "bytes was written on an error"
    message = flint.last_error()
    assert "flint_gpt2_bytes" in message and named(message, names, fields), message

    status, model = call_create(flint, fields)
    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert model is None, "create left *model set"
    message = flint.last_error()
    assert "flint_gpt2_create" in message and named(message, names, fields), message


def test_the_message_names_the_first_argument_that_is_wrong(flint):
    # In the order layers, width, heads, length, block_size, padded_vocab, vocab, sequences.
    everything = dict(layers=0, width=12, heads=0, length=12, block_size=0, padded_vocab=12, vocab=0, sequences=0)
    order = ["layers", "width", "heads", "length", "block_size", "padded_vocab", "vocab", "sequences"]
    for i, name in enumerate(order):
        fields = {k: v for k, v in everything.items() if k in order[i:]}
        status, _ = call_bytes(flint, fields)
        assert status == flint_lib.ERROR_INVALID_ARGUMENT
        assert re.search(rf"(?<![A-Za-z_]){name} \(", flint.last_error()), (name, flint.last_error())


@pytest.mark.parametrize("name", [n for n in LIMIT_CONFIGS if n != "smallest"])
def test_a_legal_config_at_its_limits_passes_validation(flint, name):
    # Validation passes, so create gets as far as the GPU: it works if the memory is there
    # and otherwise fails with ERROR_CUDA, never with an invalid argument or a wrapped size.
    fields = LIMIT_CONFIGS[name]
    total = bytes_of(flint, Gpt2Config(**fields))
    assert total == expected_bytes(**fields)
    free = free_memory()
    fits = total < free // 4
    if not fits and total < 2 * free:
        pytest.skip(f"{name} needs {total / GiB:.1f} GiB, which is neither clearly there ({free / GiB:.1f} GiB free) nor clearly not")
    # One allocation that fails needs no partial ones (the next test has those), and
    # allocating the first of many pieces of a multi-GiB model only slows the shared GPU.
    for separate in (0, 1) if fits else (0,):
        status, model = call_create(flint, {**fields, "separate_buffers": separate})
        if fits:
            assert status == flint_lib.OK and model is not None
            flint.lib.flint_gpt2_destroy(model)
        else:
            assert status == flint_lib.ERROR_CUDA, (status, flint.last_error())
            assert model is None
            assert "out of memory" in flint.last_error().lower(), flint.last_error()
        assert free - free_memory() < LEAK_TOLERANCE


def test_the_largest_legal_config_has_the_bytes_of_the_python_sum(flint):
    fields = LIMIT_CONFIGS["largest"]
    total = bytes_of(flint, Gpt2Config(**fields))
    assert total == expected_bytes(**fields)
    assert total > 2**40, "the largest config should be terabytes, and not wrapped"
    print(f"BYTES the largest legal config: {total} bytes = {total / 2**40:.1f} TiB")


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_a_config_too_large_for_the_gpu_fails_create_with_a_cuda_error_and_keeps_nothing(flint, separate):
    # GPT-2 at 8,190 sequences of 1,024 (heads 6, so that the batches are 49,140) is legal and
    # takes terabytes. With separate buffers the parameters (about 0.5 GB) are allocated
    # first, so some allocations succeed before one fails, and all must be given back.
    fields = dict(config_fields(GPT2, 8190, 1024), heads=6, separate_buffers=int(separate))
    total = bytes_of(flint, Gpt2Config(**fields))
    assert total == expected_bytes(**{k: v for k, v in fields.items() if k != "separate_buffers"})
    free, capacity = torch.cuda.mem_get_info()
    assert total > capacity
    before = free_memory()
    status, model = call_create(flint, fields)
    assert status == flint_lib.ERROR_CUDA
    assert model is None
    message = flint.last_error()
    assert "out of memory" in message.lower(), message
    after = free_memory()
    print(f"OOM create ({'separate' if separate else 'one allocation'}): asked {total / GiB:.0f} GiB, {message}; "
          f"free memory {before / MiB:.0f} -> {after / MiB:.0f} MiB")  # fmt: skip
    assert before - after < LEAK_TOLERANCE, "a failed create kept memory"


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_the_library_still_works_right_after_a_failed_create(flint, tiny_model, separate):
    # The very next call must succeed: a failed cudaMalloc leaves its error as the thread's
    # "last error", which the next kernel launch's cudaGetLastError would report as its own
    # unless create clears it.
    fields = dict(config_fields(GPT2, 8190, 1024), heads=6, separate_buffers=int(separate))
    status, model = call_create(flint, fields)
    assert status == flint_lib.ERROR_CUDA and model is None
    config = gpt2_config(TINY, 2, 32)
    batch = random_batch(TINY, 2, 32, 31)
    state = state_of(tiny_model)
    again = run_model(flint, config, state, batch, 0.5)
    expected = compose_forward(flint, config, state, state, batch, 0.5)
    assert_same_outputs(again, expected, "after the failed create", loss=False)
    assert torch.isfinite(again.logits.float()).all()


def test_the_nulls_of_create_and_bytes(flint):
    config = Gpt2Config(**GOOD)
    n = ctypes.c_size_t(0xDEAD)
    handle = ctypes.c_void_p(NOT_A_MODEL)

    assert flint.lib.flint_gpt2_bytes(None, ctypes.byref(n)) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "config is null" in flint.last_error() and "flint_gpt2_bytes" in flint.last_error()
    assert flint.lib.flint_gpt2_bytes(ctypes.byref(config), None) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "bytes is null" in flint.last_error() and "flint_gpt2_bytes" in flint.last_error()
    assert n.value == 0xDEAD

    assert flint.lib.flint_gpt2_create(None, ctypes.byref(handle)) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "config is null" in flint.last_error() and "flint_gpt2_create" in flint.last_error()
    assert handle.value is None, "create left *model set for a null config"
    assert flint.lib.flint_gpt2_create(ctypes.byref(config), None) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "model is null" in flint.last_error() and "flint_gpt2_create" in flint.last_error()
    assert flint.lib.flint_gpt2_create(None, None) == flint_lib.ERROR_INVALID_ARGUMENT


def test_the_nulls_of_a_null_model(flint):
    lib = flint.lib
    info = Gpt2ParameterInfo()
    pointer = ctypes.c_void_p()
    loss = ctypes.c_double(-1.0)
    lib.flint_gpt2_destroy(None)  # does nothing
    assert lib.flint_gpt2_parameter_count(None) == 0
    cases = {
        "flint_gpt2_parameter": lambda: lib.flint_gpt2_parameter(None, 0, ctypes.byref(info)),
        "flint_gpt2_sync_weights": lambda: lib.flint_gpt2_sync_weights(None, None),
        "flint_gpt2_forward": lambda: lib.flint_gpt2_forward(None, None, None, 1.0, None),
        "flint_gpt2_outputs": lambda: lib.flint_gpt2_outputs(None, ctypes.byref(pointer), None, None),
        "flint_gpt2_mean_loss": lambda: lib.flint_gpt2_mean_loss(None, None, ctypes.byref(loss)),
    }  # fmt: skip
    for name, call in cases.items():
        assert call() == flint_lib.ERROR_INVALID_ARGUMENT, name
        assert name in flint.last_error() and "null" in flint.last_error(), (name, flint.last_error())
    assert pointer.value is None and loss.value == -1.0


def test_parameter_and_mean_loss_check_their_other_pointers_and_the_index(flint):
    config = gpt2_config(TINY, 1, 8)
    with Gpt2(flint, config) as m:
        count = flint.lib.flint_gpt2_parameter_count(m.handle)
        assert count == 2 + 12 * TINY.n_layer + 2
        info = Gpt2ParameterInfo()
        for index in (count, count + 1, 2**32, 2**64 - 1):
            assert flint.lib.flint_gpt2_parameter(m.handle, index, ctypes.byref(info)) == flint_lib.ERROR_INVALID_ARGUMENT
            assert f"index ({index}) must be below {count}" in flint.last_error(), flint.last_error()
        assert flint.lib.flint_gpt2_parameter(m.handle, count - 1, ctypes.byref(info)) == flint_lib.OK
        assert info.name == b"ln_f.bias" and (info.rows, info.cols) == (1, TINY.n_embd)
        assert flint.lib.flint_gpt2_parameter(m.handle, 0, None) == flint_lib.ERROR_INVALID_ARGUMENT
        assert "must not be null" in flint.last_error()
        assert flint.lib.flint_gpt2_mean_loss(m.handle, None, None) == flint_lib.ERROR_INVALID_ARGUMENT
        assert "must not be null" in flint.last_error()


def test_outputs_accepts_null_output_pointers(flint):
    config = gpt2_config(TINY, 1, 8)
    with Gpt2(flint, config) as m:
        lib = flint.lib
        assert lib.flint_gpt2_outputs(m.handle, None, None, None) == flint_lib.OK
        logits, dlogits, row_loss = m.outputs()
        assert len({logits, dlogits, row_loss}) == 3
        for which in range(3):
            pointers = [ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()]
            arguments = [ctypes.byref(p) if i == which else None for i, p in enumerate(pointers)]
            assert lib.flint_gpt2_outputs(m.handle, *arguments) == flint_lib.OK
            assert pointers[which].value == (logits, dlogits, row_loss)[which]
            assert [p.value for i, p in enumerate(pointers) if i != which] == [None, None]


class BadForward:
    """A tiny model with its outputs filled with a pattern, to show that a rejected forward
    pass changes nothing."""

    PATTERN16 = as_int16(0x5A5A)
    PATTERN32 = 0x5A5A5A5A

    def __init__(self, flint, state):
        self.flint = flint
        self.config = gpt2_config(TINY, 2, 32)
        self.rows = 64
        self.m = Gpt2(flint, self.config)
        self.m.load(state)
        self.logits_at, self.dlogits_at, self.loss_at = self.m.outputs()
        self.sizes = {"logits": self.rows * TINY.padded_vocab_size * 2,
                      "dlogits": self.rows * TINY.padded_vocab_size * 2, "row_loss": self.rows * 4}  # fmt: skip
        self.addresses = {"logits": self.logits_at, "dlogits": self.dlogits_at, "row_loss": self.loss_at}
        self.batch = random_batch(TINY, 2, 32, 32)
        self.fill()

    def fill(self):
        for name, address in self.addresses.items():
            if name == "row_loss":
                pattern = torch.full((self.sizes[name] // 4,), self.PATTERN32, dtype=torch.int32, device="cuda")
            else:
                pattern = torch.full((self.sizes[name] // 2,), self.PATTERN16, dtype=torch.int16, device="cuda")
            self.flint.copy(address, pattern, self.sizes[name])
        torch.cuda.synchronize()

    def untouched(self):
        torch.cuda.synchronize()
        out = read_outputs(self.m, self.config, mean_loss=False)
        return (all_equal(int_view(out.logits), self.PATTERN16) and all_equal(int_view(out.dlogits), self.PATTERN16)
                and all_equal(int_view(out.row_loss), self.PATTERN32))  # fmt: skip

    def expect_rejected(self, reason, tokens, targets, scale=0.5):
        with pytest.raises(flint_lib.FlintError) as error:
            self.m.forward(tokens, targets, scale)
        assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
        for text in ("flint_gpt2_forward", reason):
            assert text in str(error.value) and text in self.flint.last_error(), (text, str(error.value))
        assert self.untouched(), f"a rejected forward pass ({reason}) changed the outputs"

    def close(self):
        self.m.close()


@pytest.fixture
def bad_forward(flint, tiny_model):
    bad = BadForward(flint, state_of(tiny_model))
    yield bad
    bad.close()


@pytest.mark.parametrize("which", ["tokens", "targets"])
def test_forward_rejects_a_null_pointer(bad_forward, which):
    b = bad_forward.batch
    arguments = {"tokens": b.tokens, "targets": b.target_tokens}
    arguments[which] = None
    bad_forward.expect_rejected(f"{which} is null", arguments["tokens"], arguments["targets"])


@pytest.mark.parametrize("offset", [2, 8])
@pytest.mark.parametrize("which", ["tokens", "targets"])
def test_forward_rejects_a_misaligned_pointer(bad_forward, which, offset):
    # 2 bytes is as far off as an element goes, and 8 is aligned for 64 bits but not 128.
    b = bad_forward.batch
    room = torch.zeros(b.rows + 64, dtype=torch.int16, device="cuda")
    arguments = {"tokens": b.tokens, "targets": b.target_tokens}
    arguments[which] = room.data_ptr() + offset
    bad_forward.expect_rejected(f"{which} is not 16-byte aligned", arguments["tokens"], arguments["targets"])


@pytest.mark.parametrize("scale", [math.nan, math.inf, -math.inf], ids=str)
def test_forward_rejects_a_non_finite_scale(bad_forward, scale):
    b = bad_forward.batch
    bad_forward.expect_rejected("scale (", b.tokens, b.target_tokens, scale)
    bad_forward.expect_rejected(") must be finite", b.tokens, b.target_tokens, scale)


OVERLAPS = ["same-start", "16-in", "starts-16-before-end", "ends-16-in"]


@pytest.mark.parametrize("label", OVERLAPS)
@pytest.mark.parametrize("output", ["logits", "dlogits", "row_loss"])
@pytest.mark.parametrize("which", ["tokens", "targets"])
def test_forward_rejects_tokens_or_targets_that_overlap_an_output(bad_forward, which, output, label):
    # The model's own buffers are the ones a caller could hand back by mistake. The region
    # of 2 * rows bytes is placed on the output's start, 16 bytes in, starting 16 bytes before
    # its end, and ending 16 bytes into it.
    b, size, rows_bytes = bad_forward.batch, bad_forward.sizes[output], bad_forward.rows * 2
    start = {"same-start": 0, "16-in": 16, "starts-16-before-end": size - 16, "ends-16-in": 16 - rows_bytes}[label]
    address = bad_forward.addresses[output] + start
    # The first output, in the order logits, dlogits, row_loss, that the region overlaps.
    first = next(
        name for name in ("logits", "dlogits", "row_loss")
        if address < bad_forward.addresses[name] + bad_forward.sizes[name] and bad_forward.addresses[name] < address + rows_bytes
    )  # fmt: skip
    arguments = {"tokens": b.tokens, "targets": b.target_tokens}
    arguments[which] = address
    bad_forward.expect_rejected(f"{first} and {which} overlap", arguments["tokens"], arguments["targets"])


@pytest.mark.parametrize("gap", [0, 8, 16], ids=["same-memory", "16-bytes-apart", "32-bytes-apart"])
def test_tokens_and_targets_may_share_memory(flint, tiny_model, gap):
    # They are only read. Every id is below vocab, so the same ids work as tokens and targets.
    label = f"{2 * gap} bytes apart"
    config = gpt2_config(TINY, 2, 32)
    arena = as_tokens(random_ids(TINY, 64 + 16, 33))
    tokens, targets = arena[:64], arena[gap : gap + 64]
    separate = types.SimpleNamespace(tokens=tokens.clone(), target_tokens=targets.clone())
    state = state_of(tiny_model)
    expected = run_model(flint, config, state, separate, 0.25)
    with Gpt2(flint, config) as m:
        m.load(state)
        m.forward(tokens, targets, 0.25)
        shared = read_outputs(m, config)
    assert_same_outputs(shared, expected, f"tokens and targets sharing memory ({label})")


# -- 4. The sanitizer -------------------------------------------------------------------------


@pytest.mark.parametrize("seeds", [(0, 1)], ids=str)
@pytest.mark.parametrize("sequences, length", [(1, 8), (2, 16)], ids=str)
def test_sanitized_tiny_forward_with_separate_buffers(flint, sequences, length, seeds):
    # separate_buffers gives each buffer its own allocation, so memcheck sees each one's
    # bounds. Small shapes keep the sanitized run short.
    gate_case(flint, "tiny", sequences, length, "mixed", "random", seeds=seeds, separate=True)


def test_sanitized_tiny_forward_with_one_allocation(flint):
    config = gpt2_config(TINY, 2, 16)
    model = make_model(TINY, 0, "mixed")
    batch = random_batch(TINY, 2, 16, 40)
    scale = 1.0 / batch.rows
    one = run_model(flint, config, state_of(model), batch, scale)
    separate = run_model(flint, gpt2_config(TINY, 2, 16, separate=True), state_of(model), batch, scale)
    assert_same_outputs(one, separate, "one allocation against separate buffers")
    gate_case(flint, "tiny", 2, 16, "mixed", "random", seeds=(0,), separate=False)
