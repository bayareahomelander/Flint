"""flint_adamw: one optimizer step, as PyTorch 2.6 does it with FP16 autocast, in two kernels that
never wait on the host: GradScaler's unscale_ and its found_inf, clip_grad_norm_, a fused AdamW
(decoupled weight decay), and GradScaler's update. flint_gradient_norm has already left the FP64
norm of the still-scaled FP32 gradients and the count of their values that are not finite; this
reads both, and a 24-byte state (the loss scale, the growth tracker, AdamW's step count and the
count of skipped steps) in device memory.

  - The step is skipped if the count is above 0, if the loss scale isn't finite and positive (a
    state no step or reset leaves, which then skips every step), if 1 / scale overflows FP32
    (GradScaler's own inv_scale: a scale of 0, or about 2^-128 or less), or if the unscaled norm
    is 2^127 or more, past which an unscaled value could overflow (every value GradScaler's
    unscale_ would overflow is covered, and nothing below the bound overflows). A skip halves the scale, resets the tracker and counts
    itself, and changes no weight, no moment and not AdamW's step count.
  - Otherwise unscaling and clipping fold into one multiplier, fl32(clip / scale), with the unscaled
    norm norm / scale and clip = min(1, max_norm / (unscaled norm + 1e-6)); no unscaled gradient is
    ever written. The step count advances, and the loss scale update follows: the tracker counts
    clean steps and at the growth interval the scale doubles (unless that gives inf).
  - Each value then goes through AdamW's arithmetic in the precision mix of PyTorch's fused
    kernel, whose scalars are doubles: g = gradient * multiplier in FP32; the decay, the moments
    and the denominator's eps added in FP64 and rounded to FP32; the rest in FP32. data/d32/
    precision.log measured why: over 100 steps against an FP64 reference, the same arithmetic
    with every operation in FP32 had moments with 3.5 times the error of PyTorch's own FP32 fused
    AdamW, past the gate's 2 times, while this mix stays at the peer's (0.9 times).
  - Every operation is an explicitly rounded intrinsic, nothing is summed across values and there
    are no atomics, so the bits are the same on every GPU, and a numpy transcription of the
    arithmetic (PREPARE and UPDATE below, the documented definition of the bits) must match
    them exactly.

What the tests cover:
  - the bit-exact comparison with the transcription: params, both moments and the state after
    every step of sequences with clipped and unclipped steps, decay on and off, a growth interval
    of 3 so the scale grows, and skips, on counts 0 to 2,049, more chunks than blocks (block 0
    wrapping), many tensors, misaligned arrays (which also give an aligned copy's bits), GPT-2
    124M's 148 sizes, and values' edges (zeros, -0, subnormals, FP32's maximum, lr 0, beta 0);
  - against PyTorch's fused AdamW, bit for bit where the arithmetic must agree (an unclipped step
    on its own gradients: about 99.99% of the values, the rest one ulp from PyTorch's FMA
    contraction), and by D26's rule (PLAN.md: at most twice the FP32 peer's error against an FP64
    reference whose state stays FP64, over 3 seeds) after 1, 2, 20 and 100 steps, clipped and
    unclipped; the reference itself is checked against PyTorch's float64 path;
  - the state, exactly GradScaler's: scale, growth tracker and AdamW's step count equal the peer's
    after every step of random skip patterns, at growth intervals 1, 3 and 2,000, and the edges of
    the skip rule, whose verdict is GradScaler's found_inf;
  - clipping below, at and above max_norm, with no clipping and with a tiny max_norm, and that
    the unscaled norm decides;
  - what the call reads and writes: guard bands around every array, the state and the workspace,
    the inputs unchanged, a workspace full of garbage or larger than needed, entries of count 0
    with garbage pointers; determinism, streams and the argument errors;
  - the model: flint_gpt2_step equals flint_adamw on copies of its weights, gradients, moments and
    state, and the transcription, with the FP16 copies current afterwards; decay by name, a
    skipped step, resuming from another model's state, a short training run, and the errors and
    layout of the new functions.

Under the sanitizer (scripts/sanitize.py) the inputs shrink to a few hundred thousand values at
most and a handful of steps, and the large and long cases are skipped; the wrap of the chunks over
the blocks is covered there by a table of 1,100 tiny tensors."""

import ctypes
import gc
import math
import random
import struct
import types
from collections import namedtuple

import numpy as np
import pytest
import torch

import flint_lib
import gate
from flint_lib import Gpt2, Gpt2Config
from inputs import SANITIZER, SIDE_STREAM_SLEEP_CYCLES
from test_gpt2 import (  # also puts scripts/ on sys.path
    ODD_MODELS,
    TINY,
    all_equal,
    bytes_of,
    gate_scale,
    gpt2_config,
    make_model,
    random_batch,
    read_outputs,
    state_of,
)
from test_gpt2_backward import (
    BACKWARD_BYTES,
    backward_config,
    expected_bytes_backward,
    forward_backward,
    poison_gradients,
)
from test_gradient_norm import NON_FINITE_BITS, as_int32, gpt2_sizes

import gpt2_torch as G  # noqa: E402  (test_gpt2 put scripts/ on sys.path)

F32, F64 = np.float32, np.float64
SCALE = 2.0**16
SENTINEL_BITS = 0x5A5A5A5A
HEAD = 4  # floats of guard before an array in a World's buffers
TAIL = 8  # floats of guard after it


@pytest.fixture(autouse=True)
def free_gpu_memory():
    """The laptop's GPU is 6 GiB and shared, so each test hands back what it cached."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


# -- The transcription: the documented definition of the bits ---------------------------------------
# (numpy's float32 + - * / and sqrt are correctly rounded, and Python floats are IEEE doubles.
# Only the divisions by a scale are numpy float64 here, so that a scale of 0 gives inf or NaN
# as the kernel's does instead of raising ZeroDivisionError.)


def pow_by_squaring(base, t):  # base ** t for an integer t >= 1, in FP64
    result, e = 1.0, t
    while e:
        if e & 1:
            result = result * base
        e >>= 1
        if e:
            base = base * base
    return result


def prepare(norm, nonfinite, state, c):
    """The first kernel: the step's scalars, then the new state. norm is the FP64 value,
    nonfinite the count, state a dict, c the config."""
    with np.errstate(all="ignore"):
        scale = float(F32(state["scale"]))
        unscaled = F64(norm) / F64(scale)
        clip = F64(c.max_norm) / (unscaled + 1e-6)
        if not clip < 1.0:  # clamp(max=1); also catches inf / inf
            clip = 1.0
        multiplier = F32(clip / F64(scale))
        inverse = F32(F64(1.0) / F64(scale))  # GradScaler's inv_scale
        skip = (nonfinite != 0 or not (scale > 0.0 and math.isfinite(scale))
                or not np.isfinite(inverse) or not unscaled < 2.0**127)
        new = dict(state)
        if skip:
            new["scale"] = F32(scale) * F32(0.5)
            new["growth_tracker"] = 0
            new["skipped"] = state["skipped"] + 1
            return dict(skip=True), new
        t = state["step"] + 1
        bc1 = F32(1.0 - pow_by_squaring(c.beta1, t))
        bc2_sqrt = F32(math.sqrt(1.0 - pow_by_squaring(c.beta2, t)))
        step_size = F32(c.lr / float(bc1))
        new["step"] = t
        successful = state["growth_tracker"] + 1
        if successful >= c.growth_interval:
            grown = F32(scale) * F32(2.0)
            if np.isfinite(grown):
                new["scale"] = grown
            new["growth_tracker"] = 0
        else:
            new["growth_tracker"] = successful
        return dict(skip=False, multiplier=multiplier, step_size=step_size, bc2_sqrt=bc2_sqrt), new


def update(p, grad, m, v, decay, s, c):
    """The second kernel, on one tensor's float32 arrays; returns the new p, m, v (on a skip,
    the kernel touches nothing)."""
    with np.errstate(all="ignore"):
        g = grad * s["multiplier"]
        if decay and c.weight_decay != 0:
            p64 = p.astype(F64)
            p = (p64 - (c.lr * c.weight_decay) * p64).astype(F32)
        g64 = g.astype(F64)
        m = (c.beta1 * m.astype(F64) + (1.0 - c.beta1) * g64).astype(F32)
        v = (c.beta2 * v.astype(F64) + ((1.0 - c.beta2) * g64) * g64).astype(F32)
        denom = ((np.sqrt(v) / s["bc2_sqrt"]).astype(F64) + c.eps).astype(F32)
        p = p - (s["step_size"] * m) / denom
        return p, m, v


def namespace(config):
    """A flint_adamw_config (or a dict of its fields) as the object the transcription reads."""
    fields = ("lr", "beta1", "beta2", "eps", "weight_decay", "max_norm", "growth_interval")
    if isinstance(config, dict):
        return types.SimpleNamespace(**{k: config[k] for k in fields})
    return types.SimpleNamespace(**{k: getattr(config, k) for k in fields})


def initial_state(scale=SCALE, growth_tracker=0, step=0, skipped=0):
    return {"scale": F32(scale), "growth_tracker": growth_tracker, "step": step, "skipped": skipped}


def state_bits(state):
    return struct.pack("<fiqq", float(state["scale"]), state["growth_tracker"], state["step"],
                       state["skipped"])  # fmt: skip


def states_equal(device_state, expected):
    return state_bits(device_state) == state_bits(expected)


# -- The reference: the same algorithm in float64, with the state in float64 -------------------------


def reference_step(P, M, V, scaled, decay, scale, t, c):
    """One step on float64 arrays, on FP32 scaled gradients: the unscale by the float64 scale, the
    clip coefficient from the float64 norm of the unscaled gradients, then AdamW with t the
    number of the step (1, 2, ...). On a gradient that isn't finite the step is skipped: the
    same arrays and False come back. decay is a boolean array."""
    scaled = scaled.astype(F64)
    if not np.isfinite(scaled).all():
        return P, M, V, False
    g = scaled / scale
    norm = math.sqrt(float(np.dot(g, g)))
    g = g * min(1.0, c.max_norm / (norm + 1e-6))
    bc1 = 1.0 - c.beta1**t
    bc2 = 1.0 - c.beta2**t
    P = np.where(decay, P - (c.lr * c.weight_decay) * P, P)
    M = c.beta1 * M + (1.0 - c.beta1) * g
    V = c.beta2 * V + (1.0 - c.beta2) * g * g
    P = P - (c.lr / bc1) * M / (np.sqrt(V) / math.sqrt(bc2) + c.eps)
    return P, M, V, True


def step_f32(p, m, v, g, decay, c, t):
    """The all-FP32 variant of the update (data/d32/precision.py's), for the print that shows why
    the mix is used: the scalars in FP64 rounded once, every operation in FP32."""
    bc1 = 1.0 - c.beta1**t
    bc2 = 1.0 - c.beta2**t
    decay_factor = F32(c.lr * c.weight_decay)
    step_size = F32(c.lr / bc1)
    bc2_sqrt = F32(math.sqrt(bc2))
    p = np.where(decay, p - decay_factor * p, p)
    m = F32(c.beta1) * m + F32(1.0 - c.beta1) * g
    v = F32(c.beta2) * v + (F32(1.0 - c.beta2) * g) * g
    denominator = np.sqrt(v) / bc2_sqrt + F32(c.eps)
    p = p - (step_size * m) / denominator
    return p, m, v


# -- Inputs: integers times powers of two, the same on every machine ----------------------------------


def draw(n, low, high, g):
    """n integers in [low, high] as exact FP32, from the seeded CPU generator g."""
    return torch.randint(low, high + 1, (n,), generator=g, dtype=torch.int32).float()


# Tensors above this many values are drawn in chunks of it, each from a generator of its own (the
# first chunk's is the tensor's), so that GPT-2's biggest tensors never sit in host memory whole.
CHUNK_VALUES = 1 << 22


def chunk_seed(base, c):
    return base + c * 1_000_000_007


def chunks_of(n):
    """(chunk index, first value, one past the last) of each chunk of n values (one for n = 0)."""
    return [(c, lo, min(n, lo + CHUNK_VALUES)) for c, lo in enumerate(range(0, max(n, 1), CHUNK_VALUES))]


def weights_chunk(i, n, seed, c):
    """The chunk c of the i-th tensor's FP32 weights, about N(0, 0.02): four uniform integers' sum
    times 2^-14."""
    size = chunks_of(n)[c][2] - chunks_of(n)[c][1]
    g = torch.Generator().manual_seed(chunk_seed(7919 * seed + i, c))
    return (sum(draw(size, -256, 256, g) for _ in range(4)) * 2.0**-14).numpy()


def tensor_weights(i, n, seed):
    return np.concatenate([weights_chunk(i, n, seed, c) for c, _, _ in chunks_of(n)])


def gradient_chunk(i, n, seed, step, c):
    """The chunk c of the i-th tensor's plain unscaled gradient: fresh integers of up to 10 bits
    (never 0)."""
    size = chunks_of(n)[c][2] - chunks_of(n)[c][1]
    noise = torch.Generator().manual_seed(chunk_seed(1_000_003 * seed + 1009 * step + i, c))
    integer = draw(size, -1023, 1023, noise)
    return torch.where(integer == 0, torch.ones_like(integer), integer).numpy()


def initial_weights(counts, seed):
    return [tensor_weights(i, n, seed) for i, n in enumerate(counts)]


def tensor_gradient(i, n, seed, step, wide=False):
    """The i-th tensor's unscaled gradient at this step, from generators of its own, so that any
    tensor can be made again alone. Plain: fresh integers of up to 10 bits (never 0) each step.
    Wide, GPT-2-like: each value's exponent is fixed (before the norm's power of two, from 2^-27 to
    2^-7), its integer is a persistent part plus the step's noise, 22 bits each, so that two
    steps' values almost never cancel in the first moment to far below an FP32 ulp (which would
    make the FMA that PyTorch's kernel uses and the FP64 arithmetic here differ by many ulps of a
    tiny result), and 1 in 1,000 values has no gradient at all."""
    if wide:
        persistent = torch.Generator().manual_seed(1_000_003 * seed + 17 * i + 1)
        noise = torch.Generator().manual_seed(1_000_003 * seed + 1009 * step + i + 5_000_000_000)
        exponent = (draw(n, -27, -7, persistent) - 21).numpy().astype(np.int32)
        keep = (draw(n, 0, 999, persistent) != 0).numpy()
        integer = (draw(n, -(2**21), 2**21, persistent) + draw(n, -(2**21), 2**21, noise)).numpy()
        return np.ldexp(integer * keep, exponent).astype(F32)
    return np.concatenate([gradient_chunk(i, n, seed, step, c) for c, _, _ in chunks_of(n)])


def power_of_two_for(squares, target, scale=SCALE):
    """The exponent k that puts the unscaled norm of values whose squares sum to `squares`, times
    2^k and over scale, within sqrt(2) of target."""
    return int(round(math.log2(target * scale / math.sqrt(squares)))) if squares > 0 else 0


def make_gradients(counts, seed, step, target, scale=SCALE, wide=False):
    """Scaled FP32 gradients, one array per count: tensor_gradient's values times a power of two
    (exact) that puts the unscaled norm, norm / scale, within sqrt(2) of target."""
    values = [tensor_gradient(i, n, seed, step, wide) for i, n in enumerate(counts)]
    k = power_of_two_for(sum(float(np.sum(a.astype(F64) ** 2)) for a in values), target, scale)
    return [np.ldexp(a, k).astype(F32) for a in values]


def inject(grads, kind="inf"):
    """Writes a non-finite value (a key of NON_FINITE_BITS) into the last non-empty array's 6th
    value (or its last); returns where, or None if all the arrays are empty."""
    for i in range(len(grads) - 1, -1, -1):
        if grads[i].size:
            at = min(5, grads[i].size - 1)
            grads[i].view(np.uint32)[at] = NON_FINITE_BITS[kind]
            return i, at
    return None


# -- Comparing -----------------------------------------------------------------------------------------


def mismatch(device, host, chunk=1 << 24):
    """The index of the first float32 whose bits differ between a flat device tensor and a flat
    numpy array, or None."""
    for lo in range(0, device.numel(), chunk):
        d = device[lo : lo + chunk].view(torch.int32)
        h = torch.from_numpy(host[lo : lo + chunk]).cuda().view(torch.int32)
        if not torch.equal(d, h):
            return int((d != h).nonzero()[0].item()) + lo
    return None


def ordered(a):
    """FP32 values as integers that are ordered as the values are (and equal for -0 and +0), so
    that the difference of two is their distance in ulps."""
    i = np.ascontiguousarray(a).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulps_between(a, b):
    return np.abs(ordered(a) - ordered(b))


# -- The peer: PyTorch's fused AdamW behind GradScaler, as scripts/gpt2_torch.py's train_step ----------


class Peer:
    """FP32 CUDA parameters (clones of the arrays given) and what train_step does after the
    backward pass: GradScaler.unscale_, clip_grad_norm_, the fused AdamW with a group of weight
    decay for the decayed arrays and one without, GradScaler.step and update. step takes the
    scaled FP32 gradients as CUDA tensors, the same ones Flint reads."""

    def __init__(self, arrays, decays, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                 max_norm=1.0, init_scale=SCALE, growth_interval=2000):  # fmt: skip
        self.params = [torch.nn.Parameter(torch.as_tensor(a).float().cuda().clone()) for a in arrays]
        groups = [{"params": [p for p, d in zip(self.params, decays) if d], "weight_decay": weight_decay},
                  {"params": [p for p, d in zip(self.params, decays) if not d], "weight_decay": 0.0}]  # fmt: skip
        groups = [g for g in groups if g["params"]]
        self.opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=True)
        self.scaler = torch.amp.GradScaler("cuda", init_scale=init_scale, growth_interval=growth_interval)
        self.max_norm = max_norm

    def step(self, scaled_grads):
        self.scaler.scale(torch.ones((), device="cuda"))  # creates the scale and the tracker
        for p, g in zip(self.params, scaled_grads):
            p.grad = g.clone()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.params, self.max_norm)
        self.scaler.step(self.opt)
        self.scaler.update()

    @property
    def scale(self):
        return float(self.scaler._scale.item())

    @property
    def growth_tracker(self):
        return int(self.scaler._growth_tracker.item())

    @property
    def step_count(self):
        """The fused optimizer's step count (0 before its first step, which a skip never takes)."""
        state = self.opt.state.get(self.params[0])
        return int(state["step"].item()) if state else 0

    def values(self, which):
        """The concatenated parameters ("p") or moments ("m", "v") as a numpy float32 array."""
        if which == "p":
            parts = [p.detach() for p in self.params]
        else:
            name = {"m": "exp_avg", "v": "exp_avg_sq"}[which]
            parts = [self.opt.state[p][name] if p in self.opt.state else torch.zeros_like(p) for p in self.params]
        return torch.cat([x.reshape(-1) for x in parts]).cpu().numpy()

    def arrays(self, which):
        """As values, one numpy array per parameter."""
        if which == "p":
            return [p.detach().cpu().numpy().copy() for p in self.params]
        name = {"m": "exp_avg", "v": "exp_avg_sq"}[which]
        return [self.opt.state[p][name].cpu().numpy().copy() for p in self.params]


# -- Flint's side: arrays on the GPU beside the transcription's on the host -----------------------------

STANDARD = [("clip", None), ("free", None), ("free", "inf"), ("free", "nan"),
            ("clip", None), ("free", None), ("free", None), ("clip", None)]  # fmt: skip
# The sanitizer runs a handful of steps: the first 6, with a growth interval of 2, which grows the
# scale twice (at the 2nd and the 6th step) where 8 steps with an interval of 3 grow it once.
SEQUENCE = STANDARD[:6] if SANITIZER else STANDARD
GROWTH_INTERVAL = 2 if SANITIZER else 3
GROWTHS = 2 if SANITIZER else 1
TARGETS = {"clip": 2.0, "free": 0.6}
StepResult = namedtuple("StepResult", "skip multiplier unscaled norm nonfinite")


def default_config(**kwargs):
    kwargs.setdefault("lr", 6e-4)
    kwargs.setdefault("growth_interval", GROWTH_INTERVAL)
    return flint_lib.adamw_config(**kwargs)


class World:
    """tensors of these counts (decays: whether each decays), each of the four arrays (param, grad,
    exp_avg, exp_avg_sq) living in a flat buffer of its own, filled with a sentinel pattern around
    the arrays (HEAD floats before each, TAIL after), starting `offsets` floats past a 16-byte
    boundary (a tuple for the four arrays, or a function of the entry's index to one). The state
    and the workspace sit inside buffers of a sentinel as well. The transcription runs beside:
    step() runs gradient_norm and flint_adamw on gradients (host float32 arrays), then compares
    every array of every entry with the transcription's, bit for bit, the guards, the gradients,
    the norm, the count and the table with what they were, and the state with the
    transcription's. A mismatch fails with where."""

    def __init__(self, flint, counts, decays=None, *, seed=0, offsets=(0, 0, 0, 0), config=None,
                 state=None, params=None, state_offset=16, workspace_offset=16, workspace_extra=0,
                 garbage_entries=None):  # fmt: skip
        self.flint, self.counts = flint, list(counts)
        self.decays = [True] * len(counts) if decays is None else list(decays)
        self.config = config or default_config()
        offsets_of = offsets if callable(offsets) else (lambda i: offsets)
        # The layout of each of the four flat buffers.
        self.starts = [[] for _ in range(4)]
        self.sizes = []
        for k in range(4):
            cursor = 0
            for i, n in enumerate(self.counts):
                start = -(-cursor // 4) * 4 + HEAD + offsets_of(i)[k]
                self.starts[k].append(start)
                cursor = start + n + TAIL
            self.sizes.append(cursor + HEAD)
        self.host = [np.full(size, SENTINEL_BITS, dtype=np.int32).view(F32) for size in self.sizes]
        self.dev = [torch.from_numpy(h.copy()).cuda() for h in self.host]
        self.views = [[self.host[k][s : s + n] for s, n in zip(self.starts[k], self.counts)] for k in range(4)]
        self.dviews = [[self.dev[k][s : s + n] for s, n in zip(self.starts[k], self.counts)] for k in range(4)]
        p0 = params if params is not None else initial_weights(self.counts, 100 + seed)
        self.p0 = [np.array(a, dtype=F32) for a in p0]
        for i, a in enumerate(self.p0):
            self.views[0][i][:] = a
            self.views[2][i][:] = 0
            self.views[3][i][:] = 0
        for k in (0, 2, 3):
            self.dev[k].copy_(torch.from_numpy(self.host[k]))
        # The table, with entries of count 0 and garbage pointers at the places asked for.
        entries = [(self.dviews[0][i].data_ptr(), self.dviews[1][i].data_ptr(),
                    self.dviews[2][i].data_ptr(), self.dviews[3][i].data_ptr(), n, d)
                   for i, (n, d) in enumerate(zip(self.counts, self.decays))]  # fmt: skip
        for position, entry in sorted((garbage_entries or {}).items()):
            entries.insert(position, entry)
        self.table = flint_lib.adamw_table(entries)
        self.table_count = len(entries)
        self.grad_table = flint_lib.tensor_table(self.dviews[1])
        # The state and the workspace, in buffers of sentinel bytes.
        self.state_at, self.workspace_at = state_offset, workspace_offset
        self.workspace_bytes = flint.adamw_workspace() + workspace_extra
        self.state_buffer = torch.full((state_offset + 24 + 64,), 0x5A, dtype=torch.uint8, device="cuda")
        self.workspace_buffer = torch.full((workspace_offset + self.workspace_bytes + 64,), 0x5A,
                                           dtype=torch.uint8, device="cuda")  # fmt: skip
        self.state = self.state_buffer[state_offset : state_offset + 24]
        self.workspace = self.workspace_buffer[workspace_offset : workspace_offset + self.workspace_bytes]
        self.np_state = initial_state(**(state or {}))
        self.write_state(self.np_state)
        self.norm = torch.zeros(1, dtype=torch.float64, device="cuda")
        self.nonfinite = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.norm_workspace = torch.zeros(flint.gradient_norm_workspace(), dtype=torch.uint8, device="cuda")

    def write_state(self, state):
        self.state.copy_(flint_lib.adamw_state(float(state["scale"]), state["growth_tracker"],
                                               state["step"], state["skipped"]))  # fmt: skip
        torch.cuda.synchronize()

    def read_state(self):
        return flint_lib.read_adamw_state(self.state)

    def set_arrays(self, p=None, m=None, v=None):
        """Overwrites the params and moments of every entry (lists of numpy float32 arrays)."""
        for k, arrays in ((0, p), (2, m), (3, v)):
            if arrays is None:
                continue
            for i, a in enumerate(arrays):
                self.views[k][i][:] = a
            self.dev[k].copy_(torch.from_numpy(self.host[k]))

    def device_values(self, k):
        """The concatenated values of every entry of array k (0 param, 2 exp_avg, 3 exp_avg_sq)
        as the GPU holds them, a numpy float32 array."""
        return torch.cat([d for d in self.dviews[k] if d.numel()]).cpu().numpy() if sum(self.counts) else np.zeros(0, F32)

    def where(self, k, index):
        """Which entry, and which element of it, the flat index of array k is."""
        starts = np.array(self.starts[k])
        i = int(np.searchsorted(starts, index, side="right")) - 1
        if i < 0:
            return f"the guard before entry 0 (index {index})"
        return f"entry {i} of {len(self.counts)} (count {self.counts[i]}) element {index - self.starts[k][i]}"

    def begin(self, grads, *, norm=None, nonfinite=None, workspace_fill="sentinel", defer=False):
        """Everything before the call: the gradients into their arrays (with defer, left out, and
        a device copy of them in pending["source"] for the test to copy over on its own stream),
        then the norm and the count from flint_gradient_norm, or the values given, written by
        hand; the workspace's 16 bytes filled as asked. Returns what finish compares with."""
        for i, g in enumerate(grads):
            self.views[1][i][:] = g
        pending = {"source": None, "norm_bits": None, "count_bits": None}
        if defer:
            pending["source"] = torch.from_numpy(self.host[1].copy()).cuda()
        else:
            self.dev[1].copy_(torch.from_numpy(self.host[1]))
            if norm is None:
                self.flint.gradient_norm(self.norm, self.nonfinite, self.grad_table, len(self.counts),
                                         self.norm_workspace)  # fmt: skip
            else:
                self.norm.copy_(torch.tensor([norm], dtype=torch.float64))
                self.nonfinite.copy_(torch.tensor([nonfinite], dtype=torch.int64))
        used = self.workspace[:16]
        if workspace_fill == "zero":
            used.zero_()
        elif workspace_fill == "garbage":
            used.copy_(torch.randint(0, 256, (16,), dtype=torch.int64, device="cuda").to(torch.uint8))
        elif workspace_fill == "ones":
            used.fill_(0xFF)
        else:
            used.fill_(0x5A)
        torch.cuda.synchronize()
        if not defer:
            pending["norm_bits"] = self.norm.view(torch.int64).clone()
            pending["count_bits"] = self.nonfinite.clone()
        pending["table"] = self.table.clone()
        return pending

    def launch(self, config=None, stream=None, workspace=None):
        """flint_adamw on the world's table, norm, count and state, and its workspace or the given
        one (a uint8 tensor)."""
        ws = self.workspace if workspace is None else workspace
        self.flint.adamw(self.table, self.table_count, self.norm, self.nonfinite, self.state,
                         config or self.config, ws, ws.numel(), stream)  # fmt: skip

    def finish(self, pending, config=None, check_state=True):
        """After the call: runs the transcription on the norm and the count the call read, and
        compares every array, the guards, the inputs and the state."""
        ns = namespace(config or self.config)
        torch.cuda.synchronize()
        norm_value, count = self.norm.item(), self.nonfinite.item()
        scale_before = float(F32(self.np_state["scale"]))
        with np.errstate(all="ignore"):
            unscaled = float(F64(norm_value) / F64(scale_before))
        scalars, new_state = prepare(norm_value, count, self.np_state, ns)
        if not scalars["skip"]:
            for i in range(len(self.counts)):
                p, m, v = update(self.views[0][i], self.views[1][i], self.views[2][i],
                                 self.views[3][i], self.decays[i], scalars, ns)  # fmt: skip
                self.views[0][i][:] = p
                self.views[2][i][:] = m
                self.views[3][i][:] = v
        self.np_state = new_state
        for k, label in ((0, "param"), (2, "exp_avg"), (3, "exp_avg_sq"), (1, "grad")):
            bad = mismatch(self.dev[k], self.host[k])
            assert bad is None, (f"{label}: bits differ from the transcription at {self.where(k, bad)} (step "
                                 f"{'skipped' if scalars['skip'] else 'taken'}; a guard if past an array)")  # fmt: skip
        if pending["norm_bits"] is not None:
            assert torch.equal(self.norm.view(torch.int64), pending["norm_bits"]), "the norm changed"
            assert torch.equal(self.nonfinite, pending["count_bits"]), "the count changed"
        assert torch.equal(self.table, pending["table"]), "the table changed"
        for name, buffer, at, used_bytes in (("state", self.state_buffer, self.state_at, 24),
                                             ("workspace", self.workspace_buffer, self.workspace_at, 16)):  # fmt: skip
            around = torch.cat([buffer[:at], buffer[at + used_bytes :]])
            assert all_equal(around, 0x5A), f"the call wrote around the {name}"
        if check_state:
            device_state = self.read_state()
            assert states_equal(device_state, self.np_state), (
                f"state {device_state} against the transcription's {self.np_state}")
        return StepResult(scalars["skip"], scalars.get("multiplier"), unscaled, norm_value, count)

    def step(self, grads, config=None, *, norm=None, nonfinite=None, stream=None,
             workspace_fill="sentinel"):  # fmt: skip
        """begin, launch and finish."""
        pending = self.begin(grads, norm=norm, nonfinite=nonfinite, workspace_fill=workspace_fill)
        self.launch(config, stream)
        return self.finish(pending, config)


def run_standard(world, sequence=SEQUENCE, seed=1, wide=False, **step_arguments):
    """The standard sequence (STANDARD, or its first 6 steps under the sanitizer) on a world: a
    list of (kind, injection, scale before, scale after, StepResult)."""
    results = []
    for number, (kind, injection) in enumerate(sequence):
        grads = make_gradients(world.counts, seed, number, TARGETS[kind], wide=wide)
        if injection:
            inject(grads, injection)
        before = float(world.np_state["scale"])
        result = world.step(grads, **step_arguments)
        results.append((kind, injection, before, float(world.np_state["scale"]), result))
    return results


def check_standard(world, results):
    """The sequence did what it is for, when the arrays have values to inject into: the two
    injected steps are the skipped ones, a clipped step and an unclipped one were taken, and the
    scale halved twice and grew (once, or twice under the sanitizer)."""
    if not any(world.counts):
        return
    assert [r.skip for *_, r in results] == [injection is not None for _, injection, *_ in results]
    taken = [(kind, r) for kind, _, _, _, r in results if not r.skip]
    assert any(r.unscaled > 1.0 for _, r in taken) and any(r.unscaled < 1.0 for _, r in taken)
    assert world.np_state["skipped"] == 2 and world.np_state["step"] == len(taken)
    assert sum(after > before for _, _, before, after, _ in results) == GROWTHS, "the scale grew"
    assert sum(after < before for _, _, before, after, _ in results) == 2, "and halved twice"


def sanitizer_size(full, small):
    return small if SANITIZER else full


# -- The reference against PyTorch's own float64 path -----------------------------------------------------


def test_the_reference_equals_pytorchs_float64_path():
    # The gate's reference is trusted because it is PyTorch's algorithm in float64: unscale, the
    # clip, torch.optim.AdamW (not fused: on the CPU, float64) over several steps with a skip.
    counts, decays = [3000, 700, 300], [True, False, False]
    mask = np.repeat(decays, counts)
    config = types.SimpleNamespace(lr=6e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1,
                                   max_norm=1.0, growth_interval=2000)  # fmt: skip
    p0 = [a.astype(F64) for a in initial_weights(counts, 3)]
    params = [torch.nn.Parameter(torch.from_numpy(a.copy())) for a in p0]
    groups = [{"params": [params[0]], "weight_decay": 0.1}, {"params": params[1:], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.95), eps=config.eps, foreach=False)
    P, M, V = np.concatenate(p0), np.zeros(sum(counts)), np.zeros(sum(counts))
    steps = [(2.0, None), (0.6, None), (0.6, "inf"), (2.0, None), (0.6, None)]
    t = 0
    for number, (target, injection) in enumerate(steps):
        grads = make_gradients(counts, 3, number, target)
        if injection:
            inject(grads, injection)
        scaled = np.concatenate(grads)
        P, M, V, taken = reference_step(P, M, V, scaled, mask, SCALE, t + 1, config)
        assert taken == (injection is None)
        if taken:
            t += 1
            for p, g in zip(params, grads):
                p.grad = torch.from_numpy(g.astype(F64) / SCALE)
            torch.nn.utils.clip_grad_norm_(params, config.max_norm)
            optimizer.step()
        theirs = {"p": np.concatenate([p.detach().numpy() for p in params]),
                  "m": np.concatenate([optimizer.state[p]["exp_avg"].numpy() for p in params]),
                  "v": np.concatenate([optimizer.state[p]["exp_avg_sq"].numpy() for p in params])}  # fmt: skip
        for name, mine in (("p", P), ("m", M), ("v", V)):
            relative = np.linalg.norm(mine - theirs[name]) / np.linalg.norm(theirs[name])
            assert relative <= 1e-12, f"step {number}: {name} is {relative:.3g} (relative) from PyTorch's"
    assert t == 4 and float(np.abs(M).max()) > 0


# -- The bit-exact comparison with the transcription -----------------------------------------------------


def assert_same_arrays(a, b, what, state=True):
    """The two worlds hold the same bits in every array of every entry (and the same state)."""
    for k, label in ((0, "param"), (2, "exp_avg"), (3, "exp_avg_sq")):
        x, y = a.device_values(k), b.device_values(k)
        assert np.array_equal(x.view(np.int32), y.view(np.int32)), f"{what}: {label} differs"
    assert not state or a.read_state() == b.read_state(), what


COUNT_CASES = [(n, True) for n in (0, 1, 3, 4, 5, 1023, 1024, 2047, 2048, 2049)] + [
    (n, False) for n in (1, 1023, 2048, 2049)
]


@pytest.mark.parametrize("count, decay", COUNT_CASES, ids=lambda v: str(v))
def test_params_moments_and_state_equal_the_transcription_bit_for_bit(flint, count, decay):
    # The standard sequence: a clipped step, an unclipped one, an inf (a skip), a NaN (a skip),
    # then clean steps, with a growth interval of 3 so the scale grows once.
    world = World(flint, [count], [decay], seed=count)

    results = run_standard(world, seed=count)

    check_standard(world, results)


@pytest.mark.skipif(SANITIZER, reason="2 million values; the 1,100 tiny tensors below cover the wrap")
def test_more_chunks_than_blocks_make_block_0_wrap(flint):
    # 1,024 x 2,048 + 1 values: 1,025 chunks, so block 0 takes the first and the last.
    count = 1024 * 2048 + 1
    world = World(flint, [count], [True], seed=7)

    results = run_standard(world, seed=7)

    check_standard(world, results)


def test_a_table_of_1100_tiny_tensors_wraps_the_blocks_too(flint):
    # Each tensor is one chunk: 1,100 chunks over 1,024 blocks, with several tensors to a block.
    counts = [1 + (i * 7) % 5 for i in range(1100)]
    world = World(flint, counts, [i % 3 != 0 for i in range(1100)], seed=8)

    results = run_standard(world, seed=8)

    check_standard(world, results)


def test_many_tensors_of_many_sizes_in_one_table(flint):
    counts = torch.randint(1, 3001, (sanitizer_size(600, 80),), generator=torch.Generator().manual_seed(9)).tolist()
    world = World(flint, counts, [i % 2 == 0 for i in range(len(counts))], seed=9)

    results = run_standard(world, seed=9)

    check_standard(world, results)


def test_tensors_of_every_tail_and_empty_entries_in_one_table(flint):
    counts = [17, 4096, 1, 5000, 0, 3, 8193, 4095, 4097, 12, 900, 0, 2049, 0]
    world = World(flint, counts, [i % 2 == 1 for i in range(len(counts))], seed=10)

    results = run_standard(world, seed=10)

    check_standard(world, results)


def test_wide_gradients_with_zeros_equal_the_transcription(flint):
    # GPT-2-like gradients (exponents from 2^-27 to 2^-7 before the norm's power of two, a
    # persistent part and noise, 1 in 1,000 with none).
    world = World(flint, sanitizer_size([50000, 3000, 77], [20000, 3000, 77]), [True, False, True], seed=11)

    results = run_standard(world, seed=11, wide=True)

    check_standard(world, results)


MISALIGNED_COUNTS = [2049, 777, 1, 4096 + 3]
OFFSET_CASES = [(array, offset) for array in range(4) for offset in (1, 2, 3)]
TOGETHER_CASES = [(1, 1, 1, 1), (3, 3, 3, 3), (1, 2, 3, 1), (2, 3, 1, 2)]
if SANITIZER:  # every array and every offset once
    OFFSET_CASES = [(array, 1 + array % 3) for array in range(4)]
    TOGETHER_CASES = [(1, 2, 3, 1)]
ARRAY_NAMES = ("param", "grad", "exp_avg", "exp_avg_sq")


@pytest.mark.parametrize("array, offset", OFFSET_CASES, ids=lambda v: str(v))
def test_one_array_starting_past_a_16_byte_boundary_gives_an_aligned_copys_bits(flint, array, offset):
    # With any of the four arrays 1, 2 or 3 floats off, every entry takes the scalar path; the bits
    # are the transcription's, and so an aligned copy's (compared too).
    offsets = tuple(offset if k == array else 0 for k in range(4))
    decays = [True, False, True, False]
    misaligned = World(flint, MISALIGNED_COUNTS, decays, seed=12, offsets=offsets)
    aligned = World(flint, MISALIGNED_COUNTS, decays, seed=12)
    for k in range(4):
        pointers = [view.data_ptr() % 16 for view in misaligned.dviews[k] if view.numel()]
        assert set(pointers) == {4 * offsets[k]}, (ARRAY_NAMES[k], pointers)
    assert all(view.data_ptr() % 16 == 0 for k in range(4) for view in aligned.dviews[k] if view.numel())

    results = run_standard(misaligned, seed=12)
    run_standard(aligned, seed=12)

    check_standard(misaligned, results)
    assert_same_arrays(misaligned, aligned, f"{ARRAY_NAMES[array]} {offset} floats off")


@pytest.mark.parametrize("offsets", TOGETHER_CASES, ids=str)
def test_all_four_arrays_misaligned_together_give_an_aligned_copys_bits(flint, offsets):
    decays = [True, True, False, True]
    misaligned = World(flint, MISALIGNED_COUNTS, decays, seed=13, offsets=offsets)
    aligned = World(flint, MISALIGNED_COUNTS, decays, seed=13)

    results = run_standard(misaligned, seed=13)
    run_standard(aligned, seed=13)

    check_standard(misaligned, results)
    assert_same_arrays(misaligned, aligned, f"offsets {offsets}")


def test_misaligned_entries_among_aligned_ones(flint):
    # Per entry: aligned, one array off, all off, aligned, another array off.
    plans = [(0, 0, 0, 0), (0, 0, 1, 0), (3, 2, 1, 3), (0, 0, 0, 0), (0, 1, 0, 0), (2, 0, 0, 0)]
    counts = [3000, 2049, 5000, 2048, 100, 4097]
    decays = [True, True, False, True, False, True]
    mixed = World(flint, counts, decays, seed=14, offsets=lambda i: plans[i])
    aligned = World(flint, counts, decays, seed=14)

    results = run_standard(mixed, seed=14)
    run_standard(aligned, seed=14)

    check_standard(mixed, results)
    assert_same_arrays(mixed, aligned, "mixed alignments")


def gpt2_decays():
    pattern = [False, False, True, False, True, False, False, False, True, False, True, False]
    return [True, True] + pattern * 12 + [False, False]


@pytest.mark.skipif(SANITIZER, reason="GPT-2 124M's 124 million values are too many for the sanitizer")
def test_gpt2_124ms_148_tensors_equal_the_transcription_for_two_steps(flint):
    # The transcription runs a chunk of a tensor at a time (update is elementwise) from regenerated
    # inputs, since GPT-2's four FP32 arrays and their FP64 temporaries do not fit the laptop's
    # host memory or its 6 GiB together: the device holds 2 GiB (params, gradients and moments),
    # and the host never holds more than a few chunks of 4M values.
    sizes, decays = gpt2_sizes(), gpt2_decays()
    assert len(decays) == len(sizes) == 148
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    total = int(offsets[-1])
    seed = 15
    config = default_config(lr=6e-4, growth_interval=2000)
    ns = namespace(config)
    p = torch.empty(total, dtype=torch.float32, device="cuda")
    g = torch.empty(total, dtype=torch.float32, device="cuda")
    m = torch.zeros(total, dtype=torch.float32, device="cuda")
    v = torch.zeros(total, dtype=torch.float32, device="cuda")
    for i, n in enumerate(sizes):
        for c, lo, hi in chunks_of(n):
            p[offsets[i] + lo : offsets[i] + hi].copy_(torch.from_numpy(weights_chunk(i, n, seed, c)))
    entries = [(p[offsets[i] : offsets[i + 1]], g[offsets[i] : offsets[i + 1]], m[offsets[i] : offsets[i + 1]],
                v[offsets[i] : offsets[i + 1]], decays[i]) for i in range(148)]  # fmt: skip
    table = flint_lib.adamw_table(entries)
    grad_table = flint_lib.tensor_table([e[1] for e in entries])
    state = flint_lib.adamw_state(SCALE)
    norm = torch.zeros(1, dtype=torch.float64, device="cuda")
    nonfinite = torch.zeros(1, dtype=torch.int64, device="cuda")
    norm_workspace = torch.zeros(flint.gradient_norm_workspace(), dtype=torch.uint8, device="cuda")
    workspace = torch.zeros(flint.adamw_workspace(), dtype=torch.uint8, device="cuda")
    np_state, scalars_by_step, factors = initial_state(), [], []

    def gradient(i, step, c):
        return np.ldexp(gradient_chunk(i, sizes[i], seed, step, c), factors[step - 1]).astype(F32)

    for step, target in ((1, 2.0), (2, 0.6)):
        # The norm's power of two, from a first pass over the chunks.
        squares = sum(float(np.sum(gradient_chunk(i, n, seed, step, c).astype(F64) ** 2))
                      for i, n in enumerate(sizes) for c, _, _ in chunks_of(n))  # fmt: skip
        factors.append(power_of_two_for(squares, target))
        for i, n in enumerate(sizes):
            for c, lo, hi in chunks_of(n):
                g[offsets[i] + lo : offsets[i] + hi].copy_(torch.from_numpy(gradient(i, step, c)))
        flint.gradient_norm(norm, nonfinite, grad_table, 148, norm_workspace)
        flint.adamw(table, 148, norm, nonfinite, state, config, workspace)
        torch.cuda.synchronize()
        scalars, np_state = prepare(norm.item(), nonfinite.item(), np_state, ns)
        assert not scalars["skip"] and nonfinite.item() == 0
        scalars_by_step.append(scalars)
        assert states_equal(flint_lib.read_adamw_state(state), np_state)
        # Every chunk after this many steps, from regenerated inputs.
        for i, n in enumerate(sizes):
            for c, lo, hi in chunks_of(n):
                a, b, d = weights_chunk(i, n, seed, c), np.zeros(hi - lo, F32), np.zeros(hi - lo, F32)
                for done in range(step):
                    a, b, d = update(a, gradient(i, done + 1, c), b, d, decays[i], scalars_by_step[done], ns)
                base = offsets[i] + lo
                for label, device, host in (("param", p, a), ("exp_avg", m, b), ("exp_avg_sq", v, d)):
                    bad = mismatch(device[base : base + hi - lo], host)
                    assert bad is None, (f"step {step}, tensor {i} ({n} values, decay {decays[i]}), chunk {c}: "
                                         f"{label} differs at {bad}")  # fmt: skip
                del a, b, d
    assert float(np_state["scale"]) == SCALE and np_state["step"] == 2


# -- The same bits as PyTorch's fused kernel where they must agree ----------------------------------------


def test_the_arithmetic_matches_pytorchs_fused_kernel_where_it_must(flint):
    # On an unclipped step both see bit-identical unscaled gradients (an exact power-of-two unscale,
    # a clip of 1), and the transcription is the fused kernel's arithmetic. Measured on the laptop
    # (data/d32/peer_bits.log): the first step from zero moments gave identical bits for all 1M
    # values; on later steps, each started from the peer's own state, about 0.4% of m and 0.0002%
    # of v differ by 1 ulp (PyTorch's build contracts b1 m + (1 - b1) g into one FMA) and about
    # 0.015% of p differ, by tiny amounts. Equal everywhere else, so decoupled decay, the bias
    # corrections, eps's place and the step size's FP32 rounding are PyTorch's.
    n = sanitizer_size(1 << 20, 1 << 16)
    later = sanitizer_size(5, 3)
    p0 = initial_weights([n], 21)
    config = flint_lib.adamw_config(6e-4, growth_interval=2000)
    world = World(flint, [n], [True], params=p0, config=config)
    peer = Peer(p0, [True], 6e-4)
    before = None
    for t in range(1, later + 2):
        grads = make_gradients([n], 21, t, 0.3, wide=True)
        if before is not None:
            world.set_arrays(*before)
        result = world.step(grads)
        peer.step([torch.from_numpy(grads[0]).cuda()])
        assert not result.skip and result.unscaled < 1.0
        before = tuple(peer.arrays(which) for which in "pmv")
        report = []
        for which, k in (("p", 0), ("m", 2), ("v", 3)):
            ours, theirs = world.device_values(k), peer.values(which)
            equal = float(np.mean(ours.view(np.int32) == theirs.view(np.int32)))
            ulps = ulps_between(ours, theirs)
            report.append(f"{which}: equal {equal * 100:.4f}%, max {int(ulps.max())} ulps")
            if t == 1:
                assert equal >= 0.99999 and ulps.max() <= 1, (t, which, equal, ulps.max())
            elif which == "p":
                assert equal >= 0.999, (t, which, equal)
                largest = float(np.abs(theirs).max())
                assert float(np.abs(ours.astype(F64) - theirs).max()) <= 4 * gate.fp32_ulp(largest), (t, which)
            else:
                assert ulps.max() <= 1, (t, which, ulps.max())
                assert equal >= (0.99 if which == "m" else 0.9999), (t, which, equal)
        print(f"PEER BITS step {t}: " + "; ".join(report))


# -- The gate: D26's rule against the FP32 peer ---------------------------------------------------------

GATE_COUNTS = sanitizer_size([196608, 32768, 32768], [20480, 2048, 2048])
GATE_DECAYS = [True, False, False]


def gate_weights(seed):
    """Weights about N(0, 0.02) with decay, LayerNorm-like ones near 1 and biases near 0 without."""
    g = torch.Generator().manual_seed(500 + seed)
    n_w, n_ln, n_b = GATE_COUNTS
    w = (sum(draw(n_w, -256, 256, g) for _ in range(4)) * 2.0**-14).numpy()
    ln = (1.0 + draw(n_ln, -256, 256, g) * 2.0**-12).numpy()
    b = (draw(n_b, -256, 256, g) * 2.0**-15).numpy()
    return [w, ln, b]


def run_gate(flint, lr, steps, checkpoints, *, max_norm=1.0, targets=(2.0, 0.6), variant=False):
    """Flint, the FP32 peer and the FP32 variant of the update (if variant) over `steps` steps on
    3 seeds, each step's gradients alternating the targets' norms; the gate on params and both
    moments at each checkpoint step."""
    config = flint_lib.adamw_config(lr, max_norm=max_norm, growth_interval=2000)
    ns = namespace(config)
    mask = np.repeat(GATE_DECAYS, GATE_COUNTS)
    names = ("p", "m", "v")
    errors = {kind: {t: {o: [] for o in names} for t in checkpoints} for kind in ("flint", "peer", "f32")}
    for seed in gate.SEEDS:
        p0 = gate_weights(seed)
        world = World(flint, GATE_COUNTS, GATE_DECAYS, params=p0, config=config)
        peer = Peer(p0, GATE_DECAYS, lr, max_norm=max_norm, growth_interval=2000)
        ref = (np.concatenate(p0).astype(F64), np.zeros(sum(GATE_COUNTS)), np.zeros(sum(GATE_COUNTS)))
        low = (np.concatenate(p0), np.zeros(sum(GATE_COUNTS), F32), np.zeros(sum(GATE_COUNTS), F32))
        for t in range(1, steps + 1):
            grads = make_gradients(GATE_COUNTS, seed, t, targets[0] if t % 2 else targets[1], wide=True)
            result = world.step(grads)
            assert not result.skip
            peer.step([torch.from_numpy(g).cuda() for g in grads])
            flat = np.concatenate(grads)
            *ref, taken = reference_step(*ref, flat, mask, SCALE, t, ns)
            if variant:
                low = step_f32(*low, flat * result.multiplier, mask, ns, t)
            if t in checkpoints:
                for o, k, r, f in zip(names, (0, 2, 3), ref, low):
                    for kind, out in (("flint", world.device_values(k)), ("peer", peer.values(o)), ("f32", f)):
                        if kind == "f32" and not variant:
                            continue
                        errors[kind][t][o].append(gate.errors(torch.from_numpy(out), torch.from_numpy(r)))
        del world, peer
    for t in checkpoints:
        for o in names:
            try:
                bounds = gate.check_gate(gate.FORWARD_FACTOR, errors["flint"][t][o], errors["peer"][t][o], output="fp32")
            except AssertionError as error:
                raise AssertionError(f"lr {lr:g}, max_norm {max_norm:g}, step {t}, {o}:\n{error}") from None

            def worst(kind):
                return max(e.rel_l2 for e in errors[kind][t][o]), max(e.max_abs for e in errors[kind][t][o])

            peer_rel, peer_abs = worst("peer")
            rel, ab = worst("flint")
            print(f"GATE lr {lr:g} max_norm {max_norm:g} step {t:3d} {o}: flint rel {rel:.3g} abs {ab:.3g} | "
                  f"peer rel {peer_rel:.3g} abs {peer_abs:.3g} | bounds rel {bounds.rel_l2:.3g} abs {bounds.max_abs:.3g} | "
                  f"margin rel {rel / bounds.rel_l2:.2f} abs {ab / bounds.max_abs:.2f}")  # fmt: skip
            if variant:
                rel, ab = worst("f32")
                try:
                    gate.check_gate(gate.FORWARD_FACTOR, errors["f32"][t][o], errors["peer"][t][o], output="fp32")
                    verdict = "would pass"
                except AssertionError:
                    verdict = "would FAIL"
                print(f"GATE all-FP32 variant lr {lr:g} step {t:3d} {o}: rel {rel:.3g} abs {ab:.3g} | "
                      f"margin rel {rel / bounds.rel_l2:.2f} abs {ab / bounds.max_abs:.2f} "
                      f"({verdict}; not asserted)")  # fmt: skip


SHORT = sanitizer_size(20, 6)  # the sanitizer's "20" steps
GATE_CASES = [
    pytest.param(6e-4, SHORT, (1, 2, SHORT), id=f"lr-6e-4-{SHORT}-steps"),
    pytest.param(6e-6, SHORT, (1, 2, SHORT), id=f"lr-6e-6-{SHORT}-steps"),
    pytest.param(6e-4, 100, (1, 2, 20, 100), id="lr-6e-4-100-steps",
                 marks=pytest.mark.skipif(SANITIZER, reason="100 steps are too long for the sanitizer")),  # fmt: skip
]


@pytest.mark.parametrize("lr, steps, checkpoints", GATE_CASES)
def test_params_and_moments_pass_the_gate_against_fp32_pytorch(flint, lr, steps, checkpoints):
    # Steps alternate clipped (an unscaled norm of about 2) and unclipped (about 0.6). The all-FP32
    # variant's margin on the 100-step case is printed, not asserted: it is the evidence for the mix.
    run_gate(flint, lr, steps, checkpoints, variant=steps == 100)


# -- The state, exactly as GradScaler's --------------------------------------------------------------------


def skip_steps(pattern, steps):
    """The steps (from 0) that get an inf or a NaN, for a pattern."""
    if pattern == "none":
        return set()
    if pattern == "random":
        rng = random.Random(steps)
        return {t for t in range(steps) if rng.random() < 0.35}
    if pattern == "first":  # the first step, and the next, and one more
        return {0, 1, steps // 2}
    if pattern == "row":  # a run of skips in the middle
        return set(range(steps // 3, steps // 3 + 4))
    raise ValueError(pattern)


def follow_the_scaler(flint, interval, skips, steps, *, init_scale=SCALE, n=8, seed=31):
    """Flint's state beside GradScaler's (and the fused AdamW's step count) over `steps` steps of
    one tensor, with an inf or a NaN written into the gradients of the steps in `skips`: scale and
    tracker (bit for bit) and AdamW's step count are the peer's after every step. Returns the
    scales before and after each step, and the skipped steps' count."""
    p0 = initial_weights([n], seed)
    config = flint_lib.adamw_config(1e-3, growth_interval=interval)
    world = World(flint, [n], [True], params=p0, config=config, state=dict(scale=init_scale))
    peer = Peer(p0, [True], 1e-3, init_scale=init_scale, growth_interval=interval)
    history = []
    for t in range(steps):
        grads = make_gradients([n], seed, t, 0.5 if t % 2 else 2.0, scale=init_scale)
        if t in skips:
            inject(grads, "inf" if t % 2 else "nan")
        before = world.read_state()["scale"]
        result = world.step(grads)
        peer.step([torch.from_numpy(g).cuda() for g in grads])
        state = world.read_state()
        assert result.skip == (t in skips)
        assert struct.pack("<f", state["scale"]) == struct.pack("<f", peer.scale), (
            f"step {t}: scale {state['scale']!r} against the scaler's {peer.scale!r}")
        assert state["growth_tracker"] == peer.growth_tracker, (
            f"step {t}: tracker {state['growth_tracker']} against the scaler's {peer.growth_tracker}")
        assert state["step"] == peer.step_count, f"step {t}: AdamW's step count {state['step']} against {peer.step_count}"
        history.append((before, state["scale"]))
    assert state["skipped"] == len(skips)
    return history, state


STATE_STEPS = sanitizer_size(30, 12)


@pytest.mark.parametrize("pattern", ["none", "random", "first", "row"])
@pytest.mark.parametrize("interval", [1, 3])
def test_the_state_equals_gradscalers_after_every_step(flint, interval, pattern):
    skips = skip_steps(pattern, STATE_STEPS)
    history, state = follow_the_scaler(flint, interval, skips, STATE_STEPS)
    grew = sum(after > before for before, after in history)
    halved = sum(after < before for before, after in history)
    assert halved == len(skips)
    assert grew >= 1 if interval == 1 or pattern == "none" else grew >= 0
    if pattern == "none":
        assert grew == STATE_STEPS // interval and state["step"] == STATE_STEPS
    print(f"STATE interval {interval} {pattern}: {len(skips)} skips, grew {grew} times, scale {state['scale']}")


@pytest.mark.skipif(SANITIZER, reason="4,001 steps are too long for the sanitizer")
def test_the_state_equals_gradscalers_over_4001_steps_at_the_default_interval(flint):
    # Clean runs long enough to grow at 2,000, one skip to reset the tracker before the first
    # growth would have come late, and a pair of skips that reset it again.
    skips = {3, 2990, 2991}
    history, state = follow_the_scaler(flint, 2000, skips, 4001, n=8)
    grown = [(i, after / before) for i, (before, after) in enumerate(history) if after > before]
    assert [i for i, _ in grown] == [2003], grown  # the 2,000th clean step since the skip at step 3
    assert state["skipped"] == 3 and state["step"] == 4001 - 3


def test_a_scale_of_2_to_the_127_does_not_grow_because_2_to_the_128_is_inf(flint):
    # Interval 1, so every clean step would double the scale: it stays, the tracker resets, and
    # the step is taken. A skip halves it to 2^126, and the scale then grows back to 2^127.
    top = 2.0**127
    history, state = follow_the_scaler(flint, 1, {2}, 6, init_scale=top, seed=32)
    assert [round(math.log2(after)) for _, after in history] == [127, 127, 126, 127, 127, 127]
    assert state["growth_tracker"] == 0 and state["step"] == 5 and state["skipped"] == 1


def test_a_tracker_past_a_shortened_interval_grows_on_the_next_clean_step(flint):
    # A resumed run may shorten growth_interval: the test is tracker + 1 >= interval, where
    # PyTorch's is ==, so a tracker of 10 with an interval of 3 grows at once (PyTorch's would
    # wait for the tracker to wrap), and starts over.
    world = World(flint, [100], seed=33, state=dict(growth_tracker=10, step=4))

    result = world.step(make_gradients([100], 33, 0, 0.5))

    state = world.read_state()
    assert not result.skip
    assert state["scale"] == 2.0 * SCALE and state["growth_tracker"] == 0 and state["step"] == 5


def test_a_skip_changes_no_bit_of_the_params_or_moments_and_counts_in_skipped(flint):
    world = World(flint, [3000, 5, 2049], [True, False, True], seed=34)
    for number in range(2):
        world.step(make_gradients(world.counts, 34, number, 0.6))
    before = [world.device_values(k).copy() for k in (0, 2, 3)]
    assert all(np.any(b != 0) for b in before)
    state_before = world.read_state()
    grads = make_gradients(world.counts, 34, 2, 0.6)
    inject(grads, "-inf")

    result = world.step(grads)

    assert result.skip and result.nonfinite == 1
    for b, k in zip(before, (0, 2, 3)):
        assert np.array_equal(world.device_values(k).view(np.int32), b.view(np.int32))
    state = world.read_state()
    assert state == {"scale": state_before["scale"] / 2, "growth_tracker": 0, "step": state_before["step"],
                     "skipped": state_before["skipped"] + 1}  # fmt: skip


# -- The skip rule's edges -----------------------------------------------------------------------------------

INF, NAN = math.inf, math.nan
SKIP_CASES = {
    "a-count-above-0-with-a-finite-norm": dict(norm=100.0, count=1, scale=SCALE, skip=True),
    "a-norm-of-inf-with-a-count-of-0": dict(norm=INF, count=0, scale=SCALE, skip=True),
    "a-norm-of-nan-with-a-count-of-0": dict(norm=NAN, count=0, scale=SCALE, skip=True),
    "a-norm-of-negative-nan-with-a-count-of-0": dict(norm=-NAN, count=0, scale=SCALE, skip=True),
    "a-scale-of-0-with-a-finite-norm": dict(norm=100.0, count=0, scale=0.0, skip=True),
    "a-scale-of-0-with-a-norm-of-0": dict(norm=0.0, count=0, scale=0.0, skip=True),
    "a-scale-of-2^-130-so-1-over-it-overflows": dict(norm=2.0**-140, count=0, scale=2.0**-130, skip=True),
    "the-smallest-scale-halves-to-0": dict(norm=2.0**-160, count=0, scale=2.0**-149, skip=True),
    "an-unscaled-norm-that-overflows": dict(norm=1e308, count=0, scale=2.0**-10, skip=True),
    "a-clean-step-as-a-control": dict(norm=1000.0, count=0, scale=SCALE, skip=False),
    "a-small-but-fine-scale-of-2^-100": dict(norm=2.0**-110, count=0, scale=2.0**-100, skip=False),
    "a-scale-of-2^-126": dict(norm=2.0**-127, count=0, scale=2.0**-126, skip=False),
}


@pytest.mark.parametrize("case", SKIP_CASES.values(), ids=list(SKIP_CASES))
def test_the_skip_rule_at_its_edges(flint, case):
    # The norm and the count are written by hand into the buffers flint_adamw reads.
    world = World(flint, [1000, 5], seed=40, state=dict(scale=case["scale"]))
    before = [world.device_values(k).copy() for k in (0, 2, 3)]
    # Gradients consistent with the scale where the step is taken (the multiplier would take
    # others to inf); the norm is crafted whatever they are.
    grads = make_gradients(world.counts, 40, 0, 0.5, scale=case["scale"] if case["scale"] >= 2.0**-126 else SCALE)

    result = world.step(grads, norm=case["norm"], nonfinite=case["count"])

    state = world.read_state()
    assert result.skip == case["skip"]
    if case["skip"]:
        assert state["skipped"] == 1 and state["step"] == 0 and state["growth_tracker"] == 0
        assert state["scale"] == float(F32(case["scale"]) * F32(0.5))
        for b, k in zip(before, (0, 2, 3)):
            assert np.array_equal(world.device_values(k).view(np.int32), b.view(np.int32))
    else:
        assert state["skipped"] == 0 and state["step"] == 1 and state["growth_tracker"] == 1
        assert any(not np.array_equal(world.device_values(k).view(np.int32), b.view(np.int32))
                   for b, k in zip(before, (0, 2, 3)))  # fmt: skip
    if case["scale"] == 0.0:
        assert struct.pack("<f", state["scale"]) == struct.pack("<f", 0.0), "a scale of 0 stays +0"


BAD_SCALES = {"negative": -SCALE, "inf": math.inf, "-inf": -math.inf, "nan": math.nan, "-0": -0.0}


@pytest.mark.parametrize("name", BAD_SCALES)
def test_a_scale_that_is_not_finite_and_positive_skips_the_step_and_halves_it(flint, name):
    # Clean gradients and a clean norm: only the state's scale is wrong. Nothing moves, and the
    # scale is halved (a negative one toward 0, an infinite or NaN one stays as it is), so such a
    # state skips every step. NaN is compared with isnan: the kernel's NaN bits needn't be numpy's.
    scale = BAD_SCALES[name]
    state = dict(scale=scale, growth_tracker=5, step=7, skipped=2)
    world = World(flint, [1000, 5, 2049], seed=44, state=state)
    before = [world.device_values(k).copy() for k in (0, 2, 3)]
    for number in range(2):
        pending = world.begin(make_gradients(world.counts, 44, number, 0.5))
        world.launch()
        torch.cuda.synchronize()
        assert world.nonfinite.item() == 0 and math.isfinite(world.norm.item())
        after = [world.device_values(k) for k in (0, 2, 3)]
        for b, a in zip(before, after):
            assert np.array_equal(a.view(np.int32), b.view(np.int32)), "an array changed"
        got = world.read_state()
        assert got["growth_tracker"] == 0 and got["step"] == 7 and got["skipped"] == 3 + number
        if math.isnan(scale):
            assert math.isnan(got["scale"])
        else:
            expected = float(F32(scale) * F32(0.5) ** (number + 1))
            assert struct.pack("<f", got["scale"]) == struct.pack("<f", expected), (got["scale"], expected)
        assert torch.equal(world.table, pending["table"])
        # The transcription's mirror of the state, for the next call (NaN: not compared).
        world.np_state = {"scale": F32(got["scale"]), "growth_tracker": 0, "step": 7, "skipped": 3 + number}


def scaler_unscale(grads, scale=1.0):
    """GradScaler's unscale_ on copies of the gradients (it unscales them in place), with its own
    inv_scale, the FP32 rounding of 1 / scale (inf for a scale of 2^-140). Returns (found_inf,
    whether every unscaled value is finite). PyTorch 2.6's kernel checks the values it reads
    before multiplying, so an unscaled value that overflows is inf in the result without found_inf
    set (probed in data/d32/probe_unscale.py): PyTorch would go on with an inf gradient."""
    copies = [torch.from_numpy(g.copy()).cuda() for g in grads if g.size]
    found = torch.zeros(1, device="cuda")
    with np.errstate(all="ignore"):
        inverse = F32(F64(1.0) / F64(scale))
    torch._amp_foreach_non_finite_check_and_unscale_(
        copies, found, torch.tensor([float(inverse)], dtype=torch.float32, device="cuda"))
    return bool(found.item() > 0), all(bool(torch.isfinite(c).all()) for c in copies)


def found_inf_of(grads, scale=1.0):
    """GradScaler's found_inf on the gradients."""
    return scaler_unscale(grads, scale)[0]


@pytest.mark.parametrize("kind", [None, *NON_FINITE_BITS])
def test_the_step_is_skipped_exactly_when_the_scaler_finds_an_inf(flint, kind):
    world = World(flint, [3000, 5, 1, 4097], seed=41)
    grads = make_gradients(world.counts, 41, 0, 0.5)
    if kind:
        inject(grads, kind)

    result = world.step(grads)

    assert result.skip == found_inf_of(grads) == (kind is not None)
    assert result.nonfinite == (1 if kind else 0)


def test_an_unscaled_value_that_overflows_skips_the_step(flint):
    # The review's finding 1: with max_norm = +inf (no clip) and a scale of 0.5, a scaled
    # gradient of 2^127 unscales to 2^128, past FP32's maximum: g would be inf, the weights NaN.
    # GradScaler's unscale_ (inv_scale 2) multiplies it to inf too, though PyTorch 2.6 doesn't
    # flag that as found_inf (it checks its inputs); Flint skips.
    world = World(flint, [1000, 5], seed=45, state=dict(scale=0.5),
                  config=default_config(max_norm=math.inf))  # fmt: skip
    before = [world.device_values(k).copy() for k in (0, 2, 3)]
    grads = make_gradients(world.counts, 45, 0, 0.5, scale=0.5)
    grads[0][3] = 2.0**127
    assert all(np.isfinite(g).all() for g in grads)

    result = world.step(grads)

    assert result.skip and result.nonfinite == 0
    assert scaler_unscale(grads, 0.5) == (False, False), "PyTorch's unscaled gradient has an inf"
    for b, k in zip(before, (0, 2, 3)):
        assert np.array_equal(world.device_values(k).view(np.int32), b.view(np.int32)), "an array changed"
    assert world.read_state() == {"scale": 0.25, "growth_tracker": 0, "step": 0, "skipped": 1}


def test_a_scale_whose_inverse_overflows_skips_even_when_the_multiplier_does_not(flint):
    # The review's finding 2: a scale of 2^-140 and a scaled norm of 2^-100 (max_norm 1) clip to
    # a multiplier of 2^-40 / 2^-140 = 2^100, finite; but 1 / scale is inf in FP32, so
    # GradScaler's inv_scale makes every unscaled value inf (found_inf isn't set, as above);
    # Flint skips.
    scale = 2.0**-140
    world = World(flint, [1000, 5], seed=46, state=dict(scale=scale))
    before = [world.device_values(k).copy() for k in (0, 2, 3)]
    grads = make_gradients(world.counts, 46, 0, 0.5, scale=1.0)

    result = world.step(grads, norm=2.0**-100, nonfinite=0)

    assert result.skip and scaler_unscale(grads, scale) == (False, False)
    for b, k in zip(before, (0, 2, 3)):
        assert np.array_equal(world.device_values(k).view(np.int32), b.view(np.int32)), "an array changed"
    state = world.read_state()
    assert state["skipped"] == 1 and state["scale"] == float(F32(scale) * F32(0.5))


BOUND = F32(2.0**126)  # at a scale of 0.5, a gradient of this unscales to 2^127
BOUND_CASES = {
    "just-below-2^127": (np.nextafter(BOUND, F32(0)), False),
    "at-2^127": (BOUND, True),
    "just-above-2^127": (np.nextafter(BOUND, F32(np.inf)), True),
}


@pytest.mark.parametrize("value, skip", BOUND_CASES.values(), ids=list(BOUND_CASES))
def test_the_unscaled_norm_bound_of_2_to_the_127(flint, value, skip):
    # One scaled gradient of `value` among small ones, a scale of 0.5 and no clipping: the unscaled
    # norm is 2 * value. Below 2^127 the step is taken and every g * multiplier is finite (the
    # second moments overflow to inf in FP32, as g^2 is about 2^254, but nothing is NaN), the bits
    # are the transcription's, and GradScaler's unscaled values are finite; from 2^127 on Flint
    # skips though they still are (the only extra skips: PyTorch's own FP32 clip norm is inf
    # there).
    world = World(flint, [1000, 5], seed=47, state=dict(scale=0.5),
                  config=default_config(max_norm=math.inf))  # fmt: skip
    before = [world.device_values(k).copy() for k in (0, 2, 3)]
    grads = make_gradients(world.counts, 47, 0, 0.5, scale=0.5)
    grads[0][:] = np.clip(grads[0], -1000, 1000)
    grads[0][3] = value

    result = world.step(grads)

    assert result.skip == skip and scaler_unscale(grads, 0.5) == (False, True)
    after = [world.device_values(k) for k in (0, 2, 3)]
    if skip:
        for b, a in zip(before, after):
            assert np.array_equal(a.view(np.int32), b.view(np.int32)), "an array changed"
    else:
        assert result.unscaled < 2.0**127 and np.isfinite(grads[0]).all()
        assert np.isfinite(after[0]).all() and np.isfinite(after[1]).all() and not np.isnan(after[2]).any()
        assert world.read_state()["step"] == 1


def test_finite_gradients_the_scaler_accepts_are_not_skipped_either(flint):
    # Values near FP32's maximum are finite: neither skips.
    world = World(flint, [200], seed=42)
    grads = make_gradients(world.counts, 42, 0, 0.5)
    grads[0][:4] = [3.4028235e38, -3.4028235e38, 3e38, -3e38]

    result = world.step(grads)

    assert not result.skip and not found_inf_of(grads)


# -- Clipping ------------------------------------------------------------------------------------------------

CLIP_NORMS = {"below": 0.5, "exactly-at": 1.0, "above": 2.0}


@pytest.mark.parametrize("name", CLIP_NORMS)
def test_the_unscaled_norm_below_at_and_above_max_norm(flint, name):
    # The norm is written by hand: unscaled, norm / 65,536. At max_norm itself the coefficient is
    # 1 / (1 + 1e-6), so the step clips a little, as PyTorch's does.
    world = World(flint, [3000, 77, 5000], [True, False, True], seed=50)
    unscaled = CLIP_NORMS[name]

    result = world.step(make_gradients(world.counts, 50, 0, 0.6), norm=unscaled * SCALE, nonfinite=0)

    unclipped = F32(1.0 / SCALE)
    if name == "below":
        assert result.multiplier == unclipped
    else:
        assert result.multiplier < unclipped
    if name == "exactly-at":
        assert result.multiplier == F32((1.0 / (1.0 + 1e-6)) / SCALE)
    if name == "above":
        assert result.multiplier == F32((1.0 / (2.0 + 1e-6)) / SCALE)


def test_max_norm_inf_multiplies_by_exactly_one_over_the_scale(flint):
    # Gradients whose unscaled norm of about 2 would be clipped at max_norm 1: with inf they are not.
    world = World(flint, [3000, 77], seed=51, config=default_config(max_norm=math.inf))

    results = [world.step(make_gradients(world.counts, 51, number, 2.0)) for number in range(2)]

    assert all(r.unscaled > 1.0 and r.multiplier == F32(1.0 / SCALE) for r in results)


@pytest.mark.parametrize("max_norm", [1e-3, 1e-30, 1e-300])
def test_a_tiny_max_norm_shrinks_the_gradients_to_nothing(flint, max_norm):
    world = World(flint, [3000, 77], [True, False], seed=52, config=default_config(max_norm=max_norm))

    results = [world.step(make_gradients(world.counts, 52, number, 0.6)) for number in range(3)]

    assert all(not r.skip and 0 <= r.multiplier * SCALE < 0.01 for r in results)
    if max_norm == 1e-300:
        assert all(r.multiplier == 0 for r in results), "a multiplier that underflows is 0, not a skip"


def test_the_clip_uses_the_unscaled_norm_so_two_scales_give_the_same_params(flint):
    # The same unscaled gradients at a scale of 2^16 and of 2^8: every step's multiplier, params
    # and moments are the same bits.
    counts, decays = [4000, 33, 2049], [True, False, True]
    high = World(flint, counts, decays, seed=53, state=dict(scale=2.0**16))
    low = World(flint, counts, decays, seed=53, state=dict(scale=2.0**8))
    for number in range(4):
        unscaled = make_gradients(counts, 53, number, 2.0 if number % 2 == 0 else 0.6, scale=1.0)
        a = high.step([np.ldexp(g, 16) for g in unscaled])
        b = low.step([np.ldexp(g, 8) for g in unscaled])
        assert a.unscaled == b.unscaled and a.multiplier * F32(2.0**16) == b.multiplier * F32(2.0**8)
    assert_same_arrays(high, low, "scales 2^16 and 2^8", state=False)
    assert high.np_state["step"] == low.np_state["step"] == 4


CLIP_GATES = {
    "every-step-clipped-at-1": dict(max_norm=1.0, targets=(2.0, 2.0)),
    "every-step-clipped-hard-at-0.01": dict(max_norm=0.01, targets=(2.0, 2.0)),
    "no-clipping-at-inf": dict(max_norm=math.inf, targets=(2.0, 0.6)),
}


@pytest.mark.parametrize("case", CLIP_GATES.values(), ids=list(CLIP_GATES))
def test_clipped_steps_pass_the_gate_against_fp32_pytorch(flint, case):
    last = sanitizer_size(5, 3)
    run_gate(flint, 6e-4, last, (1, last), **case)


# -- The values' edges ---------------------------------------------------------------------------------------


def run_with(world, steps, seed, target=0.6, edit=None, wide=False):
    """`steps` steps of plain gradients, each passed through edit (a function of the list of
    arrays, in place) first; returns the StepResults."""
    results = []
    for number in range(steps):
        grads = make_gradients(world.counts, seed, number, target, wide=wide)
        if edit:
            edit(grads)
        results.append(world.step(grads))
    return results


def bits_of(array):
    return np.ascontiguousarray(array).view(np.int32)


def test_zero_gradients_on_zero_params_stay_exactly_positive_zero_even_with_decay(flint):
    # wte's padded rows: nothing moves them, whatever the decay and the other tensors do.
    counts = [3000, 513, 40]
    p0 = initial_weights(counts, 60)
    p0[1][:] = 0.0
    world = World(flint, counts, [True, True, False], seed=60, params=p0)

    def edit(grads):
        grads[1][:] = 0.0

    results = run_with(world, 4, 60, edit=edit)

    assert not any(r.skip for r in results)
    for k in (0, 2, 3):
        assert not world.dviews[k][1].view(torch.int32).any(), "an array of zeros stayed +0"
    assert any(np.any(world.device_values(k)) for k in (0, 2, 3))


def test_lr_0_changes_no_param_except_that_decay_turns_minus_zero_into_plus_zero(flint):
    counts = [1000, 1000]
    p0 = initial_weights(counts, 61)
    for a in p0:
        a[0], a[1] = -0.0, 0.0
    world = World(flint, counts, [True, False], seed=61, params=p0, config=default_config(lr=0.0))

    def edit(grads):  # a positive gradient at the -0 of the undecayed tensor keeps its -0 -0
        grads[1][0] = abs(grads[1][0])

    run_with(world, 4, 61, edit=edit)

    decayed, plain = world.device_values(0)[:1000], world.device_values(0)[1000:]
    expected = p0[0] + F32(0.0)  # -0 + +0 is +0, every other value is itself
    assert np.array_equal(bits_of(decayed), bits_of(expected)) and bits_of(decayed)[0] == 0
    assert np.array_equal(bits_of(plain), bits_of(p0[1])) and bits_of(plain)[0] == bits_of(np.array([-0.0], F32))[0]
    assert np.any(world.device_values(2) != 0), "the moments still moved"


def test_weight_decay_0_leaves_a_decayed_entrys_minus_zero_as_minus_zero(flint):
    counts = [1000]
    p0 = initial_weights(counts, 62)
    p0[0][0] = -0.0

    def edit(grads):  # no gradient at that value, so nothing else touches it
        grads[0][0] = 0.0

    off = World(flint, counts, [True], seed=62, params=p0, config=default_config(weight_decay=0.0))
    on = World(flint, counts, [True], seed=62, params=p0)
    run_with(off, 4, 62, edit=edit)
    run_with(on, 4, 62, edit=edit)

    minus_zero = bits_of(np.array([-0.0], F32))[0]
    assert bits_of(off.device_values(0))[0] == minus_zero, "no decay: -0 stays"
    assert bits_of(on.device_values(0))[0] == 0, "with decay: -0 became +0"


def test_subnormal_gradients_give_subnormal_moments_and_nothing_is_flushed(flint):
    # Entry 0: unscaled gradients from 2^-135 to 2^-125, mostly FP32 subnormals (the product of the
    # scaled gradient and the multiplier is one, rounded to subnormal); its first moments are too.
    # Entry 1: gradients near 2^-62, whose squares make the second moments subnormal.
    counts = [3000, 3000]
    world = World(flint, counts, [True, False], seed=63)
    low = 1.17549435e-38
    seen = [0, 0]
    for number in range(4):
        grads = [np.ldexp(tensor_gradient(i, n, 63, number), -119 if i == 0 else -56).astype(F32)
                 for i, n in enumerate(counts)]  # fmt: skip
        result = world.step(grads)
        assert not result.skip and result.unscaled < 1e-10
        m = world.device_values(2)
        v = world.device_values(3)
        seen[0] += int(np.sum((np.abs(m[:3000]) < low) & (m[:3000] != 0)))
        seen[1] += int(np.sum((np.abs(v[3000:]) < low) & (v[3000:] != 0)))
    assert seen[0] > 5000 and seen[1] > 1000, seen
    print(f"SUBNORMALS first moments {seen[0]}, second moments {seen[1]} (over 4 steps)")


def test_gradients_near_fp32s_maximum_are_finite_and_not_skipped(flint):
    # Scaled gradients up to 3.4e38: their squares overflow FP32 by far, the norm in FP64 doesn't,
    # and the multiplier (which is a subnormal here) brings them back into range.
    counts, decays = [2000, 300], [True, False]
    world = World(flint, counts, decays, seed=64)

    def edit(grads):
        for g in grads:
            np.multiply(g, F32(2.0**117), out=g)
        grads[0][:4] = [3.4028235e38, -3.4028235e38, 3e38, -3e38]

    results = []
    for number in range(3):
        grads = [tensor_gradient(i, n, 64, number) for i, n in enumerate(counts)]
        edit(grads)
        results.append(world.step(grads))

    assert not any(r.skip for r in results) and all(r.norm > 1e38 and math.isfinite(r.norm) for r in results)
    assert world.read_state()["skipped"] == 0 and world.read_state()["step"] == 3
    assert all(np.isfinite(world.device_values(k)).all() for k in (0, 2, 3))
    assert all(r.multiplier < 1e-30 for r in results)


@pytest.mark.parametrize("beta1, beta2", [(0.0, 0.95), (0.9, 0.0), (0.0, 0.0)], ids=str)
def test_beta_0_is_allowed(flint, beta1, beta2):
    world = World(flint, [3000, 40], [True, False], seed=65, config=default_config(beta1=beta1, beta2=beta2))

    results = run_with(world, 5, 65)

    assert not any(r.skip for r in results)


@pytest.mark.parametrize("eps", [1e-30, 1.0])
def test_a_tiny_and_a_huge_eps(flint, eps):
    world = World(flint, [3000, 40], [True, False], seed=66, config=default_config(eps=eps))

    results = run_with(world, 5, 66)

    assert not any(r.skip for r in results)
    assert all(np.isfinite(world.device_values(k)).all() for k in (0, 2, 3))


# -- What the call reads and writes -------------------------------------------------------------------------


def test_the_workspace_is_16_bytes(flint):
    assert flint.adamw_workspace() == 16


@pytest.mark.parametrize("state_offset, workspace_offset", [(16, 16), (48, 80), (112, 32)], ids=str)
def test_nothing_is_written_around_the_arrays_the_state_or_the_workspace(flint, state_offset, workspace_offset):
    # World.step checks, after every call: the guard bands around each array of each entry (HEAD
    # floats before it, TAIL = 8 after), around the state's 24 bytes and the workspace's 16 and the
    # room after it, and that the gradients, the norm, the count and the table are unchanged.
    world = World(flint, [3000, 1, 4097, 12, 0, 2048], seed=70, state_offset=state_offset,
                  workspace_offset=workspace_offset, workspace_extra=4096)  # fmt: skip

    results = run_standard(world, seed=70)

    check_standard(world, results)


@pytest.mark.parametrize("fill", ["zero", "garbage", "ones"])
def test_the_workspaces_old_content_changes_nothing(flint, fill):
    world = World(flint, [9000, 5, 4097], seed=71)
    reference = World(flint, [9000, 5, 4097], seed=71)

    results = run_standard(world, seed=71, workspace_fill=fill)
    run_standard(reference, seed=71)

    check_standard(world, results)
    assert_same_arrays(world, reference, f"a workspace of {fill}")


@pytest.mark.parametrize("extra", [16, 4096])
def test_the_workspace_may_be_larger_than_needed(flint, extra):
    world = World(flint, [9000, 5, 4097], seed=72, workspace_extra=extra)
    reference = World(flint, [9000, 5, 4097], seed=72)

    results = run_standard(world, seed=72)
    run_standard(reference, seed=72)

    check_standard(world, results)
    assert_same_arrays(world, reference, f"{extra} bytes more workspace")


def test_an_entry_of_count_0_with_null_or_garbage_pointers_touches_nothing(flint):
    # Counts of 0 are never read through their pointers (memcheck stays clean), and the entries
    # are no chunks: the bits are those of the table without them.
    garbage = {0: (0, 0, 0, 0, 0, 1), 2: (0xDEADBEEF0000, 1, 0x7FFFFFFFFFFFFFF0, 3, 0, 0), 5: (1, 2, 3, 4, 0, 1),
               6: (0x10, 0x20, 0x30, 0x40, 0, 0)}  # fmt: skip
    world = World(flint, [5000, 7, 4097], [True, False, True], seed=73, garbage_entries=garbage)
    reference = World(flint, [5000, 7, 4097], [True, False, True], seed=73)
    assert world.table_count == 7

    results = run_standard(world, seed=73)
    run_standard(reference, seed=73)

    check_standard(world, results)
    assert_same_arrays(world, reference, "entries of count 0")


# -- Determinism and streams ---------------------------------------------------------------------------------


def test_run_twice_is_bitwise_identical(flint):
    counts = sanitizer_size([20000, 3, 9000, 4097], [3000, 3, 900, 2049])
    first = World(flint, counts, [True, False, True, True], seed=80)
    second = World(flint, counts, [True, False, True, True], seed=80, state_offset=64, workspace_offset=48)
    third = World(flint, counts, [True, False, True, True], seed=80, offsets=(1, 3, 2, 1))

    run_standard(first, seed=80)
    run_standard(second, seed=80, workspace_fill="garbage")
    run_standard(third, seed=80, workspace_fill="zero")

    assert_same_arrays(first, second, "the second run")
    assert_same_arrays(first, third, "the third run")


def test_two_different_tables_in_a_row_on_one_workspace(flint):
    # Four tables, their calls queued one after the other on one stream and one shared workspace
    # with no sync between: each gives its own table's bits.
    shapes = [([9000, 5], [True, False]), ([1] * 30, [True] * 30), ([4097, 4096, 3], [False, True, True]), ([20000], [True])]
    worlds = [World(flint, counts, decays, seed=90 + i) for i, (counts, decays) in enumerate(shapes)]
    shared = torch.full((16,), 0x5A, dtype=torch.uint8, device="cuda")
    for number, order in enumerate(([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2])):
        pending = {i: worlds[i].begin(make_gradients(worlds[i].counts, 90 + i, number, 0.6)) for i in order}
        for i in order:
            worlds[i].launch(workspace=shared)
        for i in order:
            worlds[i].finish(pending[i])


STREAM_MODES = ["explicit-stream", "current-stream", "default-stream"]


@pytest.mark.parametrize("mode", STREAM_MODES)
def test_runs_on_the_given_stream(flint, mode):
    counts = [20000, 9, 4097]
    world = World(flint, counts, [True, False, True], seed=95)
    # A first call loads the kernels' module, which takes longer than the sleep below: without this
    # warm-up a call on the wrong stream would start late enough to pass.
    World(flint, [100], seed=96).step(make_gradients([100], 96, 0, 0.6))
    torch.cuda.synchronize()
    pending = world.begin(make_gradients(counts, 95, 0, 0.6), defer=True)
    stream = torch.cuda.Stream()

    def enqueue(call_stream):
        # The copy of the gradients into their arrays finishes only after the sleep, so the norm and
        # the step are right only if both calls are queued on the same stream behind it (the arrays
        # hold a sentinel until then). With the current stream the calls have no stream argument,
        # and torch's current stream is the side one; with the default stream there is no side one.
        torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
        world.dev[1].copy_(pending["source"])
        flint.gradient_norm(world.norm, world.nonfinite, world.grad_table, len(counts), world.norm_workspace,
                            stream=call_stream)  # fmt: skip
        world.launch(stream=call_stream)

    if mode == "default-stream":
        enqueue(None)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            enqueue(stream if mode == "explicit-stream" else None)
        stream.synchronize()

    world.finish(pending)


# -- Argument errors -----------------------------------------------------------------------------------------

NAMES = ("state", "workspace", "norm", "nonfinite", "tensors")
TENSOR_COUNT = 2
BYTES = {"state": 24, "workspace": 16, "norm": 8, "nonfinite": 8, "tensors": 48 * TENSOR_COUNT}
SPACING = 32768
ARENA_SLOT = {name: (i + 2) * SPACING for i, name in enumerate(NAMES)}
FILL = 0x5A
LIMIT = (2**64 - 1) // 48  # the most entries whose table's size doesn't wrap


class Arena:
    """One buffer of sentinel bytes holding the call's five buffers at fixed slots (which a test
    moves around), and, outside it, the two entries' real arrays: after a rejected call, nothing
    in the buffer and none of the arrays may have changed."""

    def __init__(self):
        self.bytes = torch.full((8 * SPACING,), FILL, dtype=torch.uint8, device="cuda")
        g = torch.Generator().manual_seed(110)
        self.arrays = [[draw(8, -100, 100, g).cuda() * 2.0**-6 for _ in range(4)] for _ in range(TENSOR_COUNT)]
        self.rows = flint_lib.adamw_table([(*a, True) for a in self.arrays])
        self.slot = dict(ARENA_SLOT)

    def addresses(self):
        return {name: self.bytes.data_ptr() + self.slot[name] for name in NAMES}

    def place_table(self):
        self.bytes[self.slot["tensors"] : self.slot["tensors"] + 96] = self.rows.view(torch.uint8).flatten()

    def snapshot(self):
        return self.bytes.clone(), [[x.clone() for x in a] for a in self.arrays]

    def unchanged(self, snapshot):
        torch.cuda.synchronize()
        return torch.equal(self.bytes, snapshot[0]) and all(
            torch.equal(x.view(torch.int32), y.view(torch.int32)) for a, b in zip(self.arrays, snapshot[1]) for x, y in zip(a, b)
        )


VALID = dict(lr=6e-4)


def call(flint, a, tensor_count=TENSOR_COUNT, workspace_bytes=16, config="valid"):
    config = flint_lib.adamw_config(**VALID) if config == "valid" else config
    flint.adamw(a["tensors"], tensor_count, a["norm"], a["nonfinite"], a["state"], config, a["workspace"], workspace_bytes)


def expect_invalid_argument(flint, reason, arena, a, **kwargs):
    """The call fails with the status and a message naming the function and the problem, and
    nothing changed: no kernel ran (the state and the arrays keep their bits)."""
    before = arena.snapshot()
    with pytest.raises(flint_lib.FlintError) as error:
        call(flint, a, **kwargs)
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    for text in ("flint_adamw", reason):
        assert text in str(error.value), (text, str(error.value))
        assert text in flint.last_error()
    assert arena.unchanged(before), "the call wrote something"


def bad_config(**fields):
    config = flint_lib.adamw_config(**VALID)
    for name, value in fields.items():
        setattr(config, name, value)
    return config


CONFIG_ERRORS = {
    "null-config": ("config is null", None),
    "lr--1": ("lr (", dict(lr=-1.0)),
    "lr-nan": ("lr (", dict(lr=math.nan)),
    "lr-inf": ("lr (", dict(lr=math.inf)),
    "beta1-1": ("beta1 (", dict(beta1=1.0)),
    "beta1--0.1": ("beta1 (", dict(beta1=-0.1)),
    "beta1-nan": ("beta1 (", dict(beta1=math.nan)),
    "beta2-1": ("beta2 (", dict(beta2=1.0)),
    "beta2--0.1": ("beta2 (", dict(beta2=-0.1)),
    "beta2-nan": ("beta2 (", dict(beta2=math.nan)),
    "eps-0": ("eps (", dict(eps=0.0)),
    "eps--1": ("eps (", dict(eps=-1.0)),
    "eps-inf": ("eps (", dict(eps=math.inf)),
    "eps-nan": ("eps (", dict(eps=math.nan)),
    "weight_decay--0.1": ("weight_decay (", dict(weight_decay=-0.1)),
    "weight_decay-inf": ("weight_decay (", dict(weight_decay=math.inf)),
    "weight_decay-nan": ("weight_decay (", dict(weight_decay=math.nan)),
    "max_norm-0": ("max_norm (", dict(max_norm=0.0)),
    "max_norm--1": ("max_norm (", dict(max_norm=-1.0)),
    "max_norm-nan": ("max_norm (", dict(max_norm=math.nan)),
    "growth_interval-0": ("growth_interval (0)", dict(growth_interval=0)),
    "growth_interval--1": ("growth_interval (-1)", dict(growth_interval=-1)),
    "growth_interval-2^31": ("growth_interval (2147483648)", dict(growth_interval=2**31)),
}


@pytest.mark.parametrize("reason, fields", CONFIG_ERRORS.values(), ids=list(CONFIG_ERRORS))
def test_a_bad_config_is_rejected_naming_the_field_and_launches_nothing(flint, reason, fields):
    arena = Arena()
    arena.place_table()
    config = None if fields is None else bad_config(**fields)
    expect_invalid_argument(flint, reason, arena, arena.addresses(), config=config)


def test_boundary_configs_are_accepted(flint):
    # max_norm is allowed to be +inf, lr and weight_decay 0, the betas 0, the interval 1 and 2^31 - 1.
    for fields in (dict(max_norm=math.inf), dict(lr=0.0), dict(weight_decay=0.0), dict(beta1=0.0, beta2=0.0),
                   dict(growth_interval=1), dict(growth_interval=2**31 - 1),
                   dict(beta1=0.9999999, beta2=0.9999999)):  # fmt: skip
        world = World(flint, [300, 5], seed=111, config=default_config(**fields))
        world.step(make_gradients(world.counts, 111, 0, 0.6))


@pytest.mark.parametrize("name", NAMES)
def test_null_pointer(flint, name):
    arena = Arena()
    arena.place_table()
    expect_invalid_argument(flint, f"{name} is null", arena, {**arena.addresses(), name: None})


@pytest.mark.parametrize("name", NAMES)
def test_misaligned_pointer(flint, name):
    # 4 bytes is as far off as an FP32 goes, 8 is aligned for 64 bits but not 128.
    arena = Arena()
    arena.place_table()
    for offset in (4, 8):
        a = arena.addresses()
        a[name] += offset
        expect_invalid_argument(flint, f"{name} is not 16-byte aligned", arena, a)


@pytest.mark.parametrize("name", ["state", "workspace", "tensors"])
def test_a_buffer_may_not_run_past_the_address_space(flint, name):
    arena = Arena()
    arena.place_table()
    expect_invalid_argument(flint, f"{name} runs past the end of the address space", arena,
                            {**arena.addresses(), name: 2**64 - 16})  # fmt: skip


def test_a_workspace_may_not_be_too_small(flint):
    arena = Arena()
    arena.place_table()
    for given in (15, 8, 1, 0):
        expect_invalid_argument(flint, f"needs 16 bytes of workspace, given {given}", arena, arena.addresses(),
                                workspace_bytes=given)  # fmt: skip


def test_checks_are_made_in_order(flint):
    # The config, then tensor_count, then the workspace's size, then the pointers (per buffer:
    # null, alignment, the address space; state, workspace, norm, nonfinite, tensors), then the
    # overlaps.
    arena = Arena()
    arena.place_table()
    a = arena.addresses()
    nothing = {name: None for name in NAMES}
    expect_invalid_argument(flint, "config is null", arena, nothing, tensor_count=2**63, workspace_bytes=0, config=None)
    expect_invalid_argument(flint, f"tensor_count ({2**60}) can be up to {LIMIT}", arena, nothing,
                            tensor_count=2**60, workspace_bytes=0)  # fmt: skip
    expect_invalid_argument(flint, "needs 16 bytes of workspace, given 0", arena, nothing, workspace_bytes=0)
    expect_invalid_argument(flint, "state is null", arena, nothing)
    expect_invalid_argument(flint, "state is not 16-byte aligned", arena, {**a, "state": a["state"] + 4, "workspace": None})
    expect_invalid_argument(flint, "state runs past the end", arena, {**a, "state": 2**64 - 16, "workspace": None})
    expect_invalid_argument(flint, "workspace is null", arena, {**a, "workspace": None})
    expect_invalid_argument(flint, "norm is null", arena, {**a, "norm": None, "nonfinite": a["nonfinite"] + 4})
    expect_invalid_argument(flint, "tensors is null", arena, {**a, "tensors": None, "state": a["workspace"]})
    expect_invalid_argument(flint, "state and workspace overlap", arena, {**a, "workspace": a["state"]})


@pytest.mark.parametrize("count", [LIMIT + 1, 2**60, 2**63, 2**64 - 1], ids=str)
def test_a_tensor_count_whose_table_size_wraps_is_rejected(flint, count):
    arena = Arena()
    expect_invalid_argument(flint, f"tensor_count ({count}) can be up to {LIMIT}", arena, arena.addresses(),
                            tensor_count=count)  # fmt: skip


def test_the_largest_table_that_does_not_wrap_runs_past_the_address_space(flint):
    arena = Arena()
    expect_invalid_argument(flint, "tensors runs past the end of the address space", arena, arena.addresses(),
                            tensor_count=LIMIT)  # fmt: skip


def overlapping_offsets(first, second):
    """Where `second` starts relative to `first` (a multiple of 16) so that they overlap: at the
    same address, 16 bytes in, in first's last 16 bytes, and (negative) with second's last 16
    bytes inside first's start. Those that really overlap, without duplicates."""
    a, b = BYTES[first], BYTES[second]
    candidates = {"same-start": 0, "16-in": 16, "last-16-of-first": (a - 1) // 16 * 16,
                  "ends-in-first": -((b - 1) // 16 * 16)}  # fmt: skip
    out = {}
    for label, d in candidates.items():
        if d % 16 == 0 and d < a and d + b > 0 and d not in out.values():
            out[label] = d
    return out


# The outputs (state and workspace) may overlap neither each other nor anything after them in the
# order of the checks; norm, nonfinite and the table are only read, and may share memory.
PAIRS = [(a, b) for i, a in enumerate(NAMES[:2]) for b in NAMES[i + 1 :]]


@pytest.mark.parametrize("first, second", PAIRS, ids=lambda v: str(v))
def test_the_state_and_the_workspace_may_overlap_neither_each_other_nor_the_inputs(flint, first, second):
    # The buffers are real memory in an arena, so a call that wrongly went ahead would do no harm
    # to anything else.
    offsets = overlapping_offsets(first, second)
    assert offsets, (first, second)
    for label, offset in offsets.items():
        arena = Arena()
        a = arena.addresses()
        a[second] = a[first] + offset
        expect_invalid_argument(flint, f"{first} and {second} overlap", arena, a)


def test_the_overlap_cases_cover_the_outputs_pairs(flint):
    assert len(PAIRS) == 7 and ("state", "workspace") in PAIRS and ("workspace", "tensors") in PAIRS
    assert ("norm", "nonfinite") not in PAIRS, "the inputs may share memory"


def test_buffers_next_to_each_other_are_not_overlapping_and_the_call_runs(flint):
    # Every pair of buffers placed right after and right before each other (no gap): the call is
    # valid, changes the state, the workspace and the arrays, and nothing else in the arena.
    rounded = lambda n: -(-n // 16) * 16  # noqa: E731
    for first, second in [(a, b) for i, a in enumerate(NAMES) for b in NAMES[i + 1 :]]:
        for label in ("just-after", "just-before"):
            arena = Arena()
            start = dict(ARENA_SLOT)
            start[second] = start[first] + (rounded(BYTES[first]) if label == "just-after" else -rounded(BYTES[second]))
            arena.slot = start
            arena.place_table()
            base = arena.bytes.data_ptr()
            arena.bytes[start["state"] : start["state"] + 24] = flint_lib.adamw_state(SCALE).to(torch.uint8)
            arena.bytes[start["norm"] : start["norm"] + 8] = torch.tensor([1000.0], dtype=torch.float64).view(torch.uint8).cuda()
            arena.bytes[start["nonfinite"] : start["nonfinite"] + 8] = 0
            used = torch.zeros(arena.bytes.numel(), dtype=torch.bool, device="cuda")
            for name in ("state", "workspace"):
                used[start[name] : start[name] + BYTES[name]] = True
            outside = arena.bytes[~used].clone()
            before = [x.clone() for a in arena.arrays for x in a]

            call(flint, {name: base + start[name] for name in NAMES})
            torch.cuda.synchronize()

            assert torch.equal(arena.bytes[~used], outside), f"{first}, {second}, {label}: something else changed"
            state = flint_lib.read_adamw_state(base + start["state"])
            assert state == {"scale": SCALE, "growth_tracker": 1, "step": 1, "skipped": 0}, (first, second, label)
            assert not torch.equal(arena.arrays[0][0], before[0]), "the weights moved"


def test_an_error_leaves_nothing_behind_and_a_valid_call_works_after(flint):
    arena = Arena()
    arena.place_table()
    a = arena.addresses()
    for kwargs in (dict(tensor_count=2**60), dict(workspace_bytes=15), dict(config=None)):
        with pytest.raises(flint_lib.FlintError):
            call(flint, a, **kwargs)
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "state": None})
    with pytest.raises(flint_lib.FlintError):
        call(flint, {**a, "workspace": a["workspace"] + 4})
    arena.bytes[ARENA_SLOT["state"] : ARENA_SLOT["state"] + 24] = flint_lib.adamw_state(SCALE).to(torch.uint8)
    norm_bytes = torch.tensor([1000.0], dtype=torch.float64).view(torch.uint8).cuda()
    arena.bytes[ARENA_SLOT["norm"] : ARENA_SLOT["norm"] + 8] = norm_bytes
    arena.bytes[ARENA_SLOT["nonfinite"] : ARENA_SLOT["nonfinite"] + 8] = 0

    call(flint, a)

    assert flint_lib.read_adamw_state(a["state"]) == {"scale": SCALE, "growth_tracker": 1, "step": 1, "skipped": 0}


def test_the_workspace_function_needs_a_bytes_pointer(flint):
    status = flint.lib.flint_adamw_workspace(None)

    assert status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_adamw_workspace: bytes is null" in flint.last_error()


def test_the_workspace_function_writes_the_size(flint):
    size = ctypes.c_size_t(0xDEADBEEF)

    assert flint.lib.flint_adamw_workspace(ctypes.byref(size)) == flint_lib.OK
    assert size.value == 16


# -- The model ---------------------------------------------------------------------------------------------------

MODEL_SHAPES = {"tiny": (TINY, 2, 32), **ODD_MODELS}
if SANITIZER:
    MODEL_SHAPES = {name: MODEL_SHAPES[name] for name in ("one-head-of-8", "heads-of-24")}
SENTINEL64 = 0x5A5A5A5A5A5A5A5A
ROUNDS = 2 if SANITIZER else 3  # rounds of forward, backward and step (the sanitizer's handful)
# The model of the tests that need just one: the tiny config, or under the sanitizer a smaller one.
SMALL, SMALL_SEQUENCES, SMALL_LENGTH = ODD_MODELS["heads-of-24"] if SANITIZER else (TINY, 2, 32)


def optimizer_config(cfg, sequences, length, separate=False):
    return gpt2_config(cfg, sequences, length, separate, backward=1, optimizer=1)


def decays_by_name(state):
    """Which parameters decay: those of dimension 2 or more, as gpt2_torch.parameter_groups has it."""
    return {name: p.dim() >= 2 for name, p in state.items()}


def read_model(m):
    """Copies, in the order of the parameters, of the params, the gradients and both moments (FP32
    device tensors of (rows, cols))."""
    params = [m.read(a, (r, c), torch.float32) for _, a, r, c in m.parameters()]
    grads = [m.read(a, (r, c), torch.float32) for _, a, r, c in m.gradients()]
    moments = m.moments()
    avg = [m.read(a, (r, c), torch.float32) for _, a, v, r, c in moments]
    avg_sq = [m.read(v, (r, c), torch.float32) for _, a, v, r, c in moments]
    return params, grads, avg, avg_sq


def same_bits(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def wtes_padded_rows_are_zero(m, cfg):
    address = m.parameters()[0][1]
    rows = m.read(address, (cfg.padded_vocab_size, cfg.n_embd), torch.float32)[cfg.vocab_size :]
    return rows.numel() > 0 and not rows.view(torch.int32).any()


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
@pytest.mark.parametrize("name", MODEL_SHAPES)
def test_the_models_step_is_flint_adamw_on_copies_and_the_transcription(flint, name, separate):
    # Three rounds of forward, backward and step. Before each step, copies of the model's params,
    # gradients, moments and state go through flint_adamw (with the model's norm and count) and
    # through the transcription: the model's step must leave their bits.
    cfg, sequences, length = MODEL_SHAPES[name]
    weights = make_model(cfg, 3, "mixed")
    state = state_of(weights)
    decays = decays_by_name(state)
    config = flint_lib.adamw_config(6e-4)
    ns = namespace(config)
    model_config = optimizer_config(cfg, sequences, length, separate)
    np_state = initial_state()
    with Gpt2(flint, model_config) as m:
        m.load(state)
        m.reset_optimizer(SCALE)
        assert m.read_optimizer_state() == {"scale": SCALE, "growth_tracker": 0, "step": 0, "skipped": 0}
        names = [p[0] for p in m.parameters()]
        assert [n for n, d in decays.items() if d] == [n for n in names if decays[n]]
        assert wtes_padded_rows_are_zero(m, cfg)
        for iteration in range(ROUNDS):
            batch = random_batch(cfg, sequences, length, 20 + iteration)
            scale = gate_scale(batch.rows)
            forward_backward(m, batch, scale)
            m.gradient_norm()
            norm, count = m.read_gradient_norm()
            params, grads, avg, avg_sq = read_model(m)
            if iteration == 0:
                assert all(not t.view(torch.int32).any() for t in avg + avg_sq), "reset_optimizer gave +0 moments"
            # By hand: flint_adamw on copies.
            copies = [[t.clone() for t in group] for group in (params, avg, avg_sq)]
            table = flint_lib.adamw_table([(p, g, a, v, decays[n])
                                           for n, p, g, a, v in zip(names, copies[0], grads, copies[1], copies[2])])
            copy_state = flint_lib.adamw_state(**m.read_optimizer_state())
            flint.adamw(table, len(names), torch.tensor([norm], dtype=torch.float64, device="cuda"),
                        torch.tensor([count], dtype=torch.int64, device="cuda"), copy_state, config,
                        torch.zeros(16, dtype=torch.uint8, device="cuda"))  # fmt: skip
            # By the transcription.
            scalars, np_state = prepare(norm, count, np_state, ns)
            assert not scalars["skip"] and count == 0
            expected = [update(p.cpu().numpy().reshape(-1), g.cpu().numpy().reshape(-1), a.cpu().numpy().reshape(-1),
                               v.cpu().numpy().reshape(-1), decays[n], scalars, ns)
                        for n, p, g, a, v in zip(names, params, grads, avg, avg_sq)]  # fmt: skip

            m.step(config)
            torch.cuda.synchronize()

            after = read_model(m)
            for i, n in enumerate(names):
                for k, label in ((0, "param"), (2, "exp_avg"), (3, "exp_avg_sq")):
                    by_hand = copies[(0, None, 1, 2)[k]][i]
                    assert same_bits(after[k][i], by_hand), (
                        f"iteration {iteration}: {n} {label} differs from flint_adamw on a copy")
                    host = expected[i][(0, None, 1, 2)[k]]
                    assert np.array_equal(after[k][i].cpu().numpy().reshape(-1).view(np.int32), host.view(np.int32)), (
                        f"iteration {iteration}: {n} {label} differs from the transcription")
                assert same_bits(after[1][i], grads[i]), f"{n}: the step changed a gradient"
            device_state = m.read_optimizer_state()
            assert states_equal(device_state, np_state) and states_equal(device_state, flint_lib.read_adamw_state(copy_state))
            assert wtes_padded_rows_are_zero(m, cfg)
            if iteration == 0:
                assert device_state == {"scale": SCALE, "growth_tracker": 1, "step": 1, "skipped": 0}
                # The FP16 copies are current: a forward pass gives the bits it gives after one more sync.
                m.forward(batch.tokens, batch.target_tokens, scale)
                first = read_outputs(m, model_config).logits.clone()
                m.sync_weights()
                m.forward(batch.tokens, batch.target_tokens, scale)
                second = read_outputs(m, model_config).logits
                assert torch.equal(first.view(torch.int16), second.view(torch.int16))
        assert m.read_optimizer_state() == {"scale": SCALE, "growth_tracker": ROUNDS, "step": ROUNDS, "skipped": 0}
        moved = sum(not same_bits(p, state[n].reshape(p.shape)) for n, p in zip(names, read_model(m)[0]))
    assert moved == len(names), "every parameter moved"


def test_weight_decay_applies_to_exactly_the_two_dimensional_parameters(flint):
    # With every gradient 0 the only thing a step does is the decay: p * (1 - lr * wd), in FP64
    # rounded once, on the parameters of gpt2_torch.parameter_groups' decayed group, by name.
    weights = make_model(SMALL, 4, "mixed")
    state = state_of(weights)
    groups = G.parameter_groups(weights)
    decayed_ids = {id(p) for p in groups[0]["params"]}
    decayed = {n for n, p in weights.named_parameters() if id(p) in decayed_ids}
    assert decayed == {n for n, p in state.items() if p.dim() >= 2} and 0 < len(decayed) < len(state)
    batch = random_batch(SMALL, SMALL_SEQUENCES, SMALL_LENGTH, 5)
    lr, wd = 1e-2, 0.1
    with Gpt2(flint, optimizer_config(SMALL, SMALL_SEQUENCES, SMALL_LENGTH)) as m:
        m.load(state)
        m.reset_optimizer(SCALE)
        forward_backward(m, batch, gate_scale(batch.rows))
        for _, address, rows, cols in m.gradients():
            m.flint.copy(address, torch.zeros(rows * cols, dtype=torch.float32, device="cuda"), rows * cols * 4)
        torch.cuda.synchronize()
        before = read_model(m)[0]

        m.step(flint_lib.adamw_config(lr, weight_decay=wd))

        after = read_model(m)[0]
        for (n, *_), b, a in zip(m.parameters(), before, after):
            if n in decayed:
                p64 = b.cpu().numpy().astype(F64)
                expected = (p64 - (lr * wd) * p64).astype(F32)
                assert np.array_equal(a.cpu().numpy().view(np.int32), expected.view(np.int32)), n
                assert not same_bits(a, b) or not b.any(), f"{n} did not decay"
            else:
                assert same_bits(a, b), f"{n} changed"
        assert m.read_optimizer_state()["step"] == 1


@pytest.mark.parametrize("kind", ["inf"] if SANITIZER else ["inf", "nan"])
@pytest.mark.parametrize("place", ["wte-first", "layer-bias", "ln_f-last"])
def test_a_non_finite_gradient_skips_the_models_step_and_changes_nothing(flint, place, kind):
    weights = make_model(SMALL, 6, "mixed")
    state = state_of(weights)
    places = {"wte-first": ("wte.weight", 0), "layer-bias": (f"blocks.{SMALL.n_layer // 2}.attn.c_proj.bias", SMALL.n_embd // 2),
              "ln_f-last": ("ln_f.bias", SMALL.n_embd - 1)}  # fmt: skip
    parameter, element = places[place]
    batch = random_batch(SMALL, SMALL_SEQUENCES, SMALL_LENGTH, 7)
    config = flint_lib.adamw_config(6e-4)
    model_config = optimizer_config(SMALL, SMALL_SEQUENCES, SMALL_LENGTH)
    with Gpt2(flint, model_config) as m:
        m.load(state)
        m.reset_optimizer(SCALE)
        scale = gate_scale(batch.rows)
        forward_backward(m, batch, scale)
        m.step(config)  # a clean step first, so the moments aren't zeros
        m.forward(batch.tokens, batch.target_tokens, scale)
        logits = read_outputs(m, model_config).logits.clone()
        forward_backward(m, batch, scale)
        address = {n: a for n, a, *_ in m.gradients()}[parameter]
        bad = torch.tensor([as_int32(NON_FINITE_BITS[kind])], dtype=torch.int32, device="cuda")
        m.flint.copy(address + 4 * element, bad, 4)
        torch.cuda.synchronize()
        before = read_model(m)
        state_before = m.read_optimizer_state()
        assert state_before == {"scale": SCALE, "growth_tracker": 1, "step": 1, "skipped": 0}

        m.step(config)
        torch.cuda.synchronize()

        after = read_model(m)
        assert m.read_gradient_norm()[1] == 1
        for k, label in ((0, "param"), (2, "exp_avg"), (3, "exp_avg_sq")):
            assert all(same_bits(a, b) for a, b in zip(after[k], before[k])), f"a {label} changed"
        assert m.read_optimizer_state() == {"scale": SCALE / 2, "growth_tracker": 0, "step": 1, "skipped": 1}
        m.forward(batch.tokens, batch.target_tokens, scale)
        assert torch.equal(read_outputs(m, model_config).logits.view(torch.int16),
                           logits.view(torch.int16))


def test_a_run_resumes_from_another_models_weights_moments_and_state(flint):
    # Model A takes 3 steps; its weights, moments and state go to a fresh model B through the
    # accessors; both take 3 more rounds on the same tokens and agree bit for bit.
    weights = make_model(SMALL, 8, "mixed")
    config = flint_lib.adamw_config(1e-3)
    model_config = optimizer_config(SMALL, SMALL_SEQUENCES, SMALL_LENGTH)

    def rounds(m, first, count):
        losses = []
        for i in range(first, first + count):
            batch = random_batch(SMALL, SMALL_SEQUENCES, SMALL_LENGTH, 30 + i)
            scale = m.read_optimizer_state()["scale"] / batch.rows
            m.forward(batch.tokens, batch.target_tokens, scale)
            losses.append(m.mean_loss())
            m.backward(batch.tokens)
            m.step(config)
        torch.cuda.synchronize()
        return losses

    with Gpt2(flint, model_config) as a, Gpt2(flint, model_config) as b:
        a.load(state_of(weights))
        a.reset_optimizer(SCALE)
        rounds(a, 0, ROUNDS)
        a_params, _, a_avg, a_avg_sq = read_model(a)
        b.load({n: p for (n, *_), p in zip(a.parameters(), a_params)})
        for (_, avg_at, avg_sq_at, rows, cols), x, y in zip(b.moments(), a_avg, a_avg_sq):
            b.flint.copy(avg_at, x, rows * cols * 4)
            b.flint.copy(avg_sq_at, y, rows * cols * 4)
        torch.cuda.synchronize()
        b.write_optimizer_state(a.read_optimizer_state())
        resumed = {"scale": SCALE, "growth_tracker": ROUNDS, "step": ROUNDS, "skipped": 0}
        assert b.read_optimizer_state() == a.read_optimizer_state() == resumed

        losses_a = rounds(a, ROUNDS, ROUNDS)
        losses_b = rounds(b, ROUNDS, ROUNDS)

        assert losses_a == losses_b, (losses_a, losses_b)
        ra, rb = read_model(a), read_model(b)
        for k, label in ((0, "params"), (2, "exp_avg"), (3, "exp_avg_sq")):
            assert all(same_bits(x, y) for x, y in zip(ra[k], rb[k])), label
        finished = {"scale": SCALE, "growth_tracker": 2 * ROUNDS, "step": 2 * ROUNDS, "skipped": 0}
        assert a.read_optimizer_state() == b.read_optimizer_state() == finished


def test_reset_optimizer_zeroes_both_moments_and_the_next_step_is_a_fresh_models_first(flint):
    # After a few steps, reset_optimizer leaves every moment (the first and the second, bit for
    # bit +0) and the state as {initial_scale, 0, 0, 0}; and the next step gives the bits that a
    # fresh model, loaded with the same weights and the same gradients, gives on its first.
    weights = make_model(SMALL, 12, "mixed")
    config = flint_lib.adamw_config(1e-3)
    model_config = optimizer_config(SMALL, SMALL_SEQUENCES, SMALL_LENGTH)
    with Gpt2(flint, model_config) as a, Gpt2(flint, model_config) as b:
        a.load(state_of(weights))
        a.reset_optimizer(SCALE)
        for i in range(ROUNDS):
            batch = random_batch(SMALL, SMALL_SEQUENCES, SMALL_LENGTH, 50 + i)
            forward_backward(a, batch, gate_scale(batch.rows))
            a.step(config)
        params, grads, avg, avg_sq = read_model(a)
        assert all(t.view(torch.int32).any() for t in avg + avg_sq), "every moment moved"
        assert a.read_optimizer_state()["step"] == ROUNDS

        a.reset_optimizer(SCALE)
        torch.cuda.synchronize()

        zeroed = read_model(a)
        assert all(not t.view(torch.int32).any() for t in zeroed[2]), "exp_avg is not all +0"
        assert all(not t.view(torch.int32).any() for t in zeroed[3]), "exp_avg_sq is not all +0"
        assert a.read_optimizer_state() == {"scale": SCALE, "growth_tracker": 0, "step": 0, "skipped": 0}
        assert all(same_bits(x, y) for x, y in zip(zeroed[0], params)), "the reset touched the weights"
        assert all(same_bits(x, y) for x, y in zip(zeroed[1], grads)), "the reset touched the gradients"
        # B: A's current weights and gradients, then its own reset and first step.
        b.load({n: p for (n, *_), p in zip(a.parameters(), params)})
        for (_, address, rows, cols), g in zip(b.gradients(), grads):
            b.flint.copy(address, g, rows * cols * 4)
        b.reset_optimizer(SCALE)
        torch.cuda.synchronize()

        a.step(config)
        b.step(config)
        torch.cuda.synchronize()

        ra, rb = read_model(a), read_model(b)
        for k, label in ((0, "params"), (2, "exp_avg"), (3, "exp_avg_sq")):
            assert all(same_bits(x, y) for x, y in zip(ra[k], rb[k])), label
        assert a.read_optimizer_state() == b.read_optimizer_state()
        assert a.read_optimizer_state() == {"scale": SCALE, "growth_tracker": 1, "step": 1, "skipped": 0}


@pytest.mark.skipif(SANITIZER, reason="a training run is too long for the sanitizer")
def test_a_smoke_run_on_one_batch_lowers_the_loss(flint):
    batch = random_batch(TINY, 2, 32, 40)
    config = flint_lib.adamw_config(1e-3)
    model_config = optimizer_config(TINY, 2, 32)
    with Gpt2(flint, model_config) as m:
        m.load(state_of(make_model(TINY, 9, "init")))
        m.reset_optimizer(SCALE)
        losses = []
        for _ in range(30):
            m.forward(batch.tokens, batch.target_tokens, m.read_optimizer_state()["scale"] / batch.rows)
            losses.append(m.mean_loss())
            m.backward(batch.tokens)
            m.step(config)
        state = m.read_optimizer_state()
    print(f"SMOKE loss {losses[0]:.4f} at the first step, {losses[-1]:.4f} at the 30th; state {state}")
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0] - 1.0, losses
    assert state["step"] + state["skipped"] == 30


# -- The model's errors and layout --------------------------------------------------------------------------


def optimizer_functions(config):
    return {
        "flint_gpt2_reset_optimizer": lambda lib, handle: lib.flint_gpt2_reset_optimizer(handle, SCALE, None),
        "flint_gpt2_step": lambda lib, handle: lib.flint_gpt2_step(handle, ctypes.byref(config), None),
        "flint_gpt2_moments": lambda lib, handle: lib.flint_gpt2_moments(handle, 0, None, None),
        "flint_gpt2_optimizer_state": lambda lib, handle: lib.flint_gpt2_optimizer_state(
            handle, ctypes.byref(ctypes.c_void_p())),
    }


@pytest.mark.parametrize("function", ["flint_gpt2_reset_optimizer", "flint_gpt2_step", "flint_gpt2_moments",
                                      "flint_gpt2_optimizer_state"])  # fmt: skip
@pytest.mark.parametrize("model", ["forward-only", "backward-without-optimizer", "null"])
def test_the_optimizer_functions_fail_without_an_optimizer_or_a_model(flint, model, function):
    config = flint_lib.adamw_config(1e-3)
    call_it = optimizer_functions(config)[function]
    if model == "null":
        assert call_it(flint.lib, None) == flint_lib.ERROR_INVALID_ARGUMENT
        assert function in flint.last_error() and "model is null" in flint.last_error(), flint.last_error()
        return
    model_config = gpt2_config(SMALL, 1, 8) if model == "forward-only" else backward_config(SMALL, 1, 8)
    with Gpt2(flint, model_config) as m:
        assert call_it(flint.lib, m.handle) == flint_lib.ERROR_INVALID_ARGUMENT
        assert function in flint.last_error() and "without optimizer" in flint.last_error(), flint.last_error()


def test_a_config_with_optimizer_and_no_backward_fails_to_create(flint):
    with pytest.raises(flint_lib.FlintError) as error:
        Gpt2(flint, gpt2_config(SMALL, 1, 8, optimizer=1))
    assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
    assert "flint_gpt2_create" in str(error.value) and "optimizer" in str(error.value) and "backward" in str(error.value)
    # And flint_gpt2_bytes agrees with create.
    n = ctypes.c_size_t(0)
    config = gpt2_config(SMALL, 1, 8, optimizer=1)
    assert flint.lib.flint_gpt2_bytes(ctypes.byref(config), ctypes.byref(n)) == flint_lib.ERROR_INVALID_ARGUMENT
    assert "optimizer" in flint.last_error() and "backward" in flint.last_error()


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan], ids=str)
def test_reset_optimizer_needs_a_finite_positive_scale_and_changes_nothing_when_it_fails(flint, scale):
    with Gpt2(flint, optimizer_config(SMALL, 1, 8)) as m:
        m.reset_optimizer(SCALE)
        written = {"scale": 1024.0, "growth_tracker": 7, "step": 5, "skipped": 2}
        m.write_optimizer_state(written)
        _, _, a_at, _, _ = m.moments()[0]
        m.flint.copy(a_at, torch.full((4,), 1.5, device="cuda"), 16)
        torch.cuda.synchronize()

        with pytest.raises(flint_lib.FlintError) as error:
            m.reset_optimizer(scale)

        assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
        assert "flint_gpt2_reset_optimizer" in str(error.value) and "initial_scale" in str(error.value)
        assert m.read_optimizer_state() == written
        assert m.read(a_at, (4,), torch.float32).tolist() == [1.5] * 4, "a moment was reset"


@pytest.mark.parametrize("field, value, reason",
                         [("lr", -1.0, "lr ("), ("max_norm", 0.0, "max_norm ("), (None, None, "config is null")],
                         ids=["lr", "max_norm", "null"])  # fmt: skip
def test_step_with_a_bad_config_fails_before_anything_launches(flint, field, value, reason):
    batch = random_batch(SMALL, SMALL_SEQUENCES, SMALL_LENGTH, 41)
    with Gpt2(flint, optimizer_config(SMALL, SMALL_SEQUENCES, SMALL_LENGTH)) as m:
        m.load(state_of(make_model(SMALL, 10, "mixed")))
        m.reset_optimizer(SCALE)
        forward_backward(m, batch, gate_scale(batch.rows))
        norm_at, count_at = m.gradient_norm_outputs()
        sentinel = torch.full((1,), SENTINEL64, dtype=torch.int64, device="cuda")
        m.flint.copy(norm_at, sentinel, 8)
        m.flint.copy(count_at, sentinel, 8)
        torch.cuda.synchronize()
        before = read_model(m)
        state_before = m.read_optimizer_state()

        with pytest.raises(flint_lib.FlintError) as error:
            if field is None:
                m.flint.check(m.flint.lib.flint_gpt2_step(m.handle, None, None))
            else:
                bad = flint_lib.adamw_config(1e-3)
                setattr(bad, field, value)
                m.step(bad)

        assert error.value.status == flint_lib.ERROR_INVALID_ARGUMENT
        assert "flint_gpt2_step" in str(error.value) and reason in str(error.value), str(error.value)
        torch.cuda.synchronize()
        assert m.read(norm_at, (), torch.int64).item() == SENTINEL64, "the gradient norm ran"
        assert m.read(count_at, (), torch.int64).item() == SENTINEL64, "the gradient norm ran"
        assert m.read_optimizer_state() == state_before
        after = read_model(m)
        assert all(same_bits(x, y) for group_a, group_b in zip(before, after) for x, y in zip(group_a, group_b))


def test_moments_accepts_null_out_pointers_and_rejects_an_index_out_of_range(flint):
    with Gpt2(flint, optimizer_config(SMALL, 1, 8)) as m:
        count = flint.lib.flint_gpt2_parameter_count(m.handle)
        info = flint_lib.Gpt2ParameterInfo()
        assert flint.lib.flint_gpt2_moments(m.handle, 0, None, None) == flint_lib.OK
        assert flint.lib.flint_gpt2_moments(m.handle, 1, ctypes.byref(info), None) == flint_lib.OK
        only_first = info.data
        name, _, rows, cols = m.parameters()[1]
        assert info.name.decode() == name and info.rows * info.cols == rows * cols
        info = flint_lib.Gpt2ParameterInfo()
        assert flint.lib.flint_gpt2_moments(m.handle, 1, None, ctypes.byref(info)) == flint_lib.OK
        assert info.data != only_first and info.data % 16 == 0 and only_first % 16 == 0
        for index in (count, count + 7, 2**63):
            assert flint.lib.flint_gpt2_moments(m.handle, index, ctypes.byref(info), None) == flint_lib.ERROR_INVALID_ARGUMENT
            assert "flint_gpt2_moments" in flint.last_error() and "index" in flint.last_error(), flint.last_error()
        assert flint.lib.flint_gpt2_optimizer_state(m.handle, None) == flint_lib.ERROR_INVALID_ARGUMENT
        assert "flint_gpt2_optimizer_state" in flint.last_error() and "state is null" in flint.last_error()


OPTIMIZER_BYTES = ["tiny-2x32", "odd", "gpt2-1x8", "gpt2-16x1024", "layers-1024", "smallest"]


@pytest.mark.parametrize("key", OPTIMIZER_BYTES)
def test_bytes_with_the_optimizer_equals_an_independent_sum(flint, key):
    fields = BACKWARD_BYTES[key]
    with_optimizer = bytes_of(flint, Gpt2Config(**fields, backward=1, optimizer=1))
    without = bytes_of(flint, Gpt2Config(**fields, backward=1))
    assert with_optimizer == expected_bytes_backward(flint, **fields, optimizer=True)
    assert without == expected_bytes_backward(flint, **fields), "a model without the optimizer changed size"
    assert with_optimizer > without
    print(f"BYTES optimizer {key}: {with_optimizer} = {with_optimizer / 2**30:.3f} GiB, "
          f"{(with_optimizer - without) / 2**30:.3f} GiB more")


@pytest.mark.parametrize("separate", [False, True], ids=["one-allocation", "separate"])
def test_the_models_step_runs_on_the_given_stream(flint, separate):
    cfg, sequences, length = ODD_MODELS["heads-of-24"]
    state = state_of(make_model(cfg, 11, "mixed"))
    batch = random_batch(cfg, sequences, length, 12)
    scale = gate_scale(batch.rows)
    config = flint_lib.adamw_config(1e-3)
    model_config = optimizer_config(cfg, sequences, length, separate)

    def run(m, stream=None):
        forward_backward(m, batch, scale, stream=stream)
        m.step(config, stream)
        torch.cuda.synchronize()
        return read_model(m), m.read_optimizer_state()

    stream = torch.cuda.Stream()
    # The second model is made while the first is alive, so its memory can't be the first's with
    # the expected results still in it; its gradients are poisoned before the stream starts.
    with Gpt2(flint, model_config) as first, Gpt2(flint, model_config) as m:
        first.load(state)
        first.reset_optimizer(SCALE)
        default, default_state = run(first)
        poison_gradients(m)
        # The sleeps keep the stream busy: a step that ran on another stream would race ahead of the
        # weights' copy, the state's reset and the backward pass, and read poisoned gradients.
        with torch.cuda.stream(stream):
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.load(state, stream=stream)
            m.reset_optimizer(SCALE, stream)
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            forward_backward(m, batch, scale, stream=stream)
            torch.cuda._sleep(SIDE_STREAM_SLEEP_CYCLES)
            m.step(config, stream)
        stream.synchronize()
        side, side_state = read_model(m), m.read_optimizer_state()
    for k, label in ((0, "params"), (2, "exp_avg"), (3, "exp_avg_sq")):
        assert all(same_bits(x, y) for x, y in zip(side[k], default[k])), label
    assert side_state == default_state == {"scale": SCALE, "growth_tracker": 1, "step": 1, "skipped": 0}
