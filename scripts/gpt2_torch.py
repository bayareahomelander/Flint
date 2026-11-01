"""GPT-2 small in PyTorch: the model and training step that torch_baseline.py times.

Usage: imported by scripts/torch_baseline.py and tests/test_torch_baseline.py.

The model is the project's M4 configuration: pre-LayerNorm blocks, GELU (tanh),
a tied output layer, no dropout, and a vocabulary padded to 50,304 rows so the
output GEMM stays aligned. It is also meant as the FP32 reference and the
FP16-autocast peer of Flint's own training step, so its init, weight-decay
groups and loss follow the spec exactly rather than whatever is fastest.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    # A multiple of 128, so the tied output GEMM's N is aligned.
    padded_vocab_size: int = 50304


GPT2_CONFIG = GPTConfig()
# Small enough for tests to run in a blink, with GPT-2's real vocabulary sizes
# so the padded-row behavior is the real one.
TINY_CONFIG = GPTConfig(n_layer=2, n_head=2, n_embd=128, block_size=64)
CONFIGS = {"gpt2": GPT2_CONFIG, "tiny": TINY_CONFIG}

INIT_STD = 0.02
LEARNING_RATE = 6e-4
BETAS = (0.9, 0.95)
EPSILON = 1e-8
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=2)
        # Heads of width / n_head (64 for GPT-2); the default scale is 1/sqrt(64).
        q, k, v = (
            t.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2) for t in (q, k, v)
        )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)

    def init_weights(self, generator):
        """The spec's init, drawn from a (CPU) generator so a seed fixes it."""
        config = self.config
        # The residual projections are scaled by 1/sqrt(2 n_layer), as in GPT-2,
        # so the residual stream's variance doesn't grow with depth.
        residual_std = INIT_STD / math.sqrt(2 * config.n_layer)
        residual = {id(b.attn.c_proj) for b in self.blocks} | {id(b.mlp.c_proj) for b in self.blocks}
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    std = residual_std if id(module) in residual else INIT_STD
                    nn.init.normal_(module.weight, 0.0, std, generator=generator)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, 0.0, INIT_STD, generator=generator)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            # Padded rows never appear in a target, and stay exactly zero.
            self.wte.weight[config.vocab_size:].zero_()

    def hidden(self, idx):
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def _logits(self, x):
        # The GEMM runs over all padded rows to stay aligned; slicing the
        # columns off before the loss keeps the padded rows out of the
        # log-sum-exp, and gives them exactly zero gradient.
        flat = x.reshape(-1, x.shape[-1])
        return F.linear(flat, self.wte.weight)[:, : self.config.vocab_size]

    def logits(self, idx):
        """The logits the loss sees: [batch, length, vocab_size]."""
        return self._logits(self.hidden(idx)).view(*idx.shape, self.config.vocab_size)

    def forward(self, idx, targets):
        logits = self._logits(self.hidden(idx))
        return F.cross_entropy(logits, targets.reshape(-1))


def build_model(config, seed=1234, device="cpu"):
    """A fresh model on `device`, with FP32 parameters initialised from `seed`.
    On the meta device it only has shapes."""
    with torch.device("meta"):
        model = GPT(config)
    if device == "meta":
        return model
    # Drawn on the CPU whatever the target, so a seed gives the same weights
    # on every machine.
    model.to_empty(device="cpu")
    generator = torch.Generator("cpu").manual_seed(seed)
    model.init_weights(generator)
    return model.to(device)


def count_parameters(model, exclude_wpe=False):
    """Parameters counted once each (the tied wte included), minus wpe if asked."""
    total = sum(p.numel() for p in model.parameters())
    return total - model.wpe.weight.numel() if exclude_wpe else total


def parameter_groups(model, weight_decay=WEIGHT_DECAY):
    """Weight decay on every parameter of dimension 2 or more (wte, wpe and the
    Linear weights); none on biases and LayerNorm parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    return [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]


def configure_optimizer(model, lr=LEARNING_RATE):
    # The learning rate is constant: a schedule changes no kernel's work, so it
    # can't change throughput.
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(parameter_groups(model), lr=lr, betas=BETAS, eps=EPSILON, fused=fused)


def train_step(model, params, optimizer, scaler, batches):
    """One optimizer step over `batches`, a list of (idx, targets) micro-batches
    accumulated into a single gradient. Returns True if the GradScaler skipped
    the step (it saw a non-finite gradient and lowered its scale)."""
    scale_before = scaler.get_scale()
    for idx, targets in batches:
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(idx, targets) / len(batches)
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return scaler.get_scale() < scale_before
