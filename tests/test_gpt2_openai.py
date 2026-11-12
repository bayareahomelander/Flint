"""OpenAI's GPT-2 124M weights through scripts/gpt2_torch.py and flint_gpt2: the check that the
model we test against is GPT-2 and that Flint computes it. PLAN.md's decision "OpenAI's weights
in D25": the gate in test_gpt2.py only shows that Flint matches gpt2_torch, a file from our own
reading of the architecture, so nothing there shows that it is GPT-2.

The checkpoint is openai-community/gpt2 at revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
(data/gpt2-124M/model.safetensors, 548,105,171 bytes, sha256 checked once per module), loaded
into three FP32 models under torch.no_grad():
  - transformers' GPT2LMHeadModel(GPT2Config()), built with no network access (GPT2Config()'s
    defaults are GPT-2 124M, which the first test checks, and so is eval() mode's lack of
    dropout), its tensors loaded by name;
  - gpt2_torch with the same weights: OpenAI's Conv1D weights are [in, out] and so are
    transposed, and the padded vocabulary rows stay zero;
  - Flint, whose master weights are the same FP32 values (the FP16 copies are rounded by
    sync_weights, as autocast rounds them).
The tokens are windows of real FineWeb-Edu text, 3 each at (1, 1024), (2, 256) and (8, 64).

  - gpt2_torch is GPT-2: its logits and mean loss against transformers' in FP32. The two are
    different implementations of the same model, so they differ by FP32 noise (measured: a
    relative L2 of at most 1.1e-6, a max-abs of at most 9e-6 of max|logit| and a loss gap of
    at most 1.2e-6). The bounds are 10x that: a relative L2 of 1e-5, a max-abs of 1e-4 of
    max|logit| and a loss gap of 1e-5, which are 100x below what FP16 does (about 1e-3) and
    so would see a wrong GELU, scale, LayerNorm epsilon or mapping.
  - Flint is GPT-2: with OpenAI's exact FP32 master weights, the gate of test_gpt2.py, with
    transformers' logits as the reference and gpt2_torch under autocast as the peer; and, as
    the done rule has it, with every master weight rounded to FP16-representable, against
    gpt2_torch in FP32.
  - The loss is real (between 2.5 and 5 on real text; random weights give about 11), and
    Flint's and autocast's top-1 predictions agree with the FP32 model's on at least 95% of
    the tokens. The padded dlogits are +0 and a second pass gives the same bits.

Skipped when the weights are absent (the tests needing the shard skip when it is), and always
under the sanitizer. One shape lives on the GPU at a time: an FP32 reference at 1,024 tokens
fits the laptop only at batch 1."""

import copy
import gc

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import gate
from inputs import SANITIZER
from test_gpt2 import (  # also puts scripts/ on sys.path
    Batch,
    Gpt2,
    G,
    TorchReference,
    all_equal,
    assert_same_outputs,
    check_measured,
    data_file,
    free_memory,
    gate_scale,
    gpt2_config,
    int_view,
    measure,
    read_outputs,
    real_tokens,
    sha256_of,
    state_of,
)

CHECKPOINT = "gpt2-124M/model.safetensors"
CHECKPOINT_BYTES = 548_105_171
CHECKPOINT_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"

GPT2 = G.GPT2_CONFIG
VOCAB = GPT2.vocab_size
SHAPES = [(1, 1024), (2, 256), (8, 64)]
# gpt2_torch against transformers (see the module docstring).
TORCH_VS_HF_REL_L2 = 1e-5
TORCH_VS_HF_MAX_ABS_OF_MAX = 1e-4
TORCH_VS_HF_LOSS = 1e-5
LOSS_RANGE = (2.5, 5.0)
MIN_TOP1 = 0.95


@pytest.fixture(autouse=True)
def free_gpu_memory():
    """The laptop's GPU is 6 GiB and shared, so each test hands back what it cached."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def not_under_the_sanitizer():
    if SANITIZER:
        pytest.skip("these tests run FP32 GPT-2 124M, which the sanitizer would take hours over")


def checkpoint_names():
    names = ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]
    for i in range(GPT2.n_layer):
        for layer in ("ln_1", "ln_2", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            names += [f"h.{i}.{layer}.weight", f"h.{i}.{layer}.bias"]
    return names


@pytest.fixture(scope="module")
def checkpoint():
    """OpenAI's tensors on the CPU, by name (without transformer.), after checking the file."""
    path = data_file(CHECKPOINT)
    if path is None:
        pytest.skip(f"{CHECKPOINT} not found in data/ or ~/flint-data/")
    assert path.stat().st_size == CHECKPOINT_BYTES, f"{path} is not the pinned checkpoint's size"
    assert sha256_of(path) == CHECKPOINT_SHA256, f"{path} is not the pinned checkpoint (sha256)"
    state = {k.removeprefix("transformer."): v for k, v in load_file(str(path)).items()}
    names = checkpoint_names()
    assert set(names) <= set(state), f"missing: {sorted(set(names) - set(state))}"
    # Besides the weights the file holds only attention's causal-mask buffers.
    extra = set(state) - set(names)
    assert all(k.endswith(("attn.bias", "attn.masked_bias")) for k in extra), sorted(extra)
    assert all(v.dtype == torch.float32 for k, v in state.items() if k in names)
    return state


@pytest.fixture(scope="module")
def hf_model(checkpoint):
    """transformers' GPT-2 with OpenAI's weights, FP32, in eval mode, on the GPU, built
    with no network access."""
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(GPT2Config())
    result = model.load_state_dict({"transformer." + k: v for k, v in checkpoint.items()}, strict=False)
    # lm_head is tied to wte, so the file has no key of its own for it; the attention
    # bias buffers are not part of the model's state_dict.
    assert result.missing_keys == ["lm_head.weight"], result.missing_keys
    assert result.unexpected_keys and all(k.endswith(("attn.bias", "attn.masked_bias")) for k in result.unexpected_keys), result.unexpected_keys
    assert model.lm_head.weight is model.transformer.wte.weight
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, checkpoint[name.removeprefix("transformer.")]), name
    model.eval().cuda()
    yield model
    del model


def load_openai(checkpoint):
    """gpt2_torch's model with OpenAI's exact FP32 weights: Conv1D's are [in, out], so
    transposed, and the padded vocabulary rows stay zero."""
    model = G.build_model(GPT2, seed=0, device="cuda")

    def put(parameter, tensor, transpose=False):
        tensor = tensor.to("cuda")
        parameter.copy_(tensor.T if transpose else tensor)

    with torch.no_grad():
        model.wte.weight.zero_()
        model.wte.weight[:VOCAB].copy_(checkpoint["wte.weight"].to("cuda"))
        put(model.wpe.weight, checkpoint["wpe.weight"])
        put(model.ln_f.weight, checkpoint["ln_f.weight"])
        put(model.ln_f.bias, checkpoint["ln_f.bias"])
        for i, block in enumerate(model.blocks):
            for name, layer in (("ln_1", block.ln_1), ("ln_2", block.ln_2)):
                put(layer.weight, checkpoint[f"h.{i}.{name}.weight"])
                put(layer.bias, checkpoint[f"h.{i}.{name}.bias"])
            for name, layer in (("attn.c_attn", block.attn.c_attn), ("attn.c_proj", block.attn.c_proj),
                                ("mlp.c_fc", block.mlp.c_fc), ("mlp.c_proj", block.mlp.c_proj)):  # fmt: skip
                put(layer.weight, checkpoint[f"h.{i}.{name}.weight"], transpose=True)
                put(layer.bias, checkpoint[f"h.{i}.{name}.bias"])
    return model


@pytest.fixture(scope="module")
def openai_model(checkpoint):
    model = load_openai(checkpoint)
    yield model
    del model


def rounded_to_fp16(model):
    """A copy of a model with every master weight rounded to an FP16-representable value."""
    rounded = copy.deepcopy(model)
    with torch.no_grad():
        for p in rounded.parameters():
            p.copy_(p.half().float())
    return rounded


def hf_logits(hf, idx):
    with torch.no_grad(), gate.peer_math():
        return hf(input_ids=idx, use_cache=False).logits.reshape(-1, VOCAB)


def window_batch(sequences, length, window):
    ids = real_tokens(sequences * length + 1, window)
    assert int(ids.max()) < VOCAB
    return Batch(ids, sequences, length)


# -- The models --------------------------------------------------------------------------------


def test_transformers_builds_gpt2_124m_with_no_dropout_in_eval(hf_model):
    from transformers.activations import ACT2FN

    c = hf_model.config
    assert (c.n_layer, c.n_head, c.n_embd, c.n_positions, c.vocab_size) == (12, 12, 768, 1024, 50257)
    assert c.layer_norm_epsilon == 1e-5 and c.n_inner is None
    assert c.activation_function == "gelu_new"
    assert c.scale_attn_weights and not c.reorder_and_upcast_attn and not c.scale_attn_by_inverse_layer_idx
    assert c.tie_word_embeddings
    assert (c.attn_pdrop, c.embd_pdrop, c.resid_pdrop) == (0.1, 0.1, 0.1), "the defaults have dropout, so eval() must switch it off"
    # gpt2_torch's choices are the same: eps 1e-5, 12 x 64 heads, GELU's tanh form.
    assert all(m.eps == 1e-5 for m in hf_model.modules() if isinstance(m, torch.nn.LayerNorm))
    x = torch.randn(100_000, device="cuda") * 3
    assert (ACT2FN["gelu_new"](x) - F.gelu(x, approximate="tanh")).abs().max().item() < 1e-6
    idx = torch.randint(0, VOCAB, (1, 64), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    a, b = hf_logits(hf_model, idx), hf_logits(hf_model, idx)
    assert torch.equal(a, b) and not hf_model.training
    assert not any(m.training for m in hf_model.modules())
    try:
        hf_model.train()
        # In train mode dropout acts, so two passes differ: the check can see it.
        assert any_differs(hf_logits(hf_model, idx), hf_logits(hf_model, idx))
    finally:
        hf_model.eval()


def any_differs(a, b):
    return bool((a != b).any())


def test_openai_weights_are_mapped_exactly_into_gpt2_torch(checkpoint, openai_model):
    model = openai_model
    assert all_equal(model.wte.weight[VOCAB:], 0), "the padded rows are zero"
    assert torch.equal(model.wte.weight[:VOCAB].cpu(), checkpoint["wte.weight"])
    assert torch.equal(model.wpe.weight.cpu(), checkpoint["wpe.weight"])
    for i, block in enumerate(model.blocks):
        for name, layer in (("attn.c_attn", block.attn.c_attn), ("attn.c_proj", block.attn.c_proj),
                            ("mlp.c_fc", block.mlp.c_fc), ("mlp.c_proj", block.mlp.c_proj)):  # fmt: skip
            assert torch.equal(layer.weight.cpu(), checkpoint[f"h.{i}.{name}.weight"].T), (i, name)
            assert torch.equal(layer.bias.cpu(), checkpoint[f"h.{i}.{name}.bias"]), (i, name)
        assert torch.equal(block.ln_1.bias.cpu(), checkpoint[f"h.{i}.ln_1.bias"])
        assert torch.equal(block.ln_2.weight.cpu(), checkpoint[f"h.{i}.ln_2.weight"])
    assert sum(p.numel() for p in model.parameters()) == 124_475_904


# -- gpt2_torch is GPT-2 ---------------------------------------------------------------------------


@pytest.mark.parametrize("sequences, length", SHAPES, ids=lambda v: str(v))
def test_gpt2_torch_is_gpt2(hf_model, openai_model, sequences, length):
    failures = []
    for window in gate.SEEDS:
        batch = window_batch(sequences, length, window)
        with torch.no_grad(), gate.peer_math():
            reference = hf_logits(hf_model, batch.idx)
            ours = openai_model.logits(batch.idx).reshape(-1, VOCAB)
            hf_loss = F.cross_entropy(reference, batch.targets).item()
            our_loss = openai_model(batch.idx, batch.targets).item()
        e = gate.errors(ours, reference)
        gap = abs(our_loss - hf_loss)
        print(f"GPT2_TORCH vs transformers {sequences}x{length} window {window}: logits rel_l2 {e.rel_l2:.3e} "
              f"max_abs {e.max_abs:.3e} (max|logit| {e.ref_max:.1f}, ratio {e.max_abs / e.ref_max:.2e}); loss "
              f"transformers {hf_loss:.7f} gpt2_torch {our_loss:.7f} gap {gap:.2e}")  # fmt: skip
        if not e.rel_l2 <= TORCH_VS_HF_REL_L2:
            failures.append(f"window {window}: rel_l2 {e.rel_l2:.3e} > {TORCH_VS_HF_REL_L2}")
        if not e.max_abs <= TORCH_VS_HF_MAX_ABS_OF_MAX * e.ref_max:
            failures.append(f"window {window}: max_abs {e.max_abs:.3e} > {TORCH_VS_HF_MAX_ABS_OF_MAX} * {e.ref_max:.1f}")
        if not gap <= TORCH_VS_HF_LOSS:
            failures.append(f"window {window}: loss gap {gap:.2e} > {TORCH_VS_HF_LOSS}")
        del reference, ours, batch
        free_memory()
    assert not failures, "\n".join(failures)


# -- Flint is GPT-2 ----------------------------------------------------------------------------------

CASES = {}


def describe(label, results):
    for window, r in enumerate(results):
        print(
            f"OPENAI {label} window {window}: loss flint {r.flint_loss:.6f} fp32 {r.ref_loss:.6f} autocast {r.peer_loss:.6f} | "
            f"logits rel_l2 flint {r.logits[0].rel_l2:.3e} autocast {r.logits[1].rel_l2:.3e}, max_abs flint "
            f"{r.logits[0].max_abs:.3e} autocast {r.logits[1].max_abs:.3e} | loss err flint {r.loss[0].max_abs:.2e} "
            f"autocast {r.loss[1].max_abs:.2e} | dlogits rel_l2 flint {r.dlogits[0].rel_l2:.3e} autocast "
            f"{r.dlogits[1].rel_l2:.3e}, max_abs flint {r.dlogits[0].max_abs:.3e} autocast {r.dlogits[1].max_abs:.3e} | "
            f"top-1 vs fp32 flint {r.top1_flint:.4f} autocast {r.top1_peer:.4f}"
        )  # fmt: skip


def measured_case(flint, hf, openai, sequences, length, weights):
    """Per window, Flint's forward pass on OpenAI's weights against the FP32 reference
    and the autocast peer. weights="exact": OpenAI's FP32 values, with transformers as the
    reference and gpt2_torch under autocast as the peer. weights="fp16": every master weight
    rounded to FP16-representable, with gpt2_torch in FP32 as the reference. Memoized, as
    several tests read one run."""
    key = (sequences, length, weights)
    if key in CASES:
        return CASES[key]
    rows, scale = sequences * length, gate_scale(sequences * length)
    config = gpt2_config(GPT2, sequences, length)
    model = openai if weights == "exact" else rounded_to_fp16(openai)
    results = []
    for window in gate.SEEDS:
        batch = window_batch(sequences, length, window)
        with torch.no_grad():
            if weights == "exact":
                reference_logits = hf_logits(hf, batch.idx)
                torch_ref = TorchReference(model, batch.idx, batch.targets, with_reference=False)

                def reference(lo, hi):
                    return reference_logits[lo:hi]
            else:
                torch_ref = TorchReference(model, batch.idx, batch.targets)
                reference = torch_ref.reference
            with Gpt2(flint, config) as m:
                m.load(state_of(model))
                m.forward(batch.tokens, batch.target_tokens, scale)
                results.append(measure(m, config, reference, torch_ref.peer, batch.targets, scale, torch_ref.peer_loss))
        del batch, torch_ref
        reference_logits = reference = None
        free_memory()
    if model is not openai:
        del model
    free_memory()
    describe(f"{weights} weights {sequences}x{length}", results)
    CASES[key] = results
    return results


@pytest.mark.parametrize("sequences, length", SHAPES, ids=lambda v: str(v))
def test_flint_is_gpt2_with_openais_exact_weights(flint, hf_model, openai_model, sequences, length):
    # Reference: transformers' FP32 logits. Peer: gpt2_torch, the same weights, under autocast.
    results = measured_case(flint, hf_model, openai_model, sequences, length, "exact")
    check_measured(f"OpenAI exact FP32 weights {sequences}x{length}, real tokens", results)


@pytest.mark.parametrize("sequences, length", SHAPES, ids=lambda v: str(v))
def test_flint_passes_the_gate_with_fp16_representable_weights(flint, hf_model, openai_model, sequences, length):
    # The done rule's form: identical FP16-representable weights, gpt2_torch in FP32 as the
    # reference and autocast as the peer.
    results = measured_case(flint, hf_model, openai_model, sequences, length, "fp16")
    check_measured(f"OpenAI weights rounded to FP16 {sequences}x{length}, real tokens", results)


@pytest.mark.parametrize("sequences, length", SHAPES, ids=lambda v: str(v))
def test_the_loss_on_real_text_is_real(flint, hf_model, openai_model, sequences, length):
    results = measured_case(flint, hf_model, openai_model, sequences, length, "exact")
    low, high = LOSS_RANGE
    for window, r in enumerate(results):
        for who, loss in (("flint", r.flint_loss), ("fp32", r.ref_loss), ("autocast", r.peer_loss)):
            assert low < loss < high, f"{sequences}x{length} window {window}: {who} loss {loss} is not in ({low}, {high})"


@pytest.mark.parametrize("sequences, length", SHAPES, ids=lambda v: str(v))
def test_top_1_predictions_agree_with_the_fp32_model(flint, hf_model, openai_model, sequences, length):
    results = measured_case(flint, hf_model, openai_model, sequences, length, "exact")
    for window, r in enumerate(results):
        print(f"TOP1 {sequences}x{length} window {window}: flint {r.top1_flint:.4f} autocast {r.top1_peer:.4f}")
    worst_flint = min(r.top1_flint for r in results)
    worst_peer = min(r.top1_peer for r in results)
    assert worst_flint >= MIN_TOP1, f"flint agrees with the FP32 model on only {worst_flint:.4f}"
    assert worst_peer >= MIN_TOP1, f"autocast agrees with the FP32 model on only {worst_peer:.4f}"


def test_padded_dlogits_are_zero_and_a_second_pass_gives_the_same_bits(flint, openai_model):
    sequences, length = 2, 256
    config = gpt2_config(GPT2, sequences, length)
    batch = window_batch(sequences, length, 0)
    scale = gate_scale(batch.rows)
    with Gpt2(flint, config) as m:
        m.load(state_of(openai_model))
        m.forward(batch.tokens, batch.target_tokens, scale)
        first = read_outputs(m, config)
        m.forward(batch.tokens, batch.target_tokens, scale)
        second = read_outputs(m, config)
    assert all_equal(int_view(first.dlogits)[:, VOCAB:], 0), "dlogits past vocab are not +0"
    assert all_equal(int_view(first.logits)[:, VOCAB:], 0), "logits past vocab are not +0"
    assert torch.isfinite(first.dlogits.float()).all() and torch.isfinite(first.logits[:, :VOCAB].float()).all()
    assert_same_outputs(first, second, "second pass on real weights")
