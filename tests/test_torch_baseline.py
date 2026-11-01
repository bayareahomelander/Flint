"""scripts/gpt2_torch.py and scripts/torch_baseline.py: the model, the bookkeeping, and CLI smoke runs."""

import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

import flint_lib

SCRIPTS = flint_lib.ROOT / "scripts"
spec = importlib.util.spec_from_file_location("torch_baseline", SCRIPTS / "torch_baseline.py")
torch_baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch_baseline)
gpt2_torch = torch_baseline.gpt2_torch

# These tests run no Flint kernel. Under compute-sanitizer's unfiltered
# initcheck, PyTorch's own kernels would only slow them or false-alarm.
needs_gpu = pytest.mark.skipif(
    os.environ.get("FLINT_SANITIZER") == "1" or not torch.cuda.is_available(),
    reason="uses the GPU, which the sanitizer run and CPU-only machines skip",
)

TINY = gpt2_torch.TINY_CONFIG
MIB = 1024 * 1024


# ---------------------------------------------------------------- the model

def test_full_config_parameter_count_on_the_meta_device():
    model = gpt2_torch.build_model(gpt2_torch.GPT2_CONFIG, device="meta")
    assert gpt2_torch.count_parameters(model) == 124_475_904
    assert gpt2_torch.count_parameters(model, exclude_wpe=True) == 123_689_472


def test_flops_per_token_conventions():
    config = gpt2_torch.GPT2_CONFIG
    n = 123_689_472
    assert torch_baseline.flops_per_token(n, config.n_layer, config.n_embd, 1024, "palm") == 855_383_040
    assert torch_baseline.flops_per_token(n, config.n_layer, config.n_embd, 1024, "llmc") == 798_759_936
    with pytest.raises(ValueError, match="convention"):
        torch_baseline.flops_per_token(n, 12, 768, 1024, "other")


def test_weight_decay_groups_hold_exactly_the_matrices():
    model = gpt2_torch.build_model(gpt2_torch.GPT2_CONFIG, device="meta")
    decay, no_decay = gpt2_torch.parameter_groups(model)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    decayed = {names[id(p)] for p in decay["params"]}
    assert {"wte.weight", "wpe.weight"} <= decayed
    assert not any(n.endswith(".bias") or "ln_" in n for n in decayed)
    assert all(n.endswith(".bias") or "ln_" in n for n in (names[id(p)] for p in no_decay["params"]))
    # Every parameter lands in one group, and the tied wte only once.
    everything = [id(p) for p in decay["params"] + no_decay["params"]]
    assert len(everything) == len(set(everything)) == len(list(model.parameters()))
    assert sum(1 for p in decay["params"] if p is model.wte.weight) == 1
    # 2 embeddings + (c_attn, attn.c_proj, c_fc, mlp.c_proj) per block decay.
    assert len(decay["params"]) == 2 + 4 * 12


def test_init_follows_the_spec():
    config = gpt2_torch.GPTConfig(n_layer=12, n_head=2, n_embd=64, block_size=16)
    model = gpt2_torch.build_model(config, seed=7)
    residual = 0.02 / math.sqrt(24)
    for block in model.blocks:
        for projection in (block.attn.c_proj, block.mlp.c_proj):
            assert projection.weight.std().item() == pytest.approx(residual, rel=0.05)
        for linear in (block.attn.c_attn, block.mlp.c_fc):
            assert linear.weight.std().item() == pytest.approx(0.02, rel=0.05)
        assert block.attn.c_proj.weight.std().item() < 0.5 * block.attn.c_attn.weight.std().item()
    assert model.wpe.weight.std().item() == pytest.approx(0.02, rel=0.05)
    real_rows = model.wte.weight[: config.vocab_size]
    assert real_rows.std().item() == pytest.approx(0.02, rel=0.02)
    assert real_rows.mean().abs().item() < 1e-3
    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            assert not p.any(), name
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert (module.weight == 1).all() and (module.bias == 0).all()
    # Padded rows are exactly zero, and the rows before them are not.
    assert not model.wte.weight[config.vocab_size:].any()
    assert model.wte.weight[config.vocab_size - 1].any()
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_seed_fixes_the_weights():
    a = gpt2_torch.build_model(TINY, seed=3)
    b = gpt2_torch.build_model(TINY, seed=3)
    c = gpt2_torch.build_model(TINY, seed=4)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert not torch.equal(a.blocks[0].attn.c_attn.weight, c.blocks[0].attn.c_attn.weight)


def test_model_runs_on_the_cpu_and_the_loss_starts_near_log_vocab():
    model = gpt2_torch.build_model(TINY, seed=0)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, TINY.vocab_size, (2, 17), generator=generator)
    loss = model(tokens[:, :-1], tokens[:, 1:])
    assert loss.shape == () and loss.item() == pytest.approx(math.log(TINY.vocab_size), abs=0.1)
    logits = model.logits(tokens[:, :-1])
    assert logits.shape == (2, 16, TINY.vocab_size)


@needs_gpu
def test_padded_vocab_rows_get_no_gradient_and_stay_zero():
    model = gpt2_torch.build_model(TINY, seed=0, device="cuda")
    generator = torch.Generator("cuda").manual_seed(0)
    tokens = torch.randint(0, TINY.vocab_size, (4, 33), generator=generator, device="cuda")
    idx, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    # The loss sees only the real vocabulary's columns.
    assert model._logits(model.hidden(idx)).shape == (4 * 32, 50257)
    assert model.logits(idx).shape[-1] == 50257
    loss = model(idx, targets)
    loss.backward()
    grad = model.wte.weight.grad
    assert not grad[50257:].any()
    assert grad[:50257].any()
    before = model.wte.weight.detach().clone()
    gpt2_torch.configure_optimizer(model).step()
    after = model.wte.weight.detach()
    assert not after[50257:].any()
    assert not torch.equal(after[:50257], before[:50257])


@needs_gpu
def test_train_step_reports_skips_and_keeps_padded_rows_zero():
    model = gpt2_torch.build_model(TINY, seed=0, device="cuda")
    params = list(model.parameters())
    optimizer = gpt2_torch.configure_optimizer(model)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator("cuda").manual_seed(0)
    tokens = torch.randint(0, TINY.vocab_size, (2, 33), generator=generator, device="cuda")
    batches = [(tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())] * 2
    flags = [gpt2_torch.train_step(model, params, optimizer, scaler, batches) for _ in range(3)]
    assert all(isinstance(f, bool) for f in flags)
    assert all(torch.isfinite(p).all() for p in params)
    assert not model.wte.weight[50257:].any()
    assert all(p.grad is None for p in params)
    # A skip lowers the scale; a clean step doesn't.
    assert scaler.get_scale() == 65536.0 * 0.5 ** sum(flags)


# ------------------------------------------------------- FLOPs, MFU, days

def test_peak_and_mfu_arithmetic():
    assert torch_baseline.rated_sm_clock_mhz("NVIDIA GeForce RTX 2080 Ti") == 1545
    assert torch_baseline.rated_sm_clock_mhz("NVIDIA GeForce RTX 4050 Laptop GPU") is None
    peak = torch_baseline.peak_flops(68, 1545)
    assert peak == 512 * 68 * 1545e6 and peak / 1e12 == pytest.approx(53.79, abs=0.01)
    assert torch_baseline.mfu(1000, 855_383_040, peak) == pytest.approx(1000 * 855_383_040 / peak)
    assert torch_baseline.mfu(1000, 855_383_040, None) is None
    assert torch_baseline.days_for_tokens(1000) == pytest.approx(1e10 / 1000 / 86400)
    assert torch_baseline.days_for_tokens(1e5) == pytest.approx(1.1574, abs=1e-4)


# ------------------------------------------------------ window statistics

def sample(t, sm=1500, mem=7000, power=250_000, temp=60, reasons=0, used=1 << 30, new=False, ok=True):
    return {"t": t, "sm_mhz": sm, "mem_mhz": mem, "power_mw": power, "temp_c": temp, "reasons": reasons,
            "used_bytes": used, "processes_ok": ok, "new_process": new}


def flat_stats(sm_mhz=1500, seconds=1.0, **kwargs):
    samples = [sample(0.0, sm=sm_mhz, **kwargs), sample(seconds + 1, sm=sm_mhz, **kwargs)]
    return torch_baseline.window_stats(samples, 0.5, 0.5 + seconds)


def test_window_stats_weight_by_time():
    thermal, power_cap = 1 << 5, 1 << 2
    samples = [
        sample(0.0, sm=1000, power=None),
        sample(1.0, sm=2000, reasons=thermal, temp=70, used=5 << 30),
        sample(2.0, sm=3000, reasons=power_cap, new=True),
        sample(3.0, sm=9999),
    ]
    stats = torch_baseline.window_stats(samples, 0.5, 2.5)
    # Sample 0 holds for 0.5 s, sample 1 for 1 s, and sample 2 for 0.5 s.
    assert stats["seconds"] == pytest.approx(2.0) and stats["samples"] == 3
    assert stats["sm_mhz"] == pytest.approx((1000 * 0.5 + 2000 * 1 + 3000 * 0.5) / 2)
    assert stats["power_w"] == pytest.approx(250.0)  # the failed read is left out, not zeroed
    assert stats["failed_reads"] == 1
    assert stats["thermal"] == pytest.approx(0.5) and stats["hw_slowdown"] == 0
    assert stats["sw_power_cap"] == pytest.approx(0.25)
    assert stats["reason_shares"]["sw_thermal_slowdown"] == pytest.approx(0.5)
    assert stats["reason_shares"]["sw_power_cap"] == pytest.approx(0.25)
    assert stats["temp_c_max"] == 70 and stats["used_bytes_max"] == 5 << 30
    assert stats["new_process"] is True
    assert torch_baseline.window_stats(samples, 0.5, 0.9)["new_process"] is False


def test_window_stats_flag_hardware_slowdowns_and_ignore_other_reasons():
    stats = torch_baseline.window_stats(
        [sample(0.0, reasons=1 << 7), sample(1.0, reasons=(1 << 3) | (1 << 8)), sample(3.0)], 0.0, 2.0
    )
    assert stats["hw_slowdown"] == 1.0 and stats["thermal"] == 0.0
    stats = torch_baseline.window_stats([sample(0.0, reasons=(1 << 6)), sample(3.0)], 0.0, 1.0)
    assert stats["thermal"] == 1.0 and stats["hw_slowdown"] == 0.0
    stats = torch_baseline.window_stats([sample(0.0, reasons=(1 << 0) | (1 << 1) | (1 << 4)), sample(3.0)], 0.0, 1.0)
    assert stats["thermal"] == stats["hw_slowdown"] == stats["sw_power_cap"] == 0.0


def test_merge_stats_weights_windows_by_length():
    a = flat_stats(sm_mhz=1000, seconds=1.0)
    b = flat_stats(sm_mhz=1600, seconds=3.0, temp=80)
    merged = torch_baseline.merge_stats([a, b])
    assert merged["seconds"] == pytest.approx(4.0)
    assert merged["sm_mhz"] == pytest.approx((1000 * 1 + 1600 * 3) / 4)
    assert merged["temp_c_max"] == 80


# ------------------------------------------------------ steps and cases

def test_step_validity_rules():
    reserved, baseline = 10 << 30, 1 << 30
    stats = flat_stats(used=reserved + baseline + 200 * MIB)
    assert torch_baseline.invalid_reason(stats, reserved, baseline) is None
    stats = flat_stats(used=reserved + baseline + 300 * MIB)
    assert "300 MiB" in torch_baseline.invalid_reason(stats, reserved, baseline)
    # Exactly at the limit still counts.
    stats = flat_stats(used=reserved + baseline + 256 * MIB)
    assert torch_baseline.invalid_reason(stats, reserved, baseline) is None
    stats = flat_stats(new=True)
    assert "another process" in torch_baseline.invalid_reason(stats, reserved, baseline)
    # No memory reading at all can't be judged, and doesn't discard the step.
    stats = flat_stats(used=None)
    assert torch_baseline.invalid_reason(stats, reserved, baseline) is None
    # Flagged slowdowns are reported, not grounds to discard.
    stats = flat_stats(reasons=(1 << 5) | (1 << 7))
    assert stats["thermal"] == 1.0
    assert torch_baseline.invalid_reason(stats, reserved, baseline) is None


def step(seconds, skipped=False, new=False, tokens=1000, **kwargs):
    return torch_baseline.step_record(0, seconds, skipped, flat_stats(new=new, seconds=seconds, **kwargs), 1 << 30, 0, tokens)


def test_case_status_needs_half_the_steps_valid():
    good, bad = step(1.0), step(1.0, new=True)
    assert [good["valid"], bad["valid"]] == [True, False]
    assert torch_baseline.case_status([good, good, bad, bad]) == "ok"
    assert torch_baseline.case_status([good, bad, bad, bad]) == "too few valid steps"
    assert torch_baseline.case_status([good, bad]) == "ok"
    assert torch_baseline.case_status([good, bad, bad]) == "too few valid steps"
    assert torch_baseline.case_status([bad]) == "too few valid steps"
    assert torch_baseline.case_status([]) == "too few valid steps"


def test_summary_statistics_use_valid_steps_only():
    flops = {"palm": 855_383_040, "llmc": 798_759_936}
    steps = [step(0.5, sm_mhz=1400), step(1.0, skipped=True, sm_mhz=1600), step(2.0, sm_mhz=1500),
             step(0.1, new=True, sm_mhz=900)]
    summary = torch_baseline.summarize_steps(steps, 1000, flops, 68, 1545)
    assert (summary["timed_steps"], summary["valid_steps"], summary["skipped_steps"]) == (4, 3, 1)
    assert summary["tokens_per_second"] == {"median": 1000.0, "min": 500.0, "max": 2000.0}
    # Time-weighted over the valid steps: 0.5 s at 1400, 1 s at 1600, 2 s at 1500.
    assert summary["stats"]["sm_mhz"] == pytest.approx((1400 * 0.5 + 1600 * 1 + 1500 * 2) / 3.5)
    rated, adjusted = summary["mfu"]["rated"], summary["mfu"]["clock_adjusted"]
    peak = 512 * 68 * 1545e6
    assert rated["palm"] == pytest.approx(1000 * flops["palm"] / peak)
    assert rated["llmc"] == pytest.approx(1000 * flops["llmc"] / peak)
    clock_peak = 512 * 68 * summary["stats"]["sm_mhz"] * 1e6
    assert adjusted["palm"] == pytest.approx(1000 * flops["palm"] / clock_peak)
    assert summary["mfu"]["rated_peak_tflops"] == pytest.approx(53.79, abs=0.01)
    assert summary["days_for_10b_tokens"] == pytest.approx(1e10 / 1000 / 86400)


def test_summary_without_a_rated_clock_or_valid_steps():
    flops = {"palm": 1e9, "llmc": 1e9}
    summary = torch_baseline.summarize_steps([step(1.0)], 1000, flops, 20, None)
    assert summary["mfu"]["rated"] is None and summary["mfu"]["rated_peak_tflops"] is None
    assert summary["mfu"]["clock_adjusted"]["palm"] == pytest.approx(1000 * 1e9 / (512 * 20 * 1500e6))
    empty = torch_baseline.summarize_steps([step(1.0, new=True)], 1000, flops, 20, None)
    assert empty["valid_steps"] == 0 and empty["tokens_per_second"] is None and empty["mfu"] is None


def test_table_renders_every_kind_of_case():
    flops = {"palm": 855_383_040, "llmc": 798_759_936}
    ok = {"mode": "eager", "micro_batch": 4, "status": "ok", "steps": [],
          "summary": {**torch_baseline.summarize_steps([step(1.0)], 1000, flops, 68, 1545),
                      "peak_memory_reserved_mib": 4096.0}}
    oom = {"mode": "eager", "micro_batch": 64, "status": "oom", "steps": [], "summary": None}
    table = torch_baseline.render_table([ok, oom])
    lines = table.splitlines()
    assert len(lines) == 3 and "1,000 [1,000-1,000]" in lines[1] and "4.0" in lines[1]
    assert "oom" in lines[2]


# --------------------------------------------------------------- the sweep

def test_oom_stops_the_larger_micro_batches_of_its_mode():
    calls, cleaned, recorded = [], [], []

    def run_case(mode, micro_batch):
        calls.append((mode, micro_batch))
        if mode == "compile" and micro_batch >= 16:
            raise torch.OutOfMemoryError("fake")
        return {"mode": mode, "micro_batch": micro_batch, "status": "ok", "steps": [], "summary": None}

    cases, skipped = torch_baseline.run_sweep(
        ["eager", "compile"], [16, 4, 32, 8], run_case, cleaned.append, recorded.append
    )
    assert calls == [("eager", 4), ("eager", 8), ("eager", 16), ("eager", 32),
                     ("compile", 4), ("compile", 8), ("compile", 16)]
    assert [(c["mode"], c["micro_batch"], c["status"]) for c in cases[-3:]] == [
        ("compile", 4, "ok"), ("compile", 8, "ok"), ("compile", 16, "oom")]
    assert [(s["mode"], s["micro_batch"]) for s in skipped] == [("compile", 32)]
    assert cleaned == ["eager"] * 4 + ["compile"] * 3
    assert recorded == cases


def test_the_first_micro_batch_running_out_of_memory_skips_the_rest():
    def run_case(mode, micro_batch):
        raise torch.OutOfMemoryError("fake")

    cases, skipped = torch_baseline.run_sweep(["eager"], [4, 8], run_case, lambda mode: None)
    assert [(c["micro_batch"], c["status"]) for c in cases] == [(4, "oom")]
    assert [s["micro_batch"] for s in skipped] == [8]


def test_other_errors_are_not_swallowed():
    def run_case(mode, micro_batch):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        torch_baseline.run_sweep(["eager"], [4], run_case, lambda mode: None)


# -------------------------------------------------- the monitor, with a fake NVML

class FakeNvml:
    def __init__(self, pids=(), used=300 * MIB, utilization=0, detection=True):
        self.pids = list(pids)
        self.used = used
        self.utilization = utilization
        self.detection = detection
        self.closed = False

    def processes(self):
        return list(self.pids) if self.detection else None

    def used_bytes(self):
        return self.used

    def utilization_percent(self):
        return self.utilization

    def clock_mhz(self, kind):
        return 1234 if kind == "sm" else 5678

    def power_mw(self):
        return 100_000

    def temperature_c(self):
        return 55

    def clock_event_reasons(self):
        return 0

    def device_name(self):
        return "Fake GPU"

    def close(self):
        self.closed = True


def fast_monitor(nvml):
    return torch_baseline.Monitor(nvml, interval=0.005, utilization_reads=1, utilization_pause=0)


def test_monitor_sees_a_process_that_appears_during_a_window():
    nvml = FakeNvml(pids=[5])
    monitor = fast_monitor(nvml)
    assert monitor.processes_at_start == [5] and monitor.process_detection
    monitor.start()
    try:
        time.sleep(0.05)
        t0 = time.perf_counter()
        time.sleep(0.05)
        quiet = monitor.window(t0, time.perf_counter())  # pid 5 was there at start
        assert quiet["new_process"] is False and quiet["sm_mhz"] == 1234 and quiet["samples"] > 2
        nvml.pids.append(99)
        t1 = time.perf_counter()
        time.sleep(0.05)
        busy = monitor.window(t1, time.perf_counter())
        assert busy["new_process"] is True
        assert busy["used_bytes_max"] == 300 * MIB
    finally:
        monitor.stop()


def test_busy_at_start_rule():
    idle = torch_baseline.busy_reasons(0, 100 * MIB, 0, True)
    assert idle == []
    assert torch_baseline.busy_reasons(0, 256 * MIB, 0, True) == []  # exactly the limit is still idle
    reasons = torch_baseline.busy_reasons(2, 300 * MIB, 7, False)
    assert reasons == [
        "2 other processes",
        "300 MiB in use before our context",
        "7% utilization before we started",
        "NVML can't list processes, so others would go unseen",
    ]


def test_nvml_struct_layouts():
    assert torch_baseline.ctypes.sizeof(torch_baseline._MemoryV2) == 40
    assert torch_baseline.NVML_MEMORY_V2 == 40 | (2 << 24)
    assert torch_baseline.ctypes.sizeof(torch_baseline._ProcessInfo) == 24
    assert torch_baseline.ctypes.sizeof(torch_baseline._ProcessInfoV1) == 16


# ------------------------------------------------------------ main's rules

TINY_ARGS = ["--config", "tiny", "--modes", "eager", "--micro-batches", "2", "--tokens-per-step", "256",
             "--steps", "2", "--warmup-steps", "1"]


def test_busy_gpu_is_refused_before_any_cuda_work(tmp_path, monkeypatch, capsys):
    def no_cuda(*args, **kwargs):
        raise AssertionError("CUDA was touched")

    monkeypatch.setattr(torch_baseline, "gpu_info", no_cuda)
    monkeypatch.setattr(torch.cuda, "init", no_cuda)
    out = tmp_path / "log.json"
    nvml = FakeNvml(pids=[4321])
    code = torch_baseline.main(["--out", str(out), *TINY_ARGS], nvml_factory=lambda: nvml)
    assert code == 2
    assert "the GPU is busy: 1 other processes" in capsys.readouterr().out
    assert nvml.closed and not out.exists()


def test_allow_busy_goes_on_to_the_gpu(tmp_path, monkeypatch, capsys):
    class Reached(Exception):
        pass

    def reached(nvml):
        raise Reached

    monkeypatch.setattr(torch_baseline, "gpu_info", reached)
    with pytest.raises(Reached):
        torch_baseline.main(["--out", str(tmp_path / "log.json"), "--allow-busy", *TINY_ARGS],
                            nvml_factory=lambda: FakeNvml(pids=[4321]))
    assert "running anyway" in capsys.readouterr().err


def test_missing_nvml_is_an_error_not_a_crash(tmp_path, capsys):
    def factory():
        raise torch_baseline.NvmlError("can't load NVML")

    assert torch_baseline.main(["--out", str(tmp_path / "log.json"), *TINY_ARGS], nvml_factory=factory) == 1
    assert "can't load NVML" in capsys.readouterr().err


def test_arguments_are_validated(tmp_path):
    out = ["--out", str(tmp_path / "log.json")]
    with pytest.raises(SystemExit):
        torch_baseline.parse_args(out + ["--tokens-per-step", "100", "--modes", "eager"])
    with pytest.raises(SystemExit):
        torch_baseline.parse_args(out + ["--modes", "graph"])
    with pytest.raises(SystemExit):
        torch_baseline.parse_args(out + ["--seq-len", "2048", "--modes", "eager"])
    args = torch_baseline.parse_args(out + ["--micro-batches", "8,4,8", "--modes", "eager"])
    assert args.micro_batches == [4, 8] and args.seq_len == 1024 and args.tokens_per_step == 2**19
    assert torch_baseline.parse_args(out + ["--config", "tiny", "--modes", "eager"]).seq_len == 64


def test_new_files_have_unix_line_endings():
    for path in (SCRIPTS / "gpt2_torch.py", SCRIPTS / "torch_baseline.py", Path(__file__)):
        assert b"\r" not in path.read_bytes(), path.name


# --------------------------------------------------------- CLI smoke runs

def run_cli(tmp_path, modes, extra_env=None):
    out = tmp_path / "baseline.json"
    command = [sys.executable, str(SCRIPTS / "torch_baseline.py"), "--out", str(out), "--allow-busy",
               *TINY_ARGS]
    command[command.index("--modes") + 1] = modes
    env = dict(os.environ, **(extra_env or {}))
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=600)
    return result, out


def check_log(result, out, tmp_path, mode):
    assert result.returncode == 0, result.stdout + result.stderr
    text = out.read_text(encoding="utf-8")
    assert "\r" not in text
    for forbidden in (str(tmp_path), str(flint_lib.ROOT), str(Path.home()), Path.home().name, socket.gethostname()):
        if len(forbidden) >= 4:
            assert forbidden.lower() not in (text + result.stdout).lower(), forbidden
    log = json.loads(text)
    assert log["schema"] == "flint-torch-baseline/1" and log["complete"] is True
    assert set(log["git"]) == {"sha", "dirty"}
    assert log["versions"]["torch"] == torch.__version__
    assert log["device"]["name"] == torch.cuda.get_device_name(0)
    assert log["settings"]["allow_fp16_reduced_precision_reduction"] is False
    assert "out" not in log["settings"] and log["settings"]["allow_busy"] is True
    assert log["model"]["parameters"] > 0 and set(log["model"]["flops_per_token"]) == {"palm", "llmc"}
    assert "other_processes" in log["gpu_at_start"]
    (case,) = log["cases"]
    assert (case["mode"], case["micro_batch"], case["accum_steps"], case["status"]) == (mode, 2, 2, "ok")
    assert len(case["steps"]) == 2 and all(s["seconds"] > 0 for s in case["steps"])
    assert case["summary"]["tokens_per_second"]["median"] > 0
    assert case["summary"]["peak_memory_reserved_mib"] > 0
    assert "tokens/s" in result.stdout and mode in result.stdout
    return log


@needs_gpu
def test_eager_smoke_run(tmp_path):
    result, out = run_cli(tmp_path, "eager")
    check_log(result, out, tmp_path, "eager")


@needs_gpu
@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="torch.compile needs Triton")
def test_compile_smoke_run(tmp_path):
    # The server's /tmp is a RAM disk, so the compilers' caches go under tmp_path.
    env = {"TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"), "TRITON_CACHE_DIR": str(tmp_path / "triton")}
    result, out = run_cli(tmp_path, "compile", env)
    check_log(result, out, tmp_path, "compile")
