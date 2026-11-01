"""Measures PyTorch's GPT-2 124M training throughput (tokens/s), in eager and torch.compile modes.

Usage: python scripts/torch_baseline.py --out LOG.json [--config gpt2|tiny]
       [--modes eager,compile] [--micro-batches 4,8,16,32]
       [--tokens-per-step 524288] [--seq-len N] [--steps 10] [--warmup-steps 3]
       [--seed 1234] [--allow-busy]

Each case (a mode and a micro-batch size) builds the model fresh from the seed
and times full optimizer steps of --tokens-per-step tokens: FP16 autocast,
gradient accumulation over micro-batches, GradScaler, gradient clipping and
fused AdamW, as scripts/gpt2_torch.py defines them. The input is random token
ids generated once on the GPU, because the step's cost doesn't depend on the
token values; data loading is deliberately left out, so the numbers are the
training step alone. A micro-batch that runs out of memory is recorded as
"oom", and the larger ones for that mode are skipped.

An NVML monitor (loaded through ctypes, mirroring bench/monitor.cpp) samples
clocks, power, temperature, clock-event reasons, device memory and other
processes every ~20 ms. A step is discarded if a new process used the GPU
during it, or if other processes' memory grew by more than 256 MiB since
warm-up. Clock slowdowns are reported, not grounds to discard, as in
flint_bench. The GPU must be idle at start: otherwise the script exits with
code 2, unless --allow-busy is passed. The JSON log has the schema
flint-torch-baseline/1.
"""

import argparse
import bisect
import ctypes
import dataclasses
import gc
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpt2_torch  # noqa: E402

SCHEMA = "flint-torch-baseline/1"
MIB = 1024.0 * 1024.0
# A step is discarded if other processes' memory grows past its value after
# warm-up by more than this, and the GPU counts as busy at start past this much.
MEMORY_GROWTH_LIMIT = 256 * 1024 * 1024
SAMPLE_INTERVAL = 0.02
# FP16 tensor-core FLOP per SM per clock with FP32 accumulation, on GeForce
# sm_75 and sm_89 alike (as in bench/).
PEAK_FLOP_PER_SM_PER_CLOCK = 512
# The boost clock behind the vendor's rated peak, for the GPUs we know; the
# laptop's clocks vary by design, so it has no rated peak. Same table as
# bench/monitor.cpp.
RATED_SM_CLOCK_MHZ = {"NVIDIA GeForce RTX 2080 Ti": 1545}
TOKENS_TARGET = 10_000_000_000
# Inductor compiles in a pool of one worker per CPU by default. The server has
# 20 CPUs and little free host RAM, and compiling isn't timed.
COMPILE_THREADS = 4

# NVML's clock-event reasons, in bit order, and their names in the raw log.
REASON_NAMES = [
    "gpu_idle", "applications_clocks", "sw_power_cap", "hw_slowdown", "sync_boost",
    "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake", "display_clocks",
]
# The shares the summary flags, and the power cap it doesn't.
THERMAL_MASK = (1 << 5) | (1 << 6)
HW_SLOWDOWN_MASK = (1 << 3) | (1 << 7)
SW_POWER_CAP_MASK = 1 << 2
SAMPLE_READS = 7  # the quantities a sample reads, for counting failed reads


# ---------------------------------------------------------------- NVML

class NvmlError(RuntimeError):
    pass


class _MemoryV2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _MemoryV1(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _ProcessInfo(ctypes.Structure):
    # nvmlProcessInfo_t, which the _v3 process queries fill.
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


class _ProcessInfoV1(ctypes.Structure):
    # The older layout of the _v2 process queries.
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]


NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_TEMPERATURE_GPU = 0
# NVML_STRUCT_VERSION(Memory, 2): the struct's size with the version in the top byte.
NVML_MEMORY_V2 = ctypes.sizeof(_MemoryV2) | (2 << 24)


def load_nvml_library():
    if sys.platform == "win32":
        candidates = ["nvml.dll", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
    else:
        candidates = ["libnvidia-ml.so.1"]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise NvmlError(f"can't load NVML ({candidates[0]}); is the NVIDIA driver installed?")


class Nvml:
    """The NVML reads the monitor needs, for the machine's one GPU (index 0).
    A read that fails returns None."""

    def __init__(self):
        self._lib = load_nvml_library()
        self.own_pid = os.getpid()
        if self._bind("nvmlInit_v2")() != NVML_SUCCESS:
            raise NvmlError("nvmlInit failed")
        self._handle = ctypes.c_void_p()
        by_index = self._bind("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint, ctypes.c_void_p)
        if by_index(0, ctypes.byref(self._handle)) != NVML_SUCCESS:
            raise NvmlError("nvmlDeviceGetHandleByIndex failed")
        p = ctypes.c_void_p
        self._clock = self._bind("nvmlDeviceGetClockInfo", p, ctypes.c_uint, p)
        self._max_clock = self._bind("nvmlDeviceGetMaxClockInfo", p, ctypes.c_uint, p)
        self._power = self._bind("nvmlDeviceGetPowerUsage", p, p)
        self._temperature = self._bind("nvmlDeviceGetTemperature", p, ctypes.c_uint, p)
        # Renamed in newer drivers; the old name stays as an alias.
        self._reasons = self._bind("nvmlDeviceGetCurrentClocksEventReasons", p, p) or self._bind(
            "nvmlDeviceGetCurrentClocksThrottleReasons", p, p
        )
        # The v2 query leaves out what the driver reserves (527 MiB on the idle
        # 2080 Ti, where v1 reports it as used); v1 is the fallback.
        self._memory_v2 = self._bind("nvmlDeviceGetMemoryInfo_v2", p, p)
        self._memory_v1 = self._bind("nvmlDeviceGetMemoryInfo", p, p)
        self._utilization = self._bind("nvmlDeviceGetUtilizationRates", p, p)
        self._name = self._bind("nvmlDeviceGetName", p, ctypes.c_char_p, ctypes.c_uint)
        self._driver = self._bind("nvmlSystemGetDriverVersion", ctypes.c_char_p, ctypes.c_uint)
        self._shutdown = self._bind("nvmlShutdown")
        self._process_queries = []
        for kind in ("Compute", "Graphics"):
            for suffix, struct in (("_v3", _ProcessInfo), ("_v2", _ProcessInfoV1)):
                query = self._bind(f"nvmlDeviceGet{kind}RunningProcesses{suffix}", p, p, p)
                if query is not None:
                    self._process_queries.append((query, struct))
                    break

    def _bind(self, name, *argtypes):
        try:
            function = getattr(self._lib, name)
        except AttributeError:
            return None
        function.restype = ctypes.c_int
        function.argtypes = list(argtypes)
        return function

    def _read(self, function, ctype, *lead):
        if function is None:
            return None
        value = ctype()
        if function(self._handle, *lead, ctypes.byref(value)) != NVML_SUCCESS:
            return None
        return value.value

    def _string(self, function, *lead):
        buffer = ctypes.create_string_buffer(96)
        if function is None or function(*lead, buffer, len(buffer)) != NVML_SUCCESS:
            raise NvmlError("a string query failed")
        return buffer.value.decode("utf-8", "replace")

    def device_name(self):
        return self._string(self._name, self._handle)

    def driver_version(self):
        return self._string(self._driver)

    def max_clock_mhz(self, kind):
        return self._read(self._max_clock, ctypes.c_uint, NVML_CLOCK_SM if kind == "sm" else NVML_CLOCK_MEM)

    def clock_mhz(self, kind):
        return self._read(self._clock, ctypes.c_uint, NVML_CLOCK_SM if kind == "sm" else NVML_CLOCK_MEM)

    def power_mw(self):
        return self._read(self._power, ctypes.c_uint)

    def temperature_c(self):
        return self._read(self._temperature, ctypes.c_uint, NVML_TEMPERATURE_GPU)

    def clock_event_reasons(self):
        return self._read(self._reasons, ctypes.c_ulonglong)

    def used_bytes(self):
        if self._memory_v2 is not None:
            memory = _MemoryV2()
            memory.version = NVML_MEMORY_V2
            if self._memory_v2(self._handle, ctypes.byref(memory)) == NVML_SUCCESS:
                return memory.used
        if self._memory_v1 is not None:
            memory = _MemoryV1()
            if self._memory_v1(self._handle, ctypes.byref(memory)) == NVML_SUCCESS:
                return memory.used
        return None

    def utilization_percent(self):
        if self._utilization is None:
            return None
        utilization = _Utilization()
        if self._utilization(self._handle, ctypes.byref(utilization)) != NVML_SUCCESS:
            return None
        return utilization.gpu

    def processes(self):
        """PIDs of other processes on the GPU from the compute and graphics
        lists, or None if neither list can be read."""
        pids = set()
        any_list = False
        for query, struct in self._process_queries:
            size = 64
            infos = (struct * size)()
            count = ctypes.c_uint(size)
            result = query(self._handle, ctypes.byref(count), infos)
            if result == NVML_ERROR_INSUFFICIENT_SIZE:
                size = count.value + 16  # room for processes that start meanwhile
                infos = (struct * size)()
                count = ctypes.c_uint(size)
                result = query(self._handle, ctypes.byref(count), infos)
            # Some drivers don't support one of the lists; the other still counts.
            if result != NVML_SUCCESS:
                continue
            any_list = True
            pids.update(infos[i].pid for i in range(min(count.value, size)))
        if not any_list:
            return None
        pids.discard(self.own_pid)
        return sorted(pids)

    def close(self):
        if self._shutdown is not None:
            self._shutdown()
            self._shutdown = None


# ------------------------------------------------------------- monitor

def window_stats(samples, begin, end):
    """Summarizes the samples over [begin, end] seconds. Means and shares are
    weighted by time, holding each sample until the next, and use only the
    samples whose read of that quantity succeeded (a failed read is None)."""
    times = [s["t"] for s in samples]
    # The sample in effect at `begin` is the last one taken at or before it.
    first = bisect.bisect_right(times, begin)
    if first > 0:
        first -= 1
    stats = {"seconds": end - begin, "samples": 0, "failed_reads": 0, "temp_c_max": 0,
             "new_process": False, "used_bytes_max": 0}
    totals = {"sm": 0.0, "mem": 0.0, "power": 0.0}
    weights = {"sm": 0.0, "mem": 0.0, "power": 0.0, "reasons": 0.0}
    shares = [0.0] * len(REASON_NAMES)
    flagged = {"thermal": 0.0, "hw_slowdown": 0.0, "sw_power_cap": 0.0}
    for i in range(first, len(samples)):
        s = samples[i]
        if s["t"] >= end:
            break
        start = max(s["t"], begin)
        stop = min(samples[i + 1]["t"], end) if i + 1 < len(samples) else end
        weight = max(stop - start, 0.0)
        stats["samples"] += 1
        stats["failed_reads"] += sum(
            s[key] is None for key in ("sm_mhz", "mem_mhz", "power_mw", "temp_c", "reasons", "used_bytes")
        ) + (0 if s["processes_ok"] else 1)
        for key, field, scale in (("sm", "sm_mhz", 1.0), ("mem", "mem_mhz", 1.0), ("power", "power_mw", 1e-3)):
            if s[field] is not None:
                totals[key] += s[field] * scale * weight
                weights[key] += weight
        if s["reasons"] is not None:
            reasons = s["reasons"]
            for bit in range(len(REASON_NAMES)):
                shares[bit] += weight if reasons >> bit & 1 else 0.0
            flagged["thermal"] += weight if reasons & THERMAL_MASK else 0.0
            flagged["hw_slowdown"] += weight if reasons & HW_SLOWDOWN_MASK else 0.0
            flagged["sw_power_cap"] += weight if reasons & SW_POWER_CAP_MASK else 0.0
            weights["reasons"] += weight
        if s["temp_c"] is not None:
            stats["temp_c_max"] = max(stats["temp_c_max"], s["temp_c"])
        if s["used_bytes"] is not None:
            stats["used_bytes_max"] = max(stats["used_bytes_max"], s["used_bytes"])
        stats["new_process"] = stats["new_process"] or s["new_process"]

    def mean(total, weight):
        return total / weight if weight > 0 else 0.0

    stats["sm_mhz"] = mean(totals["sm"], weights["sm"])
    stats["mem_mhz"] = mean(totals["mem"], weights["mem"])
    stats["power_w"] = mean(totals["power"], weights["power"])
    stats["reason_shares"] = {n: mean(v, weights["reasons"]) for n, v in zip(REASON_NAMES, shares)}
    for key, total in flagged.items():
        stats[key] = mean(total, weights["reasons"])
    return stats


def merge_stats(windows):
    """Combines step windows into one, weighting each by its length."""
    merged = {"seconds": 0.0, "samples": 0, "failed_reads": 0, "temp_c_max": 0,
              "new_process": False, "used_bytes_max": 0}
    keys = ("sm_mhz", "mem_mhz", "power_w", "thermal", "hw_slowdown", "sw_power_cap")
    totals = dict.fromkeys(keys, 0.0)
    for w in windows:
        merged["seconds"] += w["seconds"]
        merged["samples"] += w["samples"]
        merged["failed_reads"] += w["failed_reads"]
        merged["temp_c_max"] = max(merged["temp_c_max"], w["temp_c_max"])
        merged["used_bytes_max"] = max(merged["used_bytes_max"], w["used_bytes_max"])
        merged["new_process"] = merged["new_process"] or w["new_process"]
        for key in keys:
            totals[key] += w[key] * w["seconds"]
    for key in keys:
        merged[key] = totals[key] / merged["seconds"] if merged["seconds"] > 0 else 0.0
    return merged


class Monitor:
    """A background NVML sampler over an Nvml (or anything with its methods).
    Construct it before torch creates the CUDA context, so the memory in use at
    start is other processes' alone."""

    def __init__(self, nvml, interval=SAMPLE_INTERVAL, utilization_reads=5, utilization_pause=0.1):
        self._nvml = nvml
        self._interval = interval
        pids = nvml.processes()
        self.process_detection = pids is not None
        self.processes_at_start = sorted(pids or [])
        used = nvml.used_bytes()
        if used is None:
            raise NvmlError("can't read device memory")
        self.used_bytes_at_start = used
        # Utilization is the share of NVML's last sampling period in which any
        # kernel ran. We've launched nothing, so it's someone else's, even a
        # process NVML hides.
        self.utilization_at_start = 0
        for _ in range(utilization_reads):
            self.utilization_at_start = max(self.utilization_at_start, nvml.utilization_percent() or 0)
            time.sleep(utilization_pause)
        self._samples = []
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def used_bytes_now(self):
        return self._nvml.used_bytes() or 0

    def clear(self):
        """Drops the samples so far, so a long sweep's samples don't pile up."""
        with self._condition:
            self._samples = []

    def _read(self):
        nvml = self._nvml
        sample = {"sm_mhz": nvml.clock_mhz("sm"), "mem_mhz": nvml.clock_mhz("mem")}
        # Stamped between the clock reads and the slower ones that follow.
        sample["t"] = time.perf_counter()
        sample["power_mw"] = nvml.power_mw()
        sample["temp_c"] = nvml.temperature_c()
        sample["reasons"] = nvml.clock_event_reasons()
        sample["used_bytes"] = nvml.used_bytes()
        pids = nvml.processes()
        sample["processes_ok"] = pids is not None
        sample["new_process"] = any(p not in self.processes_at_start for p in pids or [])
        return sample

    def _run(self):
        deadline = time.perf_counter()
        while not self._stop.is_set():
            sample = self._read()
            with self._condition:
                self._samples.append(sample)
                self._condition.notify_all()
            # Fixed deadlines keep the cadence near the interval where the OS
            # timer allows; a late sample doesn't trigger a burst to catch up.
            deadline = max(deadline + self._interval, time.perf_counter())
            self._stop.wait(max(deadline - time.perf_counter(), 0.0))

    def window(self, begin, end, timeout=10.0):
        """Summarizes [begin, end] (perf_counter seconds), waiting for the first
        sample after end."""
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._samples and self._samples[-1]["t"] >= end, timeout=timeout
            ):
                raise RuntimeError("the NVML monitor stopped sampling")
            return window_stats(self._samples, begin, end)


def busy_reasons(process_count, used_bytes, utilization, process_detection):
    """Why the GPU doesn't look idle at start, as the same phrases harness.cpp uses."""
    reasons = []
    if process_count:
        reasons.append(f"{process_count} other processes")
    if used_bytes > MEMORY_GROWTH_LIMIT:
        reasons.append(f"{int(used_bytes / MIB)} MiB in use before our context")
    if utilization > 0:
        reasons.append(f"{utilization}% utilization before we started")
    if not process_detection:
        reasons.append("NVML can't list processes, so others would go unseen")
    return reasons


# ------------------------------------------------------ FLOPs and MFU

def rated_sm_clock_mhz(name):
    """The boost clock behind the rated peak of a known GPU, else None."""
    return RATED_SM_CLOCK_MHZ.get(name)


def flops_per_token(n_params, n_layer, n_embd, seq_len, convention):
    """Training FLOPs per token: 6N for the weights plus the attention term.
    PaLM/nanoGPT counts 12 L d T for it; llm.c counts 6 L d T. N excludes wpe."""
    attention = {"palm": 12, "llmc": 6}
    if convention not in attention:
        raise ValueError(f"unknown FLOPs convention {convention!r}")
    return 6 * n_params + attention[convention] * n_layer * n_embd * seq_len


def peak_flops(sm_count, sm_mhz):
    return PEAK_FLOP_PER_SM_PER_CLOCK * sm_count * sm_mhz * 1e6


def mfu(tokens_per_second, flops_per_token_, peak):
    """The share of `peak` FLOP/s the run achieves, or None without a peak."""
    if not peak or peak <= 0:
        return None
    return tokens_per_second * flops_per_token_ / peak


def days_for_tokens(tokens_per_second, tokens=TOKENS_TARGET):
    return tokens / tokens_per_second / 86400


# ----------------------------------------------------- steps and cases

def invalid_reason(stats, reserved_at_end, other_baseline):
    """Why a timed step can't count, or None. Other processes' memory is
    estimated as NVML's device memory in use, less what this process's
    allocator holds, and must not grow past its value after warm-up."""
    if stats["new_process"]:
        return "another process used the GPU"
    if stats["used_bytes_max"] > 0:
        other = stats["used_bytes_max"] - reserved_at_end
        if other > other_baseline + MEMORY_GROWTH_LIMIT:
            return f"other processes' memory grew by {int((other - other_baseline) / MIB)} MiB"
    return None


def step_record(index, seconds, skipped, stats, reserved_at_end, other_baseline, tokens_per_step):
    reason = invalid_reason(stats, reserved_at_end, other_baseline)
    return {
        "step": index,
        "seconds": seconds,
        "tokens_per_second": tokens_per_step / seconds,
        "skipped": skipped,
        "valid": reason is None,
        "discard_reason": reason,
        "memory_reserved_bytes_end": reserved_at_end,
        "other_memory_bytes": stats["used_bytes_max"] - reserved_at_end if stats["used_bytes_max"] else None,
        "stats": stats,
    }


def case_status(steps):
    """"too few valid steps" if fewer than half the timed steps are valid."""
    valid = sum(1 for s in steps if s["valid"])
    return "ok" if steps and 2 * valid >= len(steps) else "too few valid steps"


def summarize_steps(steps, tokens_per_step, flops, sm_count, rated_mhz):
    """Statistics over a case's valid steps. `flops` maps a convention to its
    FLOPs per token; MFU is against the rated peak when the GPU has one, and
    always against the peak at the mean SM clock."""
    valid = [s for s in steps if s["valid"]]
    summary = {
        "timed_steps": len(steps),
        "valid_steps": len(valid),
        "skipped_steps": sum(1 for s in steps if s["skipped"]),
        "peak_nvml_used_mib": max((s["stats"]["used_bytes_max"] for s in steps), default=0) / MIB,
    }
    if not valid:
        return {**summary, "tokens_per_second": None, "stats": None, "mfu": None, "days_for_10b_tokens": None}
    rates = [s["tokens_per_second"] for s in valid]
    median = statistics.median(rates)
    merged = merge_stats([s["stats"] for s in valid])
    rated_peak = peak_flops(sm_count, rated_mhz) if rated_mhz else None
    adjusted_peak = peak_flops(sm_count, merged["sm_mhz"]) if merged["sm_mhz"] > 0 else None
    return {
        **summary,
        "tokens_per_second": {"median": median, "min": min(rates), "max": max(rates)},
        "stats": {k: merged[k] for k in ("seconds", "sm_mhz", "mem_mhz", "power_w", "temp_c_max",
                                         "thermal", "hw_slowdown", "sw_power_cap")},
        "mfu": {
            "rated_peak_tflops": rated_peak / 1e12 if rated_peak else None,
            "clock_adjusted_peak_tflops": adjusted_peak / 1e12 if adjusted_peak else None,
            "rated": {c: mfu(median, f, rated_peak) for c, f in flops.items()} if rated_peak else None,
            "clock_adjusted": {c: mfu(median, f, adjusted_peak) for c, f in flops.items()}
            if adjusted_peak
            else None,
        },
        "days_for_10b_tokens": days_for_tokens(median),
    }


def run_sweep(modes, micro_batches, run_case, cleanup, on_case=None):
    """Runs every mode's micro-batches in increasing order through
    `run_case(mode, micro_batch)`, which returns a case record or raises
    torch.OutOfMemoryError. A micro-batch that runs out of memory is recorded
    as "oom" and the larger ones are skipped for that mode. `cleanup(mode)`
    runs after every case. Returns (cases, skipped)."""
    cases, skipped = [], []
    for mode in modes:
        out_of_memory = False
        for micro_batch in sorted(set(micro_batches)):
            if out_of_memory:
                skipped.append({"mode": mode, "micro_batch": micro_batch,
                                "reason": "a smaller micro-batch ran out of memory"})
                continue
            try:
                case = run_case(mode, micro_batch)
            except torch.OutOfMemoryError:
                out_of_memory = True
                case = {"mode": mode, "micro_batch": micro_batch, "status": "oom", "steps": [], "summary": None}
            # Outside the except block, so the failed case's tensors, held by
            # the traceback until then, can be freed.
            cleanup(mode)
            cases.append(case)
            if on_case is not None:
                on_case(case)
    return cases, skipped


# ------------------------------------------------------------ real runs

def make_batches(config, micro_batch, seq_len, seed, count, device="cuda"):
    generator = torch.Generator(device).manual_seed(seed)
    batches = []
    for _ in range(count):
        tokens = torch.randint(0, config.vocab_size, (micro_batch, seq_len + 1), generator=generator, device=device)
        batches.append((tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()))
    return batches


def free_everything(mode):
    gc.collect()
    if mode == "compile":
        torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure_case(args, config, monitor, device, mode, micro_batch):
    seq_len = args.seq_len
    accum = args.tokens_per_step // (seq_len * micro_batch)
    print(f"running {mode}, micro-batch {micro_batch} ({accum} micro-batches per step) ...", flush=True)
    # Earlier cases' windows have been read.
    monitor.clear()
    model = gpt2_torch.build_model(config, seed=args.seed, device="cuda")
    params = list(model.parameters())
    optimizer = gpt2_torch.configure_optimizer(model)
    scaler = torch.amp.GradScaler("cuda")
    module = torch.compile(model) if mode == "compile" else model
    # A couple of micro-batches, reused: the step's cost doesn't depend on the tokens.
    distinct = make_batches(config, micro_batch, seq_len, args.seed, min(accum, 2))
    batches = [distinct[i % len(distinct)] for i in range(accum)]

    def timed_step():
        torch.cuda.synchronize()
        begin = time.perf_counter()
        skipped = gpt2_torch.train_step(module, params, optimizer, scaler, batches)
        torch.cuda.synchronize()
        return begin, time.perf_counter(), skipped

    for _ in range(args.warmup_steps):
        timed_step()
    # Memory the other processes hold, once lazily loaded modules and library
    # workspaces have settled: later growth in it means someone else.
    other_baseline = max(monitor.used_bytes_now() - torch.cuda.memory_reserved(), 0)
    raw = []
    for _ in range(args.steps):
        begin, end, skipped = timed_step()
        raw.append((begin, end, skipped, torch.cuda.memory_reserved()))
    # The windows are read after the loop, so waiting for the sampler leaves
    # no idle gap between steps.
    steps = [
        step_record(i, end - begin, skipped, monitor.window(begin, end), reserved, other_baseline, args.tokens_per_step)
        for i, (begin, end, skipped, reserved) in enumerate(raw)
    ]
    summary = summarize_steps(steps, args.tokens_per_step, args.flops, device["sm_count"], device["rated_sm_clock_mhz"])
    summary["peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved() / MIB
    summary["peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / MIB
    return {
        "mode": mode,
        "micro_batch": micro_batch,
        "accum_steps": accum,
        "status": case_status(steps),
        "steps": steps,
        "summary": summary,
    }


def gpu_info(nvml):
    if not torch.cuda.is_available():
        raise SystemExit("error: CUDA is not available")
    properties = torch.cuda.get_device_properties(0)
    if properties.name != nvml.device_name():
        raise SystemExit(f"error: NVML's GPU 0 ({nvml.device_name()}) is not torch's ({properties.name})")
    return {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_mib": properties.total_memory / MIB,
        "max_sm_clock_mhz": nvml.max_clock_mhz("sm"),
        "max_mem_clock_mhz": nvml.max_clock_mhz("mem"),
        "rated_sm_clock_mhz": rated_sm_clock_mhz(properties.name),
        "driver_version": nvml.driver_version(),
    }


def git_info():
    """The repo's HEAD and whether its tree is dirty, as cmake/git_info.cmake
    finds them: "unknown" and clean outside a git checkout."""
    sha, dirty = "unknown", False
    if (ROOT / ".git").exists():
        try:
            head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
            if head and all(c in "0123456789abcdef" for c in head):
                sha = head
                status = subprocess.run(["git", "-C", str(ROOT), "--no-optional-locks", "status", "--porcelain"],
                                        capture_output=True, text=True, check=True).stdout
                dirty = status != ""
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"sha": sha, "dirty": dirty}


def versions():
    found = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import triton

        found["triton"] = triton.__version__
    except ImportError:
        found["triton"] = None
    return found


# ------------------------------------------------------------ rendering

def pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_table(cases):
    header = (f"{'mode':<8} {'mb':>3}  {'status':<20} {'tokens/s median [min-max]':<28} {'GiB':>5}  "
              f"{'MFU PaLM / llm.c (rated)':<26} {'MFU PaLM / llm.c (clock)':<26} {'days/10B':>8}  flags")
    lines = [header]
    for case in cases:
        s = case["summary"]
        head = f"{case['mode']:<8} {case['micro_batch']:>3}  {case['status']:<20} "
        if s is None or s["tokens_per_second"] is None:
            lines.append(head + "-")
            continue
        t = s["tokens_per_second"]
        rates = f"{t['median']:,.0f} [{t['min']:,.0f}-{t['max']:,.0f}]"
        m = s["mfu"]
        rated = m["rated"]
        adjusted = m["clock_adjusted"]
        rated_text = f"{pct(rated['palm'])} / {pct(rated['llmc'])}" if rated else "n/a"
        adjusted_text = f"{pct(adjusted['palm'])} / {pct(adjusted['llmc'])}" if adjusted else "n/a"
        flags = f"thermal {pct(s['stats']['thermal'])}, hw {pct(s['stats']['hw_slowdown'])}"
        lines.append(
            head + f"{rates:<28} {s['peak_memory_reserved_mib'] / 1024:>5.1f}  {rated_text:<26} "
            f"{adjusted_text:<26} {s['days_for_10b_tokens']:>8.1f}  {flags}"
        )
    return "\n".join(lines)


def write_log(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")


# ----------------------------------------------------------------- main

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="JSON log to write")
    parser.add_argument("--config", choices=sorted(gpt2_torch.CONFIGS), default="gpt2")
    parser.add_argument("--modes", default="eager,compile", help="comma-separated: eager, compile")
    parser.add_argument("--micro-batches", default="4,8,16,32", help="comma-separated sequences per micro-batch")
    parser.add_argument("--tokens-per-step", type=int, default=2**19)
    parser.add_argument("--seq-len", type=int, help="default: the config's block size")
    parser.add_argument("--steps", type=int, default=10, help="timed steps per case")
    parser.add_argument("--warmup-steps", type=int, default=3, help="untimed steps, compilation included")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-busy", action="store_true", help="run although other work is on the GPU")
    args = parser.parse_args(argv)

    config = gpt2_torch.CONFIGS[args.config]
    args.seq_len = args.seq_len or config.block_size
    args.modes = [m for m in args.modes.split(",") if m]
    if not args.modes or any(m not in ("eager", "compile") for m in args.modes):
        parser.error("--modes takes eager and compile")
    if "compile" in args.modes and importlib.util.find_spec("triton") is None:
        parser.error("torch.compile needs Triton, which isn't installed here; use --modes eager")
    try:
        args.micro_batches = sorted({int(m) for m in args.micro_batches.split(",") if m})
    except ValueError:
        parser.error("--micro-batches takes comma-separated integers")
    if not args.micro_batches or args.micro_batches[0] < 1:
        parser.error("--micro-batches needs positive integers")
    if not 1 <= args.seq_len <= config.block_size:
        parser.error(f"--seq-len must be between 1 and the block size, {config.block_size}")
    if args.steps < 1 or args.warmup_steps < 0:
        parser.error("--steps needs at least 1, and --warmup-steps can't be negative")
    for micro_batch in args.micro_batches:
        if args.tokens_per_step % (args.seq_len * micro_batch):
            parser.error(
                f"--tokens-per-step {args.tokens_per_step} isn't a multiple of "
                f"seq-len x micro-batch = {args.seq_len} x {micro_batch}"
            )
    return args


def run(args, config, nvml, monitor, busy):
    device = gpu_info(nvml)
    # FP16 split-K reductions are off, matching Flint's FP32 accumulation.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if "compile" in args.modes:
        # Bound under its own name: a plain `import torch...` here would make
        # `torch` local to the whole function.
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = COMPILE_THREADS
        args.compile_threads = COMPILE_THREADS

    model = gpt2_torch.build_model(config, device="meta")
    n_params = gpt2_torch.count_parameters(model, exclude_wpe=True)
    args.flops = {
        c: flops_per_token(n_params, config.n_layer, config.n_embd, args.seq_len, c) for c in ("palm", "llmc")
    }
    settings = {k: v for k, v in vars(args).items() if k not in ("out", "flops")}
    document = {
        "schema": SCHEMA,
        "complete": False,
        "git": git_info(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "os": "windows" if sys.platform == "win32" else "linux",
        "versions": versions(),
        "device": device,
        "model": {
            "config": dataclasses.asdict(config),
            "parameters": gpt2_torch.count_parameters(model),
            "parameters_excluding_wpe": n_params,
            "flops_per_token": args.flops,
        },
        "settings": {
            **settings,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        },
        "gpu_at_start": {
            "other_processes": len(monitor.processes_at_start),
            "pids": monitor.processes_at_start,
            "used_bytes": monitor.used_bytes_at_start,
            "utilization_percent": monitor.utilization_at_start,
            "process_detection": monitor.process_detection,
            "busy": busy,
        },
        "cases": [],
        "skipped": [],
    }
    del model

    def record(case):
        document["cases"].append(case)
        write_log(args.out, document)  # so a crash later keeps what ran

    monitor.start()
    cases, skipped = run_sweep(
        args.modes,
        args.micro_batches,
        lambda mode, micro_batch: measure_case(args, config, monitor, device, mode, micro_batch),
        free_everything,
        record,
    )
    document["skipped"] = skipped
    document["complete"] = True
    write_log(args.out, document)
    print()
    print(render_table(cases), flush=True)
    for entry in skipped:
        print(f"skipped {entry['mode']} micro-batch {entry['micro_batch']}: {entry['reason']}")
    print(f"wrote {Path(args.out).name}", flush=True)
    return 0


def main(argv=None, nvml_factory=Nvml):
    args = parse_args(argv)
    config = gpt2_torch.CONFIGS[args.config]
    try:
        nvml = nvml_factory()
    except NvmlError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        # Before torch initializes CUDA, so the memory in use is other processes' alone.
        monitor = Monitor(nvml)
        busy = busy_reasons(len(monitor.processes_at_start), monitor.used_bytes_at_start,
                            monitor.utilization_at_start, monitor.process_detection)
        if busy:
            message = ", ".join(busy)
            if not args.allow_busy:
                print(f"the GPU is busy: {message} (pass --allow-busy to run anyway)", flush=True)
                return 2
            print(f"warning: the GPU is busy: {message}; running anyway (--allow-busy)", file=sys.stderr, flush=True)
        try:
            return run(args, config, nvml, monitor, busy)
        finally:
            monitor.stop()
    finally:
        nvml.close()


if __name__ == "__main__":
    sys.exit(main())
