# Flint: GPT-2 from scratch on hand-written CUDA kernels

Flint trains GPT-2 124M from scratch on one RTX 2080 Ti using only its own CUDA
kernels: no cuBLAS, cuDNN or PyTorch at runtime. It will then serve that model,
and OpenAI's GPT-2 checkpoints, through its own inference kernels.

Training GPT-2 without PyTorch has been done; llm.c, for one, does it in C and
CUDA, calling cuBLAS for its matrix multiplies. Flint's claim is narrower: no
NVIDIA math libraries at all. Its GEMMs and flash attention (backward included)
are hand-written, it trains in FP16 with loss scaling on Turing, its speed is
reported against cuBLAS and the hardware's peak, and it carries through to
inference.

**Status: v0.1.0, the GEMM (milestone M1).** The tensor-core GEMMs a training
step needs are done and measured; the rest of the training step comes next. The
full results, method and profile are in [docs/results/M1.md](docs/results/M1.md),
and the changes in [CHANGELOG.md](CHANGELOG.md).

## GEMM results

FP16 inputs, FP32 accumulation, GPT-2 small's shapes at 8,192 tokens (MxNxK).
Each cell is the median of 10 interleaved rounds against cuBLAS's
`cublasGemmEx` (or cuBLASLt's matching epilogue, for the fused kernels), with
the same transposes and FP32 compute. The 2080 Ti's 53.8 TFLOPS rated peak
assumes its reference 1,545 MHz boost; under these loads it runs at
1,630-1,810 MHz, power-capped.

| Layer | Shape | 2080 Ti: Flint TFLOPS | 2080 Ti: % of baseline | Laptop: % of baseline |
| --- | --- | ---: | ---: | ---: |
| **Forward, y = x wᵀ** | | | | |
| qkv | 8192x2304x768 | 44.3 | 82.1% | 88.0% |
| attention projection | 8192x768x768 | 42.7 | 85.0% | 88.3% |
| MLP up | 8192x3072x768 | 44.7 | 83.1% | 86.5% |
| MLP down | 8192x768x3072 | 47.5 | 88.2% | 86.9% |
| logits | 8192x50304x768 | 47.3 | 89.0% | 88.5% |
| **Input gradient, dX = dY W** | | | | |
| qkv | 8192x768x2304 | 49.1 | 90.6% | 88.5% |
| attention projection | 8192x768x768 | 43.3 | 85.9% | 87.2% |
| MLP up | 8192x768x3072 | 49.4 | 90.9% | 88.8% |
| MLP down | 8192x3072x768 | 46.0 | 90.4% | 86.9% |
| logits | 8192x768x50304 | 47.3 | 99.1% | 78.3% |
| **Weight gradient, dW = dYᵀ X (FP32, accumulating)** | | | | |
| qkv | 2304x768x8192 | 45.4 | 95.2% | 79.6% |
| attention projection | 768x768x8192 | 41.2 | 107.8% | 81.2% |
| MLP up | 3072x768x8192 | 41.2 | 91.4% | 82.2% |
| MLP down | 768x3072x8192 | 41.5 | 92.3% | 84.8% |
| logits | 50304x768x8192 | 51.8 | 93.9% | 91.0% |
| **Fused, against cuBLASLt** | | | | |
| qkv, + bias | 8192x2304x768 | 43.6 | 83.2% | 97.9% |
| attention projection, + bias | 8192x768x768 | 42.2 | 86.7% | 84.3% |
| MLP up, + bias | 8192x3072x768 | 43.8 | 83.7% | 86.7% |
| MLP down, + bias | 8192x768x3072 | 46.7 | 89.1% | 83.6% |
| MLP up, + bias + GELU | 8192x3072x768 | 39.3 | 105.7% | 83.7% |
| MLP down input gradient, + GELU's backward | 8192x3072x768 | 40.8 | 148.5% | 95.9% |
| attention projection, + bias + residual | 8192x768x768 | 32.5 | 85.5% | 82.8% |
| MLP down, + bias + residual | 8192x768x3072 | 44.4 | 89.5% | 84.7% |

- M1's targets on the 2080 Ti were 70% of cuBLAS for the forward and
  input-gradient GEMMs and 60% for the weight gradient. Every shape clears them.
- The fused GELU kernels beat cuBLASLt on the 2080 Ti mostly because cuBLASLt
  runs those epilogues as a separate kernel on Turing.
- The laptop (RTX 4050 Laptop GPU) shares its GPU with the desktop, so its
  numbers are indicative only.
- Each number links to a raw JSON log through the tables in
  [M1.md](docs/results/M1.md), which also give the ranges, clocks and power.
  M1.md's profile section explains where the remaining gap to cuBLAS comes from.

## Timing protocol

`flint_bench` gives Flint and the baseline identical random inputs, rotated
across buffer sets larger than 4x the L2 cache. It warms each up for 1.5 s,
then times 10 rounds that alternate the two, each sample at least 100 ms of
back-to-back launches. An NVML thread logs clocks, power, clock-event reasons
and the processes on the GPU. A round is discarded if another process appears
or device memory grows, and the benchmark refuses to start on a busy GPU unless
given `--allow-busy`. Each run writes a JSON log with the commit, GPU, driver,
clocks and every round; `scripts/results_table.py` renders the tables from those
logs.

## Building and testing

Flint needs an NVIDIA GPU of compute capability 7.5 or 8.9, the CUDA 12.4
toolkit, CMake 3.28 or newer, Ninja, and Python 3.12. The build compiles for
both sm_75 and sm_89. PyTorch 2.6 is used only by the tests, as the reference.

**Linux.** `environment.yml` describes a conda environment (CUDA 12.4.1, GCC 13,
CMake, Ninja, Python); `scripts/setup_server.sh` creates it with micromamba and
installs `requirements.txt`. Then, inside it:

```bash
bash scripts/check.sh
```

which configures, builds and runs the test suite.

**Windows.** With the CUDA 12.4 toolkit and Visual Studio 2022 Build Tools
installed, `scripts\setup_laptop.ps1` creates `.venv` from
`requirements-windows.txt`. Then:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\check.ps1
```

**By hand**, in either environment:

```bash
cmake --preset release
cmake --build --preset release
python -m pytest
python scripts/sanitize.py
build/release/bin/flint_bench gemm --layout nt
```

`sanitize.py` runs the suite under compute-sanitizer's memcheck, racecheck,
synccheck and initcheck, and checks that two deliberately broken calls are
caught. `flint_bench --help` lists the benchmarks and their options.

## Layout

- `include/flint/flint.h`: the C API.
- `src/`: the library (`kernels/` for the CUDA kernels, `api/` for argument
  checks).
- `bench/`: `flint_bench`, the only part that links cuBLAS and NVML.
- `tests/`: pytest, calling the library through ctypes.
- `scripts/`: build, test, sanitizer, profiling and result-table tools.
- `docs/results/`: one write-up per milestone, with the raw logs behind every
  number.

## Limitations

- The GEMMs accumulate in FP32 on the tensor cores, which truncate, so k is
  capped at 65,536.
- Results are bit-identical from run to run on one GPU, not across GPUs.
- The 2080 Ti's clocks are logged, not locked, and its GEMMs run power-capped.
- The laptop's numbers vary with its power state and its desktop load.

More are listed in [M1.md](docs/results/M1.md#limitations).

## Roadmap

| Milestone | Goal |
| --- | --- |
| M0 | Build, test bridge and benchmark harness ([write-up](docs/results/M0.md)) |
| M1 | Tensor-core GEMM (this release, [write-up](docs/results/M1.md)) |
| M2 | One training step matching PyTorch: embeddings, LayerNorm, attention, cross-entropy, AdamW, loss scaling |
| M3 | Flash attention, forward and backward, on Turing |
| M4 | The long run: GPT-2 124M on 10B FineWeb-Edu tokens, evaluated on HellaSwag against OpenAI's checkpoint |
| M5 | Inference: KV cache, decode attention, int8 weights, OpenAI's checkpoints up to 1.5B |

## Contributors

Paco Sun, with Claude (Anthropic) as an AI pair programmer.

## License

MIT; see [LICENSE](LICENSE).
