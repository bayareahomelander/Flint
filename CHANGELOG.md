# Changelog

## 0.1.0 (2026-10-30)

The first release: milestone M1, the GEMM, on top of M0's build and harnesses.
Results, method and limitations are in [docs/results/M1.md](docs/results/M1.md).

- **GEMMs:** FP16 tensor-core matrix multiplies with FP32 accumulation, in the
  three layouts of a linear layer's training step: `flint_gemm_nt` (forward),
  `flint_gemm_nn` (input gradient) and `flint_gemm_tn` (weight gradient, FP32,
  with a deterministic split of k).
- **Fused epilogues:** `flint_gemm_nt_bias`, `flint_gemm_nt_bias_gelu`,
  `flint_gemm_nn_dgelu` (GELU's backward) and `flint_gemm_nt_bias_residual`
  (into an FP32 residual stream).
- **Speed:** on an RTX 2080 Ti, 82-89% of cuBLAS for the forward GEMM, 86-99%
  for the input gradient and 91-108% for the weight gradient, at GPT-2 small's
  shapes. Also measured on an RTX 4050 Laptop GPU.
- **Tools:** `flint_bench` times every GEMM against cuBLAS or cuBLASLt;
  `scripts/profile_gemm.py`, `nsys_kernels.py`, `k_sweep.py` and `sass_loop.py`
  profile the gap to cuBLAS; `scripts/results_table.py` renders the write-up's
  tables from the raw logs.
- **From M0:** the CMake build for sm_75 and sm_89 on Linux and Windows, the C
  API (`include/flint/flint.h`), the pytest bridge, compute-sanitizer runs, and
  the benchmark harness, first used on `flint_copy`.
