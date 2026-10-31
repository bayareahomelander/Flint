// The gemm benchmark: flint_gemm_nt (y = x w^T), flint_gemm_nn (y = x w) or
// flint_gemm_tn (y += x^T w, with FP32 y) against cuBLAS, in FP16 with FP32
// accumulation, timed as a fraction of cuBLAS. flint_gemm_nt_bias (y = x w^T +
// bias) runs against cuBLASLt's bias epilogue instead, and
// flint_gemm_nt_bias_gelu (the same, then GELU, with the sums before it stored
// too) against its GELU_AUX_BIAS epilogue, flint_gemm_nn_dgelu (y = x w
// times GELU's derivative at pre) against its DGELU epilogue, and
// flint_gemm_nt_bias_residual (y = residual + (x w^T + bias), with FP32 y and
// residual) against its bias epilogue with the residual as C. Each cuBLASLt case
// times the heuristic's first 8 algorithms before its rounds and uses the
// fastest as its baseline (--lt-algo first: the heuristic's first choice), and
// logs them all.
#pragma once

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include "cli.h"
#include "harness.h"

namespace bench {

// A cuBLAS handle, created and destroyed with the object. Create it before
// the memory baseline is read, so cuBLAS's own allocations count as ours.
class CublasHandle {
public:
    CublasHandle();
    ~CublasHandle();
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

// A cuBLASLt handle, created and destroyed with the object, and created before
// the memory baseline like CublasHandle.
class CublasLtHandle {
public:
    CublasLtHandle();
    ~CublasLtHandle();
    CublasLtHandle(const CublasLtHandle&) = delete;
    CublasLtHandle& operator=(const CublasLtHandle&) = delete;
    cublasLtHandle_t get() const { return handle_; }

private:
    cublasLtHandle_t handle_ = nullptr;
};

CaseResult bench_gemm(const GemmShape& shape, GemmLayout layout, GemmEpilogue epilogue,
                      cublasHandle_t cublas, cublasLtHandle_t cublas_lt, const DeviceInfo& device,
                      Monitor& monitor, const Options& options, cudaStream_t stream);

}  // namespace bench
