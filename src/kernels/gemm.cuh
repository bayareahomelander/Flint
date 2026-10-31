// The tensor-core GEMM.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The largest n the grid can cover, 65,535 tiles of 128 columns in its y
// dimension. m has the same limit, which its z dimension covers with room to
// spare.
constexpr size_t kGemmMaxMN = 65535 * 128;

// Launches y = x w^T on stream, for row-major FP16 matrices: y is m x n, x is
// m x k, and w is n x k. Expects the checks flint_gemm_nt makes: m and n
// positive, k a positive multiple of 8, aligned pointers, and the size limits.
// Returns the launch's error, if any.
cudaError_t launch_gemm_nt(void* y, const void* x, const void* w, int m, int n, int k,
                           cudaStream_t stream);

// The same, adding bias (n FP32 values, one per column) to the FP32 sums
// before they round. Expects flint_gemm_nt_bias's checks.
cudaError_t launch_gemm_nt_bias(void* y, const void* x, const void* w, const void* bias, int m,
                                int n, int k, cudaStream_t stream);

// The same, also storing the biased sums rounded to FP16 in pre (m x n), and
// applying GPT-2's tanh GELU to them before they round into y. Expects
// flint_gemm_nt_bias_gelu's checks.
cudaError_t launch_gemm_nt_bias_gelu(void* y, void* pre, const void* x, const void* w,
                                     const void* bias, int m, int n, int k, cudaStream_t stream);

// The same as launch_gemm_nt_bias into FP32 y, adding the biased sums to the
// residual stream, FP32 m x n: y = residual + (x w^T + bias). Expects
// flint_gemm_nt_bias_residual's checks.
cudaError_t launch_gemm_nt_bias_residual(void* y, const void* x, const void* w, const void* bias,
                                         const void* residual, int m, int n, int k,
                                         cudaStream_t stream);

// The same for y = x w, with w k x n. Expects flint_gemm_nn's checks, which add
// that n is a multiple of 8.
cudaError_t launch_gemm_nn(void* y, const void* x, const void* w, int m, int n, int k,
                           cudaStream_t stream);

// The same, multiplying each FP32 sum by GPT-2's tanh GELU's derivative at
// pre (m x n FP16) before it rounds. Expects flint_gemm_nn_dgelu's checks.
cudaError_t launch_gemm_nn_dgelu(void* y, const void* x, const void* w, const void* pre, int m,
                                 int n, int k, cudaStream_t stream);

// The workspace an m x n x k TN GEMM needs on the current device, in bytes: 0
// when it runs as one split of k.
cudaError_t gemm_tn_workspace(int m, int n, int k, size_t* bytes);

// Launches y = x^T w, or y += x^T w with accumulate, for FP32 y (m x n) and
// FP16 x (k x m) and w (k x n), all row-major. Expects flint_gemm_tn's checks:
// m and n multiples of 8, k from 1, the size limits, aligned pointers, and a
// workspace of gemm_tn_workspace's size when that isn't 0.
cudaError_t launch_gemm_tn(void* y, const void* x, const void* w, int m, int n, int k,
                           bool accumulate, void* workspace, cudaStream_t stream);

}  // namespace flint
