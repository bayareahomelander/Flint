// LayerNorm's backward pass.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "kernels/bias_gradient.cuh"
#include "kernels/layernorm.cuh"

namespace flint {

// The widest row: a multiple of 8, since dbeta is the bias gradient's column sum
// and dy moves in 16-byte pieces of 8 FP16 values.
constexpr size_t kLayerNormBackwardMaxCols = kBiasGradientMaxCols;
// The most rows: dx has a warp per row, 8 to a block, as the forward pass does.
constexpr size_t kLayerNormBackwardMaxRows = kLayerNormMaxRows;
static_assert(kLayerNormBackwardMaxCols <= kLayerNormMaxCols, "cols must convert to FP32 exactly");
static_assert(kLayerNormBackwardMaxRows <= kBiasGradientMaxRows, "dbeta must take every row");

// The workspace layernorm_backward needs, in bytes: the bias gradient's, which dbeta
// uses first and dgamma's partial sums then reuse.
constexpr size_t layernorm_backward_workspace(size_t rows, size_t cols) {
    return bias_gradient_workspace(rows, cols);
}

// Launches LayerNorm's backward pass on stream, for dy rows x cols FP16, x rows x
// cols FP32, mean and rstd rows FP32 each (the forward pass's) and gamma cols FP32:
// with xhat = (x - mean) rstd and g = dy gamma,
//   dx = rstd (g - mean(g) - xhat mean(g xhat)), each row's means over its cols,
//   dgamma = the column sums of dy xhat, and dbeta = the column sums of dy.
// dx (rows x cols FP32) is written, or with accumulate_dx added to with one FP32
// add; dgamma and dbeta (cols FP32 each) likewise with accumulate_params. Expects
// flint_layernorm_backward's checks: rows and cols positive and within the limits,
// cols a multiple of 8, 16-byte aligned pointers, a workspace of
// layernorm_backward_workspace(rows, cols) bytes, and no overlap. Returns the first
// launch error, if any.
cudaError_t launch_layernorm_backward(void* dx, void* dgamma, void* dbeta, const void* dy,
                                      const void* x, const void* mean, const void* rstd,
                                      const void* gamma, size_t rows, size_t cols,
                                      bool accumulate_dx, bool accumulate_params, void* workspace,
                                      cudaStream_t stream);

}  // namespace flint
