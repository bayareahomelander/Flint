// LayerNorm.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The widest row: its length converts to FP32 exactly, so the mean's divisor
// is exact.
constexpr size_t kLayerNormMaxCols = size_t{1} << 24;
// The most rows: a warp per row, 8 to a block, in a grid of up to 2^31 - 1
// blocks.
constexpr size_t kLayerNormMaxRows = size_t{0x7fffffff} * 8;

// Launches LayerNorm's forward pass on stream: for each row of x (rows x cols
// FP32), y = (x - mean) rstd gamma + beta rounded to FP16, with the row's mean
// and rstd = 1 / sqrt(variance + eps) stored in mean and rstd (rows FP32
// each). Expects flint_layernorm_forward's checks: rows and cols positive and
// within the limits, and 16-byte aligned pointers. Returns the launch's error,
// if any.
cudaError_t launch_layernorm_forward(void* y, void* mean, void* rstd, const void* x,
                                     const void* gamma, const void* beta, size_t rows, int cols,
                                     float eps, cudaStream_t stream);

}  // namespace flint
