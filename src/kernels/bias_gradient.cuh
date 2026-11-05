// A linear layer's bias gradient: each column's sum over the rows.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The rows one block of the first pass sums. The workspace holds two FP32 rows per
// such chunk, the partial sums and their errors, so its size and the order of the
// additions depend only on rows and cols, not on the GPU.
constexpr size_t kBiasGradientChunkRows = 256;
// The widest dy: the first pass's grid has a block per 256 columns in y, which
// holds up to 65,535.
constexpr size_t kBiasGradientMaxCols = size_t{65535} * 256;
// The most rows: the first pass has a block per chunk in x, which holds up to
// 2^31 - 1.
constexpr size_t kBiasGradientMaxRows = size_t{0x7fffffff} * kBiasGradientChunkRows;

// The workspace bias_gradient needs, in bytes: an FP32 partial sum and its error
// per chunk of rows and column.
constexpr size_t bias_gradient_workspace(size_t rows, size_t cols) {
    return (rows + kBiasGradientChunkRows - 1) / kBiasGradientChunkRows * cols * 8;
}

// Launches the two passes on stream: dbias[j] = sum over r of dy[r, j], or with
// accumulate dbias[j] += that sum, for dy rows x cols FP16 and dbias cols FP32.
// Expects flint_bias_gradient's checks: rows and cols positive and within the
// limits, cols a multiple of 8, 16-byte aligned pointers, a workspace of
// bias_gradient_workspace(rows, cols) bytes, and no overlap. Returns the first
// launch error, if any.
cudaError_t launch_bias_gradient(void* dbias, const void* dy, size_t rows, size_t cols,
                                 bool accumulate, void* workspace, cudaStream_t stream);

}  // namespace flint
