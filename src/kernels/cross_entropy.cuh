// The output layer's softmax and cross-entropy.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The widest row: a target is 16 bits, so columns past that couldn't be targets.
constexpr size_t kCrossEntropyMaxCols = size_t{1} << 16;
// The most rows: a warp per row, 8 to a block, in a grid of up to 2^31 - 1
// blocks.
constexpr size_t kCrossEntropyMaxRows = size_t{0x7fffffff} * 8;

// Launches softmax_cross_entropy on stream. logits and dlogits hold rows x cols
// FP16, targets rows uint16 and row_loss rows FP32. Expects
// flint_softmax_cross_entropy's checks: cols a positive multiple of 8 within the
// limit, vocab from 1 to cols, rows positive and within the limit, a finite
// scale, 16-byte aligned pointers, and no overlap. Returns the launch's error, if
// any.
cudaError_t launch_softmax_cross_entropy(void* dlogits, void* row_loss, const void* logits,
                                         const void* targets, float scale, size_t rows,
                                         size_t vocab, size_t cols, cudaStream_t stream);

}  // namespace flint
