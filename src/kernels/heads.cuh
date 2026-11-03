// Attention's head layouts: splitting a row buffer into heads and merging them
// back.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The widest row of the rows layout.
constexpr size_t kHeadsMaxRowWidth = size_t{1} << 24;
// The most rows: a warp per row, 8 to a block, in a grid of up to 2^31 - 1
// blocks.
constexpr size_t kHeadsMaxRows = size_t{0x7fffffff} * 8;

// Launch heads_split and heads_merge on stream, which move FP16 values as bits,
// 16 bytes at a time. The rows layout is rows x row_width, with rows =
// batches * length. The heads layout is batches x heads x length x head_width.
// Split copies the columns [first, first + heads * head_width) of each row of
// x (rows layout) to y (heads layout), and merge copies x (heads layout) back
// to those columns of y (rows layout), leaving the other columns alone. Expect
// flint_heads_split's checks: head_width, row_width and first multiples of 8,
// heads positive, the columns within the row, row_width and rows within the
// limits and positive (length too), 16-byte aligned pointers, and no overlap.
// Return the launch's error, if any.
cudaError_t launch_heads_split(void* y, const void* x, size_t rows, size_t length, size_t heads,
                               size_t head_width, size_t row_width, size_t first,
                               cudaStream_t stream);
cudaError_t launch_heads_merge(void* y, const void* x, size_t rows, size_t length, size_t heads,
                               size_t head_width, size_t row_width, size_t first,
                               cudaStream_t stream);

}  // namespace flint
