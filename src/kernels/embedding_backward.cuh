// The embedding's backward pass: the word table's gradient, bucketed by token, and
// the position table's, a column sum over the sequences.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "kernels/bias_gradient.cuh"
#include "kernels/embedding.cuh"

namespace flint {

// The widest row: the word table's gradient has a block per 256 columns in the grid's
// y, which holds up to 65,535, as the bias gradient's first pass does.
constexpr size_t kEmbeddingBackwardMaxWidth = kBiasGradientMaxCols;
// The most rows, sequences * length: the sort keeps row indices in 32 bits.
constexpr size_t kEmbeddingBackwardMaxRows = 0xffffffffu;
// The widest position gradient, length * width: it's the bias gradient's column sum
// over the sequences, with dx viewed as sequences x (length * width).
constexpr size_t kEmbeddingBackwardMaxPositionValues = kBiasGradientMaxCols;
static_assert(kEmbeddingBackwardMaxRows <= kBiasGradientMaxRows,
              "the position gradient must take every sequence");

// The sort's tiles: a block of 8 warps takes 2,048 rows, each warp 256 of them, 32 at
// a time.
constexpr size_t kEmbeddingSortTileRows = 2048;
// A sort pass orders the rows by one 8-bit digit of the token.
constexpr size_t kEmbeddingSortDigits = 256;

// The workspace's layout, in bytes from its start, each part 16-byte aligned:
//   position partials  the bias gradient's partial sums for dx viewed as sequences x
//                      (length * width), bias_gradient_workspace's size
//   tokens             rows uint16 values, the tokens in sorted order
//   rows               rows uint32 values, the row of each sorted token: ascending
//                      within a token, since the sort is stable
//   offsets            vocab + 1 uint32 values: offsets[v] is the number of rows whose
//                      token is below v, so token v's rows are sorted positions
//                      offsets[v] to offsets[v + 1] - 1
//   tokens and rows    the first sort pass's output, rows uint16 then rows uint32
//   counts             256 uint32 values per tile of 2,048 rows: each digit's count
//                      for the tile, digit by digit, then their exclusive prefix sums
// The tests read the sorted tokens and rows, the offsets and the partial sums.
struct EmbeddingBackwardLayout {
    size_t position_partials = 0;
    size_t sorted_tokens = 0;
    size_t sorted_rows = 0;
    size_t offsets = 0;
    size_t pass_tokens = 0;
    size_t pass_rows = 0;
    size_t counts = 0;
    size_t bytes = 0;
};

constexpr size_t embedding_backward_align(size_t bytes) {
    return (bytes + 15) / 16 * 16;
}

constexpr EmbeddingBackwardLayout embedding_backward_layout(size_t sequences, size_t length,
                                                            size_t vocab, size_t width) {
    const size_t rows = sequences * length;
    const size_t tiles = (rows + kEmbeddingSortTileRows - 1) / kEmbeddingSortTileRows;
    EmbeddingBackwardLayout layout;
    size_t at = embedding_backward_align(bias_gradient_workspace(sequences, length * width));
    layout.sorted_tokens = at;
    at += embedding_backward_align(rows * 2);
    layout.sorted_rows = at;
    at += embedding_backward_align(rows * 4);
    layout.offsets = at;
    at += embedding_backward_align((vocab + 1) * 4);
    layout.pass_tokens = at;
    at += embedding_backward_align(rows * 2);
    layout.pass_rows = at;
    at += embedding_backward_align(rows * 4);
    layout.counts = at;
    at += embedding_backward_align(tiles * kEmbeddingSortDigits * 4);
    layout.bytes = at;
    return layout;
}

// Launches the embedding's backward pass on stream, for dx (sequences * length) x
// width FP32 and tokens sequences * length uint16:
//   dwte[v] = the sum of dx[r] over the rows r whose token is v, for v below vocab, in
//             each token's rows in ascending order; written to all vocab rows of dwte
//             (+0 where no row's token is v), or with accumulate_wte added with one
//             FP32 add to the rows some token reaches, the others unchanged. A token at
//             or above vocab reaches no row.
//   dwpe[t] = the sum over sequences s of dx[s * length + t], for t below length, the
//             bias gradient's column sum; written, or added with accumulate_wpe.
// Expects flint_embedding_backward's checks: sequences, length, vocab and width
// positive and within the limits, width a multiple of 8, 16-byte aligned pointers, a
// workspace of embedding_backward_layout(...).bytes bytes, and no overlap. Returns
// the first launch error, if any.
cudaError_t launch_embedding_backward(void* dwte, void* dwpe, const void* dx, const void* tokens,
                                      size_t sequences, size_t length, size_t vocab, size_t width,
                                      bool accumulate_wte, bool accumulate_wpe, void* workspace,
                                      cudaStream_t stream);

}  // namespace flint
