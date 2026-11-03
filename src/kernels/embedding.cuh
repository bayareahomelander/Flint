// The embedding's forward pass.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The widest row: the GEMMs need multiples of 8, and this is LayerNorm's limit.
constexpr size_t kEmbeddingMaxWidth = size_t{1} << 24;
// The largest vocabulary: a token is 16 bits.
constexpr size_t kEmbeddingMaxVocab = size_t{1} << 16;
// The most rows: a warp per row, 8 to a block, in a grid of up to 2^31 - 1
// blocks.
constexpr size_t kEmbeddingMaxRows = size_t{0x7fffffff} * 8;

// Launches the embedding's forward pass on stream: for each of rows tokens
// (uint16), row r of x (rows x width FP32) is row tokens[r] of wte
// (vocab x width FP32) plus row r % length of wpe (at least length x width
// FP32). Expects flint_embedding_forward's checks: width a multiple of 8 up to
// the limit, rows and length positive with rows within the limit, and 16-byte
// aligned pointers. Tokens must be below vocab, which the kernel doesn't
// check. Returns the launch's error, if any.
cudaError_t launch_embedding_forward(void* x, const void* tokens, const void* wte,
                                     const void* wpe, size_t rows, size_t length, size_t width,
                                     cudaStream_t stream);

}  // namespace flint
