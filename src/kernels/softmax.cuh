// Attention's causal softmax.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The longest row, GPT-2's longest context, and the most batches, as for the
// batched GEMMs that produce S and consume P.
constexpr size_t kSoftmaxMaxLength = 1024;
constexpr size_t kSoftmaxMaxBatches = 65535;

// Launches causal_softmax_forward on stream. p and s hold rows matrix rows of
// length FP16 values (rows = batches * length), and row t = row % length gets
// the softmax of its columns 0 to t, with +0 in the rest. Expects
// flint_causal_softmax_forward's checks: length a positive multiple of 8 within
// the limit, rows positive, 16-byte aligned pointers, and no overlap. Returns the
// launch's error, if any.
cudaError_t launch_causal_softmax_forward(void* p, const void* s, size_t rows, size_t length,
                                          cudaStream_t stream);

}  // namespace flint
