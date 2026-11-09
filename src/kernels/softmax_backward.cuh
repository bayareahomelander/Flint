// Attention's causal softmax, backward.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "kernels/softmax.cuh"

namespace flint {

// Launches causal_softmax_backward on stream. ds, p and dp hold rows matrix rows of
// length FP16 values (rows = batches * length). Row t = row % length gets
// ds = scale p (dp - D) in its columns 0 to t, with D the sum of p dp over them, and
// +0 in the rest. Expects flint_causal_softmax_backward's checks: length a positive
// multiple of 8 within kSoftmaxMaxLength, rows positive, a finite scale, 16-byte
// aligned pointers, and ds either equal to dp or overlapping neither input. Returns
// the launch's error, if any.
cudaError_t launch_causal_softmax_backward(void* ds, const void* p, const void* dp, size_t rows,
                                           size_t length, float scale, cudaStream_t stream);

}  // namespace flint
