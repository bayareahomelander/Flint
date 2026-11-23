// The gradient's L2 norm and its count of non-finite values, over a list of FP32 tensors.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The first pass's blocks, fixed so that the order of the additions depends only on the
// tensors' counts, not on the GPU. Each leaves an FP64 partial sum and a 64-bit count.
constexpr size_t kGradientNormBlocks = 1024;
// The workspace gradient_norm needs, in bytes: the partial sums, then the partial counts.
constexpr size_t kGradientNormWorkspaceBytes = kGradientNormBlocks * 16;

// Launches the two passes on stream: *norm = the square root of the sum of the squares of
// every value of the tensors, in FP64, and *nonfinite = how many of the values are inf or
// NaN. tensors holds tensor_count { const float* data; size_t count; } entries in device
// memory (flint_tensor's layout). Expects flint_gradient_norm's checks: 16-byte aligned
// pointers, a workspace of kGradientNormWorkspaceBytes bytes, and no overlap. Returns the
// first launch error, if any.
cudaError_t launch_gradient_norm(void* norm, void* nonfinite, const void* tensors,
                                 size_t tensor_count, void* workspace, cudaStream_t stream);

}  // namespace flint
