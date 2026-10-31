// The device-to-device byte copy.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// Launches a copy of bytes from src to dst on stream, which must not overlap.
// Returns the launch's error, if any.
cudaError_t launch_copy(void* dst, const void* src, size_t bytes, cudaStream_t stream);

}  // namespace flint
