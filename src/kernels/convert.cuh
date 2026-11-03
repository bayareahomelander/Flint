// Conversion between FP32 and FP16.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace flint {

// The most values: their FP32 bytes must not wrap.
constexpr size_t kConvertMaxCount = SIZE_MAX / 4;

// Launches the conversion of count FP32 values at x to FP16 at y on stream,
// each rounded to nearest with ties to even: overflow gives inf, and NaN stays
// NaN. Expects flint_fp32_to_fp16's checks: count positive and within the
// limit, 16-byte aligned pointers, and no overlap. Returns the launch's error,
// if any.
cudaError_t launch_fp32_to_fp16(void* y, const void* x, size_t count, cudaStream_t stream);

}  // namespace flint
