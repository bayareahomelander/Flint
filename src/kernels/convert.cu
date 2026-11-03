// FP32 to FP16, a flat array at a time, in grid-stride loops as copy.cu's.
#include "kernels/convert.cuh"

#include <algorithm>
#include <cstdint>

#include <cuda_fp16.h>

namespace flint {
namespace {

constexpr unsigned kThreads = 256;
// Many waves on either GPU; the grid-stride loops cover any larger size.
constexpr size_t kMaxBlocks = 4096;

// Both pointers are 16-byte aligned: converts four values at a time, then the
// last count % 4 values, one per thread.
__global__ void __launch_bounds__(kThreads)
    fp32_to_fp16(__half* __restrict__ y, const float* __restrict__ x, size_t count) {
    const size_t first = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const size_t stride = size_t{gridDim.x} * blockDim.x;
    const size_t quads = count / 4;
    const auto* in = reinterpret_cast<const float4*>(x);
    auto* out = reinterpret_cast<__half2*>(y);
    for (size_t i = first; i < quads; i += stride) {
        const float4 v = in[i];
        out[2 * i] = __floats2half2_rn(v.x, v.y);
        out[2 * i + 1] = __floats2half2_rn(v.z, v.w);
    }
    if (first < count % 4) {
        const size_t i = quads * 4 + first;
        y[i] = __float2half_rn(x[i]);
    }
}

}  // namespace

cudaError_t launch_fp32_to_fp16(void* y, const void* x, size_t count, cudaStream_t stream) {
    const size_t quads = count / 4;
    const size_t blocks = std::clamp<size_t>((quads + kThreads - 1) / kThreads, 1, kMaxBlocks);
    fp32_to_fp16<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        static_cast<__half*>(y), static_cast<const float*>(x), count);
    return cudaGetLastError();
}

}  // namespace flint
