// Flint's first kernel, a device-to-device byte copy. It exercises the build,
// the C API, and the test, sanitizer, and benchmark harnesses.
#include "kernels/copy.cuh"

#include <algorithm>
#include <cstdint>

namespace flint {
namespace {

constexpr unsigned kThreads = 256;
// Many waves on either GPU; the grid-stride loops cover any larger size.
constexpr size_t kMaxBlocks = 4096;

// Both pointers are 16-byte aligned: copies 16-byte words, then the last
// bytes % 16 bytes, one per thread.
__global__ void copy_aligned(void* __restrict__ dst, const void* __restrict__ src,
                             size_t bytes) {
    const size_t first = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const size_t stride = size_t{gridDim.x} * blockDim.x;
    const size_t words = bytes / 16;
    auto* dst_words = static_cast<uint4*>(dst);
    const auto* src_words = static_cast<const uint4*>(src);
    for (size_t i = first; i < words; i += stride) {
        dst_words[i] = src_words[i];
    }
    if (first < bytes % 16) {
        const size_t i = words * 16 + first;
        static_cast<unsigned char*>(dst)[i] = static_cast<const unsigned char*>(src)[i];
    }
}

// Any alignment: one byte per iteration.
__global__ void copy_unaligned(unsigned char* __restrict__ dst,
                               const unsigned char* __restrict__ src, size_t bytes) {
    const size_t first = size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const size_t stride = size_t{gridDim.x} * blockDim.x;
    for (size_t i = first; i < bytes; i += stride) {
        dst[i] = src[i];
    }
}

}  // namespace

cudaError_t launch_copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    if (bytes == 0) {
        return cudaSuccess;
    }
    const bool aligned =
        (reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) % 16 == 0;
    const size_t items = aligned ? bytes / 16 : bytes;
    const size_t blocks = std::clamp<size_t>((items + kThreads - 1) / kThreads, 1, kMaxBlocks);
    if (aligned) {
        copy_aligned<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(dst, src, bytes);
    } else {
        copy_unaligned<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
            static_cast<unsigned char*>(dst), static_cast<const unsigned char*>(src), bytes);
    }
    return cudaGetLastError();
}

}  // namespace flint
