// The embedding's forward pass: one warp per row, each token's row of the
// word table plus its position's row of the position table, in FP32.
//
// A row's lanes read the token once, and each takes every 32nd group of four
// floats, so the loads and stores are 128 bits wide and a warp covers 512
// bytes at a time.
#include "kernels/embedding.cuh"

#include <cstdint>

namespace flint {
namespace {

constexpr int kWarp = 32;
constexpr int kRowsPerBlock = 8;
constexpr unsigned kThreads = kWarp * kRowsPerBlock;
constexpr unsigned kAllLanes = 0xffffffffu;

// The sequences' rows follow each other, so row r is position r % length of
// its sequence. Row and table offsets are size_t: a token times width needs
// more than 32 bits at the limits.
__global__ void __launch_bounds__(kThreads)
    embedding_forward(float* __restrict__ x, const uint16_t* __restrict__ tokens,
                      const float* __restrict__ wte, const float* __restrict__ wpe, size_t rows,
                      size_t length, size_t width) {
    const size_t row = size_t{blockIdx.x} * kRowsPerBlock + threadIdx.x / kWarp;
    // A warp's lanes share a row, so whole warps leave together and the
    // shuffle below always has all 32 lanes.
    if (row >= rows) {
        return;
    }
    const unsigned lane = threadIdx.x % kWarp;
    // One load of the token, not 32 of the same value.
    const unsigned loaded = lane == 0 ? tokens[row] : 0;
    const size_t token = __shfl_sync(kAllLanes, loaded, 0);
    const size_t position = row % length;

    const auto* word = reinterpret_cast<const float4*>(wte + token * width);
    const auto* place = reinterpret_cast<const float4*>(wpe + position * width);
    auto* out = reinterpret_cast<float4*>(x + row * width);
    const unsigned groups = static_cast<unsigned>(width / 4);
    for (unsigned i = lane; i < groups; i += kWarp) {
        const float4 a = word[i];
        const float4 b = place[i];
        out[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

}  // namespace

cudaError_t launch_embedding_forward(void* x, const void* tokens, const void* wte,
                                     const void* wpe, size_t rows, size_t length, size_t width,
                                     cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    embedding_forward<<<blocks, kThreads, 0, stream>>>(
        static_cast<float*>(x), static_cast<const uint16_t*>(tokens),
        static_cast<const float*>(wte), static_cast<const float*>(wpe), rows, length, width);
    return cudaGetLastError();
}

}  // namespace flint
