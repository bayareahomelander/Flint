// Attention's causal softmax: one warp per row, all in FP32, with each output
// rounded to FP16 once.
//
// Row t of a matrix sees its columns 0 to t. Its output is exp(s - max) / sum,
// where max is the largest visible score and sum adds exp(s - max) over the
// visible columns, so max's exp is 1, the sum stays between 1 and the row's
// length, and nothing overflows. A first pass finds max, a second the sum, and
// a third writes the row, recomputing each exp. The row is re-read from global
// memory on each pass; at 2 KB, the second and third come from cache.
//
// The sums are plain FP32, unlike LayerNorm's: each output goes through at most
// about 40 additions, for a relative error near 2e-6, far below the 2.4e-4 to
// 4.9e-4 of FP16's own rounding.
//
// Non-finite scores need no special case. fmaxf skips a NaN, but its exp makes
// the sum NaN, and with it the row's visible outputs. A +inf is the max, and
// inf - inf gives the same NaN, as does a row whose visible scores are all -inf.
// A -inf among finite scores gets an exp of 0. Masked columns are never used,
// whatever they hold.
#include "kernels/softmax.cuh"

#include <cuda_fp16.h>

namespace flint {
namespace {

constexpr int kWarp = 32;
constexpr int kRowsPerBlock = 8;
constexpr unsigned kThreads = kWarp * kRowsPerBlock;
constexpr unsigned kAllLanes = 0xffffffffu;
// Rows move as 16-byte chunks of 8 FP16 values.
constexpr int kChunkValues = 8;

// A chunk's values in FP32.
struct Chunk {
    float v[kChunkValues];
};

__device__ void unpack(unsigned bits, float& low, float& high) {
    low = __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
    high = __half2float(__ushort_as_half(static_cast<unsigned short>(bits >> 16)));
}

// The i-th chunk of a row.
__device__ Chunk load(const uint4* __restrict__ row, int i) {
    const uint4 q = row[i];
    Chunk c;
    unpack(q.x, c.v[0], c.v[1]);
    unpack(q.y, c.v[2], c.v[3]);
    unpack(q.z, c.v[4], c.v[5]);
    unpack(q.w, c.v[6], c.v[7]);
    return c;
}

__device__ unsigned pack(float low, float high) {
    const __half2_raw raw = __floats2half2_rn(low, high);
    return static_cast<unsigned>(raw.x) | static_cast<unsigned>(raw.y) << 16;
}

// How many of chunk i's values row t sees, for a chunk with at least one.
__device__ int visible_values(int t, int i) {
    return min(t + 1 - i * kChunkValues, kChunkValues);
}

// Each lane takes every 32nd chunk of a row, so the order of its partial sums
// depends only on the row's position, and the butterfly's additions commute, so
// all lanes end with the same bits.
__global__ void __launch_bounds__(kThreads)
    causal_softmax_forward(uint4* __restrict__ p, const uint4* __restrict__ s, size_t rows,
                           int length) {
    const size_t row = size_t{blockIdx.x} * kRowsPerBlock + threadIdx.x / kWarp;
    // A warp's lanes share a row, so whole warps leave together and the
    // shuffles below always have all 32 lanes.
    if (row >= rows) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x % kWarp);
    const int chunks = length / kChunkValues;
    const int t = static_cast<int>(row % length);
    const int visible_chunks = t / kChunkValues + 1;
    const uint4* s_row = s + row * chunks;
    uint4* p_row = p + row * chunks;

    float max_score = -__builtin_huge_valf();
    for (int i = lane; i < visible_chunks; i += kWarp) {
        const Chunk c = load(s_row, i);
        const int visible = visible_values(t, i);
        for (int j = 0; j < visible; ++j) {
            max_score = fmaxf(max_score, c.v[j]);
        }
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        max_score = fmaxf(max_score, __shfl_xor_sync(kAllLanes, max_score, offset));
    }

    float sum = 0.0f;
    for (int i = lane; i < visible_chunks; i += kWarp) {
        const Chunk c = load(s_row, i);
        const int visible = visible_values(t, i);
        for (int j = 0; j < visible; ++j) {
            sum += expf(c.v[j] - max_score);
        }
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(kAllLanes, sum, offset);
    }

    const float scale = 1.0f / sum;
    for (int i = lane; i < chunks; i += kWarp) {
        if (i >= visible_chunks) {
            p_row[i] = make_uint4(0, 0, 0, 0);
            continue;
        }
        Chunk c = load(s_row, i);
        const int visible = visible_values(t, i);
        for (int j = 0; j < kChunkValues; ++j) {
            c.v[j] = j < visible ? expf(c.v[j] - max_score) * scale : 0.0f;
        }
        p_row[i] = make_uint4(pack(c.v[0], c.v[1]), pack(c.v[2], c.v[3]), pack(c.v[4], c.v[5]),
                              pack(c.v[6], c.v[7]));
    }
}

}  // namespace

cudaError_t launch_causal_softmax_forward(void* p, const void* s, size_t rows, size_t length,
                                          cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    causal_softmax_forward<<<blocks, kThreads, 0, stream>>>(
        static_cast<uint4*>(p), static_cast<const uint4*>(s), rows, static_cast<int>(length));
    return cudaGetLastError();
}

}  // namespace flint
