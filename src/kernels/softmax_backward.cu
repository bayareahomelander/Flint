// Attention's causal softmax, backward: one warp per row, all in FP32, with each
// output rounded to FP16 once.
//
// Row t of a matrix sees its columns 0 to t. With D the sum of p dp over them,
// ds = scale p (dp - D) there, and +0 in the rest, whatever p and dp hold. A first
// pass finds D, a second writes the row.
//
// D is kept as a compensated pair (TwoSum, as LayerNorm's and the bias gradient's
// sums are): a rounded sum s and the sum e of the rounding errors left behind in it.
// Each product p dp of two FP16 values is exact in FP32 (11 significant bits each),
// so the pair holds D to about twice FP32's precision. The pair is never added up:
// each output subtracts s, then e. Where dp is close to D, dp - s is exact and the
// difference's leading bits come from e, which a D rounded to FP32 would have lost.
// Measured on both GPUs with P from scores of std 1 to 16 and dp of five kinds,
// every output was within 1 FP16 ulp of the correctly rounded result and at least
// 99.99% equal to it; plain sums, or the pair added up first, were up to 4,468 ulps
// off where dp cancels D.
//
// Each lane takes every 32nd 16-byte chunk of the row, as the forward pass does, so
// the order of its partial sums depends only on the row's position, and a butterfly
// merges the lanes' pairs: at each step the two lanes of a pair hold the same two
// pairs in swapped order, and TwoSum's error is exact, so every lane ends with the
// same bits.
//
// ds may be dp itself: each lane reads all its chunks of dp for D before the
// butterfly, whose shuffles wait for every lane, and then reads and writes only its
// own chunks. So neither pointer is __restrict__.
#include "kernels/softmax_backward.cuh"

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
__device__ Chunk load(const uint4* row, int i) {
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

// A rounded sum and the sum of the rounding errors left behind in it.
struct CompensatedSum {
    float sum = 0.0f;
    float error = 0.0f;
};

// Adds v. Knuth's TwoSum finds the error e with sum + v = t + e exactly, for any
// order of magnitude between the two.
__device__ void add(CompensatedSum& a, float v) {
    const float t = a.sum + v;
    const float v_part = t - a.sum;
    a.error += (a.sum - (t - v_part)) + (v - v_part);
    a.sum = t;
}

// Merges b into a: b's sum through TwoSum, then both errors. Swapping a and b gives
// the same bits, since TwoSum's error is exact.
__device__ void merge(CompensatedSum& a, float b_sum, float b_error) {
    const float errors = a.error + b_error;
    a.error = 0.0f;
    add(a, b_sum);
    a.error += errors;
}

// The pair over the warp's lanes, received by every lane. When the sum isn't finite,
// TwoSum's error is NaN, and the error is dropped, so an inf sum stays inf.
__device__ CompensatedSum warp_total(CompensatedSum a) {
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        const float other_sum = __shfl_xor_sync(kAllLanes, a.sum, offset);
        const float other_error = __shfl_xor_sync(kAllLanes, a.error, offset);
        merge(a, other_sum, other_error);
    }
    if (!isfinite(a.sum)) {
        a.error = 0.0f;
    }
    return a;
}

__global__ void __launch_bounds__(kThreads)
    causal_softmax_backward(uint4* ds, const uint4* __restrict__ p, const uint4* dp, size_t rows,
                            int length, float scale) {
    const size_t row = size_t{blockIdx.x} * kRowsPerBlock + threadIdx.x / kWarp;
    // A warp's lanes share a row, so whole warps leave together and the shuffles
    // below always have all 32 lanes.
    if (row >= rows) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x % kWarp);
    const int chunks = length / kChunkValues;
    const int t = static_cast<int>(row % length);
    const int visible_chunks = t / kChunkValues + 1;
    const uint4* p_row = p + row * chunks;
    const uint4* dp_row = dp + row * chunks;
    uint4* ds_row = ds + row * chunks;

    CompensatedSum sum;
    for (int i = lane; i < visible_chunks; i += kWarp) {
        const Chunk pc = load(p_row, i);
        const Chunk dc = load(dp_row, i);
        const int visible = visible_values(t, i);
        for (int j = 0; j < visible; ++j) {
            add(sum, __fmul_rn(pc.v[j], dc.v[j]));
        }
    }
    const CompensatedSum d = warp_total(sum);

    for (int i = lane; i < chunks; i += kWarp) {
        if (i >= visible_chunks) {
            ds_row[i] = make_uint4(0, 0, 0, 0);
            continue;
        }
        const Chunk pc = load(p_row, i);
        Chunk c = load(dp_row, i);
        const int visible = visible_values(t, i);
        for (int j = 0; j < kChunkValues; ++j) {
            const float diff = __fsub_rn(__fsub_rn(c.v[j], d.sum), d.error);
            c.v[j] = j < visible ? __fmul_rn(__fmul_rn(pc.v[j], diff), scale) : 0.0f;
        }
        ds_row[i] = make_uint4(pack(c.v[0], c.v[1]), pack(c.v[2], c.v[3]), pack(c.v[4], c.v[5]),
                               pack(c.v[6], c.v[7]));
    }
}

}  // namespace

cudaError_t launch_causal_softmax_backward(void* ds, const void* p, const void* dp, size_t rows,
                                           size_t length, float scale, cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    causal_softmax_backward<<<blocks, kThreads, 0, stream>>>(
        static_cast<uint4*>(ds), static_cast<const uint4*>(p), static_cast<const uint4*>(dp), rows,
        static_cast<int>(length), scale);
    return cudaGetLastError();
}

}  // namespace flint
