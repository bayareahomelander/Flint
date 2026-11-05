// The output layer's softmax and cross-entropy in one kernel: one warp per row, all
// in FP32, with each gradient rounded to FP16 once.
//
// The softmax has attention's shape over a row's first vocab columns. With max the
// largest of those logits and sum adding exp(logit - max) over them, max's exp is
// 1, the sum stays between 1 and vocab, and nothing overflows. A first pass finds
// max, a second the sum, and a third writes the gradient, recomputing each exp. A
// row is re-read from global memory on each pass.
//
// The third pass also writes the row's loss, as log(sum) - (logit - max) rather
// than -log(p), so a target whose exp underflows still gets a finite loss. It runs
// in the lane that holds the target, so no other lane needs the target's logit.
//
// The sum is compensated, as LayerNorm's are (TwoSum, after Ogita, Rump and Oishi's
// Sum2), because its error is also the loss's. With plain FP32, GPT-2's real logits
// lose about 1e-5: a row's dominant exp is about 1, and the lane that holds it then
// absorbs some 1,500 far smaller exps, each losing part of itself to rounding, all
// in the same direction.
//
// Non-finite logits need no special case. fmaxf skips a NaN, but its exp makes the
// sum NaN, and with it the row's loss and gradients. A +inf is the max, and inf -
// inf gives the same NaN, as does a row whose logits are all -inf. A -inf among
// finite logits gets an exp of 0, and if it's the target's, the loss is +inf.
// Columns at or above vocab are never used, whatever they hold.
#include "kernels/cross_entropy.cuh"

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

// How many of chunk i's values are below vocab, for a chunk with at least one.
__device__ int vocab_values(int vocab, int i) {
    return min(vocab - i * kChunkValues, kChunkValues);
}

// A rounded sum and the sum of the rounding errors left behind in it.
struct CompensatedSum {
    float sum = 0.0f;
    float error = 0.0f;
};

// Adds v. Knuth's TwoSum finds the error e with sum + v = t + e exactly, for any
// order of magnitude between the two. The compiler fuses expf's last multiply into
// these additions, which changes nothing while that product is exact: it's a power
// of 2 times a rounded value, inexact only when its result is subnormal, about
// 1e-38 at most against a sum of at least 1.
__device__ void add(CompensatedSum& a, float v) {
    const float t = a.sum + v;
    const float v_part = t - a.sum;
    a.error += (a.sum - (t - v_part)) + (v - v_part);
    a.sum = t;
}

// The total over the warp's lanes, received by every lane. At each butterfly step
// the two lanes hold the same pair in swapped order. TwoSum's error is exact, so it
// doesn't depend on that order, and FP32 addition is commutative, so all lanes end
// with the same bits.
__device__ float warp_total(CompensatedSum a) {
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        const float other_sum = __shfl_xor_sync(kAllLanes, a.sum, offset);
        const float other_error = __shfl_xor_sync(kAllLanes, a.error, offset);
        const float error = a.error + other_error;
        a.error = 0.0f;
        add(a, other_sum);
        a.error += error;
    }
    return a.sum + a.error;
}

// Each lane takes every 32nd chunk of a row, so the order of its partial sums
// depends only on the row's width.
__global__ void __launch_bounds__(kThreads)
    softmax_cross_entropy(uint4* __restrict__ dlogits, float* __restrict__ row_loss,
                          const uint4* __restrict__ logits,
                          const unsigned short* __restrict__ targets, float scale, size_t rows,
                          int vocab, int cols) {
    const size_t row = size_t{blockIdx.x} * kRowsPerBlock + threadIdx.x / kWarp;
    // A warp's lanes share a row, so whole warps leave together and the
    // shuffles below always have all 32 lanes.
    if (row >= rows) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x % kWarp);
    const int chunks = cols / kChunkValues;
    const int vocab_chunks = (vocab + kChunkValues - 1) / kChunkValues;
    const uint4* logits_row = logits + row * chunks;
    uint4* dlogits_row = dlogits + row * chunks;

    float max_logit = -__builtin_huge_valf();
    for (int i = lane; i < vocab_chunks; i += kWarp) {
        const Chunk c = load(logits_row, i);
        const int count = vocab_values(vocab, i);
        for (int j = 0; j < count; ++j) {
            max_logit = fmaxf(max_logit, c.v[j]);
        }
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        max_logit = fmaxf(max_logit, __shfl_xor_sync(kAllLanes, max_logit, offset));
    }

    CompensatedSum partial;
    for (int i = lane; i < vocab_chunks; i += kWarp) {
        const Chunk c = load(logits_row, i);
        const int count = vocab_values(vocab, i);
        for (int j = 0; j < count; ++j) {
            add(partial, expf(c.v[j] - max_logit));
        }
    }
    const float sum = warp_total(partial);

    const int target = targets[row];
    const int target_chunk = target / kChunkValues;
    const int target_value = target % kChunkValues;
    const float inverse = 1.0f / sum;
    for (int i = lane; i < chunks; i += kWarp) {
        if (i >= vocab_chunks) {
            dlogits_row[i] = make_uint4(0, 0, 0, 0);
            continue;
        }
        Chunk c = load(logits_row, i);
        const int count = vocab_values(vocab, i);
        for (int j = 0; j < kChunkValues; ++j) {
            if (j >= count) {
                c.v[j] = 0.0f;
                continue;
            }
            const float gap = c.v[j] - max_logit;
            float p = expf(gap) * inverse;
            if (i == target_chunk && j == target_value) {
                row_loss[row] = logf(sum) - gap;
                p -= 1.0f;
            }
            c.v[j] = p * scale;
        }
        dlogits_row[i] = make_uint4(pack(c.v[0], c.v[1]), pack(c.v[2], c.v[3]),
                                    pack(c.v[4], c.v[5]), pack(c.v[6], c.v[7]));
    }
}

}  // namespace

cudaError_t launch_softmax_cross_entropy(void* dlogits, void* row_loss, const void* logits,
                                         const void* targets, float scale, size_t rows,
                                         size_t vocab, size_t cols, cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    softmax_cross_entropy<<<blocks, kThreads, 0, stream>>>(
        static_cast<uint4*>(dlogits), static_cast<float*>(row_loss),
        static_cast<const uint4*>(logits), static_cast<const unsigned short*>(targets), scale, rows,
        static_cast<int>(vocab), static_cast<int>(cols));
    return cudaGetLastError();
}

}  // namespace flint
