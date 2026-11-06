// LayerNorm's backward pass, in FP32 with no atomics, in an order that depends only on
// rows and cols, so the result is the same bits on every run and every GPU.
//
// For each value of a row, with xhat = (x - mean) rstd and g = dy gamma,
//   dx = rstd ((g - mean(g)) - xhat mean(g xhat)),
// with both means over the row's cols, and dgamma and dbeta are the column sums of
// dy xhat and of dy. xhat is rounded as the forward pass rounds it: x - mean, then
// times rstd. Every inexact operation is one FP32 add, subtract, multiply or divide,
// through an intrinsic the compiler can't fuse with another; fmaf only finds a
// product's rounding error, which is exact. So the result can be reproduced bit for
// bit in plain FP32 arithmetic.
//
// dx gives a warp a row, 8 rows to a block, as the forward pass does. Lane l takes
// pieces l, l + 32, ... of 8 values (16 bytes of dy, 32 of x and of gamma) and adds g
// and g xhat over them in order, and a butterfly adds the lanes' sums. A second pass
// over the row, from cache at GPT-2's width, writes dx, or adds it to dx with one FP32
// add. Its sums are plain FP32: plain sums gave at most 1.2 times the error against
// FP64 that compensated ones did, since the rounding of each value's own terms
// dominates it, and stayed within 1.1 times FP32 PyTorch's error on both GPUs.
//
// dgamma's sums are compensated, as the forward pass's and the bias gradient's are
// (TwoSum, after Ogita, Rump and Oishi's Sum2), with each product's rounding error added
// to the errors, as the forward pass does for its squares. Plain sums came to up to 4
// times FP32 PyTorch's error with outlier channels near 3,000. Its first pass has the
// bias gradient's layout and order: a block takes a chunk of 256 rows and a tile of 256
// columns, thread (x, y) adds rows y, y + 8, ... of the chunk for its piece of 8
// columns, thread (x, 0) merges the other 7 threads' sums in order of y, and the block
// stores the chunk's sums and errors in the workspace. The bias gradient's second pass
// merges the chunks. dbeta is the bias gradient itself, run first in the same workspace.
#include "kernels/layernorm_backward.cuh"

#include <cuda_fp16.h>

namespace flint {
namespace {

constexpr int kWarp = 32;
constexpr unsigned kAllLanes = 0xffffffffu;
// dy, x and gamma move in pieces of 8 values: 16 bytes of FP16, 32 of FP32.
constexpr int kPieceValues = 8;
// dx's block: a warp per row.
constexpr int kRowsPerBlock = 8;
// dgamma's first pass: the bias gradient's block, kTileX pieces across and kTileY
// rows down.
constexpr int kTileX = kWarp;
constexpr int kTileY = 8;
constexpr int kChunkRows = static_cast<int>(kBiasGradientChunkRows);
static_assert(kTileX * kPieceValues == 256, "the bias gradient's tiles are 256 columns");

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

// Adds the product a b, with the product's own rounding error. __fmul_rn keeps the
// compiler from fusing the product into the addition that follows.
__device__ void add_product(CompensatedSum& s, float a, float b) {
    const float p = __fmul_rn(a, b);
    s.error += fmaf(a, b, -p);
    add(s, p);
}

// Merges b into a: b's sum through TwoSum, then both errors.
__device__ void merge(CompensatedSum& a, float b_sum, float b_error) {
    const float errors = a.error + b_error;
    a.error = 0.0f;
    add(a, b_sum);
    a.error += errors;
}

// The total over the warp's lanes, received by every lane. At each butterfly step the
// two lanes of a pair add the same two values in swapped order, which gives the same
// bits, so every lane ends with the same total.
__device__ float warp_total(float v) {
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        v = __fadd_rn(v, __shfl_xor_sync(kAllLanes, v, offset));
    }
    return v;
}

// xhat, rounded as the forward pass rounds it.
__device__ float normalize(float x, float mean, float rstd) {
    return __fmul_rn(__fsub_rn(x, mean), rstd);
}

// The 8 FP16 values of a 16-byte piece, in FP32, which holds them exactly.
__device__ void unpack(float (&v)[kPieceValues], uint4 q) {
    const unsigned words[4] = {q.x, q.y, q.z, q.w};
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __half2float(__ushort_as_half(static_cast<unsigned short>(words[i])));
        v[2 * i + 1] = __half2float(__ushort_as_half(static_cast<unsigned short>(words[i] >> 16)));
    }
}

// The 8 FP32 values of the 32-byte piece at p.
__device__ void load(float (&v)[kPieceValues], const float4* __restrict__ p) {
    const float4 a = p[0];
    const float4 b = p[1];
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
    v[4] = b.x;
    v[5] = b.y;
    v[6] = b.z;
    v[7] = b.w;
}

// Warp w of block b: row 8 b + w of dx.
__global__ void __launch_bounds__(kWarp * kRowsPerBlock)
    layernorm_backward_dx(float4* __restrict__ dx, const uint4* __restrict__ dy,
                          const float4* __restrict__ x, const float* __restrict__ mean,
                          const float* __restrict__ rstd, const float4* __restrict__ gamma,
                          size_t rows, int pieces, bool accumulate) {
    const size_t row = size_t{blockIdx.x} * kRowsPerBlock + threadIdx.x / kWarp;
    // A warp's lanes share a row, so whole warps leave together and the shuffles below
    // always have all 32 lanes.
    if (row >= rows) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x % kWarp);
    const uint4* dy_row = dy + row * pieces;
    const float4* x_row = x + row * pieces * 2;
    float4* dx_row = dx + row * pieces * 2;
    const float row_mean = mean[row];
    const float row_rstd = rstd[row];

    float g_sum = 0.0f;
    float gx_sum = 0.0f;
    for (int i = lane; i < pieces; i += kWarp) {
        float d[kPieceValues];
        float xs[kPieceValues];
        float gs[kPieceValues];
        unpack(d, dy_row[i]);
        load(xs, x_row + 2 * i);
        load(gs, gamma + 2 * i);
        for (int j = 0; j < kPieceValues; ++j) {
            const float g = __fmul_rn(d[j], gs[j]);
            g_sum = __fadd_rn(g_sum, g);
            gx_sum = __fadd_rn(gx_sum, __fmul_rn(g, normalize(xs[j], row_mean, row_rstd)));
        }
    }
    const float n = static_cast<float>(pieces * kPieceValues);
    const float g_mean = __fdiv_rn(warp_total(g_sum), n);
    const float gx_mean = __fdiv_rn(warp_total(gx_sum), n);

    for (int i = lane; i < pieces; i += kWarp) {
        float d[kPieceValues];
        float xs[kPieceValues];
        float gs[kPieceValues];
        unpack(d, dy_row[i]);
        load(xs, x_row + 2 * i);
        load(gs, gamma + 2 * i);
        float out[kPieceValues];
        for (int j = 0; j < kPieceValues; ++j) {
            const float g = __fmul_rn(d[j], gs[j]);
            const float xhat = normalize(xs[j], row_mean, row_rstd);
            out[j] = __fmul_rn(row_rstd, __fsub_rn(__fsub_rn(g, g_mean), __fmul_rn(xhat, gx_mean)));
        }
        if (accumulate) {
            float old[kPieceValues];
            load(old, dx_row + 2 * i);
            for (int j = 0; j < kPieceValues; ++j) {
                out[j] = __fadd_rn(old[j], out[j]);
            }
        }
        dx_row[2 * i] = make_float4(out[0], out[1], out[2], out[3]);
        dx_row[2 * i + 1] = make_float4(out[4], out[5], out[6], out[7]);
    }
}

// Block (chunk, tile): for the tile's columns j, the chunk's compensated sum of
// dy[r, j] xhat[r, j] over its rows, the sums in the chunk's first row of partial and
// the errors in its second, as the bias gradient's first pass leaves them.
__global__ void __launch_bounds__(kTileX * kTileY)
    layernorm_backward_dgamma_partials(float4* __restrict__ partial, const uint4* __restrict__ dy,
                                       const float4* __restrict__ x,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ rstd, size_t rows, int pieces) {
    __shared__ float sums[kTileY][kTileX][kPieceValues];
    __shared__ float errors[kTileY][kTileX][kPieceValues];
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int piece = static_cast<int>(blockIdx.y) * kTileX + tx;
    const size_t first = size_t{blockIdx.x} * kChunkRows;
    const size_t end = rows - first < kChunkRows ? rows : first + kChunkRows;

    CompensatedSum sum[kPieceValues];
    if (piece < pieces) {
        for (size_t r = first + ty; r < end; r += kTileY) {
            const float row_mean = mean[r];
            const float row_rstd = rstd[r];
            const size_t at = r * pieces + piece;
            float d[kPieceValues];
            float xs[kPieceValues];
            unpack(d, dy[at]);
            load(xs, x + at * 2);
            for (int j = 0; j < kPieceValues; ++j) {
                add_product(sum[j], d[j], normalize(xs[j], row_mean, row_rstd));
            }
        }
    }
    for (int j = 0; j < kPieceValues; ++j) {
        sums[ty][tx][j] = sum[j].sum;
        errors[ty][tx][j] = sum[j].error;
    }
    __syncthreads();
    if (ty != 0 || piece >= pieces) {
        return;
    }
    for (int other = 1; other < kTileY; ++other) {
        for (int j = 0; j < kPieceValues; ++j) {
            merge(sum[j], sums[other][tx][j], errors[other][tx][j]);
        }
    }
    // A chunk's row of sums is pieces * 2 float4s, and its row of errors follows.
    float4* out = partial + size_t{blockIdx.x} * pieces * 4 + size_t{static_cast<unsigned>(piece)} * 2;
    out[0] = make_float4(sum[0].sum, sum[1].sum, sum[2].sum, sum[3].sum);
    out[1] = make_float4(sum[4].sum, sum[5].sum, sum[6].sum, sum[7].sum);
    out += size_t{static_cast<unsigned>(pieces)} * 2;
    out[0] = make_float4(sum[0].error, sum[1].error, sum[2].error, sum[3].error);
    out[1] = make_float4(sum[4].error, sum[5].error, sum[6].error, sum[7].error);
}

}  // namespace

cudaError_t launch_layernorm_backward(void* dx, void* dgamma, void* dbeta, const void* dy,
                                      const void* x, const void* mean, const void* rstd,
                                      const void* gamma, size_t rows, size_t cols,
                                      bool accumulate_dx, bool accumulate_params, void* workspace,
                                      cudaStream_t stream) {
    // dbeta first: the bias gradient is done with the workspace once its second pass
    // has run, and dgamma's partial sums take it next.
    if (const cudaError_t error = launch_bias_gradient(dbeta, dy, rows, cols, accumulate_params,
                                                       workspace, stream);
        error != cudaSuccess) {
        return error;
    }
    const auto pieces = static_cast<int>(cols / kPieceValues);
    const auto chunks = static_cast<unsigned>((rows + kChunkRows - 1) / kChunkRows);
    const auto tiles = static_cast<unsigned>((pieces + kTileX - 1) / kTileX);
    const auto* dy_q = static_cast<const uint4*>(dy);
    const auto* x_q = static_cast<const float4*>(x);
    const auto* mean_f = static_cast<const float*>(mean);
    const auto* rstd_f = static_cast<const float*>(rstd);
    layernorm_backward_dgamma_partials<<<dim3(chunks, tiles), dim3(kTileX, kTileY), 0, stream>>>(
        static_cast<float4*>(workspace), dy_q, x_q, mean_f, rstd_f, rows, pieces);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    if (const cudaError_t error = launch_bias_gradient_finish(dgamma, workspace, rows, cols,
                                                              accumulate_params, stream);
        error != cudaSuccess) {
        return error;
    }
    const auto blocks = static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    layernorm_backward_dx<<<blocks, kWarp * kRowsPerBlock, 0, stream>>>(
        static_cast<float4*>(dx), dy_q, x_q, mean_f, rstd_f, static_cast<const float4*>(gamma),
        rows, pieces, accumulate_dx);
    return cudaGetLastError();
}

}  // namespace flint
