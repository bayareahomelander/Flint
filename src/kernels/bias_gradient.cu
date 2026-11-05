// A linear layer's bias gradient, dbias = the column sums of dy, in two passes
// with no atomics, so the result is the same bits on every run.
//
// The sums are compensated, as LayerNorm's and the cross-entropy's are (TwoSum,
// after Ogita, Rump and Oishi's Sum2): each keeps its rounded value and the sum of
// the rounding errors left behind in it, and only the final total adds the two.
// With plain FP32 sums, a column of outliers near 3,000 (8,453 rows) missed the
// owner's bound of twice FP32 PyTorch's own error against FP64.
//
// The first pass gives each block a chunk of 256 rows and a tile of 256 columns.
// Its 32 x 8 threads read 16-byte pieces of 8 FP16 values, a warp covering 512
// contiguous bytes of one row. Thread (x, y) adds rows y, y + 8, ... of the chunk
// in order, then thread (x, 0) merges the other 7 threads' sums into its own in
// order of y, and the block stores the chunk's sums and errors, two FP32 rows per
// chunk, in the workspace.
//
// The second pass gives each warp 4 columns. Lane l merges chunks l, l + 32, ... in
// order, and a butterfly over the lanes merges those. At each butterfly step the
// two lanes of a pair hold the same two values in swapped order; TwoSum's error is
// exact and FP32 addition is commutative, so every lane ends with the same bits.
// Lane 0 stores sum + error, or adds that to dbias with one FP32 add.
//
// The chunks, tiles and orders depend only on rows and cols, so a GPU's SM count
// changes nothing. inf and NaN reach the column's sum, as they reach PyTorch's, and
// a non-finite sum is stored without its error, which is then NaN (see value).
#include "kernels/bias_gradient.cuh"

#include <cuda_fp16.h>

namespace flint {
namespace {

constexpr int kWarp = 32;
constexpr unsigned kAllLanes = 0xffffffffu;
// dy moves in 16-byte pieces of 8 FP16 values.
constexpr int kPieceValues = 8;
// The first pass's block: kTileX pieces of a row across, kTileY rows down.
constexpr int kTileX = kWarp;
constexpr int kTileY = 8;
constexpr int kTileCols = kTileX * kPieceValues;
constexpr int kChunkRows = static_cast<int>(kBiasGradientChunkRows);
static_assert(kTileCols == 256, "bias_gradient.cuh's column limit assumes 256-column tiles");
// The second pass's block: 8 warps, each taking 4 columns.
constexpr int kFinishWarps = 8;
constexpr int kFinishCols = 4;

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

__device__ void add_piece(CompensatedSum (&sum)[kPieceValues], uint4 q) {
    const unsigned words[4] = {q.x, q.y, q.z, q.w};
    for (int i = 0; i < 4; ++i) {
        add(sum[2 * i], __half2float(__ushort_as_half(static_cast<unsigned short>(words[i]))));
        add(sum[2 * i + 1], __half2float(__ushort_as_half(static_cast<unsigned short>(words[i] >> 16))));
    }
}

// Block (chunk, tile): for the tile's columns j, the chunk's compensated sum of
// dy[r, j] over its rows, the sums in the chunk's first row of partial and the
// errors in its second.
__global__ void __launch_bounds__(kTileX * kTileY)
    bias_gradient_partials(float4* __restrict__ partial, const uint4* __restrict__ dy, size_t rows,
                           int pieces) {
    __shared__ float sums[kTileY][kTileX][kPieceValues];
    __shared__ float errors[kTileY][kTileX][kPieceValues];
    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    const int piece = static_cast<int>(blockIdx.y) * kTileX + x;
    const size_t first = size_t{blockIdx.x} * kChunkRows;
    const size_t end = rows - first < kChunkRows ? rows : first + kChunkRows;

    CompensatedSum sum[kPieceValues];
    if (piece < pieces) {
        for (size_t r = first + y; r < end; r += kTileY) {
            add_piece(sum, dy[r * pieces + piece]);
        }
    }
    for (int i = 0; i < kPieceValues; ++i) {
        sums[y][x][i] = sum[i].sum;
        errors[y][x][i] = sum[i].error;
    }
    __syncthreads();
    if (y != 0 || piece >= pieces) {
        return;
    }
    for (int other = 1; other < kTileY; ++other) {
        for (int i = 0; i < kPieceValues; ++i) {
            merge(sum[i], sums[other][x][i], errors[other][x][i]);
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

// The sum's value: sum + error. A non-finite sum is the value itself: once an inf has
// passed through TwoSum, its error is inf - inf = NaN, which would turn a +inf column
// into NaN. A finite sum always has a finite error, since sums of FP16 values stay far
// below FP32's overflow.
__device__ float value(const CompensatedSum& a) {
    return isfinite(a.sum) ? a.sum + a.error : a.sum;
}

// Merges b into a's 4 columns.
__device__ void merge4(CompensatedSum (&a)[kFinishCols], float4 b_sum, float4 b_error) {
    merge(a[0], b_sum.x, b_error.x);
    merge(a[1], b_sum.y, b_error.y);
    merge(a[2], b_sum.z, b_error.z);
    merge(a[3], b_sum.w, b_error.w);
}

// Warp w of block b: dbias = the merged chunks' sums, for columns 4 (8 b + w) to
// 4 (8 b + w) + 3.
__global__ void __launch_bounds__(kWarp * kFinishWarps)
    bias_gradient_finish(float4* __restrict__ dbias, const float4* __restrict__ partial,
                         size_t chunks, int quads, bool accumulate) {
    const int quad = static_cast<int>(blockIdx.x) * kFinishWarps + static_cast<int>(threadIdx.x) / kWarp;
    // A warp's lanes share a quad, so whole warps leave together and the
    // shuffles below always have all 32 lanes.
    if (quad >= quads) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    CompensatedSum total[kFinishCols];
    for (size_t chunk = lane; chunk < chunks; chunk += kWarp) {
        const float4* row = partial + chunk * quads * 2;
        merge4(total, row[quad], row[quads + quad]);
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        float4 other_sum;
        float4 other_error;
        other_sum.x = __shfl_xor_sync(kAllLanes, total[0].sum, offset);
        other_sum.y = __shfl_xor_sync(kAllLanes, total[1].sum, offset);
        other_sum.z = __shfl_xor_sync(kAllLanes, total[2].sum, offset);
        other_sum.w = __shfl_xor_sync(kAllLanes, total[3].sum, offset);
        other_error.x = __shfl_xor_sync(kAllLanes, total[0].error, offset);
        other_error.y = __shfl_xor_sync(kAllLanes, total[1].error, offset);
        other_error.z = __shfl_xor_sync(kAllLanes, total[2].error, offset);
        other_error.w = __shfl_xor_sync(kAllLanes, total[3].error, offset);
        merge4(total, other_sum, other_error);
    }
    if (lane != 0) {
        return;
    }
    float4 result = make_float4(value(total[0]), value(total[1]), value(total[2]), value(total[3]));
    if (accumulate) {
        const float4 old = dbias[quad];
        result = make_float4(old.x + result.x, old.y + result.y, old.z + result.z, old.w + result.w);
    }
    dbias[quad] = result;
}

}  // namespace

cudaError_t launch_bias_gradient(void* dbias, const void* dy, size_t rows, size_t cols,
                                 bool accumulate, void* workspace, cudaStream_t stream) {
    const auto chunks = static_cast<unsigned>((rows + kBiasGradientChunkRows - 1) /
                                              kBiasGradientChunkRows);
    const auto pieces = static_cast<int>(cols / kPieceValues);
    const auto tiles = static_cast<unsigned>((pieces + kTileX - 1) / kTileX);
    bias_gradient_partials<<<dim3(chunks, tiles), dim3(kTileX, kTileY), 0, stream>>>(
        static_cast<float4*>(workspace), static_cast<const uint4*>(dy), rows, pieces);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    const auto quads = static_cast<int>(cols / kFinishCols);
    const auto blocks = static_cast<unsigned>((quads + kFinishWarps - 1) / kFinishWarps);
    bias_gradient_finish<<<blocks, kWarp * kFinishWarps, 0, stream>>>(
        static_cast<float4*>(dbias), static_cast<const float4*>(workspace), chunks, quads,
        accumulate);
    return cudaGetLastError();
}

}  // namespace flint
