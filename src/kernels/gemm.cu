// The GEMM on FP16 tensor cores with FP32 accumulation, in a linear layer's
// three layouts: y = x w^T with w n x k (the forward pass), y = x w with w
// k x n (the input gradient, dX = dY W), and y = x^T w with x k x m and w
// k x n (the weight gradient, dW = dY^T X), which writes or adds into an FP32
// y and splits k across blocks when the output has few tiles. The forward
// layout can also add a bias to its outputs before they round, and then apply
// GELU or add the result to the FP32 residual stream, and the input-gradient
// layout can multiply its outputs by GELU's derivative, GELU's backward pass.
// This first version is plain: one shared-memory stage with padded rows, and
// no overlap of loads with math. The tuning deliverables build on it.
#include "kernels/gemm.cuh"

#include <algorithm>
#include <cstdint>

#include <cuda_fp16.h>

namespace flint {
namespace {

// A block computes a 128 x 128 tile of y, stepping through k 32 at a time.
// Its 8 warps sit in a 2 x 4 grid, each computing 64 x 32 of the tile as
// 4 x 4 mma tiles of 16 x 8.
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;
constexpr int kWarpM = 64;
constexpr int kWarpN = 32;
constexpr int kWarpsN = kBlockN / kWarpN;
constexpr int kThreads = 32 * (kBlockM / kWarpM) * kWarpsN;
constexpr int kTilesM = kWarpM / 16;
constexpr int kTilesN = kWarpN / 8;

// Shared rows of x's tile, and of w's in the NT layout, hold 32 halves of k
// plus 8 of padding. The 80-byte stride puts the 8 rows that each ldmatrix
// phase reads in 8 different groups of banks.
constexpr int kRowStride = kBlockK + 8;
// Where a tile's rows are k (w in NN; x and w in TN), it has 32 rows, each 128
// halves plus 8 of padding; the 272-byte stride does the same.
constexpr int kNnRowStride = kBlockN + 8;
// Both operands' tiles load in 16-byte chunks, 8 halves each.
constexpr int kChunksPerRow = kBlockK / 8;
constexpr int kChunksPerThread = kBlockM * kChunksPerRow / kThreads;
constexpr int kNnChunksPerRow = kBlockN / 8;
// A thread's chunks in a k-row tile are this many rows apart.
constexpr int kNnRowsPerPass = kThreads / kNnChunksPerRow;
static_assert(kBlockM == kBlockN, "the two operands' tiles share one loading scheme");
static_assert(kBlockM * kChunksPerRow % kThreads == 0, "every thread loads whole chunks");
static_assert(kBlockK * kNnChunksPerRow == kBlockN * kChunksPerRow,
              "a tile has as many chunks in either orientation");

// How x and w are stored: x m x k and w n x k for y = x w^T, x m x k and w
// k x n for y = x w, and x k x m and w k x n for y = x^T w.
enum class Layout { kNT, kNN, kTN };

// What the output stage does to the FP32 sums before they round to FP16 y:
// nothing; add an FP32 bias to each column; add the bias, keep that sum as
// GELU's input for the backward pass, and apply GELU; in the backward pass,
// multiply each sum by GELU's derivative at that stored input; or add the bias
// and then the FP32 residual stream, into an FP32 y that never rounds to FP16.
enum class Epilogue { kNone, kBias, kBiasGelu, kDGelu, kBiasResidual };

// TN's split of k, and where its output goes. The grid's z holds `groups`
// groups of rows of tiles for each split in turn.
struct TnArgs {
    int split_len = 0;          // k per split, a multiple of 32
    int groups = 1;             // groups of rows of tiles per split
    float* partials = nullptr;  // with several splits, each split's m x n sums
    int accumulate = 0;         // with one split, add into y rather than write it
};

// Blocks start roughly in order of their index, x fastest, so the order tiles
// are handed out in decides what L2 holds. Row by row, each wave would read a
// new stretch of w, and a w bigger than L2 (the logits' 77 MB) would be read
// from memory again for every row of tiles. Instead, tiles go out in groups of
// kGroupM rows of tiles, column by column within a group, so the panels of x
// and w that a wave reads stay in L2 for the whole group. The grid itself has
// that order: x is the row within a group, y the column, z the group. A tile
// index computed from a linear one would need registers that the main loop
// can't spare, and on sm_75 costs 10-15% even with no regrouping.
constexpr int kGroupM = 8;

__device__ uint32_t shared_address(const void* pointer) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// Loads four 8 x 8 FP16 matrices from shared memory. Lane i gives the address
// of row i % 8 of matrix i / 8, and register j receives matrix j: in each
// thread, the pair of halves at row lane / 4, columns 2 * (lane % 4) and the
// one after it.
__device__ void ldmatrix_x4(uint32_t (&r)[4], const __half* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(shared_address(row))
                 : "memory");
}

// The same, transposed: register j receives, in each thread, the pair at
// rows 2 * (lane % 4) and the one after it, column lane / 4.
__device__ void ldmatrix_x4_trans(uint32_t (&r)[4], const __half* row) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(shared_address(row))
                 : "memory");
}

// c += a b for a 16 x 8 tile of c, with a 16 x 8 and b 8 x 8. With g = lane / 4
// and t = lane % 4: a0 holds a[g][2t..2t+1] and a1 holds a[g+8][2t..2t+1];
// b holds b[2t..2t+1][g]; c holds c[g][2t..2t+1], then c[g+8][2t..2t+1].
__device__ void mma_m16n8k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b));
}

// Addressing is relative to the block's corner of each matrix (for a matrix
// whose rows are k, the corner of the current 32 rows). Rows within a tile are
// under 128, m and n are at most 8,388,480 and k at most 65,536, so offsets
// from a corner fit an int; 64-bit offsets per thread spill on sm_75.

// The 16-byte chunk at (row, col) from a block's corner of NT's or NN's x, or
// NT's w, which has `rows` rows of k halves left, or zeros past the matrix. k
// is a multiple of 8, so a chunk is either wholly inside or wholly outside.
__device__ uint4 load_chunk(const __half* __restrict__ corner, int row, int rows, int col,
                            int k) {
    if (row < rows && col < k) {
        return *reinterpret_cast<const uint4*>(corner + (row * k + col));
    }
    return make_uint4(0, 0, 0, 0);
}

// The same for a matrix whose rows are k (NN's w; TN's x and w), with `rows`
// rows of k and `cols` columns left from the corner, and rows n long. The
// columns are a multiple of 8 too.
__device__ uint4 load_chunk_nn(const __half* __restrict__ corner, int row, int rows, int col,
                               int cols, int n) {
    if (row < rows && col < cols) {
        return *reinterpret_cast<const uint4*>(corner + (row * n + col));
    }
    return make_uint4(0, 0, 0, 0);
}

// Rounds two adjacent outputs to FP16 and stores those inside y, at (row, col)
// from the block's corner, which has `rows` rows and `cols` columns left. With
// n even, the corner, every row and col are even, so the pair is one aligned
// store; with n odd, rows are only 2-byte aligned and the halves go one at a
// time.
__device__ void store_pair(__half* __restrict__ corner, int row, int col, int rows, int cols,
                           int n, float v0, float v1) {
    if (row >= rows || col >= cols) {
        return;
    }
    __half* out = corner + (row * n + col);
    if (n % 2 == 0) {
        *reinterpret_cast<__half2*>(out) = __floats2half2_rn(v0, v1);
    } else {
        out[0] = __float2half_rn(v0);
        if (col + 1 < cols) {
            out[1] = __float2half_rn(v1);
        }
    }
}

// Reads the two FP16 values at (row, col) from a block's corner of an m x n
// matrix, which has `rows` rows and `cols` columns left, as store_pair writes
// them. Only NN reads one, and its n is a multiple of 8, so col < cols means
// the pair fits and is 4-byte aligned. Outside the matrix it gives zeros,
// which store_pair then doesn't store.
__device__ float2 load_pair(const __half* __restrict__ corner, int row, int col, int rows,
                            int cols, int n) {
    if (row >= rows || col >= cols) {
        return make_float2(0.0f, 0.0f);
    }
    return __half22float2(*reinterpret_cast<const __half2*>(corner + (row * n + col)));
}

// Stores two adjacent FP32 outputs inside y, or with `add`, adds them to what
// is there (old + sum, one rounding each). TN's m and n are multiples of 8, so
// col < cols means the pair fits, and it's 8-byte aligned.
__device__ void store_pair_f32(float* __restrict__ corner, int row, int col, int rows, int cols,
                               int n, float v0, float v1, bool add) {
    if (row >= rows || col >= cols) {
        return;
    }
    float2* out = reinterpret_cast<float2*>(corner + (row * n + col));
    if (add) {
        const float2 old = *out;
        v0 = old.x + v0;
        v1 = old.y + v1;
    }
    *out = make_float2(v0, v1);
}

// GPT-2's tanh GELU, 0.5 x (1 + tanh(u)) with u = sqrt(2/pi) (x + 0.044715 x^3),
// computed as x / (1 + exp(-2u)): the same function, without the cancellation
// in 1 + tanh(u) that costs the tanh form its relative accuracy for negative
// x. +inf gives +inf and -inf gives NaN, as the tanh form does, and a large
// negative x gives -0.
__device__ __forceinline__ float gelu(float x) {
    constexpr float kMinus2Sqrt2OverPi = -1.5957691216f;  // -2 sqrt(2/pi)
    const float e = expf(kMinus2Sqrt2OverPi * x * fmaf(0.044715f, x * x, 1.0f));
    return __fdividef(x, 1.0f + e);
}

// GELU's derivative, for the backward pass. With sigma = 1 / (1 + exp(-2u)),
// GELU is x sigma, as above, so its derivative is sigma + x sigma (1 - sigma)
// 2u', with 2u' = 2 sqrt(2/pi) (1 + 3 * 0.044715 x^2), computed as one fma
// from sigma. For x >= 0, 1 - sigma is exp(-2u) sigma, which keeps its
// relative accuracy as sigma nears 1; for negative x, sigma is under 1/2, so
// 1 - sigma is exact enough as it is, and exp(-2u) can overflow. The
// derivative crosses 0 at x = -0.7525, where, as with any FP32 formula, its
// error is a few FP32 ulps of sigma (about 1/4 there), not of itself. An
// infinite x takes GELU's limits, 0 at -inf and 1 at +inf, where the formula,
// like PyTorch's tanh form, gives NaN; NaN gives NaN.
__device__ __forceinline__ float gelu_grad(float x) {
    constexpr float kMinus2Sqrt2OverPi = -1.5957691216f;  // as in gelu
    constexpr float k2Sqrt2OverPi = 1.5957691216f;
    constexpr float k6KappaSqrt2OverPi = 0.214064449f;  // 2 sqrt(2/pi) * 3 * 0.044715
    const float e = expf(kMinus2Sqrt2OverPi * x * fmaf(0.044715f, x * x, 1.0f));
    const float sigma = __fdividef(1.0f, 1.0f + e);
    const float rest = x < 0.0f ? 1.0f - sigma : e * sigma;  // 1 - sigma
    const float q = x * rest * fmaf(k6KappaSqrt2OverPi, x * x, k2Sqrt2OverPi);
    const float grad = fmaf(sigma, q, sigma);
    return isinf(x) ? (x > 0.0f ? 1.0f : 0.0f) : grad;
}

// The epilogue's work on a thread's FP32 sums before they're stored. The
// thread holds rows row0 + 16 i and the row 8 below it, in its pairs of
// columns col0 + 8 j and the one after, of the tile at (block_m, block_n).
// aux is an epilogue's second m x n buffer: GELU's input, which the forward
// stores and the backward reads, or the residual stream. The plain one does
// nothing. It's an empty specialization, called unconditionally, because
// epilogue code in the body, even in a discarded `if constexpr`, changes how
// the plain kernels compile.
template <Epilogue E>
__device__ __forceinline__ void apply_epilogue(float (&)[kTilesM][kTilesN][4],
                                               const float* __restrict__, void* __restrict__, int,
                                               int, int, int, int, int) {}

// The bias epilogue adds bias[col] to every sum in column col. It reads the
// bias only now, after the main loop, which has no registers to hold it.
// (block_n is added here, not by the caller: the plain kernels' bias is null.)
template <>
__device__ __forceinline__ void apply_epilogue<Epilogue::kBias>(
    float (&acc)[kTilesM][kTilesN][4], const float* __restrict__ bias, void* __restrict__, int,
    int block_n, int, int col0, int, int n) {
    bias += block_n;
    const int cols = n - block_n;
#pragma unroll
    for (int j = 0; j < kTilesN; ++j) {
        const int col = col0 + j * 8;
        const float b0 = col < cols ? bias[col] : 0.0f;
        const float b1 = col + 1 < cols ? bias[col + 1] : 0.0f;
#pragma unroll
        for (int i = 0; i < kTilesM; ++i) {
            acc[i][j][0] += b0;
            acc[i][j][1] += b1;
            acc[i][j][2] += b0;
            acc[i][j][3] += b1;
        }
    }
}

// Stores a thread's outputs in y (m x n, FP16 unless the epilogue says
// otherwise): the epilogue's work on the sums, then each pair rounded to FP16.
template <Epilogue E>
__device__ __forceinline__ void store_outputs(float (&acc)[kTilesM][kTilesN][4],
                                              void* __restrict__ y,
                                              const float* __restrict__ bias,
                                              void* __restrict__ aux, int block_m, int block_n,
                                              int row0, int col0, int m, int n) {
    __half* y_corner = static_cast<__half*>(y) + size_t(block_m) * n + block_n;
    apply_epilogue<E>(acc, bias, aux, block_m, block_n, row0, col0, m, n);
#pragma unroll
    for (int i = 0; i < kTilesM; ++i) {
#pragma unroll
        for (int j = 0; j < kTilesN; ++j) {
            const int row = row0 + i * 16;
            const int col = col0 + j * 8;
            store_pair(y_corner, row, col, m - block_m, n - block_n, n, acc[i][j][0],
                       acc[i][j][1]);
            store_pair(y_corner, row + 8, col, m - block_m, n - block_n, n, acc[i][j][2],
                       acc[i][j][3]);
        }
    }
}

// The GELU epilogue adds the bias, then for each mma tile stores the sums
// rounded to FP16 in pre (aux, GELU's input, which its backward pass needs)
// and GELU of the unrounded sums in y, so each output rounds once. Doing both
// a tile at a time matters on sm_75: storing all of pre before applying GELU
// rescheduled the main loop, with more waits on ldmatrix before the HMMAs, and
// made it 30% slower at large k.
template <>
__device__ __forceinline__ void store_outputs<Epilogue::kBiasGelu>(
    float (&acc)[kTilesM][kTilesN][4], void* __restrict__ y, const float* __restrict__ bias,
    void* __restrict__ aux, int block_m, int block_n, int row0, int col0, int m, int n) {
    __half* y_corner = static_cast<__half*>(y) + size_t(block_m) * n + block_n;
    apply_epilogue<Epilogue::kBias>(acc, bias, aux, block_m, block_n, row0, col0, m, n);
    __half* pre_corner = static_cast<__half*>(aux) + size_t(block_m) * n + block_n;
#pragma unroll
    for (int i = 0; i < kTilesM; ++i) {
#pragma unroll
        for (int j = 0; j < kTilesN; ++j) {
            const int row = row0 + i * 16;
            const int col = col0 + j * 8;
            float* c = acc[i][j];
            store_pair(pre_corner, row, col, m - block_m, n - block_n, n, c[0], c[1]);
            store_pair(pre_corner, row + 8, col, m - block_m, n - block_n, n, c[2], c[3]);
            store_pair(y_corner, row, col, m - block_m, n - block_n, n, gelu(c[0]), gelu(c[1]));
            store_pair(y_corner, row + 8, col, m - block_m, n - block_n, n, gelu(c[2]),
                       gelu(c[3]));
        }
    }
}

// GELU's backward epilogue multiplies each sum by GELU's derivative at pre
// (aux, the forward's stored input, read-only here), and stores the products
// in y, so each rounds once. Like the GELU epilogue, it works a tile at a time.
template <>
__device__ __forceinline__ void store_outputs<Epilogue::kDGelu>(
    float (&acc)[kTilesM][kTilesN][4], void* __restrict__ y, const float* __restrict__,
    void* __restrict__ aux, int block_m, int block_n, int row0, int col0, int m, int n) {
    __half* y_corner = static_cast<__half*>(y) + size_t(block_m) * n + block_n;
    const __half* pre_corner = static_cast<const __half*>(aux) + size_t(block_m) * n + block_n;
#pragma unroll
    for (int i = 0; i < kTilesM; ++i) {
#pragma unroll
        for (int j = 0; j < kTilesN; ++j) {
            const int row = row0 + i * 16;
            const int col = col0 + j * 8;
            const float* c = acc[i][j];
            const float2 p0 = load_pair(pre_corner, row, col, m - block_m, n - block_n, n);
            const float2 p1 = load_pair(pre_corner, row + 8, col, m - block_m, n - block_n, n);
            store_pair(y_corner, row, col, m - block_m, n - block_n, n, c[0] * gelu_grad(p0.x),
                       c[1] * gelu_grad(p0.y));
            store_pair(y_corner, row + 8, col, m - block_m, n - block_n, n,
                       c[2] * gelu_grad(p1.x), c[3] * gelu_grad(p1.y));
        }
    }
}

// The residual epilogue adds the bias, then adds each sum to the residual
// stream (aux, FP32 m x n, read-only here) and stores the result in FP32 y:
// y = residual + (sum + bias), one FP32 add each, with nothing rounded to
// FP16. n is even, so the corner, every row and col are even, and col < cols
// means a pair fits as one aligned 8-byte access. It works a tile at a time,
// reading both of the tile's pairs before storing either: on sm_75, storing
// each pair right after reading it reordered the main loop's address
// arithmetic and cost 10% at large k, and so did paths for an odd n.
template <>
__device__ __forceinline__ void store_outputs<Epilogue::kBiasResidual>(
    float (&acc)[kTilesM][kTilesN][4], void* __restrict__ y, const float* __restrict__ bias,
    void* __restrict__ aux, int block_m, int block_n, int row0, int col0, int m, int n) {
    const size_t corner = size_t(block_m) * n + block_n;
    float* y_corner = static_cast<float*>(y) + corner;
    const float* residual_corner = static_cast<const float*>(aux) + corner;
    apply_epilogue<Epilogue::kBias>(acc, bias, aux, block_m, block_n, row0, col0, m, n);
    const int rows = m - block_m;
    const int cols = n - block_n;
#pragma unroll
    for (int i = 0; i < kTilesM; ++i) {
#pragma unroll
        for (int j = 0; j < kTilesN; ++j) {
            const int row = row0 + i * 16;
            const int col = col0 + j * 8;
            float* c = acc[i][j];
            const bool in0 = row < rows && col < cols;
            const bool in1 = row + 8 < rows && col < cols;
            float2 r0 = make_float2(0.0f, 0.0f);
            float2 r1 = make_float2(0.0f, 0.0f);
            if (in0) {
                r0 = *reinterpret_cast<const float2*>(residual_corner + (row * n + col));
            }
            if (in1) {
                r1 = *reinterpret_cast<const float2*>(residual_corner + ((row + 8) * n + col));
            }
            if (in0) {
                *reinterpret_cast<float2*>(y_corner + (row * n + col)) =
                    make_float2(r0.x + c[0], r0.y + c[1]);
            }
            if (in1) {
                *reinterpret_cast<float2*>(y_corner + ((row + 8) * n + col)) =
                    make_float2(r1.x + c[2], r1.y + c[3]);
            }
        }
    }
}

// The body of every kernel, which differ in how x and w are stored, in TN's
// FP32 output and split of k, and in the epilogue applied to the outputs of
// the others. bias and aux are the epilogue's (see apply_epilogue).
template <Layout L, Epilogue E>
__device__ __forceinline__ void gemm(void* __restrict__ y, const __half* __restrict__ x,
                                     const __half* __restrict__ w, int m, int n, int k,
                                     TnArgs tn, const float* __restrict__ bias,
                                     void* __restrict__ aux) {
    constexpr bool nt = L == Layout::kNT;
    constexpr bool tn_layout = L == Layout::kTN;
    static_assert(!tn_layout || E == Epilogue::kNone, "TN's FP32 output has no epilogue");
    // NT and NN: 128 rows of m, each 32 halves of k. TN: 32 rows of k, each
    // 128 of m.
    __shared__ alignas(16)
        __half x_tile[tn_layout ? kBlockK : kBlockM][tn_layout ? kNnRowStride : kRowStride];
    // NT: 128 rows of n, each 32 halves of k. NN and TN: 32 rows of k, each 128
    // of n.
    __shared__ alignas(16) __half w_tile[nt ? kBlockN : kBlockK][nt ? kRowStride : kNnRowStride];

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warp_m = warp / kWarpsN * kWarpM;
    const int warp_n = warp % kWarpsN * kWarpN;
    // TN's z holds the groups of each split of k in turn.
    [[maybe_unused]] int split = 0;
    int group = static_cast<int>(blockIdx.z);
    if constexpr (tn_layout) {
        split = group / tn.groups;
        group -= split * tn.groups;
    }
    // A group has gridDim.x rows of tiles, kGroupM or fewer if m has fewer.
    // The last group's rows past m have no tile; the whole block leaves
    // before its first barrier.
    const int block_m = (group * gridDim.x + blockIdx.x) * kBlockM;
    const int block_n = blockIdx.y * kBlockN;
    if (block_m >= m) {
        return;
    }
    // The block's stretch of k: all of it, or TN's split.
    const int k_begin = tn_layout ? split * tn.split_len : 0;
    const int k_end = tn_layout ? min(k, k_begin + tn.split_len) : k;
    [[maybe_unused]] const __half* x_corner = x + size_t(block_m) * k;
    // NT: the block's first row of w. (NN and TN find their corners of a
    // k-row matrix afresh each step: a pointer kept across steps is one
    // register too many.)
    [[maybe_unused]] const __half* w_corner = w + size_t(block_n) * k;
    // In a tile whose rows are k, each thread's chunks sit at rows nn_row and
    // nn_row + 16 of the step's 32 rows of k, at column nn_col.
    [[maybe_unused]] const int nn_row = threadIdx.x / kNnChunksPerRow;
    [[maybe_unused]] const int nn_col = threadIdx.x % kNnChunksPerRow * 8;

    float acc[kTilesM][kTilesN][4] = {};

    for (int k0 = k_begin; k0 < k_end; k0 += kBlockK) {
        // Chunks past m, n or k read as zeros, which add nothing to the sums.
        uint4 x_chunks[kChunksPerThread];
        uint4 w_chunks[kChunksPerThread];
#pragma unroll
        for (int i = 0; i < kChunksPerThread; ++i) {
            const int chunk = threadIdx.x + i * kThreads;
            [[maybe_unused]] const int row = chunk / kChunksPerRow;
            [[maybe_unused]] const int col = k0 + chunk % kChunksPerRow * 8;
            if constexpr (tn_layout) {
                x_chunks[i] = load_chunk_nn(x + (size_t(k0) * m + block_m),
                                            nn_row + i * kNnRowsPerPass, k_end - k0, nn_col,
                                            m - block_m, m);
            } else {
                x_chunks[i] = load_chunk(x_corner, row, m - block_m, col, k);
            }
            if constexpr (nt) {
                w_chunks[i] = load_chunk(w_corner, row, n - block_n, col, k);
            } else {
                w_chunks[i] = load_chunk_nn(w + (size_t(k0) * n + block_n),
                                            nn_row + i * kNnRowsPerPass, k_end - k0, nn_col,
                                            n - block_n, n);
            }
        }
        __syncthreads();  // every warp is done reading the previous tiles
#pragma unroll
        for (int i = 0; i < kChunksPerThread; ++i) {
            const int chunk = threadIdx.x + i * kThreads;
            [[maybe_unused]] const int row = chunk / kChunksPerRow;
            [[maybe_unused]] const int col = chunk % kChunksPerRow * 8;
            if constexpr (tn_layout) {
                *reinterpret_cast<uint4*>(&x_tile[nn_row + i * kNnRowsPerPass][nn_col]) =
                    x_chunks[i];
            } else {
                *reinterpret_cast<uint4*>(&x_tile[row][col]) = x_chunks[i];
            }
            if constexpr (nt) {
                *reinterpret_cast<uint4*>(&w_tile[row][col]) = w_chunks[i];
            } else {
                *reinterpret_cast<uint4*>(&w_tile[nn_row + i * kNnRowsPerPass][nn_col]) =
                    w_chunks[i];
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
            // Each ldmatrix covers 16 of k, two mma steps. For x, matrices
            // 0-3 are rows 0-7 and 8-15 at k 0-7, then the same at k 8-15,
            // which makes registers 0-1 the first step's a0, a1 and 2-3 the
            // second's. TN's x tile has rows of k, so the transposed load
            // gives each thread a's pair of k at one row.
            uint32_t a[kTilesM][4];
#pragma unroll
            for (int i = 0; i < kTilesM; ++i) {
                if constexpr (tn_layout) {
                    ldmatrix_x4_trans(a[i], &x_tile[kk + lane / 16 * 8 + lane % 8]
                                                   [warp_m + i * 16 + lane / 8 % 2 * 8]);
                } else {
                    ldmatrix_x4(a[i], &x_tile[warp_m + i * 16 + lane % 16][kk + lane / 16 * 8]);
                }
            }
            // For w, matrices 0-3 are tile j at k 0-7 and k 8-15, then tile
            // j + 1 the same way. NT's rows are n, so a plain load gives each
            // thread b's pair of k at one n. NN's rows are k, so the
            // transposed load gives the same pair.
            uint32_t b[kTilesN][2];
#pragma unroll
            for (int j = 0; j < kTilesN; j += 2) {
                uint32_t r[4];
                if constexpr (nt) {
                    ldmatrix_x4(r, &w_tile[warp_n + j * 8 + lane / 16 * 8 + lane % 8]
                                          [kk + lane / 8 % 2 * 8]);
                } else {
                    ldmatrix_x4_trans(r, &w_tile[kk + lane / 8 % 2 * 8 + lane % 8]
                                                [warp_n + j * 8 + lane / 16 * 8]);
                }
                b[j][0] = r[0];
                b[j][1] = r[1];
                b[j + 1][0] = r[2];
                b[j + 1][1] = r[3];
            }
#pragma unroll
            for (int step = 0; step < 2; ++step) {
#pragma unroll
                for (int i = 0; i < kTilesM; ++i) {
#pragma unroll
                    for (int j = 0; j < kTilesN; ++j) {
                        mma_m16n8k8(acc[i][j], a[i][2 * step], a[i][2 * step + 1], b[j][step]);
                    }
                }
            }
        }
    }

    const int g = lane / 4;
    const int t = lane % 4;
    if constexpr (tn_layout) {
        // One split writes y, or adds into it; several write their own sums,
        // which gemm_tn_reduce adds up.
        float* out = tn.partials ? tn.partials + size_t(split) * m * n : static_cast<float*>(y);
        float* y_corner = out + size_t(block_m) * n + block_n;
        const bool add = tn.partials == nullptr && tn.accumulate;
#pragma unroll
        for (int i = 0; i < kTilesM; ++i) {
#pragma unroll
            for (int j = 0; j < kTilesN; ++j) {
                const int row = warp_m + i * 16 + g;
                const int col = warp_n + j * 8 + 2 * t;
                store_pair_f32(y_corner, row, col, m - block_m, n - block_n, n, acc[i][j][0],
                               acc[i][j][1], add);
                store_pair_f32(y_corner, row + 8, col, m - block_m, n - block_n, n,
                               acc[i][j][2], acc[i][j][3], add);
            }
        }
    } else {
        store_outputs<E>(acc, y, bias, aux, block_m, block_n, warp_m + g, warp_n + 2 * t, m, n);
    }
}

// The kernels get their own names, so SASS, profiles and sanitizer reports
// tell the layouts and epilogues apart.
__global__ void __launch_bounds__(kThreads, 2)
    gemm_nt(__half* __restrict__ y, const __half* __restrict__ x, const __half* __restrict__ w,
            int m, int n, int k) {
    gemm<Layout::kNT, Epilogue::kNone>(y, x, w, m, n, k, TnArgs{}, nullptr, nullptr);
}

// The epilogues' arguments come last, so the others sit where gemm_nt has
// them, and the main loop compiles to the same instructions.
__global__ void __launch_bounds__(kThreads, 2)
    gemm_nt_bias(__half* __restrict__ y, const __half* __restrict__ x,
                 const __half* __restrict__ w, int m, int n, int k,
                 const float* __restrict__ bias) {
    gemm<Layout::kNT, Epilogue::kBias>(y, x, w, m, n, k, TnArgs{}, bias, nullptr);
}

__global__ void __launch_bounds__(kThreads, 2)
    gemm_nt_bias_gelu(__half* __restrict__ y, const __half* __restrict__ x,
                      const __half* __restrict__ w, int m, int n, int k,
                      const float* __restrict__ bias, __half* __restrict__ pre) {
    gemm<Layout::kNT, Epilogue::kBiasGelu>(y, x, w, m, n, k, TnArgs{}, bias, pre);
}

__global__ void __launch_bounds__(kThreads, 2)
    gemm_nn(__half* __restrict__ y, const __half* __restrict__ x, const __half* __restrict__ w,
            int m, int n, int k) {
    gemm<Layout::kNN, Epilogue::kNone>(y, x, w, m, n, k, TnArgs{}, nullptr, nullptr);
}

// pre is only read; the body's aux is writable for the forward's GELU.
__global__ void __launch_bounds__(kThreads, 2)
    gemm_nn_dgelu(__half* __restrict__ y, const __half* __restrict__ x,
                  const __half* __restrict__ w, int m, int n, int k,
                  const __half* __restrict__ pre) {
    gemm<Layout::kNN, Epilogue::kDGelu>(y, x, w, m, n, k, TnArgs{}, nullptr,
                                        const_cast<__half*>(pre));
}

// y is FP32, the residual stream after the layer, and residual is only read.
__global__ void __launch_bounds__(kThreads, 2)
    gemm_nt_bias_residual(float* __restrict__ y, const __half* __restrict__ x,
                          const __half* __restrict__ w, int m, int n, int k,
                          const float* __restrict__ bias, const float* __restrict__ residual) {
    gemm<Layout::kNT, Epilogue::kBiasResidual>(y, x, w, m, n, k, TnArgs{}, bias,
                                               const_cast<float*>(residual));
}

__global__ void __launch_bounds__(kThreads, 2)
    gemm_tn(float* __restrict__ y, const __half* __restrict__ x, const __half* __restrict__ w,
            int m, int n, int k, TnArgs tn) {
    gemm<Layout::kTN, Epilogue::kNone>(y, x, w, m, n, k, tn, nullptr, nullptr);
}

// Sums a split TN GEMM's partial outputs in split order, and writes y or adds
// into it: y = old + (p0 + p1 + ... + p(S-1)), the same order a single split
// uses (old + sum), whichever split finished first. count is a multiple of 4.
__global__ void __launch_bounds__(kThreads)
    gemm_tn_reduce(float* __restrict__ y, const float* __restrict__ partials, size_t count,
                   int splits, int accumulate) {
    const size_t vectors = count / 4;
    const auto* p = reinterpret_cast<const float4*>(partials);
    auto* out = reinterpret_cast<float4*>(y);
    for (size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < vectors;
         i += size_t{gridDim.x} * blockDim.x) {
        float4 sum = p[i];
        for (int s = 1; s < splits; ++s) {
            const float4 v = p[size_t(s) * vectors + i];
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        if (accumulate) {
            const float4 old = out[i];
            sum = make_float4(old.x + sum.x, old.y + sum.y, old.z + sum.z, old.w + sum.w);
        }
        out[i] = sum;
    }
}

// The grid: gridDim.x rows of tiles per group, a column of tiles per y, and
// the groups of each split in turn along z.
dim3 grid_for(int m, int n, int splits, int* groups) {
    const int tiles_m = (m + kBlockM - 1) / kBlockM;
    const int group = tiles_m < kGroupM ? tiles_m : kGroupM;
    *groups = (tiles_m + group - 1) / group;
    return dim3(static_cast<unsigned>(group), static_cast<unsigned>((n + kBlockN - 1) / kBlockN),
                static_cast<unsigned>(*groups * splits));
}

using Kernel = void (*)(__half*, const __half*, const __half*, int, int, int);

cudaError_t launch(Kernel kernel, void* y, const void* x, const void* w, int m, int n, int k,
                   cudaStream_t stream) {
    int groups = 0;
    const dim3 grid = grid_for(m, n, 1, &groups);
    kernel<<<grid, kThreads, 0, stream>>>(static_cast<__half*>(y), static_cast<const __half*>(x),
                                          static_cast<const __half*>(w), m, n, k);
    return cudaGetLastError();
}

// TN's split of k. Blocks run in waves of two per SM, and a last wave that's
// mostly empty wastes the rest: 36 tiles leave most of the 2080 Ti's 136
// slots idle, and 144 tiles run as one wave plus 8 blocks. So the count is
// the smallest that keeps at least kMinFill of the slots busy across all the
// waves, or else the fullest, and at most kMaxSplits. Every split but the last
// is a multiple of 32 and at least kMinSplitK long; the last takes the rest,
// which can be shorter (k = 2049 splits as 1056 and 993). On the 2080 Ti at
// 8,192 tokens it picks 3 for 768x768, 768x3072 and 3072x768, and 1 for
// 2304x768, each the fastest measured or within noise of it.
constexpr int kMinSplitK = 1024;
constexpr int kMaxSplits = 16;
constexpr double kMinFill = 0.75;

struct Splits {
    int count = 1;
    int len = 0;
};

cudaError_t tn_splits(int m, int n, int k, Splits* splits) {
    int device = 0;
    int sms = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) {
        error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    }
    if (error != cudaSuccess) {
        return error;
    }
    const long long tiles = static_cast<long long>((m + kBlockM - 1) / kBlockM) *
                            ((n + kBlockN - 1) / kBlockN);
    const long long slots = 2LL * sms;
    const int most = std::min(kMaxSplits, std::max(1, k / kMinSplitK));
    int want = 1;
    double best = 0;
    for (int s = 1; s <= most; ++s) {
        const long long blocks = tiles * s;
        const long long waves = (blocks + slots - 1) / slots;
        const double fill = static_cast<double>(blocks) / static_cast<double>(waves * slots);
        if (fill > best) {
            best = fill;
            want = s;
        }
        if (fill >= kMinFill) {
            want = s;
            break;
        }
    }
    const int per_split = (k + want - 1) / want;
    splits->len = (per_split + kBlockK - 1) / kBlockK * kBlockK;
    splits->count = (k + splits->len - 1) / splits->len;
    return cudaSuccess;
}

}  // namespace

cudaError_t launch_gemm_nt(void* y, const void* x, const void* w, int m, int n, int k,
                           cudaStream_t stream) {
    return launch(gemm_nt, y, x, w, m, n, k, stream);
}

cudaError_t launch_gemm_nt_bias(void* y, const void* x, const void* w, const void* bias, int m,
                                int n, int k, cudaStream_t stream) {
    int groups = 0;
    const dim3 grid = grid_for(m, n, 1, &groups);
    gemm_nt_bias<<<grid, kThreads, 0, stream>>>(
        static_cast<__half*>(y), static_cast<const __half*>(x), static_cast<const __half*>(w), m,
        n, k, static_cast<const float*>(bias));
    return cudaGetLastError();
}

cudaError_t launch_gemm_nt_bias_gelu(void* y, void* pre, const void* x, const void* w,
                                     const void* bias, int m, int n, int k, cudaStream_t stream) {
    int groups = 0;
    const dim3 grid = grid_for(m, n, 1, &groups);
    gemm_nt_bias_gelu<<<grid, kThreads, 0, stream>>>(
        static_cast<__half*>(y), static_cast<const __half*>(x), static_cast<const __half*>(w), m,
        n, k, static_cast<const float*>(bias), static_cast<__half*>(pre));
    return cudaGetLastError();
}

cudaError_t launch_gemm_nt_bias_residual(void* y, const void* x, const void* w, const void* bias,
                                         const void* residual, int m, int n, int k,
                                         cudaStream_t stream) {
    int groups = 0;
    const dim3 grid = grid_for(m, n, 1, &groups);
    gemm_nt_bias_residual<<<grid, kThreads, 0, stream>>>(
        static_cast<float*>(y), static_cast<const __half*>(x), static_cast<const __half*>(w), m,
        n, k, static_cast<const float*>(bias), static_cast<const float*>(residual));
    return cudaGetLastError();
}

cudaError_t launch_gemm_nn(void* y, const void* x, const void* w, int m, int n, int k,
                           cudaStream_t stream) {
    return launch(gemm_nn, y, x, w, m, n, k, stream);
}

cudaError_t launch_gemm_nn_dgelu(void* y, const void* x, const void* w, const void* pre, int m,
                                 int n, int k, cudaStream_t stream) {
    int groups = 0;
    const dim3 grid = grid_for(m, n, 1, &groups);
    gemm_nn_dgelu<<<grid, kThreads, 0, stream>>>(
        static_cast<__half*>(y), static_cast<const __half*>(x), static_cast<const __half*>(w), m,
        n, k, static_cast<const __half*>(pre));
    return cudaGetLastError();
}

cudaError_t gemm_tn_workspace(int m, int n, int k, size_t* bytes) {
    Splits splits;
    const cudaError_t error = tn_splits(m, n, k, &splits);
    if (error == cudaSuccess) {
        *bytes = splits.count > 1 ? size_t(splits.count) * m * n * sizeof(float) : 0;
    }
    return error;
}

cudaError_t launch_gemm_tn(void* y, const void* x, const void* w, int m, int n, int k,
                           bool accumulate, void* workspace, cudaStream_t stream) {
    Splits splits;
    cudaError_t error = tn_splits(m, n, k, &splits);
    if (error != cudaSuccess) {
        return error;
    }
    TnArgs tn;
    tn.split_len = splits.len;
    tn.partials = splits.count > 1 ? static_cast<float*>(workspace) : nullptr;
    tn.accumulate = accumulate ? 1 : 0;
    const dim3 grid = grid_for(m, n, splits.count, &tn.groups);
    gemm_tn<<<grid, kThreads, 0, stream>>>(static_cast<float*>(y), static_cast<const __half*>(x),
                                           static_cast<const __half*>(w), m, n, k, tn);
    error = cudaGetLastError();
    if (error != cudaSuccess || splits.count == 1) {
        return error;
    }
    const size_t count = size_t(m) * n;
    const size_t blocks = std::clamp<size_t>((count / 4 + kThreads - 1) / kThreads, 1, 4096);
    gemm_tn_reduce<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        static_cast<float*>(y), tn.partials, count, splits.count, tn.accumulate);
    return cudaGetLastError();
}

}  // namespace flint
