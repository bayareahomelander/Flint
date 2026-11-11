// The embedding's backward pass, with no atomics, in an order that depends only on the
// inputs' shape and values, so the result is the same bits on every run and every GPU.
//
// The word table's gradient buckets the rows by token first. A stable LSD radix sort
// orders the row indices by token, in two passes of 8 bits, low byte first. Each pass
// counts the digits per tile of 2,048 rows, turns the counts into each (digit, tile)'s
// first position with one block's exclusive scan, digit by digit, and scatters each
// row to its position. Within a warp, __match_any_sync finds the lanes that share a
// digit and the population count below a lane ranks it among them; each warp keeps
// its own counters in shared memory, and a counter is changed only by the lowest lane
// of its digit's group, so no two threads ever write one counter at once. The rows
// start in ascending order and both passes are stable, so the rows end up sorted by
// (token, row). Those pairs are all different, so this order is the only correct one,
// whatever the tiles.
//
// A binary search then finds, for each v from 0 to vocab, how many rows have a token
// below v, and a block per row of the table sums its token's rows of dx, in ascending
// order of row, 4 columns per thread. A token at or above vocab sorts after every
// token below it, so no block ever reaches its rows.
//
// The sums are compensated, as the bias gradient's are (TwoSum, after Ogita, Rump and
// Oishi's Sum2): each keeps its rounded value and the sum of the rounding errors left
// behind in it, and only the final total adds the two. Plain sums missed the owner's
// bound of twice FP32 PyTorch's own error against FP64 on both GPUs: the word table's
// by up to 20 times, on real tokens and on tokens repeated thousands of times, and the
// position table's by up to 46 times over thousands of sequences.
//
// The position table's gradient is the bias gradient's column sum over the sequences,
// with dx viewed as sequences x (length * width): a first pass with the bias
// gradient's layout and order, on FP32 values, then its second pass.
#include "kernels/embedding_backward.cuh"

#include <cstdint>

namespace flint {
namespace {

constexpr int kWarp = 32;
constexpr unsigned kAllLanes = 0xffffffffu;

// The sort: a tile of 2,048 rows per block of 8 warps, 256 rows to a warp.
constexpr int kSortWarps = 8;
constexpr int kSortThreads = kWarp * kSortWarps;
constexpr int kTileRows = static_cast<int>(kEmbeddingSortTileRows);
constexpr int kWarpRows = kTileRows / kSortWarps;
constexpr int kDigits = static_cast<int>(kEmbeddingSortDigits);
static_assert(kSortThreads == kDigits, "a thread per digit sums the warps' counts");
// A lane past the last row takes this digit, which no counter has.
constexpr unsigned kNoDigit = kDigits;
// The scan's block.
constexpr int kScanThreads = 1024;
// The offsets' block.
constexpr int kOffsetThreads = 256;
// The word table's gradient: a thread per 4 columns, 64 to a block.
constexpr int kWordThreads = 64;
// The position table's first pass: the bias gradient's block, kTileX pieces of 8 values
// across and kTileY rows down.
constexpr int kPieceValues = 8;
constexpr int kTileX = kWarp;
constexpr int kTileY = 8;
constexpr int kChunkRows = static_cast<int>(kBiasGradientChunkRows);
static_assert(kTileX * kPieceValues == 256, "the bias gradient's tiles are 256 columns");
static_assert(kWordThreads * 4 == 256, "the word table's limit assumes 256 columns a block");

// A rounded sum and the sum of the rounding errors left behind in it.
struct CompensatedSum {
    float sum = 0.0f;
    float error = 0.0f;
};

// Adds v. Knuth's TwoSum finds the error e with sum + v = t + e exactly, for any order
// of magnitude between the two.
__device__ void add(CompensatedSum& a, float v) {
    const float t = __fadd_rn(a.sum, v);
    const float v_part = __fsub_rn(t, a.sum);
    a.error = __fadd_rn(a.error, __fadd_rn(__fsub_rn(a.sum, __fsub_rn(t, v_part)), __fsub_rn(v, v_part)));
    a.sum = t;
}

// Merges b into a: b's sum through TwoSum, then both errors.
__device__ void merge(CompensatedSum& a, float b_sum, float b_error) {
    const float errors = __fadd_rn(a.error, b_error);
    a.error = 0.0f;
    add(a, b_sum);
    a.error = __fadd_rn(a.error, errors);
}

// The sum's value: sum + error. A non-finite sum is the value itself: once an inf has
// passed through TwoSum, its error is inf - inf = NaN, which would turn a +inf sum into
// NaN.
__device__ float value(const CompensatedSum& a) {
    return isfinite(a.sum) ? __fadd_rn(a.sum, a.error) : a.sum;
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

// Row e of the sort's input: its digit, or kNoDigit past the last row.
__device__ unsigned digit_of(const uint16_t* __restrict__ tokens, size_t e, size_t rows,
                             int shift) {
    return e < rows ? (static_cast<unsigned>(tokens[e]) >> shift) & (kDigits - 1) : kNoDigit;
}

// Counts the digits of warp w's 256 rows of the tile into counts[w], 32 rows at a time.
// The lowest lane of each group of lanes that share a digit adds the group's size; the
// groups' digits differ, so no two lanes change one counter at once.
__device__ void count_warp(unsigned (&counts)[kSortWarps][kDigits], const uint16_t* __restrict__ tokens,
                           size_t first, size_t rows, int shift, int warp, int lane) {
    for (int i = 0; i < kWarpRows / kWarp; ++i) {
        const unsigned digit = digit_of(tokens, first + i * kWarp + lane, rows, shift);
        const unsigned group = __match_any_sync(kAllLanes, digit);
        if (digit != kNoDigit && lane == __ffs(group) - 1) {
            counts[warp][digit] += __popc(group);
        }
        __syncwarp();
    }
}

// Block b: each digit's count over tile b's rows, stored at counts[digit * tiles + b].
__global__ void __launch_bounds__(kSortThreads)
    embedding_sort_count(unsigned* __restrict__ counts, const uint16_t* __restrict__ tokens,
                         size_t rows, int shift) {
    __shared__ unsigned warp_counts[kSortWarps][kDigits];
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarp;
    const int lane = thread % kWarp;
    for (int w = 0; w < kSortWarps; ++w) {
        warp_counts[w][thread] = 0;
    }
    __syncthreads();
    const size_t tile = blockIdx.x;
    count_warp(warp_counts, tokens, tile * kTileRows + warp * kWarpRows, rows, shift, warp, lane);
    __syncthreads();
    unsigned total = 0;
    for (int w = 0; w < kSortWarps; ++w) {
        total += warp_counts[w][thread];
    }
    counts[size_t{static_cast<unsigned>(thread)} * gridDim.x + tile] = total;
}

// One block: replaces the entries values of counts with their exclusive prefix sums.
__global__ void __launch_bounds__(kScanThreads)
    embedding_sort_scan(unsigned* __restrict__ counts, size_t entries) {
    __shared__ unsigned warp_sums[kScanThreads / kWarp];
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarp;
    const int lane = thread % kWarp;
    unsigned carry = 0;
    for (size_t base = 0; base < entries; base += kScanThreads) {
        const size_t i = base + thread;
        const unsigned v = i < entries ? counts[i] : 0;
        unsigned inclusive = v;
        for (int offset = 1; offset < kWarp; offset *= 2) {
            const unsigned other = __shfl_up_sync(kAllLanes, inclusive, offset);
            if (lane >= offset) {
                inclusive += other;
            }
        }
        if (lane == kWarp - 1) {
            warp_sums[warp] = inclusive;
        }
        __syncthreads();
        if (warp == 0) {
            unsigned s = warp_sums[lane];
            for (int offset = 1; offset < kWarp; offset *= 2) {
                const unsigned other = __shfl_up_sync(kAllLanes, s, offset);
                if (lane >= offset) {
                    s += other;
                }
            }
            warp_sums[lane] = s;
        }
        __syncthreads();
        const unsigned before = warp == 0 ? 0 : warp_sums[warp - 1];
        if (i < entries) {
            counts[i] = carry + before + inclusive - v;
        }
        carry += warp_sums[kScanThreads / kWarp - 1];
        __syncthreads();
    }
}

// Block b: moves tile b's rows to their sorted positions, ordered by the digit at shift
// and, within a digit, as they come. starts holds each (digit, tile)'s first position,
// at starts[digit * tiles + b]. in_rows null means the rows are 0, 1, ...
__global__ void __launch_bounds__(kSortThreads)
    embedding_sort_scatter(uint16_t* __restrict__ out_tokens, unsigned* __restrict__ out_rows,
                           const uint16_t* __restrict__ in_tokens,
                           const unsigned* __restrict__ in_rows,
                           const unsigned* __restrict__ starts, size_t rows, int shift) {
    __shared__ unsigned next[kSortWarps][kDigits];
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / kWarp;
    const int lane = thread % kWarp;
    for (int w = 0; w < kSortWarps; ++w) {
        next[w][thread] = 0;
    }
    __syncthreads();
    const size_t tile = blockIdx.x;
    const size_t first = tile * kTileRows + warp * kWarpRows;
    count_warp(next, in_tokens, first, rows, shift, warp, lane);
    __syncthreads();
    // Thread d: each warp's first position for digit d, after the earlier warps' rows.
    unsigned position = starts[size_t{static_cast<unsigned>(thread)} * gridDim.x + tile];
    for (int w = 0; w < kSortWarps; ++w) {
        const unsigned count = next[w][thread];
        next[w][thread] = position;
        position += count;
    }
    __syncthreads();
    for (int i = 0; i < kWarpRows / kWarp; ++i) {
        const size_t e = first + i * kWarp + lane;
        const unsigned digit = digit_of(in_tokens, e, rows, shift);
        const unsigned group = __match_any_sync(kAllLanes, digit);
        if (digit != kNoDigit) {
            const unsigned below = group & ((1u << lane) - 1);
            const unsigned at = next[warp][digit] + __popc(below);
            out_tokens[at] = in_tokens[e];
            out_rows[at] = in_rows == nullptr ? static_cast<unsigned>(e) : in_rows[e];
        }
        __syncwarp();
        if (digit != kNoDigit && lane == __ffs(group) - 1) {
            next[warp][digit] += __popc(group);
        }
        __syncwarp();
    }
}

// Thread v, for v from 0 to vocab: offsets[v] = the number of sorted tokens below v.
__global__ void __launch_bounds__(kOffsetThreads)
    embedding_offsets(unsigned* __restrict__ offsets, const uint16_t* __restrict__ tokens,
                      size_t rows, unsigned vocab) {
    const unsigned v = blockIdx.x * kOffsetThreads + threadIdx.x;
    if (v > vocab) {
        return;
    }
    size_t low = 0;
    size_t high = rows;
    while (low < high) {
        const size_t middle = low + (high - low) / 2;
        if (tokens[middle] < v) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    offsets[v] = static_cast<unsigned>(low);
}

// Block (v, slab): columns 4 (64 slab + thread) to 4 (64 slab + thread) + 3 of dwte's row
// v, the compensated sum of token v's rows of dx in ascending order of row.
__global__ void __launch_bounds__(kWordThreads)
    embedding_backward_words(float4* __restrict__ dwte, const float4* __restrict__ dx,
                             const unsigned* __restrict__ sorted_rows,
                             const unsigned* __restrict__ offsets, unsigned quads,
                             bool accumulate) {
    const unsigned quad = blockIdx.y * kWordThreads + threadIdx.x;
    if (quad >= quads) {
        return;
    }
    const size_t v = blockIdx.x;
    const unsigned begin = offsets[v];
    const unsigned end = offsets[v + 1];
    float4* out = dwte + v * quads + quad;
    if (begin == end) {
        if (!accumulate) {
            *out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    CompensatedSum sum[4];
    for (unsigned k = begin; k < end; ++k) {
        const float4 d = dx[size_t{sorted_rows[k]} * quads + quad];
        add(sum[0], d.x);
        add(sum[1], d.y);
        add(sum[2], d.z);
        add(sum[3], d.w);
    }
    float4 result = make_float4(value(sum[0]), value(sum[1]), value(sum[2]), value(sum[3]));
    if (accumulate) {
        const float4 old = *out;
        result = make_float4(__fadd_rn(old.x, result.x), __fadd_rn(old.y, result.y),
                             __fadd_rn(old.z, result.z), __fadd_rn(old.w, result.w));
    }
    *out = result;
}

// Block (chunk, tile): for the tile's columns j of dx viewed as sequences x
// (length * width), the chunk's compensated sum of dx[s, j] over its sequences, the sums
// in the chunk's first row of partial and the errors in its second, as the bias
// gradient's first pass leaves them.
__global__ void __launch_bounds__(kTileX * kTileY)
    embedding_backward_position_partials(float4* __restrict__ partial,
                                         const float4* __restrict__ dx, size_t sequences,
                                         int pieces) {
    __shared__ float sums[kTileY][kTileX][kPieceValues];
    __shared__ float errors[kTileY][kTileX][kPieceValues];
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int piece = static_cast<int>(blockIdx.y) * kTileX + tx;
    const size_t first = size_t{blockIdx.x} * kChunkRows;
    const size_t end = sequences - first < kChunkRows ? sequences : first + kChunkRows;

    CompensatedSum sum[kPieceValues];
    if (piece < pieces) {
        for (size_t s = first + ty; s < end; s += kTileY) {
            float v[kPieceValues];
            load(v, dx + (s * pieces + piece) * 2);
            for (int j = 0; j < kPieceValues; ++j) {
                add(sum[j], v[j]);
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

cudaError_t launch_embedding_backward(void* dwte, void* dwpe, const void* dx, const void* tokens,
                                      size_t sequences, size_t length, size_t vocab, size_t width,
                                      bool accumulate_wte, bool accumulate_wpe, void* workspace,
                                      cudaStream_t stream) {
    const size_t rows = sequences * length;
    const EmbeddingBackwardLayout layout = embedding_backward_layout(sequences, length, vocab, width);
    auto* base = static_cast<unsigned char*>(workspace);
    auto* sorted_tokens = reinterpret_cast<uint16_t*>(base + layout.sorted_tokens);
    auto* sorted_rows = reinterpret_cast<unsigned*>(base + layout.sorted_rows);
    auto* offsets = reinterpret_cast<unsigned*>(base + layout.offsets);
    auto* pass_tokens = reinterpret_cast<uint16_t*>(base + layout.pass_tokens);
    auto* pass_rows = reinterpret_cast<unsigned*>(base + layout.pass_rows);
    auto* counts = reinterpret_cast<unsigned*>(base + layout.counts);
    const auto* in_tokens = static_cast<const uint16_t*>(tokens);

    // The sort: the low byte into the pass buffers, then the high byte into the sorted
    // ones.
    const auto tiles = static_cast<unsigned>((rows + kTileRows - 1) / kTileRows);
    const size_t entries = size_t{tiles} * kDigits;
    for (int pass = 0; pass < 2; ++pass) {
        const int shift = pass * 8;
        const uint16_t* from_tokens = pass == 0 ? in_tokens : pass_tokens;
        const unsigned* from_rows = pass == 0 ? nullptr : pass_rows;
        uint16_t* to_tokens = pass == 0 ? pass_tokens : sorted_tokens;
        unsigned* to_rows = pass == 0 ? pass_rows : sorted_rows;
        embedding_sort_count<<<tiles, kSortThreads, 0, stream>>>(counts, from_tokens, rows, shift);
        if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
            return error;
        }
        embedding_sort_scan<<<1, kScanThreads, 0, stream>>>(counts, entries);
        if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
            return error;
        }
        embedding_sort_scatter<<<tiles, kSortThreads, 0, stream>>>(
            to_tokens, to_rows, from_tokens, from_rows, counts, rows, shift);
        if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
            return error;
        }
    }
    const auto offset_blocks = static_cast<unsigned>((vocab + 1 + kOffsetThreads - 1) / kOffsetThreads);
    embedding_offsets<<<offset_blocks, kOffsetThreads, 0, stream>>>(
        offsets, sorted_tokens, rows, static_cast<unsigned>(vocab));
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    const auto quads = static_cast<unsigned>(width / 4);
    const dim3 word_grid(static_cast<unsigned>(vocab), (quads + kWordThreads - 1) / kWordThreads);
    embedding_backward_words<<<word_grid, kWordThreads, 0, stream>>>(
        static_cast<float4*>(dwte), static_cast<const float4*>(dx), sorted_rows, offsets, quads,
        accumulate_wte);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }

    // The position table: dx as sequences x (length * width).
    const size_t cols = length * width;
    const auto pieces = static_cast<int>(cols / kPieceValues);
    const auto chunks = static_cast<unsigned>((sequences + kChunkRows - 1) / kChunkRows);
    const auto col_tiles = static_cast<unsigned>((pieces + kTileX - 1) / kTileX);
    void* partials = base + layout.position_partials;
    embedding_backward_position_partials<<<dim3(chunks, col_tiles), dim3(kTileX, kTileY), 0, stream>>>(
        static_cast<float4*>(partials), static_cast<const float4*>(dx), sequences, pieces);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    return launch_bias_gradient_finish(dwpe, partials, sequences, cols, accumulate_wpe, stream);
}

}  // namespace flint
