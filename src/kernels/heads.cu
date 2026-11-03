// Attention's head layouts. A row of the rows layout (rows x row_width) holds
// a block of heads * head_width columns, starting at column first, and the
// heads layout (batches x heads x length x head_width) holds each head's rows
// together. Split reads the rows layout and writes the heads layout, and merge
// does the reverse.
//
// One warp takes each row of the rows layout and moves its block of columns
// as 16-byte chunks of 8 FP16 values, which it never converts: they stay bits,
// so NaN payloads and -0 survive. Lane l takes chunks l, l + 32, ... of the
// block, which is head h = chunk / (head_width / 8) at chunk c = chunk %
// (head_width / 8) within the head.
#include "kernels/heads.cuh"

namespace flint {
namespace {

constexpr int kWarp = 32;
constexpr int kRowsPerBlock = 8;
constexpr unsigned kThreads = kWarp * kRowsPerBlock;

// The layouts' sizes in 16-byte chunks.
struct Shape {
    size_t rows;
    size_t length;
    unsigned heads;
    unsigned head_chunks;  // head_width / 8
    unsigned row_chunks;   // row_width / 8
    unsigned first_chunk;  // first / 8
};

// Moves the row's chunks. In split, x is the rows layout and y the heads
// layout; in merge, the reverse. Offsets are size_t: the layouts can hold
// more than 2^32 chunks.
template <bool Split>
__device__ void move_row(uint4* __restrict__ y, const uint4* __restrict__ x, const Shape& s) {
    const size_t row = size_t{blockIdx.x} * kRowsPerBlock + threadIdx.x / kWarp;
    if (row >= s.rows) {
        return;
    }
    const unsigned lane = threadIdx.x % kWarp;
    const size_t batch = row / s.length;
    const size_t position = row % s.length;
    const unsigned chunks = s.heads * s.head_chunks;
    // The row's block of columns starts here in the rows layout, and the
    // batch's first head here in the heads layout, in units of heads.
    const size_t row_at = row * s.row_chunks + s.first_chunk;
    const size_t head_at = batch * s.heads;

    // The lane's head and chunk within it, stepped by 32 chunks at a time
    // instead of divided out of the chunk's index in every iteration.
    unsigned head = lane / s.head_chunks;
    unsigned chunk = lane % s.head_chunks;
    const unsigned head_step = kWarp / s.head_chunks;
    const unsigned chunk_step = kWarp % s.head_chunks;
    for (unsigned i = lane; i < chunks; i += kWarp) {
        const size_t rows_chunk = row_at + i;
        const size_t heads_chunk = ((head_at + head) * s.length + position) * s.head_chunks + chunk;
        if constexpr (Split) {
            y[heads_chunk] = x[rows_chunk];
        } else {
            y[rows_chunk] = x[heads_chunk];
        }
        head += head_step;
        chunk += chunk_step;
        if (chunk >= s.head_chunks) {
            chunk -= s.head_chunks;
            ++head;
        }
    }
}

__global__ void __launch_bounds__(kThreads)
    heads_split(uint4* __restrict__ y, const uint4* __restrict__ x, Shape shape) {
    move_row<true>(y, x, shape);
}

__global__ void __launch_bounds__(kThreads)
    heads_merge(uint4* __restrict__ y, const uint4* __restrict__ x, Shape shape) {
    move_row<false>(y, x, shape);
}

Shape make_shape(size_t rows, size_t length, size_t heads, size_t head_width, size_t row_width,
                 size_t first) {
    return Shape{rows,
                 length,
                 static_cast<unsigned>(heads),
                 static_cast<unsigned>(head_width / 8),
                 static_cast<unsigned>(row_width / 8),
                 static_cast<unsigned>(first / 8)};
}

unsigned block_count(size_t rows) {
    return static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
}

}  // namespace

cudaError_t launch_heads_split(void* y, const void* x, size_t rows, size_t length, size_t heads,
                               size_t head_width, size_t row_width, size_t first,
                               cudaStream_t stream) {
    heads_split<<<block_count(rows), kThreads, 0, stream>>>(
        static_cast<uint4*>(y), static_cast<const uint4*>(x),
        make_shape(rows, length, heads, head_width, row_width, first));
    return cudaGetLastError();
}

cudaError_t launch_heads_merge(void* y, const void* x, size_t rows, size_t length, size_t heads,
                               size_t head_width, size_t row_width, size_t first,
                               cudaStream_t stream) {
    heads_merge<<<block_count(rows), kThreads, 0, stream>>>(
        static_cast<uint4*>(y), static_cast<const uint4*>(x),
        make_shape(rows, length, heads, head_width, row_width, first));
    return cudaGetLastError();
}

}  // namespace flint
