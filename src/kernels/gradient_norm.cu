// The L2 norm of a list of FP32 tensors, taken as one vector, and the count of its values
// that are inf or NaN, in two passes with no atomics, so the result is the same bits on every
// run and on every GPU. The optimizer reads both: it clips by the norm, and the loss scale's
// overflow check reads the count.
//
// The sums are in FP64. An FP32 value converts to FP64 exactly, and its square (a 48-bit
// product of 24-bit significands) is exact in FP64, so fma(d, d, s) and s + d * d give the
// same bits, whatever the compiler contracts. The sum of squares of finite FP32 values can't
// overflow FP64 (each is below 2^256, and there are fewer than 2^64 of them), so finite
// gradients always give a finite norm: unlike an FP32 sum of squares, it doesn't overflow at
// gradients near 1e19. If any value is inf or NaN, the sum and the norm aren't finite either.
// Nothing flushes subnormals to 0 (no .FTZ, no fast-math), so FP32 subnormals count at their
// exact values.
// The overflow check is the CUDA isfinite on each FP32 value, never a comparison or a max,
// which would drop a NaN.
//
// The first pass is 1,024 blocks of 256 threads, whatever the GPU. The values are cut into
// chunks of 4,096, and chunks never span tensors: tensor t has ceil(count_t / 4096) chunks,
// numbered globally in table order (tensor 0's first). Block b takes every chunk whose global
// number g has g % 1024 == b, in increasing g. A block walks the whole table, every thread
// reading the same entry, and keeps the running number of each tensor's first chunk, which
// gives its first chunk congruent to b without looping over the others. In a chunk, thread
// tid takes, for j = 0 to 3, the group of 4 values at j * 1024 + 4 * tid, in order j then 0 to
// 3 within the group. The 4 groups are loaded first, then added, so several loads are in
// flight, through the read-only path (__ldg), which also makes them global loads, since the
// compiler can't tell that a pointer read from the table is in global memory. A group in range
// of a 16-byte aligned tensor is one 16-byte load; otherwise its values load one by one, and
// those past the tensor's end are +0. The padding is exact: the running sum starts at +0 and
// squares are at least +0, so it is never -0 and s + 0 = s, and 0 is finite. A thread keeps
// one FP64 sum and one count across all its chunks.
//
// The block reduction, in this order, is the same in both passes. Within each warp, for
// offsets 16, 8, 4, 2, 1: s += shuffle down by the offset (the counts the same). Lane 0 of
// warp w stores its sum and count in shared memory, and after a barrier, thread 0 adds the 8
// warps' in order of w, one after the other. The first pass's block b stores its total in the
// workspace, the sums first and the counts after them.
//
// The second pass is one block: thread t adds the partial sums t, t + 256, t + 512 and t + 768
// in that order, then the block reduction runs, and thread 0 stores the square root (IEEE
// rounded, with no fast-math intrinsic) and the count.
//
// The orders depend only on the tensors' counts and the constants here, and 16-byte alignment
// only picks the load instruction. Both passes always launch, even for no tensors, when the
// first pass's blocks store 0s. Speed matters little: the step runs it once, for about a
// millisecond.
#include "kernels/gradient_norm.cuh"

namespace flint {
namespace {

constexpr int kWarp = 32;
constexpr unsigned kAllLanes = 0xffffffffu;
constexpr int kBlocks = static_cast<int>(kGradientNormBlocks);
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarp;
// The values of a group, which one thread loads at once (a 16-byte load).
constexpr int kGroup = 4;
// The groups a thread takes from a chunk, loaded together.
constexpr int kUnroll = 4;
constexpr size_t kChunk = size_t{kThreads} * kGroup * kUnroll;
static_assert(kChunk == 4096, "the header comment's numbers assume chunks of 4,096");
static_assert(kGradientNormWorkspaceBytes == 2 * kBlocks * sizeof(double),
              "the workspace holds a double and a count per block");
// The second pass's block adds this many partial sums per thread.
constexpr int kFinishPartials = kBlocks / kThreads;

// A table entry, flint_tensor's layout.
struct Tensor {
    const float* data;
    size_t count;
};
static_assert(sizeof(Tensor) == 16, "a flint_tensor is 16 bytes");

// Reduces each thread's sum and count over the block, in the order of the header comment. The
// total is in thread 0's s and count on return.
__device__ void reduce_block(double& s, unsigned long long& count) {
    __shared__ double warp_sums[kWarps];
    __shared__ unsigned long long warp_counts[kWarps];
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        s += __shfl_down_sync(kAllLanes, s, offset);
        count += __shfl_down_sync(kAllLanes, count, offset);
    }
    const int tid = static_cast<int>(threadIdx.x);
    if (tid % kWarp == 0) {
        warp_sums[tid / kWarp] = s;
        warp_counts[tid / kWarp] = count;
    }
    __syncthreads();
    if (tid == 0) {
        s = warp_sums[0];
        count = warp_counts[0];
        for (int w = 1; w < kWarps; ++w) {
            s += warp_sums[w];
            count += warp_counts[w];
        }
    }
}

// Block b: the sum of squares and the count of non-finite values over the chunks g of the
// tensors with g % 1024 == b, stored as partial_sums[b] and partial_counts[b].
__global__ void __launch_bounds__(kThreads)
    gradient_norm_partials(double* __restrict__ partial_sums,
                           unsigned long long* __restrict__ partial_counts,
                           const Tensor* __restrict__ tensors, size_t tensor_count) {
    const unsigned tid = threadIdx.x;
    const unsigned long long block = blockIdx.x;
    double s = 0.0;
    unsigned long long count = 0;

    // The global number of the current tensor's first chunk.
    unsigned long long first = 0;
    for (size_t t = 0; t < tensor_count; ++t) {
        const Tensor tensor = tensors[t];
        const unsigned long long chunks = tensor.count / kChunk + (tensor.count % kChunk != 0);
        // The first chunk of this tensor with g % 1024 == block, as an index into the tensor.
        const unsigned long long skip = (block + kBlocks - first % kBlocks) % kBlocks;
        first += chunks;
        const bool vector_loads = reinterpret_cast<size_t>(tensor.data) % 16 == 0;
        for (unsigned long long c = skip; c < chunks; c += kBlocks) {
            float v[kUnroll][kGroup];
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                const size_t start = c * kChunk + static_cast<size_t>(j * kThreads * kGroup) +
                                     size_t{tid} * kGroup;
                const size_t left = start < tensor.count ? tensor.count - start : 0;
                if (vector_loads && left >= kGroup) {
                    const float4 q = __ldg(reinterpret_cast<const float4*>(tensor.data + start));
                    v[j][0] = q.x;
                    v[j][1] = q.y;
                    v[j][2] = q.z;
                    v[j][3] = q.w;
                } else {
#pragma unroll
                    for (int k = 0; k < kGroup; ++k) {
                        v[j][k] =
                            static_cast<size_t>(k) < left ? __ldg(tensor.data + start + k) : 0.0f;
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
#pragma unroll
                for (int k = 0; k < kGroup; ++k) {
                    count += !isfinite(v[j][k]);
                    const double d = static_cast<double>(v[j][k]);
                    s = fma(d, d, s);
                }
            }
        }
    }
    reduce_block(s, count);
    if (tid == 0) {
        partial_sums[block] = s;
        partial_counts[block] = count;
    }
}

// The one block: *norm = the square root of the partial sums' total, *nonfinite = the
// partial counts' total.
__global__ void __launch_bounds__(kThreads)
    gradient_norm_finish(double* __restrict__ norm, unsigned long long* __restrict__ nonfinite,
                         const double* __restrict__ partial_sums,
                         const unsigned long long* __restrict__ partial_counts) {
    const int tid = static_cast<int>(threadIdx.x);
    double s = 0.0;
    unsigned long long count = 0;
    for (int i = 0; i < kFinishPartials; ++i) {
        s += partial_sums[tid + i * kThreads];
        count += partial_counts[tid + i * kThreads];
    }
    reduce_block(s, count);
    if (tid == 0) {
        *norm = sqrt(s);
        *nonfinite = count;
    }
}

}  // namespace

cudaError_t launch_gradient_norm(void* norm, void* nonfinite, const void* tensors,
                                 size_t tensor_count, void* workspace, cudaStream_t stream) {
    auto* partial_sums = static_cast<double*>(workspace);
    auto* partial_counts = reinterpret_cast<unsigned long long*>(partial_sums + kBlocks);
    gradient_norm_partials<<<kBlocks, kThreads, 0, stream>>>(
        partial_sums, partial_counts, static_cast<const Tensor*>(tensors), tensor_count);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    gradient_norm_finish<<<1, kThreads, 0, stream>>>(static_cast<double*>(norm),
                                                     static_cast<unsigned long long*>(nonfinite),
                                                     partial_sums, partial_counts);
    return cudaGetLastError();
}

}  // namespace flint
