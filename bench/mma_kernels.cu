#include "mma_kernels.cuh"

#include <cstdint>
#include <cstring>

#include <cuda_fp16.h>

namespace flint::microbench {
namespace {

// The loop only issues mma.sync: no loads, and one store per thread at the
// end. The inline assembly is volatile, so none of it can be dropped or
// merged, and the chains have no dependencies on each other, which hides the
// instruction's latency.
template <int CHAINS>
__global__ void __launch_bounds__(kMmaThreads)
    mma_m16n8k8_f32(float value, int iterations, float4* out) {
    const __half2 pair = __float2half2_rn(value);
    uint32_t operand = 0;
    std::memcpy(&operand, &pair, sizeof operand);
    float acc[CHAINS][4] = {};
    for (int i = 0; i < iterations; ++i) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) {
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
                : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
                : "r"(operand), "r"(operand), "r"(operand));
        }
    }
    float4 sum = make_float4(0, 0, 0, 0);
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) {
        sum.x += acc[c][0];
        sum.y += acc[c][1];
        sum.z += acc[c][2];
        sum.w += acc[c][3];
    }
    out[size_t{blockIdx.x} * blockDim.x + threadIdx.x] = sum;
}

}  // namespace

cudaError_t launch_mma_m16n8k8_f32(int chains, int blocks, int iterations, float value, float* out,
                                   cudaStream_t stream) {
    float4* out4 = reinterpret_cast<float4*>(out);
    switch (chains) {
    case 4:
        mma_m16n8k8_f32<4><<<blocks, kMmaThreads, 0, stream>>>(value, iterations, out4);
        break;
    case 8:
        mma_m16n8k8_f32<8><<<blocks, kMmaThreads, 0, stream>>>(value, iterations, out4);
        break;
    case 16:
        mma_m16n8k8_f32<16><<<blocks, kMmaThreads, 0, stream>>>(value, iterations, out4);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

int mma_m16n8k8_f32_blocks_per_sm(int chains) {
    int blocks = 0;
    cudaError_t error = cudaErrorInvalidValue;
    switch (chains) {
    case 4:
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, mma_m16n8k8_f32<4>,
                                                              kMmaThreads, 0);
        break;
    case 8:
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, mma_m16n8k8_f32<8>,
                                                              kMmaThreads, 0);
        break;
    case 16:
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, mma_m16n8k8_f32<16>,
                                                              kMmaThreads, 0);
        break;
    default:
        break;
    }
    return error == cudaSuccess ? blocks : 0;
}

}  // namespace flint::microbench
