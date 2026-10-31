// Tensor-core throughput kernels for the mma benchmark. They live in the
// benchmark, not the library, and in namespace flint so compute-sanitizer's
// kernel filter covers them.
#pragma once

#include <cuda_runtime_api.h>

namespace flint::microbench {

constexpr int kMmaThreads = 256;

// Launches `blocks` blocks of kMmaThreads threads. Every warp runs
// `iterations` steps of `chains` independent mma.sync m16n8k8 instructions
// (FP16 inputs, FP32 accumulation) on registers alone, with every A and B
// element equal to `value`. Each thread then stores the sum of its chains'
// accumulators as one float4 in `out`, which needs blocks * kMmaThreads of
// them. chains must be 4, 8, or 16.
cudaError_t launch_mma_m16n8k8_f32(int chains, int blocks, int iterations, float value, float* out,
                                   cudaStream_t stream);

// How many blocks of the `chains` variant one SM holds at once, so a launch
// can be exactly one full wave; 0 for an unsupported chain count.
int mma_m16n8k8_f32_blocks_per_sm(int chains);

}  // namespace flint::microbench
