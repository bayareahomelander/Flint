#include "clock_probe.cuh"

#include <algorithm>

namespace flint::microbench {
namespace {

// The reads are volatile asm, so they stay in this order. A block can be
// preempted and resumed on another SM between them, and the SMs' counters
// differ by hundreds of millions of cycles, so the pair is stored only if the
// SM was the same before and after.
__global__ void clock_probe(unsigned long long* slots) {
    if (threadIdx.x == 0) {
        unsigned long long cycles, ns;
        unsigned smid_before, smid_after;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_before));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles));
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_after));
        if (smid_before == smid_after && smid_before < kClockProbeSlots) {
            slots[2 * smid_before] = cycles;
            slots[2 * smid_before + 1] = ns;
        }
    }
}

}  // namespace

cudaError_t launch_clock_probe(int sm_count, unsigned long long* slots, cudaStream_t stream) {
    int device = 0;
    int resident_threads = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess) {
        error = cudaDeviceGetAttribute(&resident_threads, cudaDevAttrMaxThreadsPerMultiProcessor,
                                       device);
    }
    if (error != cudaSuccess) {
        return error;
    }
    // More than half of what an SM holds, so no SM takes two blocks.
    const int threads = std::min(1024, (resident_threads / 2 + 32) / 32 * 32);
    clock_probe<<<sm_count, threads, 0, stream>>>(slots);
    return cudaGetLastError();
}

}  // namespace flint::microbench
