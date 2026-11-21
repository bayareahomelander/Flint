#include "mma_bench.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "clock_probe.cuh"
#include "common.h"
#include "mma_kernels.cuh"

namespace bench {
namespace {

using flint::microbench::kClockProbeSlots;
using flint::microbench::kMmaThreads;
using flint::microbench::launch_clock_probe;
using flint::microbench::launch_mma_m16n8k8_f32;
using flint::microbench::mma_m16n8k8_f32_blocks_per_sm;

// Every warp runs the same number of mma steps whatever the chain count,
// about 2 ms a launch on the 2080 Ti.
constexpr int kStepsPerLaunch = 32768;
// 8 * 0.25 * 0.25 = 0.5 is exact, so each step adds exactly 0.5 to every
// accumulator, and every sum stays exact in FP32.
constexpr float kValue = 0.25f;
constexpr double kFlopPerMma = 2.0 * 16 * 8 * 8;

}  // namespace

double tensor_flop_per_sm_per_clock(const DeviceInfo& device) {
    // Data-center and workstation parts of the same architectures (T4, Quadro
    // RTX, L4, L40) accumulate FP32 at the full 1024, so only GeForce counts.
    const int cc = device.compute_major * 10 + device.compute_minor;
    const bool geforce = device.name.find("GeForce") != std::string::npos;
    return geforce && (cc == 75 || cc == 89) ? 512 : 0;
}

CaseResult bench_mma(int chains, const DeviceInfo& device, Monitor& monitor, const Options& options,
                     cudaStream_t stream) {
    // Exactly one full wave, so no SM waits on a tail of leftover blocks.
    const int blocks_per_sm = mma_m16n8k8_f32_blocks_per_sm(chains);
    if (blocks_per_sm == 0) {
        throw std::runtime_error("mma: no occupancy for " + std::to_string(chains) + " chains");
    }
    const int blocks = blocks_per_sm * device.sm_count;
    const int iterations = kStepsPerLaunch / chains;
    const size_t outputs = size_t{static_cast<unsigned>(blocks)} * kMmaThreads * 4;

    constexpr size_t kProbeBytes = 2 * kClockProbeSlots * sizeof(unsigned long long);
    void* buffers[3] = {nullptr, nullptr, nullptr};
    struct Free {
        void** buffers;
        ~Free() {
            cudaFree(buffers[0]);
            cudaFree(buffers[1]);
            cudaFree(buffers[2]);
        }
    } free_buffers{buffers};
    check_cuda(cudaMalloc(&buffers[0], outputs * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&buffers[1], outputs * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&buffers[2], kProbeBytes), "cudaMalloc");
    float* out = static_cast<float*>(buffers[0]);
    float* check = static_cast<float*>(buffers[1]);
    auto* probe_slots = static_cast<unsigned long long*>(buffers[2]);

    const double warps = double(blocks) * kMmaThreads / 32;
    CaseSpec spec;
    spec.benchmark = "mma";
    spec.label = "m16n8k8 f16->f32, " + std::to_string(chains) + " chains, " +
                 std::to_string(blocks_per_sm) + " blocks/SM";
    spec.flops = warps * iterations * chains * kFlopPerMma;
    spec.peak_flop_per_sm_per_clock = tensor_flop_per_sm_per_clock(device);
    spec.flint = [&](cudaStream_t s, long long, long long count) {
        for (long long i = 0; i < count; ++i) {
            check_cuda(launch_mma_m16n8k8_f32(chains, blocks, iterations, kValue, out, s),
                       "mma_m16n8k8_f32");
        }
    };
    // The SM clock, from a probe kernel's (clock64, %globaltimer) pair. Two
    // readings must come from the same SM, since the SMs' counters differ by
    // hundreds of millions of cycles: the first reading picks the lowest SM that
    // has a pair, and every later one uses it. A launch can miss an SM, so a
    // missing pair is retried, and failing loudly after that is deliberate: a
    // clock from another SM would be silently wrong.
    int pinned_sm = -1;
    spec.kernel_clock = [&](cudaStream_t s) {
        std::vector<unsigned long long> pairs(2 * kClockProbeSlots);
        for (int attempt = 0; attempt < 10; ++attempt) {
            check_cuda(cudaMemsetAsync(probe_slots, 0, kProbeBytes, s), "cudaMemsetAsync");
            check_cuda(launch_clock_probe(device.sm_count, probe_slots, s), "clock_probe");
            // Once an SM is pinned, only its two slots come back.
            const bool whole = pinned_sm < 0;
            const size_t first = whole ? 0 : 2 * static_cast<size_t>(pinned_sm);
            const size_t count = whole ? pairs.size() : 2;
            check_cuda(cudaMemcpyAsync(pairs.data(), probe_slots + first,
                                       count * sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost, s),
                       "cudaMemcpyAsync");
            check_cuda(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            if (whole) {
                for (int sm = 0; sm < kClockProbeSlots; ++sm) {
                    if (pairs[2 * sm + 1] != 0) {
                        pinned_sm = sm;
                        break;
                    }
                }
            }
            if (pinned_sm >= 0) {
                const size_t at = whole ? 2 * static_cast<size_t>(pinned_sm) : 0;
                if (pairs[at + 1] != 0) {
                    return KernelClock{pairs[at], pairs[at + 1]};
                }
            }
        }
        throw std::runtime_error("mma: the clock probe found no pair on its SM");
    };

    CaseResult result = run_case(spec, device, monitor, options, stream);

    // Every thread's sum must be exactly chains * 0.5 * iterations, and a
    // second launch must reproduce it bit for bit.
    check_cuda(cudaMemsetAsync(out, 0xA5, outputs * sizeof(float), stream), "cudaMemsetAsync");
    check_cuda(cudaMemsetAsync(check, 0x5A, outputs * sizeof(float), stream), "cudaMemsetAsync");
    check_cuda(launch_mma_m16n8k8_f32(chains, blocks, iterations, kValue, out, stream),
               "mma_m16n8k8_f32");
    check_cuda(launch_mma_m16n8k8_f32(chains, blocks, iterations, kValue, check, stream),
               "mma_m16n8k8_f32");
    std::vector<float> first(outputs), second(outputs);
    check_cuda(cudaMemcpyAsync(first.data(), out, outputs * sizeof(float), cudaMemcpyDeviceToHost,
                               stream),
               "cudaMemcpyAsync");
    check_cuda(cudaMemcpyAsync(second.data(), check, outputs * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const float expected = static_cast<float>(chains * 0.5 * iterations);
    bool exact = std::memcmp(first.data(), second.data(), outputs * sizeof(float)) == 0;
    for (size_t i = 0; exact && i < outputs; ++i) {
        exact = first[i] == expected;
    }
    result.outputs_match = exact;
    return result;
}

}  // namespace bench
