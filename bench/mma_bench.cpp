#include "mma_bench.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "mma_kernels.cuh"

namespace bench {
namespace {

using flint::microbench::kMmaThreads;
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

    void* buffers[2] = {nullptr, nullptr};
    struct Free {
        void** buffers;
        ~Free() {
            cudaFree(buffers[0]);
            cudaFree(buffers[1]);
        }
    } free_buffers{buffers};
    check_cuda(cudaMalloc(&buffers[0], outputs * sizeof(float)), "cudaMalloc");
    check_cuda(cudaMalloc(&buffers[1], outputs * sizeof(float)), "cudaMalloc");
    float* out = static_cast<float*>(buffers[0]);
    float* check = static_cast<float*>(buffers[1]);

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
