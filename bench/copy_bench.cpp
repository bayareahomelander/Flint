#include "copy_bench.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "flint/flint.h"

namespace bench {
namespace {

// Fills device memory with seeded random bytes, staged through a small host buffer.
void fill_random(void* device, size_t bytes, uint64_t seed) {
    std::vector<unsigned char> host(std::min(bytes, kHostChunk));
    uint64_t state = seed;
    for (size_t offset = 0; offset < bytes; offset += host.size()) {
        const size_t n = std::min(host.size(), bytes - offset);
        for (size_t i = 0; i < n; i += 8) {
            const uint64_t value = splitmix64(state);
            std::memcpy(&host[i], &value, std::min<size_t>(8, n - i));
        }
        check_cuda(cudaMemcpy(static_cast<char*>(device) + offset, host.data(), n,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy");
    }
}

}  // namespace

CaseResult bench_copy(unsigned long long bytes, const DeviceInfo& device, Monitor& monitor,
                      const Options& options, cudaStream_t stream) {
    const size_t size = static_cast<size_t>(bytes);
    // The sets are carved from one arena per side, 256-byte aligned like
    // cudaMalloc, so tiny sizes don't need thousands of allocations.
    const size_t stride = (size + 255) / 256 * 256;
    const int sets = rotation_sets(device, 2.0 * stride);

    // Flint and the baseline share these buffers, so their inputs are identical.
    DeviceAllocations allocations;
    char* src_arena = static_cast<char*>(allocations.allocate(stride * sets));
    char* dst_arena = static_cast<char*>(allocations.allocate(stride * sets));
    std::vector<void*> src(sets), dst(sets);
    for (int s = 0; s < sets; ++s) {
        src[s] = src_arena + stride * s;
        dst[s] = dst_arena + stride * s;
    }
    void* check = allocations.allocate(size);
    fill_random(src_arena, stride * sets, 0x5eed);
    check_cuda(cudaMemset(dst_arena, 0, stride * sets), "cudaMemset");

    CaseSpec spec;
    spec.benchmark = "copy";
    spec.label = size_label(bytes);
    spec.baseline_name = "cudaMemcpyAsync";
    spec.size_bytes = bytes;
    spec.bytes_moved = 2.0 * size;
    spec.rotation_sets = sets;
    spec.flint = [&](cudaStream_t s, long long first, long long count) {
        for (long long i = first; i < first + count; ++i) {
            const int set = static_cast<int>(i % sets);
            check_flint(flint_copy(dst[set], src[set], size, s), "flint_copy");
        }
    };
    spec.reference = [&](cudaStream_t s, long long first, long long count) {
        for (long long i = first; i < first + count; ++i) {
            const int set = static_cast<int>(i % sets);
            check_cuda(cudaMemcpyAsync(dst[set], src[set], size, cudaMemcpyDeviceToDevice, s),
                       "cudaMemcpyAsync");
        }
    };

    CaseResult result = run_case(spec, device, monitor, options, stream);

    // The timing left src's bytes in every dst, so poison both outputs with
    // different bytes first: a copy that skips any byte can't match.
    check_cuda(cudaMemsetAsync(dst[0], 0xA5, size, stream), "cudaMemsetAsync");
    check_cuda(cudaMemsetAsync(check, 0x5A, size, stream), "cudaMemsetAsync");
    check_flint(flint_copy(dst[0], src[0], size, stream), "flint_copy");
    check_cuda(cudaMemcpyAsync(check, src[0], size, cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    result.outputs_match = device_equal(dst[0], check, size);
    return result;
}

}  // namespace bench
