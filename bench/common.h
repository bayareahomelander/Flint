// Small helpers shared by the benchmark sources.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "flint/flint.h"

namespace bench {

// The largest host staging buffer for moving data to or from the device, kept
// small for the server's RAM.
constexpr size_t kHostChunk = 16u << 20;

// Throws std::runtime_error naming the call if a CUDA runtime call failed.
inline void check_cuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorName(error) + ": " +
                                 cudaGetErrorString(error));
    }
}

// Throws std::runtime_error with Flint's message if a Flint call failed.
inline void check_flint(flint_status status, const char* what) {
    if (status != FLINT_OK) {
        throw std::runtime_error(std::string(what) + ": " + flint_last_error());
    }
}

// Times work on a stream with a pair of CUDA events.
class EventTimer {
public:
    EventTimer() {
        check_cuda(cudaEventCreate(&start_), "cudaEventCreate");
        check_cuda(cudaEventCreate(&stop_), "cudaEventCreate");
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    // Milliseconds for `count` launches enqueued back to back by `launch`, a
    // callable taking (stream, first, count) like harness.h's Launcher.
    template <typename Launch>
    double time(const Launch& launch, cudaStream_t stream, long long first, long long count) {
        check_cuda(cudaEventRecord(start_, stream), "cudaEventRecord");
        launch(stream, first, count);
        check_cuda(cudaEventRecord(stop_, stream), "cudaEventRecord");
        check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize");
        float ms = 0;
        check_cuda(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

// The median of the values; for an even count, the mean of the middle two.
inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}

// Device allocations freed together when the case ends.
class DeviceAllocations {
public:
    DeviceAllocations() = default;
    DeviceAllocations(const DeviceAllocations&) = delete;
    DeviceAllocations& operator=(const DeviceAllocations&) = delete;
    ~DeviceAllocations() {
        for (void* pointer : pointers_) {
            cudaFree(pointer);
        }
    }
    void* allocate(size_t bytes) {
        void* pointer = nullptr;
        check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
        pointers_.push_back(pointer);
        return pointer;
    }

private:
    std::vector<void*> pointers_;
};

// The splitmix64 generator: advances `state` and returns the next 64 bits.
inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// Whether two device buffers hold the same bytes, compared through two host
// buffers of kHostChunk / 2 each.
inline bool device_equal(const void* a, const void* b, size_t bytes) {
    std::vector<unsigned char> host_a(std::min(bytes, kHostChunk / 2));
    std::vector<unsigned char> host_b(host_a.size());
    for (size_t offset = 0; offset < bytes; offset += host_a.size()) {
        const size_t n = std::min(host_a.size(), bytes - offset);
        check_cuda(cudaMemcpy(host_a.data(), static_cast<const char*>(a) + offset, n,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        check_cuda(cudaMemcpy(host_b.data(), static_cast<const char*>(b) + offset, n,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        if (std::memcmp(host_a.data(), host_b.data(), n) != 0) {
            return false;
        }
    }
    return true;
}

}  // namespace bench
