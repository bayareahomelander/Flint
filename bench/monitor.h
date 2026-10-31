// A background NVML sampler: clocks, power, temperature, clock-event reasons,
// device memory, and other processes on the GPU, used to judge each timed
// window of a benchmark.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nvml.h>

namespace bench {

struct DeviceInfo {
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    int sm_count = 0;
    long long l2_bytes = 0;
    int bus_width_bits = 0;
    unsigned max_sm_clock_mhz = 0;
    unsigned max_mem_clock_mhz = 0;
    // The boost clock behind the vendor's rated compute peak, from a table of
    // known GPUs; 0 if unknown, as for the laptop, whose clocks vary by design.
    unsigned rated_sm_clock_mhz = 0;
    double rated_bandwidth_gbs = 0;  // 2 x max memory clock x bus width
    std::string driver_version;
    int cuda_driver_version = 0;   // such as 13020
    int cuda_runtime_version = 0;  // such as 12040
};

// NVML's clock-event reasons, in bit order, and their names in the raw log.
constexpr int kReasonCount = 9;
const char* reason_name(int index);

// The monitor's samples summarized over a time window. Means and shares are
// weighted by time, holding each sample until the next, and use only the
// samples whose read of that quantity succeeded.
struct WindowStats {
    double seconds = 0;
    int samples = 0;
    int failed_reads = 0;  // NVML reads that failed, over all quantities
    double sm_mhz = 0;
    double mem_mhz = 0;
    double power_w = 0;  // NVML may report it averaged over about a second
    unsigned temp_c_max = 0;
    std::array<double, kReasonCount> reason_shares{};
    // The shares the summary flags, and the power cap it doesn't.
    double thermal = 0;       // software or hardware thermal slowdown
    double hw_slowdown = 0;   // hardware slowdown or power brake
    double sw_power_cap = 0;  // normal on a power-capped card
    bool new_process = false;  // a process absent at start used the GPU
    unsigned long long used_bytes_max = 0;
};

class Monitor {
public:
    // Opens NVML for the given CUDA device and starts sampling every ~10 ms.
    // Construct it before creating the CUDA context, so the memory in use at
    // start is other processes' alone.
    explicit Monitor(int cuda_device);
    ~Monitor();
    Monitor(const Monitor&) = delete;
    Monitor& operator=(const Monitor&) = delete;

    DeviceInfo device_info() const;

    // Seconds since the monitor started, on the clock its samples use.
    double now() const;

    // Summarizes [begin, end], waiting for the first sample after end.
    WindowStats window(double begin, double end) const;

    // Other processes on the GPU, device memory in use, and the highest
    // utilization over half a second, all read before sampling began.
    const std::vector<unsigned>& processes_at_start() const { return processes_at_start_; }
    unsigned long long used_bytes_at_start() const { return used_bytes_at_start_; }
    unsigned utilization_at_start() const { return utilization_at_start_; }
    // False if neither NVML process list can be read, so new processes go unseen.
    bool process_detection() const { return process_detection_; }

    // Device memory in use right now, read directly.
    unsigned long long used_bytes_now() const;

private:
    struct Sample {
        double t = 0;
        unsigned valid = 0;  // which reads succeeded, one bit per Field
        unsigned sm_mhz = 0;
        unsigned mem_mhz = 0;
        unsigned power_mw = 0;
        unsigned temp_c = 0;
        unsigned long long reasons = 0;
        unsigned long long used_bytes = 0;
        bool new_process = false;
    };

    // Other processes' PIDs, or false if no list could be read.
    bool other_processes(std::vector<unsigned>& pids) const;
    void run();

    nvmlDevice_t device_{};
    int cuda_device_ = 0;
    unsigned own_pid_ = 0;
    std::vector<unsigned> processes_at_start_;  // sorted
    unsigned long long used_bytes_at_start_ = 0;
    unsigned utilization_at_start_ = 0;
    bool process_detection_ = false;
    std::chrono::steady_clock::time_point start_;

    mutable std::mutex mutex_;
    mutable std::condition_variable sampled_;
    std::vector<Sample> samples_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

}  // namespace bench
