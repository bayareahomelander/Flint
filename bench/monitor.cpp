#include "monitor.h"

#include <algorithm>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "common.h"

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

namespace bench {
namespace {

constexpr auto kInterval = std::chrono::milliseconds(10);

// One bit per quantity a sample reads, for Sample::valid.
enum Field : unsigned {
    kSm = 1,
    kMem = 2,
    kPower = 4,
    kTemp = 8,
    kReasons = 16,
    kUsed = 32,
    kProcesses = 64,
};
constexpr int kFieldCount = 7;

constexpr const char* kReasonNames[kReasonCount] = {
    "gpu_idle",          "applications_clocks", "sw_power_cap",
    "hw_slowdown",       "sync_boost",          "sw_thermal_slowdown",
    "hw_thermal_slowdown", "hw_power_brake",    "display_clocks",
};

constexpr unsigned long long kThermal =
    nvmlClocksEventReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown;
constexpr unsigned long long kHwSlowdown =
    nvmlClocksThrottleReasonHwSlowdown | nvmlClocksThrottleReasonHwPowerBrakeSlowdown;

void check_nvml(nvmlReturn_t result, const char* what) {
    if (result != NVML_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + nvmlErrorString(result));
    }
}

void check_driver(CUresult result, const char* what) {
    if (result != CUDA_SUCCESS) {
        const char* name = "unknown error";
        cuGetErrorName(result, &name);
        throw std::runtime_error(std::string(what) + ": " + name);
    }
}

// Device memory in use. The v2 query leaves out what the driver reserves
// (527 MiB on the idle 2080 Ti, where v1 reports it as used); v1 is the
// fallback for drivers without v2.
bool read_used_bytes(nvmlDevice_t device, unsigned long long& used) {
    nvmlMemory_v2_t memory{};
    memory.version = nvmlMemory_v2;
    if (nvmlDeviceGetMemoryInfo_v2(device, &memory) == NVML_SUCCESS) {
        used = memory.used;
        return true;
    }
    nvmlMemory_t legacy{};
    if (nvmlDeviceGetMemoryInfo(device, &legacy) == NVML_SUCCESS) {
        used = legacy.used;
        return true;
    }
    return false;
}

int cuda_attribute(cudaDeviceAttr attribute, int device) {
    int value = 0;
    check_cuda(cudaDeviceGetAttribute(&value, attribute, device), "cudaDeviceGetAttribute");
    return value;
}

}  // namespace

const char* reason_name(int index) { return kReasonNames[index]; }

Monitor::Monitor(int cuda_device) : cuda_device_(cuda_device) {
    check_nvml(nvmlInit_v2(), "nvmlInit");
    // Found through the driver API, which creates no context, so the memory
    // read below is other processes' alone.
    char bus_id[32] = {};
    CUdevice device = 0;
    check_driver(cuInit(0), "cuInit");
    check_driver(cuDeviceGet(&device, cuda_device), "cuDeviceGet");
    check_driver(cuDeviceGetPCIBusId(bus_id, sizeof bus_id, device), "cuDeviceGetPCIBusId");
    check_nvml(nvmlDeviceGetHandleByPciBusId_v2(bus_id, &device_), "nvmlDeviceGetHandleByPciBusId");
    own_pid_ = static_cast<unsigned>(getpid());
    process_detection_ = other_processes(processes_at_start_);
    used_bytes_at_start_ = used_bytes_now();
    // Utilization is the share of NVML's last sampling period (1/6 to 1 s) in
    // which any kernel ran. We've launched nothing, so it's someone else's,
    // even a process NVML hides.
    for (int i = 0; i < 5; ++i) {
        nvmlUtilization_t utilization{};
        check_nvml(nvmlDeviceGetUtilizationRates(device_, &utilization),
                   "nvmlDeviceGetUtilizationRates");
        utilization_at_start_ = std::max(utilization_at_start_, utilization.gpu);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    start_ = std::chrono::steady_clock::now();
    thread_ = std::thread([this] { run(); });
}

Monitor::~Monitor() {
    stop_ = true;
    thread_.join();
    nvmlShutdown();
}

DeviceInfo Monitor::device_info() const {
    DeviceInfo info;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, cuda_device_), "cudaGetDeviceProperties");
    info.name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.sm_count = cuda_attribute(cudaDevAttrMultiProcessorCount, cuda_device_);
    info.l2_bytes = cuda_attribute(cudaDevAttrL2CacheSize, cuda_device_);
    info.bus_width_bits = cuda_attribute(cudaDevAttrGlobalMemoryBusWidth, cuda_device_);
    check_nvml(nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_SM, &info.max_sm_clock_mhz),
               "nvmlDeviceGetMaxClockInfo");
    check_nvml(nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_MEM, &info.max_mem_clock_mhz),
               "nvmlDeviceGetMaxClockInfo");
    // NVML's memory clock is half the data rate, as for the 2080 Ti's 7000 MHz
    // and 616 GB/s.
    info.rated_bandwidth_gbs = 2.0 * info.max_mem_clock_mhz * 1e6 * (info.bus_width_bits / 8) / 1e9;
    // The 2080 Ti's rated 53.8 TFLOPS assumes its 1545 MHz boost clock.
    if (info.name == "NVIDIA GeForce RTX 2080 Ti") {
        info.rated_sm_clock_mhz = 1545;
    }
    char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
    check_nvml(nvmlSystemGetDriverVersion(driver, sizeof driver), "nvmlSystemGetDriverVersion");
    info.driver_version = driver;
    check_cuda(cudaDriverGetVersion(&info.cuda_driver_version), "cudaDriverGetVersion");
    check_cuda(cudaRuntimeGetVersion(&info.cuda_runtime_version), "cudaRuntimeGetVersion");
    return info;
}

double Monitor::now() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
}

unsigned long long Monitor::used_bytes_now() const {
    unsigned long long used = 0;
    if (!read_used_bytes(device_, used)) {
        throw std::runtime_error("nvmlDeviceGetMemoryInfo failed");
    }
    return used;
}

bool Monitor::other_processes(std::vector<unsigned>& pids) const {
    using Query = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*);
    pids.clear();
    bool any_list = false;
    for (Query query : {Query{nvmlDeviceGetComputeRunningProcesses_v3},
                        Query{nvmlDeviceGetGraphicsRunningProcesses_v3}}) {
        std::vector<nvmlProcessInfo_t> infos(64);
        unsigned count = static_cast<unsigned>(infos.size());
        nvmlReturn_t result = query(device_, &count, infos.data());
        if (result == NVML_ERROR_INSUFFICIENT_SIZE) {
            infos.resize(count + 16);  // room for processes that start meanwhile
            count = static_cast<unsigned>(infos.size());
            result = query(device_, &count, infos.data());
        }
        // Some drivers don't support one of the lists; the other still counts.
        if (result != NVML_SUCCESS) {
            continue;
        }
        any_list = true;
        for (unsigned i = 0; i < count; ++i) {
            if (infos[i].pid != own_pid_) {
                pids.push_back(infos[i].pid);
            }
        }
    }
    std::sort(pids.begin(), pids.end());
    pids.erase(std::unique(pids.begin(), pids.end()), pids.end());
    return any_list;
}

void Monitor::run() {
    std::vector<unsigned> pids;
    auto deadline = std::chrono::steady_clock::now();
    while (!stop_) {
        Sample sample;
        unsigned value = 0;
        if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &value) == NVML_SUCCESS) {
            sample.sm_mhz = value;
            sample.valid |= kSm;
        }
        if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_MEM, &value) == NVML_SUCCESS) {
            sample.mem_mhz = value;
            sample.valid |= kMem;
        }
        // Stamped between the clock reads and the slower ones that follow.
        sample.t = now();
        if (nvmlDeviceGetPowerUsage(device_, &value) == NVML_SUCCESS) {
            sample.power_mw = value;
            sample.valid |= kPower;
        }
        if (nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS) {
            sample.temp_c = value;
            sample.valid |= kTemp;
        }
        if (nvmlDeviceGetCurrentClocksEventReasons(device_, &sample.reasons) == NVML_SUCCESS) {
            sample.valid |= kReasons;
        }
        if (read_used_bytes(device_, sample.used_bytes)) {
            sample.valid |= kUsed;
        }
        if (other_processes(pids)) {
            sample.valid |= kProcesses;
            for (unsigned pid : pids) {
                if (!std::binary_search(processes_at_start_.begin(), processes_at_start_.end(), pid)) {
                    sample.new_process = true;
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            samples_.push_back(sample);
        }
        sampled_.notify_all();
        // Fixed deadlines keep the cadence near 10 ms where the OS timer
        // allows; Windows' default 15.6 ms tick stretches it a little.
        deadline += kInterval;
        std::this_thread::sleep_until(deadline);
    }
}

WindowStats Monitor::window(double begin, double end) const {
    std::unique_lock<std::mutex> lock(mutex_);
    sampled_.wait(lock, [&] { return !samples_.empty() && samples_.back().t >= end; });

    // The sample in effect at `begin` is the last one taken at or before it.
    auto first = std::upper_bound(samples_.begin(), samples_.end(), begin,
                                  [](double t, const Sample& s) { return t < s.t; });
    if (first != samples_.begin()) {
        --first;
    }
    WindowStats stats;
    stats.seconds = end - begin;
    double sm_time = 0, mem_time = 0, power_time = 0, reason_time = 0;
    for (auto it = first; it != samples_.end() && it->t < end; ++it) {
        const double from = std::max(it->t, begin);
        const double to = (it + 1 != samples_.end()) ? std::min((it + 1)->t, end) : end;
        const double weight = std::max(to - from, 0.0);
        ++stats.samples;
        for (int bit = 0; bit < kFieldCount; ++bit) {
            stats.failed_reads += (it->valid >> bit & 1) ? 0 : 1;
        }
        if (it->valid & kSm) {
            stats.sm_mhz += it->sm_mhz * weight;
            sm_time += weight;
        }
        if (it->valid & kMem) {
            stats.mem_mhz += it->mem_mhz * weight;
            mem_time += weight;
        }
        if (it->valid & kPower) {
            stats.power_w += it->power_mw / 1000.0 * weight;
            power_time += weight;
        }
        if (it->valid & kReasons) {
            for (int bit = 0; bit < kReasonCount; ++bit) {
                stats.reason_shares[bit] += (it->reasons >> bit & 1) ? weight : 0;
            }
            stats.thermal += (it->reasons & kThermal) ? weight : 0;
            stats.hw_slowdown += (it->reasons & kHwSlowdown) ? weight : 0;
            stats.sw_power_cap += (it->reasons & nvmlClocksEventReasonSwPowerCap) ? weight : 0;
            reason_time += weight;
        }
        if (it->valid & kTemp) {
            stats.temp_c_max = std::max(stats.temp_c_max, it->temp_c);
        }
        if (it->valid & kUsed) {
            stats.used_bytes_max = std::max(stats.used_bytes_max, it->used_bytes);
        }
        stats.new_process = stats.new_process || it->new_process;
    }
    auto normalize = [](double& total, double time) { total = time > 0 ? total / time : 0; };
    normalize(stats.sm_mhz, sm_time);
    normalize(stats.mem_mhz, mem_time);
    normalize(stats.power_w, power_time);
    for (double& share : stats.reason_shares) {
        normalize(share, reason_time);
    }
    normalize(stats.thermal, reason_time);
    normalize(stats.hw_slowdown, reason_time);
    normalize(stats.sw_power_cap, reason_time);
    return stats;
}

}  // namespace bench
