#include "harness.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "common.h"

namespace bench {
namespace {

constexpr double kMiB = 1024.0 * 1024.0;
// A round is discarded if device memory grows past the case's own allocations
// by more than this, and the GPU counts as busy at start past this much.
constexpr double kMemoryGrowthLimit = 256 * kMiB;
// A sample shorter than this share of sample_ms discards its round.
constexpr double kShortestSample = 0.9;

void busy(const Options& options, const std::string& message) {
    if (!options.allow_busy) {
        throw std::runtime_error("the GPU is busy: " + message + " (pass --allow-busy to run anyway)");
    }
    std::cerr << "warning: the GPU is busy: " << message << "; running anyway (--allow-busy)\n";
}

WindowStats merge(const std::vector<WindowStats>& windows) {
    WindowStats merged;
    for (const WindowStats& w : windows) {
        merged.seconds += w.seconds;
        merged.samples += w.samples;
        merged.failed_reads += w.failed_reads;
        merged.sm_mhz += w.sm_mhz * w.seconds;
        merged.mem_mhz += w.mem_mhz * w.seconds;
        merged.power_w += w.power_w * w.seconds;
        merged.thermal += w.thermal * w.seconds;
        merged.hw_slowdown += w.hw_slowdown * w.seconds;
        merged.sw_power_cap += w.sw_power_cap * w.seconds;
        for (int i = 0; i < kReasonCount; ++i) {
            merged.reason_shares[i] += w.reason_shares[i] * w.seconds;
        }
        merged.temp_c_max = std::max(merged.temp_c_max, w.temp_c_max);
        merged.used_bytes_max = std::max(merged.used_bytes_max, w.used_bytes_max);
        merged.new_process = merged.new_process || w.new_process;
    }
    if (merged.seconds > 0) {
        for (double* mean : {&merged.sm_mhz, &merged.mem_mhz, &merged.power_w, &merged.thermal,
                             &merged.hw_slowdown, &merged.sw_power_cap}) {
            *mean /= merged.seconds;
        }
        for (double& share : merged.reason_shares) {
            share /= merged.seconds;
        }
    }
    return merged;
}

SideSummary summarize(long long launches_per_sample, const std::vector<const TimedSample*>& samples,
                      const CaseSpec& spec, const DeviceInfo& device) {
    SideSummary side;
    side.launches_per_sample = launches_per_sample;
    if (samples.empty()) {
        return side;
    }
    std::vector<double> ms_per_launch;
    std::vector<WindowStats> windows;
    side.min_sample_ms = samples.front()->sample_ms;
    for (const TimedSample* sample : samples) {
        ms_per_launch.push_back(sample->ms_per_launch);
        windows.push_back(sample->stats);
        side.min_sample_ms = std::min(side.min_sample_ms, sample->sample_ms);
    }
    side.ms_per_launch = median(ms_per_launch);
    side.stats = merge(windows);
    if (spec.bytes_moved > 0) {
        side.gbs = spec.bytes_moved / (side.ms_per_launch * 1e-3) / 1e9;
        if (device.rated_bandwidth_gbs > 0) {
            side.rated_fraction = side.gbs / device.rated_bandwidth_gbs;
        }
        const double clock_adjusted_gbs =
            2.0 * side.stats.mem_mhz * 1e6 * (device.bus_width_bits / 8) / 1e9;
        if (clock_adjusted_gbs > 0) {
            side.clock_adjusted_fraction = side.gbs / clock_adjusted_gbs;
        }
    }
    if (spec.flops > 0) {
        side.tflops = spec.flops / (side.ms_per_launch * 1e-3) / 1e12;
        const double per_mhz = spec.peak_flop_per_sm_per_clock * device.sm_count * 1e6 / 1e12;
        if (per_mhz > 0 && device.rated_sm_clock_mhz > 0) {
            side.rated_compute_fraction = side.tflops / (per_mhz * device.rated_sm_clock_mhz);
        }
        if (per_mhz > 0 && side.stats.sm_mhz > 0) {
            side.clock_adjusted_compute_fraction = side.tflops / (per_mhz * side.stats.sm_mhz);
        }
    }
    return side;
}

}  // namespace

const char* lt_algo_name(LtAlgo selection) {
    return selection == LtAlgo::First ? "first" : "fastest";
}

int rotation_sets(const DeviceInfo& device, double bytes_per_set) {
    return static_cast<int>(std::floor(4.0 * device.l2_bytes / bytes_per_set)) + 1;
}

bool standard_protocol(const Options& options) {
    return !options.quick && options.rounds >= 10 && options.warmup_seconds >= 1.0 &&
           options.sample_ms >= 100;
}

std::string size_label(unsigned long long bytes) {
    const unsigned long long mib = 1024ull * 1024;
    if (bytes >= mib && bytes % mib == 0) {
        return std::to_string(bytes / mib) + " MiB";
    }
    return std::to_string(bytes) + " B";
}

unsigned require_idle_gpu(const Monitor& monitor, const Options& options) {
    const unsigned utilization = monitor.utilization_at_start();
    std::vector<std::string> reasons;
    if (!monitor.processes_at_start().empty()) {
        reasons.push_back(std::to_string(monitor.processes_at_start().size()) + " other processes");
    }
    if (monitor.used_bytes_at_start() > kMemoryGrowthLimit) {
        reasons.push_back(std::to_string(static_cast<long long>(monitor.used_bytes_at_start() / kMiB)) +
                          " MiB in use before our context");
    }
    if (utilization > 0) {
        reasons.push_back(std::to_string(utilization) + "% utilization before we started");
    }
    if (!monitor.process_detection()) {
        reasons.push_back("NVML can't list processes, so others would go unseen");
    }
    if (!reasons.empty()) {
        std::string message = reasons[0];
        for (size_t i = 1; i < reasons.size(); ++i) {
            message += ", " + reasons[i];
        }
        busy(options, message);
    }
    return utilization;
}

void require_memory_unchanged(const Monitor& monitor, const Options& options,
                              unsigned long long reference) {
    const double grown = static_cast<double>(monitor.used_bytes_now()) - static_cast<double>(reference);
    if (grown > kMemoryGrowthLimit) {
        busy(options, "device memory grew by " + std::to_string(static_cast<long long>(grown / kMiB)) +
                          " MiB between cases");
    }
}

CaseResult run_case(const CaseSpec& spec, const DeviceInfo& device, Monitor& monitor,
                    const Options& options, cudaStream_t stream) {
    CaseResult result;
    result.benchmark = spec.benchmark;
    result.label = spec.label;
    result.layout = spec.layout;
    result.epilogue = spec.epilogue;
    result.baseline_name = spec.baseline_name;
    result.size_bytes = spec.size_bytes;
    result.bytes_moved = spec.bytes_moved;
    result.rotation_sets = spec.rotation_sets;

    // Setup copies may still run on other streams; nothing may overlap the timing.
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const double setup_memory = static_cast<double>(monitor.used_bytes_now());
    EventTimer timer;
    const Launcher* sides[2] = {&spec.flint, &spec.reference};
    const bool one_sided = !spec.reference;
    const int side_count = one_sided ? 1 : 2;
    result.one_sided = one_sided;
    result.flops = spec.flops;
    result.peak_flop_per_sm_per_clock = spec.peak_flop_per_sm_per_clock;
    long long next = 0;  // the next launch's rotation index, continuing across samples

    // Warm-up: alternate the sides for at least warmup_seconds, doubling each
    // side's launch count until a batch takes half a sample.
    const auto warmup_start = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - warmup_start).count();
    };
    long long counts[2] = {1, 1};
    double per_launch[2] = {0, 0};
    do {
        for (int i = 0; i < side_count; ++i) {
            const double ms = timer.time(*sides[i], stream, next, counts[i]);
            next += counts[i];
            per_launch[i] = std::max(ms, 1e-3) / counts[i];  // events resolve ~0.5 us
            if (ms < options.sample_ms / 2) {
                counts[i] *= 2;
            }
        }
    } while (elapsed() < options.warmup_seconds);

    // Calibration: size each side's sample from its estimate, time one sample at
    // that size, and refine, so short warm-ups don't leave samples too short.
    long long launches[2] = {1, 1};
    for (int i = 0; i < side_count; ++i) {
        for (int pass = 0; pass < 2; ++pass) {
            launches[i] = std::max(1LL, static_cast<long long>(
                                            std::ceil(options.sample_ms * 1.05 / per_launch[i])));
            const double ms = timer.time(*sides[i], stream, next, launches[i]);
            next += launches[i];
            per_launch[i] = std::max(ms, 1e-3) / launches[i];
        }
    }
    result.warmup_seconds = elapsed();

    // The memory baseline comes after warm-up, so lazily loaded modules and
    // library workspaces count as ours, but growth during warm-up past the
    // setup's own memory still means someone else.
    const double memory_baseline = static_cast<double>(monitor.used_bytes_now());
    if (memory_baseline > setup_memory + kMemoryGrowthLimit) {
        busy(options, "device memory grew by " +
                          std::to_string(static_cast<long long>((memory_baseline - setup_memory) / kMiB)) +
                          " MiB during warm-up");
    }

    // Times one sample; its window's stats are filled in after the round, so
    // the two samples of a round run back to back.
    auto timed_sample = [&](int side, double& begin, double& end) {
        TimedSample sample;
        sample.launches = launches[side];
        begin = monitor.now();
        sample.sample_ms = timer.time(*sides[side], stream, next, sample.launches);
        end = monitor.now();
        next += sample.launches;
        sample.ms_per_launch = sample.sample_ms / sample.launches;
        // Launches keep getting faster after calibration for small kernels, so
        // grow later samples back past sample_ms whenever one falls short.
        if (sample.sample_ms < options.sample_ms) {
            launches[side] = static_cast<long long>(
                std::ceil(sample.launches * options.sample_ms * 1.05 / std::max(sample.sample_ms, 1e-3)));
        }
        return sample;
    };

    // Interleaved rounds. Which side goes first alternates over valid rounds,
    // so a discarded round is redone in the same order. A one-sided round is
    // Flint's sample alone.
    for (int r = 0; r < options.max_rounds && result.valid_rounds < options.rounds; ++r) {
        Round round;
        round.flint_first = one_sided || result.valid_rounds % 2 == 0;
        double flint_begin = 0, flint_end = 0, baseline_begin = 0, baseline_end = 0;
        if (one_sided) {
            round.flint = timed_sample(0, flint_begin, flint_end);
        } else if (round.flint_first) {
            round.flint = timed_sample(0, flint_begin, flint_end);
            round.baseline = timed_sample(1, baseline_begin, baseline_end);
        } else {
            round.baseline = timed_sample(1, baseline_begin, baseline_end);
            round.flint = timed_sample(0, flint_begin, flint_end);
        }
        round.flint.stats = monitor.window(flint_begin, flint_end);
        if (!one_sided) {
            round.baseline.stats = monitor.window(baseline_begin, baseline_end);
            round.ratio = round.baseline.ms_per_launch / round.flint.ms_per_launch;
        }
        for (int side = 0; side < side_count; ++side) {
            const TimedSample* sample = side == 0 ? &round.flint : &round.baseline;
            if (!round.valid) {
                break;
            }
            if (sample->stats.new_process) {
                round.valid = false;
                round.discard_reason = "another process used the GPU";
            } else if (sample->stats.used_bytes_max > memory_baseline + kMemoryGrowthLimit) {
                round.valid = false;
                round.discard_reason =
                    "device memory grew by " +
                    std::to_string(static_cast<long long>(
                        (sample->stats.used_bytes_max - memory_baseline) / kMiB)) +
                    " MiB";
            } else if (sample->sample_ms < kShortestSample * options.sample_ms) {
                round.valid = false;
                round.discard_reason = "a sample ran " + std::to_string(sample->sample_ms) +
                                       " ms, short of the protocol's floor";
            }
        }
        result.valid_rounds += round.valid ? 1 : 0;
        result.rounds.push_back(round);
    }
    result.enough_rounds = result.valid_rounds >= options.rounds;

    std::vector<double> ratios;
    std::vector<const TimedSample*> flint_samples, baseline_samples;
    for (const Round& round : result.rounds) {
        if (!round.valid) {
            continue;
        }
        ratios.push_back(round.ratio);
        flint_samples.push_back(&round.flint);
        baseline_samples.push_back(&round.baseline);
    }
    if (!one_sided && !ratios.empty()) {
        result.ratio_median = median(ratios);
        result.ratio_min = *std::min_element(ratios.begin(), ratios.end());
        result.ratio_max = *std::max_element(ratios.begin(), ratios.end());
    }
    result.flint = summarize(launches[0], flint_samples, spec, device);
    if (!one_sided) {
        result.baseline = summarize(launches[1], baseline_samples, spec, device);
    }
    return result;
}

}  // namespace bench
