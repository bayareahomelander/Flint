// Flint's measurement protocol (PLAN.md, Measurement): a Flint kernel timed
// against a baseline with a time-based warm-up, samples of at least ~100 ms,
// inputs rotated past 4x the L2 cache, and interleaved rounds, each judged by
// the NVML monitor.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "monitor.h"

namespace bench {

// Which of cuBLASLt's candidate algorithms an epilogue GEMM case uses as its
// baseline: the one that times fastest on the case's own buffers, or the
// heuristic's first choice, as PyTorch does.
enum class LtAlgo { Fastest, First };

// "fastest" or "first".
const char* lt_algo_name(LtAlgo selection);

struct Options {
    double warmup_seconds = 1.5;
    double sample_ms = 100;  // the shortest timed sample
    int rounds = 10;         // valid rounds required
    int max_rounds = 30;     // rounds tried before giving up
    bool allow_busy = false;
    bool quick = false;  // a shortened protocol for tests, never a real result
    LtAlgo lt_algo = LtAlgo::Fastest;  // gemm: the cuBLASLt baseline's algorithm
};

// Enqueues `count` back-to-back launches on `stream`. Launch i, counting from
// `first`, uses buffer set i % rotation_sets. Throws on failure.
using Launcher = std::function<void(cudaStream_t stream, long long first, long long count)>;

struct CaseSpec {
    std::string benchmark;      // such as "copy"
    std::string label;          // such as "16 MiB"
    std::string layout;         // such as "nn" for a gemm; empty otherwise
    std::string epilogue;       // "none", "bias", "bias-gelu", "dgelu" or "bias-residual" for a gemm; empty otherwise
    std::string baseline_name;  // such as "cudaMemcpyAsync"; empty when one-sided
    unsigned long long size_bytes = 0;
    double bytes_moved = 0;  // read plus written per launch; 0 if not bandwidth-bound
    double flops = 0;        // per launch; 0 if not compute-bound
    // The peak the FLOPs are judged against, per SM per clock; 0 if unknown.
    double peak_flop_per_sm_per_clock = 0;
    int rotation_sets = 1;
    Launcher flint;
    // Empty for a one-sided case, which has no baseline to interleave with
    // and is judged against the peaks alone.
    Launcher reference;
};

struct TimedSample {
    long long launches = 0;
    double ms_per_launch = 0;
    double sample_ms = 0;
    WindowStats stats;
};

struct Round {
    bool flint_first = false;
    TimedSample flint;
    TimedSample baseline;
    double ratio = 0;  // baseline time / Flint time: Flint's speed relative to the baseline
    bool valid = true;
    std::string discard_reason;
};

// One side's results over the valid rounds.
struct SideSummary {
    long long launches_per_sample = 0;  // at the end of the run
    double min_sample_ms = 0;           // the shortest valid sample
    double ms_per_launch = 0;           // median
    double gbs = 0;            // 0 unless the case moves bytes
    double rated_fraction = 0;           // of DeviceInfo::rated_bandwidth_gbs
    double clock_adjusted_fraction = 0;  // of 2 x mean memory clock x bus width
    double tflops = 0;                   // 0 unless the case counts FLOPs
    // Of peak_flop_per_sm_per_clock x SMs x the rated SM clock (0 where that
    // clock is unknown), and of the same at the mean SM clock in the samples.
    double rated_compute_fraction = 0;
    double clock_adjusted_compute_fraction = 0;
    WindowStats stats;  // merged, weighted by time
};

// What a GEMM case's output check measured: Flint's output against the
// baseline's, and Flint against itself.
struct OutputCheck {
    double rel_l2 = 0;                 // ||flint - baseline||_2 / ||baseline||_2
    double max_abs = 0;                // max |flint - baseline|
    double max_abs_baseline = 0;       // max |baseline|
    unsigned long long nonfinite = 0;  // NaN and infinite values, both sides together
    // Two Flint runs, into differently poisoned outputs, match bit for bit.
    bool deterministic = false;
};

// One of the algorithms cuBLASLt's heuristic offered for a case, with its
// configuration (-1 where cuBLASLt wouldn't report it) and its timing.
struct LtCandidate {
    int index = 0;  // in the heuristic's order, counting only the usable ones
    long long algo_id = -1;
    long long tile_id = -1;
    long long stages_id = -1;
    long long splitk_num = -1;
    long long reduction_scheme = -1;
    long long cta_swizzling = -1;
    long long custom_option = -1;
    unsigned long long workspace_bytes = 0;  // from the heuristic's result
    double waves_count = 0;                  // likewise
    std::string status;                      // "CUBLAS_STATUS_SUCCESS" or the launch error
    std::optional<double> ms_per_launch;     // median over a few short samples; unset if it failed
};

// How a cuBLASLt baseline's algorithm was chosen.
struct BaselineAlgorithm {
    LtAlgo selection = LtAlgo::Fastest;
    int used = 0;  // index into candidates
    std::vector<LtCandidate> candidates;
};

struct CaseResult {
    std::string benchmark;
    std::string label;
    std::string layout;  // a gemm's layout, "nt", "nn" or "tn"; empty for other benchmarks
    std::string epilogue;  // a gemm's epilogue, "none", "bias", "bias-gelu", "dgelu" or "bias-residual"; empty for other benchmarks
    std::string baseline_name;
    unsigned long long size_bytes = 0;
    double bytes_moved = 0;
    double flops = 0;
    double peak_flop_per_sm_per_clock = 0;
    int rotation_sets = 1;
    bool one_sided = false;     // no baseline: the ratio and baseline are unset
    double warmup_seconds = 0;  // spent, calibration included
    std::vector<Round> rounds;
    int valid_rounds = 0;
    bool enough_rounds = false;
    double ratio_median = 0;
    double ratio_min = 0;
    double ratio_max = 0;
    SideSummary flint;
    SideSummary baseline;
    // The case's output check passed. Copy's output equals the baseline's bit
    // for bit and mma's the exactly known result; a GEMM's outputs are finite,
    // the same on a second run, and within error bounds of the baseline's.
    bool outputs_match = false;
    // The numbers behind outputs_match for a case that compares numerically;
    // unset for the exact checks of copy and mma.
    std::optional<OutputCheck> output_check;
    // A bias-gelu case's second output, the sums before GELU, checked the same
    // way; output_check is then its GELU output.
    std::optional<OutputCheck> pre_check;
    // The cuBLASLt algorithms a GEMM epilogue case chose its baseline from;
    // unset for the cases without cuBLASLt.
    std::optional<BaselineAlgorithm> baseline_algorithm;
    bool ok() const { return enough_rounds && outputs_match; }
};

// How many buffer sets of bytes_per_set it takes to exceed 4x the L2 cache.
int rotation_sets(const DeviceInfo& device, double bytes_per_set);

// "16 MiB" for whole MiB, "4097 B" otherwise.
std::string size_label(unsigned long long bytes);

// Whether options meet PLAN.md's protocol: at least 10 rounds, a 1 s warm-up,
// and 100 ms samples, outside --quick. Only such runs may be published.
bool standard_protocol(const Options& options);

// Throws unless the GPU looked free of other users when the monitor started:
// no other processes, at most 256 MiB in use, NVML able to list processes,
// and zero utilization. With options.allow_busy it warns on stderr instead.
// Returns the utilization seen, in percent.
unsigned require_idle_gpu(const Monitor& monitor, const Options& options);

// Between cases, when our buffers are freed: throws (or warns, with
// allow_busy) if device memory grew more than 256 MiB past `reference`, the
// memory in use once our context existed, since a process NVML hides would
// otherwise be absorbed into the next case's baseline.
void require_memory_unchanged(const Monitor& monitor, const Options& options,
                              unsigned long long reference);

// Runs the protocol for one case. Allocate and fill the case's buffers first:
// device memory growing during the run is taken as another process.
CaseResult run_case(const CaseSpec& spec, const DeviceInfo& device, Monitor& monitor,
                    const Options& options, cudaStream_t stream);

}  // namespace bench
