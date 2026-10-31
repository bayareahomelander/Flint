// flint_bench's output: a console summary and the JSON raw log.
#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "harness.h"

namespace bench {

// Everything about a run except its cases. It holds no hostname or absolute
// path, since raw logs get committed.
struct RunInfo {
    std::string benchmark;
    std::vector<std::string> args;  // without argv[0]
    std::string git_sha;            // "unknown" outside a git checkout
    bool git_dirty = false;         // the tree had uncommitted changes
    std::string timestamp_utc;      // such as "2026-10-23T08:15:00Z"
    std::string os;                 // "windows" or "linux"
    std::string flint_version;
    DeviceInfo device;
    Options options;
    bool standard_protocol = false;  // see standard_protocol() in harness.h
    std::vector<unsigned> processes_at_start;  // other processes on the GPU
    unsigned long long used_bytes_at_start = 0;  // before our CUDA context
    unsigned utilization_at_start = 0;           // percent, while we were idle
    bool process_detection = false;              // NVML could list processes
};

void print_summary(std::ostream& out, const RunInfo& run, const std::vector<CaseResult>& cases);

// Writes the raw log, creating parent directories. Throws on failure.
void write_json(const std::string& path, const RunInfo& run, const std::vector<CaseResult>& cases);

// FLINT_BENCH_OUT_DIR/<benchmark>-<gpu>-<timestamp>.json
std::string default_json_path(const RunInfo& run);

}  // namespace bench
