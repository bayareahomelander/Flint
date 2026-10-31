#include "report.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#ifndef FLINT_BENCH_OUT_DIR
#define FLINT_BENCH_OUT_DIR "bench-results"
#endif

namespace bench {

namespace {

std::string fixed(double value, int decimals) {
    char text[64];
    std::snprintf(text, sizeof text, "%.*f", decimals, value);
    return text;
}

// 12040 -> "12.4"
std::string cuda_version(int version) {
    return std::to_string(version / 1000) + "." + std::to_string(version % 1000 / 10);
}

std::string scientific(double value) {
    char text[64];
    std::snprintf(text, sizeof text, "%.2e", value);
    return text;
}

std::string percent(double fraction) { return fixed(100 * fraction, 1) + "%"; }

std::string git_label(const RunInfo& run) {
    return run.git_sha.substr(0, 12) + (run.git_dirty ? "-dirty" : "");
}

std::string flags(const CaseResult& result) {
    std::string text;
    const auto add = [&text](const char* flag) {
        text += text.empty() ? "" : ",";
        text += flag;
    };
    const bool two_sided = !result.one_sided;
    if (result.flint.stats.thermal > 0 || (two_sided && result.baseline.stats.thermal > 0)) {
        add("thermal");
    }
    if (result.flint.stats.hw_slowdown > 0 || (two_sided && result.baseline.stats.hw_slowdown > 0)) {
        add("hw-slowdown");
    }
    if (result.flint.stats.failed_reads > 0 || (two_sided && result.baseline.stats.failed_reads > 0)) {
        add("nvml-errors");
    }
    if (!result.outputs_match) {
        add("OUTPUT MISMATCH");
    }
    if (!result.enough_rounds) {
        add("TOO FEW ROUNDS");
    }
    return text.empty() ? "-" : text;
}

std::vector<std::string> summary_row(const CaseResult& c) {
    const bool bandwidth = c.bytes_moved > 0;
    std::vector<std::string> row;
    // An epilogue shows beside the layout, as in "gemm nt+bias 8192x2304x768".
    row.push_back(c.benchmark + (c.layout.empty() ? "" : " " + c.layout) +
                  (c.epilogue.empty() || c.epilogue == "none" ? "" : "+" + c.epilogue) + " " +
                  c.label);
    if (c.one_sided) {
        row.push_back("flint " + fixed(c.flint.tflops, 1) + " TFLOPS");
    } else {
        const bool tflops = !bandwidth && c.flops > 0;
        row.push_back(bandwidth ? "flint " + fixed(c.flint.gbs, 1) + " GB/s"
                                : tflops ? "flint " + fixed(c.flint.tflops, 1) + " TFLOPS" : "flint");
        row.push_back(c.baseline_name +
                      (bandwidth ? " " + fixed(c.baseline.gbs, 1) + " GB/s"
                                 : tflops ? " " + fixed(c.baseline.tflops, 1) + " TFLOPS" : ""));
        row.push_back(percent(c.ratio_median) + " [" + fixed(100 * c.ratio_min, 1) + "-" +
                      fixed(100 * c.ratio_max, 1) + "]");
    }
    if (bandwidth) {
        row.push_back(percent(c.flint.rated_fraction) + " rated");
        row.push_back(percent(c.flint.clock_adjusted_fraction) + " clk-adj");
    }
    if (c.flops > 0) {
        // A two-sided row names the side, since both have fractions.
        const auto add_fractions = [&row](const std::string& prefix, const SideSummary& side) {
            row.push_back(prefix + (side.rated_compute_fraction > 0
                                        ? percent(side.rated_compute_fraction) + " rated"
                                        : "rated n/a"));
            row.push_back(prefix + percent(side.clock_adjusted_compute_fraction) + " clk-adj");
        };
        add_fractions(c.one_sided ? "" : "flint ", c.flint);
        if (!c.one_sided) {
            add_fractions(c.baseline_name + " ", c.baseline);
        }
    }
    if (c.output_check) {
        row.push_back("rel L2 " + scientific(c.output_check->rel_l2));
    }
    row.push_back(std::to_string(c.valid_rounds) + "/" + std::to_string(c.rounds.size()));
    const double min_sample_ms =
        c.one_sided ? c.flint.min_sample_ms : std::min(c.flint.min_sample_ms, c.baseline.min_sample_ms);
    row.push_back(fixed(min_sample_ms, 1) + " ms");
    row.push_back(flags(c));
    return row;
}

// Writes pretty-printed JSON with two-space indentation.
class JsonWriter {
public:
    explicit JsonWriter(std::ostream& out) : out_(out) {}

    void begin_object() { open('{'); }
    void end_object() { close('}'); }
    void begin_array() { open('['); }
    void end_array() { close(']'); }

    void key(const std::string& name) {
        separate();
        write_string(name);
        out_ << ": ";
        after_key_ = true;
    }

    void null() {
        before_value();
        out_ << "null";
    }

    template <class T>
    void value(const T& v) {
        before_value();
        if constexpr (std::is_same_v<T, bool>) {
            out_ << (v ? "true" : "false");
        } else if constexpr (std::is_floating_point_v<T>) {
            write_double(static_cast<double>(v));
        } else if constexpr (std::is_integral_v<T>) {
            out_ << v;
        } else {
            write_string(v);
        }
    }

    template <class T>
    void field(const std::string& name, const T& v) {
        key(name);
        value(v);
    }

private:
    void open(char bracket) {
        before_value();
        out_ << bracket;
        counts_.push_back(0);
    }

    void close(char bracket) {
        const int count = counts_.back();
        counts_.pop_back();
        if (count > 0) {
            out_ << '\n' << std::string(2 * counts_.size(), ' ');
        }
        out_ << bracket;
    }

    // The comma and line break before an element of the enclosing container.
    void separate() {
        if (counts_.back()++ > 0) {
            out_ << ',';
        }
        out_ << '\n' << std::string(2 * counts_.size(), ' ');
    }

    void before_value() {
        if (after_key_) {
            after_key_ = false;
        } else if (!counts_.empty()) {
            separate();
        }
    }

    void write_double(double v) {
        if (!std::isfinite(v)) {
            out_ << "null";
            return;
        }
        char text[40];
        std::snprintf(text, sizeof text, "%.17g", v);
        out_ << text;
    }

    void write_string(const std::string& text) {
        out_ << '"';
        for (const char c : text) {
            const unsigned char u = static_cast<unsigned char>(c);
            if (c == '"') {
                out_ << "\\\"";
            } else if (c == '\\') {
                out_ << "\\\\";
            } else if (u < 0x20) {
                char escape[8];
                std::snprintf(escape, sizeof escape, "\\u%04x", static_cast<unsigned>(u));
                out_ << escape;
            } else {
                out_ << c;
            }
        }
        out_ << '"';
    }

    std::ostream& out_;
    std::vector<int> counts_;  // elements written so far at each nesting level
    bool after_key_ = false;
};

void write_window(JsonWriter& json, const char* name, const WindowStats& s) {
    json.key(name);
    json.begin_object();
    json.field("seconds", s.seconds);
    json.field("samples", s.samples);
    json.field("failed_reads", s.failed_reads);
    json.field("sm_mhz", s.sm_mhz);
    json.field("mem_mhz", s.mem_mhz);
    json.field("power_w", s.power_w);
    json.field("temp_c_max", s.temp_c_max);
    json.field("thermal", s.thermal);
    json.field("hw_slowdown", s.hw_slowdown);
    json.field("sw_power_cap", s.sw_power_cap);
    json.key("reason_shares");
    json.begin_object();
    for (int i = 0; i < kReasonCount; ++i) {
        json.field(reason_name(i), s.reason_shares[i]);
    }
    json.end_object();
    json.field("new_process", s.new_process);
    json.field("used_bytes_max", s.used_bytes_max);
    json.end_object();
}

void write_side(JsonWriter& json, const char* name, const SideSummary& s) {
    json.key(name);
    json.begin_object();
    json.field("launches_per_sample", s.launches_per_sample);
    json.field("min_sample_ms", s.min_sample_ms);
    json.field("ms_per_launch", s.ms_per_launch);
    json.field("gbs", s.gbs);
    json.field("rated_fraction", s.rated_fraction);
    json.field("clock_adjusted_fraction", s.clock_adjusted_fraction);
    json.field("tflops", s.tflops);
    json.key("rated_compute_fraction");
    if (s.rated_compute_fraction > 0) {
        json.value(s.rated_compute_fraction);
    } else {
        json.null();
    }
    json.key("clock_adjusted_compute_fraction");
    if (s.clock_adjusted_compute_fraction > 0) {
        json.value(s.clock_adjusted_compute_fraction);
    } else {
        json.null();
    }
    write_window(json, "stats", s.stats);
    json.end_object();
}

void write_sample(JsonWriter& json, const char* name, const TimedSample& s) {
    json.key(name);
    json.begin_object();
    json.field("launches", s.launches);
    json.field("sample_ms", s.sample_ms);
    json.field("ms_per_launch", s.ms_per_launch);
    write_window(json, "stats", s.stats);
    json.end_object();
}

void write_round(JsonWriter& json, const Round& r, bool one_sided) {
    json.begin_object();
    json.field("flint_first", r.flint_first);
    json.key("ratio");
    if (one_sided) {
        json.null();
    } else {
        json.value(r.ratio);
    }
    json.field("valid", r.valid);
    json.key("discard_reason");
    if (r.valid) {
        json.null();
    } else {
        json.value(r.discard_reason);
    }
    write_sample(json, "flint", r.flint);
    if (one_sided) {
        json.key("baseline");
        json.null();
    } else {
        write_sample(json, "baseline", r.baseline);
    }
    json.end_object();
}

void write_output_check(JsonWriter& json, const char* name,
                        const std::optional<OutputCheck>& check) {
    json.key(name);
    if (!check) {
        json.null();
        return;
    }
    json.begin_object();
    json.field("rel_l2", check->rel_l2);
    json.field("max_abs", check->max_abs);
    json.field("max_abs_baseline", check->max_abs_baseline);
    json.field("nonfinite", check->nonfinite);
    json.field("deterministic", check->deterministic);
    json.end_object();
}

// A configuration value cuBLASLt wouldn't report is -1 in the candidate, null
// in the log.
void write_config(JsonWriter& json, const char* name, long long value) {
    json.key(name);
    if (value < 0) {
        json.null();
    } else {
        json.value(value);
    }
}

void write_baseline_algorithm(JsonWriter& json, const std::optional<BaselineAlgorithm>& algorithm) {
    json.key("baseline_algorithm");
    if (!algorithm) {
        json.null();
        return;
    }
    json.begin_object();
    json.field("selection", std::string(lt_algo_name(algorithm->selection)));
    json.field("used", algorithm->used);
    json.key("candidates");
    json.begin_array();
    for (const LtCandidate& candidate : algorithm->candidates) {
        json.begin_object();
        json.field("index", candidate.index);
        write_config(json, "algo_id", candidate.algo_id);
        write_config(json, "tile_id", candidate.tile_id);
        write_config(json, "stages_id", candidate.stages_id);
        write_config(json, "splitk_num", candidate.splitk_num);
        write_config(json, "reduction_scheme", candidate.reduction_scheme);
        write_config(json, "cta_swizzling", candidate.cta_swizzling);
        write_config(json, "custom_option", candidate.custom_option);
        json.field("workspace_bytes", candidate.workspace_bytes);
        json.field("waves_count", candidate.waves_count);
        json.field("status", candidate.status);
        json.key("ms_per_launch");
        if (candidate.ms_per_launch) {
            json.value(*candidate.ms_per_launch);
        } else {
            json.null();
        }
        json.end_object();
    }
    json.end_array();
    json.end_object();
}

void write_case(JsonWriter& json, const CaseResult& c) {
    json.begin_object();
    json.field("benchmark", c.benchmark);
    json.field("label", c.label);
    json.key("layout");
    if (c.layout.empty()) {
        json.null();
    } else {
        json.value(c.layout);
    }
    json.key("epilogue");
    if (c.epilogue.empty()) {
        json.null();
    } else {
        json.value(c.epilogue);
    }
    json.field("baseline_name", c.baseline_name);
    write_baseline_algorithm(json, c.baseline_algorithm);
    json.field("size_bytes", c.size_bytes);
    json.field("bytes_moved", c.bytes_moved);
    json.field("flops", c.flops);
    json.field("peak_flop_per_sm_per_clock", c.peak_flop_per_sm_per_clock);
    json.field("one_sided", c.one_sided);
    json.field("rotation_sets", c.rotation_sets);
    json.field("warmup_seconds", c.warmup_seconds);
    json.field("ok", c.ok());
    json.field("outputs_match", c.outputs_match);
    write_output_check(json, "output_check", c.output_check);
    write_output_check(json, "pre_check", c.pre_check);
    json.field("enough_rounds", c.enough_rounds);
    json.field("valid_rounds", c.valid_rounds);
    json.key("ratio");
    if (c.one_sided) {
        json.null();
    } else {
        json.begin_object();
        json.field("median", c.ratio_median);
        json.field("min", c.ratio_min);
        json.field("max", c.ratio_max);
        json.end_object();
    }
    write_side(json, "flint", c.flint);
    if (c.one_sided) {
        json.key("baseline");
        json.null();
    } else {
        write_side(json, "baseline", c.baseline);
    }
    json.key("rounds");
    json.begin_array();
    for (const Round& round : c.rounds) {
        write_round(json, round, c.one_sided);
    }
    json.end_array();
    json.end_object();
}

}  // namespace

void print_summary(std::ostream& out, const RunInfo& run, const std::vector<CaseResult>& cases) {
    const DeviceInfo& d = run.device;
    out << d.name << ", driver " << d.driver_version << ", CUDA runtime "
        << cuda_version(d.cuda_runtime_version) << " / driver " << cuda_version(d.cuda_driver_version)
        << ", rated " << fixed(d.rated_bandwidth_gbs, 1) << " GB/s, "
        << (d.rated_sm_clock_mhz > 0 ? "rated SM clock " + std::to_string(d.rated_sm_clock_mhz) + " MHz, " : "")
        << "git " << git_label(run) << "\n";
    if (!run.processes_at_start.empty()) {
        out << "warning: " << run.processes_at_start.size() << " other processes were on the GPU at start\n";
    }
    if (run.options.quick) {
        out << "quick run: shortened protocol, not a real result\n";
    } else if (!run.standard_protocol) {
        out << "warning: options below the standard protocol; not a publishable result\n";
    }

    std::vector<std::vector<std::string>> rows;
    for (const CaseResult& c : cases) {
        rows.push_back(summary_row(c));
    }
    std::vector<std::size_t> widths;
    for (const auto& row : rows) {
        widths.resize(std::max(widths.size(), row.size()), 0);
        for (std::size_t i = 0; i < row.size(); ++i) {
            widths[i] = std::max(widths[i], row[i].size());
        }
    }
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            out << (i == 0 ? "" : "  ") << row[i];
            if (i + 1 < row.size()) {
                out << std::string(widths[i] - row[i].size(), ' ');
            }
        }
        out << "\n";
    }
}

void write_json(const std::string& path, const RunInfo& run, const std::vector<CaseResult>& cases) {
    namespace fs = std::filesystem;
    const fs::path file(path);
    try {
        if (file.has_parent_path()) {
            fs::create_directories(file.parent_path());
        }
    } catch (const fs::filesystem_error& error) {
        throw std::runtime_error(std::string("cannot create the directory for ") + path + ": " +
                                 error.what());
    }
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open " + path + " for writing");
    }

    JsonWriter json(out);
    const DeviceInfo& d = run.device;
    json.begin_object();
    json.field("schema", "flint-bench/1");
    json.field("benchmark", run.benchmark);
    json.field("quick", run.options.quick);
    json.field("standard_protocol", run.standard_protocol);
    json.key("args");
    json.begin_array();
    for (const std::string& arg : run.args) {
        json.value(arg);
    }
    json.end_array();
    json.key("git");
    json.begin_object();
    json.field("sha", run.git_sha);
    json.field("dirty", run.git_dirty);
    json.end_object();
    json.field("timestamp_utc", run.timestamp_utc);
    json.field("os", run.os);
    json.field("flint_version", run.flint_version);

    json.key("device");
    json.begin_object();
    json.field("name", d.name);
    json.field("compute_capability", std::to_string(d.compute_major) + "." + std::to_string(d.compute_minor));
    json.field("sm_count", d.sm_count);
    json.field("l2_bytes", d.l2_bytes);
    json.field("bus_width_bits", d.bus_width_bits);
    json.field("max_sm_clock_mhz", d.max_sm_clock_mhz);
    json.field("max_mem_clock_mhz", d.max_mem_clock_mhz);
    json.field("rated_bandwidth_gbs", d.rated_bandwidth_gbs);
    json.key("rated_sm_clock_mhz");
    if (d.rated_sm_clock_mhz > 0) {
        json.value(d.rated_sm_clock_mhz);
    } else {
        json.null();
    }
    json.field("driver_version", d.driver_version);
    json.field("cuda_driver_version", d.cuda_driver_version);
    json.field("cuda_runtime_version", d.cuda_runtime_version);
    json.end_object();

    const Options& o = run.options;
    json.key("options");
    json.begin_object();
    json.field("warmup_seconds", o.warmup_seconds);
    json.field("sample_ms", o.sample_ms);
    json.field("rounds", o.rounds);
    json.field("max_rounds", o.max_rounds);
    json.field("allow_busy", o.allow_busy);
    json.field("quick", o.quick);
    json.field("lt_algo", std::string(lt_algo_name(o.lt_algo)));
    json.end_object();

    json.key("gpu_at_start");
    json.begin_object();
    json.field("other_processes", run.processes_at_start.size());  // a count: no PIDs in a raw log
    json.field("used_bytes", run.used_bytes_at_start);
    json.field("utilization_percent", run.utilization_at_start);
    json.field("process_detection", run.process_detection);
    json.end_object();

    json.key("cases");
    json.begin_array();
    for (const CaseResult& c : cases) {
        write_case(json, c);
    }
    json.end_array();
    json.end_object();
    out << "\n";

    out.flush();
    if (!out) {
        throw std::runtime_error("failed writing " + path);
    }
}

std::string default_json_path(const RunInfo& run) {
    std::string slug;
    for (const char c : run.device.name) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (std::isalnum(u)) {
            slug += static_cast<char>(std::tolower(u));
        } else if (!slug.empty() && slug.back() != '-') {
            slug += '-';
        }
    }
    if (!slug.empty() && slug.back() == '-') {
        slug.pop_back();
    }
    std::string stamp;
    for (const char c : run.timestamp_utc) {
        if (c != '-' && c != ':') {
            stamp += c;
        }
    }
    return std::string(FLINT_BENCH_OUT_DIR) + "/" + run.benchmark + "-" + slug + "-" + stamp + ".json";
}

}  // namespace bench
