// flint_bench: times Flint's kernels against their baselines following
// PLAN.md's measurement protocol, and writes a JSON raw log.
#include <ctime>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>

#include "cli.h"
#include "common.h"
#include "copy_bench.h"
#include "flint/flint.h"
#include "gemm_bench.h"
#include "git_info.h"
#include "mma_bench.h"
#include "report.h"

namespace {

std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char text[32];
    std::strftime(text, sizeof text, "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace bench;
    try {
        const CommandLine command = parse_command_line(argc, argv);
        if (command.help) {
            std::cout << usage();
            return 0;
        }
        // The monitor reads memory and utilization before our context exists;
        // since CUDA 12, cudaSetDevice already creates it.
        Monitor monitor(0);
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        check_cuda(cudaFree(nullptr), "creating the CUDA context");
        // cuBLAS's and cuBLASLt's handles come before the memory baseline, so
        // their workspaces count as ours.
        std::optional<CublasHandle> cublas;
        std::optional<CublasLtHandle> cublas_lt;
        if (command.benchmark == "gemm") {
            cublas.emplace();
            cublas_lt.emplace();
        }
        const unsigned long long reference = monitor.used_bytes_now();

        RunInfo run;
        run.benchmark = command.benchmark;
        run.args = command.args;
        run.git_sha = FLINT_GIT_SHA;
        run.git_dirty = FLINT_GIT_DIRTY != 0;
        run.timestamp_utc = utc_timestamp();
#ifdef _WIN32
        run.os = "windows";
#else
        run.os = "linux";
#endif
        run.flint_version = flint_version();
        run.device = monitor.device_info();
        run.options = command.options;
        run.standard_protocol = standard_protocol(command.options);
        run.processes_at_start = monitor.processes_at_start();
        run.used_bytes_at_start = monitor.used_bytes_at_start();
        run.process_detection = monitor.process_detection();
        run.utilization_at_start = require_idle_gpu(monitor, command.options);

        cudaStream_t stream = nullptr;
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        std::vector<CaseResult> cases;
        if (command.benchmark == "copy") {
            for (unsigned long long size : command.sizes) {
                require_memory_unchanged(monitor, command.options, reference);
                cases.push_back(bench_copy(size, run.device, monitor, command.options, stream));
            }
        } else if (command.benchmark == "gemm") {
            for (const GemmShape& shape : command.shapes) {
                require_memory_unchanged(monitor, command.options, reference);
                cases.push_back(bench_gemm(shape, command.layout, command.epilogue, cublas->get(),
                                           cublas_lt->get(), run.device, monitor, command.options,
                                           stream));
            }
        } else {
            for (int chains : command.chains) {
                require_memory_unchanged(monitor, command.options, reference);
                cases.push_back(bench_mma(chains, run.device, monitor, command.options, stream));
            }
        }
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

        print_summary(std::cout, run, cases);
        const std::string path = command.out.empty() ? default_json_path(run) : command.out;
        write_json(path, run, cases);
        std::cout << "wrote " << path << "\n";
        // Under compute-sanitizer on Windows, the process can end without
        // flushing std::cout, and the summary would be lost.
        std::cout.flush();

        for (const CaseResult& result : cases) {
            if (!result.ok()) {
                return 1;
            }
        }
        return 0;
    } catch (const std::invalid_argument& error) {
        std::cerr << "flint_bench: " << error.what() << "\n\n" << usage();
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "flint_bench: " << error.what() << "\n";
        return 1;
    }
}
