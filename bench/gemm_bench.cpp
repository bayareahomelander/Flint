#include "gemm_bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "common.h"
#include "mma_bench.h"

namespace bench {
namespace {

constexpr double kInputSigmaX = 1.0;
constexpr double kInputSigmaW = 0.02;
constexpr uint64_t kSeedX = 0x5eed0001;
constexpr uint64_t kSeedW = 0x5eed0002;
constexpr uint64_t kSeedY = 0x5eed0003;  // tn's initial y, N(0, 1)
constexpr uint64_t kSeedBias = 0x5eed0004;
constexpr double kBiasSigma = 0.5;
// dgelu's pre is an MLP's pre-activations, about N(0, 2^2): a spread that puts
// most of its values where GELU's derivative is neither 0 nor 1.
constexpr uint64_t kSeedPre = 0x5eed0005;
constexpr double kPreSigma = 2.0;
// bias-residual's residual stream: FP32, about N(0, 1).
constexpr uint64_t kSeedResidual = 0x5eed0006;
constexpr double kResidualSigma = 1.0;
// The workspace cuBLASLt may use for an epilogue case: cuBLAS's own default size
// on these pre-Hopper GPUs.
constexpr size_t kLtWorkspaceBytes = size_t{4} << 20;
// How many algorithms to ask cuBLASLt's heuristic for. PyTorch takes the first;
// the baseline here is the fastest of these, or the first with --lt-algo first.
constexpr int kLtCandidates = 8;
// Two GEMMs that accumulate in FP32 and round once differ by at most one FP16
// ulp, in a small fraction of the outputs. The check allows 2^-10 relative L2
// and 4 ulps of the largest output.
// With a bias, cuBLASLt adds it rounded to FP16 and Flint adds the FP32 value,
// so the outputs can also differ by that rounding; the same bounds apply, to
// the GELU outputs and the sums before GELU alike. With dgelu, cuBLASLt seems
// to round the sums to FP16 before multiplying by the derivative where Flint
// rounds once, which is one more FP16 rounding of difference; the same bounds
// apply.
constexpr double kMaxRelL2 = 1.0 / 1024;
constexpr double kMaxUlps = 4;
// tn's outputs are FP32, so the ulp bound doesn't apply. Both sides accumulate
// in FP32, in different orders and splits, onto the same initial y. The
// tensor cores' truncating accumulation makes the difference grow about in
// proportion to k. On the GPT-2 shapes (k = 8192) it measured up to 1.0e-5
// relative L2 and 1.3e-5 max-abs as a fraction of the largest cuBLAS output;
// at k = 65536 up to 1.0e-4 and 1.1e-4. The bounds are about 5x those, and
// scale with k above 8192.
constexpr double kTnMaxRelL2 = 5e-5;
constexpr double kTnMaxAbsFraction = 6e-5;
constexpr double kTnBoundK = 8192;
// bias-residual's outputs are FP32 too. cuBLASLt adds the bias and the residual
// in an order that depends on the algorithm it picks, (s + r) + b or
// (s + b) + r, which differ by an FP32 ulp of y in some elements, and it
// accumulates in another order, or splits k. The truncating accumulation makes
// that difference grow about in proportion to k. On the RTX 4050 it measured 0
// (bit for bit) on the GPT-2 shapes; up to 4.4e-8 relative L2 and 1.0e-7
// max-abs as a fraction of the largest cuBLASLt output (an ulp in some
// elements) on shapes where it accumulates as Flint does; and, where it
// doesn't, 7.9e-7 and 1.0e-6 at k = 1024, 2.8e-6 and 3.3e-6 at 3072, 1.1e-5
// and 1.4e-5 at 8192 (tiles few enough to split), and 1.2e-4 and 1.2e-4 at
// 65536. The bounds are a floor of 3e-7 and 6e-7, for the ulps, plus 5e-9 and
// 1e-8 per unit of k: at least 2.7x over all of those.
constexpr double kResidualRelL2Floor = 3e-7;
constexpr double kResidualRelL2PerK = 5e-9;
constexpr double kResidualMaxAbsFloor = 6e-7;
constexpr double kResidualMaxAbsPerK = 1e-8;

void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + cublasGetStatusName(status));
    }
}

// cublasLtMatmul with an epilogue, using the same column-major view of the
// row-major buffers as bench_gemm's cublasGemmEx call, and one algorithm, chosen
// up front from the heuristic's candidates (the first until use() says
// otherwise):
//   kBias      y = x w^T + bias, with an FP16 bias (the FP32 bias Flint reads,
//              rounded to FP16);
//   kBiasGelu  y = GELU(x w^T + bias), the sums before GELU stored too in an aux
//              output;
//   kDGelu     y = (x w) GELU'(aux), GELU's backward pass, with the forward's
//              sums before GELU as an aux input;
//   kBiasResidual  y = residual + (x w^T + bias), with an FP32 bias and FP32 y
//              and residual (the matmul's C, with beta = 1, in a buffer of its
//              own: D is a different pointer).
class LtGemm {
public:
    enum class Kind { kBias, kBiasGelu, kDGelu, kBiasResidual };

    // Throws if cuBLASLt has no algorithm for the case. It keeps up to
    // kLtCandidates of the heuristic's, the usable ones. bias holds n FP16
    // values (kBias and kBiasGelu) or FP32 ones (kBiasResidual). The aux buffer
    // (kBiasGelu and kDGelu) is m x n FP16 like y, the one run() is given, and
    // `aux` is the first one (the algorithm is chosen with it set). With
    // kBiasResidual the aux buffer is the residual, m x n FP32 like y.
    LtGemm(cublasLtHandle_t lt, Kind kind, size_t m, size_t n, size_t k, const void* bias,
           void* aux, void* workspace)
        : lt_(lt), kind_(kind), workspace_(workspace) {
        try {
            create(m, n, k, bias, aux);
        } catch (...) {
            destroy();
            throw;
        }
    }
    ~LtGemm() { destroy(); }
    LtGemm(const LtGemm&) = delete;
    LtGemm& operator=(const LtGemm&) = delete;

    // The candidates, in the heuristic's order, with their configurations; the
    // caller fills in the timings.
    const std::vector<LtCandidate>& candidates() const { return candidates_; }

    // Makes candidate i the algorithm run() and launch() use.
    void use(size_t i) { algo_ = algos_.at(i); }

    // y, and aux (ignored with kBias; with kBiasResidual the residual, which is
    // only read).
    void run(const void* x, const void* w, void* y, void* aux, cudaStream_t stream) {
        check_cublas(launch(x, w, y, aux, stream), "cublasLtMatmul");
    }

    // run() that returns cublasLtMatmul's status instead of throwing on it.
    cublasStatus_t launch(const void* x, const void* w, void* y, void* aux, cudaStream_t stream) {
        const bool residual = kind_ == Kind::kBiasResidual;
        if (kind_ == Kind::kBiasGelu || kind_ == Kind::kDGelu) {
            set(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, aux,
                "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER");
        }
        const float alpha = 1.0f;
        const float beta = residual ? 1.0f : 0.0f;
        return cublasLtMatmul(lt_, op_, &alpha, w, a_, x, b_, &beta, residual ? aux : y,
                              residual ? c_ : d_, y, d_, &algo_, workspace_, kLtWorkspaceBytes,
                              stream);
    }

private:
    template <typename T>
    void set(cublasLtMatmulDescAttributes_t attribute, const T& value, const char* what) {
        check_cublas(cublasLtMatmulDescSetAttribute(op_, attribute, &value, sizeof value), what);
    }

    void create(size_t m, size_t n, size_t k, const void* bias, void* aux) {
        check_cublas(cublasLtMatmulDescCreate(&op_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                     "cublasLtMatmulDescCreate");
        const bool dgelu = kind_ == Kind::kDGelu;
        const bool residual = kind_ == Kind::kBiasResidual;
        set(CUBLASLT_MATMUL_DESC_TRANSA, dgelu ? CUBLAS_OP_N : CUBLAS_OP_T,
            "CUBLASLT_MATMUL_DESC_TRANSA");
        set(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N, "CUBLASLT_MATMUL_DESC_TRANSB");
        set(CUBLASLT_MATMUL_DESC_EPILOGUE,
            kind_ == Kind::kBiasGelu ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS
            : dgelu                  ? CUBLASLT_EPILOGUE_DGELU
                                     : CUBLASLT_EPILOGUE_BIAS,
            "CUBLASLT_MATMUL_DESC_EPILOGUE");
        if (!dgelu) {
            set(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias, "CUBLASLT_MATMUL_DESC_BIAS_POINTER");
            // CUDA 12.4's cuBLASLt rejects an FP32 bias with FP16 output (the
            // heuristic returns CUBLAS_STATUS_INVALID_VALUE), so the FP16
            // output cases read the bias rounded to FP16, as PyTorch's
            // autocast linear does. Reading n halves instead of n floats
            // doesn't change their time measurably. With FP32 output, as in
            // kBiasResidual, it takes the FP32 bias.
            set(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, residual ? CUDA_R_32F : CUDA_R_16F,
                "CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE");
        }
        if (kind_ == Kind::kBiasGelu || dgelu) {
            // The aux buffer has D's layout: column-major n x m, ld n. It is
            // the sums before GELU, which cuBLASLt keeps in FP16: stored by
            // kBiasGelu and read by kDGelu.
            set(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, aux,
                "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER");
            set(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, static_cast<int64_t>(n),
                "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD");
            set(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, CUDA_R_16F,
                "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE");
        }

        // NT: w's buffer read column-major is k x n (ld k), and OP_T makes it
        // the n x k A. NN (kDGelu): w's is n x k (ld n), A as it is. x's is
        // k x m (ld k), the B. D is y^T, n x m (ld n), and C is D, except that
        // kBiasResidual's C is the residual, a separate n x m (ld n) matrix
        // like D, and both are FP32.
        const auto layout = [](cublasLtMatrixLayout_t& out, cudaDataType type, size_t rows,
                               size_t cols) {
            check_cublas(cublasLtMatrixLayoutCreate(&out, type, rows, cols,
                                                    static_cast<int64_t>(rows)),
                         "cublasLtMatrixLayoutCreate");
        };
        if (dgelu) {
            layout(a_, CUDA_R_16F, n, k);
        } else {
            layout(a_, CUDA_R_16F, k, n);
        }
        layout(b_, CUDA_R_16F, k, m);
        layout(d_, residual ? CUDA_R_32F : CUDA_R_16F, n, m);
        if (residual) {
            layout(c_, CUDA_R_32F, n, m);
        }

        // Split-K parts are never summed in FP16: only the FP32 reductions are
        // allowed (none, or in the compute type). This is the cuBLASLt
        // counterpart of the CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
        // flag on the cuBLAS handle; INPLACE and OUTPUT_TYPE would sum in FP16.
        check_cublas(cublasLtMatmulPreferenceCreate(&preference_),
                     "cublasLtMatmulPreferenceCreate");
        const uint64_t workspace_bytes = kLtWorkspaceBytes;
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
                         sizeof workspace_bytes),
                     "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES");
        const uint32_t reductions =
            CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
        check_cublas(cublasLtMatmulPreferenceSetAttribute(
                         preference_, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reductions,
                         sizeof reductions),
                     "CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK");

        cublasLtMatmulHeuristicResult_t found[kLtCandidates] = {};
        int returned = 0;
        check_cublas(cublasLtMatmulAlgoGetHeuristic(lt_, op_, a_, b_, residual ? c_ : d_, d_,
                                                    preference_, kLtCandidates, found, &returned),
                     "cublasLtMatmulAlgoGetHeuristic");
        for (int i = 0; i < returned; ++i) {
            if (found[i].state != CUBLAS_STATUS_SUCCESS) {
                continue;
            }
            const cublasLtMatmulAlgo_t& algo = found[i].algo;
            LtCandidate candidate;
            candidate.index = static_cast<int>(algos_.size());
            candidate.algo_id = config(algo, CUBLASLT_ALGO_CONFIG_ID);
            candidate.tile_id = config(algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
            candidate.stages_id = config(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
            candidate.splitk_num = config(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
            candidate.reduction_scheme = config(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
            candidate.cta_swizzling = config(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
            candidate.custom_option = config(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
            candidate.workspace_bytes = found[i].workspaceSize;
            candidate.waves_count = found[i].wavesCount;
            candidate.status = cublasGetStatusName(CUBLAS_STATUS_SUCCESS);
            algos_.push_back(algo);
            candidates_.push_back(candidate);
        }
        if (algos_.empty()) {
            throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic: no algorithm for the case");
        }
        algo_ = algos_.front();
    }

    // One of an algorithm's configuration values, or -1 if cuBLASLt won't
    // report it.
    static long long config(const cublasLtMatmulAlgo_t& algo,
                            cublasLtMatmulAlgoConfigAttributes_t attribute) {
        uint32_t value = 0;
        size_t written = 0;
        if (cublasLtMatmulAlgoConfigGetAttribute(&algo, attribute, &value, sizeof value,
                                                 &written) != CUBLAS_STATUS_SUCCESS ||
            written != sizeof value) {
            return -1;
        }
        return value;
    }

    // Teardown can't throw, and a failure here changes nothing.
    void destroy() {
        if (preference_ != nullptr) {
            cublasLtMatmulPreferenceDestroy(preference_);
        }
        for (cublasLtMatrixLayout_t layout : {c_, d_, b_, a_}) {
            if (layout != nullptr) {
                cublasLtMatrixLayoutDestroy(layout);
            }
        }
        if (op_ != nullptr) {
            cublasLtMatmulDescDestroy(op_);
        }
        preference_ = nullptr;
        c_ = d_ = b_ = a_ = nullptr;
        op_ = nullptr;
    }

    cublasLtHandle_t lt_;
    Kind kind_;
    void* workspace_;
    cublasLtMatmulDesc_t op_ = nullptr;
    cublasLtMatrixLayout_t a_ = nullptr;
    cublasLtMatrixLayout_t b_ = nullptr;
    cublasLtMatrixLayout_t c_ = nullptr;  // kBiasResidual only
    cublasLtMatrixLayout_t d_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    cublasLtMatmulAlgo_t algo_{};
    std::vector<cublasLtMatmulAlgo_t> algos_;
    std::vector<LtCandidate> candidates_;
};

// Chooses the algorithm an LtGemm runs in a case, from its candidates, and
// returns the choice with every candidate's timing. Each candidate gets a
// warm-up of a sixth of the case's own, then five samples of half the case's
// sample length, on the case's rotating buffer sets, and its time is the median
// of them. The fastest wins, unless `selection` is First, which takes the
// heuristic's first choice but times the rest all the same. A candidate whose
// launch fails is recorded with its status and skipped; it throws if the one
// that was to be used fails.
BaselineAlgorithm choose_algorithm(LtGemm& lt, LtAlgo selection, int sets,
                                   const std::vector<const void*>& x,
                                   const std::vector<const void*>& w, const std::vector<void*>& y,
                                   const std::vector<void*>& aux, const Options& options,
                                   cudaStream_t stream) {
    constexpr int kSamples = 5;
    const double warmup_seconds = options.warmup_seconds / 6;
    const double sample_ms = options.sample_ms / 2;
    BaselineAlgorithm chosen;
    chosen.selection = selection;
    chosen.candidates = lt.candidates();

    EventTimer timer;
    long long next = 0;  // the next launch's rotation index, continuing across candidates
    cublasStatus_t failure = CUBLAS_STATUS_SUCCESS;
    const auto launcher = [&](cudaStream_t s, long long first, long long count) {
        for (long long i = first; i < first + count && failure == CUBLAS_STATUS_SUCCESS; ++i) {
            const int set = static_cast<int>(i % sets);
            failure = lt.launch(x[set], w[set], y[set], aux[set], s);
        }
    };
    // Milliseconds for `count` launches, or 0 once one has failed.
    const auto time = [&](long long count) {
        const double ms = timer.time(launcher, stream, next, count);
        next += count;
        return failure == CUBLAS_STATUS_SUCCESS ? ms : 0.0;
    };

    for (LtCandidate& candidate : chosen.candidates) {
        lt.use(static_cast<size_t>(candidate.index));
        failure = CUBLAS_STATUS_SUCCESS;
        // Warm-up, doubling the batch until it takes a sample.
        const auto start = std::chrono::steady_clock::now();
        long long count = 1;
        double per_launch = 0;
        do {
            const double ms = time(count);
            per_launch = std::max(ms, 1e-3) / static_cast<double>(count);
            if (ms < sample_ms) {
                count *= 2;
            }
        } while (failure == CUBLAS_STATUS_SUCCESS &&
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() <
                     warmup_seconds);
        std::vector<double> samples;
        for (int i = 0; i < kSamples && failure == CUBLAS_STATUS_SUCCESS; ++i) {
            count = std::max(1LL, static_cast<long long>(std::ceil(sample_ms * 1.05 / per_launch)));
            const double ms = time(count);
            per_launch = std::max(ms, 1e-3) / static_cast<double>(count);
            samples.push_back(per_launch);
        }
        if (failure != CUBLAS_STATUS_SUCCESS) {
            // The failed call may leave a non-sticky error behind.
            cudaGetLastError();
            candidate.status = cublasGetStatusName(failure);
            continue;
        }
        candidate.ms_per_launch = median(samples);
    }

    chosen.used = 0;
    if (selection == LtAlgo::Fastest) {
        for (const LtCandidate& candidate : chosen.candidates) {
            const std::optional<double>& best = chosen.candidates[chosen.used].ms_per_launch;
            if (candidate.ms_per_launch && (!best || *candidate.ms_per_launch < *best)) {
                chosen.used = candidate.index;
            }
        }
    }
    const LtCandidate& used = chosen.candidates[chosen.used];
    if (!used.ms_per_launch) {
        throw std::runtime_error(std::string("cublasLtMatmul: ") +
                                 (selection == LtAlgo::Fastest ? "no candidate algorithm works: "
                                                               : "the first candidate fails: ") +
                                 used.status);
    }
    lt.use(static_cast<size_t>(chosen.used));
    return chosen;
}

// Gaussian values by Box-Muller from splitmix64, two per pair of draws.
class NormalGenerator {
public:
    explicit NormalGenerator(uint64_t seed) : state_(seed) {}

    double next() {
        if (have_spare_) {
            have_spare_ = false;
            return spare_;
        }
        // u1 is in (0, 1], so the logarithm is finite; u2 is in [0, 1).
        const double u1 = static_cast<double>((splitmix64(state_) >> 11) + 1) * 0x1p-53;
        const double u2 = static_cast<double>(splitmix64(state_) >> 11) * 0x1p-53;
        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double angle = 2.0 * 3.14159265358979323846 * u2;
        spare_ = radius * std::sin(angle);
        have_spare_ = true;
        return radius * std::cos(angle);
    }

private:
    uint64_t state_;
    double spare_ = 0;
    bool have_spare_ = false;
};

// Fills `count` FP16 values on the device with N(0, sigma^2), staged through a
// host buffer of at most kHostChunk bytes.
void fill_normal(void* device, size_t count, double sigma, uint64_t seed) {
    NormalGenerator generator(seed);
    std::vector<uint16_t> host(std::min(count, kHostChunk / sizeof(uint16_t)));
    for (size_t offset = 0; offset < count; offset += host.size()) {
        const size_t n = std::min(host.size(), count - offset);
        for (size_t i = 0; i < n; ++i) {
            host[i] = __half_as_ushort(__float2half(static_cast<float>(sigma * generator.next())));
        }
        check_cuda(cudaMemcpy(static_cast<uint16_t*>(device) + offset, host.data(),
                              n * sizeof(uint16_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy");
    }
}

// Fills `count` FP32 values on the device with N(0, sigma^2), staged like
// fill_normal.
void fill_normal_f32(void* device, size_t count, double sigma, uint64_t seed) {
    NormalGenerator generator(seed);
    std::vector<float> host(std::min(count, kHostChunk / sizeof(float)));
    for (size_t offset = 0; offset < count; offset += host.size()) {
        const size_t n = std::min(host.size(), count - offset);
        for (size_t i = 0; i < n; ++i) {
            host[i] = static_cast<float>(sigma * generator.next());
        }
        check_cuda(cudaMemcpy(static_cast<float*>(device) + offset, host.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy");
    }
}

// Fills `count` FP32 values on the device with N(0, sigma^2), as
// fill_normal_f32 does, and `count` FP16 values with the same values rounded
// to nearest (one rounding), staged through host buffers.
void fill_bias(void* device_f32, void* device_f16, size_t count, double sigma, uint64_t seed) {
    NormalGenerator generator(seed);
    const size_t chunk = std::min(count, kHostChunk / (sizeof(float) + sizeof(uint16_t)));
    std::vector<float> host_f32(chunk);
    std::vector<uint16_t> host_f16(chunk);
    for (size_t offset = 0; offset < count; offset += chunk) {
        const size_t n = std::min(chunk, count - offset);
        for (size_t i = 0; i < n; ++i) {
            host_f32[i] = static_cast<float>(sigma * generator.next());
            host_f16[i] = __half_as_ushort(__float2half_rn(host_f32[i]));
        }
        check_cuda(cudaMemcpy(static_cast<float*>(device_f32) + offset, host_f32.data(),
                              n * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy");
        check_cuda(cudaMemcpy(static_cast<uint16_t*>(device_f16) + offset, host_f16.data(),
                              n * sizeof(uint16_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy");
    }
}

struct Difference {
    double diff_sq = 0;  // sum of (a - b)^2 over the finite pairs
    double ref_sq = 0;   // sum of b^2 over the same pairs
    double max_abs = 0;  // max |a - b|
    double max_ref = 0;  // max |b|
    unsigned long long nonfinite = 0;
};

// Compares two device buffers of `count` values of type T on the host in
// float64, through two host buffers of kHostChunk / 2 bytes each. `a` is
// Flint's and `b` the reference; `to_double` decodes a stored value. A value
// that is NaN or infinite counts in `nonfinite`, and its pair in nothing else.
template <typename T, typename Decode>
Difference compare_device(const void* a, const void* b, size_t count, Decode to_double) {
    Difference d;
    std::vector<T> host_a(std::min(count, kHostChunk / 2 / sizeof(T)));
    std::vector<T> host_b(host_a.size());
    for (size_t offset = 0; offset < count; offset += host_a.size()) {
        const size_t n = std::min(host_a.size(), count - offset);
        check_cuda(cudaMemcpy(host_a.data(), static_cast<const T*>(a) + offset, n * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        check_cuda(cudaMemcpy(host_b.data(), static_cast<const T*>(b) + offset, n * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy");
        for (size_t i = 0; i < n; ++i) {
            const double x = to_double(host_a[i]);
            const double y = to_double(host_b[i]);
            d.nonfinite += (std::isfinite(x) ? 0 : 1) + (std::isfinite(y) ? 0 : 1);
            if (!std::isfinite(x) || !std::isfinite(y)) {
                continue;
            }
            const double diff = x - y;
            d.diff_sq += diff * diff;
            d.ref_sq += y * y;
            d.max_abs = std::max(d.max_abs, std::fabs(diff));
            d.max_ref = std::max(d.max_ref, std::fabs(y));
        }
    }
    return d;
}

Difference compare_fp16(const void* a, const void* b, size_t count) {
    return compare_device<uint16_t>(
        a, b, count, [](uint16_t v) -> double { return __half2float(__ushort_as_half(v)); });
}

Difference compare_fp32(const void* a, const void* b, size_t count) {
    return compare_device<float>(a, b, count, [](float v) -> double { return v; });
}

// One FP16 ulp at magnitude v: 2^(floor(log2 v) - 10), or the subnormal
// spacing 2^-24 below the smallest normal, 2^-14.
double fp16_ulp(double v) { return v >= 0x1p-14 ? std::ldexp(1.0, std::ilogb(v) - 10) : 0x1p-24; }

}  // namespace

CublasHandle::CublasHandle() {
    check_cublas(cublasCreate(&handle_), "cublasCreate");
    // By default a split-K GEMM may reduce its partial sums in the output's
    // FP16. Disallowing that keeps cuBLAS in FP32 throughout, as PyTorch does
    // with allow_fp16_reduced_precision_reduction off.
    check_cublas(cublasSetMathMode(handle_, static_cast<cublasMath_t>(
                                                CUBLAS_DEFAULT_MATH |
                                                CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)),
                 "cublasSetMathMode");
}

CublasHandle::~CublasHandle() {
    if (handle_ != nullptr) {
        cublasDestroy(handle_);
    }
}

CublasLtHandle::CublasLtHandle() { check_cublas(cublasLtCreate(&handle_), "cublasLtCreate"); }

CublasLtHandle::~CublasLtHandle() {
    if (handle_ != nullptr) {
        cublasLtDestroy(handle_);
    }
}

CaseResult bench_gemm(const GemmShape& shape, GemmLayout layout, GemmEpilogue epilogue,
                      cublasHandle_t cublas, cublasLtHandle_t cublas_lt, const DeviceInfo& device,
                      Monitor& monitor, const Options& options, cudaStream_t stream) {
    const size_t m = static_cast<size_t>(shape.m);
    const size_t n = static_cast<size_t>(shape.n);
    const size_t k = static_cast<size_t>(shape.k);
    const bool tn = layout == GemmLayout::TN;
    const bool gelu = epilogue == GemmEpilogue::BiasGelu;
    const bool dgelu = epilogue == GemmEpilogue::DGelu;
    const bool residual = epilogue == GemmEpilogue::BiasResidual;
    // gelu and residual add the bias too.
    const bool bias = epilogue == GemmEpilogue::Bias || gelu || residual;
    const size_t x_bytes = m * k * sizeof(uint16_t);  // m x k, or k x m for TN
    const size_t w_bytes = n * k * sizeof(uint16_t);  // n x k for NT, k x n for NN and TN
    // FP32 for TN and bias-residual.
    const size_t y_bytes = m * n * (tn || residual ? sizeof(float) : sizeof(uint16_t));
    // The second m x n buffer of the epilogue cases, y's size: gelu writes pre,
    // the sums before GELU, dgelu reads it, and residual reads the residual
    // stream (FP32 like y).
    const size_t aux_bytes = gelu || dgelu || residual ? y_bytes : 0;
    const double set_bytes = static_cast<double>(x_bytes) + static_cast<double>(w_bytes) +
                             static_cast<double>(y_bytes) + static_cast<double>(aux_bytes);
    const int sets = rotation_sets(device, set_bytes);

    // Each set has its own x, w, y and aux, carved from one arena per matrix with
    // 256-byte strides like cudaMalloc, so small shapes don't need thousands
    // of allocations. Flint and cuBLAS read the same buffers.
    const auto stride = [](size_t bytes) { return (bytes + 255) / 256 * 256; };
    DeviceAllocations allocations;
    char* x_arena = static_cast<char*>(allocations.allocate(stride(x_bytes) * sets));
    char* w_arena = static_cast<char*>(allocations.allocate(stride(w_bytes) * sets));
    char* y_arena = static_cast<char*>(allocations.allocate(stride(y_bytes) * sets));
    char* aux_arena = aux_bytes > 0
                          ? static_cast<char*>(allocations.allocate(stride(aux_bytes) * sets))
                          : nullptr;
    // The output check's second output, beside set 0's y, and with gelu its
    // second pre. (dgelu's pre and the residual are inputs, set 0's for both
    // sides.)
    void* check = allocations.allocate(y_bytes);
    void* check_pre = gelu ? allocations.allocate(aux_bytes) : nullptr;
    // The bias is one FP32 buffer of n values for Flint, shared by every set
    // (it's tiny), and, except with residual, the same values rounded to FP16
    // for cuBLASLt (which reads the FP32 ones with an FP32 output). cuBLASLt
    // also gets a workspace.
    const size_t bias_bytes = bias ? n * sizeof(float) : 0;
    void* bias_buffer = nullptr;
    void* bias_buffer_f16 = nullptr;
    void* lt_workspace = nullptr;
    if (bias) {
        bias_buffer = allocations.allocate(bias_bytes);
        if (residual) {
            fill_normal_f32(bias_buffer, n, kBiasSigma, kSeedBias);
        } else {
            bias_buffer_f16 = allocations.allocate(n * sizeof(uint16_t));
            fill_bias(bias_buffer, bias_buffer_f16, n, kBiasSigma, kSeedBias);
        }
    }
    if (bias || dgelu) {
        lt_workspace = allocations.allocate(kLtWorkspaceBytes);
    }
    // TN splits k for small outputs, and then needs a workspace. The launches
    // run in order on one stream, so every set shares one.
    size_t workspace_bytes = 0;
    void* workspace = nullptr;
    if (tn) {
        check_flint(flint_gemm_tn_workspace(m, n, k, &workspace_bytes), "flint_gemm_tn_workspace");
        if (workspace_bytes > 0) {
            workspace = allocations.allocate(workspace_bytes);
        }
    }
    std::vector<const void*> x(sets), w(sets);
    std::vector<void*> y(sets), aux(sets, nullptr);
    for (int s = 0; s < sets; ++s) {
        x[s] = x_arena + stride(x_bytes) * s;
        w[s] = w_arena + stride(w_bytes) * s;
        y[s] = y_arena + stride(y_bytes) * s;
        if (aux_arena != nullptr) {
            aux[s] = aux_arena + stride(aux_bytes) * s;
        }
    }
    fill_normal(x_arena, stride(x_bytes) * sets / sizeof(uint16_t), kInputSigmaX, kSeedX);
    fill_normal(w_arena, stride(w_bytes) * sets / sizeof(uint16_t), kInputSigmaW, kSeedW);
    check_cuda(cudaMemset(y_arena, 0, stride(y_bytes) * sets), "cudaMemset");
    if (gelu) {
        check_cuda(cudaMemset(aux_arena, 0, stride(aux_bytes) * sets), "cudaMemset");
    }
    if (dgelu) {
        fill_normal(aux_arena, stride(aux_bytes) * sets / sizeof(uint16_t), kPreSigma, kSeedPre);
    }
    if (residual) {
        fill_normal_f32(aux_arena, stride(aux_bytes) * sets / sizeof(float), kResidualSigma,
                        kSeedResidual);
    }

    const float alpha = 1.0f;
    // TN accumulates into y, as training adds each micro-batch's gradient, so
    // both sides add there.
    const float beta = tn ? 1.0f : 0.0f;
    // cuBLAS is column-major, so it sees the row-major buffers y (m x n) and x
    // (m x k) as the transposes y^T (n x m, ld n) and x^T (k x m, ld k).
    // NT: w (n x k) is seen as w^T (k x n, ld k). The row-major y = x w^T is
    // the column-major y^T = op(w^T) x^T with op = transpose: A is w's buffer
    // with OP_T, B is x's buffer with OP_N.
    // NN: w (k x n) is seen as w^T (n x k, ld n). The row-major y = x w is the
    // column-major y^T = w^T x^T: A is w's buffer and B is x's buffer, both
    // with OP_N.
    // TN: x is k x m, w is k x n, and y (FP32) is m x n. The row-major
    // y = x^T w is the column-major y^T = w^T x. w's buffer read column-major
    // is n x k (ld n), which is w^T, so OP_N; x's buffer read column-major is
    // m x k (ld m), which is x^T, so OP_T gives x. A is w's buffer and B is
    // x's.
    const bool nn = layout == GemmLayout::NN;
    // The epilogue cases' baseline is cuBLASLt's matmul with its bias epilogue
    // (GELU_AUX_BIAS for gelu, DGELU for dgelu, and for residual the bias
    // epilogue with the residual as C and beta = 1), with the same view of the
    // buffers as the cublasGemmEx call, set up once.
    std::optional<LtGemm> lt_gemm;
    std::optional<BaselineAlgorithm> baseline_algorithm;
    if (bias || dgelu) {
        lt_gemm.emplace(cublas_lt,
                        dgelu      ? LtGemm::Kind::kDGelu
                        : gelu     ? LtGemm::Kind::kBiasGelu
                        : residual ? LtGemm::Kind::kBiasResidual
                                   : LtGemm::Kind::kBias,
                        m, n, k, residual ? bias_buffer : bias_buffer_f16, aux[0], lt_workspace);
        // The baseline algorithm comes from timing the heuristic's candidates
        // on these buffers, before the case's own warm-up and rounds.
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        baseline_algorithm = choose_algorithm(*lt_gemm, options.lt_algo, sets, x, w, y, aux,
                                              options, stream);
    }
    const auto cublas_gemm = [&](const void* xs, const void* ws, void* ys, void* auxs,
                                 cudaStream_t s) {
        if (bias || dgelu) {
            lt_gemm->run(xs, ws, ys, auxs, s);
            return;
        }
        if (tn) {
            check_cublas(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(n),
                                      static_cast<int>(m), static_cast<int>(k), &alpha, ws,
                                      CUDA_R_16F, static_cast<int>(n), xs, CUDA_R_16F,
                                      static_cast<int>(m), &beta, ys, CUDA_R_32F,
                                      static_cast<int>(n), CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT),
                         "cublasGemmEx");
            return;
        }
        check_cublas(cublasGemmEx(cublas, nn ? CUBLAS_OP_N : CUBLAS_OP_T, CUBLAS_OP_N,
                                  static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                                  &alpha, ws, CUDA_R_16F, static_cast<int>(nn ? n : k), xs,
                                  CUDA_R_16F, static_cast<int>(k), &beta, ys, CUDA_R_16F,
                                  static_cast<int>(n), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx");
    };
    const auto flint_gemm = [&](cudaStream_t s, const void* xs, const void* ws, void* ys,
                                void* auxs) {
        if (tn) {
            check_flint(flint_gemm_tn(ys, xs, ws, m, n, k, 1, workspace, workspace_bytes, s),
                        "flint_gemm_tn");
        } else if (dgelu) {
            check_flint(flint_gemm_nn_dgelu(ys, xs, ws, auxs, m, n, k, s), "flint_gemm_nn_dgelu");
        } else if (nn) {
            check_flint(flint_gemm_nn(ys, xs, ws, m, n, k, s), "flint_gemm_nn");
        } else if (gelu) {
            check_flint(flint_gemm_nt_bias_gelu(ys, auxs, xs, ws, bias_buffer, m, n, k, s),
                        "flint_gemm_nt_bias_gelu");
        } else if (residual) {
            check_flint(flint_gemm_nt_bias_residual(ys, xs, ws, bias_buffer, auxs, m, n, k, s),
                        "flint_gemm_nt_bias_residual");
        } else if (bias) {
            check_flint(flint_gemm_nt_bias(ys, xs, ws, bias_buffer, m, n, k, s),
                        "flint_gemm_nt_bias");
        } else {
            check_flint(flint_gemm_nt(ys, xs, ws, m, n, k, s), "flint_gemm_nt");
        }
    };

    CaseSpec spec;
    spec.benchmark = "gemm";
    spec.label = std::to_string(shape.m) + "x" + std::to_string(shape.n) + "x" + std::to_string(shape.k);
    spec.layout = layout_name(layout);
    spec.epilogue = epilogue_name(epilogue);
    spec.baseline_name = residual ? "cuBLASLt, residual in C"
                         : bias   ? "cuBLASLt, FP16 bias"
                         : dgelu  ? "cuBLASLt DGELU"
                                  : "cuBLAS";
    // Flint's side: one set's x, w, y (and pre or the residual), and its FP32
    // bias.
    spec.size_bytes = static_cast<unsigned long long>(set_bytes) + bias_bytes;
    spec.flops = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
                 static_cast<double>(shape.k);
    spec.peak_flop_per_sm_per_clock = tensor_flop_per_sm_per_clock(device);
    spec.rotation_sets = sets;
    spec.flint = [&](cudaStream_t s, long long first, long long count) {
        for (long long i = first; i < first + count; ++i) {
            const int set = static_cast<int>(i % sets);
            flint_gemm(s, x[set], w[set], y[set], aux[set]);
        }
    };
    spec.reference = [&](cudaStream_t s, long long first, long long count) {
        check_cublas(cublasSetStream(cublas, s), "cublasSetStream");
        for (long long i = first; i < first + count; ++i) {
            const int set = static_cast<int>(i % sets);
            cublas_gemm(x[set], w[set], y[set], aux[set], s);
        }
    };

    CaseResult result = run_case(spec, device, monitor, options, stream);
    result.baseline_algorithm = baseline_algorithm;

    if (tn) {
        // The timing left results in every y. Both outputs start from the same
        // random values and both GEMMs add into them, so a tile that either
        // one skips keeps its initial value and differs from the other's. The fill is a
        // host copy; the device sync makes sure it has landed before the
        // stream reads it.
        fill_normal_f32(y[0], m * n, 1.0, kSeedY);
        check_cuda(cudaMemcpy(check, y[0], y_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    } else {
        // The timing left results in every y, so poison both outputs first
        // with bytes that decode as NaN: an output either GEMM skips counts as
        // non-finite.
        check_cuda(cudaMemsetAsync(y[0], 0xFF, y_bytes, stream), "cudaMemsetAsync");
        if (gelu) {
            check_cuda(cudaMemsetAsync(aux[0], 0xFF, aux_bytes, stream), "cudaMemsetAsync");
        }
    }
    // The aux buffer the check runs read: gelu's second output is its own, and
    // dgelu's pre and the residual are inputs, set 0's for both sides. (Nothing
    // ever writes the residual.)
    void* const check_aux = dgelu || residual ? aux[0] : check_pre;
    flint_gemm(stream, x[0], w[0], y[0], aux[0]);
    if (!tn) {
        check_cuda(cudaMemsetAsync(check, 0xFF, y_bytes, stream), "cudaMemsetAsync");
        if (gelu) {
            check_cuda(cudaMemsetAsync(check_pre, 0xFF, aux_bytes, stream), "cudaMemsetAsync");
        }
    }
    check_cublas(cublasSetStream(cublas, stream), "cublasSetStream");
    cublas_gemm(x[0], w[0], check, check_aux, stream);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const auto measure = [](const Difference& d) {
        OutputCheck output;
        output.rel_l2 =
            d.ref_sq > 0 ? std::sqrt(d.diff_sq / d.ref_sq) : (d.diff_sq > 0 ? INFINITY : 0.0);
        output.max_abs = d.max_abs;
        output.max_abs_baseline = d.max_ref;
        output.nonfinite = d.nonfinite;
        return output;
    };
    OutputCheck output = measure(tn || residual ? compare_fp32(y[0], check, m * n)
                                                : compare_fp16(y[0], check, m * n));
    OutputCheck pre_output;
    if (gelu) {
        pre_output = measure(compare_fp16(aux[0], check_pre, m * n));
    }

    // A second Flint run, into an output poisoned with different NaN bytes (for
    // tn, refilled with the same initial values), must reproduce the first bit
    // for bit.
    if (tn) {
        fill_normal_f32(check, m * n, 1.0, kSeedY);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    } else {
        check_cuda(cudaMemsetAsync(check, 0x7F, y_bytes, stream), "cudaMemsetAsync");
        if (gelu) {
            check_cuda(cudaMemsetAsync(check_pre, 0x7F, aux_bytes, stream), "cudaMemsetAsync");
        }
    }
    flint_gemm(stream, x[0], w[0], check, check_aux);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    output.deterministic = device_equal(y[0], check, y_bytes);
    if (gelu) {
        pre_output.deterministic = device_equal(aux[0], check_pre, aux_bytes);
    }

    result.output_check = output;
    const double tn_scale = std::max(1.0, static_cast<double>(k) / kTnBoundK);
    const auto within_bounds = [&](const OutputCheck& c) {
        if (tn) {
            return c.rel_l2 <= kTnMaxRelL2 * tn_scale &&
                   c.max_abs <= kTnMaxAbsFraction * tn_scale * c.max_abs_baseline;
        }
        if (residual) {
            const double kd = static_cast<double>(k);
            return c.rel_l2 <= kResidualRelL2Floor + kResidualRelL2PerK * kd &&
                   c.max_abs <=
                       (kResidualMaxAbsFloor + kResidualMaxAbsPerK * kd) * c.max_abs_baseline;
        }
        return c.rel_l2 <= kMaxRelL2 && c.max_abs <= kMaxUlps * fp16_ulp(c.max_abs_baseline);
    };
    const auto passes = [&](const OutputCheck& c) {
        return c.deterministic && c.nonfinite == 0 && within_bounds(c);
    };
    result.outputs_match = passes(output);
    if (gelu) {
        result.pre_check = pre_output;
        result.outputs_match = result.outputs_match && passes(pre_output);
    }
    return result;
}

}  // namespace bench
