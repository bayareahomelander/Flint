// flint_gemm_nt, flint_gemm_nt_bias, flint_gemm_nt_bias_gelu,
// flint_gemm_nt_bias_residual, flint_gemm_nn, flint_gemm_nn_dgelu and
// flint_gemm_tn: check the arguments, then launch the GEMM kernel for their
// layout and epilogue.
#include <cstdint>

#include "common/error.h"
#include "kernels/gemm.cuh"

namespace {

// The tensor cores' FP32 accumulation truncates, so the error grows with the
// length of k's chain, while the peer's (PyTorch's FP16 path) stays flat. At
// 2^16 it's 1.1x the peer's, and at 2^18 2.2x, past the correctness gate.
constexpr size_t kMaxK = size_t{1} << 16;

bool aligned16(const void* pointer) { return reinterpret_cast<uintptr_t>(pointer) % 16 == 0; }

// Whether [pointer, pointer + bytes) stays inside the address space.
bool fits(const void* pointer, size_t bytes) {
    return bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(pointer);
}

// Whether [a, a + a_bytes) and [b, b + b_bytes) overlap, for ranges that fit.
bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    const uintptr_t a_begin = reinterpret_cast<uintptr_t>(a);
    const uintptr_t b_begin = reinterpret_cast<uintptr_t>(b);
    return a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
}

// The checks NT and NN share, for m and n positive (TN has its own, below). w
// holds n * k halves either way. When its rows are n long, as in NN, they load
// in 16-byte chunks too, so n must be a multiple of 8. y's elements are FP16,
// or FP32 (4 bytes) into the residual stream.
flint_status check_nt_nn(const char* name, bool n_in_chunks, void* y, const void* x,
                         const void* w, size_t m, size_t n, size_t k, size_t y_element = 2) {
    if (k == 0 || k % 8 != 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: k (%zu) must be a positive multiple of 8", name, k);
    }
    if (n_in_chunks && n % 8 != 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: n (%zu) must be a multiple of 8",
                           name, n);
    }
    if (m > flint::kGemmMaxMN || n > flint::kGemmMaxMN || k > kMaxK) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: %zu x %zu x %zu is too large: m and n can be up to %zu, and k up "
                           "to %zu",
                           name, m, n, k, flint::kGemmMaxMN, kMaxK);
    }
    if (y == nullptr || x == nullptr || w == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: %s is null", name,
                           y == nullptr ? "y" : x == nullptr ? "x" : "w");
    }
    if (!aligned16(y) || !aligned16(x) || !aligned16(w)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: %s is not 16-byte aligned", name,
                           !aligned16(y) ? "y" : !aligned16(x) ? "x" : "w");
    }
    // The size limits keep these products far below 2^64.
    const size_t y_bytes = m * n * y_element;
    const size_t x_bytes = m * k * 2;
    const size_t w_bytes = n * k * 2;
    if (!fits(y, y_bytes) || !fits(x, x_bytes) || !fits(w, w_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: %s runs past the end of the address space", name,
                           !fits(y, y_bytes) ? "y" : !fits(x, x_bytes) ? "x" : "w");
    }
    if (overlap(y, y_bytes, x, x_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: y and x overlap", name);
    }
    if (overlap(y, y_bytes, w, w_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: y and w overlap", name);
    }
    return FLINT_OK;
}

// The bias epilogues' checks on bias, n FP32 values, after NT's checks, for y
// of y_element bytes per value.
flint_status check_bias(const char* name, const void* y, const void* bias, size_t m, size_t n,
                        size_t y_element = 2) {
    if (bias == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: bias is null", name);
    }
    if (!aligned16(bias)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: bias is not 16-byte aligned", name);
    }
    const size_t bias_bytes = n * 4;
    if (!fits(bias, bias_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: bias runs past the end of the address space", name);
    }
    if (overlap(y, m * n * y_element, bias, bias_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: y and bias overlap", name);
    }
    return FLINT_OK;
}

// The GELU epilogues' checks on pre, m x n FP16 like y, which the forward
// stores and the backward reads, after NT's or NN's checks.
flint_status check_pre(const char* name, const void* y, const void* pre, size_t m, size_t n) {
    if (pre == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: pre is null", name);
    }
    if (!aligned16(pre)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: pre is not 16-byte aligned", name);
    }
    // pre has y's size, and the size limits keep it far below 2^64.
    const size_t pre_bytes = m * n * 2;
    if (!fits(pre, pre_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: pre runs past the end of the address space", name);
    }
    if (overlap(pre, pre_bytes, y, pre_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: y and pre overlap", name);
    }
    return FLINT_OK;
}

// TN's dimensions: x (k x m) and w (k x n) load in 16-byte chunks along m and
// n, so those are multiples of 8, while k, their row count, can be anything
// from 1.
flint_status check_tn_dims(const char* name, size_t m, size_t n, size_t k) {
    if (k == 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: k must be positive", name);
    }
    if (m % 8 != 0 || n % 8 != 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: %s (%zu) must be a multiple of 8",
                           name, m % 8 != 0 ? "m" : "n", m % 8 != 0 ? m : n);
    }
    if (m > flint::kGemmMaxMN || n > flint::kGemmMaxMN || k > kMaxK) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: %zu x %zu x %zu is too large: m and n can be up to %zu, and k up "
                           "to %zu",
                           name, m, n, k, flint::kGemmMaxMN, kMaxK);
    }
    return FLINT_OK;
}

}  // namespace

flint_status flint_gemm_nt(void* y, const void* x, const void* w, size_t m, size_t n, size_t k,
                           flint_stream stream) {
    if (m == 0 || n == 0) {
        return FLINT_OK;
    }
    if (const flint_status status = check_nt_nn("flint_gemm_nt", false, y, x, w, m, n, k);
        status != FLINT_OK) {
        return status;
    }
    FLINT_CUDA_CHECK(flint::launch_gemm_nt(y, x, w, static_cast<int>(m), static_cast<int>(n),
                                           static_cast<int>(k), stream));
    return FLINT_OK;
}

flint_status flint_gemm_nt_bias(void* y, const void* x, const void* w, const void* bias, size_t m,
                                size_t n, size_t k, flint_stream stream) {
    if (m == 0 || n == 0) {
        return FLINT_OK;
    }
    constexpr const char* name = "flint_gemm_nt_bias";
    if (const flint_status status = check_nt_nn(name, false, y, x, w, m, n, k);
        status != FLINT_OK) {
        return status;
    }
    if (const flint_status status = check_bias(name, y, bias, m, n); status != FLINT_OK) {
        return status;
    }
    FLINT_CUDA_CHECK(flint::launch_gemm_nt_bias(y, x, w, bias, static_cast<int>(m),
                                                static_cast<int>(n), static_cast<int>(k), stream));
    return FLINT_OK;
}

flint_status flint_gemm_nt_bias_gelu(void* y, void* pre, const void* x, const void* w,
                                     const void* bias, size_t m, size_t n, size_t k,
                                     flint_stream stream) {
    if (m == 0 || n == 0) {
        return FLINT_OK;
    }
    constexpr const char* name = "flint_gemm_nt_bias_gelu";
    if (const flint_status status = check_nt_nn(name, false, y, x, w, m, n, k);
        status != FLINT_OK) {
        return status;
    }
    if (const flint_status status = check_bias(name, y, bias, m, n); status != FLINT_OK) {
        return status;
    }
    if (const flint_status status = check_pre(name, y, pre, m, n); status != FLINT_OK) {
        return status;
    }
    // The size limits keep these products far below 2^64.
    const size_t pre_bytes = m * n * 2;
    if (overlap(pre, pre_bytes, x, m * k * 2)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: pre and x overlap", name);
    }
    if (overlap(pre, pre_bytes, w, n * k * 2)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: pre and w overlap", name);
    }
    if (overlap(pre, pre_bytes, bias, n * 4)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: pre and bias overlap", name);
    }
    FLINT_CUDA_CHECK(flint::launch_gemm_nt_bias_gelu(y, pre, x, w, bias, static_cast<int>(m),
                                                     static_cast<int>(n), static_cast<int>(k),
                                                     stream));
    return FLINT_OK;
}

// y and residual are FP32. The residual is an input, so it may overlap x, w or
// bias, as inputs may overlap each other.
flint_status flint_gemm_nt_bias_residual(void* y, const void* x, const void* w, const void* bias,
                                         const void* residual, size_t m, size_t n, size_t k,
                                         flint_stream stream) {
    if (m == 0 || n == 0) {
        return FLINT_OK;
    }
    constexpr const char* name = "flint_gemm_nt_bias_residual";
    if (const flint_status status = check_nt_nn(name, false, y, x, w, m, n, k, 4);
        status != FLINT_OK) {
        return status;
    }
    // The epilogue reads and writes FP32 pairs as single 8-byte accesses, which
    // keeps the main loop at full speed on sm_75.
    if (n % 2 != 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: n (%zu) must be even", name, n);
    }
    if (const flint_status status = check_bias(name, y, bias, m, n, 4); status != FLINT_OK) {
        return status;
    }
    if (residual == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: residual is null", name);
    }
    if (!aligned16(residual)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: residual is not 16-byte aligned",
                           name);
    }
    // The size limits keep this far below 2^64.
    const size_t residual_bytes = m * n * 4;
    if (!fits(residual, residual_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: residual runs past the end of the address space", name);
    }
    if (overlap(y, residual_bytes, residual, residual_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: y and residual overlap", name);
    }
    FLINT_CUDA_CHECK(flint::launch_gemm_nt_bias_residual(y, x, w, bias, residual,
                                                         static_cast<int>(m), static_cast<int>(n),
                                                         static_cast<int>(k), stream));
    return FLINT_OK;
}

flint_status flint_gemm_nn(void* y, const void* x, const void* w, size_t m, size_t n, size_t k,
                           flint_stream stream) {
    if (m == 0 || n == 0) {
        return FLINT_OK;
    }
    if (const flint_status status = check_nt_nn("flint_gemm_nn", true, y, x, w, m, n, k);
        status != FLINT_OK) {
        return status;
    }
    FLINT_CUDA_CHECK(flint::launch_gemm_nn(y, x, w, static_cast<int>(m), static_cast<int>(n),
                                           static_cast<int>(k), stream));
    return FLINT_OK;
}

// pre is an input here, so it may overlap x or w, as inputs may overlap each
// other.
flint_status flint_gemm_nn_dgelu(void* y, const void* x, const void* w, const void* pre, size_t m,
                                 size_t n, size_t k, flint_stream stream) {
    if (m == 0 || n == 0) {
        return FLINT_OK;
    }
    constexpr const char* name = "flint_gemm_nn_dgelu";
    if (const flint_status status = check_nt_nn(name, true, y, x, w, m, n, k);
        status != FLINT_OK) {
        return status;
    }
    if (const flint_status status = check_pre(name, y, pre, m, n); status != FLINT_OK) {
        return status;
    }
    FLINT_CUDA_CHECK(flint::launch_gemm_nn_dgelu(y, x, w, pre, static_cast<int>(m),
                                                 static_cast<int>(n), static_cast<int>(k), stream));
    return FLINT_OK;
}

flint_status flint_gemm_tn_workspace(size_t m, size_t n, size_t k, size_t* bytes) {
    if (bytes == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_gemm_tn_workspace: bytes is null");
    }
    *bytes = 0;
    if (m == 0 || n == 0) {
        return FLINT_OK;
    }
    if (const flint_status status = check_tn_dims("flint_gemm_tn_workspace", m, n, k);
        status != FLINT_OK) {
        return status;
    }
    FLINT_CUDA_CHECK(flint::gemm_tn_workspace(static_cast<int>(m), static_cast<int>(n),
                                              static_cast<int>(k), bytes));
    return FLINT_OK;
}

flint_status flint_gemm_tn(void* y, const void* x, const void* w, size_t m, size_t n, size_t k,
                           int accumulate, void* workspace, size_t workspace_bytes,
                           flint_stream stream) {
    if (m == 0 || n == 0) {
        return FLINT_OK;
    }
    if (const flint_status status = check_tn_dims("flint_gemm_tn", m, n, k);
        status != FLINT_OK) {
        return status;
    }
    if (y == nullptr || x == nullptr || w == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_gemm_tn: %s is null",
                           y == nullptr ? "y" : x == nullptr ? "x" : "w");
    }
    if (!aligned16(y) || !aligned16(x) || !aligned16(w)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "flint_gemm_tn: %s is not 16-byte aligned",
                           !aligned16(y) ? "y" : !aligned16(x) ? "x" : "w");
    }
    // The size limits keep these products far below 2^64.
    const size_t y_bytes = m * n * 4;
    const size_t x_bytes = k * m * 2;
    const size_t w_bytes = k * n * 2;
    if (!fits(y, y_bytes) || !fits(x, x_bytes) || !fits(w, w_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "flint_gemm_tn: %s runs past the end of the address space",
                           !fits(y, y_bytes) ? "y" : !fits(x, x_bytes) ? "x" : "w");
    }
    if (overlap(y, y_bytes, x, x_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_gemm_tn: y and x overlap");
    }
    if (overlap(y, y_bytes, w, w_bytes)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_gemm_tn: y and w overlap");
    }
    size_t needed = 0;
    FLINT_CUDA_CHECK(flint::gemm_tn_workspace(static_cast<int>(m), static_cast<int>(n),
                                              static_cast<int>(k), &needed));
    if (needed > 0) {
        if (workspace == nullptr || workspace_bytes < needed) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "flint_gemm_tn: needs %zu bytes of workspace, given %zu", needed,
                               workspace == nullptr ? size_t{0} : workspace_bytes);
        }
        if (!aligned16(workspace)) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "flint_gemm_tn: workspace is not 16-byte aligned");
        }
        if (!fits(workspace, needed)) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "flint_gemm_tn: workspace runs past the end of the address space");
        }
        if (overlap(workspace, needed, y, y_bytes) || overlap(workspace, needed, x, x_bytes) ||
            overlap(workspace, needed, w, w_bytes)) {
            return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                               "flint_gemm_tn: workspace overlaps y, x or w");
        }
    }
    FLINT_CUDA_CHECK(flint::launch_gemm_tn(y, x, w, static_cast<int>(m), static_cast<int>(n),
                                           static_cast<int>(k), accumulate != 0, workspace,
                                           stream));
    return FLINT_OK;
}
