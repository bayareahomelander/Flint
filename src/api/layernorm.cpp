// LayerNorm's forward and backward passes and the backward pass's workspace size:
// check the arguments, then launch the kernels.
#include <cmath>
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/layernorm.cuh"
#include "kernels/layernorm_backward.cuh"

flint_status flint_layernorm_forward(void* y, void* mean, void* rstd, const void* x,
                                     const void* gamma, const void* beta, size_t rows,
                                     size_t cols, float eps, flint_stream stream) {
    constexpr const char* name = "flint_layernorm_forward";
    if (cols == 0 || cols > flint::kLayerNormMaxCols) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: cols (%zu) must be from 1 to %zu",
                           name, cols, flint::kLayerNormMaxCols);
    }
    if (rows > flint::kLayerNormMaxRows) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: rows (%zu) can be up to %zu", name,
                           rows, flint::kLayerNormMaxRows);
    }
    if (!std::isfinite(eps) || eps < 0.0f) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: eps (%g) must be finite and non-negative", name,
                           static_cast<double>(eps));
    }
    if (rows == 0) {
        return FLINT_OK;
    }

    // Outputs first, then inputs, in the order of the parameters. The limits keep
    // rows * cols * 4 far below 2^64.
    const flint::Buffer buffers[] = {
        {"y", y, rows * cols * 2},
        {"mean", mean, rows * 4},
        {"rstd", rstd, rows * 4},
        {"x", x, rows * cols * 4},
        {"gamma", gamma, cols * 4},
        {"beta", beta, cols * 4},
    };
    // Each output must not overlap any other buffer; the inputs may share
    // memory, since they are only read.
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 3);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(flint::launch_layernorm_forward(y, mean, rstd, x, gamma, beta, rows,
                                                     static_cast<int>(cols), eps, stream));
    return FLINT_OK;
}

namespace {

flint_status check_backward_dims(const char* name, size_t rows, size_t cols) {
    if (cols == 0 || cols % 8 != 0 || cols > flint::kLayerNormBackwardMaxCols) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: cols (%zu) must be a multiple of 8 from 8 to %zu", name, cols,
                           flint::kLayerNormBackwardMaxCols);
    }
    if (rows == 0 || rows > flint::kLayerNormBackwardMaxRows) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: rows (%zu) must be from 1 to %zu",
                           name, rows, flint::kLayerNormBackwardMaxRows);
    }
    return FLINT_OK;
}

}  // namespace

flint_status flint_layernorm_backward(void* dx, void* dgamma, void* dbeta, const void* dy,
                                      const void* x, const void* mean, const void* rstd,
                                      const void* gamma, size_t rows, size_t cols,
                                      int accumulate_dx, int accumulate_params, void* workspace,
                                      size_t workspace_bytes, flint_stream stream) {
    constexpr const char* name = "flint_layernorm_backward";
    if (const flint_status status = check_backward_dims(name, rows, cols); status != FLINT_OK) {
        return status;
    }
    const size_t needed = flint::layernorm_backward_workspace(rows, cols);
    if (workspace_bytes < needed) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: needs %zu bytes of workspace, given %zu",
                           name, needed, workspace_bytes);
    }
    // Outputs first, then inputs, in the order of the parameters. The limits keep
    // rows * cols * 4 far below 2^64.
    const flint::Buffer buffers[] = {
        {"dx", dx, rows * cols * 4},
        {"dgamma", dgamma, cols * 4},
        {"dbeta", dbeta, cols * 4},
        {"workspace", workspace, needed},
        {"dy", dy, rows * cols * 2},
        {"x", x, rows * cols * 4},
        {"mean", mean, rows * 4},
        {"rstd", rstd, rows * 4},
        {"gamma", gamma, cols * 4},
    };
    // Each output must not overlap any other buffer; the inputs may share memory,
    // since they are only read.
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 4);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(flint::launch_layernorm_backward(
        dx, dgamma, dbeta, dy, x, mean, rstd, gamma, rows, cols, accumulate_dx != 0,
        accumulate_params != 0, workspace, stream));
    return FLINT_OK;
}

flint_status flint_layernorm_backward_workspace(size_t rows, size_t cols, size_t* bytes) {
    constexpr const char* name = "flint_layernorm_backward_workspace";
    if (bytes == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: bytes is null", name);
    }
    if (const flint_status status = check_backward_dims(name, rows, cols); status != FLINT_OK) {
        return status;
    }
    *bytes = flint::layernorm_backward_workspace(rows, cols);
    return FLINT_OK;
}
