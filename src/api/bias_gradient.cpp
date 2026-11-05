// flint_bias_gradient and its workspace size: check the arguments, then launch the
// kernels.
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/bias_gradient.cuh"

namespace {

flint_status check_dims(const char* name, size_t rows, size_t cols) {
    if (cols % 8 != 0 || cols > flint::kBiasGradientMaxCols) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: cols (%zu) must be a multiple of 8 up to %zu", name, cols,
                           flint::kBiasGradientMaxCols);
    }
    if (rows == 0 || rows > flint::kBiasGradientMaxRows) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: rows (%zu) must be from 1 to %zu",
                           name, rows, flint::kBiasGradientMaxRows);
    }
    return FLINT_OK;
}

}  // namespace

flint_status flint_bias_gradient_workspace(size_t rows, size_t cols, size_t* bytes) {
    constexpr const char* name = "flint_bias_gradient_workspace";
    if (bytes == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: bytes is null", name);
    }
    if (const flint_status status = check_dims(name, rows, cols); status != FLINT_OK) {
        return status;
    }
    *bytes = flint::bias_gradient_workspace(rows, cols);
    return FLINT_OK;
}

flint_status flint_bias_gradient(void* dbias, const void* dy, size_t rows, size_t cols,
                                 int accumulate, void* workspace, size_t workspace_bytes,
                                 flint_stream stream) {
    constexpr const char* name = "flint_bias_gradient";
    if (const flint_status status = check_dims(name, rows, cols); status != FLINT_OK) {
        return status;
    }
    if (cols == 0) {
        return FLINT_OK;
    }
    // The limits keep rows * cols * 2 below 2^64, and the workspace below 2^58.
    const size_t needed = flint::bias_gradient_workspace(rows, cols);
    if (workspace_bytes < needed) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: needs %zu bytes of workspace, given %zu",
                           name, needed, workspace_bytes);
    }
    // Outputs first: dbias, and the workspace, which the first pass writes. Each must
    // not overlap any buffer after it.
    const flint::Buffer buffers[] = {
        {"dbias", dbias, cols * 4},
        {"workspace", workspace, needed},
        {"dy", dy, rows * cols * 2},
    };
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 2);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(
        flint::launch_bias_gradient(dbias, dy, rows, cols, accumulate != 0, workspace, stream));
    return FLINT_OK;
}
