// flint_layernorm_forward: checks the arguments, then launches the kernel.
#include <cmath>
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/layernorm.cuh"

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
