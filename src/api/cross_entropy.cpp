// flint_softmax_cross_entropy: checks the arguments, then launches the kernel.
#include <cmath>
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/cross_entropy.cuh"

flint_status flint_softmax_cross_entropy(void* dlogits, void* row_loss, const void* logits,
                                         const void* targets, float scale, size_t rows,
                                         size_t vocab, size_t cols, flint_stream stream) {
    constexpr const char* name = "flint_softmax_cross_entropy";
    if (cols == 0 || cols % 8 != 0 || cols > flint::kCrossEntropyMaxCols) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: cols (%zu) must be a multiple of 8 from 8 to %zu", name, cols,
                           flint::kCrossEntropyMaxCols);
    }
    if (vocab == 0 || vocab > cols) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: vocab (%zu) must be from 1 to cols (%zu)", name, vocab, cols);
    }
    if (rows > flint::kCrossEntropyMaxRows) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: rows (%zu) can be up to %zu", name,
                           rows, flint::kCrossEntropyMaxRows);
    }
    if (!std::isfinite(scale)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: scale (%g) must be finite", name,
                           static_cast<double>(scale));
    }
    if (rows == 0) {
        return FLINT_OK;
    }
    // The limits keep rows * cols * 2 far below 2^64.
    const flint::Buffer buffers[] = {
        {"dlogits", dlogits, rows * cols * 2},
        {"row_loss", row_loss, rows * 4},
        {"logits", logits, rows * cols * 2},
        {"targets", targets, rows * 2},
    };
    // The two outputs must not overlap each other or an input; the inputs may
    // share memory, since they are only read.
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 2);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(flint::launch_softmax_cross_entropy(dlogits, row_loss, logits, targets, scale,
                                                         rows, vocab, cols, stream));
    return FLINT_OK;
}
