// Attention's causal softmax, forward and backward: check the arguments, then launch the
// kernels.
#include <cmath>
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/softmax.cuh"
#include "kernels/softmax_backward.cuh"

flint_status flint_causal_softmax_forward(void* p, const void* s, size_t batches, size_t length,
                                          flint_stream stream) {
    constexpr const char* name = "flint_causal_softmax_forward";
    if (length % 8 != 0 || length > flint::kSoftmaxMaxLength) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: length (%zu) must be a multiple of 8 up to %zu", name, length,
                           flint::kSoftmaxMaxLength);
    }
    if (batches > flint::kSoftmaxMaxBatches) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: batches (%zu) can be up to %zu", name,
                           batches, flint::kSoftmaxMaxBatches);
    }
    if (batches == 0 || length == 0) {
        return FLINT_OK;
    }
    // The limits keep this product far below 2^64.
    const size_t bytes = batches * length * length * 2;
    const flint::Buffer buffers[] = {
        {"p", p, bytes},
        {"s", s, bytes},
    };
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 1);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(flint::launch_causal_softmax_forward(p, s, batches * length, length, stream));
    return FLINT_OK;
}

flint_status flint_causal_softmax_backward(void* ds, const void* p, const void* dp, size_t batches,
                                           size_t length, float scale, flint_stream stream) {
    constexpr const char* name = "flint_causal_softmax_backward";
    if (length % 8 != 0 || length > flint::kSoftmaxMaxLength) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: length (%zu) must be a multiple of 8 up to %zu", name, length,
                           flint::kSoftmaxMaxLength);
    }
    if (batches > flint::kSoftmaxMaxBatches) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: batches (%zu) can be up to %zu", name,
                           batches, flint::kSoftmaxMaxBatches);
    }
    if (!std::isfinite(scale)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: scale (%g) must be finite", name,
                           static_cast<double>(scale));
    }
    if (batches == 0 || length == 0) {
        return FLINT_OK;
    }
    // The limits keep this product far below 2^64.
    const size_t bytes = batches * length * length * 2;
    // ds may be dp itself (in place), and then dp isn't listed separately; otherwise ds
    // must overlap neither input. The inputs may share memory, since they are only read.
    const flint::Buffer buffers[] = {
        {"ds", ds, bytes},
        {"p", p, bytes},
        {"dp", dp, bytes},
    };
    const flint_status checked =
        flint::check_buffers(name, buffers, ds == dp ? 2 : std::size(buffers), 1);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(flint::launch_causal_softmax_backward(ds, p, dp, batches * length, length,
                                                           scale, stream));
    return FLINT_OK;
}
