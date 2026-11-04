// flint_causal_softmax_forward: checks the arguments, then launches the kernel.
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/softmax.cuh"

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
