// flint_fp32_to_fp16: checks the arguments, then launches the kernel.
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/convert.cuh"

flint_status flint_fp32_to_fp16(void* y, const void* x, size_t count, flint_stream stream) {
    constexpr const char* name = "flint_fp32_to_fp16";
    if (count > flint::kConvertMaxCount) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: count (%zu) can be up to %zu", name,
                           count, flint::kConvertMaxCount);
    }
    if (count == 0) {
        return FLINT_OK;
    }

    // The limit keeps count * 4 below 2^64.
    const flint::Buffer buffers[] = {
        {"y", y, count * 2},
        {"x", x, count * 4},
    };
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 1);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(flint::launch_fp32_to_fp16(y, x, count, stream));
    return FLINT_OK;
}
