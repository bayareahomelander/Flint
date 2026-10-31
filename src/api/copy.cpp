// flint_copy: checks the arguments, then launches the copy kernel.
#include <cstdint>

#include "common/error.h"
#include "kernels/copy.cuh"

flint_status flint_copy(void* dst, const void* src, size_t bytes, flint_stream stream) {
    if (bytes == 0) {
        return FLINT_OK;
    }
    if (dst == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_copy: dst is null");
    }
    if (src == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_copy: src is null");
    }
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t s = reinterpret_cast<uintptr_t>(src);
    if (bytes > UINTPTR_MAX - d || bytes > UINTPTR_MAX - s) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "flint_copy: %zu bytes run past the end of the address space", bytes);
    }
    if (d < s + bytes && s < d + bytes) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_copy: dst and src overlap");
    }
    FLINT_CUDA_CHECK(flint::launch_copy(dst, src, bytes, stream));
    return FLINT_OK;
}
