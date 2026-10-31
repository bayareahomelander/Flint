// Error reporting for the C API: each failure records a message that
// flint_last_error() returns on the same thread.
#pragma once

#include <cuda_runtime_api.h>

#include "flint/flint.h"

#if defined(__GNUC__)
#define FLINT_PRINTF_FORMAT(format_index, first_arg) \
    __attribute__((__format__(__printf__, format_index, first_arg)))
#else
#define FLINT_PRINTF_FORMAT(format_index, first_arg)
#endif

namespace flint {

// Records a printf-style message for flint_last_error() and returns status,
// so a failing API function can end with `return fail(...)`.
flint_status fail(flint_status status, const char* format, ...) FLINT_PRINTF_FORMAT(2, 3);

}  // namespace flint

// Returns FLINT_ERROR_CUDA from the calling function if a CUDA runtime call
// fails, recording the error's name and description and where it happened.
#define FLINT_CUDA_CHECK(call)                                                       \
    do {                                                                             \
        const cudaError_t flint_cuda_error_ = (call);                                \
        if (flint_cuda_error_ != cudaSuccess) {                                      \
            return ::flint::fail(FLINT_ERROR_CUDA, "%s: %s (%s:%d)",                 \
                                 cudaGetErrorName(flint_cuda_error_),                \
                                 cudaGetErrorString(flint_cuda_error_), __FILE__,    \
                                 __LINE__);                                          \
        }                                                                            \
    } while (0)
