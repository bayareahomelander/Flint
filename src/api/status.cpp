// The version and status strings: the part of the C API that never touches
// the GPU.
#include "flint/flint.h"

const char* flint_version(void) { return FLINT_VERSION; }

const char* flint_status_string(flint_status status) {
    switch (status) {
    case FLINT_OK:
        return "ok";
    case FLINT_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case FLINT_ERROR_CUDA:
        return "CUDA error";
    default:
        return "unknown status";
    }
}
