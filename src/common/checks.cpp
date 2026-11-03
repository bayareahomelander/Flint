#include "common/checks.h"

#include "common/error.h"

namespace flint {

flint_status check_buffers(const char* function, const Buffer* buffers, size_t count,
                           size_t outputs) {
    for (size_t i = 0; i < count; ++i) {
        const Buffer& b = buffers[i];
        if (b.pointer == nullptr) {
            return fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: %s is null", function, b.name);
        }
        if (!aligned16(b.pointer)) {
            return fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: %s is not 16-byte aligned", function,
                        b.name);
        }
        if (!fits(b.pointer, b.bytes)) {
            return fail(FLINT_ERROR_INVALID_ARGUMENT,
                        "%s: %s runs past the end of the address space", function, b.name);
        }
    }
    for (size_t i = 0; i < outputs; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            const Buffer& a = buffers[i];
            const Buffer& b = buffers[j];
            if (overlap(a.pointer, a.bytes, b.pointer, b.bytes)) {
                return fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: %s and %s overlap", function,
                            a.name, b.name);
            }
        }
    }
    return FLINT_OK;
}

}  // namespace flint
