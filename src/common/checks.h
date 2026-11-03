// Pointer checks the C API's argument validation shares.
#pragma once

#include <cstddef>
#include <cstdint>

#include "flint/flint.h"

namespace flint {

inline bool aligned16(const void* pointer) {
    return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}

// Whether [pointer, pointer + bytes) stays inside the address space.
inline bool fits(const void* pointer, size_t bytes) {
    return bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(pointer);
}

// Whether [a, a + a_bytes) and [b, b + b_bytes) overlap, for ranges that fit.
inline bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    const uintptr_t a_begin = reinterpret_cast<uintptr_t>(a);
    const uintptr_t b_begin = reinterpret_cast<uintptr_t>(b);
    return a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
}

// A buffer an API function uses.
struct Buffer {
    const char* name;
    const void* pointer;
    size_t bytes;
};

// Checks count buffers for function, recording the first failure as its error:
// in order, each must be non-null, 16-byte aligned and inside the address
// space, and then each of the first outputs (the buffers it writes, listed
// first) must not overlap any buffer after it. The buffers after those are only
// read, so they may share memory. Returns FLINT_OK if all pass.
flint_status check_buffers(const char* function, const Buffer* buffers, size_t count,
                           size_t outputs);

}  // namespace flint
