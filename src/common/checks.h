// Pointer checks the C API's argument validation shares.
#pragma once

#include <cstddef>
#include <cstdint>

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

}  // namespace flint
