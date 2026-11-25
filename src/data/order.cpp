#include "data/order.h"

#include <utility>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace flint::data {

namespace {

constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ULL;

}  // namespace

uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return x;
}

uint64_t next(uint64_t& state) {
    state += kGamma;
    return mix(state);
}

uint64_t hash(uint64_t x) { return mix(x + kGamma); }

uint64_t stream_start(uint64_t seed, uint64_t epoch, uint64_t key) {
    return hash(hash(hash(seed) ^ epoch) ^ key);
}

uint64_t draw(uint64_t r, uint64_t n) {
#if defined(_MSC_VER) && !defined(__clang__)
    return __umulh(r, n);
#else
    // __extension__ keeps -Wpedantic quiet about the 128-bit type.
    __extension__ typedef unsigned __int128 uint128;
    return static_cast<uint64_t>((static_cast<uint128>(r) * n) >> 64);
#endif
}

void permutation(std::vector<uint32_t>& out, uint64_t n, uint64_t state) {
    if (n > out.capacity()) {
        // Release the old buffer before taking the new one. Growing a vector in place reallocates
        // geometrically, with the old buffer alive during the copy, so a file just past the
        // previous one's size could hold up to twice the 4 bytes per element for good. Emptied
        // first, reserve(n) allocates exactly n elements on both libstdc++ and MSVC's library.
        std::vector<uint32_t>().swap(out);
        out.reserve(static_cast<size_t>(n));
    }
    out.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<uint32_t>(i);
    }
    for (uint64_t i = n; i > 1; --i) {
        // i - 1 is the position being fixed, and its partner is drawn from [0, i).
        const uint64_t j = draw(next(state), i);
        std::swap(out[static_cast<size_t>(i - 1)], out[static_cast<size_t>(j)]);
    }
}

}  // namespace flint::data
