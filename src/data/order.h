// The arithmetic of the loader's order: SplitMix64, the streams' keys, the bounded draw and
// Fisher-Yates. Everything is unsigned 64-bit integers, wrapping mod 2^64, with no floating
// point and nothing from <random> or std::shuffle (their algorithms are each standard library's
// own), so every machine and compiler gives the same order from one seed. No I/O happens here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flint::data {

// The key of the stream that orders an epoch's files. File f's sequences use key f.
constexpr uint64_t kFilesKey = 0xFFFFFFFFFFFFFFFFULL;

// SplitMix64's finalizer.
uint64_t mix(uint64_t x);

// SplitMix64's step: advances state, and returns the output for the new state.
uint64_t next(uint64_t& state);

// The output SplitMix64 gives from state x: mix(x + 0x9E3779B97F4A7C15).
uint64_t hash(uint64_t x);

// The starting state of the stream for (seed, epoch, key): hash(hash(hash(seed) ^ epoch) ^ key).
uint64_t stream_start(uint64_t seed, uint64_t epoch, uint64_t key);

// A value in [0, n) from the 64-bit output r: the high 64 bits of the 128-bit product r x n. It
// is biased by at most n / 2^64 and has no rejection loop. n must be at least 1.
uint64_t draw(uint64_t r, uint64_t n);

// Fisher-Yates of [0, 1, ..., n - 1] into out, drawing from the stream that starts at state: for
// i from n - 1 down to 1, swap out[i] with out[draw(next(state), i + 1)]. An n of 0 or 1 draws
// nothing. n must be at most 2^32, so every element fits in a uint32. out's buffer is replaced,
// not grown, when n exceeds its capacity, so it holds 4 bytes per element at most.
void permutation(std::vector<uint32_t>& out, uint64_t n, uint64_t state);

}  // namespace flint::data
