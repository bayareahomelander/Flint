// flint_embedding_forward: checks the arguments, then launches the kernel.
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/embedding.cuh"

flint_status flint_embedding_forward(void* x, const void* tokens, const void* wte, const void* wpe,
                                     size_t sequences, size_t length, size_t vocab, size_t width,
                                     flint_stream stream) {
    constexpr const char* name = "flint_embedding_forward";
    if (width == 0 || width % 8 != 0 || width > flint::kEmbeddingMaxWidth) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: width (%zu) must be a multiple of 8 from 8 to %zu", name, width,
                           flint::kEmbeddingMaxWidth);
    }
    if (vocab == 0 || vocab > flint::kEmbeddingMaxVocab) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: vocab (%zu) must be from 1 to %zu",
                           name, vocab, flint::kEmbeddingMaxVocab);
    }
    // Divides before it multiplies, so a product that wraps is still too large.
    if (length != 0 && sequences > flint::kEmbeddingMaxRows / length) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: sequences (%zu) x length (%zu) can be up to %zu rows", name,
                           sequences, length, flint::kEmbeddingMaxRows);
    }
    if (sequences == 0 || length == 0) {
        return FLINT_OK;
    }
    const size_t rows = sequences * length;

    // The limits keep rows * width * 4 far below 2^64. Only length rows of wpe
    // are read.
    const flint::Buffer buffers[] = {
        {"x", x, rows * width * 4},
        {"tokens", tokens, rows * 2},
        {"wte", wte, vocab * width * 4},
        {"wpe", wpe, length * width * 4},
    };
    // x must not overlap any input; the inputs may share memory, since they
    // are only read.
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 1);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(
        flint::launch_embedding_forward(x, tokens, wte, wpe, rows, length, width, stream));
    return FLINT_OK;
}
