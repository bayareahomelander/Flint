// The embedding's forward and backward passes and the backward pass's workspace size:
// check the arguments, then launch the kernels.
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/embedding.cuh"
#include "kernels/embedding_backward.cuh"

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

namespace {

flint_status check_backward_dims(const char* name, size_t sequences, size_t length, size_t vocab,
                                 size_t width) {
    if (width == 0 || width % 8 != 0 || width > flint::kEmbeddingBackwardMaxWidth) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: width (%zu) must be a multiple of 8 from 8 to %zu", name, width,
                           flint::kEmbeddingBackwardMaxWidth);
    }
    if (vocab == 0 || vocab > flint::kEmbeddingMaxVocab) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: vocab (%zu) must be from 1 to %zu",
                           name, vocab, flint::kEmbeddingMaxVocab);
    }
    if (length == 0 || length > flint::kEmbeddingBackwardMaxPositionValues / width) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: length (%zu) must be from 1 to %zu, so that length x width (%zu) "
                           "is at most %zu",
                           name, length, flint::kEmbeddingBackwardMaxPositionValues / width, width,
                           flint::kEmbeddingBackwardMaxPositionValues);
    }
    // Divides before it multiplies, so a product that wraps is still too large.
    if (sequences == 0 || sequences > flint::kEmbeddingBackwardMaxRows / length) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: sequences (%zu) must be from 1 to %zu, so that sequences x length "
                           "is at most %zu rows",
                           name, sequences, flint::kEmbeddingBackwardMaxRows / length,
                           flint::kEmbeddingBackwardMaxRows);
    }
    return FLINT_OK;
}

}  // namespace

flint_status flint_embedding_backward(void* dwte, void* dwpe, const void* dx, const void* tokens,
                                      size_t sequences, size_t length, size_t vocab, size_t width,
                                      int accumulate_wte, int accumulate_wpe, void* workspace,
                                      size_t workspace_bytes, flint_stream stream) {
    constexpr const char* name = "flint_embedding_backward";
    if (const flint_status status = check_backward_dims(name, sequences, length, vocab, width);
        status != FLINT_OK) {
        return status;
    }
    const size_t needed = flint::embedding_backward_layout(sequences, length, vocab, width).bytes;
    if (workspace_bytes < needed) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: needs %zu bytes of workspace, given %zu",
                           name, needed, workspace_bytes);
    }
    // Outputs first, then inputs, in the order of the parameters. The limits keep
    // rows * width * 4 far below 2^64.
    const size_t rows = sequences * length;
    const flint::Buffer buffers[] = {
        {"dwte", dwte, vocab * width * 4},
        {"dwpe", dwpe, length * width * 4},
        {"workspace", workspace, needed},
        {"dx", dx, rows * width * 4},
        {"tokens", tokens, rows * 2},
    };
    // Each output must not overlap any other buffer; the inputs may share memory,
    // since they are only read.
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 3);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(flint::launch_embedding_backward(dwte, dwpe, dx, tokens, sequences, length,
                                                      vocab, width, accumulate_wte != 0,
                                                      accumulate_wpe != 0, workspace, stream));
    return FLINT_OK;
}

flint_status flint_embedding_backward_workspace(size_t sequences, size_t length, size_t vocab,
                                                size_t width, size_t* bytes) {
    constexpr const char* name = "flint_embedding_backward_workspace";
    if (bytes == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: bytes is null", name);
    }
    if (const flint_status status = check_backward_dims(name, sequences, length, vocab, width);
        status != FLINT_OK) {
        return status;
    }
    *bytes = flint::embedding_backward_layout(sequences, length, vocab, width).bytes;
    return FLINT_OK;
}
