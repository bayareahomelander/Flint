// flint_heads_split and flint_heads_merge: check the arguments, then launch the
// kernels. They differ only in which of y and x is the heads layout, so one
// function checks both.
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/heads.cuh"

namespace {

flint_status move_heads(bool split, void* y, const void* x, size_t batches, size_t length,
                        size_t heads, size_t head_width, size_t row_width, size_t first,
                        flint_stream stream) {
    const char* name = split ? "flint_heads_split" : "flint_heads_merge";
    if (row_width == 0 || row_width % 8 != 0 || row_width > flint::kHeadsMaxRowWidth) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: row_width (%zu) must be a multiple of 8 from 8 to %zu", name,
                           row_width, flint::kHeadsMaxRowWidth);
    }
    if (head_width == 0 || head_width % 8 != 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: head_width (%zu) must be a positive multiple of 8", name,
                           head_width);
    }
    if (first % 8 != 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: first (%zu) must be a multiple of 8",
                           name, first);
    }
    if (heads == 0) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: heads must be positive", name);
    }
    // Divides before it multiplies, so a product that wraps is still too large.
    if (heads > row_width / head_width || first > row_width - heads * head_width) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: %zu heads of %zu columns from column %zu run past row_width (%zu)",
                           name, heads, head_width, first, row_width);
    }
    if (length != 0 && batches > flint::kHeadsMaxRows / length) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: batches (%zu) x length (%zu) can be up to %zu rows", name, batches,
                           length, flint::kHeadsMaxRows);
    }
    if (batches == 0 || length == 0) {
        return FLINT_OK;
    }
    const size_t rows = batches * length;

    // The limits keep rows * row_width * 2 far below 2^64. The heads layout
    // never holds more than the rows layout: heads * head_width <= row_width.
    const size_t rows_bytes = rows * row_width * 2;
    const size_t heads_bytes = rows * heads * head_width * 2;
    const flint::Buffer buffers[] = {
        {"y", y, split ? heads_bytes : rows_bytes},
        {"x", x, split ? rows_bytes : heads_bytes},
    };
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 1);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK((split ? flint::launch_heads_split : flint::launch_heads_merge)(
        y, x, rows, length, heads, head_width, row_width, first, stream));
    return FLINT_OK;
}

}  // namespace

flint_status flint_heads_split(void* y, const void* x, size_t batches, size_t length, size_t heads,
                               size_t head_width, size_t row_width, size_t first,
                               flint_stream stream) {
    return move_heads(true, y, x, batches, length, heads, head_width, row_width, first, stream);
}

flint_status flint_heads_merge(void* y, const void* x, size_t batches, size_t length, size_t heads,
                               size_t head_width, size_t row_width, size_t first,
                               flint_stream stream) {
    return move_heads(false, y, x, batches, length, heads, head_width, row_width, first, stream);
}
