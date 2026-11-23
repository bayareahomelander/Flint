// flint_gradient_norm and its workspace size: check the arguments, then launch the kernels.
#include <cstddef>
#include <cstdint>
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/gradient_norm.cuh"

// The kernels read the table in this layout.
static_assert(sizeof(flint_tensor) == 16 && offsetof(flint_tensor, count) == 8,
              "a flint_tensor must be a pointer and a size_t, 16 bytes");

flint_status flint_gradient_norm_workspace(size_t* bytes) {
    if (bytes == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "flint_gradient_norm_workspace: bytes is null");
    }
    *bytes = flint::kGradientNormWorkspaceBytes;
    return FLINT_OK;
}

flint_status flint_gradient_norm(void* norm, void* nonfinite, const void* tensors,
                                 size_t tensor_count, void* workspace, size_t workspace_bytes,
                                 flint_stream stream) {
    constexpr const char* name = "flint_gradient_norm";
    if (tensor_count > SIZE_MAX / sizeof(flint_tensor)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: tensor_count (%zu) can be up to %zu",
                           name, tensor_count, SIZE_MAX / sizeof(flint_tensor));
    }
    constexpr size_t needed = flint::kGradientNormWorkspaceBytes;
    if (workspace_bytes < needed) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: needs %zu bytes of workspace, given %zu", name, needed,
                           workspace_bytes);
    }
    // Outputs first: norm, nonfinite and the workspace, which the first pass writes. Each
    // must not overlap any buffer after it. The table is only read.
    const flint::Buffer buffers[] = {
        {"norm", norm, 8},
        {"nonfinite", nonfinite, 8},
        {"workspace", workspace, needed},
        {"tensors", tensors, tensor_count * sizeof(flint_tensor)},
    };
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 3);
    if (checked != FLINT_OK) {
        return checked;
    }
    FLINT_CUDA_CHECK(
        flint::launch_gradient_norm(norm, nonfinite, tensors, tensor_count, workspace, stream));
    return FLINT_OK;
}
