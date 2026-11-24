// flint_adamw and its workspace size: check the arguments, then launch the kernels.
#include "api/adamw_config.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>

#include "common/checks.h"
#include "common/error.h"
#include "kernels/adamw.cuh"

// The kernels read the table and the state in these layouts.
static_assert(sizeof(flint_adamw_tensor) == 48 && offsetof(flint_adamw_tensor, count) == 32 &&
                  offsetof(flint_adamw_tensor, decay) == 40,
              "a flint_adamw_tensor must be four pointers, a size_t and an int, 48 bytes");
static_assert(sizeof(flint_adamw_state) == 24 && offsetof(flint_adamw_state, step) == 8 &&
                  offsetof(flint_adamw_state, skipped) == 16,
              "a flint_adamw_state must be 24 bytes");
static_assert(sizeof(flint_adamw_config) == 56, "a flint_adamw_config must be 56 bytes");

namespace flint {

flint_status check_adamw_config(const char* function, const flint_adamw_config* config) {
    if (config == nullptr) {
        return fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: config is null", function);
    }
    const flint_adamw_config& c = *config;
    if (!(std::isfinite(c.lr) && c.lr >= 0.0)) {
        return fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: lr (%g) must be finite and at least 0",
                    function, c.lr);
    }
    if (!(c.beta1 >= 0.0 && c.beta1 < 1.0)) {
        return fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: beta1 (%g) must be at least 0 and below 1",
                    function, c.beta1);
    }
    if (!(c.beta2 >= 0.0 && c.beta2 < 1.0)) {
        return fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: beta2 (%g) must be at least 0 and below 1",
                    function, c.beta2);
    }
    if (!(std::isfinite(c.eps) && c.eps > 0.0)) {
        return fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: eps (%g) must be finite and above 0",
                    function, c.eps);
    }
    if (!(std::isfinite(c.weight_decay) && c.weight_decay >= 0.0)) {
        return fail(FLINT_ERROR_INVALID_ARGUMENT,
                    "%s: weight_decay (%g) must be finite and at least 0", function,
                    c.weight_decay);
    }
    if (!(c.max_norm > 0.0)) {
        return fail(FLINT_ERROR_INVALID_ARGUMENT,
                    "%s: max_norm (%g) must be above 0, or +inf for no clipping", function,
                    c.max_norm);
    }
    if (c.growth_interval < 1 || c.growth_interval > INT32_MAX) {
        return fail(FLINT_ERROR_INVALID_ARGUMENT,
                    "%s: growth_interval (%lld) must be from 1 to %d", function,
                    c.growth_interval, INT32_MAX);
    }
    return FLINT_OK;
}

}  // namespace flint

flint_status flint_adamw_workspace(size_t* bytes) {
    if (bytes == nullptr) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "flint_adamw_workspace: bytes is null");
    }
    *bytes = flint::kAdamwWorkspaceBytes;
    return FLINT_OK;
}

flint_status flint_adamw(const void* tensors, size_t tensor_count, const void* norm,
                         const void* nonfinite, void* state, const flint_adamw_config* config,
                         void* workspace, size_t workspace_bytes, flint_stream stream) {
    constexpr const char* name = "flint_adamw";
    const flint_status configured = flint::check_adamw_config(name, config);
    if (configured != FLINT_OK) {
        return configured;
    }
    if (tensor_count > SIZE_MAX / sizeof(flint_adamw_tensor)) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT, "%s: tensor_count (%zu) can be up to %zu",
                           name, tensor_count, SIZE_MAX / sizeof(flint_adamw_tensor));
    }
    constexpr size_t needed = flint::kAdamwWorkspaceBytes;
    if (workspace_bytes < needed) {
        return flint::fail(FLINT_ERROR_INVALID_ARGUMENT,
                           "%s: needs %zu bytes of workspace, given %zu", name, needed,
                           workspace_bytes);
    }
    // Outputs first: the state and the workspace, which the first kernel writes. Each must not
    // overlap any buffer after it. The norm, the count and the table are only read.
    const flint::Buffer buffers[] = {
        {"state", state, sizeof(flint_adamw_state)},
        {"workspace", workspace, needed},
        {"norm", norm, 8},
        {"nonfinite", nonfinite, 8},
        {"tensors", tensors, tensor_count * sizeof(flint_adamw_tensor)},
    };
    const flint_status checked = flint::check_buffers(name, buffers, std::size(buffers), 2);
    if (checked != FLINT_OK) {
        return checked;
    }
    const flint::AdamwSettings settings{config->lr,           config->beta1,
                                        config->beta2,        config->eps,
                                        config->weight_decay, config->max_norm,
                                        config->growth_interval};
    FLINT_CUDA_CHECK(
        flint::launch_adamw(tensors, tensor_count, norm, nonfinite, state, settings, workspace,
                            stream));
    return FLINT_OK;
}
