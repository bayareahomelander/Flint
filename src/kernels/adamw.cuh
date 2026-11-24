// One AdamW step over a list of FP32 master weights, with the gradient's unscaling, clipping
// and skipping folded in, and the loss scale's update.
#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace flint {

// The workspace adamw needs, in bytes: the step's scalars, which the first kernel leaves for
// the second.
constexpr size_t kAdamwWorkspaceBytes = 16;

// A step's hyperparameters, flint_adamw_config's fields.
struct AdamwSettings {
    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    double max_norm;
    long long growth_interval;
};

// Launches the two kernels on stream: the first reads *norm, *nonfinite and the state
// (flint_adamw_state's layout, 24 bytes) and writes the step's scalars to the workspace and
// the new state; the second updates every tensor, unless the step is skipped. tensors holds
// tensor_count flint_adamw_tensor entries in device memory. Expects flint_adamw's checks and
// settings. Returns the first launch error, if any.
cudaError_t launch_adamw(const void* tensors, size_t tensor_count, const void* norm,
                         const void* nonfinite, void* state, const AdamwSettings& settings,
                         void* workspace, cudaStream_t stream);

}  // namespace flint
