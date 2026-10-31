// The copy benchmark: flint_copy against cudaMemcpyAsync, device to device.
#pragma once

#include <cuda_runtime_api.h>

#include "harness.h"

namespace bench {

CaseResult bench_copy(unsigned long long bytes, const DeviceInfo& device, Monitor& monitor,
                      const Options& options, cudaStream_t stream);

}  // namespace bench
