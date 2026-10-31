// The mma benchmark: the tensor cores' FP16 throughput with FP32 accumulation,
// timed one-sided against the GPU's peak.
#pragma once

#include <cuda_runtime_api.h>

#include "harness.h"

namespace bench {

// FP16 tensor FLOP per SM per clock with FP32 accumulation: 512 on GeForce
// Turing (sm_75) and Ada (sm_89), half their FP16-accumulate rate. 0 for
// other GPUs, whose peaks Flint doesn't know.
double tensor_flop_per_sm_per_clock(const DeviceInfo& device);

CaseResult bench_mma(int chains, const DeviceInfo& device, Monitor& monitor, const Options& options,
                     cudaStream_t stream);

}  // namespace bench
