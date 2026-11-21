// A tiny kernel that reads the SM's cycle counter and the global timer back to
// back, so the benchmark can measure the SM clock from outside the kernel under
// test, which stays as it is. It lives in the benchmark, not the library, and
// in namespace flint so compute-sanitizer's kernel filter covers it.
//
// Two probes on the same SM give that SM's mean clock over the interval
// between them: its cycles per nanosecond. The benchmark takes one right
// before and one right after a timed sample, so the interval is the sample
// plus about 0.14 ms of idle GPU around it (a sync and a probe on each side,
// as measured on the laptop): about 1% of a quick sample and 0.1% of a
// standard one, which reads the clock slightly low. NVML's clock, by contrast,
// can lag a short sample by half a second.
//
// PTX documents %globaltimer as intended for NVIDIA's tools, specific to the
// target, and liable to change. The benchmark's tests bound the clock this
// gives (against the device's maximum clock and the tensor peak), so a change
// in what the timer counts shows up there.
#pragma once

#include <cuda_runtime_api.h>

namespace flint::microbench {

// smid values are below this, so a slot array has 2 * kClockProbeSlots entries.
constexpr int kClockProbeSlots = 1024;

// Launches one block per SM. Thread 0 of each block stores (clock64,
// %globaltimer), read back to back in that order, at slots[2 * smid] and
// slots[2 * smid + 1] for the SM it ran on, unless it moved to another SM
// while reading (then it stores nothing). The caller zeroes the slots it reads
// before each probe and reads the pair of one SM, the same SM every time:
// PTX gives SM identifiers as 0 to %nsmid - 1 and does not promise they are
// contiguous, so the caller picks the lowest that has a pair.
//
// One SM is used because the SMs' clock64 counters differ by hundreds of
// millions of cycles, in a few clusters, and the clusters drift apart by about
// 0.1% of the elapsed cycles, so two probes compared across SMs would be wrong
// by far more than the clock they measure. The global timer is the same on
// every SM. A single SM's counter runs at its clock and never stops or resets,
// even when the GPU idles.
//
// Each block has more than half the threads an SM can hold, so no SM takes two
// blocks, and a probe launched on an idle GPU reaches every SM. That holds
// where an SM holds fewer than 2048 threads (sm_75: 1024, sm_89: 1536). With
// 2048 a block is exactly half, two can share an SM and another SM can go
// without. A launch can miss an SM in any case, and its slot's timer stays
// zero, so the caller retries.
cudaError_t launch_clock_probe(int sm_count, unsigned long long* slots, cudaStream_t stream);

}  // namespace flint::microbench
