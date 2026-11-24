// One optimizer step: AdamW over a list of FP32 master weights, as PyTorch's
// GradScaler.unscale_, clip_grad_norm_, a fused AdamW and GradScaler.update do it, in two
// kernels that never wait on the host. The gradients are still scaled; the gradient norm
// (gradient_norm.cu) has left their FP64 norm and the count of their values that aren't finite
// in device memory, and these kernels read both.
//
// The first kernel is one block of one thread (the work is a few dozen FP64 operations). From
// the norm, the count and the state (the loss scale, the growth tracker, AdamW's step count and
// the count of skipped steps) it decides the step and leaves its scalars in the workspace:
//  - The step is skipped if any gradient value isn't finite, if the loss scale isn't finite and
//    positive (a state that no step or reset leaves: a negative, infinite or NaN scale skips
//    every step, a visible stall instead of a wrong step), or if unscaling isn't safe: 1 / scale
//    overflows FP32 (GradScaler's own inv_scale: a scale of 0, or of about 2^-128 or less), or
//    the unscaled norm, norm / scale, isn't below 2^127 (which includes an infinite or NaN one).
//    The bound is why no unscaled value can overflow: a value whose unscaling overflows FP32 has
//    |g| / scale >= FLT_MAX, so the unscaled norm is past 2^127 and this skips, while below it
//    every |g * multiplier| < 2^127 (1 + 2^-23) < FLT_MAX. PyTorch 2.6 doesn't skip at these two
//    edges: GradScaler's unscale_ checks the gradients before it multiplies them, so an
//    overflowing inverse or value goes on into its clip and step as inf, and the weights become
//    NaN. A skip halves the scale, resets the growth tracker, counts the skip and changes
//    nothing else: no weight, no moment, not AdamW's step count.
//  - Otherwise the gradient's unscaling and clipping fold into one multiplier. The unscaled norm
//    is norm / scale (the division is exact for power-of-two scales, which GradScaler's are: from
//    2^16 they are only doubled and halved), the clip coefficient is max_norm / (unscaled + 1e-6)
//    clamped to at most 1 (a NaN from inf / inf also gives 1), and each gradient value is
//    multiplied by the FP32 rounding of coefficient / scale, finite since 1 / scale is. No
//    unscaled gradient is ever stored.
//  - Then the step count advances and the bias corrections follow from it, in FP64, as fused
//    AdamW computes them: beta ** step by repeated squaring, bc1 = 1 - beta1 ** step rounded to
//    FP32, bc2's square root the same, and step_size = lr / bc1 rounded to FP32.
//  - The loss scale update is torch._amp_update_scale_'s: a clean step advances the growth
//    tracker, and when it reaches the growth interval the scale doubles, unless that gives inf,
//    and the tracker resets. (The test is >= where PyTorch's is ==, the same for a fixed
//    interval.)
//
// The second kernel is 1,024 blocks of 256 threads, whatever the GPU, over the same walk as
// gradient_norm_partials': values are cut into chunks that never span tensors, numbered globally
// in table order, and block b takes the chunks g with g % 1024 == b. Chunks are 2,048 values:
// thread tid takes, for j = 0 and 1, the group of 4 values at c * 2048 + j * 1024 + 4 * tid.
// Every thread first reads the workspace's scalars and returns at once if the step is skipped
// (the same branch for the whole grid). Otherwise it loads both groups of the weight, the
// gradient and the two moments, then computes, then stores. A group is 16-byte loads and stores
// when all four of its tensor's arrays are 16-byte aligned and the whole group is in range;
// otherwise the values in range go one by one, and nothing is stored past the tensor's count.
// The loads and stores are written in PTX as global instructions, since the compiler can't tell
// that a pointer read from the table is in global memory. The grid is fixed and each value's
// arithmetic doesn't depend on the mapping, so the mapping only affects speed.
//
// Each value's update, with the precision mix of PyTorch's fused AdamW, whose scalars are
// doubles. It was measured over 100 steps on 524,288 values against an FP64 reference: with
// everything in FP32, the moments' error reaches 3.5 times that of PyTorch's FP32 fused AdamW,
// past the gate's 2 times, and with this mix it stays at 0.9 times:
//  - g = gradient * multiplier, in FP32;
//  - the decay, if the tensor has it and weight_decay isn't 0 (even when lr is 0): the weight
//    in FP64, minus (lr * weight_decay) * weight, rounded to FP32;
//  - the moments in FP64, from the FP32 values and g, rounded to FP32: m = beta1 * m +
//    (1 - beta1) * g, and v = beta2 * v + ((1 - beta2) * g) * g;
//  - the denominator is sqrt(v) / bc2_sqrt in FP32, then + eps in FP64, rounded to FP32;
//  - weight = weight - (step_size * m) / denominator, in FP32.
//
// The result is deterministic and the same bits on every GPU: nothing is summed across values
// and there are no atomics. Every floating-point operation is an intrinsic with its rounding
// written out (__dmul_rn, __fdiv_rn and so on), so that the compiler can't contract two into
// an FMA, and a numpy transcription (the tests') matches bit for bit. Nothing flushes
// subnormals to 0 (no .FTZ, no fast-math): v is often subnormal for tiny gradients, and must
// stay exact. The host computes the three doubles that the update needs and that are one IEEE
// operation each (1 - beta1, 1 - beta2 and lr * weight_decay), which are the same bits as the
// device's.
#include "kernels/adamw.cuh"

namespace flint {
namespace {

constexpr int kBlocks = 1024;
constexpr int kThreads = 256;
// The values of a group, which one thread loads at once (a 16-byte load).
constexpr int kGroup = 4;
// The groups a thread takes from a chunk, loaded together.
constexpr int kUnroll = 2;
constexpr size_t kChunk = size_t{kThreads} * kGroup * kUnroll;
static_assert(kChunk == 2048, "the header comment's numbers assume chunks of 2,048");

// A table entry, flint_adamw_tensor's layout.
struct Tensor {
    float* param;
    const float* grad;
    float* exp_avg;
    float* exp_avg_sq;
    size_t count;
    int decay;
};
static_assert(sizeof(Tensor) == 48, "a flint_adamw_tensor is 48 bytes");

// The state, flint_adamw_state's layout.
struct State {
    float scale;
    int growth_tracker;
    long long step;
    long long skipped;
};
static_assert(sizeof(State) == 24, "a flint_adamw_state is 24 bytes");

// What the first kernel leaves for the second. On a skip only skip is meaningful.
struct Scalars {
    float multiplier;
    float step_size;
    float bc2_sqrt;
    int skip;
};
static_assert(sizeof(Scalars) == kAdamwWorkspaceBytes, "the workspace holds one Scalars");

// The doubles the update needs.
struct Update {
    double beta1;
    double one_minus_beta1;
    double beta2;
    double one_minus_beta2;
    double eps;
    double decay;  // lr * weight_decay
    int use_decay;  // weight_decay != 0
};

// base ** t for t >= 1 by repeated squaring, every product rounded to FP64.
__device__ double pow_by_squaring(double base, long long t) {
    double result = 1.0;
    unsigned long long e = static_cast<unsigned long long>(t);
    while (e != 0) {
        if (e & 1) {
            result = __dmul_rn(result, base);
        }
        e >>= 1;
        if (e != 0) {
            base = __dmul_rn(base, base);
        }
    }
    return result;
}

// One thread: the step's scalars and the new state, as the header comment says.
__global__ void adamw_prepare(Scalars* __restrict__ scalars, State* __restrict__ state,
                              const double* __restrict__ norm,
                              const unsigned long long* __restrict__ nonfinite,
                              AdamwSettings settings) {
    const State old = *state;
    State next = old;
    const double scale = static_cast<double>(old.scale);
    const double unscaled = __ddiv_rn(*norm, scale);
    double clip = __ddiv_rn(settings.max_norm, __dadd_rn(unscaled, 1e-6));
    if (!(clip < 1.0)) {
        clip = 1.0;
    }
    const float multiplier = __double2float_rn(__ddiv_rn(clip, scale));
    // GradScaler's inv_scale: self._scale.double().reciprocal().float().
    const float inverse = __double2float_rn(__ddiv_rn(1.0, scale));
    const bool skip = *nonfinite != 0 || !(old.scale > 0.0f && isfinite(old.scale)) ||
                      !isfinite(inverse) || !(unscaled < 0x1p127);
    Scalars out{};
    if (skip) {
        next.scale = __fmul_rn(old.scale, 0.5f);
        next.growth_tracker = 0;
        next.skipped = old.skipped + 1;
        out.skip = 1;
    } else {
        const long long t = old.step + 1;
        const float bc1 = __double2float_rn(__dsub_rn(1.0, pow_by_squaring(settings.beta1, t)));
        const float bc2_sqrt = __double2float_rn(
            __dsqrt_rn(__dsub_rn(1.0, pow_by_squaring(settings.beta2, t))));
        out.multiplier = multiplier;
        out.step_size = __double2float_rn(__ddiv_rn(settings.lr, static_cast<double>(bc1)));
        out.bc2_sqrt = bc2_sqrt;
        out.skip = 0;
        next.step = t;
        const long long successful = static_cast<long long>(old.growth_tracker) + 1;
        if (successful >= settings.growth_interval) {
            const float grown = __fmul_rn(old.scale, 2.0f);
            if (isfinite(grown)) {
                next.scale = grown;
            }
            next.growth_tracker = 0;
        } else {
            next.growth_tracker = static_cast<int>(successful);
        }
    }
    *scalars = out;
    *state = next;
}

// Global loads and stores, in PTX because a pointer from the table could be in any space as far
// as the compiler knows. The weights and moments are also written, so they can't take the
// read-only path (__ldg).
__device__ __forceinline__ float4 load4(const float* pointer) {
    float4 q;
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(q.x), "=f"(q.y), "=f"(q.z), "=f"(q.w)
                 : "l"(pointer)
                 : "memory");
    return q;
}

__device__ __forceinline__ float load1(const float* pointer) {
    float value;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(value) : "l"(pointer) : "memory");
    return value;
}

__device__ __forceinline__ void store4(float* pointer, float x, float y, float z, float w) {
    asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(pointer), "f"(x), "f"(y), "f"(z), "f"(w)
                 : "memory");
}

__device__ __forceinline__ void store1(float* pointer, float value) {
    asm volatile("st.global.f32 [%0], %1;" : : "l"(pointer), "f"(value) : "memory");
}

// One value's update, in the order and precision of the header comment.
__device__ __forceinline__ void update_value(float& p, float& m, float& v, float gradient,
                                             bool decay, const Scalars& s, const Update& u) {
    const float g = __fmul_rn(gradient, s.multiplier);
    if (decay) {
        const double p64 = static_cast<double>(p);
        p = __double2float_rn(__dsub_rn(p64, __dmul_rn(u.decay, p64)));
    }
    const double g64 = static_cast<double>(g);
    m = __double2float_rn(
        __dadd_rn(__dmul_rn(u.beta1, static_cast<double>(m)), __dmul_rn(u.one_minus_beta1, g64)));
    v = __double2float_rn(__dadd_rn(__dmul_rn(u.beta2, static_cast<double>(v)),
                                    __dmul_rn(__dmul_rn(u.one_minus_beta2, g64), g64)));
    const float root = __fdiv_rn(__fsqrt_rn(v), s.bc2_sqrt);
    const float denominator = __double2float_rn(__dadd_rn(static_cast<double>(root), u.eps));
    p = __fsub_rn(p, __fdiv_rn(__fmul_rn(s.step_size, m), denominator));
}

// Block b: every group of the chunks g with g % 1024 == b, unless the step is skipped.
__global__ void __launch_bounds__(kThreads)
    adamw_update(const Scalars* __restrict__ scalars, const Tensor* __restrict__ tensors,
                 size_t tensor_count, Update update) {
    const Scalars s = *scalars;
    if (s.skip != 0) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned long long block = blockIdx.x;

    // The global number of the current tensor's first chunk.
    unsigned long long first = 0;
    for (size_t t = 0; t < tensor_count; ++t) {
        const Tensor tensor = tensors[t];
        const unsigned long long chunks = tensor.count / kChunk + (tensor.count % kChunk != 0);
        // The first chunk of this tensor with g % 1024 == block, as an index into the tensor.
        const unsigned long long skip = (block + kBlocks - first % kBlocks) % kBlocks;
        first += chunks;
        const bool vector_access =
            (reinterpret_cast<size_t>(tensor.param) | reinterpret_cast<size_t>(tensor.grad) |
             reinterpret_cast<size_t>(tensor.exp_avg) |
             reinterpret_cast<size_t>(tensor.exp_avg_sq)) %
                16 ==
            0;
        const bool decay = tensor.decay != 0 && update.use_decay != 0;
        for (unsigned long long c = skip; c < chunks; c += kBlocks) {
            float g[kUnroll][kGroup] = {};
            float p[kUnroll][kGroup] = {};
            float m[kUnroll][kGroup] = {};
            float v[kUnroll][kGroup] = {};
            size_t start[kUnroll];
            size_t left[kUnroll];
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                start[j] = c * kChunk + static_cast<size_t>(j * kThreads * kGroup) +
                           size_t{tid} * kGroup;
                left[j] = start[j] < tensor.count ? tensor.count - start[j] : 0;
                if (vector_access && left[j] >= kGroup) {
                    const float4 qg = load4(tensor.grad + start[j]);
                    const float4 qp = load4(tensor.param + start[j]);
                    const float4 qm = load4(tensor.exp_avg + start[j]);
                    const float4 qv = load4(tensor.exp_avg_sq + start[j]);
                    g[j][0] = qg.x, g[j][1] = qg.y, g[j][2] = qg.z, g[j][3] = qg.w;
                    p[j][0] = qp.x, p[j][1] = qp.y, p[j][2] = qp.z, p[j][3] = qp.w;
                    m[j][0] = qm.x, m[j][1] = qm.y, m[j][2] = qm.z, m[j][3] = qm.w;
                    v[j][0] = qv.x, v[j][1] = qv.y, v[j][2] = qv.z, v[j][3] = qv.w;
                } else {
#pragma unroll
                    for (int k = 0; k < kGroup; ++k) {
                        if (static_cast<size_t>(k) < left[j]) {
                            g[j][k] = load1(tensor.grad + start[j] + k);
                            p[j][k] = load1(tensor.param + start[j] + k);
                            m[j][k] = load1(tensor.exp_avg + start[j] + k);
                            v[j][k] = load1(tensor.exp_avg_sq + start[j] + k);
                        }
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
#pragma unroll
                for (int k = 0; k < kGroup; ++k) {
                    if (static_cast<size_t>(k) < left[j]) {
                        update_value(p[j][k], m[j][k], v[j][k], g[j][k], decay, s, update);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                if (vector_access && left[j] >= kGroup) {
                    store4(tensor.param + start[j], p[j][0], p[j][1], p[j][2], p[j][3]);
                    store4(tensor.exp_avg + start[j], m[j][0], m[j][1], m[j][2], m[j][3]);
                    store4(tensor.exp_avg_sq + start[j], v[j][0], v[j][1], v[j][2], v[j][3]);
                } else {
#pragma unroll
                    for (int k = 0; k < kGroup; ++k) {
                        if (static_cast<size_t>(k) < left[j]) {
                            store1(tensor.param + start[j] + k, p[j][k]);
                            store1(tensor.exp_avg + start[j] + k, m[j][k]);
                            store1(tensor.exp_avg_sq + start[j] + k, v[j][k]);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

cudaError_t launch_adamw(const void* tensors, size_t tensor_count, const void* norm,
                         const void* nonfinite, void* state, const AdamwSettings& settings,
                         void* workspace, cudaStream_t stream) {
    auto* scalars = static_cast<Scalars*>(workspace);
    adamw_prepare<<<1, 1, 0, stream>>>(scalars, static_cast<State*>(state),
                                       static_cast<const double*>(norm),
                                       static_cast<const unsigned long long*>(nonfinite),
                                       settings);
    if (const cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
        return error;
    }
    // One IEEE operation each, the same bits as the device's.
    const Update update{settings.beta1,
                        1.0 - settings.beta1,
                        settings.beta2,
                        1.0 - settings.beta2,
                        settings.eps,
                        settings.lr * settings.weight_decay,
                        settings.weight_decay != 0.0};
    adamw_update<<<kBlocks, kThreads, 0, stream>>>(
        scalars, static_cast<const Tensor*>(tensors), tensor_count, update);
    return cudaGetLastError();
}

}  // namespace flint
