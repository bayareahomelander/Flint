// LayerNorm's forward pass: one warp per row, all in FP32, with each output
// rounded to FP16 once.
//
// A first pass sums the row for its mean, and a second sums the squared
// deviations from that mean for the variance, which stays accurate when the
// mean is large against the spread. Both sums are compensated: each addition's
// rounding error is captured exactly (TwoSum, and fmaf for a square's) and
// summed alongside, as in Ogita, Rump and Oishi's Sum2, so a sum is about as
// accurate as one computed in twice the precision and rounded once. Without
// that, GPT-2's outlier channels near 3,000 lose digits: every small value
// added to a partial sum of about 3,000 rounds to its FP32 spacing there,
// 2.4e-4, and the mean came out less accurate than PyTorch's Welford. A third
// pass writes y. The row is re-read from global memory on each pass; at
// GPT-2's width (3 KB) the second and third come from cache.
#include "kernels/layernorm.cuh"

#include <cuda_fp16.h>

namespace flint {
namespace {

constexpr int kWarp = 32;
constexpr int kRowsPerBlock = 8;
constexpr unsigned kThreads = kWarp * kRowsPerBlock;
constexpr unsigned kAllLanes = 0xffffffffu;

// A rounded sum and the sum of the rounding errors left behind in it.
struct CompensatedSum {
    float sum = 0.0f;
    float error = 0.0f;
};

// Adds v. Knuth's TwoSum finds the error e with sum + v = t + e exactly, for
// any order of magnitude between the two.
__device__ void add(CompensatedSum& a, float v) {
    const float t = a.sum + v;
    const float v_part = t - a.sum;
    a.error += (a.sum - (t - v_part)) + (v - v_part);
    a.sum = t;
}

// Adds d^2, with the product's own rounding error. __fmul_rn keeps the
// compiler from fusing the product into the addition that follows.
__device__ void add_square(CompensatedSum& a, float d) {
    const float p = __fmul_rn(d, d);
    a.error += fmaf(d, d, -p);
    add(a, p);
}

// The total over the warp's lanes, received by every lane. At each butterfly
// step the two lanes hold the same pair in swapped order. TwoSum's error is
// exact, so it doesn't depend on that order, and FP32 addition is
// commutative, so all lanes end with the same bits, and every call adds in
// the same order.
__device__ float warp_total(CompensatedSum a) {
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        const float other_sum = __shfl_xor_sync(kAllLanes, a.sum, offset);
        const float other_error = __shfl_xor_sync(kAllLanes, a.error, offset);
        const float error = a.error + other_error;
        a.error = 0.0f;
        add(a, other_sum);
        a.error += error;
    }
    return a.sum + a.error;
}

// Vec consecutive FP32 values of a row.
template <int Vec>
struct Floats {
    float v[Vec];
};

// The i-th group of Vec values at p, which must be 16-byte aligned for Vec 4.
template <int Vec>
__device__ Floats<Vec> load(const float* __restrict__ p, int i) {
    Floats<Vec> f;
    if constexpr (Vec == 4) {
        const float4 q = reinterpret_cast<const float4*>(p)[i];
        f.v[0] = q.x;
        f.v[1] = q.y;
        f.v[2] = q.z;
        f.v[3] = q.w;
    } else {
        f.v[0] = p[i];
    }
    return f;
}

// Rounds the i-th group of Vec values to FP16 at p, which must be 8-byte
// aligned for Vec 4.
template <int Vec>
__device__ void store(__half* __restrict__ p, int i, const Floats<Vec>& f) {
    if constexpr (Vec == 4) {
        __half2* out = reinterpret_cast<__half2*>(p + 4 * i);
        out[0] = __floats2half2_rn(f.v[0], f.v[1]);
        out[1] = __floats2half2_rn(f.v[2], f.v[3]);
    } else {
        p[i] = __float2half_rn(f.v[0]);
    }
}

// Vec is 4 when cols is a multiple of 4, so every row starts 16-byte aligned
// and its values move four at a time, and 1 otherwise. Each lane takes every
// 32nd group of a row, so the order of its partial sums depends only on cols.
template <int Vec>
__global__ void __launch_bounds__(kThreads)
    layernorm_forward(__half* __restrict__ y, float* __restrict__ mean,
                      float* __restrict__ rstd, const float* __restrict__ x,
                      const float* __restrict__ gamma, const float* __restrict__ beta,
                      size_t rows, int cols, float eps) {
    const size_t row = size_t{blockIdx.x} * kRowsPerBlock + threadIdx.x / kWarp;
    // A warp's lanes share a row, so whole warps leave together and the
    // shuffles below always have all 32 lanes.
    if (row >= rows) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x % kWarp);
    const float* x_row = x + row * cols;
    __half* y_row = y + row * cols;
    const int groups = cols / Vec;
    const float n = static_cast<float>(cols);

    CompensatedSum sum;
    for (int i = lane; i < groups; i += kWarp) {
        const Floats<Vec> f = load<Vec>(x_row, i);
        for (int j = 0; j < Vec; ++j) {
            add(sum, f.v[j]);
        }
    }
    const float row_mean = warp_total(sum) / n;

    CompensatedSum squares;
    for (int i = lane; i < groups; i += kWarp) {
        const Floats<Vec> f = load<Vec>(x_row, i);
        for (int j = 0; j < Vec; ++j) {
            add_square(squares, f.v[j] - row_mean);
        }
    }
    // Correctly rounded, unlike rsqrtf.
    const float row_rstd = 1.0f / sqrtf(warp_total(squares) / n + eps);

    for (int i = lane; i < groups; i += kWarp) {
        Floats<Vec> f = load<Vec>(x_row, i);
        const Floats<Vec> g = load<Vec>(gamma, i);
        const Floats<Vec> b = load<Vec>(beta, i);
        for (int j = 0; j < Vec; ++j) {
            f.v[j] = fmaf((f.v[j] - row_mean) * row_rstd, g.v[j], b.v[j]);
        }
        store<Vec>(y_row, i, f);
    }
    if (lane == 0) {
        mean[row] = row_mean;
        rstd[row] = row_rstd;
    }
}

}  // namespace

cudaError_t launch_layernorm_forward(void* y, void* mean, void* rstd, const void* x,
                                     const void* gamma, const void* beta, size_t rows, int cols,
                                     float eps, cudaStream_t stream) {
    const auto blocks = static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    auto* y_half = static_cast<__half*>(y);
    auto* mean_f = static_cast<float*>(mean);
    auto* rstd_f = static_cast<float*>(rstd);
    const auto* x_f = static_cast<const float*>(x);
    const auto* gamma_f = static_cast<const float*>(gamma);
    const auto* beta_f = static_cast<const float*>(beta);
    if (cols % 4 == 0) {
        layernorm_forward<4><<<blocks, kThreads, 0, stream>>>(y_half, mean_f, rstd_f, x_f,
                                                              gamma_f, beta_f, rows, cols, eps);
    } else {
        layernorm_forward<1><<<blocks, kThreads, 0, stream>>>(y_half, mean_f, rstd_f, x_f,
                                                              gamma_f, beta_f, rows, cols, eps);
    }
    return cudaGetLastError();
}

}  // namespace flint
