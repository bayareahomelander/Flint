/* Flint's public C API.
 *
 * The tests call it from Python through ctypes and the benchmarks link it
 * directly, so it stays plain C and needs no CUDA headers. */
#ifndef FLINT_FLINT_H
#define FLINT_FLINT_H

#include <stddef.h>

#if defined(_WIN32)
#  if defined(FLINT_BUILD)
#    define FLINT_API __declspec(dllexport)
#  else
#    define FLINT_API __declspec(dllimport)
#  endif
#else
/* The library builds with hidden visibility, so on Linux too, only functions
 * marked FLINT_API are exported. */
#  define FLINT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Every function that can fail returns one of these codes. */
typedef int flint_status;
enum {
    FLINT_OK = 0,
    FLINT_ERROR_INVALID_ARGUMENT = 1,
    FLINT_ERROR_CUDA = 2
};

/* The library version, such as "0.1.0". */
FLINT_API const char* flint_version(void);

/* A short description of a status code, or "unknown status". */
FLINT_API const char* flint_status_string(flint_status status);

/* The message for the latest failed call on this thread, or "" if none has
 * failed. Successful calls leave it unchanged. */
FLINT_API const char* flint_last_error(void);

/* The same type as cudaStream_t, declared here so this header needs no CUDA
 * headers. A null stream is the legacy default stream. */
typedef struct CUstream_st* flint_stream;

/* Copies bytes from src to dst, both in device memory, asynchronously on
 * stream. The ranges must not overlap. Any size and alignment works, and
 * 16-byte aligned pointers take the fast path. */
FLINT_API flint_status flint_copy(void* dst, const void* src, size_t bytes, flint_stream stream);

/* Multiplies y = x w^T, the forward layout of a linear layer, asynchronously
 * on stream. y is m x n, x is m x k, and w is n x k: row-major, contiguous
 * FP16 matrices in device memory. Products accumulate in FP32, and each output
 * rounds to FP16 once, to nearest, so overflow gives inf. The result is
 * deterministic.
 *
 * k must be a positive multiple of 8, and all three pointers 16-byte aligned.
 * m and n can be up to 8,388,480, and k up to 65,536: the tensor cores' FP32
 * accumulation truncates, so the error grows with k. y must not overlap x or
 * w. An m or n of 0 does nothing. */
FLINT_API flint_status flint_gemm_nt(void* y, const void* x, const void* w, size_t m, size_t n,
                                     size_t k, flint_stream stream);

/* Multiplies y = x w^T + bias, the forward layout with a linear layer's bias,
 * asynchronously on stream. bias holds n FP32 values in device memory, one per
 * column of y, and each is added to its column's FP32 sums before they round
 * to FP16, once. Everything else is as for flint_gemm_nt; bias must also be
 * 16-byte aligned and must not overlap y. */
FLINT_API flint_status flint_gemm_nt_bias(void* y, const void* x, const void* w, const void* bias,
                                          size_t m, size_t n, size_t k, flint_stream stream);

/* Computes s = x w^T + bias as flint_gemm_nt_bias does, then stores s rounded
 * to FP16 in pre (m x n), which GELU's backward pass needs, and GELU(s) rounded
 * to FP16 in y, asynchronously on stream. GELU is GPT-2's tanh approximation,
 * 0.5 s (1 + tanh(sqrt(2/pi) (s + 0.044715 s^3))), computed in FP32 on the
 * unrounded s, so each output rounds once. Everything else is as for
 * flint_gemm_nt_bias; pre must also be 16-byte aligned and must not overlap y,
 * x, w or bias. */
FLINT_API flint_status flint_gemm_nt_bias_gelu(void* y, void* pre, const void* x, const void* w,
                                               const void* bias, size_t m, size_t n, size_t k,
                                               flint_stream stream);

/* Adds a linear layer's output to the residual stream, asynchronously on
 * stream: y = residual + (x w^T + bias), with y and residual m x n FP32
 * matrices in device memory, row-major and contiguous. The FP32 sums of x w^T
 * get the bias, as in flint_gemm_nt_bias, then the residual, one FP32 add
 * each, and nothing rounds to FP16. n must be even. Everything else is as for
 * flint_gemm_nt_bias; residual must also be 16-byte aligned and must not
 * overlap y. */
FLINT_API flint_status flint_gemm_nt_bias_residual(void* y, const void* x, const void* w,
                                                   const void* bias, const void* residual,
                                                   size_t m, size_t n, size_t k,
                                                   flint_stream stream);

/* Multiplies y = x w, the layout of a linear layer's input gradient
 * (dX = dY W), asynchronously on stream. y is m x n, x is m x k, and w is
 * k x n; everything else is as for flint_gemm_nt, except that n must also be
 * a multiple of 8. */
FLINT_API flint_status flint_gemm_nn(void* y, const void* x, const void* w, size_t m, size_t n,
                                     size_t k, flint_stream stream);

/* GELU's backward pass, dpre = dh GELU'(pre) with dh = x w, asynchronously on
 * stream: computes the FP32 sums of x w as flint_gemm_nn does, multiplies each
 * by the derivative of GPT-2's tanh GELU at pre, and rounds the products to
 * FP16 y, once. pre (m x n FP16) holds GELU's inputs, as
 * flint_gemm_nt_bias_gelu stores them, and the derivative is computed in FP32
 * from them. An infinite pre takes GELU's limits, 0 at -inf and 1 at +inf,
 * and a NaN gives NaN. Everything else is as for flint_gemm_nn; pre must also
 * be 16-byte aligned and must not overlap y. */
FLINT_API flint_status flint_gemm_nn_dgelu(void* y, const void* x, const void* w,
                                           const void* pre, size_t m, size_t n, size_t k,
                                           flint_stream stream);

/* Multiplies y = x^T w, the layout of a linear layer's weight gradient
 * (dW = dY^T X), asynchronously on stream. y is m x n FP32; x is k x m and w
 * is k x n FP16, all row-major and contiguous. With accumulate nonzero it adds
 * x^T w into y instead, with ordinary FP32 adds, as accumulating over
 * micro-batches needs.
 *
 * m and n must be multiples of 8, up to 8,388,480, and k from 1 to 65,536.
 * All pointers must be 16-byte aligned, and y must not overlap x or w. An m or
 * n of 0 does nothing.
 *
 * An output with few tiles splits k across blocks, and then needs the
 * workspace that flint_gemm_tn_workspace sizes, which must not overlap y, x or
 * w; when it needs none, workspace may be null. The call uses the workspace
 * until the stream has run it, so calls on other streams need their own. The
 * splits' sums add up in a fixed order, so the result is deterministic on a
 * given GPU. */
FLINT_API flint_status flint_gemm_tn(void* y, const void* x, const void* w, size_t m, size_t n,
                                     size_t k, int accumulate, void* workspace,
                                     size_t workspace_bytes, flint_stream stream);

/* The workspace flint_gemm_tn needs for m x n x k on the current GPU, in bytes,
 * in *bytes: 0 when it needs none. It depends on the GPU's SM count. */
FLINT_API flint_status flint_gemm_tn_workspace(size_t m, size_t n, size_t k, size_t* bytes);

#ifdef __cplusplus
}
#endif

#endif /* FLINT_FLINT_H */
