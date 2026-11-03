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

/* Runs y[j] = scale x[j] w[j]^T for batches matrices at a time in one launch,
 * asynchronously on stream, as attention's heads need. Each batch's y[j]
 * (m x n), x[j] (m x k) and w[j] (n x k) directly follow the last batch's, all
 * row-major, contiguous FP16 in device memory. Products accumulate in FP32, the
 * sums are multiplied by scale in FP32, and each output rounds to FP16 once, so
 * a sum past FP16's range that scale brings back in stays finite. With a scale
 * of 1, each batch's result is flint_gemm_nt's bit for bit, except that a NaN
 * may carry other bits. The result is deterministic.
 *
 * batches can be up to 65,535, and m up to 1,024; scale must be finite.
 * Everything else is as for flint_gemm_nt, with y, x and w covering all batches.
 * A batches, m or n of 0 does nothing. */
FLINT_API flint_status flint_gemm_nt_batched(void* y, const void* x, const void* w, size_t batches,
                                             size_t m, size_t n, size_t k, float scale,
                                             flint_stream stream);

/* Runs y[j] = x[j] w[j] for batches matrices at a time, with x[j] m x k and w[j]
 * k x n. Each batch's result is flint_gemm_nn's bit for bit. The limits on batches
 * and m are flint_gemm_nt_batched's; everything else is as for flint_gemm_nn, with
 * y, x and w covering all batches. */
FLINT_API flint_status flint_gemm_nn_batched(void* y, const void* x, const void* w, size_t batches,
                                             size_t m, size_t n, size_t k, flint_stream stream);

/* Runs y[j] = x[j]^T w[j] for batches matrices at a time, with x[j] k x m and w[j]
 * k x n. Unlike flint_gemm_tn, y is FP16: each output's FP32 sum rounds once. k
 * never splits, so there's no workspace. Each batch's result is flint_gemm_tn's
 * rounded to FP16 bit for bit, wherever that doesn't split k, except that a NaN may
 * carry other bits. The limits on batches and m are flint_gemm_nt_batched's;
 * everything else is as for flint_gemm_tn, with y, x and w covering all batches. */
FLINT_API flint_status flint_gemm_tn_batched(void* y, const void* x, const void* w, size_t batches,
                                             size_t m, size_t n, size_t k, flint_stream stream);

/* LayerNorm's forward pass, asynchronously on stream. For each row of x, a
 * rows x cols FP32 matrix, it computes the row's mean and its biased variance,
 * the mean of squared deviations from the mean, then
 * y = (x - mean) * rstd * gamma + beta with rstd = 1 / sqrt(variance + eps).
 * Everything is computed in FP32, and each value of y (rows x cols) rounds to
 * FP16 once, to nearest, so overflow gives inf. The row's mean and rstd are
 * stored in mean and rstd (rows FP32 values each) for the backward pass.
 * gamma and beta hold cols FP32 values. Matrices are row-major and contiguous,
 * in device memory. Each row's result depends only on that row, gamma and
 * beta, and is deterministic. A row holding an inf or NaN gives NaN mean,
 * rstd and y, and so does a row whose sum overflows FP32. A row whose sum of
 * squared deviations overflows (a deviation from the mean of about 1.8e19 is
 * enough) gives NaN rstd and y. PyTorch's LayerNorm can stay finite on both.
 *
 * cols must be from 1 to 2^24, rows can be up to 17,179,869,176, and eps must
 * be finite and non-negative. All pointers must be 16-byte aligned, and y,
 * mean and rstd must not overlap each other or the inputs. A rows of 0 does
 * nothing. */
FLINT_API flint_status flint_layernorm_forward(void* y, void* mean, void* rstd, const void* x,
                                               const void* gamma, const void* beta, size_t rows,
                                               size_t cols, float eps, flint_stream stream);

/* The embedding's forward pass, asynchronously on stream. tokens holds
 * sequences * length uint16 values, one per row of x, sequence after sequence;
 * wte is the word table (vocab x width FP32) and wpe the position table (at
 * least length x width FP32; only its first length rows are read). For row
 * r = s * length + t, x[r, :] = wte[tokens[r], :] + wpe[t, :], one FP32 add
 * each, so positions restart at 0 in every sequence. x is
 * (sequences * length) x width FP32. Matrices are row-major and contiguous, in
 * device memory. The result is deterministic.
 *
 * Every token must be less than vocab: the function doesn't check, since that
 * would need a wait on the GPU, and a larger token reads out of bounds. The
 * caller checks.
 *
 * width must be a positive multiple of 8, up to 2^24, and vocab from 1 to
 * 65,536. The number of rows, sequences * length, can be up to
 * 17,179,869,176. All pointers must be 16-byte aligned. x must not overlap
 * tokens, wte or wpe; those may share memory. A sequences or length of 0 does
 * nothing. */
FLINT_API flint_status flint_embedding_forward(void* x, const void* tokens, const void* wte,
                                               const void* wpe, size_t sequences, size_t length,
                                               size_t vocab, size_t width, flint_stream stream);

/* Converts count FP32 values at x to FP16 at y, asynchronously on stream. Each
 * value rounds to nearest, ties to even, so overflow gives inf (nothing
 * saturates) and a NaN stays a NaN. Any count works, not only multiples of 4,
 * up to SIZE_MAX / 4. Both pointers must be 16-byte aligned, and y must not
 * overlap x. A count of 0 does nothing. */
FLINT_API flint_status flint_fp32_to_fp16(void* y, const void* x, size_t count,
                                          flint_stream stream);

/* Splits a row buffer into attention heads, asynchronously on stream. x holds
 * rows of row_width FP16 values, batches * length of them ("rows layout"), and
 * y is batches x heads x length x head_width FP16 ("heads layout"), both
 * row-major and contiguous, in device memory. For the columns
 * [first, first + heads * head_width) of each row,
 * y[b, h, t, c] = x[b * length + t, first + h * head_width + c]. The values
 * move as bits, never through a float, so NaN payloads and -0 survive exactly.
 * GPT-2's fused qkv rows have row_width 3 * 768, and Q, K and V are the splits
 * with first = 0, 768 and 1536 (heads 12, head_width 64).
 *
 * head_width, row_width and first must be multiples of 8, with head_width and
 * row_width positive and heads positive. The columns must lie within the row:
 * first + heads * head_width <= row_width. row_width can be up to 2^24, and the
 * number of rows, batches * length, up to 17,179,869,176. Both pointers must
 * be 16-byte aligned, and y must not overlap x (the whole of either layout's
 * buffer counts). A batches or length of 0 does nothing. */
FLINT_API flint_status flint_heads_split(void* y, const void* x, size_t batches, size_t length,
                                         size_t heads, size_t head_width, size_t row_width,
                                         size_t first, flint_stream stream);

/* The reverse of flint_heads_split, asynchronously on stream: x is the heads
 * layout and y the rows layout, and y[b * length + t, first + h * head_width + c]
 * = x[b, h, t, c]. It writes only the columns [first, first + heads *
 * head_width) of each row of y and leaves the others untouched, so three
 * merges (Q, K and V) can fill one row buffer. Everything else is as for
 * flint_heads_split. */
FLINT_API flint_status flint_heads_merge(void* y, const void* x, size_t batches, size_t length,
                                         size_t heads, size_t head_width, size_t row_width,
                                         size_t first, flint_stream stream);

#ifdef __cplusplus
}
#endif

#endif /* FLINT_FLINT_H */
