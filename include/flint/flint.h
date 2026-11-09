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

/* A linear layer's bias gradient, asynchronously on stream: dbias[j] is the sum
 * over the rows of dy[r, j], for dy rows x cols FP16 (row-major, contiguous) and
 * dbias cols FP32, all in device memory. With accumulate nonzero it adds the sums
 * into dbias instead, with one FP32 add each, as accumulating over micro-batches
 * needs. The sums are compensated FP32 (TwoSum), in an order that depends only on
 * rows and cols, so the result is deterministic, and the same on every GPU. An inf
 * or NaN in dy reaches its column's sum.
 *
 * cols must be a multiple of 8 up to 16,776,960, and rows from 1 to
 * 549,755,813,632. It needs the workspace that flint_bias_gradient_workspace
 * sizes. All pointers must be 16-byte aligned, and dbias, the workspace and dy
 * must not overlap. The call uses the workspace until the stream has run it, so
 * calls on other streams need their own. A cols of 0 does nothing. */
FLINT_API flint_status flint_bias_gradient(void* dbias, const void* dy, size_t rows, size_t cols,
                                           int accumulate, void* workspace,
                                           size_t workspace_bytes, flint_stream stream);

/* The workspace flint_bias_gradient needs for rows x cols, in bytes, in *bytes:
 * 8 cols bytes per 256 rows, or part of them. */
FLINT_API flint_status flint_bias_gradient_workspace(size_t rows, size_t cols, size_t* bytes);

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

/* LayerNorm's backward pass, asynchronously on stream. dy (rows x cols FP16) is the
 * gradient of the forward pass's y; x (rows x cols FP32), mean and rstd (rows FP32
 * values each) are the forward pass's input and statistics, and gamma holds cols FP32
 * values. With xhat = (x - mean) * rstd and g = dy * gamma, it computes
 *   dx = rstd * (g - mean(g) - xhat * mean(g * xhat)),
 * with both means over the row, rows x cols FP32; dgamma, the column sums of
 * dy * xhat; and dbeta, the column sums of dy, cols FP32 values each. With
 * accumulate_dx nonzero it adds dx into dx's old values instead, with one FP32 add
 * each, as the residual stream's gradient needs; with accumulate_params nonzero it adds
 * dgamma and dbeta into theirs, as accumulating over micro-batches needs. Everything is
 * computed in FP32, with compensated (TwoSum) column sums for dgamma and dbeta, in an
 * order that depends only on rows and cols, so the result is deterministic, and the
 * same on every GPU. dbeta is exactly what flint_bias_gradient gives for dy. An inf or
 * NaN in an input reaches the outputs that depend on it.
 *
 * cols must be a multiple of 8 from 8 to 16,776,960, narrower than the forward pass
 * allows, and rows from 1 to 17,179,869,176. It needs the workspace that
 * flint_layernorm_backward_workspace sizes. All pointers must be 16-byte aligned, and
 * dx, dgamma, dbeta and the workspace must not overlap each other or the inputs. The
 * call uses the workspace until the stream has run it, so calls on other streams need
 * their own. */
FLINT_API flint_status flint_layernorm_backward(void* dx, void* dgamma, void* dbeta,
                                                const void* dy, const void* x, const void* mean,
                                                const void* rstd, const void* gamma, size_t rows,
                                                size_t cols, int accumulate_dx,
                                                int accumulate_params, void* workspace,
                                                size_t workspace_bytes, flint_stream stream);

/* The workspace flint_layernorm_backward needs for rows x cols, in bytes, in *bytes:
 * flint_bias_gradient_workspace's size, so one workspace can serve both. */
FLINT_API flint_status flint_layernorm_backward_workspace(size_t rows, size_t cols,
                                                          size_t* bytes);

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

/* Attention's causal softmax, asynchronously on stream. s and p are batches x
 * length x length FP16, row-major and contiguous, in device memory, each batch
 * directly after the last. Row t of a batch sees only its columns 0 to t:
 * p[b, t, c] = exp(s[b, t, c] - max) / sum for c <= t, where max is the largest
 * of those scores and sum adds exp(s[b, t, c] - max) over them. Everything is
 * computed in FP32, and each output rounds to FP16 once, to nearest, so small
 * probabilities can round to subnormals or 0. p[b, t, c] is +0 for c > t, and
 * s's values there are ignored, even an inf or NaN (they must still be
 * initialized). Each row's result depends only on t and its visible scores, and
 * is deterministic.
 *
 * A visible NaN or +inf gives NaN in all the row's visible outputs, and so does
 * a row whose visible scores are all -inf. A -inf among finite scores gives 0.
 * A NaN's own bits aren't promised.
 *
 * length must be a multiple of 8 up to 1,024, and batches can be up to 65,535.
 * Both pointers must be 16-byte aligned, and p must not overlap s. A batches or
 * length of 0 does nothing. */
FLINT_API flint_status flint_causal_softmax_forward(void* p, const void* s, size_t batches,
                                                    size_t length, flint_stream stream);

/* Attention's causal softmax, backward, asynchronously on stream. p (the forward
 * pass's output), dp (its gradient) and ds are batches x length x length FP16, laid out
 * as for flint_causal_softmax_forward. For row t of a batch, with D the sum of
 * p[b, t, c] * dp[b, t, c] over its visible columns c <= t,
 * ds[b, t, c] = scale * p[b, t, c] * (dp[b, t, c] - D) for c <= t, and +0 for c > t,
 * whatever p and dp hold there (they must still be initialized). scale carries the
 * scores' own scale (1/8 for GPT-2's heads of 64), so the gradient GEMMs that follow
 * need none. Everything is computed in FP32: each product p * dp is exact, D is a
 * compensated (TwoSum) sum, and each output subtracts D's rounded sum and then its
 * error, so where dp is close to D the difference keeps its bits. Each output rounds
 * to FP16 once, to nearest, so a value past FP16's range gives inf. The
 * order of the additions depends only on t, so each row's result is deterministic and
 * the same on every GPU. A visible inf or NaN reaches the row's outputs as it does in
 * FP32 arithmetic. A NaN's own bits aren't promised.
 *
 * length must be a multiple of 8 up to 1,024, batches can be up to 65,535, and scale
 * must be finite. All pointers must be 16-byte aligned. ds may be dp itself, computing
 * in place; otherwise it must not overlap dp, and it must never overlap p. A batches or
 * length of 0 does nothing. */
FLINT_API flint_status flint_causal_softmax_backward(void* ds, const void* p, const void* dp,
                                                     size_t batches, size_t length, float scale,
                                                     flint_stream stream);

/* The output layer's softmax and cross-entropy in one kernel, asynchronously on
 * stream. logits and dlogits are rows x cols FP16, row-major and contiguous,
 * targets holds rows uint16 tokens and row_loss rows FP32, all in device memory.
 * Only a row's first vocab columns count. With max the largest of those logits,
 * sum adding exp(logits[r, c] - max) over them, and p[r, c] = exp(logits[r, c] -
 * max) / sum:
 *
 *   row_loss[r] = log(sum) - (logits[r, targets[r]] - max)
 *   dlogits[r, c] = scale * (p[r, c] - 1), for c = targets[r]
 *   dlogits[r, c] = scale * p[r, c], for the other c below vocab
 *
 * Everything is computed in FP32, and each dlogits value rounds to FP16 once, to
 * nearest, so small gradients can round to subnormals or 0 and large ones to an
 * inf. dlogits[r, c] is +0 for c at or above vocab, and logits' values there are
 * ignored, even an inf or NaN (they must still be initialized). Each row's result
 * depends only on its own logits, its target, vocab and scale, and is
 * deterministic.
 *
 * row_loss holds one row's cross-entropy each, for the caller to add up. dlogits
 * is the gradient of scale times their sum, so for a mean loss over a step's
 * tokens with a loss scale, scale is the loss scale over the tokens.
 *
 * A NaN or +inf among a row's first vocab logits gives NaN in its row_loss and in
 * those columns of dlogits, and so does a row whose vocab logits are all -inf. A
 * -inf among finite logits gets a p of 0, and if it is the target's, a row_loss
 * of +inf. A NaN's own bits aren't promised.
 *
 * Targets must be below vocab; checking would need a wait on the GPU, so the
 * function doesn't. A target at or above vocab reads and writes nothing outside
 * the buffers, but its row's row_loss may stay unwritten and its dlogits are
 * unspecified.
 *
 * cols must be a multiple of 8 from 8 to 65,536, vocab from 1 to cols, rows up to
 * 17,179,869,176, and scale finite. All four pointers must be 16-byte aligned,
 * and neither output may overlap the other or an input. A rows of 0 does
 * nothing. */
FLINT_API flint_status flint_softmax_cross_entropy(void* dlogits, void* row_loss,
                                                   const void* logits, const void* targets,
                                                   float scale, size_t rows, size_t vocab,
                                                   size_t cols, flint_stream stream);

/* GPT-2's forward pass, from tokens to logits, each row's loss and the logits'
 * gradient, through the functions above. A model owns its weights and buffers. */
typedef struct flint_gpt2 flint_gpt2;

/* A model's shape. width must be a multiple of 8 up to 16,384, heads must split it
 * into multiples of 8, length a multiple of 8 up to 1,024, block_size from length
 * to 1,024, padded_vocab a multiple of 8 up to 65,536, and vocab from 1 to
 * padded_vocab. layers can be from 1 to 1,024. sequences must be positive, with
 * sequences * heads up to 65,535 and sequences * length up to 8,388,480.
 * separate_buffers nonzero gives each buffer its own allocation, so the sanitizer's
 * memcheck sees each buffer's bounds; the results are the same bits. */
typedef struct {
    size_t layers;
    size_t heads;
    size_t width;
    size_t block_size;
    size_t vocab;
    size_t padded_vocab;
    size_t sequences;
    size_t length;
    int separate_buffers;
} flint_gpt2_config;

/* A master weight: rows x cols FP32 in device memory, row-major and contiguous. A
 * bias or a LayerNorm's parameter has 1 row. name lasts as long as the model. */
typedef struct {
    const char* name;
    void* data;
    size_t rows;
    size_t cols;
} flint_gpt2_parameter_info;

/* The device memory flint_gpt2_create would allocate, in *bytes. It doesn't touch
 * the GPU, so a caller can hold it against a memory cap first. */
FLINT_API flint_status flint_gpt2_bytes(const flint_gpt2_config* config, size_t* bytes);

/* Creates a model, with one allocation of flint_gpt2_bytes' size (more, in smaller
 * pieces, with separate_buffers). Its weights start uninitialized: write them
 * through flint_gpt2_parameter, then call flint_gpt2_sync_weights. Returns
 * FLINT_ERROR_CUDA if the memory isn't there, with *model null and nothing kept. */
FLINT_API flint_status flint_gpt2_create(const flint_gpt2_config* config, flint_gpt2** model);

/* Frees a model and its memory. Anything still using them on a stream must have
 * finished. A null model does nothing. */
FLINT_API void flint_gpt2_destroy(flint_gpt2* model);

/* How many master weights a model has, or 0 for a null model. */
FLINT_API size_t flint_gpt2_parameter_count(const flint_gpt2* model);

/* The index-th master weight, in the order of PyTorch's named_parameters and with
 * its names ("wte.weight", "blocks.0.attn.c_attn.bias", "ln_f.weight"). A linear
 * layer's weight is out x in, as PyTorch's is. wte has padded_vocab rows. */
FLINT_API flint_status flint_gpt2_parameter(const flint_gpt2* model, size_t index,
                                            flint_gpt2_parameter_info* info);

/* Rounds the master weights to their FP16 compute copies, asynchronously on stream:
 * wte and the linear layers' weights. Biases and LayerNorm's parameters stay FP32
 * and need no copy. Call it after changing the weights, before a forward pass. */
FLINT_API flint_status flint_gpt2_sync_weights(flint_gpt2* model, flint_stream stream);

/* Runs the model, asynchronously on stream. tokens and targets each hold sequences
 * * length uint16 values in device memory, sequence after sequence, and must not
 * overlap the model's outputs. Every token must be below padded_vocab and every
 * target below vocab; as with flint_embedding_forward and
 * flint_softmax_cross_entropy, the function doesn't check. scale must be finite,
 * and goes to flint_softmax_cross_entropy.
 *
 * The buffers are reused, so a pass overwrites the last one's outputs, and the
 * model's calls must all use one stream at a time. If a launch fails partway, the
 * error returns and the outputs are unspecified. The result is deterministic. */
FLINT_API flint_status flint_gpt2_forward(flint_gpt2* model, const void* tokens,
                                          const void* targets, float scale, flint_stream stream);

/* Where a forward pass leaves its outputs, in device memory, as
 * flint_softmax_cross_entropy describes them: logits and dlogits (sequences * length)
 * x padded_vocab FP16, row_loss sequences * length FP32. Any of the three pointers
 * may be null, if that output isn't wanted. */
FLINT_API flint_status flint_gpt2_outputs(const flint_gpt2* model, const void** logits,
                                          const void** dlogits, const void** row_loss);

/* Waits for stream, copies row_loss to the host, and stores its mean in *loss, added
 * up in FP64. It reads whatever row_loss holds: before a first pass that's
 * uninitialized, and after a failed one it's the last pass's values or a mix, so a
 * caller must check flint_gpt2_forward's status. */
FLINT_API flint_status flint_gpt2_mean_loss(flint_gpt2* model, flint_stream stream, double* loss);

#ifdef __cplusplus
}
#endif

#endif /* FLINT_FLINT_H */
