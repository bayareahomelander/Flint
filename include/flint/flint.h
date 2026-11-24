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
    FLINT_ERROR_CUDA = 2,
    FLINT_ERROR_IO = 3,     /* the operating system failed a file call, or host memory ran out */
    FLINT_ERROR_FORMAT = 4  /* a file isn't valid: flint_last_error names the rule broken */
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

/* The embedding's backward pass, asynchronously on stream. dx is the gradient of the
 * forward pass's x, (sequences * length) x width FP32, and tokens its sequences *
 * length uint16 values, as flint_embedding_forward takes them. It computes
 *   dwte[v, :], the sum of dx[r, :] over the rows r whose token is v, for each v below
 *     vocab: vocab x width FP32. Every row is written, +0 where no row's token is v;
 *     with accumulate_wte nonzero, each row some token reaches gets its sum added
 *     with one FP32 add, and the others are unchanged.
 *   dwpe[t, :], the sum of dx[s * length + t, :] over the sequences s, for each t
 *     below length: the first length rows of dwpe, written, or with accumulate_wpe
 *     nonzero added to with one FP32 add each. Rows past length aren't touched.
 * The sums are compensated FP32 (TwoSum), in an order that depends only on the
 * inputs' shape and values: a token's rows in ascending order, and dwpe as
 * flint_bias_gradient sums the columns of dx viewed as sequences x (length * width).
 * There are no atomics, so the result is deterministic, and the same on every GPU. The
 * tokens are bucketed by a sort in the workspace first. An inf or NaN in dx reaches the
 * sums that include it. Rows with -0 sum to +0.
 *
 * A token at or above vocab reaches no row of dwte, so the call never writes out of
 * bounds, but its row of dx is lost to dwte: flint_embedding_forward reads out of
 * bounds for such a token, so the caller checks the tokens. Its row still reaches dwpe.
 *
 * width must be a multiple of 8 from 8 to 16,776,960, vocab from 1 to 65,536, and
 * length from 1 with length * width at most 16,776,960; sequences must be from 1, with
 * sequences * length at most 4,294,967,295. It needs the workspace that
 * flint_embedding_backward_workspace sizes. All pointers must be 16-byte aligned, and
 * dwte, dwpe and the workspace must not overlap each other or the inputs. The call
 * uses the workspace until the stream has run it, so calls on other streams need their
 * own. */
FLINT_API flint_status flint_embedding_backward(void* dwte, void* dwpe, const void* dx,
                                                const void* tokens, size_t sequences,
                                                size_t length, size_t vocab, size_t width,
                                                int accumulate_wte, int accumulate_wpe,
                                                void* workspace, size_t workspace_bytes,
                                                flint_stream stream);

/* The workspace flint_embedding_backward needs for sequences x length tokens, vocab and
 * width, in bytes, in *bytes: flint_bias_gradient_workspace's size for sequences x
 * (length * width), plus about 12 bytes per row and 4 per token of vocab. */
FLINT_API flint_status flint_embedding_backward_workspace(size_t sequences, size_t length,
                                                          size_t vocab, size_t width,
                                                          size_t* bytes);

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

/* A tensor for flint_gradient_norm: count FP32 values, contiguous, in device memory. */
typedef struct {
    const void* data;
    size_t count;
} flint_tensor;

/* The L2 norm of a list of FP32 tensors taken as one vector, and how many of their values
 * are not finite, asynchronously on stream. *norm (one FP64 value in device memory) receives
 * the square root of the sum of the squares of every value of the tensor_count tensors,
 * each square and the sum in FP64, and *nonfinite (one uint64) receives how many of those
 * values are inf or NaN, by isfinite. If that count is above 0, the norm isn't finite
 * either. Finite values always give a finite norm: an FP32 value squares exactly in FP64,
 * and the sum can't overflow it. Zero tensors, or only counts of 0, give +0 and 0. An
 * optimizer step reads the norm to clip and the count to check the loss scale for overflow.
 *
 * tensors points to tensor_count flint_tensor entries in device memory, which the kernels
 * read on the GPU, so the function can't check them: each data must be 4-byte aligned and
 * hold count FP32 values in device memory (16-byte aligned data reads faster; the bits don't
 * depend on alignment), and no tensor's values may overlap norm, nonfinite or the workspace.
 * Tensors may overlap each other: a value listed twice counts twice. An entry with a count
 * of 0 reads nothing. The sum is in a fixed order that depends only on the tensors' counts,
 * with no atomics, so the result is deterministic and the same bits on every GPU.
 *
 * norm, nonfinite, tensors and workspace must be non-null and 16-byte aligned, and norm,
 * nonfinite and the workspace must not overlap each other or the table of entries (the
 * function checks the table's bytes, not the tensors' values). workspace_bytes must be at
 * least flint_gradient_norm_workspace's value. The call uses the workspace until the stream
 * has run it, so calls on other streams need their own. */
FLINT_API flint_status flint_gradient_norm(void* norm, void* nonfinite, const void* tensors,
                                           size_t tensor_count, void* workspace,
                                           size_t workspace_bytes, flint_stream stream);

/* The workspace flint_gradient_norm needs, in bytes, in *bytes: a fixed 16 KiB (1,024 FP64
 * partial sums and 1,024 uint64 partial counts), whatever the tensors. */
FLINT_API flint_status flint_gradient_norm_workspace(size_t* bytes);

/* One master weight for flint_adamw: count FP32 values in each of four arrays in device
 * memory, the weight, its gradient and AdamW's two moments. decay nonzero applies the weight
 * decay. */
typedef struct {
    void* param;
    const void* grad;
    void* exp_avg;
    void* exp_avg_sq;
    size_t count;
    int decay;
} flint_adamw_tensor;                       /* 48 bytes, on every platform Flint builds for */

/* The state flint_adamw reads and updates, in device memory (24 bytes). */
typedef struct {
    float scale;                            /* the loss scale */
    int growth_tracker;                     /* clean steps in a row since the last growth or skip */
    long long step;                         /* AdamW's step count: the steps taken, not skipped */
    long long skipped;                      /* the steps skipped (Flint's own, for logging) */
} flint_adamw_state;

/* A step's hyperparameters, on the host. */
typedef struct {
    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    double max_norm;                        /* +inf turns clipping off */
    long long growth_interval;              /* GradScaler's default is 2,000 */
} flint_adamw_config;                       /* 56 bytes */

/* One optimizer step, asynchronously on stream, over the tensor_count entries of tensors, a
 * table of flint_adamw_tensor in device memory. It does what PyTorch 2.6 does with FP16
 * autocast, scaler.unscale_(optimizer), clip_grad_norm_(parameters, max_norm),
 * scaler.step(optimizer) and scaler.update(), with a fused AdamW, and never waits on the
 * host: it reads *norm (one FP64 value) and *nonfinite (one uint64), which flint_gradient_norm
 * left for the same, still scaled, gradients, and the state, which it updates.
 *
 * The step is skipped if *nonfinite is above 0 (GradScaler's found_inf), if the state's scale
 * isn't finite and positive, if 1 / scale overflows FP32 (GradScaler's inv_scale: a scale of 0
 * or of about 2^-128 or less), and if the unscaled norm, norm / scale, is 2^127 or more, past
 * which an unscaled value could overflow FP32. PyTorch 2.6 doesn't skip at those last two: its
 * unscale_ checks the gradients before it multiplies them, so an overflow there reaches its
 * step as inf and NaN. Where the unscaled norm is above about 1.8e19, PyTorch's FP32 clip norm
 * overflows, and with a finite max_norm its step goes ahead with all-zero gradients (with +inf,
 * NaN ones), while this one's FP64 norm clips as usual.
 * A skipped step changes no weight, no moment and not the state's step count; it halves the
 * scale, resets the growth tracker and adds 1 to skipped. A taken step multiplies every
 * gradient value by clip / scale, rounded to FP32, where the unscaled norm is norm / scale and
 * clip is max_norm / (unscaled norm + 1e-6), at most 1; no unscaled gradient is ever written.
 * It then updates the weights and moments as PyTorch's fused AdamW does, with its precisions: the
 * gradient times that multiplier in FP32, the decay, the moments and the denominator's eps in
 * FP64, rounded to FP32, the rest in FP32, with the bias corrections from the state's
 * step count + 1, which it then stores. Finally the loss scale update: the growth tracker
 * advances, and when it reaches growth_interval the scale doubles, unless that gives inf, and
 * the tracker resets. The growth and backoff factors are fixed at 2 and 0.5, GradScaler's
 * defaults. The decay applies where an entry's decay is nonzero and weight_decay isn't 0, even
 * if lr is 0, as PyTorch's fused kernel does. The learning rate is the caller's, per call.
 *
 * The result is deterministic and the same bits on every GPU: every value's arithmetic is
 * fixed, in explicitly rounded operations, and nothing is summed across values.
 *
 * config must be non-null with lr finite and at least 0, beta1 and beta2 from 0 up to but
 * not including 1, eps finite and above 0, weight_decay finite and at least 0, max_norm above
 * 0 (+inf is allowed, NaN is not) and growth_interval from 1 to 2,147,483,647. state
 * (24 bytes), workspace, norm (8), nonfinite (8) and tensors (tensor_count * 48) must be
 * non-null and 16-byte aligned, and state and workspace must not overlap each other or the
 * others (the function checks the table's bytes, not the arrays). workspace_bytes must be at
 * least flint_adamw_workspace's value; the call uses the workspace until the stream has run
 * it, so calls on other streams need their own.
 *
 * The table can't be checked, as flint_gradient_norm's can't: each entry's four pointers
 * must be 4-byte aligned and hold count FP32 values in device memory (all four 16-byte
 * aligned reads and writes faster; the bits don't depend on alignment), and no array may
 * overlap another array of any entry (the gradient may not alias the weight, either) or any
 * of the function's own buffers. An entry with a count of 0 touches nothing. A state whose
 * scale isn't finite and positive skips every step (halving a negative scale, and leaving an
 * infinite or NaN one as it is): flint_gpt2_reset_optimizer and the steps themselves leave only
 * a positive scale, or 0 after about 166 skips in a row from 2^16. */
FLINT_API flint_status flint_adamw(const void* tensors, size_t tensor_count, const void* norm,
                                   const void* nonfinite, void* state,
                                   const flint_adamw_config* config, void* workspace,
                                   size_t workspace_bytes, flint_stream stream);

/* The workspace flint_adamw needs, in bytes, in *bytes: a fixed 16, whatever the tensors. */
FLINT_API flint_status flint_adamw_workspace(size_t* bytes);

/* GPT-2's forward pass, from tokens to logits, each row's loss and the logits'
 * gradient, and its backward pass, from that gradient to every parameter's, through
 * the functions above. A model owns its weights and buffers. */
typedef struct flint_gpt2 flint_gpt2;

/* A model's shape. width must be a multiple of 8 up to 16,384, heads must split it
 * into multiples of 8, length a multiple of 8 up to 1,024, block_size from length
 * to 1,024, padded_vocab a multiple of 8 up to 65,536, and vocab from 1 to
 * padded_vocab. layers can be from 1 to 1,024. sequences must be positive, with
 * sequences * heads up to 65,535 and sequences * length up to 8,388,480.
 * separate_buffers nonzero gives each buffer its own allocation, so the sanitizer's
 * memcheck sees each buffer's bounds; the results are the same bits.
 *
 * backward nonzero lets the model run flint_gpt2_backward: it also holds a gradient
 * per master weight, the backward pass's buffers, and each layer's activations,
 * where a forward-only model reuses one set for every layer, so it takes far more
 * memory. Its forward pass gives the same bits. It narrows two limits:
 * sequences * length can be up to 65,536 (the weight gradients' GEMMs sum over the
 * rows), and length * width up to 16,776,960 (the embedding's backward pass). Zero
 * gives the forward-only model.
 *
 * optimizer nonzero needs backward nonzero, and lets the model run flint_gpt2_step: it adds
 * AdamW's two FP32 moments per master weight (8 bytes per parameter value, 0.93 GiB for
 * GPT-2 124M), the table of arrays that flint_adamw reads and the optimizer's state. Zero
 * gives the model of before, with the same bytes, layout and bits. */
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
    int backward;
    int optimizer;
} flint_gpt2_config;

/* A master weight: rows x cols FP32 in device memory, row-major and contiguous. A
 * bias or a LayerNorm's parameter has 1 row. name lasts as long as the model. */
typedef struct {
    const char* name;
    void* data;
    size_t rows;
    size_t cols;
} flint_gpt2_parameter_info;

/* The device memory flint_gpt2_create would allocate, in *bytes, so a caller can
 * hold it against a memory cap first. Without backward it doesn't touch the GPU.
 * With backward it includes flint_gemm_tn's workspace, whose size depends on the
 * current GPU's SM count, so it asks that GPU (flint_gemm_tn_workspace), and
 * returns FLINT_ERROR_CUDA if it can't. */
FLINT_API flint_status flint_gpt2_bytes(const flint_gpt2_config* config, size_t* bytes);

/* Creates a model, with one allocation of flint_gpt2_bytes' size (more, in smaller
 * pieces, with separate_buffers). Its weights start uninitialized: write them
 * through flint_gpt2_parameter, then call flint_gpt2_sync_weights. Returns
 * FLINT_ERROR_CUDA if the memory isn't there, with *model null and nothing kept.
 * With backward, flint_gemm_tn's workspace is sized for the current GPU's SM count,
 * so the backward pass must run on a GPU with the same count: on another,
 * flint_gemm_tn may need more workspace and fail. With backward, flint_gpt2_create also
 * writes the table of gradients that flint_gpt2_gradient_norm reads, and with optimizer
 * the table that flint_gpt2_step reads, and waits for the GPU until the copies have landed.
 * With optimizer, the moments and the optimizer's state start uninitialized, like the
 * weights: call flint_gpt2_reset_optimizer before the first step. */
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
 * error returns and the outputs are unspecified. The result is deterministic. With
 * backward, the pass also keeps what flint_gpt2_backward reads. */
FLINT_API flint_status flint_gpt2_forward(flint_gpt2* model, const void* tokens,
                                          const void* targets, float scale, flint_stream stream);

/* GPT-2's backward pass, asynchronously on stream: from the dlogits that the last
 * flint_gpt2_forward left, the gradient of every master weight, which
 * flint_gpt2_gradient gives. dlogits already carries the forward pass's scale, so the
 * gradients are those of scale times the sum of row_loss. The model must have been
 * created with backward nonzero, or the call fails with FLINT_ERROR_INVALID_ARGUMENT.
 * flint_gpt2_forward must have run on the same tokens, on the same stream, with
 * nothing in between that changes the model: no other forward pass, and no change to
 * the weights or flint_gpt2_sync_weights. The function can't check that, and
 * otherwise the gradients are unspecified. tokens is as flint_gpt2_forward takes it,
 * and must also not overlap the gradients.
 *
 * With accumulate zero, every gradient is written; with it nonzero, this pass's
 * gradient is added to each, with one FP32 add per value, as accumulating over
 * micro-batches needs, except for wte, which takes two adds in a fixed order:
 * (old + the output layer's part) + the embedding's part. An accumulated wte is
 * therefore not old + (the gradient a pass with accumulate zero would write) bit for
 * bit. wpe's rows from length to block_size get no gradient: with accumulate zero
 * they are set to +0, by a memset on the stream, not a kernel, and with it nonzero
 * they are left as they are.
 *
 * It runs the layers last to first, through flint_gemm_tn, flint_bias_gradient,
 * flint_layernorm_backward, flint_embedding_backward and the attention's and GELU's
 * backward functions above, with the precisions they document: parameter gradients
 * and the residual stream's gradient are FP32, and a linear layer's output gradient
 * is the residual stream's rounded to FP16, as autocast's is. The sums are in fixed
 * orders and there are no atomics, so the result is deterministic on a given GPU.
 * If a launch fails partway, the error returns and the gradients are unspecified. */
FLINT_API flint_status flint_gpt2_backward(flint_gpt2* model, const void* tokens, int accumulate,
                                           flint_stream stream);

/* The gradient of the index-th master weight: the name, rows and cols of
 * flint_gpt2_parameter's, with data pointing at the FP32 gradient in device memory.
 * Fails with FLINT_ERROR_INVALID_ARGUMENT for a model created without backward. The
 * gradients start uninitialized, until a flint_gpt2_backward with accumulate zero. */
FLINT_API flint_status flint_gpt2_gradient(const flint_gpt2* model, size_t index,
                                           flint_gpt2_parameter_info* info);

/* flint_gradient_norm over every master weight's gradient, in flint_gpt2_parameter's
 * order, asynchronously on stream, as the gradients stand: a caller runs it after the last
 * micro-batch's flint_gpt2_backward, on the same stream. The results go to the model's own
 * two buffers, which flint_gpt2_gradient_norm_outputs locates: the FP64 norm and the uint64
 * count of values that are not finite. Before the first call they are uninitialized. The
 * call uses the model's workspace, so it must not run at the same time as another of the
 * model's calls on a different stream. Fails with FLINT_ERROR_INVALID_ARGUMENT for a null
 * model or one created without backward. */
FLINT_API flint_status flint_gpt2_gradient_norm(flint_gpt2* model, flint_stream stream);

/* Where flint_gpt2_gradient_norm leaves its results, in device memory: the FP64 norm and
 * the uint64 count of values that are not finite. Either pointer may be null, if that
 * result isn't wanted. Fails with FLINT_ERROR_INVALID_ARGUMENT for a null model or one
 * created without backward. */
FLINT_API flint_status flint_gpt2_gradient_norm_outputs(const flint_gpt2* model,
                                                        const void** norm,
                                                        const void** nonfinite);

/* Starts the optimizer over, asynchronously on stream: every moment is set to +0, and the
 * state to {initial_scale, 0, 0, 0}, so the first step is AdamW's first. initial_scale must
 * be finite and positive (GradScaler's default is 2^16, 65,536). The model must have been
 * created with optimizer nonzero, or the call fails with FLINT_ERROR_INVALID_ARGUMENT, as
 * for a null model. These are a memset per moment buffer and a copy of the state from the
 * host, outside the public API's functions; the copy's source is safe to reuse once the
 * call returns. */
FLINT_API flint_status flint_gpt2_reset_optimizer(flint_gpt2* model, float initial_scale,
                                                  flint_stream stream);

/* One optimizer step over every master weight, asynchronously on stream:
 * flint_gpt2_gradient_norm, then flint_adamw over the model's weights, gradients and
 * moments with that norm and count and the model's state, then flint_gpt2_sync_weights, so
 * that the FP16 copies are current afterwards. Run it after the last micro-batch's
 * flint_gpt2_backward, on the same stream. config is checked as flint_adamw checks it,
 * before anything launches. Its lr is the caller's schedule, by iteration; bias correction
 * uses the state's own step count, which a skipped step doesn't advance. On a skipped step
 * the weights are unchanged (the sync rewrites the same FP16 values). The call uses the
 * model's workspace, so it must not run at the same time as another of the model's calls on
 * a different stream. Fails with FLINT_ERROR_INVALID_ARGUMENT for a null model or one
 * created without optimizer. */
FLINT_API flint_status flint_gpt2_step(flint_gpt2* model, const flint_adamw_config* config,
                                       flint_stream stream);

/* The moments of the index-th master weight: the name, rows and cols of
 * flint_gpt2_parameter's, with data pointing at the FP32 first moment (exp_avg) and the
 * second (exp_avg_sq) in device memory. Either pointer may be null, if that moment isn't
 * wanted. Fails with FLINT_ERROR_INVALID_ARGUMENT for a null model, one created without
 * optimizer, or an index out of range. */
FLINT_API flint_status flint_gpt2_moments(const flint_gpt2* model, size_t index,
                                          flint_gpt2_parameter_info* exp_avg,
                                          flint_gpt2_parameter_info* exp_avg_sq);

/* Where the model's flint_adamw_state is, in device memory, in *state: a caller may read it
 * (after the stream has run), or write it to resume a run. Fails with
 * FLINT_ERROR_INVALID_ARGUMENT for a null model, a null state, or a model created without
 * optimizer. */
FLINT_API flint_status flint_gpt2_optimizer_state(const flint_gpt2* model, void** state);

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

/* safetensors files, read and written on the host, with no GPU involved. A file is the
 * header's length n (8 bytes, little-endian), n bytes of UTF-8 JSON, and then the data. The JSON
 * is one object that maps each tensor's name to {"dtype": ..., "shape": [...], "data_offsets":
 * [begin, end]}, with the bytes [begin, end) of the data being the tensor's, and may also have
 * the key "__metadata__", whose value is an object of strings (or null). Flint accepts exactly
 * what Python's safetensors 0.8.0 accepts (RFC 8259's JSON, Rust's rule for UTF-8), except that
 * it also refuses a name or metadata key given twice and a NUL in a name, a metadata key or a
 * metadata value, since the strings it hands out are NUL-terminated.
 *
 * Flint is 64-bit only: shapes and offsets are 64-bit. Paths are UTF-8 (on Windows a path that
 * isn't is FLINT_ERROR_INVALID_ARGUMENT) and may not be empty. A null pointer argument is
 * FLINT_ERROR_INVALID_ARGUMENT, except where a function says it may be null. A reader or writer
 * is used by one thread at a time. */

/* The 22 dtypes, with the bits per element: "F4" (4); "F6_E2M3" and "F6_E3M2" (6); "BOOL", "U8",
 * "I8", "F8_E5M2", "F8_E4M3", "F8_E8M0", "F8_E4M3FNUZ" and "F8_E5M2FNUZ" (8); "I16", "U16", "F16"
 * and "BF16" (16); "I32", "U32" and "F32" (32); and "C64", "F64", "I64" and "U64" (64). A
 * tensor's bytes are its element count times the bits, over 8, and the bits must make a whole
 * number of bytes. */

/* A tensor in a safetensors file. The reader fills every field. The writer reads name, dtype,
 * rank and shape, and fills bits, offset and bytes. */
typedef struct {
    const char* name;      /* UTF-8, NUL-terminated */
    const char* dtype;     /* one of the 22 names above, such as "F32" */
    size_t rank;
    const size_t* shape;   /* rank values; may be null when rank is 0 */
    size_t bits;           /* bits per element: 4, 6, 8, 16, 32 or 64 */
    size_t offset;         /* the tensor's first byte, from the data's start */
    size_t bytes;          /* end - begin */
} flint_safetensors_tensor;

typedef struct flint_safetensors_reader flint_safetensors_reader;
typedef struct flint_safetensors_writer flint_safetensors_writer;

/* Opens a safetensors file and reads its whole header, checking every rule before it returns: the
 * file holds at least 8 bytes; n is at most 100,000,000 and fits in the file; the header is valid
 * UTF-8 and one JSON object; "__metadata__" is null or an object of strings; each other key is a
 * tensor's name, mapping to an object with a "dtype" (a string, one of the 22 names), a "shape"
 * (an array of unsigned integers) and "data_offsets" (an array of two), each once, and any other
 * field is skipped, though its value must still be valid JSON; each tensor's end - begin is its
 * dtype and shape's bytes, with the multiplications checked for overflow; and sorted by begin,
 * the tensors cover the data exactly, with no hole, overlap or byte left over (a tensor of size 0
 * fits wherever the next byte is). The header's order is kept, and the data's order doesn't
 * matter. A broken rule is FLINT_ERROR_FORMAT, with a message that names the path, the rule and
 * the tensor or key; an operating system failure (a missing file, no permission, a short read)
 * or a failed host allocation is FLINT_ERROR_IO, with the call, the path and the system's
 * message. On failure *reader is null and nothing stays open. The file stays open until
 * flint_safetensors_close. */
FLINT_API flint_status flint_safetensors_open(const char* path, flint_safetensors_reader** reader);

/* Closes a reader and frees it, and everything it handed out. A null reader does nothing. */
FLINT_API void flint_safetensors_close(flint_safetensors_reader* reader);

/* How many tensors the file has, in *count. */
FLINT_API flint_status flint_safetensors_tensor_count(const flint_safetensors_reader* reader,
                                                      size_t* count);

/* The index-th tensor, in the header's order, in *info, with every field filled. The strings and
 * the shape array last as long as the reader. An index at or above the count is
 * FLINT_ERROR_INVALID_ARGUMENT. */
FLINT_API flint_status flint_safetensors_tensor_info(const flint_safetensors_reader* reader,
                                                     size_t index, flint_safetensors_tensor* info);

/* The index of the tensor named name (an exact comparison of bytes), in *index. A name that no
 * tensor has is FLINT_ERROR_INVALID_ARGUMENT, and *index is left as it was. It takes constant
 * time on average. */
FLINT_API flint_status flint_safetensors_find(const flint_safetensors_reader* reader,
                                              const char* name, size_t* index);

/* How many pairs "__metadata__" has, in *count: 0 when the file has none, or a null one. */
FLINT_API flint_status flint_safetensors_metadata_count(const flint_safetensors_reader* reader,
                                                        size_t* count);

/* The index-th metadata pair, in the header's order: its key and value, which last as long as
 * the reader. An index at or above the count is FLINT_ERROR_INVALID_ARGUMENT. */
FLINT_API flint_status flint_safetensors_metadata(const flint_safetensors_reader* reader,
                                                  size_t index, const char** key,
                                                  const char** value);

/* The value of the metadata key key, in *value, which lasts as long as the reader. A key that
 * isn't there is FLINT_ERROR_INVALID_ARGUMENT, and *value is left as it was. */
FLINT_API flint_status flint_safetensors_metadata_find(const flint_safetensors_reader* reader,
                                                       const char* key, const char** value);

/* Copies the bytes [offset, offset + bytes) of the index-th tensor to dst in host memory, which
 * may be null when bytes is 0. The range must lie within the tensor's own bytes, or the call is
 * FLINT_ERROR_INVALID_ARGUMENT and reads nothing. It reads at a position, sharing no file
 * pointer, in chunks of at most 1 GiB. If the file ends first (it shrank after it opened) the
 * call is FLINT_ERROR_IO. A tensor larger than host memory can be read in pieces. */
FLINT_API flint_status flint_safetensors_read(flint_safetensors_reader* reader, size_t index,
                                              size_t offset, size_t bytes, void* dst);

/* Starts writing a safetensors file, to path, given every tensor's name, dtype and shape up
 * front. The tensors are laid out in the order given, each starting where the last ended, with
 * no padding. The header is the compact JSON that serde_json writes: "__metadata__" first, when
 * metadata_count is above 0 (the pairs keys[i], values[i], in order), and then the tensors in
 * order, padded with spaces to a multiple of 8 bytes. The call writes the 8-byte length and the
 * header to path + ".tmp", which it creates or empties, and on success fills each entry's bits,
 * offset and bytes and returns the writer. tensors may be null when tensor_count is 0, and keys
 * and values when metadata_count is 0. path itself isn't touched until
 * flint_safetensors_commit.
 *
 * These are FLINT_ERROR_INVALID_ARGUMENT, found before any file is made: a null name, dtype, key
 * or value, or a null shape with a rank above 0; a name, key or value that isn't valid UTF-8; a
 * name that is "__metadata__"; two tensors with one name, or two pairs with one key; a dtype that
 * isn't one of the 22 names; a shape whose size overflows 64 bits (each multiplication, left to
 * right, counts) or isn't a whole number of bytes; a total size that overflows; and a padded
 * header over 100,000,000 bytes. A file that can't be made is FLINT_ERROR_IO, as is a failed host
 * allocation. On failure *writer is null, no entry is changed and no temporary file is left. */
FLINT_API flint_status flint_safetensors_create(const char* path,
                                                flint_safetensors_tensor* tensors,
                                                size_t tensor_count, const char* const* keys,
                                                const char* const* values, size_t metadata_count,
                                                flint_safetensors_writer** writer);

/* Appends the next bytes bytes of the data to the temporary file: the tensors' bytes in the
 * order declared, where one call may span tensors or carry part of one (the writer only counts
 * them). src may be null when bytes is 0. Each call goes straight to the operating system, in
 * chunks of at most 1 GiB, with no buffer of Flint's. A write that would pass the declared total
 * is FLINT_ERROR_INVALID_ARGUMENT and writes nothing, and so is a call after the commit. If the
 * system fails the write, the call is FLINT_ERROR_IO, and the writer fails: every later write or
 * commit is FLINT_ERROR_IO ("an earlier write failed"). */
FLINT_API flint_status flint_safetensors_write(flint_safetensors_writer* writer, const void* src,
                                               size_t bytes);

/* Finishes the file: every declared byte must have been written, or the call is
 * FLINT_ERROR_INVALID_ARGUMENT and the writer stays usable. Then it flushes the temporary file to
 * disk and moves it over path, replacing any file there, and makes the move durable: on Windows
 * FlushFileBuffers and MoveFileExW with replace and write-through; elsewhere fsync, rename, and
 * an fsync of path's folder. On success path holds the whole file. If the system fails, the
 * call is FLINT_ERROR_IO, the writer fails, and the temporary file is deleted, best effort. On
 * Windows the move fails (FLINT_ERROR_IO) while any handle has path open, a reader's included,
 * so close the readers of path before committing over it; on Linux the rename replaces it, and
 * an open reader keeps reading the old file. On Linux, an I/O error after the rename (syncing
 * the folder) leaves the new file at path, and the message says so. */
FLINT_API flint_status flint_safetensors_commit(flint_safetensors_writer* writer);

/* Frees a writer. If it wasn't committed, it also closes and deletes the temporary file, best
 * effort, ignoring errors; path itself is never touched. A null writer does nothing. */
FLINT_API void flint_safetensors_destroy(flint_safetensors_writer* writer);

#ifdef __cplusplus
}
#endif

#endif /* FLINT_FLINT_H */
