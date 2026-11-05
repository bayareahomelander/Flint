"""ctypes bindings for Flint's C API, used by the test suite.

The library comes from the FLINT_LIB environment variable if it is set (a path
to flint.dll or libflint.so), and otherwise from the release preset's build.

Pointer arguments take a torch tensor (its data pointer), an int address, or
None for null. Stream arguments take a torch.cuda.Stream, an int handle, or
None for torch's current stream.
"""

import ctypes
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "include" / "flint" / "flint.h"
LIBRARY_NAME = "flint.dll" if sys.platform == "win32" else "libflint.so"

# Status codes, mirroring flint.h.
OK = 0
ERROR_INVALID_ARGUMENT = 1
ERROR_CUDA = 2


class Gpt2Config(ctypes.Structure):
    """flint_gpt2_config."""

    _fields_ = [
        ("layers", ctypes.c_size_t),
        ("heads", ctypes.c_size_t),
        ("width", ctypes.c_size_t),
        ("block_size", ctypes.c_size_t),
        ("vocab", ctypes.c_size_t),
        ("padded_vocab", ctypes.c_size_t),
        ("sequences", ctypes.c_size_t),
        ("length", ctypes.c_size_t),
        ("separate_buffers", ctypes.c_int),
    ]


class Gpt2ParameterInfo(ctypes.Structure):
    """flint_gpt2_parameter_info."""

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("rows", ctypes.c_size_t),
        ("cols", ctypes.c_size_t),
    ]


# Every function flint.h declares, as name: (restype, argtypes).
SIGNATURES = {
    "flint_version": (ctypes.c_char_p, []),
    "flint_status_string": (ctypes.c_char_p, [ctypes.c_int]),
    "flint_last_error": (ctypes.c_char_p, []),
    "flint_copy": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
    ),
    "flint_gemm_nt": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_nt_bias": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_nt_bias_gelu": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_nt_bias_residual": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_nn": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_nn_dgelu": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_tn": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_tn_workspace": (
        ctypes.c_int,
        [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)],
    ),
    "flint_gemm_nt_batched": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_nn_batched": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gemm_tn_batched": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_layernorm_forward": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_void_p,
        ],
    ),
    "flint_embedding_forward": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_fp32_to_fp16": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p],
    ),
    "flint_heads_split": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_heads_merge": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_causal_softmax_forward": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_softmax_cross_entropy": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gpt2_bytes": (
        ctypes.c_int,
        [ctypes.POINTER(Gpt2Config), ctypes.POINTER(ctypes.c_size_t)],
    ),
    "flint_gpt2_create": (
        ctypes.c_int,
        [ctypes.POINTER(Gpt2Config), ctypes.POINTER(ctypes.c_void_p)],
    ),
    "flint_gpt2_destroy": (None, [ctypes.c_void_p]),
    "flint_gpt2_parameter_count": (ctypes.c_size_t, [ctypes.c_void_p]),
    "flint_gpt2_parameter": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(Gpt2ParameterInfo)],
    ),
    "flint_gpt2_sync_weights": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
    "flint_gpt2_forward": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p],
    ),
    "flint_gpt2_outputs": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    "flint_gpt2_mean_loss": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)],
    ),
}


class FlintError(RuntimeError):
    """A Flint call returned a status other than OK."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def pointer(value):
    if value is None or isinstance(value, int):
        return value
    return value.data_ptr()


def stream_handle(stream):
    if stream is None:
        return torch.cuda.current_stream().cuda_stream
    if isinstance(stream, int):
        return stream
    return stream.cuda_stream


def library_path() -> Path:
    override = os.environ.get("FLINT_LIB")
    if override:
        return Path(override)
    return ROOT / "build" / "release" / "bin" / LIBRARY_NAME


class Flint:
    """The loaded library, with its C signatures declared."""

    def __init__(self, path=None):
        self.path = Path(path) if path else library_path()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"{self.path} not found; build it with "
                "`cmake --preset release && cmake --build --preset release`"
            )
        self.lib = ctypes.CDLL(str(self.path))
        for name, (restype, argtypes) in SIGNATURES.items():
            function = getattr(self.lib, name)
            function.restype = restype
            function.argtypes = argtypes

    def version(self) -> str:
        return self.lib.flint_version().decode()

    def status_string(self, status: int) -> str:
        return self.lib.flint_status_string(status).decode()

    def last_error(self) -> str:
        return self.lib.flint_last_error().decode()

    def check(self, status: int) -> None:
        """Raises FlintError, with this thread's last error, unless status is OK."""
        if status != OK:
            message = f"{self.status_string(status)} (status {status})"
            detail = self.last_error()
            raise FlintError(status, f"{message}: {detail}" if detail else message)

    def copy(self, dst, src, nbytes: int, stream=None) -> None:
        self.check(
            self.lib.flint_copy(pointer(dst), pointer(src), nbytes, stream_handle(stream))
        )

    def gemm_nt(self, y, x, w, m: int, n: int, k: int, stream=None) -> None:
        """y = x w^T for row-major FP16 matrices: y is m x n, x m x k, w n x k."""
        self.check(
            self.lib.flint_gemm_nt(
                pointer(y), pointer(x), pointer(w), m, n, k, stream_handle(stream)
            )
        )

    def gemm_nt_bias(self, y, x, w, bias, m: int, n: int, k: int, stream=None) -> None:
        """y = x w^T + bias for row-major FP16 matrices (y is m x n, x m x k,
        w n x k) and n FP32 bias values, one per column of y, added before y
        rounds to FP16."""
        self.check(
            self.lib.flint_gemm_nt_bias(
                pointer(y), pointer(x), pointer(w), pointer(bias), m, n, k, stream_handle(stream)
            )
        )

    def gemm_nt_bias_gelu(self, y, pre, x, w, bias, m: int, n: int, k: int, stream=None) -> None:
        """s = x w^T + bias as gemm_nt_bias computes it, then pre = s and
        y = GELU(s) (the tanh approximation, on the unrounded s), both m x n
        FP16 and rounded once."""
        self.check(
            self.lib.flint_gemm_nt_bias_gelu(
                pointer(y), pointer(pre), pointer(x), pointer(w), pointer(bias), m, n, k,
                stream_handle(stream),
            )
        )

    def gemm_nt_bias_residual(
        self, y, x, w, bias, residual, m: int, n: int, k: int, stream=None
    ) -> None:
        """y = residual + (x w^T + bias): x (m x k) and w (n x k) are FP16, bias
        holds n FP32 values, and residual and y are m x n FP32. The FP32 sums
        get the bias as gemm_nt_bias adds it, then the residual, with no
        rounding to FP16. n must be even."""
        self.check(
            self.lib.flint_gemm_nt_bias_residual(
                pointer(y), pointer(x), pointer(w), pointer(bias), pointer(residual), m, n, k,
                stream_handle(stream),
            )
        )

    def gemm_nn(self, y, x, w, m: int, n: int, k: int, stream=None) -> None:
        """y = x w for row-major FP16 matrices: y is m x n, x m x k, w k x n."""
        self.check(
            self.lib.flint_gemm_nn(
                pointer(y), pointer(x), pointer(w), m, n, k, stream_handle(stream)
            )
        )

    def gemm_nn_dgelu(self, y, x, w, pre, m: int, n: int, k: int, stream=None) -> None:
        """GELU's backward pass: y = (x w) GELU'(pre), with the FP32 sums of
        x w as gemm_nn computes them multiplied by the derivative of the tanh
        GELU at pre (m x n FP16, the forward's stored input) before y (m x n
        FP16) rounds, once."""
        self.check(
            self.lib.flint_gemm_nn_dgelu(
                pointer(y), pointer(x), pointer(w), pointer(pre), m, n, k, stream_handle(stream)
            )
        )

    def gemm_tn_workspace(self, m: int, n: int, k: int) -> int:
        """The bytes of workspace gemm_tn needs for m x n x k on this GPU."""
        size = ctypes.c_size_t()
        self.check(self.lib.flint_gemm_tn_workspace(m, n, k, ctypes.byref(size)))
        return size.value

    def gemm_tn(
        self, y, x, w, m: int, n: int, k: int, accumulate=False, workspace=None,
        workspace_bytes=None, stream=None,
    ) -> None:
        """y = x^T w, or y += x^T w with accumulate, for FP32 y (m x n) and FP16
        x (k x m) and w (k x n). workspace_bytes defaults to a workspace
        tensor's size, or 0."""
        if workspace_bytes is None:
            workspace_bytes = 0 if workspace is None or isinstance(workspace, int) else (
                workspace.numel() * workspace.element_size()
            )
        self.check(
            self.lib.flint_gemm_tn(
                pointer(y), pointer(x), pointer(w), m, n, k, int(bool(accumulate)),
                pointer(workspace), workspace_bytes, stream_handle(stream),
            )
        )

    def gemm_nt_batched(
        self, y, x, w, batches: int, m: int, n: int, k: int, scale=1.0, stream=None
    ) -> None:
        """y[j] = scale x[j] w[j]^T for batches FP16 matrices in a row: y[j] is
        m x n, x[j] m x k, w[j] n x k. The FP32 sums get the scale before they
        round."""
        self.check(
            self.lib.flint_gemm_nt_batched(
                pointer(y), pointer(x), pointer(w), batches, m, n, k, scale, stream_handle(stream)
            )
        )

    def gemm_nn_batched(self, y, x, w, batches: int, m: int, n: int, k: int, stream=None) -> None:
        """y[j] = x[j] w[j] for batches FP16 matrices in a row: y[j] is m x n,
        x[j] m x k, w[j] k x n."""
        self.check(
            self.lib.flint_gemm_nn_batched(
                pointer(y), pointer(x), pointer(w), batches, m, n, k, stream_handle(stream)
            )
        )

    def gemm_tn_batched(self, y, x, w, batches: int, m: int, n: int, k: int, stream=None) -> None:
        """y[j] = x[j]^T w[j] for batches matrices in a row: FP16 y[j] (m x n),
        rounded once from FP32 sums, and FP16 x[j] (k x m) and w[j] (k x n)."""
        self.check(
            self.lib.flint_gemm_tn_batched(
                pointer(y), pointer(x), pointer(w), batches, m, n, k, stream_handle(stream)
            )
        )

    def layernorm_forward(
        self, y, mean, rstd, x, gamma, beta, rows: int, cols: int, eps=1e-5, stream=None
    ) -> None:
        """LayerNorm's forward pass over each row of x (rows x cols FP32):
        y = (x - mean) * rstd * gamma + beta, rounded to FP16 (rows x cols),
        with rstd = 1 / sqrt(biased variance + eps). Stores each row's FP32
        mean and rstd (rows values each); gamma and beta hold cols FP32
        values."""
        self.check(
            self.lib.flint_layernorm_forward(
                pointer(y), pointer(mean), pointer(rstd), pointer(x), pointer(gamma),
                pointer(beta), rows, cols, eps, stream_handle(stream),
            )
        )

    def embedding_forward(
        self, x, tokens, wte, wpe, sequences: int, length: int, vocab: int, width: int, stream=None
    ) -> None:
        """x[s * length + t] = wte[tokens[s * length + t]] + wpe[t], in FP32:
        tokens are uint16 (sequences * length), wte is vocab x width, wpe at
        least length x width and x (sequences * length) x width. Tokens must
        be below vocab; the function doesn't check."""
        self.check(
            self.lib.flint_embedding_forward(
                pointer(x), pointer(tokens), pointer(wte), pointer(wpe), sequences, length, vocab,
                width, stream_handle(stream),
            )
        )

    def fp32_to_fp16(self, y, x, count: int, stream=None) -> None:
        """Converts count FP32 values at x to FP16 at y, to nearest with ties
        to even (overflow gives inf)."""
        self.check(self.lib.flint_fp32_to_fp16(pointer(y), pointer(x), count, stream_handle(stream)))

    def heads_split(
        self, y, x, batches: int, length: int, heads: int, head_width: int, row_width: int,
        first: int = 0, stream=None,
    ) -> None:
        """Splits columns [first, first + heads * head_width) of the FP16 rows
        x (batches * length rows of row_width) into y (batches x heads x length
        x head_width): y[b, h, t, c] = x[b * length + t, first + h * head_width + c]."""
        self.check(
            self.lib.flint_heads_split(
                pointer(y), pointer(x), batches, length, heads, head_width, row_width, first,
                stream_handle(stream),
            )
        )

    def heads_merge(
        self, y, x, batches: int, length: int, heads: int, head_width: int, row_width: int,
        first: int = 0, stream=None,
    ) -> None:
        """The reverse of heads_split: x is the heads layout, and only the
        columns [first, first + heads * head_width) of each row of y change."""
        self.check(
            self.lib.flint_heads_merge(
                pointer(y), pointer(x), batches, length, heads, head_width, row_width, first,
                stream_handle(stream),
            )
        )

    def causal_softmax_forward(self, p, s, batches: int, length: int, stream=None) -> None:
        """P from S, both FP16 batches x length x length: row t of a batch is
        the softmax of its columns 0 to t in FP32, rounded to FP16 once, with
        +0 in the rest. s's columns past t are ignored."""
        self.check(
            self.lib.flint_causal_softmax_forward(
                pointer(p), pointer(s), batches, length, stream_handle(stream)
            )
        )

    def softmax_cross_entropy(
        self, dlogits, row_loss, logits, targets, scale: float, rows: int, vocab: int, cols: int,
        stream=None,
    ) -> None:
        """Each row's cross-entropy (row_loss, FP32) and logit gradient (dlogits,
        FP16 rows x cols) from FP16 logits and uint16 targets, over its first
        vocab columns; dlogits is +0 in the rest, and scale * (p - one-hot) there.
        Targets must be below vocab; the function doesn't check."""
        self.check(
            self.lib.flint_softmax_cross_entropy(
                pointer(dlogits), pointer(row_loss), pointer(logits), pointer(targets), scale, rows,
                vocab, cols, stream_handle(stream),
            )
        )


class Gpt2:
    """A flint_gpt2 model, freed by close() or on leaving a with block. Pointers to its
    weights and outputs are device addresses, as ints, which flint_copy accepts."""

    def __init__(self, flint: Flint, config: Gpt2Config):
        self.flint = flint
        self.config = config
        handle = ctypes.c_void_p()
        flint.check(flint.lib.flint_gpt2_create(ctypes.byref(config), ctypes.byref(handle)))
        self.handle = handle.value

    def close(self) -> None:
        if self.handle is not None:
            self.flint.lib.flint_gpt2_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def parameters(self) -> list:
        """(name, address, rows, cols) of each master weight, in the library's order."""
        out = []
        for i in range(self.flint.lib.flint_gpt2_parameter_count(self.handle)):
            info = Gpt2ParameterInfo()
            self.flint.check(self.flint.lib.flint_gpt2_parameter(self.handle, i, ctypes.byref(info)))
            out.append((info.name.decode(), info.data, info.rows, info.cols))
        return out

    def load(self, state, stream=None) -> None:
        """Copies state[name] (any FP32 device tensor of the right size, from a
        state_dict for example) to each master weight, then syncs the compute copies."""
        for name, address, rows, cols in self.parameters():
            tensor = state[name].contiguous()
            if tensor.dtype != torch.float32 or tensor.numel() != rows * cols:
                raise ValueError(f"{name}: {tensor.dtype} {tuple(tensor.shape)}, expected FP32 {rows} x {cols}")
            self.flint.copy(address, tensor, rows * cols * 4, stream)
        self.sync_weights(stream)

    def sync_weights(self, stream=None) -> None:
        self.flint.check(self.flint.lib.flint_gpt2_sync_weights(self.handle, stream_handle(stream)))

    def forward(self, tokens, targets, scale: float, stream=None) -> None:
        self.flint.check(
            self.flint.lib.flint_gpt2_forward(
                self.handle, pointer(tokens), pointer(targets), scale, stream_handle(stream)
            )
        )

    def outputs(self) -> tuple:
        """The device addresses of logits, dlogits and row_loss."""
        logits, dlogits, row_loss = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        self.flint.check(
            self.flint.lib.flint_gpt2_outputs(
                self.handle, ctypes.byref(logits), ctypes.byref(dlogits), ctypes.byref(row_loss)
            )
        )
        return logits.value, dlogits.value, row_loss.value

    def mean_loss(self, stream=None) -> float:
        loss = ctypes.c_double()
        self.flint.check(
            self.flint.lib.flint_gpt2_mean_loss(self.handle, stream_handle(stream), ctypes.byref(loss))
        )
        return loss.value

    def read(self, address: int, shape, dtype, first_element: int = 0, stream=None):
        """A copy, as a device tensor of this shape and dtype, of the device memory at
        address, starting first_element elements in."""
        out = torch.empty(shape, dtype=dtype, device="cuda")
        size = torch.empty((), dtype=dtype).element_size()
        self.flint.copy(out, address + first_element * size, out.numel() * size, stream)
        torch.cuda.synchronize()
        return out
