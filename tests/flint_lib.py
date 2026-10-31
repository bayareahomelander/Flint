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
