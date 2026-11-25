"""ctypes bindings for Flint's C API, used by the test suite.

The library comes from the FLINT_LIB environment variable if it is set (a path
to flint.dll or libflint.so), and otherwise from the release preset's build.

Pointer arguments take a torch tensor (its data pointer), an int address, or
None for null. Stream arguments take a torch.cuda.Stream, an int handle, or
None for torch's current stream.
"""

import ctypes
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "include" / "flint" / "flint.h"
LIBRARY_NAME = "flint.dll" if sys.platform == "win32" else "libflint.so"

# Status codes, mirroring flint.h.
OK = 0
ERROR_INVALID_ARGUMENT = 1
ERROR_CUDA = 2
ERROR_IO = 3
ERROR_FORMAT = 4


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
        ("backward", ctypes.c_int),
        ("optimizer", ctypes.c_int),
    ]


class AdamwConfig(ctypes.Structure):
    """flint_adamw_config."""

    _fields_ = [
        ("lr", ctypes.c_double),
        ("beta1", ctypes.c_double),
        ("beta2", ctypes.c_double),
        ("eps", ctypes.c_double),
        ("weight_decay", ctypes.c_double),
        ("max_norm", ctypes.c_double),
        ("growth_interval", ctypes.c_longlong),
    ]


class Gpt2ParameterInfo(ctypes.Structure):
    """flint_gpt2_parameter_info."""

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("rows", ctypes.c_size_t),
        ("cols", ctypes.c_size_t),
    ]


class SafetensorsTensor(ctypes.Structure):
    """flint_safetensors_tensor."""

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dtype", ctypes.c_char_p),
        ("rank", ctypes.c_size_t),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("bits", ctypes.c_size_t),
        ("offset", ctypes.c_size_t),
        ("bytes", ctypes.c_size_t),
    ]


class LoaderConfig(ctypes.Structure):
    """flint_loader_config."""

    _fields_ = [
        ("length", ctypes.c_size_t),
        ("max_tokens", ctypes.c_size_t),
        ("vocab", ctypes.c_size_t),
        ("seed", ctypes.c_ulonglong),
        ("shuffle", ctypes.c_int),
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
    "flint_bias_gradient": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_bias_gradient_workspace": (
        ctypes.c_int,
        [ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)],
    ),
    "flint_gradient_norm": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_gradient_norm_workspace": (ctypes.c_int, [ctypes.POINTER(ctypes.c_size_t)]),
    "flint_adamw": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(AdamwConfig),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_adamw_workspace": (ctypes.c_int, [ctypes.POINTER(ctypes.c_size_t)]),
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
    "flint_layernorm_backward": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_layernorm_backward_workspace": (
        ctypes.c_int,
        [ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)],
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
    "flint_embedding_backward": (
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
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
    ),
    "flint_embedding_backward_workspace": (
        ctypes.c_int,
        [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
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
    "flint_causal_softmax_backward": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
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
    "flint_gpt2_backward": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p],
    ),
    "flint_gpt2_gradient": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(Gpt2ParameterInfo)],
    ),
    "flint_gpt2_gradient_norm": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
    "flint_gpt2_gradient_norm_outputs": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)],
    ),
    "flint_gpt2_reset_optimizer": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p],
    ),
    "flint_gpt2_step": (ctypes.c_int, [ctypes.c_void_p, ctypes.POINTER(AdamwConfig), ctypes.c_void_p]),
    "flint_gpt2_moments": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(Gpt2ParameterInfo),
            ctypes.POINTER(Gpt2ParameterInfo),
        ],
    ),
    "flint_gpt2_optimizer_state": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
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
    "flint_safetensors_open": (
        ctypes.c_int,
        [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
    ),
    "flint_safetensors_close": (None, [ctypes.c_void_p]),
    "flint_safetensors_tensor_count": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
    ),
    "flint_safetensors_tensor_info": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(SafetensorsTensor)],
    ),
    "flint_safetensors_find": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)],
    ),
    "flint_safetensors_metadata_count": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
    ),
    "flint_safetensors_metadata": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ],
    ),
    "flint_safetensors_metadata_find": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)],
    ),
    "flint_safetensors_read": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p],
    ),
    "flint_safetensors_create": (
        ctypes.c_int,
        [
            ctypes.c_char_p,
            ctypes.POINTER(SafetensorsTensor),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    "flint_safetensors_write": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t],
    ),
    "flint_safetensors_commit": (ctypes.c_int, [ctypes.c_void_p]),
    "flint_safetensors_destroy": (None, [ctypes.c_void_p]),
    "flint_loader_open": (
        ctypes.c_int,
        [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_size_t,
            ctypes.POINTER(LoaderConfig),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    "flint_loader_close": (None, [ctypes.c_void_p]),
    "flint_loader_epoch_sequences": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
    ),
    "flint_loader_locate": (
        ctypes.c_int,
        [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ],
    ),
    "flint_loader_read": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p],
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


def tensor_table(entries):
    """The table flint_gradient_norm reads, in device memory: a CUDA int64 tensor of shape
    (n, 2) holding [address, count] per row, which is the flint_tensor layout (16 bytes each).
    Each entry is a contiguous FP32 CUDA tensor, or an (address, count) pair. With no entries it
    returns a (1, 2) tensor of zeros, so the pointer is valid; pass a tensor_count of 0."""
    rows = []
    for entry in entries:
        if isinstance(entry, torch.Tensor):
            if entry.dtype != torch.float32 or not entry.is_cuda or not entry.is_contiguous():
                raise ValueError("a tensor entry must be a contiguous FP32 CUDA tensor")
            rows.append([entry.data_ptr(), entry.numel()])
        else:
            address, count = entry
            rows.append([int(address), int(count)])
    if not rows:
        return torch.zeros((1, 2), dtype=torch.int64, device="cuda")
    return torch.tensor(rows, dtype=torch.int64, device="cuda")


def adamw_config(
    lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1, max_norm=1.0, growth_interval=2000
) -> AdamwConfig:
    """A flint_adamw_config; the defaults are PLAN.md's M4 settings."""
    return AdamwConfig(lr, beta1, beta2, eps, weight_decay, max_norm, growth_interval)


def adamw_table(entries):
    """The table flint_adamw reads, in device memory: a CUDA int64 tensor of shape (n, 6)
    holding the 48-byte flint_adamw_tensor layout per row: four addresses, the count, and the
    decay int in the low 4 bytes of the last column (0 in the high 4). Each entry is
    (param, grad, exp_avg, exp_avg_sq, decay), four contiguous FP32 CUDA tensors of one size each
    and a bool, or (param_address, grad_address, exp_avg_address, exp_avg_sq_address, count,
    decay), ints. With no entries it returns a (1, 6) tensor of zeros, so the pointer is valid;
    pass a tensor_count of 0."""
    rows = []
    for entry in entries:
        if isinstance(entry[0], torch.Tensor):
            *arrays, decay = entry
            for array in arrays:
                if array.dtype != torch.float32 or not array.is_cuda or not array.is_contiguous():
                    raise ValueError("an array must be a contiguous FP32 CUDA tensor")
            count = arrays[0].numel()
            if any(array.numel() != count for array in arrays):
                raise ValueError("an entry's four arrays must have the same size")
            rows.append([array.data_ptr() for array in arrays] + [count, int(bool(decay))])
        else:
            *addresses, count, decay = entry
            rows.append([int(a) for a in addresses] + [int(count), int(bool(decay))])
    if not rows:
        return torch.zeros((1, 6), dtype=torch.int64, device="cuda")
    return torch.tensor(rows, dtype=torch.int64, device="cuda")


def adamw_state(scale=2.0**16, growth_tracker=0, step=0, skipped=0):
    """A flint_adamw_state as a CUDA uint8 tensor of 24 bytes."""
    packed = struct.pack("<fiqq", scale, growth_tracker, step, skipped)
    return torch.frombuffer(bytearray(packed), dtype=torch.uint8).to("cuda")


class _DeviceBytes:
    """An address in device memory, as an object torch can wrap without copying."""

    def __init__(self, address, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (address, False), "version": 2,
        }  # fmt: skip


def read_adamw_state(address_or_tensor) -> dict:
    """The four fields of a flint_adamw_state in device memory, given as a CUDA tensor or an
    address. It synchronizes first."""
    torch.cuda.synchronize()
    if isinstance(address_or_tensor, torch.Tensor):
        raw = address_or_tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    else:
        raw = torch.as_tensor(_DeviceBytes(address_or_tensor, 24), device="cuda").cpu().numpy().tobytes()
    scale, growth_tracker, step, skipped = struct.unpack("<fiqq", raw[:24])
    return {"scale": scale, "growth_tracker": growth_tracker, "step": step, "skipped": skipped}


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

    def bias_gradient_workspace(self, rows: int, cols: int) -> int:
        """The bytes of workspace bias_gradient needs for rows x cols."""
        size = ctypes.c_size_t()
        self.check(self.lib.flint_bias_gradient_workspace(rows, cols, ctypes.byref(size)))
        return size.value

    def bias_gradient(
        self, dbias, dy, rows: int, cols: int, accumulate=False, workspace=None,
        workspace_bytes=None, stream=None,
    ) -> None:
        """dbias = the column sums of dy, or dbias += them with accumulate, for FP32
        dbias (cols) and FP16 dy (rows x cols). workspace_bytes defaults to a
        workspace tensor's size, or 0."""
        if workspace_bytes is None:
            workspace_bytes = 0 if workspace is None or isinstance(workspace, int) else (
                workspace.numel() * workspace.element_size()
            )
        self.check(
            self.lib.flint_bias_gradient(
                pointer(dbias), pointer(dy), rows, cols, int(bool(accumulate)), pointer(workspace),
                workspace_bytes, stream_handle(stream),
            )
        )

    def gradient_norm_workspace(self) -> int:
        """The bytes of workspace gradient_norm needs."""
        size = ctypes.c_size_t()
        self.check(self.lib.flint_gradient_norm_workspace(ctypes.byref(size)))
        return size.value

    def gradient_norm(
        self, norm, nonfinite, tensors, tensor_count: int, workspace=None, workspace_bytes=None,
        stream=None,
    ) -> None:
        """norm (one FP64) = the L2 norm of the tensor_count FP32 tensors that the table
        tensors (see tensor_table) lists, and nonfinite (one uint64) = how many of their values
        are inf or NaN. workspace_bytes defaults to a workspace tensor's size, or 0."""
        if workspace_bytes is None:
            workspace_bytes = 0 if workspace is None or isinstance(workspace, int) else (
                workspace.numel() * workspace.element_size()
            )
        self.check(
            self.lib.flint_gradient_norm(
                pointer(norm), pointer(nonfinite), pointer(tensors), tensor_count,
                pointer(workspace), workspace_bytes, stream_handle(stream),
            )
        )

    def adamw_workspace(self) -> int:
        """The bytes of workspace adamw needs."""
        size = ctypes.c_size_t()
        self.check(self.lib.flint_adamw_workspace(ctypes.byref(size)))
        return size.value

    def adamw(
        self, tensors, tensor_count: int, norm, nonfinite, state, config, workspace=None,
        workspace_bytes=None, stream=None,
    ) -> None:
        """One optimizer step over the tensor_count entries that the table tensors (see
        adamw_table) lists, reading norm (one FP64) and nonfinite (one uint64), which
        gradient_norm left, and updating state (see adamw_state). config is an AdamwConfig, or
        None for a null pointer. workspace_bytes defaults to a workspace tensor's size, or 0."""
        if workspace_bytes is None:
            workspace_bytes = 0 if workspace is None or isinstance(workspace, int) else (
                workspace.numel() * workspace.element_size()
            )
        self.check(
            self.lib.flint_adamw(
                pointer(tensors), tensor_count, pointer(norm), pointer(nonfinite), pointer(state),
                None if config is None else ctypes.byref(config), pointer(workspace),
                workspace_bytes, stream_handle(stream),
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

    def layernorm_backward_workspace(self, rows: int, cols: int) -> int:
        """The bytes of workspace layernorm_backward needs for rows x cols."""
        size = ctypes.c_size_t()
        self.check(self.lib.flint_layernorm_backward_workspace(rows, cols, ctypes.byref(size)))
        return size.value

    def layernorm_backward(
        self, dx, dgamma, dbeta, dy, x, mean, rstd, gamma, rows: int, cols: int,
        accumulate_dx=False, accumulate_params=False, workspace=None, workspace_bytes=None,
        stream=None,
    ) -> None:
        """LayerNorm's backward pass: dx (rows x cols FP32), dgamma and dbeta (cols FP32
        each) from dy (rows x cols FP16), x (rows x cols FP32), the forward pass's mean
        and rstd (rows FP32 each) and gamma (cols FP32). accumulate_dx adds dx into dx,
        accumulate_params adds dgamma and dbeta into theirs. workspace_bytes defaults
        to a workspace tensor's size, or 0."""
        if workspace_bytes is None:
            workspace_bytes = 0 if workspace is None or isinstance(workspace, int) else (
                workspace.numel() * workspace.element_size()
            )
        self.check(
            self.lib.flint_layernorm_backward(
                pointer(dx), pointer(dgamma), pointer(dbeta), pointer(dy), pointer(x),
                pointer(mean), pointer(rstd), pointer(gamma), rows, cols,
                int(bool(accumulate_dx)), int(bool(accumulate_params)), pointer(workspace),
                workspace_bytes, stream_handle(stream),
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

    def embedding_backward_workspace(
        self, sequences: int, length: int, vocab: int, width: int
    ) -> int:
        """The bytes of workspace embedding_backward needs."""
        size = ctypes.c_size_t()
        self.check(
            self.lib.flint_embedding_backward_workspace(
                sequences, length, vocab, width, ctypes.byref(size)
            )
        )
        return size.value

    def embedding_backward(
        self, dwte, dwpe, dx, tokens, sequences: int, length: int, vocab: int, width: int,
        accumulate_wte=False, accumulate_wpe=False, workspace=None, workspace_bytes=None,
        stream=None,
    ) -> None:
        """The embedding's backward pass: dwte[v] (vocab x width FP32) = the sum of dx[r]
        over the rows whose token is v, and dwpe[t] (the first length rows, FP32) = the
        sum of dx[s * length + t] over the sequences, from dx ((sequences * length) x
        width FP32) and uint16 tokens. accumulate_wte and accumulate_wpe add into the old
        values instead. workspace_bytes defaults to a workspace tensor's size, or 0."""
        if workspace_bytes is None:
            workspace_bytes = 0 if workspace is None or isinstance(workspace, int) else (
                workspace.numel() * workspace.element_size()
            )
        self.check(
            self.lib.flint_embedding_backward(
                pointer(dwte), pointer(dwpe), pointer(dx), pointer(tokens), sequences, length,
                vocab, width, int(bool(accumulate_wte)), int(bool(accumulate_wpe)),
                pointer(workspace), workspace_bytes, stream_handle(stream),
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

    def causal_softmax_backward(self, ds, p, dp, batches: int, length: int, scale=0.125,
                                stream=None) -> None:
        """dS from P and dP, all FP16 batches x length x length: row t of a batch gets
        scale * p * (dp - D) in its columns 0 to t, with D the FP32 sum of p * dp over
        them, rounded to FP16 once, and +0 in the rest. ds may be dp itself."""
        self.check(
            self.lib.flint_causal_softmax_backward(
                pointer(ds), pointer(p), pointer(dp), batches, length, scale, stream_handle(stream)
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
        return self._infos(self.flint.lib.flint_gpt2_parameter)

    def gradients(self) -> list:
        """(name, address, rows, cols) of each master weight's FP32 gradient, in the same
        order; the model must have been created with backward."""
        return self._infos(self.flint.lib.flint_gpt2_gradient)

    def _infos(self, function) -> list:
        out = []
        for i in range(self.flint.lib.flint_gpt2_parameter_count(self.handle)):
            info = Gpt2ParameterInfo()
            self.flint.check(function(self.handle, i, ctypes.byref(info)))
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

    def backward(self, tokens, accumulate=False, stream=None) -> None:
        """Every master weight's gradient from the last forward pass, which must have run on
        the same tokens and stream; written, or with accumulate added to each."""
        self.flint.check(
            self.flint.lib.flint_gpt2_backward(
                self.handle, pointer(tokens), int(accumulate), stream_handle(stream)
            )
        )

    def gradient_norm(self, stream=None) -> None:
        """The norm of every gradient as they stand, and the count of their values that are not
        finite, into the model's own two results (gradient_norm_outputs); the model must have
        been created with backward."""
        self.flint.check(
            self.flint.lib.flint_gpt2_gradient_norm(self.handle, stream_handle(stream))
        )

    def gradient_norm_outputs(self) -> tuple:
        """The device addresses of the gradient norm's FP64 norm and uint64 count."""
        norm, nonfinite = ctypes.c_void_p(), ctypes.c_void_p()
        self.flint.check(
            self.flint.lib.flint_gpt2_gradient_norm_outputs(
                self.handle, ctypes.byref(norm), ctypes.byref(nonfinite)
            )
        )
        return norm.value, nonfinite.value

    def read_gradient_norm(self, stream=None) -> tuple:
        """(norm as a float, count as an int) that the last gradient_norm left. It runs nothing:
        it synchronizes, then reads the two results."""
        norm_at, nonfinite_at = self.gradient_norm_outputs()
        norm = self.read(norm_at, (), torch.float64, 0, stream)
        nonfinite = self.read(nonfinite_at, (), torch.int64, 0, stream)
        return norm.item(), nonfinite.item()

    def reset_optimizer(self, initial_scale=2.0**16, stream=None) -> None:
        """Zeroes every moment and sets the optimizer's state to {initial_scale, 0, 0, 0}; the
        model must have been created with optimizer."""
        self.flint.check(
            self.flint.lib.flint_gpt2_reset_optimizer(
                self.handle, initial_scale, stream_handle(stream)
            )
        )

    def step(self, config, stream=None) -> None:
        """One optimizer step (an AdamwConfig) over the gradients as they stand, then
        sync_weights; the model must have been created with optimizer."""
        self.flint.check(
            self.flint.lib.flint_gpt2_step(self.handle, ctypes.byref(config), stream_handle(stream))
        )

    def moments(self) -> list:
        """(name, exp_avg_address, exp_avg_sq_address, rows, cols) of each master weight, in the
        library's order."""
        out = []
        for i in range(self.flint.lib.flint_gpt2_parameter_count(self.handle)):
            avg, avg_sq = Gpt2ParameterInfo(), Gpt2ParameterInfo()
            self.flint.check(
                self.flint.lib.flint_gpt2_moments(
                    self.handle, i, ctypes.byref(avg), ctypes.byref(avg_sq)
                )
            )
            out.append((avg.name.decode(), avg.data, avg_sq.data, avg.rows, avg.cols))
        return out

    def optimizer_state_address(self) -> int:
        """The device address of the model's flint_adamw_state."""
        state = ctypes.c_void_p()
        self.flint.check(
            self.flint.lib.flint_gpt2_optimizer_state(self.handle, ctypes.byref(state))
        )
        return state.value

    def read_optimizer_state(self) -> dict:
        """The optimizer's state as a dict (see read_adamw_state); it synchronizes."""
        return read_adamw_state(self.optimizer_state_address())

    def write_optimizer_state(self, state: dict, stream=None) -> None:
        """Copies a state (a dict with read_adamw_state's keys, any of which may be left out
        for its default) to the model's; it waits for the copy."""
        source = adamw_state(**state)
        self.flint.copy(self.optimizer_state_address(), source, 24, stream)
        torch.cuda.synchronize()

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


# --- safetensors -----------------------------------------------------------------------------


def _text_bytes(value):
    """A str as UTF-8 bytes; bytes pass through, so a test can hand over invalid UTF-8."""
    return value if isinstance(value, bytes) else str(value).encode("utf-8")


class SafetensorsInfo(tuple):
    """One tensor of a safetensors file, as a tuple (name, dtype, shape, bits, offset, bytes)
    with those fields also readable by name; name and dtype are str, shape is a tuple."""

    _names = ("name", "dtype", "shape", "bits", "offset", "bytes")

    def __new__(cls, name, dtype, shape, bits, offset, nbytes):
        return super().__new__(cls, (name, dtype, tuple(shape), bits, offset, nbytes))

    def __getattr__(self, field):
        try:
            return self[self._names.index(field)]
        except ValueError:
            raise AttributeError(field) from None


def safetensors_open_status(flint, path) -> int:
    """The status of flint_safetensors_open(path); a reader it opens is closed again."""
    handle = ctypes.c_void_p()
    status = flint.lib.flint_safetensors_open(_text_bytes(os.fspath(path)), ctypes.byref(handle))
    if status == OK:
        flint.lib.flint_safetensors_close(handle)
    elif handle.value is not None:
        raise AssertionError("flint_safetensors_open failed but returned a reader")
    return status


class SafetensorsReader:
    """A flint_safetensors_reader, closed by close() or on leaving a with block. Opening raises
    FlintError (with the status) when the library refuses the file."""

    def __init__(self, flint: Flint, path):
        self.flint = flint
        self.path = os.fspath(path)
        handle = ctypes.c_void_p()
        status = flint.lib.flint_safetensors_open(_text_bytes(self.path), ctypes.byref(handle))
        flint.check(status)
        self.handle = handle.value

    def close(self) -> None:
        if self.handle is not None:
            self.flint.lib.flint_safetensors_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def count(self) -> int:
        count = ctypes.c_size_t()
        self.flint.check(
            self.flint.lib.flint_safetensors_tensor_count(self.handle, ctypes.byref(count))
        )
        return count.value

    def info(self, index: int) -> SafetensorsInfo:
        info = SafetensorsTensor()
        self.flint.check(
            self.flint.lib.flint_safetensors_tensor_info(self.handle, index, ctypes.byref(info))
        )
        shape = [info.shape[d] for d in range(info.rank)]
        return SafetensorsInfo(
            info.name.decode("utf-8"), info.dtype.decode("ascii"), shape, info.bits, info.offset,
            info.bytes,
        )

    def infos(self) -> list:
        return [self.info(i) for i in range(self.count())]

    def find(self, name) -> int:
        index = ctypes.c_size_t()
        self.flint.check(
            self.flint.lib.flint_safetensors_find(
                self.handle, _text_bytes(name), ctypes.byref(index)
            )
        )
        return index.value

    def metadata_count(self) -> int:
        count = ctypes.c_size_t()
        self.flint.check(
            self.flint.lib.flint_safetensors_metadata_count(self.handle, ctypes.byref(count))
        )
        return count.value

    def metadata_pair(self, index: int) -> tuple:
        """(key, value) as UTF-8 bytes."""
        key, value = ctypes.c_char_p(), ctypes.c_char_p()
        self.flint.check(
            self.flint.lib.flint_safetensors_metadata(
                self.handle, index, ctypes.byref(key), ctypes.byref(value)
            )
        )
        return key.value, value.value

    def metadata_pairs(self) -> list:
        """[(key, value)] as str, in the header's order."""
        pairs = [self.metadata_pair(i) for i in range(self.metadata_count())]
        return [(k.decode("utf-8"), v.decode("utf-8")) for k, v in pairs]

    def metadata_find(self, key) -> bytes:
        value = ctypes.c_char_p()
        self.flint.check(
            self.flint.lib.flint_safetensors_metadata_find(
                self.handle, _text_bytes(key), ctypes.byref(value)
            )
        )
        return value.value

    def read(self, index: int, offset: int = 0, nbytes=None) -> bytes:
        """Bytes [offset, offset + nbytes) of tensor index; the rest of it with nbytes None."""
        if nbytes is None:
            nbytes = self.info(index).bytes - offset
        buffer = ctypes.create_string_buffer(nbytes) if nbytes else None
        self.flint.check(
            self.flint.lib.flint_safetensors_read(self.handle, index, offset, nbytes, buffer)
        )
        return buffer.raw if nbytes else b""


def safetensors_declare(tensors, metadata=()):
    """The arguments of flint_safetensors_create for tensors [(name, dtype, shape), ...] and
    metadata [(key, value), ...] (str or bytes): (array, keys, values, keepalive). The array
    holds the flint_safetensors_tensor entries; keepalive holds what they point to."""
    array = (SafetensorsTensor * max(len(tensors), 1))()
    keepalive = []
    for slot, (name, dtype, shape) in zip(array, tensors):
        name, dtype = _text_bytes(name), _text_bytes(dtype)
        shape_array = (ctypes.c_size_t * max(len(shape), 1))(*shape)
        keepalive.append((name, dtype, shape_array))
        slot.name, slot.dtype, slot.rank = name, dtype, len(shape)
        slot.shape = ctypes.cast(shape_array, ctypes.POINTER(ctypes.c_size_t))
    keys = (ctypes.c_char_p * max(len(metadata), 1))(*[_text_bytes(k) for k, _ in metadata])
    values = (ctypes.c_char_p * max(len(metadata), 1))(*[_text_bytes(v) for _, v in metadata])
    return array, keys, values, keepalive


def safetensors_create(flint, path, tensors, metadata=()):
    """flint_safetensors_create without raising: (status, writer handle or None, array), the
    array being the entries the call fills (see safetensors_declare)."""
    array, keys, values, keepalive = safetensors_declare(tensors, metadata)
    writer = ctypes.c_void_p()
    status = flint.lib.flint_safetensors_create(
        _text_bytes(os.fspath(path)), array, len(tensors), keys, values, len(metadata),
        ctypes.byref(writer),
    )
    del keepalive
    return status, writer.value, array


class SafetensorsWriter:
    """A flint_safetensors_writer, destroyed by destroy() or on leaving a with block (a writer
    that was never committed then removes its temporary file). Creating raises FlintError."""

    def __init__(self, flint: Flint, path, tensors, metadata=()):
        self.flint = flint
        self.path = os.fspath(path)
        status, handle, self.array = safetensors_create(flint, path, tensors, metadata)
        flint.check(status)
        self.handle = handle
        # What create filled in: (bits, offset, bytes) per tensor.
        self.entries = [(t.bits, t.offset, t.bytes) for t in self.array[: len(tensors)]]

    @property
    def tmp_path(self) -> str:
        return self.path + ".tmp"

    def write(self, data) -> None:
        self.flint.check(self.write_status(data))

    def write_status(self, data) -> int:
        return self.flint.lib.flint_safetensors_write(self.handle, data, len(data))

    def commit(self) -> None:
        self.flint.check(self.commit_status())

    def commit_status(self) -> int:
        return self.flint.lib.flint_safetensors_commit(self.handle)

    def destroy(self) -> None:
        if self.handle is not None:
            self.flint.lib.flint_safetensors_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.destroy()


def write_safetensors(flint, path, tensors, metadata=(), chunks=None):
    """Writes a whole file: tensors is [(name, dtype, shape, payload bytes), ...], in the
    order they sit in the file. The payloads are written as one run, in one call or in pieces
    of the sizes chunks lists (the last repeats; an int means pieces of that size). Returns
    the entries create filled in, [(bits, offset, bytes), ...]."""
    with SafetensorsWriter(flint, path, [t[:3] for t in tensors], metadata) as writer:
        data = b"".join(t[3] for t in tensors)
        if chunks is None:
            writer.write(data)
        else:
            sizes = [chunks] if isinstance(chunks, int) else list(chunks)
            position, step = 0, 0
            while position < len(data):
                size = sizes[min(step, len(sizes) - 1)]
                writer.write(data[position : position + size])
                position += size
                step += 1
        writer.commit()
        return writer.entries


# --- the data loader -------------------------------------------------------------------------


def _loader_paths(paths):
    """The arguments flint_loader_open takes for a list of paths (str, bytes or path-like; None
    gives a null pointer): (array, keepalive). The array points into the byte strings keepalive
    holds, so keep both alive for the call."""
    texts = [None if path is None else _text_bytes(os.fspath(path)) for path in paths]
    array = (ctypes.c_char_p * max(len(texts), 1))(*texts)
    return array, texts


def _loader_config(length, vocab, seed, shuffle, max_tokens) -> LoaderConfig:
    return LoaderConfig(length, max_tokens, vocab, seed, int(shuffle))


def loader_open_status(
    flint, paths, *, length, vocab, seed=0, shuffle=True, max_tokens=0
) -> int:
    """The status of flint_loader_open; a loader it opens is closed again."""
    array, keepalive = _loader_paths(paths)
    config = _loader_config(length, vocab, seed, shuffle, max_tokens)
    handle = ctypes.c_void_p()
    status = flint.lib.flint_loader_open(
        array, len(paths), ctypes.byref(config), ctypes.byref(handle)
    )
    del keepalive
    if status == OK:
        flint.lib.flint_loader_close(handle)
    elif handle.value is not None:
        raise AssertionError("flint_loader_open failed but returned a loader")
    return status


def _buffer_address(buffer):
    """The address of a buffer for flint_loader_read: None, an int, or an object with one (a
    numpy array's ctypes.data, a torch tensor's data_ptr())."""
    if buffer is None or isinstance(buffer, int):
        return buffer
    if hasattr(buffer, "ctypes"):
        return buffer.ctypes.data
    return buffer.data_ptr()


class Loader:
    """A flint_loader, closed by close() or on leaving a with block. Opening raises FlintError
    (with the status) when the library refuses the files."""

    def __init__(
        self, flint: Flint, paths, length, vocab, seed=0, shuffle=True, max_tokens=0
    ):
        self.flint = flint
        self.paths = [os.fspath(p) for p in paths]
        self.length = length
        array, keepalive = _loader_paths(self.paths)
        config = _loader_config(length, vocab, seed, shuffle, max_tokens)
        handle = ctypes.c_void_p()
        status = flint.lib.flint_loader_open(
            array, len(self.paths), ctypes.byref(config), ctypes.byref(handle)
        )
        del keepalive
        flint.check(status)
        self.handle = handle.value

    def close(self) -> None:
        if self.handle is not None:
            self.flint.lib.flint_loader_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def epoch_sequences(self) -> int:
        sequences = ctypes.c_size_t()
        self.flint.check(
            self.flint.lib.flint_loader_epoch_sequences(self.handle, ctypes.byref(sequences))
        )
        return sequences.value

    def locate(self, index: int) -> tuple:
        """(epoch, file, sequence) of sequence number index."""
        epoch, file, sequence = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        self.flint.check(
            self.flint.lib.flint_loader_locate(
                self.handle, index, ctypes.byref(epoch), ctypes.byref(file),
                ctypes.byref(sequence),
            )
        )
        return epoch.value, file.value, sequence.value

    def read_into(self, index: int, count: int, tokens, targets) -> int:
        """flint_loader_read without raising: the raw status. tokens and targets are numpy arrays
        (or anything with an address), or None for a null pointer."""
        return self.flint.lib.flint_loader_read(
            self.handle, index, count, _buffer_address(tokens), _buffer_address(targets)
        )

    def read(self, index: int, count: int) -> tuple:
        """(tokens, targets), numpy uint16 arrays of shape (count, length)."""
        tokens = np.empty((count, self.length), dtype=np.uint16)
        targets = np.empty((count, self.length), dtype=np.uint16)
        self.flint.check(self.read_into(index, count, tokens, targets))
        return tokens, targets
