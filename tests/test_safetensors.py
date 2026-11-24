"""Tests for D33: Flint's safetensors reader and writer, against Python's safetensors 0.8.0.

The ground truth for every format rule is that package; data/d33/spec.md lists the rules and
probe_python*.py the probes. A file is "accepted" by Python when safetensors.safe_open opens it
(its names, dtypes, shapes and metadata can then be read) and "rejected" when that raises
SafetensorError. Flint's verdict is the status of flint_safetensors_open: OK, or
FLINT_ERROR_FORMAT for a malformed file. The two agree on every file except the documented
differences: Flint also rejects a tensor name or a metadata key given twice and a NUL in a
name, a metadata key or a metadata value (Python accepts them), and a number in an unknown
field within about 1e-16 of the largest FP64 value, where serde_json's float parsing isn't
correctly rounded.

Sections: 1 Python writes and Flint reads, 2 OpenAI's file, 3 verdicts against Python, 4 Flint
writes and Python reads, 5 writer errors and durability, 6 reader API errors, 7 64-bit reads.

Inputs are bytes from random.Random(seed).randbytes (never torch.randn, which differs between
machines). Nothing here launches a kernel, and compute-sanitizer can't see host code, so the
whole module is skipped under the sanitizer.
"""

import ctypes
import gc
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter, namedtuple
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import safetensors
import safetensors.torch
import torch
from safetensors import SafetensorError, safe_open

import flint_lib
from flint_lib import (
    ERROR_FORMAT,
    ERROR_INVALID_ARGUMENT,
    ERROR_IO,
    OK,
    FlintError,
    SafetensorsInfo,
    SafetensorsReader,
    SafetensorsWriter,
    safetensors_create,
    safetensors_declare,
    safetensors_open_status,
    write_safetensors,
)
from inputs import SANITIZER
from test_gpt2 import data_file, sha256_of

pytestmark = pytest.mark.skipif(
    SANITIZER, reason="host-only tests launch no kernel; D33 has its own host-sanitizer run"
)

MiB = 1 << 20
U64_MAX = 2**64 - 1
MAX_HEADER = 100_000_000

# The 22 dtypes and their bits per element.
DTYPE_BITS = {
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E5M2": 8, "F8_E4M3": 8, "F8_E8M0": 8, "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "C64": 64, "F64": 64, "I64": 64, "U64": 64,
}  # fmt: skip
# The dtypes torch 2.6 has (the other four, F4, F6_E2M3, F6_E3M2 and F8_E8M0, it lacks).
TORCH_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "F8_E5M2": torch.float8_e5m2,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E4M3FNUZ": torch.float8_e4m3fnuz,
    "F8_E5M2FNUZ": torch.float8_e5m2fnuz, "I16": torch.int16, "U16": torch.uint16,
    "F16": torch.float16, "BF16": torch.bfloat16, "I32": torch.int32, "U32": torch.uint32,
    "F32": torch.float32, "C64": torch.complex64, "F64": torch.float64, "I64": torch.int64,
    "U64": torch.uint64,
}  # fmt: skip
SUBBYTE_SHAPES = {"F4": (2, 8), "F6_E2M3": (4, 2), "F6_E3M2": (4, 2)}

# What the 22 names must come to, for the tests that don't trust the library's table.
assert len(DTYPE_BITS) == 22 and len(TORCH_DTYPES) == 18

D8 = bytes(range(8))
U8_8 = '{"dtype":"U8","shape":[8],"data_offsets":[0,8]}'


# --- helpers: files --------------------------------------------------------------------------


def T(dtype, shape, begin, end):
    """A tensor's header entry."""
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, end]}


def pad8(raw):
    return raw + b" " * ((8 - len(raw) % 8) % 8)


def file_bytes(header, data=b"", pad=True):
    """8-byte length, header (str or bytes, padded with spaces to a multiple of 8 unless pad is
    False) and data."""
    raw = header.encode("utf-8") if isinstance(header, str) else bytes(header)
    if pad:
        raw = pad8(raw)
    return struct.pack("<Q", len(raw)) + raw + data


def tensor_with(extra, key="x"):
    """A one-tensor file whose tensor has the unknown field key set to the JSON text extra."""
    header = '{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"' + key + '":' + extra + "}}"
    return file_bytes(header, D8)


def split_blob(blob):
    n = struct.unpack("<Q", blob[:8])[0]
    return n, blob[8 : 8 + n], blob[8 + n :]


def element_bytes(dtype, shape):
    bits = DTYPE_BITS[dtype] * math.prod(shape)
    assert bits % 8 == 0
    return bits // 8


def make_payload(dtype, nbytes, seed):
    """nbytes seeded bytes; for BOOL only 0 and 1, which are the values torch's bool holds."""
    data = random.Random(seed).randbytes(nbytes)
    return bytes(b & 1 for b in data) if dtype == "BOOL" else data


def torch_tensor(dtype, shape, payload):
    if not payload:
        return torch.empty(tuple(shape), dtype=TORCH_DTYPES[dtype])
    return torch.frombuffer(bytearray(payload), dtype=TORCH_DTYPES[dtype]).reshape(tuple(shape))


def torch_raw(tensor):
    """A torch tensor's bytes."""
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def hand_built(tensors, metadata=None, pad=True):
    """A file as the test lays it out: tensors [(name, dtype, shape, payload)] in this order, a
    compact header, padding. metadata is a dict, or None for no __metadata__ key."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    for name, dtype, shape, payload in tensors:
        assert len(payload) == element_bytes(dtype, shape)
        header[name] = T(dtype, shape, offset, offset + len(payload))
        offset += len(payload)
    text = json.dumps(header, ensure_ascii=False, separators=(",", ":"))
    return file_bytes(text, b"".join(t[3] for t in tensors), pad)


def reference_header(tensors, metadata=()):
    """The header Flint's writer must produce, as the spec describes it: json.dumps compact and
    unescaped, __metadata__ first and only with metadata, padded with spaces to 8 bytes."""
    header = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for name, dtype, shape, payload in tensors:
        header[name] = T(dtype, shape, offset, offset + len(payload))
        offset += len(payload)
    text = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return pad8(text)


def tensor_fields(value):
    """(dtype, shape, offsets) of a tensor entry that json.loads read with object_pairs_hook=list:
    the object form (a list of pairs) or the array form (three elements), with the dtype as a
    string or as a one-key map (a list of one pair)."""
    if value and isinstance(value[0], tuple):
        fields = dict(value)
        dtype, shape, offsets = fields["dtype"], fields["shape"], fields["data_offsets"]
    else:
        dtype, shape, offsets = value
    if isinstance(dtype, list):
        ((dtype, _),) = dtype
    return dtype, shape, offsets


def header_entries(text):
    """The test's own parse of a valid header (bytes): ([(name, dtype, shape, begin, end)] and
    [(key, value)], each in the header's order)."""
    top = json.loads(text.decode("utf-8"), object_pairs_hook=list)
    tensors, metadata = [], []
    for key, value in top:
        if key == "__metadata__":
            metadata = [] if value is None else list(value)
        else:
            dtype, shape, (begin, end) = tensor_fields(value)
            tensors.append((key, dtype, shape, begin, end))
    return tensors, metadata


# --- helpers: verdicts -----------------------------------------------------------------------

Verdict = namedtuple("Verdict", "accepted tensors metadata message")


def python_verdict(path):
    """Python's verdict on the file: safe_open opens it (accepted) or raises SafetensorError
    (rejected). For an accepted file, tensors is {name: (dtype, shape)} and metadata the dict
    (or None). Any other exception is the test's error, except on an empty file, where the
    package may report the failed mmap in its own way."""
    path = str(path)
    try:
        with safe_open(path, framework="np") as f:
            keys = list(f.keys())
            metadata = f.metadata()
            tensors = {}
            for key in keys:
                piece = f.get_slice(key)
                tensors[key] = (piece.get_dtype(), tuple(piece.get_shape()))
                del piece
    except SafetensorError as error:
        return Verdict(False, None, None, str(error))
    except Exception as error:  # noqa: BLE001
        if os.path.getsize(path) == 0:
            return Verdict(False, None, None, f"empty file: {error}")
        raise
    return Verdict(True, tensors, metadata, "")


def flint_status(flint, path):
    return safetensors_open_status(flint, path)


def check_reads(flint, path, blob):
    """Flint opens the file at path, whose content is blob, and every info, metadata pair and
    byte is what the test's own parse of the header says."""
    n, text, data = split_blob(blob)
    tensors, metadata = header_entries(text)
    with SafetensorsReader(flint, path) as reader:
        assert reader.count() == len(tensors)
        for index, (name, dtype, shape, begin, end) in enumerate(tensors):
            expected = SafetensorsInfo(name, dtype, shape, DTYPE_BITS[dtype], begin, end - begin)
            assert reader.info(index) == expected
            assert reader.find(name) == index
            assert reader.read(index) == data[begin:end], name
        assert reader.metadata_count() == len(metadata)
        assert reader.metadata_pairs() == [tuple(pair) for pair in metadata]
        for key, value in metadata:
            assert reader.metadata_find(key) == value.encode("utf-8")


def check_matches_python(flint, path, verdict):
    """Flint's names, dtypes, shapes and metadata equal Python's (names as sorted lists, since
    safe_open keeps its own order)."""
    assert verdict.accepted
    with SafetensorsReader(flint, path) as reader:
        infos = reader.infos()
        assert sorted(i.name for i in infos) == sorted(verdict.tensors)
        for info in infos:
            assert verdict.tensors[info.name] == (info.dtype, info.shape), info.name
        pairs = reader.metadata_pairs()
        assert dict(pairs) == (verdict.metadata or {})
        assert len(pairs) == len(verdict.metadata or {})


def accepted_by_both(flint, path, blob=None):
    """Python accepts the file, Flint opens it, and they agree on what it holds. With blob (the
    file's content) every byte is checked too."""
    verdict = python_verdict(path)
    assert verdict.accepted, verdict.message
    check_matches_python(flint, path, verdict)
    if blob is not None:
        check_reads(flint, path, blob)


def rejected_by_both(flint, path):
    verdict = python_verdict(path)
    assert not verdict.accepted, "Python accepted the file"
    assert flint_status(flint, path) == ERROR_FORMAT, flint.last_error()


def write_blob(tmp_path, blob, name="case.safetensors"):
    path = tmp_path / name
    path.write_bytes(blob)
    return path


# =============================================================================================
# 1. Python writes, Flint reads
# =============================================================================================


@pytest.mark.parametrize("dtype", list(DTYPE_BITS))
def test_every_dtype_written_by_python_is_read(flint, tmp_path, dtype):
    shape = SUBBYTE_SHAPES.get(dtype, (3, 5))
    nbytes = element_bytes(dtype, shape)
    payload = make_payload(dtype, nbytes, seed=hash_seed(dtype))
    head, tail = b"\x07", b"xyz"
    path = tmp_path / "dtype.safetensors"
    if dtype in TORCH_DTYPES:
        safetensors.torch.save_file(
            {
                "head": torch.tensor([7], dtype=torch.uint8),
                "x": torch_tensor(dtype, shape, payload),
                "tail": torch.frombuffer(bytearray(tail), dtype=torch.uint8),
            },
            str(path),
        )
        blob = path.read_bytes()
    else:
        blob = hand_built(
            [("head", "U8", (1,), head), ("x", dtype, shape, payload), ("tail", "U8", (3,), tail)]
        )
        path.write_bytes(blob)
    verdict = python_verdict(path)
    assert verdict.accepted, verdict.message
    assert verdict.tensors["x"] == (dtype, tuple(shape))
    with SafetensorsReader(flint, path) as reader:
        info = reader.info(reader.find("x"))
        assert info.dtype == dtype
        assert info.bits == DTYPE_BITS[dtype]
        assert info.shape == tuple(shape)
        assert info.bytes == nbytes
        _, text, data = split_blob(blob)
        entries = {e[0]: e for e in header_entries(text)[0]}
        assert info.offset == entries["x"][3]
        assert reader.read(reader.find("x")) == payload
    check_matches_python(flint, path, verdict)
    check_reads(flint, path, blob)
    if dtype in TORCH_DTYPES:
        loaded = safetensors.torch.load_file(str(path))["x"]
        assert loaded.dtype == TORCH_DTYPES[dtype]
        assert torch_raw(loaded) == payload


def hash_seed(text):
    """A seed that is the same on every machine (Python's str hash isn't)."""
    return sum(ord(c) * 31**i for i, c in enumerate(text)) % 2**31


@pytest.mark.parametrize(
    "dtype, shape, nbytes",
    [
        ("F4", (2,), 1),
        ("F4", (16,), 8),
        ("F4", (2, 8), 8),
        ("F4", (1, 1, 2), 1),
        ("F4", (0,), 0),
        ("F4", (3, 0), 0),
        ("F6_E2M3", (4,), 3),
        ("F6_E2M3", (8,), 6),
        ("F6_E3M2", (4, 2), 6),
        ("F6_E3M2", (2, 2, 2, 2), 12),
        ("F6_E3M2", (0, 3), 0),
    ],
)
def test_subbyte_shapes_that_end_on_a_byte(flint, tmp_path, dtype, shape, nbytes):
    assert element_bytes(dtype, shape) == nbytes
    payload = make_payload(dtype, nbytes, seed=nbytes + 5)
    blob = hand_built([("s", dtype, shape, payload), ("u", "U8", (2,), b"ab")])
    path = write_blob(tmp_path, blob)
    accepted_by_both(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        info = reader.info(reader.find("s"))
        assert (info.bits, info.shape, info.bytes) == (DTYPE_BITS[dtype], tuple(shape), nbytes)


@pytest.mark.parametrize("rank", range(9))
def test_ranks_0_to_8_written_by_python(flint, tmp_path, rank):
    shape = (2, 3, 1, 2, 1, 2, 2, 1)[:rank]
    nbytes = element_bytes("F16", shape)
    payload = make_payload("F16", nbytes, seed=rank)
    path = tmp_path / "rank.safetensors"
    safetensors.torch.save_file(
        {"t": torch_tensor("F16", shape, payload), "u": torch.zeros(3, dtype=torch.uint8)},
        str(path),
    )
    blob = path.read_bytes()
    accepted_by_both(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        info = reader.info(reader.find("t"))
        assert info.shape == shape and info.bytes == nbytes
        assert reader.read(reader.find("t")) == payload


@pytest.mark.parametrize("shape", [(0,), (3, 0, 2), (0, 0), (5, 0), (0, 7, 3)])
def test_zero_size_shapes_written_by_python(flint, tmp_path, shape):
    path = tmp_path / "zero.safetensors"
    safetensors.torch.save_file(
        {
            "z": torch.empty(shape, dtype=torch.float32),
            "u": torch.frombuffer(bytearray(b"1234"), dtype=torch.uint8),
        },
        str(path),
    )
    blob = path.read_bytes()
    accepted_by_both(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        info = reader.info(reader.find("z"))
        assert info.shape == shape and info.bytes == 0
        assert reader.read(reader.find("z")) == b""
        assert reader.read(reader.find("u")) == b"1234"


def _zero_size_layouts():
    """Layouts (name, shape, begin, end) with zero-size tensors at one offset, between tensors
    and at the end, in several declaration orders, over data of 8 bytes."""
    a, b = ("a", [4], 0, 4), ("b", [4], 4, 8)
    z = lambda name, at: (name, [0], at, at)  # noqa: E731
    return {
        "three at 0, then a": ([z("z1", 0), z("z2", 0), z("z3", 0), ("a", [8], 0, 8)], 8),
        "a, then three at 0": ([("a", [8], 0, 8), z("z1", 0), z("z2", 0), z("z3", 0)], 8),
        "one at 0 on both sides": ([z("z1", 0), ("a", [8], 0, 8), z("z2", 0)], 8),
        "two between": ([a, z("z1", 4), z("z2", 4), b], 8),
        "two between, declared first": ([z("z1", 4), a, b, z("z2", 4)], 8),
        "two between, declared apart": ([b, z("z1", 4), a, z("z2", 4)], 8),
        "at the end": ([("a", [8], 0, 8), z("z1", 8), z("z2", 8)], 8),
        "at the end, declared first": ([z("z1", 8), z("z2", 8), ("a", [8], 0, 8)], 8),
        "everywhere": ([z("z0", 0), a, z("z1", 4), z("z2", 4), b, z("z3", 8), z("z4", 8)], 8),
        "only zero-size": ([z("z1", 0), z("z2", 0)], 0),
    }


@pytest.mark.parametrize("layout", list(_zero_size_layouts()))
def test_zero_size_tensors_at_one_offset_between_and_at_the_end(flint, tmp_path, layout):
    entries, data_length = _zero_size_layouts()[layout]
    header = {name: T("U8", shape, begin, end) for name, shape, begin, end in entries}
    blob = file_bytes(json.dumps(header, separators=(",", ":")), bytes(range(data_length)))
    path = write_blob(tmp_path, blob)
    accepted_by_both(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        assert [i.name for i in reader.infos()] == [e[0] for e in entries]


def test_2000_tensors_in_one_file(flint, tmp_path):
    rng = random.Random(2000)
    choices = [("U8", 1), ("F32", 4), ("I64", 8), ("F16", 2), ("BOOL", 1), ("C64", 8)]
    declared, payload, offset = [], bytearray(), 0
    for i in range(2000):
        dtype, size = rng.choice(choices)
        count = rng.randrange(0, 6)
        shape = [count] if i % 3 else [count, 2]
        nbytes = size * math.prod(shape)
        declared.append((f"model.layers.{i}.w", dtype, shape, offset, offset + nbytes))
        payload += rng.randbytes(nbytes)
        offset += nbytes
    # The header lists the tensors in an order that isn't the data's.
    rng.shuffle(declared)
    header = {name: T(dtype, shape, begin, end) for name, dtype, shape, begin, end in declared}
    blob = file_bytes(json.dumps(header, separators=(",", ":")), bytes(payload))
    path = write_blob(tmp_path, blob)
    verdict = python_verdict(path)
    assert verdict.accepted, verdict.message
    check_matches_python(flint, path, verdict)
    check_reads(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        assert [i.name for i in reader.infos()] == [d[0] for d in declared]
        assert [reader.find(d[0]) for d in declared] == list(range(2000))
        assert reader.count() == 2000


def _metadata_cases():
    fifty = {f"key{i:02d}": f"value {i}" for i in range(50)}
    awkward = {
        "k\u0001\n": 'v/"\\ é 日本   \U0001f600',
        "path/to/file": "a/b/c",
        "\u0001": "\u001f\u007f",
        "": "",
        "empty value": "",
        "日本語": "テスト",
        "tab\there": "line\nbreak\r\n",
        "quote\"": "back\\slash",
    }
    return {
        "none": (None, None),
        "null": (None, "null"),
        "empty object": ({}, "{}"),
        "one pair": ({"format": "pt"}, None),
        "fifty pairs": (fifty, None),
        "awkward, escaped": (awkward, None),
        "awkward, raw": (awkward, "raw"),
        "empty key and value": ({"": ""}, None),
        "after the tensors": ({"format": "pt", "x": "y"}, "last"),
    }


@pytest.mark.parametrize("case", list(_metadata_cases()))
def test_metadata_written_by_python_is_read(flint, tmp_path, case):
    pairs, variant = _metadata_cases()[case]
    entry = T("U8", [8], 0, 8)
    if case == "none":
        header = {"a": entry}
    elif variant == "null":
        header = {"__metadata__": None, "a": entry}
    elif variant == "last":
        header = {"a": entry, "__metadata__": pairs}
    else:
        header = {"__metadata__": pairs, "a": entry}
    text = json.dumps(header, ensure_ascii=variant != "raw", separators=(",", ":"))
    blob = file_bytes(text, D8)
    path = write_blob(tmp_path, blob)
    verdict = python_verdict(path)
    assert verdict.accepted, verdict.message
    assert (verdict.metadata or {}) == (pairs or {})
    check_matches_python(flint, path, verdict)
    check_reads(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        assert reader.metadata_count() == len(pairs or {})
        for key, value in (pairs or {}).items():
            assert reader.metadata_find(key) == value.encode("utf-8")
        for missing in ("absent", "format ", "KEY00", "nopeé"):
            if missing not in (pairs or {}):
                with pytest.raises(FlintError) as error:
                    reader.metadata_find(missing)
                assert error.value.status == ERROR_INVALID_ARGUMENT
                assert missing in str(error.value)


NAME_CASES = {
    "empty": ('""', ""),
    "non-ASCII, raw": ('"é日本😀"', "é日本😀"),
    "quote": ('"a\\"b"', 'a"b'),
    "backslashes": ('"a\\\\b\\\\"', "a\\b\\"),
    "control characters": ('"a\\u0001\\u001f\\n\\t\\b\\f\\r"', "a\x01\x1f\n\t\b\f\r"),
    "escaped solidus": ('"a\\/b"', "a/b"),
    "raw solidus": ('"a/b"', "a/b"),
    "\\u escapes": ('"\\u00e9\\u65e5\\u0041"', "é日A"),
    "\\u uppercase hex": ('"\\u00E9\\u65E5"', "é日"),
    "surrogate pair": ('"\\ud83d\\ude00"', "😀"),
    "surrogate pair, uppercase": ('"\\uD83D\\uDE00"', "😀"),
    "surrogate pair inside text": ('"x\\ud83d\\ude00y\\ud835\\udc00z"', "x😀y\U0001d400z"),
    "raw DEL": ('"a\x7fb"', "a\x7fb"),
    "raw U+2028": ('"a b"', "a b"),
    "a quote written as \\u0022": ('"a\\u0022b"', 'a"b'),
    "long": ('"' + "n" * 4000 + '"', "n" * 4000),
    "a near miss of the metadata key": ('"__metadata"', "__metadata"),
    "starts with underscores": ('"__x__"', "__x__"),
}


@pytest.mark.parametrize("case", list(NAME_CASES))
def test_tensor_names(flint, tmp_path, case):
    key_text, expected = NAME_CASES[case]
    blob = file_bytes('{' + key_text + ":" + U8_8 + "}", D8)
    path = write_blob(tmp_path, blob)
    verdict = python_verdict(path)
    assert verdict.accepted, verdict.message
    assert list(verdict.tensors) == [expected]
    with SafetensorsReader(flint, path) as reader:
        assert reader.count() == 1
        assert reader.info(0).name == expected
        assert reader.find(expected) == 0
        assert reader.read(0) == D8
    check_reads(flint, path, blob)


def test_names_that_differ_by_one_byte_are_different_tensors(flint, tmp_path):
    names = ["a", "A", "a ", " a", "á", "á", "ab", "b", "a\x7f"]
    header = {name: T("U8", [1], i, i + 1) for i, name in enumerate(names)}
    blob = file_bytes(json.dumps(header, separators=(",", ":")), bytes(range(len(names))))
    path = write_blob(tmp_path, blob)
    accepted_by_both(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        assert [reader.find(n) for n in names] == list(range(len(names)))
        for missing in ("", "c", "á́", "ABC"):
            with pytest.raises(FlintError):
                reader.find(missing)


VALID_UTF8_NAMES = {
    "U+0080": b"\xc2\x80",
    "U+07FF": b"\xdf\xbf",
    "U+0800": b"\xe0\xa0\x80",
    "U+D7FF": b"\xed\x9f\xbf",
    "U+E000": b"\xee\x80\x80",
    "U+FFFF": b"\xef\xbf\xbf",
    "U+10000": b"\xf0\x90\x80\x80",
    "U+10FFFF": b"\xf4\x8f\xbf\xbf",
}


@pytest.mark.parametrize("case", list(VALID_UTF8_NAMES))
def test_the_edges_of_valid_utf8_in_a_name(flint, tmp_path, case):
    raw = VALID_UTF8_NAMES[case]
    blob = file_bytes(b'{"a' + raw + b'":' + U8_8.encode() + b"}", D8)
    path = write_blob(tmp_path, blob)
    accepted_by_both(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        assert reader.info(0).name == "a" + raw.decode("utf-8")


def _layout_cases():
    one = '"a":' + U8_8
    return {
        "unpadded header": (file_bytes('{"a":' + U8_8 + "}", D8, pad=False), None),
        "padded with tabs": (file_bytes('{"a":' + U8_8 + "}" + "\t" * 5, D8, pad=False), None),
        "padded with newlines": (file_bytes('{"a":' + U8_8 + "}" + "\n" * 6, D8, pad=False), None),
        "padded with carriage returns": (
            file_bytes('{"a":' + U8_8 + "}" + "\r" * 3, D8, pad=False),
            None,
        ),
        "padded with a mix": (
            file_bytes('{"a":' + U8_8 + "}" + " \t\n\r" * 4, D8, pad=False),
            None,
        ),
        "whitespace before the brace": (file_bytes(" \t\r\n{" + one + "}", D8), None),
        "whitespace between tokens": (
            file_bytes(
                '{ "a" :\n{ "dtype" : "U8" ,\t"shape" : [ 8 ] , "data_offsets" : [ 0 , 8 ] } }',
                D8,
            ),
            None,
        ),
        "whitespace in an empty array and object": (
            file_bytes('{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"x":[ ],"y":{ }}}', D8),
            None,
        ),
        "data unaligned for its dtype": (
            file_bytes(
                json.dumps({"a": T("U8", [1], 0, 1), "b": T("F32", [1], 1, 5)}), bytes(range(5))
            ),
            None,
        ),
        "offsets out of key order": (
            file_bytes(json.dumps({"b": T("U8", [4], 4, 8), "a": T("U8", [4], 0, 4)}), D8),
            None,
        ),
        "an unknown field with a nested array": (
            tensor_with('[1,[2,[3,[]]],{"a":{"b":[{}]}}]'),
            None,
        ),
        "an unknown field with an object": (tensor_with('{"k":{"j":{"i":"h"}},"l":[]}'), None),
        "an unknown field, true": (tensor_with("true"), None),
        "an unknown field, false": (tensor_with("false"), None),
        "an unknown field, null": (tensor_with("null"), None),
        "an unknown field, numbers": (
            tensor_with("[0,-1,1.5,-2.5e-3,1E+2,1e2,0.0,123456789012345678901234567890]"),
            None,
        ),
        "an unknown field, a string with escapes": (
            tensor_with('"a\\n\\u00e9\\ud83d\\ude00 \\/"'),
            None,
        ),
        "unknown fields before and after the known ones": (
            file_bytes(
                '{"a":{"p":1,"dtype":"U8","q":[2],"shape":[8],"r":null,'
                '"data_offsets":[0,8],"s":{}}}',
                D8,
            ),
            None,
        ),
        "an unknown field twice": (
            file_bytes(
                '{"a":{"dtype":"U8","x":1,"shape":[8],"x":"two","data_offsets":[0,8]}}', D8
            ),
            None,
        ),
        "an unknown field named like a top-level one": (tensor_with("1", key="__metadata__"), None),
        "an escaped dtype key": (
            file_bytes('{"a":{"dt\\u0079pe":"U8","shape":[8],"data_offsets":[0,8]}}', D8),
            None,
        ),
        "an escaped shape key": (
            file_bytes('{"a":{"dtype":"U8","sh\\u0061pe":[8],"data_offsets":[0,8]}}', D8),
            None,
        ),
        "an escaped data_offsets key": (
            file_bytes('{"a":{"dtype":"U8","shape":[8],"data\\u005foffsets":[0,8]}}', D8),
            None,
        ),
        "an escaped dtype value": (
            file_bytes('{"a":{"dtype":"\\u0055\\u0038","shape":[8],"data_offsets":[0,8]}}', D8),
            None,
        ),
        "the metadata key spelled with an escape": (
            file_bytes('{"\\u005f_metadata__":{"k":"v"},' + one + "}", D8),
            None,
        ),
        "shape with leading zero-free big numbers": (
            file_bytes('{"a":{"dtype":"U8","shape":[1,2,4],"data_offsets":[0,8]}}', D8),
            None,
        ),
        "a dtype name escaped letter by letter": (
            file_bytes(
                '{"a":{"\\u0064type":"\\u0046\\u0033\\u0032","shape":[2],"data_offsets":[0,8]}}', D8
            ),
            None,
        ),
    }


@pytest.mark.parametrize("case", list(_layout_cases()))
def test_layouts_python_never_writes(flint, tmp_path, case):
    blob, _ = _layout_cases()[case]
    path = write_blob(tmp_path, blob)
    accepted_by_both(flint, path, blob)


def test_ranged_reads_of_a_9_byte_tensor(flint, tmp_path):
    payload = bytes(range(10, 19))
    blob = hand_built([("t", "U8", (9,), payload)])
    path = write_blob(tmp_path, blob)
    with SafetensorsReader(flint, path) as reader:
        for offset in range(10):
            for nbytes in range(10 - offset):
                assert reader.read(0, offset, nbytes) == payload[offset : offset + nbytes], (
                    offset,
                    nbytes,
                )
        assert reader.read(0, 0, 9) == payload


@pytest.mark.parametrize("offset", range(10))
def test_ranged_read_from_each_offset_of_a_9_byte_tensor(flint, tmp_path, offset):
    payload = bytes(range(100, 109))
    blob = hand_built([("pad", "U8", (3,), b"abc"), ("t", "U8", (9,), payload)])
    path = write_blob(tmp_path, blob)
    with SafetensorsReader(flint, path) as reader:
        index = reader.find("t")
        for nbytes in range(10 - offset):
            assert reader.read(index, offset, nbytes) == payload[offset : offset + nbytes], nbytes


def test_200_random_ranges_of_a_1_MiB_tensor(flint, tmp_path):
    payload = random.Random(1).randbytes(MiB)
    blob = hand_built(
        [
            ("head", "U8", (5,), b"12345"),
            ("big", "U8", (MiB,), payload),
            ("tail", "F32", (1,), b"abcd"),
        ]
    )
    path = write_blob(tmp_path, blob)
    rng = random.Random(2)
    ranges = []
    for i in range(200):
        offset = rng.randrange(0, MiB + 1)
        room = MiB - offset
        if i % 4 == 0:
            nbytes = rng.randrange(0, min(room, 64) + 1)
        elif i % 4 == 1:
            nbytes = rng.randrange(0, min(room, 4096) + 1)
        else:
            nbytes = rng.randrange(0, room + 1)
        ranges.append((offset, nbytes))
    with SafetensorsReader(flint, path) as reader:
        index = reader.find("big")
        for offset, nbytes in ranges:
            assert reader.read(index, offset, nbytes) == payload[offset : offset + nbytes], (
                offset,
                nbytes,
            )


def test_reads_that_span_the_whole_tensor_and_zero_byte_reads_with_a_null_dst(flint, tmp_path):
    payloads = [make_payload("U8", n, seed=n) for n in (1, 0, 4096, 70_001)]
    blob = hand_built([(f"t{i}", "U8", (len(p),), p) for i, p in enumerate(payloads)])
    path = write_blob(tmp_path, blob)
    with SafetensorsReader(flint, path) as reader:
        for index, payload in enumerate(payloads):
            assert reader.read(index) == payload
            assert reader.read(index, 0, len(payload)) == payload
            # A zero-byte read takes a null dst, at the start, the middle and the end.
            for offset in {0, len(payload) // 2, len(payload)}:
                status = flint.lib.flint_safetensors_read(reader.handle, index, offset, 0, None)
                assert status == OK, flint.last_error()
            # And a dst that isn't null is left alone.
            guard = ctypes.create_string_buffer(b"\xaa" * 8, 8)
            status = flint.lib.flint_safetensors_read(
                reader.handle, index, len(payload), 0, guard
            )
            assert status == OK and guard.raw == b"\xaa" * 8


def test_a_tensor_larger_than_the_chunk_of_one_read_is_read_in_pieces(flint, tmp_path):
    payload = random.Random(3).randbytes(3 * MiB + 17)
    blob = hand_built([("t", "U8", (len(payload),), payload)])
    path = write_blob(tmp_path, blob)
    with SafetensorsReader(flint, path) as reader:
        pieces = [
            reader.read(0, o, min(MiB, len(payload) - o)) for o in range(0, len(payload), MiB)
        ]
        assert b"".join(pieces) == payload
        assert reader.read(0) == payload


# =============================================================================================
# 2. OpenAI's file
# =============================================================================================

OPENAI_FILE = "gpt2-124M/model.safetensors"
OPENAI_BYTES = 548_105_171
OPENAI_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"


@pytest.fixture(scope="module")
def openai_path():
    """The pinned file, or a skip when it isn't in data/ or ~/flint-data/."""
    path = data_file(OPENAI_FILE)
    if path is None:
        pytest.skip(f"{OPENAI_FILE} not found in data/ or ~/flint-data/")
    assert path.stat().st_size == OPENAI_BYTES
    assert sha256_of(path) == OPENAI_SHA256
    return path


@pytest.fixture(scope="module")
def openai_header(openai_path):
    """(n, the header's bytes, the test's own parse of it)."""
    with open(openai_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        text = f.read(n)
    tensors, metadata = header_entries(text)
    return n, text, tensors, metadata


def test_openai_infos_equal_the_header(flint, openai_path, openai_header):
    n, text, tensors, metadata = openai_header
    assert len(tensors) == 160
    assert metadata == [("format", "pt")]
    with SafetensorsReader(flint, openai_path) as reader:
        assert reader.count() == 160
        for index, (name, dtype, shape, begin, end) in enumerate(tensors):
            expected = SafetensorsInfo(name, dtype, shape, DTYPE_BITS[dtype], begin, end - begin)
            assert reader.info(index) == expected
            assert reader.find(name) == index
        assert reader.metadata_pairs() == [("format", "pt")]
        assert reader.metadata_find("format") == b"pt"
        assert reader.metadata_count() == 1


def test_openai_file_has_the_cases_plan_md_names(flint, openai_path, openai_header):
    n, text, tensors, metadata = openai_header
    assert 8 + n == 14_291
    with SafetensorsReader(flint, openai_path) as reader:
        infos = reader.infos()
        # The data starts at byte 14,291, which isn't a multiple of 4, so no tensor starts at a
        # file offset that is.
        assert all((8 + n + info.offset) % 4 != 0 for info in infos)
        offsets = [info.offset for info in infos]
        assert offsets != sorted(offsets)
        bias = reader.info(reader.find("h.0.attn.bias"))
        assert bias.shape == (1, 1, 1024, 1024)
        assert bias.dtype == "F32" and bias.bits == 32
        assert bias.bytes == 4 * 1024 * 1024
    assert python_verdict(openai_path).accepted


def test_openai_bytes_equal_pythons_in_chunks_of_64_MiB(flint, openai_path):
    chunk = 64 * MiB
    with SafetensorsReader(flint, openai_path) as reader, safe_open(
        str(openai_path), framework="pt"
    ) as reference:
        names = [info.name for info in reader.infos()]
        assert sorted(names) == sorted(reference.keys())
        for index, name in enumerate(names):
            expected = reference.get_tensor(name).contiguous().reshape(-1).view(torch.uint8).numpy()
            assert reader.info(index).bytes == expected.nbytes, name
            for offset in range(0, expected.nbytes, chunk):
                nbytes = min(chunk, expected.nbytes - offset)
                got = reader.read(index, offset, nbytes)
                assert got == expected[offset : offset + nbytes].tobytes(), (name, offset)
            del expected
        gc.collect()


# =============================================================================================
# 3. Verdicts against Python
# =============================================================================================

# The listed differences (Python accepts, Flint rejects, except where it says otherwise): a
# duplicate tensor name, a duplicate metadata key, and \u0000 in a tensor name, a metadata key or
# a metadata value; and, the other way, the largest FP64 value's last digit in an unknown field.
LISTED_DIFFERENCES = set()

CASES = []


def case(name, blob, python_accepts, flint_verdict=None, listed=False):
    """A hand-built file with the verdicts Python's package and Flint must give: Flint's is OK
    for an accepted file and FORMAT for a rejected one, unless flint_verdict says otherwise."""
    if flint_verdict is None:
        flint_verdict = OK if python_accepts else ERROR_FORMAT
    if listed:
        assert python_accepts != (flint_verdict == OK)
        LISTED_DIFFERENCES.add(name)
    else:
        assert python_accepts == (flint_verdict == OK), name
    CASES.append(pytest.param(blob, python_accepts, flint_verdict, id=name))


def built(tensors, data=D8, pad=True, extra=""):
    return file_bytes(json.dumps(tensors) + extra, data, pad)


# From probe_python.py.
case("p1 valid F32[2]", built({"a": T("F32", [2], 0, 8)}), True)
case("p1 unpadded header", built({"a": T("F32", [2], 0, 8)}, pad=False), True)
case(
    "p1 trailing tab padding",
    file_bytes(json.dumps({"a": T("F32", [2], 0, 8)}) + "\t", D8, False),
    True,
)
case("p1 keys out of offset order", built({"b": T("U8", [4], 4, 8), "a": T("U8", [4], 0, 4)}), True)
case(
    "p1 unaligned F32 after U8[1]",
    built({"a": T("U8", [1], 0, 1), "b": T("F32", [1], 1, 5)}, D8[:5]),
    True,
)
case("p1 hole between tensors", built({"a": T("U8", [2], 0, 2), "b": T("U8", [4], 4, 8)}), False)
case("p1 overlap", built({"a": T("U8", [4], 0, 4), "b": T("U8", [4], 2, 6)}, D8[:6]), False)
case("p1 trailing bytes after data", built({"a": T("U8", [4], 0, 4)}), False)
case("p1 data shorter than offsets", built({"a": T("U8", [8], 0, 8)}, D8[:6]), False)
case("p1 shape and bytes disagree", built({"a": T("F32", [3], 0, 8)}), False)
case(
    "p1 duplicate key, the last leaves a hole",
    file_bytes(
        '{"a":' + json.dumps(T("U8", [4], 0, 4)) + ',"a":' + json.dumps(T("U8", [4], 4, 8)) + "}",
        D8,
    ),
    False,
)
case("p1 unknown field", built({"a": {**T("U8", [8], 0, 8), "extra": 1}}), True)
case("p1 unknown dtype", built({"a": T("Q8", [8], 0, 8)}), False)
case("p1 rank 0", built({"a": T("F32", [], 0, 4)}, D8[:4]), True)
case("p1 zero-size tensor", built({"a": T("F32", [0, 3], 0, 0), "b": T("U8", [8], 0, 8)}), True)
case(
    "p1 two zero-size at one offset",
    built({"a": T("F32", [0], 0, 0), "b": T("F32", [0], 0, 0)}, b""),
    True,
)
case(
    "p1 metadata value not a string",
    built({"__metadata__": {"k": 1}, "a": T("U8", [8], 0, 8)}),
    False,
)
case(
    "p1 metadata with unicode and escapes",
    built({"__metadata__": {"ké": 'v\n"中'}, "a": T("U8", [8], 0, 8)}),
    True,
)
case("p1 raw UTF-8 key", file_bytes('{"é":' + json.dumps(T("U8", [8], 0, 8)) + "}", D8), True)
case("p1 negative offset", built({"a": T("U8", [8], -1, 7)}), False)
case("p1 float in shape", built({"a": T("U8", [8.0], 0, 8)}), False)
case("p1 begin after end", built({"a": T("U8", [0], 8, 0)}), False)
case(
    "p1 offsets with 3 entries",
    built({"a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8, 8]}}),
    False,
)
case("p1 header not an object", file_bytes("[]", b""), False)
case("p1 empty header object", file_bytes("{}", b""), True)
case("p1 header length past the file", struct.pack("<Q", 1 << 20) + b"{}", False)
case("p1 header length 100,000,001", struct.pack("<Q", 100_000_001) + b"{}", False)
case("p1 space before the brace", file_bytes(" " + json.dumps({"a": T("U8", [8], 0, 8)}), D8), True)
case("p1 BOM", file_bytes("﻿" + json.dumps({"a": T("U8", [8], 0, 8)}), D8), False)
case("p1 shape overflow", built({"a": T("U8", [2**32, 2**32], 0, 8)}), False)
case("p1 BOOL", built({"a": T("BOOL", [8], 0, 8)}), True)
case("p1 F8_E4M3", built({"a": T("F8_E4M3", [8], 0, 8)}), True)
case("p1 F4 [16]", built({"a": T("F4", [16], 0, 8)}), True)
case("p1 __metadata__ not an object", built({"__metadata__": "x", "a": T("U8", [8], 0, 8)}), False)
case(
    "p1 trailing garbage after the brace",
    file_bytes(json.dumps({"a": T("U8", [8], 0, 8)}) + "x", D8),
    False,
)
case("p1 NUL padding", file_bytes(json.dumps({"a": T("U8", [8], 0, 8)}) + "\0", D8, False), False)
# Python keeps the last of two equal keys, so these are valid for it and not for Flint.
case(
    "dup tensor name, the last one fits",
    file_bytes(
        '{"a":' + json.dumps(T("U8", [2], 0, 2)) + ',"a":' + json.dumps(T("U8", [4], 0, 4)) + "}",
        D8[:4],
    ),
    True,
    ERROR_FORMAT,
    listed=True,
)
case(
    "dup tensor name, identical",
    file_bytes('{"a":' + U8_8 + ',"a":' + U8_8 + "}", D8),
    True, ERROR_FORMAT, listed=True,
)  # fmt: skip
case(
    "dup tensor name around another",
    file_bytes(
        (
            '{"a":'
            + json.dumps(T("U8", [4], 0, 4))
            + ',"b":'
            + json.dumps(T("U8", [4], 4, 8))
            + ',"a":'
            + json.dumps(T("U8", [4], 0, 4))
            + "}"
        ),
        D8,
    ),
    True,
    ERROR_FORMAT,
    listed=True,
)

# From probe_python2.py: header lengths.
case("p2 length 0", struct.pack("<Q", 0), False)
case("p2 length 0 with data", struct.pack("<Q", 0) + D8, False)
case("p2 length 1 `{`", struct.pack("<Q", 1) + b"{", False)
case("p2 length 2 `{}`", struct.pack("<Q", 2) + b"{}", True)
case("p2 file shorter than 8", b"\x02\x00\x00", False)
case("p2 length 2^64-1", struct.pack("<Q", 2**64 - 1) + b"{}", False)
case("p2 empty file", b"", False)
# Whitespace.
case("p2 CR LF around", file_bytes("\r\n" + '{"a":' + U8_8 + "}" + "\r\n", D8), True)
case("p2 form feed after", file_bytes('{"a":' + U8_8 + "}\f", D8), False)
case("p2 vertical tab after", file_bytes('{"a":' + U8_8 + "}\v", D8), False)
case("p2 NBSP after", file_bytes('{"a":' + U8_8 + "} ", D8), False)
# Numbers in shape and offsets.
BAD_SHAPE = '{"a":{"dtype":"U8","shape":%s,"data_offsets":[0,8]}}'
case(
    "p2 shape -0",
    file_bytes('{"a":{"dtype":"U8","shape":[-0,8],"data_offsets":[0,0]}}', b""),
    False,
)
case("p2 shape 08", file_bytes(BAD_SHAPE % "[08]", D8), False)
case("p2 shape 8e0", file_bytes(BAD_SHAPE % "[8e0]", D8), False)
case("p2 shape 2^64", file_bytes(BAD_SHAPE % "[18446744073709551616]", D8), False)
case(
    "p2 shape 2^64-1 and 0",
    file_bytes('{"a":{"dtype":"U8","shape":[18446744073709551615,0],"data_offsets":[0,0]}}', b""),
    True,
)
case("p2 shape 2^63 x 2", file_bytes(BAD_SHAPE % "[9223372036854775808,2]", D8), False)
case(
    "p2 shape 2^63, 2, 0",
    file_bytes('{"a":{"dtype":"U8","shape":[9223372036854775808,2,0],"data_offsets":[0,0]}}', b""),
    False,
)
case(
    "p2 shape 2^62 F32",
    file_bytes('{"a":{"dtype":"F32","shape":[4611686018427387904],"data_offsets":[0,8]}}', D8),
    False,
)
case("p2 shape 2^61 U8", file_bytes(BAD_SHAPE % "[2305843009213693952]", D8), False)
case(
    "p2 shape 0, 2^63, 4",
    file_bytes('{"a":{"dtype":"U8","shape":[0,9223372036854775808,4],"data_offsets":[0,0]}}', b""),
    True,
)
case(
    "p2 offsets end 2^64-1",
    file_bytes('{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,18446744073709551615]}}', D8),
    False,
)
case("p2 shape not an array", file_bytes(BAD_SHAPE % "8", D8), False)
case("p2 shape null", file_bytes(BAD_SHAPE % "null", D8), False)
case(
    "p2 dtype lowercase",
    file_bytes('{"a":{"dtype":"u8","shape":[8],"data_offsets":[0,8]}}', D8),
    False,
)
case("p2 missing shape", file_bytes('{"a":{"dtype":"U8","data_offsets":[0,8]}}', D8), False)
case(
    "p2 dtype twice",
    file_bytes('{"a":{"dtype":"U8","dtype":"U8","shape":[8],"data_offsets":[0,8]}}', D8),
    False,
)
case("p2 tensor not an object", file_bytes('{"a":5}', b""), False)
case("p2 tensor null", file_bytes('{"a":null}', b""), False)
# Metadata.
case("p2 __metadata__ null", file_bytes('{"__metadata__":null,"a":' + U8_8 + "}", D8), True)
case("p2 __metadata__ empty", file_bytes('{"__metadata__":{},"a":' + U8_8 + "}", D8), True)
case(
    "p2 __metadata__ twice",
    file_bytes('{"__metadata__":{"k":"1"},"__metadata__":{"k":"2"},"a":' + U8_8 + "}", D8),
    False,
)
case(
    "p2 metadata duplicate key",
    file_bytes('{"__metadata__":{"k":"1","k":"2"},"a":' + U8_8 + "}", D8),
    True,
    ERROR_FORMAT,
    listed=True,
)
case(
    "p2 metadata null value",
    file_bytes('{"__metadata__":{"k":null},"a":' + U8_8 + "}", D8),
    False,
)
case("p2 metadata last", file_bytes('{"a":' + U8_8 + ',"__metadata__":{"k":"v"}}', D8), True)
case("p2 only metadata, data 0", file_bytes('{"__metadata__":{"k":"v"}}', b""), True)
case(
    "p2 metadata value with an escaped \\u0041",
    file_bytes('{"__metadata__":{"k":"\\u0041"},"a":' + U8_8 + "}", D8),
    True,
)
case(
    "p2 metadata value, lone surrogate",
    file_bytes('{"__metadata__":{"k":"\\ud800"},"a":' + U8_8 + "}", D8),
    False,
)
# Strings and escapes in a name.
case(
    "p2 name with \\u0000",
    file_bytes('{"a\\u0000b":' + U8_8 + "}", D8),
    True,
    ERROR_FORMAT,
    listed=True,
)
case(
    "p2 metadata key with \\u0000",
    file_bytes('{"__metadata__":{"k\\u0000":"v"},"a":' + U8_8 + "}", D8),
    True,
    ERROR_FORMAT,
    listed=True,
)
case(
    "p2 metadata value with \\u0000",
    file_bytes('{"__metadata__":{"k":"v\\u0000"},"a":' + U8_8 + "}", D8),
    True,
    ERROR_FORMAT,
    listed=True,
)
case("p2 name empty", file_bytes('{"":' + U8_8 + "}", D8), True)
case("p2 name with \\/", file_bytes('{"a\\/b":' + U8_8 + "}", D8), True)
case("p2 name with \\x", file_bytes('{"a\\xb":' + U8_8 + "}", D8), False)
case("p2 name with a lone high surrogate", file_bytes('{"\\ud800":' + U8_8 + "}", D8), False)
case("p2 name with a lone low surrogate", file_bytes('{"\\udc00":' + U8_8 + "}", D8), False)
case("p2 name with a surrogate pair", file_bytes('{"\\ud83d\\ude00":' + U8_8 + "}", D8), True)
case("p2 name with a reversed pair", file_bytes('{"\\ude00\\ud83d":' + U8_8 + "}", D8), False)
case(
    "p2 name with a high surrogate then a letter",
    file_bytes('{"\\ud83dx":' + U8_8 + "}", D8),
    False,
)
case("p2 name with uppercase hex", file_bytes('{"\\u00E9":' + U8_8 + "}", D8), True)
case("p2 name with a short \\u", file_bytes('{"\\u00e":' + U8_8 + "}", D8), False)
case(
    "p2 name with a raw control character",
    file_bytes(b'{"a\x01":' + U8_8.encode() + b"}", D8),
    False,
)
case("p2 name with a raw DEL", file_bytes(b'{"a\x7f":' + U8_8.encode() + b"}", D8), True)
case("p2 name with a raw tab", file_bytes(b'{"a\tb":' + U8_8.encode() + b"}", D8), False)
case("p2 name with invalid UTF-8 0xff", file_bytes(b'{"a\xff":' + U8_8.encode() + b"}", D8), False)
case(
    "p2 name with an overlong UTF-8",
    file_bytes(b'{"a\xc0\xaf":' + U8_8.encode() + b"}", D8),
    False,
)
case(
    "p2 name with a UTF-8 surrogate",
    file_bytes(b'{"a\xed\xa0\x80":' + U8_8.encode() + b"}", D8),
    False,
)
case("p2 invalid UTF-8 in the padding", file_bytes(b'{"a":' + U8_8.encode() + b"}\xff", D8), False)
# Unknown fields: values the parser must skip.
case("p2 unknown nested object", tensor_with('{"y":[1,{"z":null}]}'), True)
case("p2 unknown true, false, null", tensor_with("[true,false,null]"), True)
case("p2 unknown 1.5e300", tensor_with("1.5e300"), True)
case("p2 unknown 1e999", tensor_with("1e999"), False)
case("p2 unknown -0.0", tensor_with("-0.0"), True)
case("p2 unknown 2^70", tensor_with("1180591620717411303424"), True)
case("p2 unknown 01", tensor_with("01"), False)
case("p2 unknown 1.", tensor_with("1."), False)
case("p2 unknown .5", tensor_with(".5"), False)
case("p2 unknown +1", tensor_with("+1"), False)
case("p2 unknown NaN", tensor_with("NaN"), False)
case("p2 unknown a lone surrogate string", tensor_with('"\\ud800"'), False)
case(
    "p2 unknown a raw control character",
    file_bytes(b'{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"x":"\x01"}}', D8),
    False,
)
case("p2 unknown an invalid escape", tensor_with('"\\q"'), False)
case("p2 unknown duplicate keys inside", tensor_with('{"k":1,"k":2}'), True)
case("p2 unknown a trailing comma", tensor_with("[1,]"), False)
case("p2 unknown a misspelled true", tensor_with("tru"), False)
case("p2 unknown a capital E exponent", tensor_with("1E5"), True)
case("p2 unknown a plus exponent", tensor_with("1e+5"), True)
case("p2 unknown -", tensor_with("-"), False)
case("p2 unknown 1e", tensor_with("1e"), False)
case("p2 unknown Infinity", tensor_with("Infinity"), False)
# Depth, the arrays and objects of probe_python2.py and probe_python3.py.
for depth in (124, 125, 126, 127, 128, 129):
    case(f"p2 unknown {depth} arrays deep", tensor_with("[" * depth + "]" * depth), depth <= 125)
for depth in (124, 125, 126, 127):
    case(
        f"p2 unknown {depth} objects deep",
        tensor_with('{"k":' * depth + "1" + "}" * depth),
        depth <= 125,
    )
# Where a tensor may go: zero-size placement.
Z4 = '{"dtype":"F32","shape":[0],"data_offsets":[%d,%d]}'
case("p2 zero-size at the end", file_bytes('{"a":' + U8_8 + ',"z":' + Z4 % (8, 8) + "}", D8), True)
case(
    "p2 zero-size inside a tensor",
    file_bytes('{"a":' + U8_8 + ',"z":' + Z4 % (4, 4) + "}", D8),
    False,
)
case(
    "p2 zero-size past the end",
    file_bytes('{"a":' + U8_8 + ',"z":' + Z4 % (9, 9) + "}", D8),
    False,
)
case(
    "p2 zero-size at 0, then a tensor",
    file_bytes('{"z":' + Z4 % (0, 0) + ',"a":' + U8_8 + "}", D8),
    True,
)
case("p2 only zero-size, data 0", file_bytes('{"z":' + Z4 % (0, 0) + "}", b""), True)
case("p2 only zero-size, data 8", file_bytes('{"z":' + Z4 % (0, 0) + "}", D8), False)
case("p2 no tensors, data 8", file_bytes("{}", D8), False)
# From probe_ties.log: Python sorts by (begin, end), so a zero-size tensor fits in any order.
A8 = '{"dtype":"U8","shape":[8],"data_offsets":[0,8]}'
A4 = '{"dtype":"U8","shape":[4],"data_offsets":[0,4]}'
B4 = '{"dtype":"U8","shape":[4],"data_offsets":[4,8]}'
Z8 = '{"dtype":"U8","shape":[0],"data_offsets":[%d,%d]}'
case("ties a(0,8) then z(0,0)", file_bytes('{"a":' + A8 + ',"z":' + Z8 % (0, 0) + "}", D8), True)
case("ties z(0,0) then a(0,8)", file_bytes('{"z":' + Z8 % (0, 0) + ',"a":' + A8 + "}", D8), True)
case(
    "ties a b z(4,4)",
    file_bytes('{"a":' + A4 + ',"b":' + B4 + ',"z":' + Z8 % (4, 4) + "}", D8),
    True,
)
case(
    "ties b a z(4,4)",
    file_bytes('{"b":' + B4 + ',"a":' + A4 + ',"z":' + Z8 % (4, 4) + "}", D8),
    True,
)
case(
    "ties big(0,8) then 20 zero-size at 0",
    file_bytes(
        '{"big":' + A8 + "," + ",".join(f'"z{i:02d}":' + Z8 % (0, 0) for i in range(20)) + "}",
        D8,
    ),
    True,
)
case(
    "ties 20 zero-size at 0 then big, names reversed",
    file_bytes(
        (
            "{"
            + ",".join(f'"z{i:02d}":' + Z8 % (0, 0) for i in reversed(range(20)))
            + ',"big":'
            + A8
            + "}"
        ),
        D8,
    ),
    True,
)
case(
    "ties a zero-size at 8 declared before a(0,8)",
    file_bytes('{"z":' + Z8 % (8, 8) + ',"a":' + A8 + "}", D8),
    True,
)
case(
    "ties a zero-size at 8 and one at 0 around a",
    file_bytes('{"z8":' + Z8 % (8, 8) + ',"a":' + A8 + ',"z0":' + Z8 % (0, 0) + "}", D8),
    True,
)
case(
    "ties a zero-size at 3 inside a(0,8)",
    file_bytes('{"a":' + A8 + ',"z":' + Z8 % (3, 3) + "}", D8),
    False,
)
case(
    "ties a zero-size at 9 after a(0,8)",
    file_bytes('{"a":' + A8 + ',"z":' + Z8 % (9, 9) + "}", D8),
    False,
)
# From probe_forms.py: serde's other spellings of a tensor entry, a 3-element array and a dtype
# written as a one-key map. Each case is (name, header, data, Python accepts it).
FORM_CASES = [
    ("array", '{"a":["U8",[8],[0,8]]}', D8, True),
    ("array, spaces", '{"a": [ "U8" , [ 8 ] , [ 0 , 8 ] ] }', D8, True),
    ("array, rank 0", '{"a":["F32",[],[0,4]]}', D8[:4], True),
    ("array, F4", '{"a":["F4",[16],[0,8]]}', D8, True),
    ("array, with metadata", '{"__metadata__":{"k":"v"},"a":["U8",[8],[0,8]]}', D8, True),
    (
        "array and object mixed",
        '{"a":["U8",[4],[0,4]],"b":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}',
        D8,
        True,
    ),
    ("array, 2 elements", '{"a":["U8",[8]]}', D8, False),
    ("array, 4 elements", '{"a":["U8",[8],[0,8],1]}', D8, False),
    ("array, empty", '{"a":[]}', b"", False),
    ("array, 1 element", '{"a":["U8"]}', D8, False),
    ("array, trailing comma", '{"a":["U8",[8],[0,8],]}', D8, False),
    ("array, wrong order", '{"a":[[8],"U8",[0,8]]}', D8, False),
    ("array, offsets 3", '{"a":["U8",[8],[0,8,8]]}', D8, False),
    ("array, bad dtype", '{"a":["Q8",[8],[0,8]]}', D8, False),
    ("array, hole", '{"a":["U8",[4],[0,4]],"b":["U8",[2],[6,8]]}', D8, False),
    ("array, shape mismatch", '{"a":["F32",[3],[0,8]]}', D8, False),
    ("array, dtype null", '{"a":[null,[8],[0,8]]}', D8, False),
    ("array, nested deeper", '{"a":[["U8"],[8],[0,8]]}', D8, False),
    ("array as metadata", '{"__metadata__":[["k","v"]],"a":["U8",[8],[0,8]]}', D8, False),
    ("array, NUL name", '{"a\\u0000":["U8",[8],[0,8]]}', D8, True),
    ("map dtype", '{"a":{"dtype":{"U8":null},"shape":[8],"data_offsets":[0,8]}}', D8, True),
    (
        "map dtype, spaces",
        '{"a":{"dtype": { "U8" : null } ,"shape":[8],"data_offsets":[0,8]}}',
        D8,
        True,
    ),
    (
        "map dtype, escaped key",
        '{"a":{"dtype":{"\\u0055\\u0038":null},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        True,
    ),
    ("map dtype in array", '{"a":[{"F32":null},[2],[0,8]]}', D8, True),
    (
        "map dtype, two keys",
        '{"a":{"dtype":{"U8":null,"I8":null},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        "map dtype, same key twice",
        '{"a":{"dtype":{"U8":null,"U8":null},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    ("map dtype, empty", '{"a":{"dtype":{},"shape":[8],"data_offsets":[0,8]}}', D8, False),
    (
        "map dtype, value []",
        '{"a":{"dtype":{"U8":[]},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        "map dtype, value {}",
        '{"a":{"dtype":{"U8":{}},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        "map dtype, value 1",
        '{"a":{"dtype":{"U8":1},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        'map dtype, value ""',
        '{"a":{"dtype":{"U8":""},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        "map dtype, unknown",
        '{"a":{"dtype":{"Q8":null},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        "map dtype, trailing comma",
        '{"a":{"dtype":{"U8":null,},"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        "dtype as array",
        '{"a":{"dtype":["U8"],"shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        "dtype twice, map and string",
        '{"a":{"dtype":{"U8":null},"dtype":"U8","shape":[8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    ("shape as map", '{"a":{"dtype":"U8","shape":{"0":8},"data_offsets":[0,8]}}', D8, False),
    (
        "offsets as map",
        '{"a":{"dtype":"U8","shape":[8],"data_offsets":{"0":0,"1":8}}}',
        D8,
        False,
    ),
    ("top-level array of pairs", '[["a",["U8",[8],[0,8]]]]', D8, False),
    (
        "metadata null in array world",
        '{"__metadata__":null,"a":["U8",[8],[0,8]]}',
        D8,
        True,
    ),
    # Duplicate and missing fields (the object form), and a missing dtype as a short array.
    (
        "shape twice",
        '{"a":{"dtype":"U8","shape":[8],"shape":[],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    (
        "data_offsets twice",
        '{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"data_offsets":[0,8]}}',
        D8,
        False,
    ),
    ("missing dtype", '{"a":{"shape":[8],"data_offsets":[0,8]}}', D8, False),
    ("missing dtype, short array", '{"a":[[8],[0,8]]}', D8, False),
    ("missing shape, short array", '{"a":["U8",[0,8]]}', D8, False),
]
for form_name, form_header, form_data, form_accepts in FORM_CASES:
    if form_name == "array, NUL name":  # the listed difference: NUL in a tensor name
        case("forms " + form_name, file_bytes(form_header, form_data), True, ERROR_FORMAT, True)
    else:
        case("forms " + form_name, file_bytes(form_header, form_data), form_accepts)

# Integers that would wrap: 2^64 + k read as k makes a consistent file, but serde reads an
# integer at or past 2^64 as a float, so Python rejects them, and so must Flint. (A reader
# that drops its overflow check on an unsigned integer accepts these.)
WRAP = '{"a":{"dtype":"U8","shape":%s,"data_offsets":%s}}'
case("wrap shape 2^64 + 8", file_bytes(WRAP % ("[18446744073709551624]", "[0,8]"), D8), False)
case("wrap offset end 2^64 + 8", file_bytes(WRAP % ("[8]", "[0,18446744073709551624]"), D8), False)
case("wrap offset begin 2^64", file_bytes(WRAP % ("[8]", "[18446744073709551616,8]"), D8), False)
case("wrap shape 2^64 x 10", file_bytes(WRAP % ("[184467440737095516160]", "[0,0]"), b""), False)
case("wrap shape 2^64 x 10 + 8", file_bytes(WRAP % ("[184467440737095516168]", "[0,8]"), D8), False)
case(
    "wrap the array form",
    file_bytes('{"a":["U8",[18446744073709551624],[0,8]]}', D8),
    False,
)
case("wrap shape 2^65 + 8", file_bytes(WRAP % ("[36893488147419103240]", "[0,8]"), D8), False)


# Number range in an unknown field, in a one-line file each (more are parametrized below).
case("range 1e308", tensor_with("1e308"), True)
case("range 1e309", tensor_with("1e309"), False)
# The one band difference (Python rejects, Flint accepts): below H, within serde_json's error.
case("band 1.7976931348623158e308", tensor_with("1.7976931348623158e308"), False, OK, listed=True)
case("band 1.797693134862316e308", tensor_with("1.797693134862316e308"), False, ERROR_FORMAT)


@pytest.mark.parametrize("blob, python_accepts, flint_verdict", CASES)
def test_verdict_table(flint, tmp_path, blob, python_accepts, flint_verdict):
    path = write_blob(tmp_path, blob)
    verdict = python_verdict(path)
    assert verdict.accepted == python_accepts, verdict.message
    status = flint_status(flint, path)
    assert status == flint_verdict, flint.last_error()
    if status != OK:
        assert flint.last_error() != ""
    if verdict.accepted and status == OK:
        check_matches_python(flint, path, verdict)
        check_reads(flint, path, blob)


def test_the_two_verdicts_agree_on_every_case_but_the_listed_differences():
    assert len(CASES) > 120
    for param in CASES:
        blob, python_accepts, flint_verdict = param.values
        name = param.id
        agree = python_accepts == (flint_verdict == OK)
        assert agree != (name in LISTED_DIFFERENCES), name
    # The five listed rules and the band, each with a case that holds it.
    for needle in ("duplicate", "dup tensor", "\\u0000", "band 1.797"):
        assert any(needle in name for name in LISTED_DIFFERENCES), needle


@pytest.mark.parametrize("order", list(permutations("abz")), ids="".join)
def test_a_zero_size_tensor_at_a_boundary_in_any_declaration_order(flint, tmp_path, order):
    entries = {"a": A4, "b": B4, "z": Z8 % (4, 4)}
    header = "{" + ",".join(f'"{name}":{entries[name]}' for name in order) + "}"
    blob = file_bytes(header, D8)
    path = write_blob(tmp_path, blob)
    accepted_by_both(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        assert [i.name for i in reader.infos()] == list(order)


# --- the array form of a tensor and the map form of a dtype --------------------------------------


def spellings(dtype, shape, begin, end):
    """The ways one tensor entry can be written: the object form and the array form, each with
    the dtype as a string and as a one-key map."""
    shape_text = json.dumps(list(shape))
    return {
        "object": f'{{"dtype":"{dtype}","shape":{shape_text},"data_offsets":[{begin},{end}]}}',
        "object, map dtype": (
            f'{{"dtype":{{"{dtype}":null}},"shape":{shape_text},"data_offsets":[{begin},{end}]}}'
        ),
        "array": f'["{dtype}",{shape_text},[{begin},{end}]]',
        "array, map dtype": f'[{{"{dtype}":null}},{shape_text},[{begin},{end}]]',
    }


def test_a_file_that_mixes_both_tensor_forms_and_both_dtype_forms(flint, tmp_path):
    header = (
        '{"o1":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},'
        '"o2":{"dtype":{"I16":null},"shape":[2],"data_offsets":[4,8]},'
        '"a1":["F32",[2],[8,16]],'
        '"a2":[{"U8":null},[3],[16,19]],'
        '"__metadata__":{"k":"v","e":"\\u00e9"},'
        '"a3":[ {"F6_E3M2" : null} , [ 4 , 2 ] , [ 19 , 25 ] ],'
        '"o3":{"shape":[1,1],"data_offsets":[25,27],"dtype":{"BF16":null}}}'
    )
    blob = file_bytes(header, bytes(range(27)))
    path = write_blob(tmp_path, blob)
    accepted_by_both(flint, path, blob)  # every info, metadata pair and byte against the header
    with SafetensorsReader(flint, path) as reader:
        assert [(i.name, i.dtype, i.shape, i.bits) for i in reader.infos()] == [
            ("o1", "U8", (4,), 8),
            ("o2", "I16", (2,), 16),
            ("a1", "F32", (2,), 32),
            ("a2", "U8", (3,), 8),
            ("a3", "F6_E3M2", (4, 2), 6),
            ("o3", "BF16", (1, 1), 16),
        ]
        assert reader.read(reader.find("a1")) == bytes(range(8, 16))
        assert reader.read(reader.find("a3")) == bytes(range(19, 25))
        assert reader.read(reader.find("o3")) == bytes([25, 26])
        assert reader.metadata_pairs() == [("k", "v"), ("e", "é")]
    # Python reads the same dtypes and shapes.
    verdict = python_verdict(path)
    assert verdict.tensors["a3"] == ("F6_E3M2", (4, 2))
    assert verdict.tensors["o2"] == ("I16", (2,))


@pytest.mark.parametrize("dtype", list(DTYPE_BITS))
def test_every_dtype_in_the_array_form(flint, tmp_path, dtype):
    shape = SUBBYTE_SHAPES.get(dtype, (3, 5))
    nbytes = element_bytes(dtype, shape)
    payload = make_payload(dtype, nbytes, seed=hash_seed(dtype) + 1)
    for spelling in ("array", "array, map dtype", "object, map dtype"):
        header = '{"head":["U8",[1],[0,1]],"x":' + spellings(dtype, shape, 1, 1 + nbytes)[spelling]
        blob = file_bytes(header + "}", b"\x07" + payload)
        path = write_blob(tmp_path, blob, spelling.replace(", ", "_").replace(" ", "_") + ".st")
        verdict = python_verdict(path)
        assert verdict.accepted, (spelling, verdict.message)
        assert verdict.tensors["x"] == (dtype, tuple(shape)), spelling
        check_matches_python(flint, path, verdict)
        check_reads(flint, path, blob)
        with SafetensorsReader(flint, path) as reader:
            info = reader.info(reader.find("x"))
            assert info == SafetensorsInfo("x", dtype, shape, DTYPE_BITS[dtype], 1, nbytes)
            assert reader.read(reader.find("x")) == payload, spelling


MAP_KEYS = {
    "\\u0055\\u0038": "U8",
    "U\\u0038": "U8",
    "\\u0046\\u0033\\u0032": "F32",
    "F6\\u005fE2M3": "F6_E2M3",
    "\\u0042\\u004f\\u004f\\u004c": "BOOL",
    "F8_E4M3\\u0046NUZ": "F8_E4M3FNUZ",
}


@pytest.mark.parametrize("key, dtype", list(MAP_KEYS.items()), ids=list(MAP_KEYS))
@pytest.mark.parametrize("form", ["object", "array"])
def test_a_map_dtype_with_an_escaped_key(flint, tmp_path, form, key, dtype):
    nbytes = element_bytes(dtype, (8,))
    dtype_text = '{ "' + key + '" : null }'
    if form == "object":
        header = '{"a":{"dtype":' + dtype_text + ',"shape":[8],"data_offsets":[0,%d]}}' % nbytes
    else:
        header = '{"a":[' + dtype_text + ",[8],[0,%d]]}" % nbytes
    blob = file_bytes(header, bytes(range(nbytes)))
    path = write_blob(tmp_path, blob)
    accepted_by_both(flint, path, blob)
    with SafetensorsReader(flint, path) as reader:
        assert reader.info(0).dtype == dtype


# --- number range, in an unknown field ---------------------------------------------------------

H = (1 << 1024) - (1 << 970)

IN_RANGE = {
    "1e308": "1e308",
    "1.7976931348623157e308": "1.7976931348623157e308",
    "1e-999": "1e-999",
    "0e99999999999999999999": "0e99999999999999999999",
    "-0.0": "-0.0",
    "309-digit integer": "1" + "0" * 308,
    "400-digit 0.000...01": "0." + "0" * 398 + "1",
    "1e-99999999999999999999": "1e-99999999999999999999",
    "0 with a long fraction": "0." + "0" * 5000,
    "a long mantissa under H": "1." + "7" * 400 + "e307",
    "1e0000000000000000000000000308": "1e0000000000000000000000000308",
    "5e-324": "5e-324",
}
OUT_OF_RANGE = {
    "1e309": "1e309",
    "-1e999": "-1e999",
    "310-digit integer": "1" + "0" * 309,
    "1e99999999999999999999": "1e99999999999999999999",
    "-1e99999999999999999999": "-1e99999999999999999999",
    "0.1e310": "0.1e310",
    "a 400-digit integer": "9" * 400,
    "1e+309": "1e+309",
}


@pytest.mark.parametrize("text", list(IN_RANGE.values()), ids=list(IN_RANGE))
def test_numbers_in_range_in_an_unknown_field(flint, tmp_path, text):
    blob = tensor_with(text)
    path = write_blob(tmp_path, blob)
    assert python_verdict(path).accepted
    assert flint_status(flint, path) == OK, flint.last_error()


@pytest.mark.parametrize("text", list(OUT_OF_RANGE.values()), ids=list(OUT_OF_RANGE))
def test_numbers_out_of_range_in_an_unknown_field(flint, tmp_path, text):
    path = write_blob(tmp_path, tensor_with(text))
    rejected_by_both(flint, path)
    assert "range" in flint.last_error()


def test_H_is_the_309_digit_string_of_the_spec():
    assert str(H) == (
        "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649"
        "017977587207096330286416692887910946555547851940402630657488671505820681908902000708383"
        "676273854845817711531764475730270069855571366959622842914819860834936475292719074168444"
        "365510704342711559699508093042880177904174497792"
    )
    assert len(str(H)) == 309


@pytest.mark.parametrize("delta, flint_accepts", [(-1, True), (0, False), (1, False)])
def test_the_midpoint_H_written_as_an_integer(
    flint, tmp_path, record_property, delta, flint_accepts
):
    path = write_blob(tmp_path, tensor_with(str(H + delta)))
    status = flint_status(flint, path)
    assert status == (OK if flint_accepts else ERROR_FORMAT), flint.last_error()
    # Python's verdict is recorded, not asserted: all three are in the band where serde_json's
    # float parsing isn't correctly rounded.
    verdict = python_verdict(path)
    record_property(f"python_accepts_H{delta:+d}", verdict.accepted)


def test_numbers_around_H_written_as_decimals(flint, tmp_path):
    # Exactly the 309 digits of H, with a fraction or an exponent instead of an integer.
    digits = str(H)
    for text, accepts in [
        (digits[:1] + "." + digits[1:] + "e308", False),
        (digits[:1] + "." + digits[1:-1] + str(int(digits[-1]) - 1) + "e308", True),
        ("0." + digits + "e309", False),
        ("0." + str(H - 1) + "e309", True),
        (digits + ".0", False),
        (str(H - 1) + ".9999", True),
        ("0.1797693134862315807937289714053034150799341327100378269361737789804449682927647509466"
         "49017977587207096330286416692887910946555547851940402630657488671505820681908902000708383"
         "676273854845817711531764475730270069855571366959622842914819860834936475292719074168444"
         "365510704342711559699508093042880177904174497792e309", False),
    ]:  # fmt: skip
        path = write_blob(tmp_path, tensor_with(text))
        expected = OK if accepts else ERROR_FORMAT
        assert flint_status(flint, path) == expected, (text, flint.last_error())


@pytest.mark.parametrize(
    "text", ["01", "1.", ".5", "+1", "1e", "-", "NaN", "Infinity", "tru"]
)
def test_grammar_errors_in_an_unknown_field(flint, tmp_path, text):
    path = write_blob(tmp_path, tensor_with(text))
    rejected_by_both(flint, path)


# --- depth -------------------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [60, 125])
@pytest.mark.parametrize("shape", ["arrays", "objects", "mixed", "arrays with values"])
def test_nesting_up_to_125_levels_is_accepted(flint, tmp_path, shape, depth):
    path = write_blob(tmp_path, tensor_with(nest(shape, depth)))
    accepted_by_both(flint, path)


@pytest.mark.parametrize("depth", [126, 127, 1000, 20_000])
@pytest.mark.parametrize("shape", ["arrays", "objects", "mixed", "arrays with values"])
def test_nesting_of_126_levels_and_more_is_rejected(flint, tmp_path, shape, depth):
    path = write_blob(tmp_path, tensor_with(nest(shape, depth)))
    rejected_by_both(flint, path)
    assert "deep" in flint.last_error()


def nest(shape, depth):
    if shape == "arrays":
        return "[" * depth + "]" * depth
    if shape == "objects":
        return '{"k":' * depth + "1" + "}" * depth
    if shape == "mixed":
        opens = ["[" if i % 2 == 0 else '{"k":' for i in range(depth)]
        closes = ["]" if i % 2 == 0 else "}" for i in range(depth)]
        return "".join(opens) + "1" + "".join(reversed(closes))
    return "[1," * depth + "2" + "]" * depth


def test_a_50_MB_header_of_open_brackets_opens_without_crashing(flint, tmp_path):
    path = tmp_path / "brackets.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 50_000_000))
        f.write(b"[" * 50_000_000)
    try:
        assert flint_status(flint, path) == ERROR_FORMAT, flint.last_error()
    finally:
        path.unlink()


def test_a_50_MB_header_of_open_brackets_inside_a_tensor_opens_without_crashing(flint, tmp_path):
    path = tmp_path / "deep.safetensors"
    head = b'{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"x":'
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 50_000_000))
        f.write(head)
        f.write(b"[" * (50_000_000 - len(head)))
    try:
        assert flint_status(flint, path) == ERROR_FORMAT, flint.last_error()
        assert "deep" in flint.last_error()
    finally:
        path.unlink()


def test_a_50_MB_header_of_open_braces_inside_a_tensor_opens_without_crashing(flint, tmp_path):
    path = tmp_path / "deep.safetensors"
    head = b'{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"x":'
    body = b'{"k":' * ((50_000_000 - len(head)) // 5)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head) + len(body)))
        f.write(head)
        f.write(body)
    try:
        assert flint_status(flint, path) == ERROR_FORMAT, flint.last_error()
    finally:
        path.unlink()


# --- the header's length -----------------------------------------------------------------------


def write_big_header(path, n, data=b""):
    """A file whose header is `{}` and spaces, n bytes long, then data."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(b"{}")
        remaining = n - 2
        block = b" " * MiB
        while remaining > 0:
            f.write(block[: min(remaining, MiB)])
            remaining -= MiB
        f.write(data)


def test_a_header_of_exactly_100_000_000_bytes_is_accepted(flint, tmp_path):
    path = tmp_path / "big.safetensors"
    write_big_header(path, MAX_HEADER)
    try:
        assert path.stat().st_size == 8 + MAX_HEADER
        verdict = python_verdict(path)
        assert verdict.accepted, verdict.message
        assert flint_status(flint, path) == OK, flint.last_error()
        with SafetensorsReader(flint, path) as reader:
            assert reader.count() == 0 and reader.metadata_count() == 0
    finally:
        path.unlink()


def test_a_header_of_100_000_001_bytes_is_rejected(flint, tmp_path):
    path = tmp_path / "bigger.safetensors"
    write_big_header(path, MAX_HEADER + 1)
    try:
        assert path.stat().st_size == 8 + MAX_HEADER + 1
        rejected_by_both(flint, path)
        assert "100,000,000" in flint.last_error()
    finally:
        path.unlink()


def test_a_header_of_100_000_000_bytes_with_a_tensor_at_the_end(flint, tmp_path):
    path = tmp_path / "big.safetensors"
    text = b'{"a":' + U8_8.encode() + b"}"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", MAX_HEADER))
        f.write(text + b" " * (MAX_HEADER - len(text)))
        f.write(D8)
    try:
        accepted_by_both(flint, path)
        with SafetensorsReader(flint, path) as reader:
            assert reader.read(0) == D8
    finally:
        path.unlink()


@pytest.mark.parametrize(
    "prefix, rest",
    [
        (0, b""),
        (0, D8),
        (2**64 - 1, b"{}"),
        (2**64 - 8, b"{}"),
        (2**63, b"{}"),
        (2**32, b"{}"),
        (MAX_HEADER + 1, b"{}"),
        (2**64 - 1, b""),
        (3, b"{}"),
        (10, b"{}"),
        (1, b"{}"),
    ],
)
def test_a_header_length_the_file_cannot_hold_or_the_format_forbids(flint, tmp_path, prefix, rest):
    path = write_blob(tmp_path, struct.pack("<Q", prefix) + rest)
    rejected_by_both(flint, path)


@pytest.mark.parametrize("size", range(8))
def test_a_file_shorter_than_8_bytes(flint, tmp_path, size):
    path = write_blob(tmp_path, b"\x02\x00\x00\x00\x00\x00\x00\x00"[:size])
    rejected_by_both(flint, path)


def test_a_header_length_too_big_for_the_file_names_the_rule(flint, tmp_path):
    path = write_blob(tmp_path, struct.pack("<Q", 5000) + b"{}")
    assert flint_status(flint, path) == ERROR_FORMAT
    assert "5000" in flint.last_error() and "past the end" in flint.last_error()


# --- UTF-8 -------------------------------------------------------------------------------------

INVALID_UTF8 = {
    "stray continuation 80": b"\x80",
    "truncated 2-byte C3": b"\xc3",
    "truncated 3-byte E2 82": b"\xe2\x82",
    "truncated 4-byte F0 9F 98": b"\xf0\x9f\x98",
    "overlong C0 AF": b"\xc0\xaf",
    "overlong E0 80 80": b"\xe0\x80\x80",
    "overlong F0 80 80 80": b"\xf0\x80\x80\x80",
    "surrogate ED A0 80": b"\xed\xa0\x80",
    "above U+10FFFF F4 90 80 80": b"\xf4\x90\x80\x80",
    "above U+10FFFF F5 80 80 80": b"\xf5\x80\x80\x80",
    "FF": b"\xff",
    "a lead without its continuation, then ASCII": b"\xe2\x82a",
}


@pytest.mark.parametrize("form", list(INVALID_UTF8))
def test_invalid_utf8_in_a_name(flint, tmp_path, form):
    blob = file_bytes(b'{"a' + INVALID_UTF8[form] + b'":' + U8_8.encode() + b"}", D8)
    path = write_blob(tmp_path, blob)
    rejected_by_both(flint, path)
    assert "UTF-8" in flint.last_error()


@pytest.mark.parametrize("form", list(INVALID_UTF8))
def test_invalid_utf8_in_the_padding(flint, tmp_path, form):
    blob = file_bytes(b'{"a":' + U8_8.encode() + b"}" + INVALID_UTF8[form], D8)
    path = write_blob(tmp_path, blob)
    rejected_by_both(flint, path)
    assert "UTF-8" in flint.last_error()


@pytest.mark.parametrize("form", ["overlong C0 AF", "surrogate ED A0 80", "FF"])
def test_invalid_utf8_in_a_metadata_value_and_an_unknown_field(flint, tmp_path, form):
    bad = INVALID_UTF8[form]
    for header in (
        b'{"__metadata__":{"k":"v' + bad + b'"},"a":' + U8_8.encode() + b"}",
        b'{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"x":"' + bad + b'"}}',
        b'{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8],"x' + bad + b'":1}}',
    ):
        path = write_blob(tmp_path, file_bytes(header, D8))
        rejected_by_both(flint, path)


# --- every prefix of a valid file ------------------------------------------------------------


def test_every_prefix_of_a_valid_file_is_rejected_by_both(flint, tmp_path):
    header = (
        '{"__metadata__":{"format":"pt","n":"\\u00e9"},'
        '"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"x":{"y":[1,2.5e3,null]}},'
        '"b":{"dtype":"U8","shape":[3],"data_offsets":[8,11]},'
        '"z":{"dtype":"I64","shape":[0],"data_offsets":[11,11]}}'
    )
    blob = file_bytes(header, bytes(range(11)))
    path = tmp_path / "prefix.safetensors"
    path.write_bytes(blob)
    accepted_by_both(flint, path, blob)
    for length in range(len(blob)):
        path.write_bytes(blob[:length])
        assert not python_verdict(path).accepted, f"Python accepted the first {length} bytes"
        assert flint_status(flint, path) == ERROR_FORMAT, (length, flint.last_error())
    path.write_bytes(blob)
    assert flint_status(flint, path) == OK


# --- speed -------------------------------------------------------------------------------------


def test_a_header_of_a_million_zero_size_tensors_opens_in_seconds(flint, tmp_path):
    path = tmp_path / "million.safetensors"
    entry = '":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}'
    # The header is written in blocks, and its length is filled in afterwards.
    with open(path, "wb") as f:
        f.write(bytes(8))
        length = 0
        for first in range(0, 1_000_000, 100_000):
            text = ",".join(f'"t{i}{entry}' for i in range(first, first + 100_000))
            text = ("{" if first == 0 else ",") + text + ("}" if first == 900_000 else "")
            length += len(text)
            f.write(text.encode("ascii"))
        padding = (8 - length % 8) % 8
        f.write(b" " * padding)
        length += padding
        assert 55_000_000 < length < MAX_HEADER
        f.seek(0)
        f.write(struct.pack("<Q", length))
    try:
        started = time.perf_counter()
        reader = SafetensorsReader(flint, path)
        took = time.perf_counter() - started
        with reader:
            assert took < 10, f"opening took {took:.1f} s"
            assert reader.count() == 1_000_000
            assert reader.find("t0") == 0
            assert reader.find("t999999") == 999_999
            assert reader.find("t500000") == 500_000
            assert reader.info(999_999) == SafetensorsInfo("t999999", "U8", (0,), 8, 0, 0)
            with pytest.raises(FlintError):
                reader.find("t1000000")
    finally:
        path.unlink()


# --- the differential fuzz ---------------------------------------------------------------------


def fuzz_bases():
    """Five small valid files. Together they hold metadata, escapes, an unknown field with
    nesting, a zero-size tensor, the array form of a tensor, the map form of a dtype, and U8,
    F32 and I64 tensors. They are padded with names, values
    and whitespace that the format lets through, so that a mutant often is still valid."""
    u8 = bytes(range(10, 15))
    f32 = struct.pack("<2f", 1.5, -2.25)
    i64 = struct.pack("<q", -123456789012)
    data = u8 + f32 + i64  # 5 + 8 + 8 bytes
    a = (
        '{"__metadata__":{"key1":"value one \\u0001 here","key2":"value two \\u00e9\\n",'
        '"key3":"v\\/3 \\"q\\" \\\\ end","key4":"\\u0001\\u0002\\u001f","format":"pt"},'
        '"tensor.one":{"dtype":"U8","shape":[5],"data_offsets":[0,5]},'
        '"tensor.two":{"dtype":"F32","shape":[2],"data_offsets":[5,13]},'
        '"tensor.three":{"dtype":"I64","shape":[1],"data_offsets":[13,21]}}'
    )
    b = (
        '{"u":{"dtype":"U8","extra":{"key1":[1,2.5e3,true,null,{"z":"some text \\u00e9 here"}],'
        '"key2":"more text, and more text"},"shape":[5],"data_offsets":[0,5]},'
        '"f":{"dtype":"F32","shape":[2],"data_offsets":[5,13],"note":[[],{},"a string"]},'
        '"i":{"dtype":"I64","shape":[1],"data_offsets":[13,21],"scale":-0.5e-3,"flag":false}}'
    )
    c = (
        '{"i64.name":{"dtype":"I64","shape":[1],"data_offsets":[13,21]},'
        '"zero.size":{"dtype":"F32","shape":[0,3],"data_offsets":[5,5]},'
        '"u8.name":{"dtype":"U8","shape":[5],"data_offsets":[0,5]},'
        '"f32.name":{"dtype":"F32","shape":[2],"data_offsets":[5,13]}}'
        "        "
    )
    d = (
        '{ "__metadata__" : { "k\\u0001" : "v\\/w x y z" , "e" : "" , "name" : "a b c d e f" } ,\n'
        '"\\u00e9u":{"dtype":"U8","shape":[5],"data_offsets":[0,5],"x":-0.5e-3},\n'
        ' "f":{"dtype":"F32","shape":[2],"data_offsets":[5,13]} ,\n'
        '"i":{"dtype":"I64","shape":[1],"data_offsets":[13,21]},'
        '"zz":{"dtype":"BOOL","shape":[0],"data_offsets":[21,21]} }'
    )
    e = (
        '{"__metadata__":{"key1":"one","key2":"two","format":"pt"},'
        '"arr.u8":["U8",[5],[0,5]],'
        '"arr.f32":[ {"F32":null} , [2] , [5,13] ],'
        '"obj.i64":{"dtype":{"I64":null},"shape":[1],"data_offsets":[13,21],"note":[1,{"a":2}]},'
        '"arr.bool":[{"\\u0042OOL":null},[0],[21,21]]}'
    )
    return [file_bytes(text, data) for text in (a, b, c, d, e)]


def random_byte(rng):
    """Any byte, or half the time one that JSON uses (so that the mutant is often valid)."""
    if rng.random() < 0.5:
        return rng.randrange(256)
    return rng.choice(b' "\\,:{}[]0123456789abcxyz\t\n\r\x00\x01u/-.eE')


def mutate(rng, blob):
    """One or two mutations of the file's bytes: replace a byte with a random one, insert a
    random byte, delete a byte, duplicate or swap a short slice, replace a digit with another
    digit, or change the length prefix by -8..-1 or 1..8. Most land in the header, and after
    one that changes the header's length the prefix is nearly always corrected, so that the mutant
    is more often a file that is nearly valid than one whose length is simply wrong."""
    for _ in range(1 + (rng.random() < 0.3)):
        n = struct.unpack("<Q", blob[:8])[0]
        op = rng.choice(["replace", "insert", "delete", "duplicate", "swap", "digit", "length"])
        if op == "length":
            delta = rng.choice([-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
            blob = struct.pack("<Q", (n + delta) % 2**64) + blob[8:]
            continue
        # The bytes the mutation works on: the header (7 times in 10) or the whole file.
        in_header = rng.random() < 0.85
        start, stop = (8, 8 + n) if in_header else (0, len(blob))
        body = bytearray(blob[start:stop])
        if not body:
            continue
        pos = rng.randrange(len(body))
        if op == "replace":
            body[pos] = random_byte(rng)
        elif op == "insert":
            body.insert(pos, random_byte(rng))
        elif op == "delete":
            del body[pos]
        elif op == "duplicate":
            size = rng.randrange(1, 9)
            body[pos + size : pos + size] = body[pos : pos + size]
        elif op == "swap":
            size = rng.randrange(1, 5)
            first, second = sorted((pos, rng.randrange(len(body))))
            if second >= first + size and second + size <= len(body):
                one, two = bytes(body[first : first + size]), bytes(body[second : second + size])
                body[first : first + size] = two
                body[second : second + size] = one
        else:
            digits = [i for i, byte in enumerate(body) if 0x30 <= byte <= 0x39]
            # Some of the time, the last digit of a \u000X escape, which can become \u0000.
            escaped = [i for i in digits if bytes(body[max(i - 5, 0) : i]) == b"\\u000"]
            if escaped and rng.random() < 0.4:
                digits = escaped
            if digits:
                at = rng.choice(digits)
                body[at] = 0x30 + (body[at] - 0x30 + rng.randrange(1, 10)) % 10
        mutated = blob[:start] + bytes(body) + blob[stop:]
        if in_header and len(body) != n and rng.random() < 0.9:
            mutated = struct.pack("<Q", len(body)) + mutated[8:]
        blob = mutated
    return blob


NUMPY_DTYPES = {"U8", "F32", "I64"}
FUZZ_BASES = fuzz_bases()


def run_fuzz_seed(flint, seed, folder):
    """100 mutants of the 5 bases, seeded: Python's and Flint's verdicts on each. Returns the
    counts; raises AssertionError, naming the seed, the index and the mutant in hex, for a
    verdict that differs without being one of the listed differences."""
    path = Path(folder) / f"fuzz{seed}.safetensors"
    stats = Counter()
    for index in range(100):
        rng = random.Random(seed * 1000 + index)
        mutant = mutate(rng, FUZZ_BASES[index % len(FUZZ_BASES)])
        where = f"seed {seed}, mutant {index}: {mutant.hex()}"
        path.write_bytes(mutant)
        verdict = python_verdict(path)
        status = flint_status(flint, path)
        assert status in (OK, ERROR_FORMAT), f"{where}: status {status}, {flint.last_error()}"
        if verdict.accepted and status == OK:
            stats["both accept"] += 1
            check_fuzz_agreement(flint, path, mutant, verdict, where, stats)
        elif not verdict.accepted and status == ERROR_FORMAT:
            stats["both reject"] += 1
            assert flint.last_error() != "", where
        elif verdict.accepted and status == ERROR_FORMAT:
            causes = listed_causes(mutant)
            assert causes, f"Python accepts, Flint rejects: {flint.last_error()}; {where}"
            stats["differ: " + causes[0]] += 1
        else:
            raise AssertionError(f"Python rejects ({verdict.message}), Flint accepts; {where}")
    path.unlink()
    return stats


def listed_causes(mutant):
    """The listed differences a header that Python accepts holds, read with json.loads: a tensor
    name or a metadata key given twice, and a NUL in a name, a metadata key or a value."""
    n, text, _ = split_blob(mutant)
    try:
        top = json.loads(text.decode("utf-8"), object_pairs_hook=list)
    except (ValueError, RecursionError):
        return []
    names = [key for key, _ in top if key != "__metadata__"]
    causes = []
    if len(set(names)) != len(names):
        causes.append("duplicate tensor name")
    if any("\0" in name for name in names):
        causes.append("NUL in a tensor name")
    for key, pairs in top:
        if key == "__metadata__" and pairs:
            keys = [k for k, _ in pairs]
            if len(set(keys)) != len(keys):
                causes.append("duplicate metadata key")
            if any("\0" in k for k in keys):
                causes.append("NUL in a metadata key")
            if any("\0" in v for _, v in pairs):
                causes.append("NUL in a metadata value")
    return causes


def check_fuzz_agreement(flint, path, mutant, verdict, where, stats):
    """Both accept: names (in key order), dtypes, shapes and metadata agree, and so do the
    bytes of every U8, F32 and I64 tensor."""
    check_matches_python(flint, path, verdict)
    n, text, data = split_blob(mutant)
    try:
        tensors, metadata = header_entries(text)
    except (ValueError, RecursionError):
        stats["json.loads could not read the header"] += 1
        tensors = None
    with SafetensorsReader(flint, path) as reader:
        infos = reader.infos()
        if tensors is not None:
            assert [i.name for i in infos] == [t[0] for t in tensors], where
            assert reader.metadata_pairs() == [tuple(p) for p in metadata], where
        picked = [i for i in infos if i.dtype in NUMPY_DTYPES]
        if picked:
            with safe_open(str(path), framework="np") as reference:
                for info in picked:
                    expected = reference.get_tensor(info.name)
                    assert reader.read(reader.find(info.name)) == expected.tobytes(), where
                    stats["tensors compared with safetensors.numpy"] += 1
                del expected


@pytest.mark.parametrize("seed", range(20))
def test_differential_fuzz(flint, tmp_path, seed):
    stats = run_fuzz_seed(flint, seed, tmp_path)
    assert sum(v for k, v in stats.items() if k.startswith(("both", "differ"))) == 100
    # The mutants must exercise both verdicts, or the comparison is empty.
    assert stats["both accept"] >= 3 and stats["both reject"] >= 3, dict(stats)


# =============================================================================================
# 4. Flint writes, Python reads
# =============================================================================================


def check_written(flint, path, tensors, metadata=()):
    """The file at path, written by Flint for tensors [(name, dtype, shape, payload)] and
    metadata [(key, value)]: the header is the reference's, the data is the payloads, Python
    accepts it and reads the same names, dtypes, shapes, metadata and bytes, and Flint reads it
    back. No temporary file is left."""
    path = Path(path)
    assert path.is_file() and not Path(str(path) + ".tmp").exists()
    raw = path.read_bytes()
    n, text, data = split_blob(raw)
    assert text == reference_header(tensors, metadata)
    assert n % 8 == 0 and (8 + n) % 8 == 0
    assert data == b"".join(t[3] for t in tensors)

    verdict = python_verdict(path)
    assert verdict.accepted, verdict.message
    assert sorted(verdict.tensors) == sorted(t[0] for t in tensors)
    for name, dtype, shape, payload in tensors:
        assert verdict.tensors[name] == (dtype, tuple(shape)), name
    assert (verdict.metadata or {}) == dict(metadata)
    # The bytes through the torch binding, for every tensor of a dtype torch has.
    with safe_open(str(path), framework="pt") as reference:
        for name, dtype, shape, payload in tensors:
            if dtype in TORCH_DTYPES:
                tensor = reference.get_tensor(name)
                assert tensor.dtype == TORCH_DTYPES[dtype], name
                assert tuple(tensor.shape) == tuple(shape), name
                assert torch_raw(tensor) == payload, name
                del tensor
    check_reads(flint, path, raw)
    return raw


def sized(dtype, shape, seed):
    return make_payload(dtype, element_bytes(dtype, shape), seed)


@pytest.mark.parametrize("dtype", list(DTYPE_BITS))
def test_every_dtype_written_by_flint_is_read_by_python(flint, tmp_path, dtype):
    shape = SUBBYTE_SHAPES.get(dtype, (3, 5))
    tensors = [
        ("head", "U8", (1,), b"\x07"),
        ("x." + dtype, dtype, shape, sized(dtype, shape, hash_seed(dtype))),
        ("tail", "U8", (3,), b"xyz"),
    ]
    path = tmp_path / "out.safetensors"
    entries = write_safetensors(flint, path, tensors)
    assert entries[1] == (DTYPE_BITS[dtype], 1, element_bytes(dtype, shape))
    check_written(flint, path, tensors)


@pytest.mark.parametrize("rank", range(9))
def test_ranks_0_to_8_written_by_flint(flint, tmp_path, rank):
    shape = (2, 3, 1, 2, 1, 2, 2, 1)[:rank]
    tensors = [("t", "F16", shape, sized("F16", shape, rank)), ("u", "U8", (2,), b"ab")]
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, tensors)
    check_written(flint, path, tensors)


@pytest.mark.parametrize("layout", list(_zero_size_layouts()))
def test_zero_size_tensors_written_by_flint(flint, tmp_path, layout):
    entries, data_length = _zero_size_layouts()[layout]
    # Flint lays tensors out in the order given, so each non-empty one gets its bytes in turn.
    tensors = []
    for name, shape, begin, end in sorted(entries, key=lambda e: (e[2], e[3])):
        if end > begin:
            tensors.append((name, "U8", shape, bytes(range(begin, end))))
        else:
            tensors.append((name, "U8", [0], b""))
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, tensors)
    check_written(flint, path, tensors)


@pytest.mark.parametrize("shape", [(0,), (3, 0, 2), (0, 0, 0, 0, 0, 0, 0, 0)])
def test_zero_size_shapes_written_by_flint(flint, tmp_path, shape):
    tensors = [("z", "F32", shape, b""), ("a", "U8", (4,), b"abcd"), ("y", "BF16", shape, b"")]
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, tensors)
    check_written(flint, path, tensors)


@pytest.mark.parametrize("case", list(NAME_CASES))
def test_names_written_by_flint(flint, tmp_path, case):
    name = NAME_CASES[case][1]
    tensors = [("z", "U8", (1,), b"z"), (name, "F32", (2,), b"abcdefgh"), ("zz", "U8", (1,), b"y")]
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, tensors)
    check_written(flint, path, tensors)


def test_many_unusual_names_in_one_file(flint, tmp_path):
    names = [NAME_CASES[c][1] for c in NAME_CASES]
    names = list(dict.fromkeys(names))
    tensors = [(n, "U8", (2,), bytes([i, i + 1])) for i, n in enumerate(names)]
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, tensors)
    check_written(flint, path, tensors)


METADATA_WRITE_CASES = {
    "none": [],
    "one pair": [("format", "pt")],
    "fifty pairs": [(f"key{i:02d}", f"value {i}") for i in range(50)],
    "escapes": [
        ("k\u0001\n", 'v/"\\ é 日本   \U0001f600'),
        ("path/to/file", "a/b/c"),
        ("\u0001", "\u001f\u007f"),
        ("", ""),
        ("empty value", ""),
        ("日本語", "テスト"),
        ("tab\there", "line\nbreak\r\n"),
        ('quote"', "back\\slash"),
        ("bell\b\f", "\x08\x0c"),
    ],
    "long": [("k" * 3000, "v" * 100_000)],
    "keys that differ in case and whitespace": [("A", "1"), ("a", "2"), ("a ", "3"), (" a", "4")],
}


@pytest.mark.parametrize("case", list(METADATA_WRITE_CASES))
def test_metadata_written_by_flint(flint, tmp_path, case):
    metadata = METADATA_WRITE_CASES[case]
    tensors = [("a", "F32", (2,), b"abcdefgh"), ("b", "U8", (3,), b"xyz")]
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, tensors, metadata)
    check_written(flint, path, tensors, metadata)
    with SafetensorsReader(flint, path) as reader:
        assert reader.metadata_pairs() == metadata
        assert reader.metadata_count() == len(metadata)


def test_no_tensors_and_no_metadata_gives_an_empty_object(flint, tmp_path):
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, [])
    raw = path.read_bytes()
    assert raw == struct.pack("<Q", 8) + b"{}" + b" " * 6
    assert python_verdict(path).accepted
    with SafetensorsReader(flint, path) as reader:
        assert reader.count() == 0 and reader.metadata_count() == 0


def test_metadata_without_tensors(flint, tmp_path):
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, [], [("k", "v")])
    raw = path.read_bytes()
    assert raw[8:] == pad8(b'{"__metadata__":{"k":"v"}}')
    verdict = python_verdict(path)
    assert verdict.accepted and verdict.metadata == {"k": "v"}


def test_a_header_whose_length_is_already_a_multiple_of_8_gets_no_padding(flint, tmp_path):
    # {"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}} is 53 bytes; "aaaaaaaa" in place of
    # "a" makes it 60 and a 3-byte name makes it 55; find the name length that gives a multiple.
    for length in range(1, 12):
        name = "n" * length
        text = json.dumps({name: T("U8", [1], 0, 1)}, separators=(",", ":"))
        if len(text) % 8 == 0:
            break
    else:
        raise AssertionError("no length fits")
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, [(name, "U8", (1,), b"x")])
    raw = path.read_bytes()
    n, header, _ = split_blob(raw)
    assert n == len(text) and header == text.encode()
    check_written(flint, path, [(name, "U8", (1,), b"x")])


def python_writer_case(tensors, metadata):
    """Python's own bytes for torch tensors [(name, dtype, shape, payload)] and a metadata
    dict, and the tensors and the metadata pairs in the order Python wrote them (Python's
    metadata order differs from one process to the next, so it is read back from its header)."""
    blob = safetensors.torch.save(
        {name: torch_tensor(dtype, shape, payload) for name, dtype, shape, payload in tensors},
        metadata=metadata or None,
    )
    n, text, _ = split_blob(blob)
    ordered, pairs = header_entries(text)
    by_name = {t[0]: t for t in tensors}
    assert sorted(by_name) == sorted(o[0] for o in ordered)
    return blob, [by_name[o[0]] for o in ordered], [tuple(p) for p in pairs]


def python_writer_cases():
    first = [
        ("w", "F32", (2, 3), sized("F32", (2, 3), 1)),
        ("b", "F16", (3,), sized("F16", (3,), 2)),
        ("z", "U8", (5,), sized("U8", (5,), 3)),
    ]
    # probe_python2.py's case: awkward names, and metadata with a backspace, a form feed, a
    # carriage return, U+001F, U+2028 and a 4-byte character.
    awkward = [
        ("bé", "U8", (1,), b"\x11"),
        ('a"q\\/\n\t\x01\x7f', "F32", (2,), sized("F32", (2,), 4)),
        ("c", "F16", (3,), sized("F16", (3,), 5)),
        ("d", "F64", (1,), sized("F64", (1,), 6)),
        ("e", "BOOL", (1,), b"\x01"),
    ]
    many = [
        ("embed", "BF16", (4, 2), sized("BF16", (4, 2), 7)),
        ("a.f64", "F64", (3,), sized("F64", (3,), 8)),
        ("b.f32", "F32", (5,), sized("F32", (5,), 9)),
        ("c.i64", "I64", (2,), sized("I64", (2,), 10)),
        ("d.i32", "I32", (3, 1), sized("I32", (3, 1), 11)),
        ("e.i16", "I16", (7,), sized("I16", (7,), 12)),
        ("f.i8", "I8", (9,), sized("I8", (9,), 13)),
        ("g.u8", "U8", (11,), sized("U8", (11,), 14)),
        ("h.bool", "BOOL", (13,), sized("BOOL", (13,), 15)),
        ("i.f16", "F16", (2, 2), sized("F16", (2, 2), 16)),
        ("j.empty", "F32", (0, 4), b""),
        ("k.c64", "C64", (1,), sized("C64", (1,), 17)),
        ("l.u16", "U16", (3,), sized("U16", (3,), 18)),
        ("m.f8", "F8_E4M3", (6,), sized("F8_E4M3", (6,), 19)),
    ]
    return {
        "three tensors, format pt": (first, {"format": "pt"}),
        "awkward names and metadata": (
            awkward,
            {"k\b\f\r\x1f": "v \U0001f600", "format": "pt"},
        ),
        "fourteen tensors, no metadata": (many, {}),
    }


@pytest.mark.parametrize("case", list(python_writer_cases()))
def test_file_is_byte_identical_with_pythons_writer(flint, tmp_path, case):
    tensors, metadata = python_writer_cases()[case]
    blob, ordered, pairs = python_writer_case(tensors, metadata)
    assert dict(pairs) == metadata
    path = tmp_path / "out.safetensors"
    write_safetensors(flint, path, ordered, pairs)
    assert path.read_bytes() == blob
    check_written(flint, path, ordered, pairs)


def three_tensor_file():
    return [
        ("a", "F32", (2, 3), sized("F32", (2, 3), 21)),
        ("b", "U8", (5,), sized("U8", (5,), 22)),
        ("c", "I64", (2,), sized("I64", (2,), 23)),
    ]


CHUNKINGS = {
    "one call": None,
    "1 byte": 1,
    "2 bytes": 2,
    "3 bytes": 3,
    "7 bytes": 7,
    "44 and 1": [44, 1],
    "straddling the first boundary": [23, 2, 100],
    "straddling both boundaries": [25, 4, 1, 3, 100],
    "on the boundaries": [24, 5, 16],
    "a tensor at a time, backwards sizes": [16, 5, 24],
    "one byte, then the rest": [1, 44],
    "zero-byte calls between": [0, 10, 0, 0, 35],
}


@pytest.mark.parametrize("name", list(CHUNKINGS))
def test_chunked_writes_make_the_same_file(flint, tmp_path, name):
    tensors = three_tensor_file()
    whole = tmp_path / "whole.safetensors"
    write_safetensors(flint, whole, tensors, [("format", "pt")])
    chunked = tmp_path / "chunked.safetensors"
    write_safetensors(flint, chunked, tensors, [("format", "pt")], CHUNKINGS[name])
    assert chunked.read_bytes() == whole.read_bytes()
    check_written(flint, chunked, tensors, [("format", "pt")])


def test_create_fills_bits_offset_and_bytes(flint, tmp_path):
    declared = [
        ("f4", "F4", (2, 8)),
        ("f6", "F6_E2M3", (4, 2)),
        ("u8", "U8", (3,)),
        ("zero", "F32", (0,)),
        ("f32", "F32", (2, 2)),
        ("c64", "C64", (1,)),
        ("scalar", "I16", ()),
    ]
    expected = [
        (4, 0, 8), (6, 8, 6), (8, 14, 3), (32, 17, 0), (32, 17, 16), (64, 33, 8), (16, 41, 2),
    ]
    status, writer, array = safetensors_create(flint, tmp_path / "e.safetensors", declared)
    assert status == OK, flint.last_error()
    try:
        assert [(t.bits, t.offset, t.bytes) for t in array[: len(declared)]] == expected
    finally:
        flint.lib.flint_safetensors_destroy(writer)


FAILING_CREATES = {
    "a duplicate name": ([("a", "U8", (1,)), ("a", "U8", (1,))], (), "dir"),
    "an unknown dtype": ([("a", "U8", (1,)), ("b", "X8", (1,))], (), "dir"),
    "an overflowing shape": ([("a", "U8", (1,)), ("b", "U8", (2**32, 2**32))], (), "dir"),
    "a duplicate key": ([("a", "U8", (1,))], [("k", "1"), ("k", "2")], "dir"),
    "a missing folder": ([("a", "U8", (1,)), ("b", "F32", (2,))], (), "missing"),
}


@pytest.mark.parametrize("case", list(FAILING_CREATES))
def test_entries_are_left_alone_when_create_fails(flint, tmp_path, case):
    declared, metadata, where = FAILING_CREATES[case]
    folder = tmp_path if where == "dir" else tmp_path / "no" / "such" / "folder"
    path = folder / "x.safetensors"
    array, keys, values, keepalive = safetensors_declare(declared, metadata)
    for slot in array[: len(declared)]:
        slot.bits, slot.offset, slot.bytes = 777, 888, 999
    writer = ctypes.c_void_p(12345)
    status = flint.lib.flint_safetensors_create(
        str(path).encode(), array, len(declared), keys, values, len(metadata), ctypes.byref(writer)
    )
    assert status in (ERROR_INVALID_ARGUMENT, ERROR_IO)
    assert writer.value is None
    assert [(t.bits, t.offset, t.bytes) for t in array[: len(declared)]] == [(777, 888, 999)] * len(
        declared
    )
    assert not path.exists() and not Path(str(path) + ".tmp").exists()
    del keepalive


def test_flint_writes_flint_reads_every_info_and_byte(flint, tmp_path):
    tensors = []
    for i, dtype in enumerate(DTYPE_BITS):
        shape = SUBBYTE_SHAPES.get(dtype, (i % 4 + 1, 3))
        tensors.append((f"layer{i}.{dtype}", dtype, shape, sized(dtype, shape, 100 + i)))
    tensors.append(("scalar", "F32", (), sized("F32", (), 1)))
    tensors.append(("empty", "I8", (0, 2), b""))
    metadata = [("format", "flint"), ("note", "é\n")]
    path = tmp_path / "out.safetensors"
    entries = write_safetensors(flint, path, tensors, metadata, chunks=[1, 7, 1000])
    with SafetensorsReader(flint, path) as reader:
        assert reader.count() == len(tensors)
        offset = 0
        for index, (name, dtype, shape, payload) in enumerate(tensors):
            info = reader.info(index)
            expected = SafetensorsInfo(
                name, dtype, shape, DTYPE_BITS[dtype], offset, len(payload)
            )
            assert info == expected
            assert entries[index] == (DTYPE_BITS[dtype], offset, len(payload))
            assert reader.find(name) == index
            assert reader.read(index) == payload
            offset += len(payload)
        assert reader.metadata_pairs() == metadata
    check_written(flint, path, tensors, metadata)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows holds the temporary file locked")
def test_a_tensor_of_2_32_plus_16_bytes_gets_64_bit_offsets_in_the_header(flint, tmp_path):
    big = 2**32 + 16
    path = tmp_path / "big.safetensors"
    declared = [("big", "U8", (big,)), ("after", "U8", (3,))]
    with SafetensorsWriter(flint, path, declared) as writer:
        assert writer.entries == [(8, 0, big), (8, big, 3)]
        tmp = Path(writer.tmp_path)
        raw = tmp.read_bytes()
        n, text, rest = split_blob(raw)
        assert rest == b""
        tensors, _ = header_entries(text)
        assert tensors == [
            ("big", "U8", [big], 0, big),
            ("after", "U8", [3], big, big + 3),
        ]
        expected = pad8(
            json.dumps(
                {"big": T("U8", [big], 0, big), "after": T("U8", [3], big, big + 3)},
                separators=(",", ":"),
            ).encode()
        )
        assert text == expected
        assert n == len(expected)
    assert not tmp.exists()
    assert not path.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="the Linux variant reads the header itself")
def test_a_tensor_of_2_32_plus_16_bytes_gets_64_bit_offsets_in_the_header_windows(flint, tmp_path):
    # Windows opens the temporary file without sharing, so it can't be read while the writer is
    # open; its size is the header's length, which has the 10-digit offsets in it.
    big = 2**32 + 16
    path = tmp_path / "big.safetensors"
    declared = [("big", "U8", (big,)), ("after", "U8", (3,))]
    expected = pad8(
        json.dumps(
            {"big": T("U8", [big], 0, big), "after": T("U8", [3], big, big + 3)},
            separators=(",", ":"),
        ).encode()
    )
    with SafetensorsWriter(flint, path, declared) as writer:
        assert writer.entries == [(8, 0, big), (8, big, 3)]
        tmp = Path(writer.tmp_path)
        assert tmp.stat().st_size == 8 + len(expected)
    assert not tmp.exists()
    assert not path.exists()


# =============================================================================================
# 5. Writer errors and durability
# =============================================================================================


def tmp_of(path):
    return Path(str(path) + ".tmp")


def no_files(path):
    assert not Path(path).exists(), f"{path} exists"
    assert not tmp_of(path).exists(), f"{tmp_of(path)} exists"


def create_with(flint, target, declared=(("a", "U8", (2,)),), metadata=(), tweak=None, **override):
    """flint_safetensors_create for the file target, with the arguments the keyword overrides
    replace: path, tensors, count, keys, values, metadata_count, writer. tweak(array, keys,
    values) edits the arrays first. Returns (status, the writer handle or None)."""
    declared = list(declared)
    array, keys, values, keepalive = safetensors_declare(declared, metadata)
    if tweak is not None:
        tweak(array, keys, values)
    writer = ctypes.c_void_p(12345)
    arguments = {
        "path": os.fspath(target).encode("utf-8"),
        "tensors": array,
        "count": len(declared),
        "keys": keys,
        "values": values,
        "metadata_count": len(metadata),
        "writer": ctypes.byref(writer),
    }
    arguments.update(override)
    status = flint.lib.flint_safetensors_create(
        arguments["path"], arguments["tensors"], arguments["count"], arguments["keys"],
        arguments["values"], arguments["metadata_count"], arguments["writer"],
    )  # fmt: skip
    del keepalive
    return status, (None if "writer" in override else writer.value)


def _set(array, index, field, value):
    setattr(array[index], field, value)


def _long_name_header_base():
    return len(json.dumps({"": T("U8", [1], 0, 1)}, separators=(",", ":")))


A2 = (("a", "U8", (2,)),)
CREATE_ERRORS = {
    "a null path": dict(path=None),
    "a null writer pointer": dict(writer=None),
    "tensors null with a count": dict(tensors=None),
    "keys null with a metadata count": dict(metadata=[("k", "v")], keys=None),
    "values null with a metadata count": dict(metadata=[("k", "v")], values=None),
    "a null name": dict(tweak=lambda a, k, v: _set(a, 0, "name", None)),
    "a null dtype": dict(tweak=lambda a, k, v: _set(a, 0, "dtype", None)),
    "a null shape with a rank above 0": dict(tweak=lambda a, k, v: _set(a, 0, "shape", None)),
    "a null key": dict(metadata=[("k", "v")], tweak=lambda a, k, v: k.__setitem__(0, None)),
    "a null value": dict(metadata=[("k", "v")], tweak=lambda a, k, v: v.__setitem__(0, None)),
    "an empty path": dict(path=b""),
    "invalid UTF-8 in a name": dict(declared=[("aé", "U8", (1,)), (b"b\xff", "U8", (1,))]),
    "an overlong form in a name": dict(declared=[(b"a\xc0\xaf", "U8", (1,))]),
    "a surrogate in a name": dict(declared=[(b"a\xed\xa0\x80", "U8", (1,))]),
    "a truncated sequence in a name": dict(declared=[(b"a\xe2\x82", "U8", (1,))]),
    "invalid UTF-8 in a key": dict(metadata=[("ok", "v"), (b"k\xff", "v")]),
    "invalid UTF-8 in a value": dict(metadata=[("ok", "v"), ("k", b"v\xc3")]),
    "__metadata__ as a tensor name": dict(
        declared=[("a", "U8", (1,)), ("__metadata__", "U8", (1,))]
    ),
    "a duplicate name": dict(declared=[("a", "U8", (1,)), ("b", "U8", (1,)), ("a", "U8", (1,))]),
    "a duplicate name, adjacent": dict(declared=[("a", "U8", (1,)), ("a", "F32", (1,))]),
    "a duplicate key": dict(metadata=[("k", "1"), ("j", "2"), ("k", "3")]),
    "an unknown dtype": dict(declared=[("a", "X8", (1,))]),
    "a lowercase dtype": dict(declared=[("a", "u8", (1,))]),
    "an empty dtype": dict(declared=[("a", "", (1,))]),
    "a dtype with a trailing space": dict(declared=[("a", "U8 ", (1,))]),
    "a dtype of a prefix of a real one": dict(declared=[("a", "F8", (1,))]),
    "a shape that overflows": dict(declared=[("a", "U8", (2**32, 2**32))]),
    "a shape that overflows before a zero": dict(declared=[("a", "U8", (2**63, 2, 0))]),
    "a shape whose bits overflow": dict(declared=[("a", "F32", (2**62,))]),
    "a shape of 2^64 - 1 twice": dict(declared=[("a", "U8", (U64_MAX, U64_MAX))]),
    "F4 [3]": dict(declared=[("a", "F4", (3,))]),
    "F4 [1]": dict(declared=[("a", "F4", (1, 1))]),
    "F6_E2M3 [1]": dict(declared=[("a", "F6_E2M3", (1,))]),
    "F6_E3M2 [3, 3]": dict(declared=[("a", "F6_E3M2", (3, 3))]),
    "a total size that overflows": dict(declared=[("a", "U8", (2**63,)), ("b", "U8", (2**63,))]),
    "a total size that overflows by one": dict(
        declared=[("a", "U8", (U64_MAX,)), ("b", "U8", (1,))]
    ),
    # Callables, so the 100 MB strings exist only while their test runs.
    "a header over 100 MB": dict(declared=lambda: [(b"n" * MAX_HEADER, "U8", (1,))]),
    "a header just over 100 MB": dict(
        declared=lambda: [(b"n" * (MAX_HEADER - _long_name_header_base() + 1), "U8", (1,))]
    ),
    "metadata over 100 MB": dict(metadata=lambda: [("k", b"v" * MAX_HEADER)]),
}


@pytest.mark.parametrize("case", list(CREATE_ERRORS))
def test_create_argument_errors(flint, tmp_path, case):
    arguments = dict(CREATE_ERRORS[case])
    declared = arguments.pop("declared", A2)
    if callable(declared):
        declared = declared()
    if callable(arguments.get("metadata")):
        arguments["metadata"] = arguments["metadata"]()
    path = tmp_path / "x.safetensors"
    status, writer = create_with(flint, path, declared, **arguments)
    assert status == ERROR_INVALID_ARGUMENT, flint.last_error()
    assert writer is None
    assert flint.last_error().startswith("flint_safetensors_create: ")
    no_files(path)
    assert list(tmp_path.iterdir()) == []


def test_a_header_of_exactly_100_000_000_bytes_can_be_written(flint, tmp_path):
    name = "n" * (MAX_HEADER - _long_name_header_base())
    path = tmp_path / "x.safetensors"
    try:
        entries = write_safetensors(flint, path, [(name, "U8", (1,), b"x")])
        assert entries == [(8, 0, 1)]
        assert path.stat().st_size == 8 + MAX_HEADER + 1
        with SafetensorsReader(flint, path) as reader:
            assert reader.count() == 1 and reader.info(0).name == name
            assert reader.read(0) == b"x"
    finally:
        for leftover in (path, tmp_of(path)):
            if leftover.exists():
                leftover.unlink()


def test_a_failed_create_leaves_an_existing_file_and_temporary_file_alone(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    path.write_bytes(b"old target")
    tmp_of(path).write_bytes(b"old leftover")
    status, writer = create_with(flint, path, [("a", "X8", (1,))])
    assert status == ERROR_INVALID_ARGUMENT and writer is None
    # The arguments are checked before any file is made, so both stay as they were.
    assert path.read_bytes() == b"old target"
    assert tmp_of(path).read_bytes() == b"old leftover"


def small_writer(flint, path, **kwargs):
    tensors = [("a", "F32", (2,), b"abcdefgh"), ("b", "U8", (3,), b"xyz")]
    return tensors, SafetensorsWriter(flint, path, [t[:3] for t in tensors], **kwargs)


def test_a_write_past_the_declared_total_writes_nothing(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors, writer = small_writer(flint, path)
    with writer:
        # The data is 11 bytes. Each refused write leaves the count where it was.
        assert writer.write_status(b"abcdefghxyz!") == ERROR_INVALID_ARGUMENT
        assert "pass" in flint.last_error()
        writer.write(b"abcde")
        assert writer.write_status(b"fghxyz!") == ERROR_INVALID_ARGUMENT  # 5 + 7
        writer.write(b"fghx")
        assert writer.write_status(b"yz!") == ERROR_INVALID_ARGUMENT  # 9 + 3
        writer.write(b"yz")
        assert writer.write_status(b"!") == ERROR_INVALID_ARGUMENT  # 11 + 1
        writer.commit()
    check_written(flint, path, tensors)


def test_a_write_past_the_total_by_a_huge_count_is_refused_without_wrapping(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors, writer = small_writer(flint, path)
    with writer:
        status = flint.lib.flint_safetensors_write(writer.handle, b"abcd", U64_MAX)
        assert status == ERROR_INVALID_ARGUMENT
        status = flint.lib.flint_safetensors_write(writer.handle, b"abcd", 2**63)
        assert status == ERROR_INVALID_ARGUMENT
        writer.write(b"".join(t[3] for t in tensors))
        writer.commit()
    check_written(flint, path, tensors)


def test_a_commit_with_bytes_missing_is_refused_and_the_writer_stays_usable(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors, writer = small_writer(flint, path)
    with writer:
        assert writer.commit_status() == ERROR_INVALID_ARGUMENT  # nothing written yet
        writer.write(b"abcdefghxy")
        assert writer.commit_status() == ERROR_INVALID_ARGUMENT  # one byte short
        assert "10" in flint.last_error() and "11" in flint.last_error()
        assert not path.exists() and tmp_of(path).exists()
        writer.write(b"z")
        writer.commit()
    check_written(flint, path, tensors)


def test_a_write_or_commit_after_a_commit_is_refused(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors, writer = small_writer(flint, path)
    with writer:
        writer.write(b"abcdefghxyz")
        writer.commit()
        before = path.read_bytes()
        assert writer.write_status(b"!") == ERROR_INVALID_ARGUMENT
        assert "committed" in flint.last_error()
        assert writer.write_status(b"") == ERROR_INVALID_ARGUMENT
        assert writer.commit_status() == ERROR_INVALID_ARGUMENT
        assert "committed" in flint.last_error()
        assert path.read_bytes() == before
    assert path.read_bytes() == before
    check_written(flint, path, tensors)


def test_a_write_with_a_null_src_and_bytes_is_refused_but_zero_bytes_may_be_null(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors, writer = small_writer(flint, path)
    with writer:
        assert flint.lib.flint_safetensors_write(writer.handle, None, 0) == OK
        assert flint.lib.flint_safetensors_write(writer.handle, None, 3) == ERROR_INVALID_ARGUMENT
        writer.write(b"abcdefghxyz")
        assert flint.lib.flint_safetensors_write(writer.handle, None, 0) == OK
        writer.commit()
    check_written(flint, path, tensors)


def test_a_file_with_no_data_commits_without_a_write(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors = [("z", "F32", (0,), b"")]
    with SafetensorsWriter(flint, path, [t[:3] for t in tensors]) as writer:
        assert writer.write_status(b"x") == ERROR_INVALID_ARGUMENT
        writer.commit()
    check_written(flint, path, tensors)


def test_an_existing_file_is_unchanged_until_commit_replaces_it(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    old = hand_built([("old", "U8", (4,), b"1234")])
    path.write_bytes(old)
    tensors, writer = small_writer(flint, path, metadata=[("k", "v")])
    with writer:
        assert path.read_bytes() == old
        assert tmp_of(path).exists()
        writer.write(b"abcdef")
        assert path.read_bytes() == old
        assert writer.commit_status() == ERROR_INVALID_ARGUMENT
        assert path.read_bytes() == old
        writer.write(b"ghxyz")
        assert path.read_bytes() == old
        writer.commit()
        assert path.read_bytes() != old
        assert not tmp_of(path).exists()
    check_written(flint, path, tensors, [("k", "v")])


def test_the_temporary_file_exists_while_writing_and_is_gone_after_commit(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors, writer = small_writer(flint, path)
    with writer:
        assert tmp_of(path).is_file() and not path.exists()
        assert writer.tmp_path == str(tmp_of(path))
        writer.write(b"abcdefghxyz")
        assert tmp_of(path).is_file() and not path.exists()
        writer.commit()
        assert path.is_file() and not tmp_of(path).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.safetensors"]


def test_destroying_an_uncommitted_writer_removes_the_temporary_file(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    path.write_bytes(b"the target")
    tensors, writer = small_writer(flint, path)
    writer.write(b"abcde")
    assert tmp_of(path).exists()
    writer.destroy()
    assert not tmp_of(path).exists()
    assert path.read_bytes() == b"the target"


def test_destroying_a_writer_after_a_refused_commit_removes_the_temporary_file(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors, writer = small_writer(flint, path)
    writer.write(b"abcdefgh")
    assert writer.commit_status() == ERROR_INVALID_ARGUMENT
    writer.destroy()
    no_files(path)


def test_destroying_a_committed_writer_leaves_the_file(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors, writer = small_writer(flint, path)
    writer.write(b"abcdefghxyz")
    writer.commit()
    writer.destroy()
    check_written(flint, path, tensors)


def test_destroy_of_null_is_fine(flint):
    flint.lib.flint_safetensors_destroy(None)
    flint.lib.flint_safetensors_close(None)


def test_a_leftover_temporary_file_is_overwritten(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tmp_of(path).write_bytes(b"\xee" * 100_000)  # longer than the file that follows
    tensors = [("a", "F32", (2,), b"abcdefgh"), ("b", "U8", (3,), b"xyz")]
    write_safetensors(flint, path, tensors)
    check_written(flint, path, tensors)


def test_a_temporary_file_left_by_a_writer_that_was_never_destroyed_is_overwritten(flint, tmp_path):
    # A crash leaves the writer's file behind with some data in it.
    path = tmp_path / "x.safetensors"
    declared = [("a", "U8", (6,))]
    status, writer, _ = safetensors_create(flint, path, declared)
    assert status == OK
    flint.lib.flint_safetensors_write(writer, b"123456", 6)
    flint.lib.flint_safetensors_destroy(writer)
    tmp_of(path).write_bytes(b"crash" * 1000)
    tensors = [("a", "U8", (2,), b"hi")]
    write_safetensors(flint, path, tensors)
    check_written(flint, path, tensors)


def test_a_second_writer_to_the_same_path_replaces_the_first_one_s_file(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    first = [("a", "U8", (3,), b"one")]
    second = [("b", "U8", (4,), b"two2")]
    write_safetensors(flint, path, first)
    write_safetensors(flint, path, second)
    check_written(flint, path, second)


def test_a_missing_folder_is_an_io_error_naming_the_path(flint, tmp_path):
    path = tmp_path / "no" / "such" / "folder" / "x.safetensors"
    status, writer, array = safetensors_create(flint, path, [("a", "U8", (2,))])
    assert status == ERROR_IO
    assert writer is None
    message = flint.last_error()
    assert message.startswith("flint_safetensors_create: ")
    assert str(path)[:100] in message
    assert not (tmp_path / "no").exists()


def test_a_target_that_is_a_folder_fails_at_commit_and_removes_the_temporary_file(flint, tmp_path):
    path = tmp_path / "target"
    path.mkdir()
    (path / "inside").write_bytes(b"x")
    tensors, writer = small_writer(flint, path)
    with writer:
        writer.write(b"abcdefghxyz")
        assert tmp_of(path).exists()
        assert writer.commit_status() == ERROR_IO
        assert str(path)[:100] in flint.last_error()
        assert not tmp_of(path).exists()
        assert path.is_dir() and (path / "inside").read_bytes() == b"x"
        # The writer has failed: it refuses everything with an I/O error.
        assert writer.write_status(b"") == ERROR_IO
        assert "earlier write failed" in flint.last_error()
        assert writer.commit_status() == ERROR_IO
        assert "earlier write failed" in flint.last_error()
    assert path.is_dir() and not tmp_of(path).exists()


def test_a_temporary_path_that_is_a_folder_fails_create(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tmp_of(path).mkdir()
    status, writer, array = safetensors_create(flint, path, [("a", "U8", (2,))])
    assert status == ERROR_IO
    assert writer is None
    assert not path.exists() and tmp_of(path).is_dir()


def test_a_non_ASCII_folder_works_for_writing_and_reading(flint, tmp_path):
    folder = tmp_path / "dä日本"
    folder.mkdir()
    path = folder / "mödel-\U0001f600.safetensors"
    tensors = [("a", "F32", (2,), b"abcdefgh"), ("b", "U8", (3,), b"xyz")]
    write_safetensors(flint, path, tensors, [("k", "v")])
    check_written(flint, path, tensors, [("k", "v")])
    with SafetensorsReader(flint, path) as reader:
        assert reader.read(reader.find("b")) == b"xyz"


def test_a_commit_over_a_file_a_reader_has_open(flint, tmp_path):
    # A checkpoint is replaced while a reader may still hold the previous one. On Windows the
    # move fails while any handle has the path open; on Linux the rename replaces the file and
    # the open reader keeps reading the old one (flint.h documents both).
    path = tmp_path / "x.safetensors"
    first = [("a", "U8", (4,), b"1234")]
    second = [("a", "U8", (4,), b"5678"), ("b", "U8", (2,), b"zz")]
    write_safetensors(flint, path, first)
    old_bytes = path.read_bytes()
    with SafetensorsReader(flint, path) as reader:
        with SafetensorsWriter(flint, path, [t[:3] for t in second]) as writer:
            writer.write(b"".join(t[3] for t in second))
            if sys.platform == "win32":
                assert writer.commit_status() == ERROR_IO
                assert "MoveFileExW" in flint.last_error()
                assert str(path)[:100] in flint.last_error()
                assert not tmp_of(path).exists()
                assert path.read_bytes() == old_bytes
                assert reader.read(0) == b"1234"
                # The writer has failed: it refuses what follows.
                assert writer.write_status(b"") == ERROR_IO
                assert writer.commit_status() == ERROR_IO
            else:
                writer.commit()
                assert not tmp_of(path).exists()
                assert reader.read(0) == b"1234"
                with SafetensorsReader(flint, path) as fresh:
                    assert fresh.count() == 2 and fresh.read(0) == b"5678"
    if sys.platform == "win32":
        # With the reader closed, a fresh writer commits over the old file.
        write_safetensors(flint, path, second)
    check_written(flint, path, second)


def test_the_reader_keeps_working_after_its_file_is_deleted(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    tensors = [("a", "U8", (4,), b"1234")]
    write_safetensors(flint, path, tensors)
    with SafetensorsReader(flint, path) as reader:
        os.remove(path)
        assert reader.read(0) == b"1234"
        assert reader.count() == 1


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows doesn't let a file shrink under a reader"
)
def test_a_read_after_the_file_shrank_is_an_io_error(flint, tmp_path):
    path = tmp_path / "x.safetensors"
    blob = hand_built([("a", "U8", (100,), bytes(range(100)))])
    path.write_bytes(blob)
    with SafetensorsReader(flint, path) as reader:
        os.truncate(path, len(blob) - 40)
        assert reader.read(0, 0, 60) == bytes(range(60))
        buffer = ctypes.create_string_buffer(40)
        status = flint.lib.flint_safetensors_read(reader.handle, 0, 60, 40, buffer)
        assert status == ERROR_IO
        assert "shrink" in flint.last_error()


# =============================================================================================
# 6. Reader API errors
# =============================================================================================


@pytest.fixture
def small_file(flint, tmp_path):
    tensors = [("a", "F32", (2,), b"abcdefgh"), ("b", "U8", (3,), b"xyz")]
    path = tmp_path / "small.safetensors"
    write_safetensors(flint, path, tensors, [("k", "v"), ("j", "w")])
    return path


def _null_calls():
    c = ctypes
    size = lambda: c.byref(c.c_size_t())  # noqa: E731
    text = lambda: c.byref(c.c_char_p())  # noqa: E731
    buffer = c.create_string_buffer(16)
    return {
        "open: path": lambda L, r, w, p: L.flint_safetensors_open(None, c.byref(c.c_void_p())),
        "open: reader": lambda L, r, w, p: L.flint_safetensors_open(p, None),
        "tensor_count: reader": lambda L, r, w, p: L.flint_safetensors_tensor_count(None, size()),
        "tensor_count: count": lambda L, r, w, p: L.flint_safetensors_tensor_count(r, None),
        "tensor_info: reader": lambda L, r, w, p: L.flint_safetensors_tensor_info(
            None, 0, c.byref(flint_lib.SafetensorsTensor())),
        "tensor_info: info": lambda L, r, w, p: L.flint_safetensors_tensor_info(r, 0, None),
        "find: reader": lambda L, r, w, p: L.flint_safetensors_find(None, b"a", size()),
        "find: name": lambda L, r, w, p: L.flint_safetensors_find(r, None, size()),
        "find: index": lambda L, r, w, p: L.flint_safetensors_find(r, b"a", None),
        "metadata_count: reader": lambda L, r, w, p: L.flint_safetensors_metadata_count(
            None, size()
        ),
        "metadata_count: count": lambda L, r, w, p: L.flint_safetensors_metadata_count(r, None),
        "metadata: reader": lambda L, r, w, p: L.flint_safetensors_metadata(
            None, 0, text(), text()
        ),
        "metadata: key": lambda L, r, w, p: L.flint_safetensors_metadata(r, 0, None, text()),
        "metadata: value": lambda L, r, w, p: L.flint_safetensors_metadata(r, 0, text(), None),
        "metadata_find: reader": lambda L, r, w, p: L.flint_safetensors_metadata_find(
            None, b"k", text()
        ),
        "metadata_find: key": lambda L, r, w, p: L.flint_safetensors_metadata_find(r, None, text()),
        "metadata_find: value": lambda L, r, w, p: L.flint_safetensors_metadata_find(r, b"k", None),
        "read: reader": lambda L, r, w, p: L.flint_safetensors_read(None, 0, 0, 1, buffer),
        "read: dst with bytes": lambda L, r, w, p: L.flint_safetensors_read(r, 0, 0, 1, None),
        "write: writer": lambda L, r, w, p: L.flint_safetensors_write(None, b"x", 1),
        "write: src with bytes": lambda L, r, w, p: L.flint_safetensors_write(w, None, 1),
        "commit: writer": lambda L, r, w, p: L.flint_safetensors_commit(None),
    }  # fmt: skip


NULL_CALLS = _null_calls()


@pytest.mark.parametrize("call", list(NULL_CALLS))
def test_null_arguments_are_invalid(flint, tmp_path, small_file, call):
    out = tmp_path / "out.safetensors"
    with SafetensorsReader(flint, small_file) as reader, SafetensorsWriter(
        flint, out, [("t", "U8", (4,))]
    ) as writer:
        status = NULL_CALLS[call](flint.lib, reader.handle, writer.handle, str(small_file).encode())
        assert status == ERROR_INVALID_ARGUMENT
        assert flint.last_error().startswith("flint_safetensors_")
        # Nothing was disturbed: the reader and writer still work.
        assert reader.count() == 2
        writer.write(b"abcd")
        writer.commit()
    assert out.read_bytes()[-4:] == b"abcd"


def test_open_with_a_null_path_clears_the_reader_it_returns(flint):
    handle = ctypes.c_void_p(12345)
    assert flint.lib.flint_safetensors_open(None, ctypes.byref(handle)) == ERROR_INVALID_ARGUMENT
    assert handle.value is None


@pytest.mark.parametrize("index", [2, 3, 1000, 2**63, U64_MAX])
def test_an_index_past_the_count_is_invalid(flint, small_file, index):
    with SafetensorsReader(flint, small_file) as reader:
        info = flint_lib.SafetensorsTensor()
        info.name = b"untouched"
        status = flint.lib.flint_safetensors_tensor_info(reader.handle, index, ctypes.byref(info))
        assert status == ERROR_INVALID_ARGUMENT
        assert "index" in flint.last_error() and "count" in flint.last_error()
        assert info.name == b"untouched"
        key, value = ctypes.c_char_p(b"k0"), ctypes.c_char_p(b"v0")
        status = flint.lib.flint_safetensors_metadata(
            reader.handle, index, ctypes.byref(key), ctypes.byref(value)
        )
        assert status == ERROR_INVALID_ARGUMENT
        assert (key.value, value.value) == (b"k0", b"v0")
        buffer = ctypes.create_string_buffer(b"\xaa" * 4, 4)
        status = flint.lib.flint_safetensors_read(reader.handle, index, 0, 4, buffer)
        assert status == ERROR_INVALID_ARGUMENT
        assert buffer.raw == b"\xaa" * 4
        status = flint.lib.flint_safetensors_read(reader.handle, index, 0, 0, None)
        assert status == ERROR_INVALID_ARGUMENT


def test_the_last_index_is_valid(flint, small_file):
    with SafetensorsReader(flint, small_file) as reader:
        assert reader.info(1).name == "b"
        assert reader.metadata_pair(1) == (b"j", b"w")


@pytest.mark.parametrize(
    "offset, nbytes",
    [
        (0, 9),
        (1, 8),
        (7, 2),
        (8, 1),
        (9, 0),
        (9, 1),
        (100, 0),
        (U64_MAX, 2),
        (U64_MAX, 1),
        (U64_MAX, 0),
        (U64_MAX - 1, 3),
        (2, U64_MAX),
        (0, U64_MAX),
        (1, 2**63),
        (2**63, 2**63),
        (4, 2**63 + 5),
        (2**32, 2**32),
    ],
)
def test_a_read_past_the_end_of_the_tensor_is_invalid_and_does_not_wrap(
    flint, small_file, offset, nbytes
):
    with SafetensorsReader(flint, small_file) as reader:
        buffer = ctypes.create_string_buffer(b"\xaa" * 16, 16)
        status = flint.lib.flint_safetensors_read(reader.handle, 0, offset, nbytes, buffer)
        assert status == ERROR_INVALID_ARGUMENT
        assert "pass" in flint.last_error()
        assert buffer.raw == b"\xaa" * 16
        assert reader.read(0) == b"abcdefgh"


def test_the_end_of_the_tensor_is_not_past_it(flint, small_file):
    with SafetensorsReader(flint, small_file) as reader:
        assert reader.read(0, 8, 0) == b""
        assert reader.read(0, 7, 1) == b"h"
        assert reader.read(1, 3, 0) == b""
        assert reader.read(1, 2, 1) == b"z"


def test_find_of_a_missing_name_names_it(flint, small_file):
    with SafetensorsReader(flint, small_file) as reader:
        index = ctypes.c_size_t(777)
        for name in (b"missing", b"", b"A", b"a ", "dä".encode()):
            status = flint.lib.flint_safetensors_find(reader.handle, name, ctypes.byref(index))
            assert status == ERROR_INVALID_ARGUMENT
            assert index.value == 777
            message = flint.last_error()
            assert f"`{name.decode()}`" in message and "no tensor" in message
        # A name longer than 128 bytes is cut in the message.
        status = flint.lib.flint_safetensors_find(reader.handle, b"n" * 300, ctypes.byref(index))
        assert status == ERROR_INVALID_ARGUMENT
        assert "n" * 128 in flint.last_error() and "n" * 129 not in flint.last_error()


def test_metadata_find_of_a_missing_key_names_it(flint, small_file):
    with SafetensorsReader(flint, small_file) as reader:
        value = ctypes.c_char_p(b"untouched")
        for key in (b"missing", b"", b"K", b"k ", "ké".encode()):
            status = flint.lib.flint_safetensors_metadata_find(
                reader.handle, key, ctypes.byref(value)
            )
            assert status == ERROR_INVALID_ARGUMENT
            assert value.value == b"untouched"
            assert f"`{key.decode()}`" in flint.last_error()


def test_a_missing_file_is_an_io_error_naming_the_path(flint, tmp_path):
    path = tmp_path / "nothing.safetensors"
    handle = ctypes.c_void_p(12345)
    status = flint.lib.flint_safetensors_open(str(path).encode(), ctypes.byref(handle))
    assert status == ERROR_IO
    assert handle.value is None
    message = flint.last_error()
    assert message.startswith("flint_safetensors_open: ")
    assert str(path)[:100] in message
    assert "CreateFileW" in message if sys.platform == "win32" else "open" in message


def test_a_folder_opened_as_a_file_is_an_io_error(flint, tmp_path):
    folder = tmp_path / "folder.safetensors"
    folder.mkdir()
    handle = ctypes.c_void_p(12345)
    status = flint.lib.flint_safetensors_open(str(folder).encode(), ctypes.byref(handle))
    assert status == ERROR_IO
    assert handle.value is None
    assert str(folder)[:100] in flint.last_error()


def test_an_empty_path_is_an_invalid_argument_for_open(flint):
    handle = ctypes.c_void_p(12345)
    assert flint.lib.flint_safetensors_open(b"", ctypes.byref(handle)) == ERROR_INVALID_ARGUMENT
    assert handle.value is None
    assert "empty" in flint.last_error()


def test_a_failed_open_returns_no_reader_and_keeps_nothing_open(flint, tmp_path):
    path = write_blob(tmp_path, file_bytes('{"a":' + U8_8 + "}", D8[:4]))
    handle = ctypes.c_void_p(12345)
    status = flint.lib.flint_safetensors_open(str(path).encode(), ctypes.byref(handle))
    assert status == ERROR_FORMAT
    assert handle.value is None
    # Nothing holds the file: it can be replaced and deleted, which Windows refuses while open.
    path.write_bytes(b"")
    path.unlink()


FORMAT_MESSAGES = {
    "a hole": (
        file_bytes(json.dumps({"a": T("U8", [2], 0, 2), "bb": T("U8", [4], 4, 8)}), D8),
        ["hole", "`bb`", "byte 2"],
    ),
    "an overlap": (
        file_bytes(json.dumps({"a": T("U8", [4], 0, 4), "bb": T("U8", [4], 2, 6)}), D8[:6]),
        ["overlap", "`bb`"],
    ),
    "trailing bytes": (
        file_bytes(json.dumps({"a": T("U8", [4], 0, 4)}), D8),
        ["trailing", "4 of the data's 8"],
    ),
    "data too short": (
        file_bytes(json.dumps({"a": T("U8", [8], 0, 8)}), D8[:6]),
        ["past the data", "`a`"],
    ),
    "too large": (struct.pack("<Q", MAX_HEADER + 1) + b"{}", ["100,000,000", "100000001"]),
    "past the end of the file": (struct.pack("<Q", 5000) + b"{}", ["5000", "past the end"]),
    "too short": (b"\x01\x00\x00", ["too short"]),
    "invalid UTF-8": (file_bytes(b'{"a\xff":' + U8_8.encode() + b"}", D8), ["UTF-8"]),
    "an unknown dtype": (
        file_bytes(json.dumps({"tt": T("Q8", [8], 0, 8)}), D8),
        ["unknown dtype", "`tt`", "`Q8`"],
    ),
    "the wrong size": (
        file_bytes(json.dumps({"tt": T("F32", [3], 0, 8)}), D8),
        ["`tt`", "8 bytes", "12"],
    ),
    "an overflowing shape": (
        file_bytes(json.dumps({"tt": T("U8", [2**32, 2**32], 0, 8)}), D8),
        ["`tt`", "overflow"],
    ),
    "bits that aren't a whole byte": (
        file_bytes(json.dumps({"tt": T("F4", [3], 0, 2)}), D8[:2]),
        ["`tt`", "whole number of bytes"],
    ),
    "a duplicate name": (
        file_bytes(
            '{"dd":'
            + json.dumps(T("U8", [4], 0, 4))
            + ',"dd":'
            + json.dumps(T("U8", [4], 4, 8))
            + "}",
            D8,
        ),
        ["`dd`", "twice"],
    ),
    "a duplicate metadata key": (
        file_bytes('{"__metadata__":{"kk":"1","kk":"2"},"a":' + U8_8 + "}", D8),
        ["`kk`", "twice"],
    ),
    "a number out of range": (tensor_with("1e999"), ["range"]),
    "nesting": (tensor_with("[" * 130 + "]" * 130), ["deep"]),
    "a missing field": (
        file_bytes('{"tt":{"dtype":"U8","shape":[8]}}', D8),
        ["`tt`", "data_offsets"],
    ),
    "a float shape": (file_bytes(BAD_SHAPE % "[8.0]", D8), ["unsigned"]),
    "trailing characters": (file_bytes('{"a":' + U8_8 + "}x", D8), ["after the header"]),
}


@pytest.mark.parametrize("case", list(FORMAT_MESSAGES))
def test_format_errors_name_the_rule(flint, tmp_path, case):
    blob, needles = FORMAT_MESSAGES[case]
    path = write_blob(tmp_path, blob)
    assert flint_status(flint, path) == ERROR_FORMAT
    message = flint.last_error()
    assert message.startswith("flint_safetensors_open: ")
    assert str(path)[:100] in message
    for needle in needles:
        assert needle in message, (needle, message)


NUL_FILES = {
    "a tensor name": file_bytes('{"a\\u0000b":' + U8_8 + "}", D8),
    "a metadata key": file_bytes('{"__metadata__":{"k\\u0000":"v"},"a":' + U8_8 + "}", D8),
    "a metadata value": file_bytes('{"__metadata__":{"kk":"v\\u0000"},"a":' + U8_8 + "}", D8),
}
# A name or key in a message is JSON-escaped, so a NUL shows as \\u0000 and the message isn't cut
# there. A value isn't quoted, so its message names the rule and the key.
NUL_MESSAGES = {
    "a tensor name": ["the tensor name", "`a\\u0000b`", "NUL"],
    "a metadata key": ["the metadata key", "`k\\u0000`", "NUL"],
    "a metadata value": ["the value of metadata key `kk`", "NUL"],
}


@pytest.mark.parametrize("case", list(NUL_FILES))
def test_a_nul_in_a_string_is_named_in_the_message(flint, tmp_path, case):
    path = write_blob(tmp_path, NUL_FILES[case])
    assert python_verdict(path).accepted
    assert flint_status(flint, path) == ERROR_FORMAT
    message = flint.last_error()
    assert str(path)[:100] in message
    for needle in NUL_MESSAGES[case]:
        assert needle in message, (needle, message)


def test_names_in_messages_are_json_escaped(flint, small_file):
    name = 'a"b\\c\n\x01\x7f/\u00e9'
    escaped = 'a\\"b\\\\c\\n\\u0001\x7f/\u00e9'
    with SafetensorsReader(flint, small_file) as reader:
        for call, kind in ((reader.find, "find"), (reader.metadata_find, "metadata_find")):
            with pytest.raises(FlintError) as error:
                call(name)
            assert error.value.status == ERROR_INVALID_ARGUMENT
            assert f"`{escaped}`" in str(error.value), kind


def test_a_long_name_is_cut_to_128_bytes_at_a_character_boundary(flint, tmp_path):
    name = "日" * 100  # 300 bytes; 128 falls inside the 43rd character
    header = {"a": T("U8", [2], 0, 2), name: T("U8", [4], 4, 8)}
    path = write_blob(tmp_path, file_bytes(json.dumps(header, ensure_ascii=False), D8))
    assert flint_status(flint, path) == ERROR_FORMAT
    message = flint.last_error()  # decoding it strictly checks that it is valid UTF-8
    assert "日" * 42 in message and "日" * 43 not in message
    assert len(message.encode("utf-8")) < 512


def test_every_long_message_is_valid_utf8_even_when_it_is_cut(flint, tmp_path):
    # Long non-ASCII paths, names and dtypes make the longest messages the library builds; the
    # last error is cut at 512 bytes and must still decode.
    folder = tmp_path
    for part in ("ä" * 20, "日" * 20):
        folder = folder / part
        folder.mkdir()
    name = "日" * 60
    cases = [
        file_bytes(
            json.dumps({"a": T("U8", [2], 0, 2), name: T("U8", [4], 4, 8)}, ensure_ascii=False),
            D8,
        ),
        file_bytes(json.dumps({name: T("ä" * 200, [8], 0, 8)}, ensure_ascii=False), D8),
        file_bytes(json.dumps({name: T("F32", [3], 0, 8)}, ensure_ascii=False), D8),
        file_bytes('{"' + name + '":' + U8_8 + ',"' + name + '":' + U8_8 + "}", D8),
    ]
    for blob in cases:
        path = write_blob(folder, blob, "möd" * 10 + ".safetensors")
        assert flint_status(flint, path) == ERROR_FORMAT
        message = flint.last_error()
        assert len(message.encode("utf-8")) <= 511
        message.encode("utf-8").decode("utf-8")
    # The same through the writer: a long non-ASCII name and dtype in a message.
    status, writer, array = safetensors_create(
        flint, folder / "w.safetensors", [(name, "ä" * 200, (1,))]
    )
    assert status == ERROR_INVALID_ARGUMENT
    assert len(flint.last_error().encode("utf-8")) <= 511
    status, writer, array = safetensors_create(
        flint, folder / "w.safetensors", [(name, "U8", (1,)), (name, "U8", (1,))]
    )
    assert status == ERROR_INVALID_ARGUMENT
    assert len(flint.last_error().encode("utf-8")) <= 511


def test_close_of_null_is_fine(flint):
    flint.lib.flint_safetensors_close(None)


def test_strings_the_reader_hands_out_live_until_close(flint, tmp_path):
    names = [f"tensor.{i}" for i in range(300)]
    tensors = [(n, "U8", (i % 3 + 1,), bytes([i % 256]) * (i % 3 + 1)) for i, n in enumerate(names)]
    path = tmp_path / "x.safetensors"
    write_safetensors(flint, path, tensors, [(f"k{i}", f"v{i}") for i in range(20)])
    with SafetensorsReader(flint, path) as reader:
        infos = []
        for i in range(300):
            info = flint_lib.SafetensorsTensor()
            flint.check(
                flint.lib.flint_safetensors_tensor_info(reader.handle, i, ctypes.byref(info))
            )
            infos.append(info)
        pairs = []
        for i in range(20):
            key, value = ctypes.c_char_p(), ctypes.c_char_p()
            flint.check(
                flint.lib.flint_safetensors_metadata(
                    reader.handle, i, ctypes.byref(key), ctypes.byref(value)
                )
            )
            pairs.append((key, value))
        # Many more calls later, the earlier pointers still hold their strings and shapes.
        for i in range(300):
            reader.info(i)
        assert [t.name.decode() for t in infos] == names
        assert [t.dtype for t in infos] == [b"U8"] * 300
        assert [t.shape[0] for t in infos] == [i % 3 + 1 for i in range(300)]
        expected_pairs = [(f"k{i}".encode(), f"v{i}".encode()) for i in range(20)]
        assert [(k.value, v.value) for k, v in pairs] == expected_pairs


def test_two_readers_of_one_file_are_independent(flint, tmp_path):
    payload = random.Random(5).randbytes(5000)
    path = write_blob(tmp_path, hand_built([("t", "U8", (5000,), payload)]))
    with SafetensorsReader(flint, path) as one, SafetensorsReader(flint, path) as two:
        assert one.read(0, 100, 50) == payload[100:150]
        assert two.read(0, 4000, 1000) == payload[4000:]
        assert one.read(0, 0, 5000) == payload
    # And one file read from several threads, since reads share no file pointer.
    import threading

    errors = []

    def work(seed):
        rng = random.Random(seed)
        try:
            with SafetensorsReader(flint, path) as reader:
                for _ in range(200):
                    offset = rng.randrange(5000)
                    nbytes = rng.randrange(5001 - offset)
                    assert reader.read(0, offset, nbytes) == payload[offset : offset + nbytes]
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]


# =============================================================================================
# 7. 64-bit reads
# =============================================================================================


def make_sparse(path, size, header, marker):
    """Writes header (8-byte length and JSON) at the start of path, extends the file to size
    bytes without storing the gap, and puts marker at the end. False if the file can't be
    made sparse (the file is removed then)."""
    with open(path, "wb") as f:
        if sys.platform == "win32":
            import msvcrt

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.DeviceIoControl.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
            ]  # fmt: skip
            kernel32.DeviceIoControl.restype = ctypes.c_int
            returned = ctypes.c_uint32(0)
            handle = msvcrt.get_osfhandle(f.fileno())
            if not kernel32.DeviceIoControl(  # FSCTL_SET_SPARSE
                handle, 0x900C4, None, 0, None, 0, ctypes.byref(returned), None
            ):
                return False
        f.write(header)
        # Writing at the far end extends the sparse file without storing the gap. (truncate
        # would not: on Windows Python's truncate fills the new bytes with zeros.)
        f.seek(size - len(marker))
        f.write(marker)
    return allocated_bytes(path) < 256 * MiB


def allocated_bytes(path):
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCompressedFileSizeW.argtypes = [
            ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32),
        ]
        kernel32.GetCompressedFileSizeW.restype = ctypes.c_uint32
        high = ctypes.c_uint32(0)
        low = kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
        return (high.value << 32) | low
    return os.stat(path).st_blocks * 512


def test_a_tensor_of_2_32_plus_16_bytes_is_read_at_64_bit_offsets(flint, tmp_path):
    big = 2**32 + 16
    marker = bytes(range(1, 17))
    header = pad8(json.dumps({"big": T("U8", [big], 0, big)}, separators=(",", ":")).encode())
    prefix = struct.pack("<Q", len(header)) + header
    path = tmp_path / "sparse.safetensors"
    try:
        if not make_sparse(path, len(prefix) + big, prefix, marker):
            pytest.skip("this file system can't make the 4 GiB file sparse")
        with SafetensorsReader(flint, path) as reader:
            info = reader.info(0)
            assert info == SafetensorsInfo("big", "U8", (big,), 8, 0, big)
            assert info.bytes == 2**32 + 16
            assert reader.find("big") == 0
            # The last 16 bytes are the marker.
            assert reader.read(0, big - 16, 16) == marker
            # A read straddling byte 2^32 of the data: zeros, then the marker's start.
            assert reader.read(0, 2**32 - 8, 12) == b"\0" * 8 + marker[:4]
            assert reader.read(0, 2**32 - 4, 20) == b"\0" * 4 + marker
            assert reader.read(0, 2**32, 16) == marker
            assert reader.read(0, 2**32 - 1, 1) == b"\0"
            assert reader.read(0, 2**31, 16) == b"\0" * 16
            # Reads past the end are still refused at this size.
            buffer = ctypes.create_string_buffer(8)
            status = flint.lib.flint_safetensors_read(reader.handle, 0, big - 4, 5, buffer)
            assert status == ERROR_INVALID_ARGUMENT
        verdict = python_verdict(path)
        assert verdict.accepted, verdict.message
        assert verdict.tensors == {"big": ("U8", (big,))}
    finally:
        gc.collect()
        if path.exists():
            path.unlink()


def test_a_sparse_file_with_tensors_on_both_sides_of_2_32(flint, tmp_path):
    size = 2**32 + 8
    marker = b"ABCDEFGH"
    header = pad8(
        json.dumps(
            {"first": T("U8", [4], 0, 4), "big": T("U8", [size - 4], 4, size)},
            separators=(",", ":"),
        ).encode()
    )
    prefix = struct.pack("<Q", len(header)) + header
    path = tmp_path / "sparse2.safetensors"
    try:
        if not make_sparse(path, len(prefix) + size, prefix, marker):
            pytest.skip("this file system can't make the 4 GiB file sparse")
        with open(path, "r+b") as f:
            f.seek(len(prefix))
            f.write(b"1234")
        with SafetensorsReader(flint, path) as reader:
            assert reader.read(reader.find("first")) == b"1234"
            big = reader.find("big")
            assert reader.info(big).offset == 4
            assert reader.read(big, size - 12, 8) == marker
            assert reader.read(big, 0, 4) == b"\0" * 4
        assert python_verdict(path).accepted
    finally:
        gc.collect()
        if path.exists():
            path.unlink()


# =============================================================================================
# Round 3: messages, 1 GiB chunking, an I/O error inside write, a failed folder sync
# =============================================================================================

BAD_NAMES = {
    b"\xff": "\\xFF",
    b"a\xc3": "a\\xC3",
    b"\xed\xa0\x80": "\\xED\\xA0\\x80",
}


@pytest.mark.parametrize("raw", list(BAD_NAMES), ids=[r.hex() for r in BAD_NAMES])
@pytest.mark.parametrize("function", ["find", "metadata_find"])
def test_a_message_with_a_name_that_isnt_utf8_is_valid_utf8(flint, small_file, function, raw):
    with SafetensorsReader(flint, small_file) as reader:
        index = ctypes.c_size_t(777)
        value = ctypes.c_char_p(b"untouched")
        if function == "find":
            status = flint.lib.flint_safetensors_find(reader.handle, raw, ctypes.byref(index))
        else:
            status = flint.lib.flint_safetensors_metadata_find(
                reader.handle, raw, ctypes.byref(value)
            )
        assert status == ERROR_INVALID_ARGUMENT
        message = flint.last_error()  # decodes strictly, so invalid UTF-8 would raise here
        assert BAD_NAMES[raw] in message, message
        assert index.value == 777 and value.value == b"untouched"


@pytest.mark.skipif(sys.platform != "win32", reason="Linux paths are bytes, so any is valid")
def test_a_path_that_isnt_utf8_is_an_invalid_argument_on_windows(flint, tmp_path):
    bad = os.fsencode(str(tmp_path)) + b"\\bad\xff.safetensors"
    handle = ctypes.c_void_p(12345)
    assert flint.lib.flint_safetensors_open(bad, ctypes.byref(handle)) == ERROR_INVALID_ARGUMENT
    assert handle.value is None
    assert "bad\\xFF" in flint.last_error()
    array, keys, values, keepalive = safetensors_declare([("a", "U8", (1,))])
    writer = ctypes.c_void_p(12345)
    status = flint.lib.flint_safetensors_create(
        bad, array, 1, keys, values, 0, ctypes.byref(writer)
    )
    assert status == ERROR_INVALID_ARGUMENT
    assert writer.value is None
    assert "bad\\xFF" in flint.last_error()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="a Linux path may be any bytes")
def test_a_missing_path_that_isnt_utf8_is_an_io_error_with_a_decodable_message(flint, tmp_path):
    bad = os.fsencode(str(tmp_path)) + b"/bad\xff.safetensors"
    handle = ctypes.c_void_p(12345)
    assert flint.lib.flint_safetensors_open(bad, ctypes.byref(handle)) == ERROR_IO
    assert handle.value is None
    assert "bad\\xFF" in flint.last_error()  # decoding it strictly is the check


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="a Linux path may be any bytes")
def test_a_format_error_at_a_path_that_isnt_utf8_has_a_decodable_message(flint, tmp_path):
    bad = os.fsencode(str(tmp_path)) + b"/bad\xff.safetensors"
    with open(bad, "wb") as f:
        f.write(b"\x01\x00\x00")  # 3 bytes: too short to hold the header's length
    handle = ctypes.c_void_p(12345)
    assert flint.lib.flint_safetensors_open(bad, ctypes.byref(handle)) == ERROR_FORMAT
    assert handle.value is None
    message = flint.last_error()  # decodes strictly, so invalid UTF-8 would raise here
    assert "bad\\xFF" in message, message


# --- 1 GiB chunking (Windows: a call moves at most 1 GiB, as its byte count is 32 bits) --------

GiB = 1 << 30


@pytest.mark.skipif(sys.platform != "win32", reason="Linux's calls have no 32-bit count")
def test_one_read_of_1_GiB_plus_16_bytes_crosses_a_chunk_boundary(flint, tmp_path):
    big = 2**32 + 16
    start = big - (GiB + 16)  # the read starts here and ends in the last 16 bytes of the tensor
    end_marker, start_marker = bytes(range(1, 17)), bytes(range(101, 117))
    header = pad8(json.dumps({"big": T("U8", [big], 0, big)}, separators=(",", ":")).encode())
    prefix = struct.pack("<Q", len(header)) + header
    path = tmp_path / "chunks.safetensors"
    try:
        if not make_sparse(path, len(prefix) + big, prefix, end_marker):
            pytest.skip("this file system can't make the 4 GiB file sparse")
        with open(path, "r+b") as f:
            f.seek(len(prefix) + start)
            f.write(start_marker)
        buffer = ctypes.create_string_buffer(GiB + 16)
        with SafetensorsReader(flint, path) as reader:
            status = flint.lib.flint_safetensors_read(reader.handle, 0, start, GiB + 16, buffer)
            assert status == OK, flint.last_error()
        view = np.frombuffer(buffer, dtype=np.uint8)
        assert bytes(view[:16]) == start_marker
        assert bytes(view[-16:]) == end_marker
        assert not view[16:-16].any()
        del view, buffer
    finally:
        gc.collect()
        if path.exists():
            path.unlink()


@pytest.mark.skipif(sys.platform != "win32", reason="Linux's calls have no 32-bit count")
def test_one_write_of_1_GiB_plus_16_bytes_crosses_a_chunk_boundary(flint, tmp_path):
    size = GiB + 16
    if shutil.disk_usage(tmp_path).free < 3 * GiB:
        pytest.skip("less than 3 GiB of free disk")
    marker = bytes(range(1, 17))
    middle = bytes(range(201, 209))
    buffer = (ctypes.c_char * size)()
    ctypes.memmove(ctypes.addressof(buffer) + size - 16, marker, 16)
    ctypes.memmove(ctypes.addressof(buffer) + GiB - 8, middle, 8)  # ends at the 1 GiB mark
    path = tmp_path / "one_write.safetensors"
    try:
        with SafetensorsWriter(flint, path, [("t", "U8", (size,))]) as writer:
            assert writer.entries == [(8, 0, size)]
            writer.write(buffer)  # one call, from one buffer
            writer.commit()
        del buffer
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        assert path.stat().st_size == 8 + n + size
        assert not tmp_of(path).exists()
        with open(path, "rb") as f:
            f.seek(-16, os.SEEK_END)
            assert f.read(16) == marker
            f.seek(8 + n + GiB - 8)
            assert f.read(16) == middle + marker[:8]  # across the 1 GiB mark
            f.seek(8 + n + GiB - 100)
            assert f.read(92) == bytes(92)
        with SafetensorsReader(flint, path) as reader:
            assert reader.info(0).bytes == size
            assert reader.read(0, size - 16, 16) == marker
    finally:
        gc.collect()
        for leftover in (path, tmp_of(path)):
            if leftover.exists():
                leftover.unlink()


# --- an I/O error inside write (Linux: RLIMIT_FSIZE makes write fail with EFBIG) --------------

RLIMIT_SCRIPT = r'''
import json
import os
import resource
import signal
import sys

sys.path.insert(0, sys.argv[1])
import flint_lib

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
flint = flint_lib.Flint()
lib = flint.lib
path = sys.argv[2]
size = 1 << 20
status, writer, _ = flint_lib.safetensors_create(flint, path, [("t", "U8", (size,))])
out = {"create": status}
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
out["write"] = lib.flint_safetensors_write(writer, bytes(size), size)
out["write_error"] = flint.last_error()
out["write_again"] = lib.flint_safetensors_write(writer, b"x", 1)
out["write_again_error"] = flint.last_error()
out["commit"] = lib.flint_safetensors_commit(writer)
out["commit_error"] = flint.last_error()
out["tmp_before_destroy"] = os.path.exists(path + ".tmp")
lib.flint_safetensors_destroy(writer)
out["tmp_after_destroy"] = os.path.exists(path + ".tmp")
out["target"] = os.path.exists(path)
print(json.dumps(out))
'''


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs RLIMIT_FSIZE")
def test_a_writer_fails_for_good_after_an_io_error_inside_write(tmp_path):
    path = tmp_path / "limited.safetensors"
    done = subprocess.run(
        [sys.executable, "-c", RLIMIT_SCRIPT, str(Path(__file__).resolve().parent), str(path)],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["create"] == OK
    assert out["write"] == ERROR_IO
    assert out["write_again"] == ERROR_IO
    assert "an earlier write failed" in out["write_again_error"]
    assert out["commit"] == ERROR_IO
    assert "an earlier write failed" in out["commit_error"]
    assert out["tmp_before_destroy"] is True
    assert out["tmp_after_destroy"] is False
    assert out["target"] is False
    assert not path.exists() and not tmp_of(path).exists()


# --- a failed folder sync after the rename (Linux) ---------------------------------------------


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's commit syncs the folder")
def test_a_failed_folder_sync_after_the_rename_says_the_file_is_in_place(flint, tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores a folder's mode")
    folder = tmp_path / "sync"
    folder.mkdir()
    path = folder / "x.safetensors"
    tensors = [("a", "F32", (2,), b"abcdefgh"), ("b", "U8", (3,), b"xyz")]
    with SafetensorsWriter(flint, path, [t[:3] for t in tensors]) as writer:
        writer.write(b"".join(t[3] for t in tensors))
        # Execute without read: the rename works, opening the folder to sync it doesn't.
        os.chmod(folder, 0o311)
        try:
            status = writer.commit_status()
            message = flint.last_error()
        finally:
            os.chmod(folder, 0o755)
        assert status == ERROR_IO
        assert "in place" in message and "syncing its folder failed" in message, message
        assert str(path)[:100] in message
        assert writer.write_status(b"") == ERROR_IO  # the writer has failed
    check_written(flint, path, tensors)
