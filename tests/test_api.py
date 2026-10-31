"""The C API's GPU-free surface: exports, the version, and status codes."""

import re

import pytest

import flint_lib


def test_declared_functions_export(flint):
    # Loading the library binds every name in SIGNATURES, so a function the
    # header declares but the library doesn't export fails the fixture.
    header = flint_lib.HEADER.read_text()
    declared = re.findall(r"FLINT_API\s[^;(]*?\b(flint_\w+)\s*\(", header)
    assert sorted(declared) == sorted(flint_lib.SIGNATURES)


def test_status_codes_match_header():
    header = flint_lib.HEADER.read_text()
    codes = dict(re.findall(r"\bFLINT_(OK|ERROR_\w+)\s*=\s*(\d+)", header))
    assert {name: getattr(flint_lib, name) for name in codes} == {
        name: int(value) for name, value in codes.items()
    }


def test_version(flint):
    assert re.fullmatch(r"\d+\.\d+\.\d+", flint.version())


@pytest.mark.parametrize(
    "status, text",
    [
        (flint_lib.OK, "ok"),
        (flint_lib.ERROR_INVALID_ARGUMENT, "invalid argument"),
        (flint_lib.ERROR_CUDA, "CUDA error"),
        (-1, "unknown status"),
        (3, "unknown status"),
        (2**31 - 1, "unknown status"),
    ],
)
def test_status_string(flint, status, text):
    assert flint.status_string(status) == text


def test_check(flint):
    flint.check(flint_lib.OK)
    with pytest.raises(flint_lib.FlintError, match=r"^CUDA error \(status 2\)") as error:
        flint.check(flint_lib.ERROR_CUDA)
    assert error.value.status == flint_lib.ERROR_CUDA
