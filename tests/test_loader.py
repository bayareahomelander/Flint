"""Tests for D34: Flint's token files and data loader (flint_loader_*).

The contract is data/d34/spec.md's "The order" and "The C API". tests/loader_reference.py is an
independent pure-Python transcription of the order (written from the design alone), and the
golden values in section 1 pin both it and the C++ to the same literal numbers, so the two can't
drift together.

Sections: 1 golden values, 2 against the reference on synthetic files, 3 reads are stateless,
4 layout and max_tokens, 5 the token range check, 6 headers and files, 7 arguments, 8 64-bit
offsets and the 2^32-sequence limit (sparse files), 9 real data, 10 the page cache (Linux).

Inputs are numpy.random.default_rng(seed) integers (never torch.randn, which differs between
machines). Nothing here launches a kernel, and compute-sanitizer can't see host code, so the whole
module is skipped under the sanitizer.
"""

import ctypes
import gc
import hashlib
import importlib.util
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import flint_lib
import loader_reference as lr
from flint_lib import (
    ERROR_FORMAT,
    ERROR_INVALID_ARGUMENT,
    ERROR_IO,
    OK,
    FlintError,
    Loader,
    LoaderConfig,
    _loader_paths,
    loader_open_status,
)
from inputs import SANITIZER
from loader_reference import Reference
from test_gpt2 import data_file

pytestmark = pytest.mark.skipif(
    SANITIZER, reason="host-only tests launch no kernel; D34 has its own host-sanitizer run"
)

_spec = importlib.util.spec_from_file_location("fineweb", flint_lib.ROOT / "scripts" / "fineweb.py")
fineweb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fineweb)

VOCAB = 50257
U64_MAX = 2**64 - 1
MiB = 1 << 20
GiB = 1 << 30
MAGIC = b"FLINTTOK"


# --- helpers ---------------------------------------------------------------------------------


def header_bytes(count, *, magic=MAGIC, version=1, tokenizer=1, reserved=bytes(40)):
    return struct.pack("<8sIIQ", magic, version, tokenizer, count) + reserved


def write_tokens(path, tokens):
    """A valid token file (an empty one by hand: write_token_file can't cast zero bytes)."""
    tokens = np.asarray(tokens, dtype=np.uint16)
    if tokens.size == 0:
        Path(path).write_bytes(header_bytes(0))
    else:
        fineweb.write_token_file(path, tokens)


def random_tokens(seed, count, vocab=VOCAB):
    return np.random.default_rng(seed).integers(0, vocab, count).astype(np.uint16)


def make_files(folder, counts, seed=0, vocab=VOCAB, stem="f"):
    """Token files of the given token counts in folder: (paths, token arrays)."""
    paths, arrays = [], []
    for i, count in enumerate(counts):
        tokens = random_tokens(seed * 1000 + i, count, vocab)
        path = Path(folder) / f"{stem}{i}.bin"
        write_tokens(path, tokens)
        paths.append(path)
        arrays.append(tokens)
    return paths, arrays


def sequences_of(count, length, max_tokens=0):
    """(usable - 1) // length, written independently of the reference."""
    usable = min(count, max_tokens) if max_tokens else count
    return 0 if usable == 0 else (usable - 1) // length


def raw_open(
    flint, paths, *, length=4, vocab=VOCAB, seed=0, shuffle=True, max_tokens=0, files=None,
    null_paths=False, null_config=False, null_loader=False,
):  # fmt: skip
    """flint_loader_open through ctypes: (status, the handle left in *loader or None, the last
    error message). *loader starts non-null, so a failure that doesn't null it shows. A loader
    that opens is closed again."""
    array, keepalive = _loader_paths(paths)
    config = LoaderConfig(length, max_tokens, vocab, seed, int(shuffle))
    handle = ctypes.c_void_p(0x1000)
    status = flint.lib.flint_loader_open(
        None if null_paths else array,
        len(paths) if files is None else files,
        None if null_config else ctypes.byref(config),
        None if null_loader else ctypes.byref(handle),
    )
    message = flint.last_error()
    value = handle.value
    del keepalive
    if status == OK and not null_loader:
        flint.lib.flint_loader_close(handle)
    return status, value, message


def open_fails(flint, paths, status, *words, **config):
    """Opening fails with status, *loader null, and a message naming flint_loader_open and each
    of words (lower-cased text compared against the lower-cased message)."""
    got, handle, message = raw_open(flint, paths, **config)
    assert got == status, (got, message)
    assert handle is None, "*loader was not set to null"
    assert "flint_loader_open" in message, message
    for word in words:
        assert str(word).lower() in message.lower(), (word, message)
    return message


def epoch_pairs(counts_per_file):
    """Every (file, sequence) once, files in list order and sequences ascending."""
    return [(f, s) for f, c in enumerate(counts_per_file) for s in range(c)]


def cpp_order(loader, epoch):
    """The (file, sequence) pairs of one epoch in the order the loader's locate gives them."""
    size = loader.epoch_sequences()
    return [loader.locate(epoch * size + r)[1:] for r in range(size)]


def same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def assert_read_equals(loader, ref, g, count):
    tokens, targets = loader.read(g, count)
    expected_tokens, expected_targets = ref.read(g, count)
    assert same(tokens, expected_tokens), f"tokens differ at g={g} count={count}"
    assert same(targets, expected_targets), f"targets differ at g={g} count={count}"


# File sets by length: the token counts of each file, in sequences of the given length.
def set_one(length):
    return [40 * length + 1 + (length - 1) // 2]


def set_three(length):
    return [17 * length + 1, 5 * length + length, 9 * length + 1 + (length - 1)]


def set_twelve(length):
    # Files with no sequence (0, 1, length tokens), one of exactly one sequence (length + 1),
    # and some with a tail that doesn't fill a sequence.
    return [0, length, length + 1, 3 * length + 1, 1, 2 * length, 7 * length + length,
            0, length + 2, 2 * length + 1, 5 * length + 1, 6]  # fmt: skip


def set_random(length):
    rng = np.random.default_rng(1000 + length)
    return [int(n) for n in rng.integers(0, 25 * length + 3, 12)]


FILE_SETS = {"one": set_one, "three": set_three, "twelve": set_twelve, "random": set_random}


# --- 1. golden values ------------------------------------------------------------------------


def test_splitmix64_golden_values():
    state, outputs = 0, []
    for _ in range(3):
        state, out = lr.next(state)
        outputs.append(out)
    assert outputs == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    # hash(x) is the output next gives from state x.
    assert lr.hash(0) == 0xE220A8397B1DCDAF


def test_stream_start_golden_values():
    assert lr.start(0, 0, 0) == 0x238275BC38FCBE91
    assert lr.start(20261124, 3, 0xFFFFFFFFFFFFFFFF) == 0x5B8F5484991BC0C1


def test_fisher_yates_golden_values():
    assert lr.fisher_yates(10, lr.start(7, 2, 5)) == [3, 8, 6, 2, 7, 5, 1, 9, 0, 4]
    assert lr.fisher_yates(0, 123) == []
    assert lr.fisher_yates(1, 123) == [0]


CONFIG_A = [
    (0, 0, 2), (0, 0, 1), (0, 0, 3), (0, 0, 0), (0, 0, 4), (0, 2, 1), (0, 2, 2), (0, 2, 0),
    (0, 3, 1), (0, 3, 2), (0, 3, 0), (0, 3, 4), (0, 3, 6), (0, 3, 3), (0, 3, 5), (1, 3, 6),
    (1, 3, 2), (1, 3, 1), (1, 3, 4), (1, 3, 0), (1, 3, 3), (1, 3, 5), (1, 0, 2), (1, 0, 0),
    (1, 0, 3), (1, 0, 1), (1, 0, 4), (1, 2, 2), (1, 2, 0), (1, 2, 1),
]  # fmt: skip


def test_config_a_golden_locations(flint, tmp_path):
    paths, _ = make_files(tmp_path, [21, 3, 13, 29])
    ref = Reference(paths, 4, seed=20261124, shuffle=True)
    assert ref.epoch_sequences() == 15
    assert [ref.locate(g) for g in range(30)] == CONFIG_A
    with Loader(flint, paths, 4, VOCAB, seed=20261124, shuffle=True) as loader:
        assert loader.epoch_sequences() == 15
        assert [loader.locate(g) for g in range(30)] == CONFIG_A


def test_one_big_file_golden_locations(flint, tmp_path):
    path = tmp_path / "big.bin"
    write_tokens(path, random_tokens(5, 97_657))
    expected = [70256, 26940, 63068, 61092, 33932, 31571, 21806, 91815]
    ref = Reference([path], 1, seed=1, shuffle=True)
    assert ref.epoch_sequences() == 97_656
    assert [ref.locate(g) for g in range(8)] == [(0, 0, s) for s in expected]
    with Loader(flint, [path], 1, VOCAB, seed=1, shuffle=True) as loader:
        assert loader.epoch_sequences() == 97_656
        assert [loader.locate(g) for g in range(8)] == [(0, 0, s) for s in expected]


# Seeds of 2^32 and more (config A's files, length 4, shuffle on): locate(g) for g = 0 .. 14, from
# the orchestrator's own transcription of the order (data/d34/golden_seeds.log).
SEED_GOLDEN = {
    0xDEADBEEFCAFEBABE: [
        (0, 2, 1), (0, 2, 2), (0, 2, 0), (0, 3, 4), (0, 3, 1), (0, 3, 6), (0, 3, 5), (0, 3, 0),
        (0, 3, 2), (0, 3, 3), (0, 0, 4), (0, 0, 1), (0, 0, 0), (0, 0, 2), (0, 0, 3),
    ],
    0xFFFFFFFFFFFFFFFF: [
        (0, 0, 4), (0, 0, 2), (0, 0, 3), (0, 0, 1), (0, 0, 0), (0, 3, 3), (0, 3, 6), (0, 3, 0),
        (0, 3, 2), (0, 3, 5), (0, 3, 1), (0, 3, 4), (0, 2, 1), (0, 2, 0), (0, 2, 2),
    ],
    0x0000000100000000: [
        (0, 3, 6), (0, 3, 3), (0, 3, 5), (0, 3, 1), (0, 3, 4), (0, 3, 2), (0, 3, 0), (0, 0, 3),
        (0, 0, 1), (0, 0, 2), (0, 0, 4), (0, 0, 0), (0, 2, 1), (0, 2, 2), (0, 2, 0),
    ],
}  # fmt: skip


@pytest.mark.parametrize("seed", list(SEED_GOLDEN), ids=lambda seed: f"seed-{seed:#x}")
def test_seeds_of_2_32_and_more_golden_locations(flint, tmp_path, seed):
    paths, _ = make_files(tmp_path, [21, 3, 13, 29])
    expected = SEED_GOLDEN[seed]
    ref = Reference(paths, 4, seed=seed, shuffle=True)
    assert [ref.locate(g) for g in range(15)] == expected
    with Loader(flint, paths, 4, VOCAB, seed=seed, shuffle=True) as loader:
        assert [loader.locate(g) for g in range(15)] == expected


@pytest.mark.parametrize("shuffle", [2, -1], ids=["shuffle-2", "shuffle-minus-1"])
def test_any_nonzero_shuffle_shuffles(flint, tmp_path, shuffle):
    paths, _ = make_files(tmp_path, [21, 3, 13, 29])
    ref = Reference(paths, 4, seed=20261124, shuffle=True)
    # The binding passes the int through (int(shuffle)), it doesn't make it a bool.
    assert flint_lib._loader_config(4, VOCAB, 20261124, shuffle, 0).shuffle == shuffle
    with Loader(flint, paths, 4, VOCAB, seed=20261124, shuffle=shuffle) as loader:
        assert [loader.locate(g) for g in range(30)] == CONFIG_A
        for g in range(30):
            assert_read_equals(loader, ref, g, 1)
    with Loader(flint, paths, 4, VOCAB, seed=20261124, shuffle=0) as off:
        assert [off.locate(g) for g in range(30)] != CONFIG_A


BIG_PERMUTATION_SHA256 = "c4cb2f27b4e2d9d1f21abd0328801c5d2bab34a675f248364bec8ef6a33edce6"
BIG_PERMUTATION_HEAD = [906103, 845029, 785687, 517030, 264843, 755130, 129612, 903627]


def test_a_whole_large_permutation_is_the_references(flint, tmp_path):
    # A 32-bit draw instead of the 64-bit one changes 74 of this permutation's 2^20 - 1 draws and
    # passes every test with a small n (data/d34/golden_big.log).
    n = 1 << 20
    tokens_all = random_tokens(99, n + 1, vocab=65_536)
    path = tmp_path / "large.bin"
    write_tokens(path, tokens_all)
    q = lr.fisher_yates(n, lr.start(99, 0, 0))
    assert q[:8] == BIG_PERMUTATION_HEAD
    packed = np.asarray(q, dtype="<u4").tobytes()
    assert hashlib.sha256(packed).hexdigest() == BIG_PERMUTATION_SHA256
    q = np.asarray(q, dtype=np.int64)
    with Loader(flint, [path], 1, 65_536, seed=99, shuffle=True) as loader:
        assert loader.epoch_sequences() == n
        tokens, targets = loader.read(0, n)
        assert np.array_equal(tokens[:, 0], tokens_all[q])
        assert np.array_equal(targets[:, 0], tokens_all[q + 1])
        indices = random.Random(99).sample(range(n), 300) + [0, 1, n - 2, n - 1]
        for g in indices:
            assert loader.locate(g) == (0, 0, int(q[g])), g


# --- 2. against the reference on synthetic files ---------------------------------------------


@pytest.mark.parametrize("seed", [3, 4])
@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffle", "noshuffle"])
@pytest.mark.parametrize("set_name", list(FILE_SETS))
@pytest.mark.parametrize("length", [1, 2, 7, 8, 1024])
def test_loader_equals_reference(flint, tmp_path, length, set_name, shuffle, seed):
    counts = FILE_SETS[set_name](length)
    paths, _ = make_files(tmp_path, counts, seed=length)
    per_file = [sequences_of(c, length) for c in counts]
    ref = Reference(paths, length, seed=seed, shuffle=shuffle)
    assert ref.counts == per_file
    size = sum(per_file)
    with Loader(flint, paths, length, VOCAB, seed=seed, shuffle=shuffle) as loader:
        assert loader.epoch_sequences() == ref.epoch_sequences() == size
        for g in range(3 * size):
            assert loader.locate(g) == ref.locate(g), g
            assert_read_equals(loader, ref, g, 1)
        orders = [cpp_order(loader, e) for e in range(3)]
        with Loader(flint, paths, length, VOCAB, seed=seed + 1, shuffle=shuffle) as other:
            other_order = cpp_order(other, 0)
    # Each epoch is every (file, sequence) pair exactly once.
    pairs = epoch_pairs(per_file)
    for order in orders:
        assert sorted(order) == pairs
    if shuffle:
        if size >= 20:
            assert orders[0] != orders[1]
            assert orders[1] != orders[2]
            assert orders[0] != other_order
    else:
        for order in orders:
            assert order == pairs
        assert other_order == pairs


# --- 3. reads are stateless ------------------------------------------------------------------


@pytest.fixture
def three_files(tmp_path):
    paths, _ = make_files(tmp_path, [7 * 61 + 1, 7 * 4 + 5, 7 * 33 + 1 + 3], seed=31)
    return paths


LENGTH3 = 7


def test_reads_in_any_order_equal_the_reference(flint, three_files):
    ref = Reference(three_files, LENGTH3, seed=9, shuffle=True)
    size = ref.epoch_sequences()
    with Loader(flint, three_files, LENGTH3, VOCAB, seed=9, shuffle=True) as loader:
        total = 3 * size
        orders = {
            "forwards": list(range(total)),
            "backwards": list(range(total - 1, -1, -1)),
            "random": random.Random(77).sample(range(total), total),
            # The same position in epoch 0 and epoch 1, alternately.
            "alternating": [g + e * size for g in range(size) for e in (0, 1, 0, 2)],
        }
        for name, order in orders.items():
            for g in order:
                tokens, targets = loader.read(g, 1)
                expected_tokens, expected_targets = ref.read(g, 1)
                assert same(tokens, expected_tokens) and same(targets, expected_targets), (name, g)


def test_multi_sequence_reads_equal_single_reads_and_the_reference(flint, three_files):
    ref = Reference(three_files, LENGTH3, seed=9, shuffle=True)
    size = ref.epoch_sequences()
    rng = random.Random(5)
    counts = [1, 3, 5, 16, size - 1, size, size + 1, 2 * size + 3]
    starts = [0, 1, size - 2, size - 1, size, 2 * size - 1, 2 * size, 3 * size + 4]
    with Loader(flint, three_files, LENGTH3, VOCAB, seed=9, shuffle=True) as loader:
        for count in counts:
            for g in starts + [rng.randrange(3 * size) for _ in range(3)]:
                tokens, targets = loader.read(g, count)
                expected_tokens, expected_targets = ref.read(g, count)
                assert same(tokens, expected_tokens), (g, count)
                assert same(targets, expected_targets), (g, count)
                singles = [loader.read(g + i, 1) for i in range(count)]
                assert same(tokens, np.concatenate([t for t, _ in singles])), (g, count)
                assert same(targets, np.concatenate([t for _, t in singles])), (g, count)


def test_a_fresh_loader_and_two_interleaved_loaders_agree(flint, three_files):
    with Loader(flint, three_files, LENGTH3, VOCAB, seed=9) as first:
        size = first.epoch_sequences()
        expected = first.read(size - 3, 20)
        # The first loader moves on to other files and epochs; the answer must not change.
        for g in (0, size, 3 * size - 1, size // 2, 2 * size + 5):
            first.read(g, 9)
        again = first.read(size - 3, 20)
        assert same(again[0], expected[0]) and same(again[1], expected[1])
        with Loader(flint, three_files, LENGTH3, VOCAB, seed=9) as fresh:
            tokens, targets = fresh.read(size - 3, 20)
            assert same(tokens, expected[0]) and same(targets, expected[1])
            # Interleaved reads of one file set from two loaders.
            rng = random.Random(3)
            for _ in range(60):
                g, count = rng.randrange(4 * size), rng.randrange(1, 8)
                a = first.read(g, count)
                b = fresh.read(g, count)
                assert same(a[0], b[0]) and same(a[1], b[1])
                assert first.locate(g) == fresh.locate(g)


SUBPROCESS_SCRIPT = """
import hashlib, sys
sys.path.insert(0, sys.argv[1])
from flint_lib import Flint, Loader
flint = Flint()
length, seed = int(sys.argv[2]), int(sys.argv[3])
indices = [int(v) for v in sys.argv[4].split(",")]
paths = sys.argv[5:]
digest = hashlib.sha256()
with Loader(flint, paths, length, 50257, seed=seed, shuffle=True) as loader:
    for g in indices:
        tokens, targets = loader.read(g, 1)
        digest.update(tokens.tobytes())
        digest.update(targets.tobytes())
print(digest.hexdigest())
"""


def test_another_process_reads_the_same_bytes(flint, three_files):
    with Loader(flint, three_files, LENGTH3, VOCAB, seed=12) as loader:
        size = loader.epoch_sequences()
        indices = [random.Random(8).randrange(5 * size) for _ in range(12)] + [0, size, 4 * size]
        digest = hashlib.sha256()
        for g in indices:
            tokens, targets = loader.read(g, 1)
            digest.update(tokens.tobytes())
            digest.update(targets.tobytes())
    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT, str(Path(__file__).resolve().parent),
         str(LENGTH3), "12", ",".join(map(str, indices)), *map(str, three_files)],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == digest.hexdigest()


# --- 4. layout and max_tokens ----------------------------------------------------------------


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffle", "noshuffle"])
def test_a_read_is_the_file_tokens_at_the_located_sequence(flint, tmp_path, shuffle):
    length = 6
    counts = [6 * 11 + 1, 6 * 3 + 4, 6 * 20 + 1 + 2]
    paths, arrays = make_files(tmp_path, counts, seed=4)
    for path, array in zip(paths, arrays):
        assert np.array_equal(fineweb.read_token_file(path), array)  # the file is what we wrote
    with Loader(flint, paths, length, VOCAB, seed=2, shuffle=shuffle) as loader:
        size = loader.epoch_sequences()
        for g in [0, 1, 7, size - 1, size, size + 3, 2 * size + 11, 5 * size - 1]:
            e, f, s = loader.locate(g)
            assert e == g // size
            assert 0 <= s < sequences_of(counts[f], length)
            tokens, targets = loader.read(g, 1)
            assert np.array_equal(tokens[0], arrays[f][s * length : s * length + length])
            assert np.array_equal(targets[0], arrays[f][s * length + 1 : s * length + length + 1])


def max_tokens_files(folder, length, max_tokens):
    """Three files: A (longer than most max_tokens, every token from max_tokens on is at vocab),
    B (shorter than max_tokens when that is big) and C."""
    count_a = 5 * length + 3
    tokens_a = random_tokens(60, count_a)
    if max_tokens < count_a:
        tokens_a[max_tokens:] = VOCAB
    tokens_b = random_tokens(61, 2 * length + 1)
    tokens_c = random_tokens(62, 3 * length + 1)
    paths = []
    for i, tokens in enumerate([tokens_a, tokens_b, tokens_c]):
        path = Path(folder) / f"m{i}.bin"
        write_tokens(path, tokens)
        paths.append(path)
    return paths, [count_a, len(tokens_b), len(tokens_c)]


@pytest.mark.parametrize("shuffle", [True, False], ids=["shuffle", "noshuffle"])
@pytest.mark.parametrize(
    "max_tokens",
    ["length", "length+1", "2length+1", "count-1", "count", "count+1", "2^60"],
)
def test_max_tokens_limits_each_file(flint, tmp_path, max_tokens, shuffle):
    length = 4
    count_a = 5 * length + 3
    value = {
        "length": length, "length+1": length + 1, "2length+1": 2 * length + 1,
        "count-1": count_a - 1, "count": count_a, "count+1": count_a + 1, "2^60": 2**60,
    }[max_tokens]  # fmt: skip
    paths, counts = max_tokens_files(tmp_path, length, value)
    per_file = [sequences_of(c, length, value) for c in counts]
    ref = Reference(paths, length, seed=6, shuffle=shuffle, max_tokens=value)
    assert ref.counts == per_file
    if sum(per_file) == 0:
        assert loader_open_status(
            flint, paths, length=length, vocab=VOCAB, shuffle=shuffle, max_tokens=value
        ) == ERROR_INVALID_ARGUMENT
        return
    with Loader(
        flint, paths, length, VOCAB, seed=6, shuffle=shuffle, max_tokens=value
    ) as loader:
        assert loader.epoch_sequences() == sum(per_file)
        # Every read of 3 epochs succeeds although tokens at vocab sit from max_tokens on, so
        # nothing past the usable tokens is read.
        for g in range(3 * sum(per_file)):
            assert loader.locate(g) == ref.locate(g)
            assert_read_equals(loader, ref, g, 1)
        assert_read_equals(loader, ref, 0, 3 * sum(per_file))
        orders = cpp_order(loader, 1)
        assert sorted(orders) == epoch_pairs(per_file)
    # Files shorter than max_tokens keep all their tokens: B and C count in full when
    # max_tokens exceeds them.
    if value >= counts[2]:
        assert per_file[1:] == [sequences_of(counts[1], length), sequences_of(counts[2], length)]


# --- 5. the token range check ----------------------------------------------------------------

BAD_LENGTHS = [2, 5]


def bad_token_setup(folder, length, where, value=VOCAB, vocab=VOCAB):
    """Files clean0, bad1 (7 sequences, tail of length - 1 tokens, one token at value at the
    place `where`) and clean2: (paths, position, set of bad sequences of file 1, tokens)."""
    sequences = 7
    tail = length - 1
    count = sequences * length + 1 + tail
    position = {
        "first": 0,
        "last": sequences * length,
        "boundary": 3 * length,
        "inside": 2 * length + 1,
    }[where]
    tokens = random_tokens(70 + length, count, vocab)
    tokens[position] = value
    paths = []
    for i, array in enumerate([random_tokens(71, 4 * length + 2, vocab), tokens,
                               random_tokens(72, 9 * length + 1, vocab)]):  # fmt: skip
        path = Path(folder) / f"{'bad' if i == 1 else 'clean'}{i}.bin"
        write_tokens(path, array)
        paths.append(path)
    bad = {s for s in range(sequences) if s * length <= position <= s * length + length}
    return paths, position, bad


@pytest.mark.parametrize("where", ["first", "last", "boundary", "inside"])
@pytest.mark.parametrize("length", BAD_LENGTHS)
def test_a_token_at_vocab_fails_exactly_the_reads_holding_it(flint, tmp_path, length, where):
    paths, position, bad = bad_token_setup(tmp_path, length, where)
    ref = Reference(paths, length, seed=8, shuffle=True)
    size = ref.epoch_sequences()
    expected_bad_count = {"first": 1, "last": 1, "boundary": 2, "inside": 1}[where]
    assert len(bad) == expected_bad_count
    with Loader(flint, paths, length, VOCAB, seed=8, shuffle=True) as loader:
        failures = 0
        for g in range(2 * size):
            for count in (1, 4):
                holds = any(
                    ref.locate(g + i)[1] == 1 and ref.locate(g + i)[2] in bad
                    for i in range(count)
                )
                tokens = np.zeros((count, length), dtype=np.uint16)
                targets = np.zeros((count, length), dtype=np.uint16)
                status = loader.read_into(g, count, tokens, targets)
                if holds:
                    failures += 1
                    assert status == ERROR_FORMAT, (g, count, status)
                    message = flint.last_error()
                    for word in ("flint_loader_read", "bad1.bin", f"token {position}",
                                 f"{VOCAB}", "vocab"):  # fmt: skip
                        assert word in message, (word, message)
                    assert f"token {position} is {VOCAB}" in message, message
                else:
                    assert status == OK, (g, count, flint.last_error())
                    expected_tokens, expected_targets = ref.read(g, count)
                    assert same(tokens, expected_tokens) and same(targets, expected_targets)
        assert failures > 0
        # The loader keeps working after a failure.
        good = next(
            g for g in range(size)
            if not (ref.locate(g)[1] == 1 and ref.locate(g)[2] in bad)
        )  # fmt: skip
        assert_read_equals(loader, ref, good, 1)
        assert loader.locate(good) == ref.locate(good)


@pytest.mark.parametrize("length", BAD_LENGTHS)
def test_a_bad_token_in_the_dropped_tail_never_fails(flint, tmp_path, length):
    sequences = 6
    count = sequences * length + 1 + (length - 1)
    tokens = random_tokens(80, count)
    tokens[sequences * length + 1 :] = 65_535  # past the last usable token
    path = tmp_path / "tail.bin"
    write_tokens(path, tokens)
    ref = Reference([path], length, seed=1, shuffle=True)
    with Loader(flint, [path], length, VOCAB, seed=1) as loader:
        assert loader.epoch_sequences() == sequences
        assert_read_equals(loader, ref, 0, 3 * sequences)
        for g in range(3 * sequences):
            assert_read_equals(loader, ref, g, 1)


def test_the_range_check_is_against_the_configured_vocab(flint, tmp_path):
    length = 4
    # vocab - 1 passes.
    tokens = random_tokens(90, 8 * length + 1)
    tokens[:] = VOCAB - 1
    path = tmp_path / "edge.bin"
    write_tokens(path, tokens)
    with Loader(flint, [path], length, VOCAB, shuffle=False) as loader:
        t, g = loader.read(0, loader.epoch_sequences())
        assert (t == VOCAB - 1).all() and (g == VOCAB - 1).all()
    # vocab 65,536 passes a token of 65,535.
    tokens[:] = 65_535
    write_tokens(path, tokens)
    with Loader(flint, [path], length, 65_536, shuffle=False) as loader:
        t, g = loader.read(0, loader.epoch_sequences())
        assert (t == 65_535).all() and (g == 65_535).all()
    # A far larger value fails against vocab 50,257, naming it.
    tokens[:] = 5
    tokens[10] = 65_535
    write_tokens(path, tokens)
    with Loader(flint, [path], length, VOCAB, shuffle=False) as loader:
        status = loader.read_into(
            2, 1, np.zeros((1, length), np.uint16), np.zeros((1, length), np.uint16)
        )
        assert status == ERROR_FORMAT
        message = flint.last_error()
        assert "token 10 is 65535" in message and f"{VOCAB}" in message, message
    # The same file passes at vocab 65,536.
    with Loader(flint, [path], length, 65_536, shuffle=False) as loader:
        loader.read(0, loader.epoch_sequences())


# --- 6. headers and files --------------------------------------------------------------------

GOOD_COUNT = 10


def valid_file_bytes(count=GOOD_COUNT):
    return header_bytes(count) + random_tokens(3, count).tobytes()


def with_reserved(i):
    reserved = bytearray(40)
    reserved[i - 24] = 1
    return header_bytes(GOOD_COUNT, reserved=bytes(reserved)) + random_tokens(3, 10).tobytes()


def resized(delta):
    data = valid_file_bytes()
    return data[:delta] if delta < 0 else data + bytes(delta)


def flipped(index):
    data = bytearray(valid_file_bytes())
    data[index] ^= 0x01
    return bytes(data)


def counted(count, size):
    """A file whose header claims count tokens and which is size bytes long."""
    data = header_bytes(count) + random_tokens(3, 400).tobytes()
    return data[:size]


HEADER_CASES = [
    pytest.param(b"", ("shorter",), id="size-0"),
    pytest.param(b"F", ("shorter",), id="size-1"),
    pytest.param(valid_file_bytes()[:63], ("shorter",), id="size-63"),
    pytest.param(flipped(0), ("magic",), id="magic-first-byte"),
    pytest.param(flipped(7), ("magic",), id="magic-last-byte"),
    pytest.param(valid_file_bytes().replace(MAGIC, b"FLINTTOX", 1), ("magic",), id="magic-X"),
    pytest.param(header_bytes(10, version=0) + bytes(20), ("version",), id="version-0"),
    pytest.param(header_bytes(10, version=2) + bytes(20), ("version",), id="version-2"),
    pytest.param(header_bytes(10, tokenizer=0) + bytes(20), ("tokenizer",), id="tokenizer-0"),
    pytest.param(header_bytes(10, tokenizer=2) + bytes(20), ("tokenizer",), id="tokenizer-2"),
    *[pytest.param(with_reserved(i), ("reserved",), id=f"reserved-{i}") for i in range(24, 64)],
    pytest.param(resized(-2), (str(64 + 2 * 10 - 2), str(64 + 2 * 10)), id="size-2-short"),
    pytest.param(resized(-1), (str(64 + 2 * 10 - 1), str(64 + 2 * 10)), id="size-1-short"),
    pytest.param(resized(1), (str(64 + 2 * 10 + 1), str(64 + 2 * 10)), id="size-1-long"),
    pytest.param(resized(2), (str(64 + 2 * 10 + 2), str(64 + 2 * 10)), id="size-2-long"),
    # A count whose 64 + 2 x count wraps around 2^64 to the size of the file would be accepted by
    # an implementation that overflows: 2^63 wraps to 64, 2^63 + 5 to 74, 2^64 - 1 to 62.
    pytest.param(counted(2**63, 64), ("bytes",), id="count-2^63-size-64"),
    pytest.param(counted(2**63, 100), ("bytes",), id="count-2^63-size-100"),
    pytest.param(counted(2**63 + 5, 74), ("bytes",), id="count-2^63+5-size-74"),
    pytest.param(counted(2**64 - 1, 64), ("bytes",), id="count-2^64-1-size-64"),
    pytest.param(counted(2**64 - 1, 100), ("bytes",), id="count-2^64-1-size-100"),
]


@pytest.mark.parametrize(("content", "words"), HEADER_CASES)
def test_a_malformed_file_fails_the_open_and_passes_once_fixed(flint, tmp_path, content, words):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(content)
    open_fails(flint, [bad], ERROR_FORMAT, "bad.bin", *words, length=4)
    # The failed open closed the file (on Windows an open file can't be overwritten), and the
    # same file made valid opens.
    bad.write_bytes(valid_file_bytes(40))
    assert loader_open_status(flint, [bad], length=4, vocab=VOCAB) == OK


SECOND_CASES = [
    pytest.param(*HEADER_CASES[i].values, id=HEADER_CASES[i].id)
    for i in (0, 2, 3, 7, 9, 10, 40, 54)
]


@pytest.mark.parametrize(("content", "words"), SECOND_CASES)
def test_a_bad_file_second_in_the_list_fails_the_open_too(flint, tmp_path, content, words):
    good = tmp_path / "good.bin"
    write_tokens(good, random_tokens(1, 50))
    bad = tmp_path / "bad.bin"
    bad.write_bytes(content)
    open_fails(flint, [good, bad], ERROR_FORMAT, "bad.bin", *words, length=4)
    # The good file isn't left open either.
    good.write_bytes(valid_file_bytes(30))
    bad.write_bytes(valid_file_bytes(30))
    assert loader_open_status(flint, [good, bad], length=4, vocab=VOCAB) == OK


def test_the_wrong_version_and_tokenizer_name_their_values(flint, tmp_path):
    bad = tmp_path / "bad.bin"
    for version in (0, 2, 77):
        bad.write_bytes(header_bytes(10, version=version) + bytes(20))
        message = open_fails(flint, [bad], ERROR_FORMAT, "version")
        assert str(version) in message.split("bad.bin")[-1], message
    for tokenizer in (0, 2, 88):
        bad.write_bytes(header_bytes(10, tokenizer=tokenizer) + bytes(20))
        message = open_fails(flint, [bad], ERROR_FORMAT, "tokenizer")
        assert str(tokenizer) in message.split("bad.bin")[-1], message


def test_a_header_with_count_zero_is_valid(flint, tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(header_bytes(0))
    assert empty.stat().st_size == 64
    full = tmp_path / "full.bin"
    write_tokens(full, random_tokens(2, 41))
    with Loader(flint, [empty, full], 4, VOCAB, shuffle=False) as loader:
        assert loader.epoch_sequences() == 10
        assert loader.locate(0) == (0, 1, 0)  # the empty file takes a slot and gives nothing
    with Loader(flint, [full, empty], 4, VOCAB, shuffle=True, seed=3) as loader:
        assert loader.epoch_sequences() == 10
        assert {loader.locate(g)[1] for g in range(10)} == {0}
    # Alone, or among files too short for a sequence, it holds no sequence.
    open_fails(flint, [empty], ERROR_INVALID_ARGUMENT, "sequence", length=4)
    short = tmp_path / "short.bin"
    write_tokens(short, random_tokens(2, 4))
    one = tmp_path / "one.bin"
    write_tokens(one, random_tokens(2, 1))
    open_fails(flint, [empty, short, one], ERROR_INVALID_ARGUMENT, "sequence", length=4)
    # One token more makes the short file a sequence.
    write_tokens(short, random_tokens(2, 5))
    assert loader_open_status(flint, [empty, short, one], length=4, vocab=VOCAB) == OK


def test_a_missing_file_and_a_folder_are_io_errors_naming_the_path(flint, tmp_path):
    missing = tmp_path / "nothing_here.bin"
    open_fails(flint, [missing], ERROR_IO, "nothing_here.bin")
    folder = tmp_path / "a_folder"
    folder.mkdir()
    open_fails(flint, [folder], ERROR_IO, "a_folder")
    # Second in the list too.
    good = tmp_path / "good.bin"
    write_tokens(good, random_tokens(1, 50))
    open_fails(flint, [good, missing], ERROR_IO, "nothing_here.bin")
    open_fails(flint, [good, folder], ERROR_IO, "a_folder")


@pytest.mark.skipif(sys.platform == "win32", reason="Windows won't truncate a file held open")
def test_a_file_that_shrinks_after_open_fails_the_reads_past_the_cut(flint, tmp_path):
    length = 4
    count = 20 * length + 1
    paths, _ = make_files(tmp_path, [count], seed=12)
    ref = Reference(paths, length, shuffle=False)  # built before the cut
    with Loader(flint, paths, length, VOCAB, shuffle=False) as loader:
        assert loader.epoch_sequences() == 20
        cut = count // 2  # tokens left: 0 .. 39, so sequence 8 is whole and sequence 9 isn't
        os.truncate(paths[0], 64 + 2 * cut)
        tokens = np.full((1, length), SENTINEL, dtype=np.uint16)
        targets = np.full((1, length), SENTINEL, dtype=np.uint16)
        for g in (9, 10, 19):  # across the cut, and past it
            assert loader.read_into(g, 1, tokens, targets) == ERROR_IO, g
            assert "flint_loader_read" in flint.last_error()
        for g in (0, 5, 8):
            assert_read_equals(loader, ref, g, 1)
        assert_read_equals(loader, ref, 0, 9)  # a read wholly before the cut
        # The loader is still usable after the failures.
        assert loader.read_into(19, 1, tokens, targets) == ERROR_IO
        assert_read_equals(loader, ref, 3, 2)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths are UTF-16; Linux's are bytes")
def test_a_path_that_is_not_utf8_is_invalid_on_windows(flint, tmp_path):
    good = tmp_path / "good.bin"
    write_tokens(good, random_tokens(1, 50))
    bad = os.fspath(tmp_path / "x").encode() + b"\xff\xfe.bin"
    got, handle, message = raw_open(flint, [bad])
    assert got == ERROR_INVALID_ARGUMENT, message
    assert handle is None and "flint_loader_open" in message
    got, handle, message = raw_open(flint, [good, bad])
    assert got == ERROR_INVALID_ARGUMENT, message
    assert handle is None


# --- 7. arguments ----------------------------------------------------------------------------


@pytest.fixture
def one_file(tmp_path):
    path = tmp_path / "one.bin"
    write_tokens(path, random_tokens(5, 4 * 25 + 1))
    return path


def test_open_rejects_bad_arguments(flint, one_file):
    p = one_file
    got, _, message = raw_open(flint, [p], null_loader=True)
    assert got == ERROR_INVALID_ARGUMENT and "flint_loader_open" in message
    open_fails(flint, [p], ERROR_INVALID_ARGUMENT, null_paths=True)
    open_fails(flint, [p], ERROR_INVALID_ARGUMENT, null_config=True)
    open_fails(flint, [p], ERROR_INVALID_ARGUMENT, files=0)
    # The count check comes before any file opens, so copies of one path will do.
    open_fails(flint, [p] * 65_537, ERROR_INVALID_ARGUMENT)
    for bad_length in (0, 65_537):
        open_fails(flint, [p], ERROR_INVALID_ARGUMENT, "length", length=bad_length)
    for bad_vocab in (0, 65_537):
        open_fails(flint, [p], ERROR_INVALID_ARGUMENT, "vocab", vocab=bad_vocab)


def test_open_names_the_index_of_a_null_or_empty_path(flint, one_file):
    message = open_fails(flint, [one_file, one_file, None], ERROR_INVALID_ARGUMENT)
    assert "2" in message, message
    message = open_fails(flint, [one_file, None, one_file], ERROR_INVALID_ARGUMENT)
    assert "1" in message, message
    message = open_fails(flint, [one_file, one_file, b""], ERROR_INVALID_ARGUMENT)
    assert "2" in message, message
    message = open_fails(flint, [b"", one_file], ERROR_INVALID_ARGUMENT)
    assert "0" in message, message


def test_the_largest_length_and_vocab_open(flint, tmp_path):
    path = tmp_path / "long.bin"
    write_tokens(path, random_tokens(6, 65_537, vocab=65_536))
    with Loader(flint, [path], 65_536, 65_536, shuffle=False) as loader:
        assert loader.epoch_sequences() == 1
        tokens, targets = loader.read(0, 1)
        tokens_all = fineweb.read_token_file(path)
        assert np.array_equal(tokens[0], tokens_all[:65_536])
        assert np.array_equal(targets[0], tokens_all[1:])
    assert loader_open_status(flint, [path], length=65_536, vocab=65_536) == OK
    assert loader_open_status(flint, [path], length=1, vocab=1) == OK


def test_length_and_vocab_of_65_537_are_refused_by_their_own_checks(flint, tmp_path):
    # The file holds enough tokens for one sequence at length 65,536, so a refusal at 65,537 can
    # only come from the length check itself.
    path = tmp_path / "enough.bin"
    write_tokens(path, random_tokens(7, 70_000, vocab=65_536))
    open_fails(flint, [path], ERROR_INVALID_ARGUMENT, "length", length=65_537, vocab=65_536)
    with Loader(flint, [path], 65_536, 65_536, shuffle=False) as loader:
        assert loader.epoch_sequences() == 1
    open_fails(flint, [path], ERROR_INVALID_ARGUMENT, "vocab", length=4, vocab=65_537)
    assert loader_open_status(flint, [path], length=4, vocab=65_536) == OK


def test_a_failed_open_raises_flint_error_with_the_status(flint, tmp_path):
    with pytest.raises(FlintError) as error:
        Loader(flint, [tmp_path / "absent.bin"], 4, VOCAB)
    assert error.value.status == ERROR_IO and "absent.bin" in str(error.value)


def test_close_of_null_does_nothing(flint):
    flint.lib.flint_loader_close(None)


def test_epoch_sequences_and_locate_reject_null(flint, one_file):
    with Loader(flint, [one_file], 4, VOCAB) as loader:
        size = ctypes.c_size_t()
        a, b, c = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t()
        lib = flint.lib
        assert lib.flint_loader_epoch_sequences(loader.handle, None) == ERROR_INVALID_ARGUMENT
        assert "flint_loader_epoch_sequences" in flint.last_error()
        assert lib.flint_loader_epoch_sequences(None, ctypes.byref(size)) == ERROR_INVALID_ARGUMENT
        refs = [ctypes.byref(a), ctypes.byref(b), ctypes.byref(c)]
        for i in range(3):
            args = list(refs)
            args[i] = None
            assert lib.flint_loader_locate(loader.handle, 0, *args) == ERROR_INVALID_ARGUMENT
            assert "flint_loader_locate" in flint.last_error()
        assert lib.flint_loader_locate(None, 0, *refs) == ERROR_INVALID_ARGUMENT
        # Still usable.
        assert loader.epoch_sequences() == 25


SENTINEL = 0xABCD


def sentinel_buffers(count, length, extra=0):
    tokens = np.full(count * length + extra, SENTINEL, dtype=np.uint16)
    targets = np.full(count * length + extra, SENTINEL, dtype=np.uint16)
    return tokens, targets


def test_read_rejects_bad_arguments_and_leaves_the_outputs_alone(flint, one_file):
    length = 4
    with Loader(flint, [one_file], length, VOCAB) as loader:
        tokens, targets = sentinel_buffers(3, length)
        addr_t, addr_g = tokens.ctypes.data, targets.ctypes.data
        lib = flint.lib

        def call(index=0, count=3, t=addr_t, g=addr_g):
            return lib.flint_loader_read(loader.handle, index, count, t, g)

        assert lib.flint_loader_read(None, 0, 3, addr_t, addr_g) == ERROR_INVALID_ARGUMENT
        assert call(t=None) == ERROR_INVALID_ARGUMENT
        assert call(g=None) == ERROR_INVALID_ARGUMENT
        assert "flint_loader_read" in flint.last_error()
        # An odd address: a view one byte into a buffer.
        big = np.full(4 * 3 * length, SENTINEL, dtype=np.uint16)
        assert call(t=addr_t + 1) == ERROR_INVALID_ARGUMENT
        assert call(g=addr_g + 1) == ERROR_INVALID_ARGUMENT
        # Tokens and targets the same buffer, and overlapping by one value.
        assert call(g=addr_t) == ERROR_INVALID_ARGUMENT
        wide = np.full(2 * 3 * length, SENTINEL, dtype=np.uint16)
        base = wide.ctypes.data
        n = 3 * length
        assert call(t=base, g=base + 2 * (n - 1)) == ERROR_INVALID_ARGUMENT
        assert call(t=base + 2 * (n - 1), g=base) == ERROR_INVALID_ARGUMENT
        assert (wide == SENTINEL).all()
        # index + count overflow.
        assert call(index=U64_MAX, count=1) == ERROR_INVALID_ARGUMENT
        assert call(index=U64_MAX - 1, count=2) == ERROR_INVALID_ARGUMENT
        # count * length * 2 overflows size_t (length 4: 2^62 x 4 = 2^64).
        assert call(count=2**62) == ERROR_INVALID_ARGUMENT
        assert call(count=2**63) == ERROR_INVALID_ARGUMENT
        assert call(count=U64_MAX) == ERROR_INVALID_ARGUMENT
        assert (tokens == SENTINEL).all() and (targets == SENTINEL).all()
        assert (big == SENTINEL).all()
        # Buffers that touch without overlapping are fine.
        assert lib.flint_loader_read(loader.handle, 0, 3, base, base + 2 * n) == OK
        assert not (wide == SENTINEL).all()
        # The loader still works.
        assert loader.read(0, 3)[0].shape == (3, length)


@pytest.mark.parametrize("length", [1, 3, 4, 65_536])
def test_read_rejects_the_first_count_whose_bytes_overflow(flint, tmp_path, length):
    # count x length x 2 must fit in 64 bits; this is the smallest count that doesn't, so the
    # factor 2 matters (at 2^61 x 4 the byte count wraps to 0).
    #
    # The boundary calls use a loader built to fail cleanly if the check is broken. If the check
    # lost its factor 2, the read would get past it and run for 2^63-ish sequences, overrunning the
    # buffers and crashing the process. So this loader has shuffle off and a file whose tokens are
    # all VOCAB - 1, opened with vocab=VOCAB - 1: a read that gets past the size check then fails
    # the range check on its first token, before anything is copied. The test then sees
    # FLINT_ERROR_FORMAT instead of FLINT_ERROR_INVALID_ARGUMENT, and the buffers still hold the
    # sentinel. A second, ordinary loader shows that reads work afterwards.
    path = tmp_path / "top.bin"
    write_tokens(path, np.full(2 * length + 1, VOCAB - 1, dtype=np.uint16))
    boundary = (U64_MAX // 2 // length) + 1
    assert boundary * length * 2 > U64_MAX
    with Loader(flint, [path], length, VOCAB - 1, shuffle=False) as loader:
        tokens, targets = sentinel_buffers(2, length)
        assert loader.read_into(0, boundary, tokens, targets) == ERROR_INVALID_ARGUMENT
        message = flint.last_error()
        assert "flint_loader_read" in message and "overflow" in message, message
        assert loader.read_into(5, boundary, tokens, targets) == ERROR_INVALID_ARGUMENT
        assert (tokens == SENTINEL).all() and (targets == SENTINEL).all()
    paths, _ = make_files(tmp_path, [2 * length + 1], seed=length)
    with Loader(flint, paths, length, VOCAB) as loader:
        assert loader.read(0, 2)[0].shape == (2, length)


def test_read_of_count_zero_writes_nothing(flint, one_file):
    length = 4
    with Loader(flint, [one_file], length, VOCAB) as loader:
        tokens, targets = sentinel_buffers(2, length)
        assert loader.read_into(0, 0, tokens, targets) == OK
        assert loader.read_into(10**6, 0, tokens, targets) == OK
        assert (tokens == SENTINEL).all() and (targets == SENTINEL).all()
        assert loader.read_into(0, 0, None, targets) == ERROR_INVALID_ARGUMENT
        assert loader.read_into(0, 0, tokens, None) == ERROR_INVALID_ARGUMENT
        tokens_none, targets_none = loader.read(0, 0)
        assert tokens_none.shape == (0, length) and targets_none.shape == (0, length)


def test_huge_indices_work(flint, tmp_path):
    paths, _ = make_files(tmp_path, [7 * 9 + 1, 7 * 4 + 3, 7 * 12 + 1], seed=2)
    ref = Reference(paths, 7, seed=11, shuffle=True)
    with Loader(flint, paths, 7, VOCAB, seed=11) as loader:
        assert loader.locate(U64_MAX) == ref.locate(U64_MAX)
        assert loader.locate(U64_MAX - 1) == ref.locate(U64_MAX - 1)
        assert loader.locate(2**63) == ref.locate(2**63)
        assert_read_equals(loader, ref, U64_MAX - 1, 1)
        assert_read_equals(loader, ref, 2**63 + 12345, 1)
        assert_read_equals(loader, ref, U64_MAX - 5, 5)


# --- 8. 64-bit offsets and the 2^32-sequence limit (sparse files) -----------------------------


def allocated_bytes(path):
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCompressedFileSizeW.argtypes = [
            ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32),
        ]  # fmt: skip
        kernel32.GetCompressedFileSizeW.restype = ctypes.c_uint32
        high = ctypes.c_uint32(0)
        low = kernel32.GetCompressedFileSizeW(str(path), ctypes.byref(high))
        return (high.value << 32) | low
    return os.stat(path).st_blocks * 512


def make_sparse(path, size, header, marker):
    """Copied from tests/test_safetensors.py (importing that module would load safetensors and
    torch): writes header at the start of path, extends the file to size bytes without storing
    the gap, and puts marker at the end. False if the file can't be made sparse (the file is
    removed then)."""
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
        # Writing at the far end extends the sparse file without storing the gap.
        f.seek(size - len(marker))
        f.write(marker)
    ok = allocated_bytes(path) < 256 * MiB
    if not ok:
        os.remove(path)
    return ok


def write_tokens_at(path, first_token, values):
    """Writes uint16 values at token index first_token of an existing token file."""
    with open(path, "r+b") as f:
        f.seek(64 + 2 * first_token)
        f.write(np.asarray(values, dtype="<u2").tobytes())


def sparse_token_file(path, count, end_value):
    """A sparse token file of count tokens, zero but for end_value as the last token. Skips the
    test when the file system can't."""
    marker = struct.pack("<H", end_value)
    if not make_sparse(path, 64 + 2 * count, header_bytes(count), marker):
        pytest.skip("this file system can't make the sparse file")


def test_a_file_past_4_gib_is_read_at_64_bit_offsets(flint, tmp_path):
    count = 2**31 + 1000
    path = tmp_path / "past4gib.bin"
    last8 = (count - 1) // 8 - 1  # the last sequence at length 8
    boundary_marker = np.arange(41, dtype=np.uint16) + 1000  # tokens 2^31 - 40 .. 2^31
    last_marker = np.arange(9, dtype=np.uint16) + 300  # the last sequence's 9 tokens
    try:
        sparse_token_file(path, count, 7)
        write_tokens_at(path, 2**31 - 40, boundary_marker)
        write_tokens_at(path, (last8 + 1) * 8 - 8, last_marker)
        # length 8, shuffle off: sequence s is tokens 8s .. 8s + 8.
        ref = Reference([path], 8, shuffle=False)
        size = ref.epoch_sequences()
        assert size == last8 + 1 == 268_435_580
        with Loader(flint, [path], 8, VOCAB, shuffle=False) as loader:
            assert loader.epoch_sequences() == size
            assert loader.locate(size - 1) == (0, 0, size - 1)
            tokens, targets = loader.read(size - 1, 1)
            assert np.array_equal(tokens[0], last_marker[:8])
            assert np.array_equal(targets[0], last_marker[1:])
            assert_read_equals(loader, ref, size - 1, 1)
            # Around token 2^31, where the byte offset passes 2^32.
            first = 2**28 - 6
            assert_read_equals(loader, ref, first, 8)
            tokens, targets = loader.read(2**28 - 5, 1)
            assert np.array_equal(tokens[0], boundary_marker[:8])
            assert np.array_equal(targets[0], boundary_marker[1:9])
            tokens, _ = loader.read(2**28, 1)
            assert tokens[0, 0] == boundary_marker[40]
            # The next epoch's first sequence is zero tokens.
            assert loader.locate(size) == (1, 0, 0)
            assert not loader.read(size, 1)[0].any()
        del ref
        gc.collect()
        # length 65,536, shuffle on: the last sequence, wherever the order puts it.
        ref = Reference([path], 65_536, seed=7, shuffle=True)
        size = ref.epoch_sequences()
        assert size == 32_768
        position = ref.sequence_order(0, 0).index(size - 1)
        with Loader(flint, [path], 65_536, VOCAB, seed=7, shuffle=True) as loader:
            assert loader.epoch_sequences() == size
            assert loader.locate(position) == (0, 0, size - 1) == ref.locate(position)
            tokens, targets = loader.read(position, 1)
            assert np.array_equal(tokens[0, -40:], boundary_marker[:40])
            assert np.array_equal(targets[0, -40:], boundary_marker[1:])
            assert_read_equals(loader, ref, position, 1)
            for g in (0, 1, 5, size - 1, size + 3):
                assert loader.locate(g) == ref.locate(g)
                assert_read_equals(loader, ref, g, 1)
        del ref, tokens, targets
        gc.collect()
    finally:
        gc.collect()
        if path.exists():
            os.remove(path)


def test_a_file_with_2_32_sequences_is_refused_and_max_tokens_brings_it_under(flint, tmp_path):
    count = 2**32 + 1
    path = tmp_path / "limit.bin"
    try:
        sparse_token_file(path, count, 13)
        write_tokens_at(path, 2**32 - 4, [10, 11, 12, 14])  # tokens 2^32 - 4 .. 2^32 - 1
        # length 1: 2^32 sequences.
        open_fails(flint, [path], ERROR_FORMAT, "limit.bin", length=1, shuffle=False)
        open_fails(flint, [path], ERROR_FORMAT, "limit.bin", length=1, shuffle=True)
        # max_tokens 2^32 leaves 2^32 - 1.
        with Loader(flint, [path], 1, VOCAB, shuffle=False, max_tokens=2**32) as loader:
            assert loader.epoch_sequences() == 2**32 - 1
            tokens, targets = loader.read(2**32 - 2, 1)
            assert tokens[0, 0] == 12 and targets[0, 0] == 14  # not the 13 past max_tokens
            assert loader.locate(2**32 - 2) == (0, 0, 2**32 - 2)
            assert loader.locate(2**32 - 1) == (1, 0, 0)
        # length 2: 2^31 sequences.
        with Loader(flint, [path], 2, VOCAB, shuffle=False) as loader:
            assert loader.epoch_sequences() == 2**31
            tokens, targets = loader.read(2**31 - 1, 1)
            assert tokens[0].tolist() == [12, 14] and targets[0].tolist() == [14, 13]
    finally:
        gc.collect()
        if path.exists():
            os.remove(path)


def test_the_shuffle_cap_is_2_26_sequences_and_only_with_shuffle_on(flint, tmp_path):
    cap = 2**26  # 67,108,864 sequences: a 256 MiB permutation
    at_cap = tmp_path / "at_cap.bin"
    over_cap = tmp_path / "over_cap.bin"
    try:
        # length 1: a file of 2^26 + 1 tokens holds 2^26 sequences.
        sparse_token_file(at_cap, cap + 1, 7)
        with Loader(flint, [at_cap], 1, VOCAB, seed=5, shuffle=True) as loader:
            assert loader.epoch_sequences() == cap
            # One locate builds the 256 MiB permutation (no comparison with the reference, which
            # would take minutes in Python).
            where = loader.locate(cap - 1)
            assert where[:2] == (0, 0) and 0 <= where[2] < cap, where
            assert loader.locate(cap - 1) == where
        # One sequence more, and shuffle on refuses it, naming the file and the cap.
        sparse_token_file(over_cap, cap + 2, 7)
        message = open_fails(
            flint, [over_cap], ERROR_INVALID_ARGUMENT, "over_cap.bin", "67,108,864",
            length=1, shuffle=True,
        )  # fmt: skip
        assert "permutation" in message, message
        # max_tokens brings the file under the cap: the cap counts the sequences left after it
        # (2^26 at length 1), not the header's count.
        assert loader_open_status(
            flint, [over_cap], length=1, vocab=VOCAB, shuffle=True, max_tokens=cap + 1
        ) == OK  # fmt: skip
        # Shuffle off has no cap: it builds no permutation.
        with Loader(flint, [over_cap], 1, VOCAB, shuffle=False) as loader:
            assert loader.epoch_sequences() == cap + 1
            assert loader.locate(cap) == (0, 0, cap)
    finally:
        gc.collect()
        for path in (at_cap, over_cap):
            if path.exists():
                os.remove(path)


# --- 9. real data ----------------------------------------------------------------------------


def shard(name):
    path = data_file(f"fineweb-edu-10B/{name}")
    if path is None:
        pytest.skip(f"fineweb-edu-10B/{name} isn't here")
    return path


def read_all(loader, size, chunk=4096):
    for g in range(0, size, chunk):
        yield g, loader.read(g, min(chunk, size - g))


def test_a_real_shard_reads_and_matches_the_reference(flint):
    path = shard("shard_001.bin")
    memmap = fineweb.read_token_file(path)
    length = 1024
    with Loader(flint, [path], length, VOCAB, shuffle=False) as loader:
        size = loader.epoch_sequences()
        assert size == 97_656 == (len(memmap) - 1) // length
        for g, (tokens, targets) in read_all(loader, size):
            n = len(tokens)
            lo, hi = g * length, (g + n) * length
            assert np.array_equal(tokens.reshape(-1), memmap[lo:hi])
            assert np.array_equal(targets.reshape(-1), memmap[lo + 1 : hi + 1])
    ref = Reference([path], length, seed=20261124, shuffle=True)
    rng = random.Random(4)
    indices = [rng.randrange(2 * 97_656) for _ in range(300)]
    with Loader(flint, [path], length, VOCAB, seed=20261124, shuffle=True) as loader:
        for g in indices:
            assert loader.locate(g) == ref.locate(g)
            assert_read_equals(loader, ref, g, 1)


def test_the_validation_and_training_shards_on_the_server(flint):
    val_a = data_file("fineweb-edu-10B/shard_000.bin")
    val_b = data_file("fineweb-edu-10B/val_b.bin")
    if val_a is None or val_b is None:
        pytest.skip("shard_000.bin and val_b.bin aren't here")
    length = 1024
    with Loader(flint, [val_a], length, VOCAB, shuffle=False, max_tokens=10_506_240) as loader:
        assert loader.epoch_sequences() == 10_259
        memmap = fineweb.read_token_file(val_a)
        for g, (tokens, targets) in read_all(loader, 10_259):
            lo, hi = g * length, (g + len(tokens)) * length
            assert np.array_equal(tokens.reshape(-1), memmap[lo:hi])
    with Loader(flint, [val_b], length, VOCAB, shuffle=False) as loader:
        assert loader.epoch_sequences() == 9_347
        for _ in read_all(loader, 9_347):
            pass


def test_all_the_training_shards_open_together_on_the_server(flint):
    paths = [data_file(f"fineweb-edu-10B/shard_{i:03d}.bin") for i in range(1, 100)]
    if any(p is None for p in paths):
        pytest.skip("the 99 training shards aren't here")
    ref = Reference(paths, 1024, seed=20261124, shuffle=True)
    with Loader(flint, paths, 1024, VOCAB, seed=20261124, shuffle=True) as loader:
        assert loader.epoch_sequences() == 9_613_662 == ref.epoch_sequences()
        rng = random.Random(6)
        for g in [rng.randrange(2 * 9_613_662) for _ in range(50)]:
            assert loader.locate(g) == ref.locate(g)
            assert_read_equals(loader, ref, g, 1)


# --- 10. the page cache (Linux) --------------------------------------------------------------


def resident_pages(path):
    """(pages of the file in the page cache, pages) through mincore on a mapping."""
    import mmap

    page = os.sysconf("SC_PAGE_SIZE")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        pages = (size + page - 1) // page
        vector = (ctypes.c_ubyte * pages)()
        view = np.frombuffer(mapping, dtype=np.uint8)  # the address, without touching the pages
        address = view.__array_interface__["data"][0]
        status = libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vector)
        assert status == 0, os.strerror(ctypes.get_errno())
        resident = sum(v & 1 for v in vector)
        del view
        mapping.close()
        return resident, pages
    finally:
        os.close(fd)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="posix_fadvise is Linux's")
def test_the_loader_holds_about_one_file_of_page_cache(flint, tmp_path):
    length = 1024
    counts = [2 * MiB // 2 + 1, 2 * MiB // 2 + 1]  # two files of 2 MiB of tokens each
    paths = []
    for i, count in enumerate(counts):
        path = tmp_path / f"cache{i}.bin"
        tokens = random_tokens(40 + i, count)
        path.write_bytes(header_bytes(count) + tokens.tobytes())
        fd = os.open(path, os.O_RDONLY)
        os.fsync(fd)  # clean pages, so the kernel is free to drop them
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        paths.append(path)
    a_pages = resident_pages(paths[0])[1]
    if resident_pages(paths[0])[0] or resident_pages(paths[1])[0]:
        pytest.skip("this file system can't drop the pages (tmpfs, such as a RAM-disk /tmp)")
    sequences_a = (counts[0] - 1) // length
    with Loader(flint, paths, length, VOCAB, shuffle=False) as loader:
        # The header read leaves at most its own page cached: with the RANDOM advice given first
        # there is no readahead (without it the kernel cached 4 pages of each file).
        assert resident_pages(paths[0])[0] <= 1 and resident_pages(paths[1])[0] <= 1
        for g in range(0, sequences_a, 64):
            loader.read(g, min(64, sequences_a - g))
        assert resident_pages(paths[0])[0] > a_pages * 0.9
        assert resident_pages(paths[1])[0] <= 1  # B may hold its header's page and no more
        # The next read moves into the second file: the first one's pages go.
        loader.read(sequences_a, 1)
        assert resident_pages(paths[0])[0] == 0
        assert resident_pages(paths[1])[0] > 0
        for g in range(sequences_a, loader.epoch_sequences(), 64):
            loader.read(g, min(64, loader.epoch_sequences() - g))
        assert resident_pages(paths[1])[0] > a_pages * 0.9
    # Closing drops the last file read.
    assert resident_pages(paths[1])[0] == 0
    assert resident_pages(paths[0])[0] == 0
