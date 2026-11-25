"""A pure-Python reference for the data loader's order (D34), written from the design's "The
order" section alone and never from the C++.

Everything is Python integers (exact, wrapping mod 2^64 by hand) and nothing comes from numpy
except the token arrays a read returns. Reference reads the token files the way the format's
ground truth does (scripts/fineweb.py's read_token_file) and answers the same questions as the
loader: epoch_sequences, locate and read.

The arithmetic, in short:

- SplitMix64: next(state) adds 0x9E3779B97F4A7C15 to the state and returns mix(state); hash(x)
  is mix(x + 0x9E3779B97F4A7C15), the output next gives from state x.
- A stream for (seed, epoch, key) starts at hash(hash(hash(seed) ^ epoch) ^ key); the files'
  order uses key 2^64 - 1 and file f's sequences use key f.
- draw(r, n) is the high 64 bits of r * n; Fisher-Yates runs i from n - 1 down to 1, takes the
  next output, swaps a[i] with a[draw(r, i + 1)].
- File f has c_f = (usable - 1) // length sequences; an epoch is every sequence of every file;
  within epoch e the files come in the order P and each file's sequences in the order Q_f (the
  identity both, with shuffle off).
"""

import importlib.util
from pathlib import Path

import numpy as np

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
MUL1 = 0xBF58476D1CE4E5B9
MUL2 = 0x94D049BB133111EB
FILES_KEY = MASK
HEADER_BYTES = 64

# Files up to this many tokens are copied out of their memmap, so the reference holds no file open.
COPY_LIMIT = 1 << 24

_fineweb = None


def fineweb():
    """scripts/fineweb.py, loaded once by path."""
    global _fineweb
    if _fineweb is None:
        path = Path(__file__).resolve().parent.parent / "scripts" / "fineweb.py"
        spec = importlib.util.spec_from_file_location("fineweb_for_loader_reference", path)
        _fineweb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_fineweb)
    return _fineweb


def mix(x):
    """SplitMix64's finalizer."""
    x &= MASK
    x ^= x >> 30
    x = (x * MUL1) & MASK
    x ^= x >> 27
    x = (x * MUL2) & MASK
    x ^= x >> 31
    return x


def next(state):  # noqa: A001 (the design's name)
    """SplitMix64's step: (the new state, the output)."""
    state = (state + GAMMA) & MASK
    return state, mix(state)


def hash(x):  # noqa: A001 (the design's name)
    """mix(x + GAMMA), the output next gives from state x."""
    return mix((x + GAMMA) & MASK)


def start(seed, epoch, key):
    """The starting state of the stream for (seed, epoch, key)."""
    return hash(hash(hash(seed) ^ epoch) ^ key)


def draw(r, n):
    """A number in [0, n) from the 64-bit output r: the high 64 bits of r * n."""
    return (r * n) >> 64


def fisher_yates(n, state):
    """A permutation of [0, n) from the starting state."""
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        state, r = next(state)
        j = draw(r, i + 1)
        a[i], a[j] = a[j], a[i]
    return a


def sequences(usable, length):
    """The sequences of a file with usable tokens."""
    return 0 if usable == 0 else (usable - 1) // length


class Reference:
    """The loader's answers, computed from the files directly.

    paths are token files, read as memmaps (those over COPY_LIMIT tokens stay memmaps, so a
    sparse file too big to materialize is read only where a test asks, and with shuffle off no
    permutation is built)."""

    def __init__(self, paths, length, seed=0, shuffle=True, max_tokens=0):
        fw = fineweb()
        self.paths = [Path(p) for p in paths]
        self.length = length
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.max_tokens = max_tokens
        self.tokens = []
        self.usable = []
        for path in self.paths:
            tokens = fw.read_token_file(path)
            count = len(tokens)
            usable = min(count, max_tokens) if max_tokens else count
            if count <= COPY_LIMIT:
                tokens = np.array(tokens, dtype=np.uint16)
            self.tokens.append(tokens)
            self.usable.append(usable)
        self.counts = [sequences(u, length) for u in self.usable]
        self.epoch_size = sum(self.counts)
        self._file_orders = {}
        self._sequence_orders = {}

    def close(self):
        self.tokens = []

    def epoch_sequences(self):
        return self.epoch_size

    def file_order(self, epoch):
        """P: the files in the order epoch takes them."""
        if not self.shuffle:
            return range(len(self.paths))
        order = self._file_orders.get(epoch)
        if order is None:
            order = fisher_yates(len(self.paths), start(self.seed, epoch, FILES_KEY))
            self._file_orders[epoch] = order
        return order

    def sequence_order(self, epoch, f):
        """Q_f: file f's sequences in the order epoch takes them."""
        if not self.shuffle:
            return range(self.counts[f])
        key = (epoch, f)
        order = self._sequence_orders.get(key)
        if order is None:
            order = fisher_yates(self.counts[f], start(self.seed, epoch, f))
            self._sequence_orders[key] = order
        return order

    def locate(self, g):
        """(epoch, file, sequence) of index g."""
        epoch, r = divmod(g, self.epoch_size)
        for f in self.file_order(epoch):
            c = self.counts[f]
            if r < c:
                return epoch, f, self.sequence_order(epoch, f)[r]
            r -= c
        raise AssertionError("an index inside the epoch found no sequence")

    def sequence_tokens(self, f, s):
        """The length + 1 tokens of sequence s of file f."""
        begin = s * self.length
        return np.asarray(self.tokens[f][begin : begin + self.length + 1])

    def read(self, g, count):
        """(tokens, targets), uint16 arrays of shape (count, length)."""
        tokens = np.empty((count, self.length), dtype=np.uint16)
        targets = np.empty((count, self.length), dtype=np.uint16)
        for i in range(count):
            _, f, s = self.locate(g + i)
            seq = self.sequence_tokens(f, s)
            assert len(seq) == self.length + 1
            tokens[i] = seq[:-1]
            targets[i] = seq[1:]
        return tokens, targets

    def epoch_order(self, epoch):
        """Every (file, sequence) of the epoch in order, for tests of the whole permutation."""
        out = []
        for f in self.file_order(epoch):
            out.extend((f, s) for s in self.sequence_order(epoch, f))
        return out
