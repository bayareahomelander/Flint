"""Builds Flint's training data from FineWeb-Edu's 10B-token sample: download, tokenize, shard, verify.

Usage: python scripts/fineweb.py download --raw DIR [--files SPEC]
       python scripts/fineweb.py tokenize --raw DIR --work DIR [--files SPEC] [--threads 4]
       python scripts/fineweb.py shard --work DIR --out DIR [--files SPEC]
              [--shard-tokens 100000000] [--val-b-rate 1000]
       python scripts/fineweb.py verify --out DIR
       python scripts/fineweb.py check --raw DIR --work DIR --file NAME --docs N

SPEC picks files by index in sorted order: 0, 0-3, 0,5,13 (default: all 14).

The data is HuggingFaceFW/fineweb-edu at a pinned revision, folder sample/10BT:
14 parquet files, 28.5 GB, whose sizes and sha256 sums are constants below.
`download` fetches and verifies them. `tokenize` turns each file into raw
uint16 tokens (tiktoken's GPT-2 encoding) plus a per-document index in the work
directory, one file at a time, so a killed run resumes at the first file
without a done-record. `shard` cuts the token streams into Flint's token files.
`verify` rechecks the outputs against the manifest; `check` compares the first
documents of one file with a plain single-threaded reference loop.

Tokens. Each document becomes [50256] + enc.encode_ordinary(text), in file
order (sorted names) and row order: llm.c's fineweb.py produces the same stream.
The parquet files are read one row group at a time and the tokens are written
as they come, so memory stays far below the data size. The dataset's own
token_count column was computed by some other implementation; the documents
where it differs from ours are counted and listed (first few ids per file) in
the done-records and the manifest. That is a cross-check, not an error.

Token file format (what M2's C++ loader reads). Everything is little-endian. A
64-byte header, then the tokens as uint16, with no padding:
    bytes  0-7   magic b"FLINTTOK"
    bytes  8-11  uint32 version = 1
    bytes 12-15  uint32 tokenizer id = 1 (tiktoken "gpt2": 50,257 tokens,
                 <|endoftext|> = 50256)
    bytes 16-23  uint64 token count
    bytes 24-63  zero
The file size is exactly 64 + 2 x count.

Outputs of `shard`, in --out:
    shard_000.bin   held out: the first --shard-tokens tokens of the full,
                    unfiltered stream, so its first 10,506,240 tokens (val-A)
                    equal the set another project evaluates on. The document
                    that straddles its end loses its remaining tokens (they
                    are dropped, counted in the manifest, and not trained
                    on); training starts at the next document.
    val_b.bin       among the documents that start after shard 0's end, those
                    whose id hashes into 1 in --val-b-rate, whole and in order.
    shard_001.bin.. every other document, in order, cut into shards of exactly
                    --shard-tokens tokens (a document may span two shards, as
                    in llm.c); the last shard is partial.
    manifest.json   dataset revision, file checksums, rules, counts, git state.

Shard step rule for val-B: a document is in val-B iff the first 8 bytes of
sha256(id as UTF-8), read as a little-endian integer, are divisible by the rate.
It is decided here, after shard 0's end is known, so the tokenize step only
stores each document's length and hash.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT = Path(__file__).resolve().parent.parent

REPO_ID = "HuggingFaceFW/fineweb-edu"
REPO_TYPE = "dataset"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SUBFOLDER = "sample/10BT"
# file name -> (size in bytes, sha256), as the hub lists them at REVISION.
FILES = {
    "000_00000.parquet": (2152819114, "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871"),
    "001_00000.parquet": (2152222432, "3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693"),
    "002_00000.parquet": (2151796315, "547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7"),
    "003_00000.parquet": (2152437524, "22184e6eb25759ddd97783751ffc73e1705dfa2542e630dae1f2a8bac8ee6ddb"),
    "004_00000.parquet": (2152338550, "33557ddd87a07a4ae6fcaf7a4789c7b484e5cc0c273ca12a65b74200e6d8748b"),
    "005_00000.parquet": (2152189947, "e08d79927ecb377786572cd854817c748ceb8880878b1a0eb91abf8c85d505ea"),
    "006_00000.parquet": (2152689867, "554fa2613c9261d6c9c396caab881de3160175794c6c3e1856bf28a0e9cb9b76"),
    "007_00000.parquet": (2150686637, "4b7d1f697d4afff7f5f65cb7aa4d83acc6db716da4d46ea23399d0c50608c6e3"),
    "008_00000.parquet": (2151274846, "a6d9dcc0c72ecd7c7f0173c2d06b3a4249a2ccaa94961469d1a522ca689bbf1a"),
    "009_00000.parquet": (2151913277, "f836cd3c70b95776699eb6c356b2dbf702816e25dcf39992f5c80a29029d23c3"),
    "010_00000.parquet": (2152798864, "e5a2eae25f057f0856a10bfae314c6ca8ea8bb08456d2131e9e89b2b8305e2f6"),
    "011_00000.parquet": (2152323681, "db71cf0425bb3d1813a09f12ff1acd6dabdfb91a2e4f141960254ee5a7f036e7"),
    "012_00000.parquet": (2152069689, "08b47a3e1c25161f796d2f8dbf99ccf60affdebdcea4910833d0d5783315551f"),
    "013_00000.parquet": (540632672, "b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de"),
}

MAGIC = b"FLINTTOK"
FORMAT_VERSION = 1
TOKENIZER_ID = 1
HEADER_BYTES = 64
HEADER = struct.Struct("<8sIIQ")  # magic, version, tokenizer id, token count; the rest is zero
ENCODING = "gpt2"
N_VOCAB = 50257
EOT = 50256

DEFAULT_SHARD_TOKENS = 100_000_000
DEFAULT_VAL_B_RATE = 1000
VAL_A_TOKENS = 10_506_240
THREADS = 4
FIRST_IDS = 5  # mismatching document ids listed per file
IO_CHUNK = 16 * 1024 * 1024
COLUMNS = ["id", "text", "token_count"]
DONE_SCHEMA = "flint-fineweb-tokenized/1"
MANIFEST_SCHEMA = "flint-fineweb/1"
MANIFEST = "manifest.json"
VAL_B_RULE = (
    "among the documents that start after the end of shard_000.bin, a document is in val-B iff the "
    "first 8 bytes of sha256(id as UTF-8), read as a little-endian integer, are divisible by val_b_rate; "
    "val-B documents are written whole and in order, and excluded from the training shards"
)


class FineWebError(Exception):
    """A failure with a message fit to print: it never names a local path."""


# ------------------------------------------------------------ small helpers

def sha256_file(path):
    digest = hashlib.sha256()
    buffer = bytearray(IO_CHUNK)
    view = memoryview(buffer)
    with open(path, "rb") as f:
        while True:
            n = f.readinto(buffer)
            if not n:
                return digest.hexdigest()
            digest.update(view[:n])


def atomic_write(path, write):
    """Calls write(f) on `<name>.tmp`, flushes it to disk and renames it over
    `path`, so a killed run leaves either the old file or the new one whole.
    Returns write's result."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "wb") as f:
            result = write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return result


def write_json(path, document):
    text = json.dumps(document, indent=2) + "\n"
    atomic_write(path, lambda f: f.write(text.encode("utf-8")))


def read_exactly(source, view):
    """Fills the writable byte view from the stream, or raises at a short file."""
    filled = 0
    while filled < len(view):
        n = source.readinto(view[filled:])
        if not n:
            raise FineWebError("a token file ended early")
        filled += n


def stem_of(name):
    return name.removesuffix(".parquet")


def expected_names():
    return sorted(FILES)


def parse_file_spec(spec, names):
    """Names picked by a spec of indices into the sorted list: 0, 0-3, 0,5,13."""
    if spec is None:
        return list(names)
    chosen = set()
    for part in spec.split(","):
        part = part.strip()
        try:
            low, dash, high = part.partition("-")
            indices = range(int(low), int(high) + 1) if dash else [int(low)]
        except ValueError:
            raise FineWebError(f"bad --files item {part!r}; use indices like 0, 0-3 or 0,5,13") from None
        if not indices:
            raise FineWebError(f"empty range {part!r} in --files")
        for index in indices:
            if not 0 <= index < len(names):
                raise FineWebError(f"--files index {index} is outside 0-{len(names) - 1}")
            chosen.add(index)
    return [names[i] for i in sorted(chosen)]


def raw_file_path(raw, name):
    """Where a downloaded parquet file lives: hf_hub_download with local_dir
    keeps the repo's relative path (sample/10BT/<name>); a file placed directly
    in the directory is found too."""
    nested = Path(raw) / SUBFOLDER / name
    flat = Path(raw) / name
    return flat if flat.exists() and not nested.exists() else nested


def peak_rss_mib():
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)


def report_resources(started):
    text = f"elapsed {time.perf_counter() - started:.1f} s"
    rss = peak_rss_mib()
    if rss is not None:
        text += f", peak RSS {rss:.0f} MiB"
    print(text, flush=True)


def git_info():
    """The repo's HEAD and whether its tree is dirty, as cmake/git_info.cmake
    finds them: "unknown" and clean outside a git checkout."""
    sha, dirty = "unknown", False
    if (ROOT / ".git").exists():
        try:
            head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
            if head and all(c in "0123456789abcdef" for c in head):
                sha = head
                status = subprocess.run(["git", "-C", str(ROOT), "--no-optional-locks", "status", "--porcelain"],
                                        capture_output=True, text=True, check=True).stdout
                dirty = status != ""
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"sha": sha, "dirty": dirty}


# ------------------------------------------------------------- token files

def write_token_file(path, tokens):
    """Writes `tokens` (uint16 values) as a Flint token file, atomically, and
    returns the sha256 of the whole file."""
    tokens = np.asarray(tokens)
    if tokens.dtype != np.uint16:
        if tokens.size and (tokens.min() < 0 or tokens.max() > 0xFFFF):
            raise ValueError("a token doesn't fit in uint16")
        tokens = tokens.astype(np.uint16)
    tokens = np.ascontiguousarray(tokens, dtype="<u2")
    header = HEADER.pack(MAGIC, FORMAT_VERSION, TOKENIZER_ID, tokens.size).ljust(HEADER_BYTES, b"\0")

    def write(f):
        digest = hashlib.sha256(header)
        f.write(header)
        data = memoryview(tokens).cast("B")
        for begin in range(0, len(data), IO_CHUNK):
            chunk = data[begin:begin + IO_CHUNK]
            digest.update(chunk)
            f.write(chunk)
        return digest.hexdigest()

    return atomic_write(path, write)


def parse_header(header, file_size):
    """The token count in a 64-byte header, or a FineWebError if the header or
    the file's size is wrong."""
    if len(header) < HEADER_BYTES:
        raise FineWebError("token file is shorter than its header")
    magic, version, tokenizer, count = HEADER.unpack(header[:HEADER.size])
    if magic != MAGIC:
        raise FineWebError("token file has the wrong magic")
    if version != FORMAT_VERSION:
        raise FineWebError(f"token file version {version} isn't {FORMAT_VERSION}")
    if tokenizer != TOKENIZER_ID:
        raise FineWebError(f"token file tokenizer id {tokenizer} isn't {TOKENIZER_ID}")
    if any(header[HEADER.size:HEADER_BYTES]):
        raise FineWebError("token file header's reserved bytes aren't zero")
    if file_size != HEADER_BYTES + 2 * count:
        raise FineWebError(f"token file is {file_size} bytes, but {count} tokens need {HEADER_BYTES + 2 * count}")
    return count


def read_token_count(path):
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
    return parse_header(header, os.path.getsize(path))


def read_token_file(path):
    """The tokens of a Flint token file as a read-only uint16 memmap, after
    checking magic, version, tokenizer id and size."""
    count = read_token_count(path)
    if count == 0:
        empty = np.empty(0, dtype="<u2")
        empty.flags.writeable = False
        return empty
    return np.memmap(path, dtype="<u2", mode="r", offset=HEADER_BYTES, shape=(count,))


# ---------------------------------------------------------------- tokenizer

def configure_tiktoken_cache():
    """tiktoken caches its BPE files under the temp dir unless told otherwise,
    and the server's temp dir (a RAM disk) must stay clean."""
    if "TIKTOKEN_CACHE_DIR" not in os.environ:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(Path.home() / ".cache" / "tiktoken")


def get_encoder():
    configure_tiktoken_cache()
    enc = tiktoken.get_encoding(ENCODING)
    assert enc.n_vocab == N_VOCAB and enc.eot_token == EOT, "unexpected GPT-2 encoding"
    return enc


def tokenize_texts(enc, texts, threads):
    """[EOT] + encode_ordinary(text) for each text, in order, as one flat uint16
    array and the uint32 length of each document (EOT included)."""
    encoded = enc.encode_ordinary_batch(texts, num_threads=threads)
    lengths = np.fromiter((len(e) + 1 for e in encoded), dtype=np.uint32, count=len(encoded))
    flat = np.empty(int(lengths.sum()), dtype="<u2")
    position = 0
    for tokens, length in zip(encoded, lengths.tolist()):
        flat[position] = EOT
        flat[position + 1:position + length] = tokens
        position += length
    return flat, lengths


def doc_hash(doc_id):
    return int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:8], "little")


# ----------------------------------------------------------------- download

def hf_download(**kwargs):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def file_is_right(path, size, sha256):
    return path.is_file() and path.stat().st_size == size and sha256_file(path) == sha256


def download_files(raw, names, downloader=hf_download):
    """Fetches each named file unless a verified copy is already there, and
    checks the sha256 of what arrives. `downloader` has hf_hub_download's
    keyword interface and returns the file's path."""
    for name in names:
        size, sha256 = FILES[name]
        path = raw_file_path(raw, name)
        if file_is_right(path, size, sha256):
            print(f"{name}: already downloaded and verified", flush=True)
            continue
        if path.exists():
            # hf_hub_download trusts a file it finds, so a bad one must go first.
            print(f"{name}: present but its size or sha256 is wrong; removing it", flush=True)
            path.unlink()
        print(f"{name}: downloading {size / 1e9:.2f} GB", flush=True)
        began = time.perf_counter()
        fetched = Path(downloader(repo_id=REPO_ID, filename=f"{SUBFOLDER}/{name}", repo_type=REPO_TYPE,
                                  revision=REVISION, local_dir=str(raw)))
        if not file_is_right(fetched, size, sha256):
            fetched.unlink(missing_ok=True)
            raise FineWebError(f"{name}: the download's size or sha256 doesn't match the pinned value; deleted it")
        print(f"{name}: verified in {time.perf_counter() - began:.0f} s", flush=True)


# ----------------------------------------------------------------- tokenize

def tokens_path(work, name):
    return Path(work) / f"tokens_{stem_of(name)}.bin"


def docs_path(work, name):
    return Path(work) / f"docs_{stem_of(name)}.npz"


def done_path(work, name):
    return Path(work) / f"done_{stem_of(name)}.json"


def load_done(work, name):
    """The done-record of a fully tokenized file, or None if the file still needs
    tokenizing: the record is missing or unreadable, or the outputs it describes
    don't match it."""
    try:
        record = json.loads(done_path(work, name).read_text(encoding="utf-8"))
        ok = (
            record["schema"] == DONE_SCHEMA
            and record["file"] == name
            and record["source_size"] == FILES[name][0]
            and tokens_path(work, name).stat().st_size == 2 * record["tokens"]
            and docs_path(work, name).is_file()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return record if ok else None


def tokenize_file(raw_path, work, name, enc, threads):
    """Tokenizes one parquet file into `tokens_<stem>.bin` (raw uint16),
    `docs_<stem>.npz` (each document's token length and id hash) and, last,
    `done_<stem>.json`. Returns the done-record."""
    began = time.perf_counter()
    # An old record must not outlive the outputs it describes.
    done_path(work, name).unlink(missing_ok=True)
    lengths_parts, hash_parts = [], []
    documents = tokens = mismatches = 0
    first_ids = []

    def write_tokens(f):
        nonlocal documents, tokens, mismatches
        digest = hashlib.sha256()
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=COLUMNS)
            ids = table.column("id").to_pylist()
            texts = table.column("text").to_pylist()
            counts = table.column("token_count").to_pylist()
            if any(text is None for text in texts) or any(i is None for i in ids):
                raise FineWebError(f"{name}: a row has no id or no text")
            flat, lengths = tokenize_texts(enc, texts, threads)
            data = memoryview(flat).cast("B")
            digest.update(data)
            f.write(data)
            lengths_parts.append(lengths)
            hash_parts.append(np.fromiter((doc_hash(i) for i in ids), dtype=np.uint64, count=len(ids)))
            for doc_id, length, count in zip(ids, lengths.tolist(), counts):
                # The dataset's token_count excludes our EOT.
                if length - 1 != count:
                    mismatches += 1
                    if len(first_ids) < FIRST_IDS:
                        first_ids.append(doc_id)
            documents += len(ids)
            tokens += flat.size
        return digest.hexdigest()

    with pq.ParquetFile(raw_path) as parquet:
        sha256 = atomic_write(tokens_path(work, name), write_tokens)
    lengths = np.concatenate(lengths_parts) if lengths_parts else np.empty(0, dtype=np.uint32)
    hashes = np.concatenate(hash_parts) if hash_parts else np.empty(0, dtype=np.uint64)
    atomic_write(docs_path(work, name), lambda f: np.savez(f, lengths=lengths, hashes=hashes))
    record = {
        "schema": DONE_SCHEMA,
        "file": name,
        "source_size": FILES[name][0],
        "documents": documents,
        "tokens": tokens,
        "tokens_sha256": sha256,
        "token_count_mismatches": mismatches,
        "mismatch_first_ids": first_ids,
        "seconds": time.perf_counter() - began,
    }
    # Last, so a killed run leaves a file without a record and redoes it.
    write_json(done_path(work, name), record)
    return record


def tokenize_files(raw, work, names, threads=THREADS):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    enc = None
    for name in names:
        if load_done(work, name) is not None:
            print(f"{name}: already tokenized, skipping", flush=True)
            continue
        path = raw_file_path(raw, name)
        if not path.is_file():
            raise FineWebError(f"{name} isn't in the raw directory; run download first")
        if path.stat().st_size != FILES[name][0]:
            raise FineWebError(f"{name} has the wrong size; run download again")
        enc = enc or get_encoder()
        record = tokenize_file(path, work, name, enc, threads)
        seconds = record["seconds"]
        print(
            f"{name}: {record['documents']:,} documents, {record['tokens']:,} tokens, {seconds:.1f} s, "
            f"{record['tokens'] / max(seconds, 1e-9):,.0f} tokens/s, "
            f"{record['token_count_mismatches']:,} token_count mismatches"
            + (f" (first ids: {', '.join(record['mismatch_first_ids'])})" if record["mismatch_first_ids"] else ""),
            flush=True,
        )


# -------------------------------------------------------------------- shard

class ShardWriter:
    """Fills one preallocated buffer with the stream's tokens and writes it as a
    token file each time it is full (and, at close, once more if partly full).
    Shard 0 is the held-out one."""

    def __init__(self, out, shard_tokens):
        self.out = Path(out)
        self.buffer = np.empty(shard_tokens, dtype="<u2")
        self.bytes = memoryview(self.buffer).cast("B")
        self.fill = 0
        self.documents = 0
        self.records = []

    def add(self, source, count, starts_document):
        """Reads `count` tokens from the stream into the buffer, flushing
        whenever it fills. A document is counted in the shard its first token
        lands in."""
        first = True
        while count > 0:
            # Flushing lazily keeps a document that begins right after a full
            # shard in the next shard's count.
            if self.fill == len(self.buffer):
                self.flush()
            take = min(count, len(self.buffer) - self.fill)
            read_exactly(source, self.bytes[2 * self.fill:2 * (self.fill + take)])
            if first and starts_document:
                self.documents += 1
            first = False
            self.fill += take
            count -= take

    def flush(self):
        index = len(self.records)
        name = f"shard_{index:03d}.bin"
        sha256 = write_token_file(self.out / name, self.buffer[:self.fill])
        self.records.append({"name": name, "role": "held_out" if index == 0 else "train", "tokens": self.fill,
                             "documents_started": self.documents, "sha256": sha256})
        self.fill = 0
        self.documents = 0

    def close(self):
        if self.fill:
            self.flush()


def require_done_records(work, names):
    records = {}
    missing = []
    for name in names:
        record = load_done(work, name)
        if record is None:
            missing.append(name)
        else:
            records[name] = record
    if missing:
        raise FineWebError("no valid done-record for " + ", ".join(missing) + "; run tokenize first")
    return records


def remove_old_outputs(out):
    """Clears what an earlier shard run left, the manifest first, so a run that
    dies midway can't pass for a finished one."""
    out = Path(out)
    (out / MANIFEST).unlink(missing_ok=True)
    for pattern in ("shard_*.bin", "shard_*.bin.tmp", "val_b.bin", "val_b.bin.tmp", MANIFEST + ".tmp"):
        for path in out.glob(pattern):
            path.unlink()


def build_manifest(names, records, writer_records, val_b_record, dropped, shard_tokens, val_b_rate):
    held_out = writer_records[0]["tokens"]
    train = sum(r["tokens"] for r in writer_records[1:])
    tokenized = sum(r["tokens"] for r in records.values())
    mismatches = {name: {"count": r["token_count_mismatches"], "first_ids": r["mismatch_first_ids"]}
                  for name, r in records.items()}
    return {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "repo": REPO_ID,
            "repo_type": REPO_TYPE,
            "revision": REVISION,
            "folder": SUBFOLDER,
            "files": [
                {"name": name, "size": FILES[name][0], "sha256": FILES[name][1],
                 "documents": records[name]["documents"], "tokens": records[name]["tokens"],
                 "tokens_sha256": records[name]["tokens_sha256"]}
                for name in names
            ],
        },
        "tokenizer": {
            "library": "tiktoken",
            "version": importlib.metadata.version("tiktoken"),
            "encoding": ENCODING,
            "n_vocab": N_VOCAB,
            "eot": EOT,
            "tokenizer_id": TOKENIZER_ID,
            "document": "[eot] + encode_ordinary(text), in file order and row order",
        },
        "format": {"magic": MAGIC.decode("ascii"), "version": FORMAT_VERSION, "header_bytes": HEADER_BYTES,
                   "dtype": "uint16", "byte_order": "little"},
        "shard_tokens": shard_tokens,
        "val_b": {"rule": VAL_B_RULE, "val_b_rate": val_b_rate},
        "val_a": {"file": "shard_000.bin", "tokens": min(VAL_A_TOKENS, shard_tokens)},
        "dropped_straddle_tokens": dropped,
        "files": writer_records + [val_b_record],
        "totals": {
            "tokenized": tokenized,
            "documents": sum(r["documents"] for r in records.values()),
            "held_out": held_out,
            "train": train,
            "val_b": val_b_record["tokens"],
            "dropped": dropped,
            "train_shards": len(writer_records) - 1,
        },
        "token_count_mismatches": {"documents": sum(m["count"] for m in mismatches.values()), "per_file": mismatches},
        "git": git_info(),
    }


def shard_files(work, out, names, shard_tokens=DEFAULT_SHARD_TOKENS, val_b_rate=DEFAULT_VAL_B_RATE):
    if shard_tokens < 1 or val_b_rate < 1:
        raise FineWebError("--shard-tokens and --val-b-rate must be positive")
    if sys.byteorder != "little":
        raise FineWebError("token files are little-endian and this machine isn't")
    work, out = Path(work), Path(out)
    records = require_done_records(work, names)
    tokenized = sum(r["tokens"] for r in records.values())
    if tokenized < shard_tokens:
        raise FineWebError(f"only {tokenized:,} tokens were tokenized, fewer than one shard of {shard_tokens:,}")
    for name in names:
        # The tokenize step may be long past; a token file that changed since is caught here.
        if sha256_file(tokens_path(work, name)) != records[name]["tokens_sha256"]:
            raise FineWebError(f"{name}: the token file doesn't match its done-record's sha256; tokenize it again")
    out.mkdir(parents=True, exist_ok=True)
    remove_old_outputs(out)

    writer = ShardWriter(out, shard_tokens)
    val_b_parts = []
    val_b_documents = dropped = 0
    base = 0  # global index of the current file's first token
    for name in names:
        record = records[name]
        with np.load(docs_path(work, name)) as index:
            lengths = index["lengths"].astype(np.int64)
            in_val_b = (index["hashes"] % np.uint64(val_b_rate) == 0).tolist()
        if len(lengths) != record["documents"] or int(lengths.sum()) != record["tokens"]:
            raise FineWebError(f"{name}: the document index doesn't match its done-record")
        starts = (base + np.cumsum(lengths) - lengths).tolist()
        with open(tokens_path(work, name), "rb", buffering=IO_CHUNK) as source:
            for start, length, vb in zip(starts, lengths.tolist(), in_val_b):
                if start < shard_tokens:
                    # Shard 0 is the stream's first shard_tokens tokens as they
                    # come; the straddling document keeps only what fits.
                    take = min(length, shard_tokens - start)
                    writer.add(source, take, True)
                    if take < length:
                        source.seek(2 * (length - take), os.SEEK_CUR)
                        dropped += length - take
                elif vb:
                    part = np.empty(length, dtype="<u2")
                    read_exactly(source, memoryview(part).cast("B"))
                    val_b_parts.append(part)
                    val_b_documents += 1
                else:
                    writer.add(source, length, True)
            if source.read(1):
                raise FineWebError(f"{name}: the token file is longer than its document index")
        base += record["tokens"]
    writer.close()

    val_b = np.concatenate(val_b_parts) if val_b_parts else np.empty(0, dtype="<u2")
    val_b_record = {"name": "val_b.bin", "role": "val_b", "tokens": int(val_b.size),
                    "documents": val_b_documents, "sha256": write_token_file(out / "val_b.bin", val_b)}
    manifest = build_manifest(names, records, writer.records, val_b_record, dropped, shard_tokens, val_b_rate)
    totals = manifest["totals"]
    if totals["held_out"] + totals["train"] + totals["val_b"] + totals["dropped"] != totals["tokenized"]:
        raise FineWebError("internal error: the outputs don't add up to the tokenized total")
    write_json(out / MANIFEST, manifest)
    return manifest


# ------------------------------------------------------------------- verify

def check_file_entry(out, entry):
    """Problems with one output file: header, size, checksum, token range."""
    name = entry["name"]
    path = Path(out) / name
    if not path.is_file():
        return [f"{name}: missing"]
    problems = []
    digest = hashlib.sha256()
    biggest = first = None
    eots = 0
    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
        try:
            count = parse_header(header, path.stat().st_size)
        except FineWebError as error:
            return [f"{name}: {error}"]
        digest.update(header)
        # One pass for the checksum and the token scan, in even-sized chunks.
        while True:
            chunk = f.read(IO_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            tokens = np.frombuffer(chunk, dtype="<u2")
            biggest = max(int(tokens.max()), biggest or 0)
            eots += int(np.count_nonzero(tokens == EOT))
            if first is None:
                first = int(tokens[0])
    if count != entry["tokens"]:
        problems.append(f"{name}: the header says {count} tokens, the manifest {entry['tokens']}")
    if digest.hexdigest() != entry["sha256"]:
        problems.append(f"{name}: sha256 doesn't match the manifest")
    if biggest is not None and biggest > EOT:
        problems.append(f"{name}: has a token above {EOT} ({biggest})")
    # Documents begin with EOT, and encode_ordinary never emits it, so val-B's
    # EOT count is its document count. Shard 1 starts at a document too.
    if name in ("shard_000.bin", "shard_001.bin", "val_b.bin") and first is not None and first != EOT:
        problems.append(f"{name}: doesn't start with EOT")
    if entry["role"] == "val_b" and eots != entry["documents"]:
        problems.append(f"{name}: {eots} EOT tokens for {entry['documents']} documents")
    return problems


def verify_outputs(out):
    """Every problem found in a shard output directory, as messages."""
    out = Path(out)
    try:
        manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
        entries = manifest["files"]
        totals = manifest["totals"]
        shard_tokens = manifest["shard_tokens"]
        sources = manifest["dataset"]["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return [f"{MANIFEST}: missing or unreadable"]
    problems = []
    names = [e["name"] for e in entries]
    shards = [e for e in entries if e["role"] in ("held_out", "train")]
    if names != [f"shard_{i:03d}.bin" for i in range(len(shards))] + ["val_b.bin"] or (
        shards and shards[0]["role"] != "held_out"
    ):
        problems.append("the manifest's file list isn't shard_000.bin, shard_001.bin, ..., val_b.bin")
    for position, entry in enumerate(shards):
        last = position == len(shards) - 1
        if entry["tokens"] != shard_tokens and (position == 0 or not last):
            problems.append(f"{entry['name']}: {entry['tokens']} tokens, not {shard_tokens}")
        if entry["tokens"] > shard_tokens or entry["tokens"] == 0:
            problems.append(f"{entry['name']}: {entry['tokens']} tokens isn't within 1..{shard_tokens}")
    for entry in entries:
        problems.extend(check_file_entry(out, entry))
    listed = set(names) | {MANIFEST}
    for path in sorted(out.iterdir()):
        if path.name not in listed:
            problems.append(f"{path.name}: not in the manifest")
    held_out = sum(e["tokens"] for e in entries if e["role"] == "held_out")
    train = sum(e["tokens"] for e in entries if e["role"] == "train")
    val_b = sum(e["tokens"] for e in entries if e["role"] == "val_b")
    expected = {"held_out": held_out, "train": train, "val_b": val_b}
    for key, value in expected.items():
        if totals.get(key) != value:
            problems.append(f"totals.{key} is {totals.get(key)}, the files add up to {value}")
    tokenized = sum(s["tokens"] for s in sources)
    if totals.get("tokenized") != tokenized:
        problems.append(f"totals.tokenized is {totals.get('tokenized')}, the source files add up to {tokenized}")
    dropped = manifest.get("dropped_straddle_tokens")
    if totals.get("dropped") != dropped:
        problems.append("totals.dropped differs from dropped_straddle_tokens")
    if held_out + train + val_b + (dropped or 0) != tokenized:
        problems.append(
            f"held-out {held_out} + train {train} + val-B {val_b} + dropped {dropped} "
            f"= {held_out + train + val_b + (dropped or 0)}, not the {tokenized} tokenized"
        )
    return problems


# -------------------------------------------------------------------- check

def check_against_reference(raw, work, name, docs, enc):
    """Tokenizes the first `docs` documents of one file with llm.c's plain loop
    and compares them with the start of the file's token file. Returns
    (documents compared, tokens compared, mismatch description or None)."""
    path = raw_file_path(raw, name)
    if not path.is_file():
        raise FineWebError(f"{name} isn't in the raw directory")
    if not tokens_path(work, name).is_file():
        raise FineWebError(f"{name} hasn't been tokenized")
    with np.load(docs_path(work, name)) as index:
        lengths = index["lengths"]
    compared = tokens_seen = 0
    with open(tokens_path(work, name), "rb") as stored, pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=1000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if compared == docs:
                    return compared, tokens_seen, None
                reference = np.array([EOT] + enc.encode_ordinary(text), dtype="<u2")
                if compared >= len(lengths) or int(lengths[compared]) != len(reference):
                    return compared, tokens_seen, f"document {compared}: the index lengths differ from the reference"
                found = np.empty(len(reference), dtype="<u2")
                read_exactly(stored, memoryview(found).cast("B"))
                if not np.array_equal(found, reference):
                    where = int(np.flatnonzero(found != reference)[0])
                    return compared, tokens_seen, f"document {compared}: first difference at its token {where}"
                compared += 1
                tokens_seen += len(reference)
    return compared, tokens_seen, None


# ---------------------------------------------------------------------- CLI

def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add_files(command):
        command.add_argument("--files", help="file indices: 0, 0-3 or 0,5,13 (default: all)")

    download = commands.add_parser("download", help="fetch and verify the parquet files")
    download.add_argument("--raw", required=True, help="directory for the parquet files")
    add_files(download)

    tokenize = commands.add_parser("tokenize", help="tokenize each parquet file into the work directory")
    tokenize.add_argument("--raw", required=True)
    tokenize.add_argument("--work", required=True)
    tokenize.add_argument("--threads", type=int, default=THREADS)
    add_files(tokenize)

    shard = commands.add_parser("shard", help="write the shards, val_b.bin and manifest.json")
    shard.add_argument("--work", required=True)
    shard.add_argument("--out", required=True)
    shard.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    shard.add_argument("--val-b-rate", type=int, default=DEFAULT_VAL_B_RATE)
    add_files(shard)

    verify = commands.add_parser("verify", help="recheck the outputs against the manifest")
    verify.add_argument("--out", required=True)

    check = commands.add_parser("check", help="compare a file's first documents with a reference loop")
    check.add_argument("--raw", required=True)
    check.add_argument("--work", required=True)
    check.add_argument("--file", required=True, help="a parquet file name")
    check.add_argument("--docs", type=int, required=True)
    return parser.parse_args(argv)


def run_command(args, downloader):
    names = expected_names()
    if args.command == "download":
        download_files(args.raw, parse_file_spec(args.files, names), downloader)
    elif args.command == "tokenize":
        if args.threads < 1:
            raise FineWebError("--threads must be positive")
        tokenize_files(args.raw, args.work, parse_file_spec(args.files, names), args.threads)
    elif args.command == "shard":
        manifest = shard_files(args.work, args.out, parse_file_spec(args.files, names),
                               args.shard_tokens, args.val_b_rate)
        totals = manifest["totals"]
        print(f"wrote {len(manifest['files'])} files: held out {totals['held_out']:,} tokens, "
              f"train {totals['train']:,} tokens in {totals['train_shards']} shards, "
              f"val-B {totals['val_b']:,} tokens, dropped {totals['dropped']:,} of {totals['tokenized']:,}")
    elif args.command == "verify":
        problems = verify_outputs(args.out)
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        if problems:
            print(f"verify failed: {len(problems)} problem(s)", file=sys.stderr)
            return 1
        manifest = json.loads((Path(args.out) / MANIFEST).read_text(encoding="utf-8"))
        print(f"verify ok: {len(manifest['files'])} files, {manifest['totals']['tokenized']:,} tokens accounted for")
    elif args.command == "check":
        if args.file not in FILES or args.docs < 1:
            raise FineWebError("--file must be one of the dataset's file names, and --docs positive")
        compared, tokens, problem = check_against_reference(args.raw, args.work, args.file, args.docs, get_encoder())
        if problem is not None:
            print(f"check FAILED after {compared} documents: {problem}", file=sys.stderr)
            return 1
        print(f"check ok: {compared:,} documents, {tokens:,} tokens identical to the reference loop")
    return 0


def main(argv=None, downloader=hf_download):
    args = parse_args(argv)
    started = time.perf_counter()
    try:
        code = run_command(args, downloader)
    except FineWebError as error:
        print(f"error: {error}", file=sys.stderr)
        code = 1
    report_resources(started)
    return code


if __name__ == "__main__":
    sys.exit(main())
