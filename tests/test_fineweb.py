"""scripts/fineweb.py: token files, tokenizing, shard rules, resume, verify, download and check, on synthetic data.

Nothing here touches the network except tiktoken's first fetch of its small
GPT-2 BPE files. The synthetic parquet files stand in for FineWeb-Edu's: tiny
shards and a small val-B rate make every rule fire within a few thousand tokens.
"""

import getpass
import hashlib
import importlib.util
import json
import os
import random
import shutil
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import flint_lib

SCRIPTS = flint_lib.ROOT / "scripts"
spec = importlib.util.spec_from_file_location("fineweb", SCRIPTS / "fineweb.py")
fineweb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fineweb)

EOT = 50256
NAMES = ["000_00000.parquet", "001_00000.parquet", "002_00000.parquet"]
DOCS_PER_FILE = [20, 17, 23]
ROW_GROUP = 6
SHARD_TOKENS = 400
VAL_B_RATE = 3
PLANTED = {"000_00000.parquet": {7: 3}, "002_00000.parquet": {2: 5, 11: -1}}  # row -> error in token_count

WORDS = ("the of and to in is that for it as was with be by on not he this are or his from at which but have "
         "an had they you were their one all we can her has there been if more when will would who so no").split()
UNICODE = ["naïve café — “quotes”", "日本語のテキストです。", "emoji 😀🎉 and ligature ﬁ", "Ünïcödé Ω≈ç√∫",
           "Привет, мир", "line one\r\nline two\n\n\ttabbed"]


# ------------------------------------------------------------ the synthetic set

def make_text(rng, file_index, row):
    if row == 1:
        return ""
    if row == 2:
        return "  \n\t \n "
    if row == 3:
        return " ".join(UNICODE)
    if row == 4:
        return "before <|endoftext|> after <|endoftext|><|endoftext|>"
    if (file_index, row) in ((0, 9), (1, 5), (1, 6), (2, 8)):  # val-B and training
        return " ".join(rng.choice(WORDS) for _ in range(650))  # longer than a shard
    parts = [rng.choice(WORDS) for _ in range(rng.randint(3, 90))]
    if rng.random() < 0.25:
        parts.insert(rng.randrange(len(parts)), rng.choice(UNICODE))
    return " ".join(parts)


def reference_tokens(enc, text):
    """llm.c's loop body."""
    return [EOT] + enc.encode_ordinary(text)


def expected_outputs(stream, shard_tokens, rate):
    """The shard rules, written the slow obvious way, from [(id, tokens)]."""
    flat = [t for _, tokens in stream for t in tokens]
    held_out = flat[:shard_tokens]
    train, val_b, val_b_documents, dropped, position = [], [], 0, 0, 0
    inside = []  # ids of documents that start in shard 0
    for doc_id, tokens in stream:
        start, end = position, position + len(tokens)
        position = end
        if start < shard_tokens:
            inside.append(doc_id)
            dropped += max(end - shard_tokens, 0)
        elif int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:8], "little") % rate == 0:
            val_b += tokens
            val_b_documents += 1
        else:
            train += tokens
    shards = [train[i:i + shard_tokens] for i in range(0, len(train), shard_tokens)]
    return SimpleNamespace(held_out=held_out, shards=shards, train=train, val_b=val_b,
                           val_b_documents=val_b_documents, dropped=dropped, inside=inside, total=len(flat))


@pytest.fixture(scope="module")
def enc():
    return fineweb.get_encoder()


@pytest.fixture(scope="module", autouse=True)
def synthetic(enc, tmp_path_factory):
    """Writes the parquet files, points fineweb.FILES at them for the whole
    module, and tokenizes them once into a shared work directory."""
    base = tmp_path_factory.mktemp("fineweb")
    raw = base / "raw"
    raw.mkdir()
    rng = random.Random(20260301)
    docs, table = [], {}
    for file_index, (name, count) in enumerate(zip(NAMES, DOCS_PER_FILE)):
        ids, texts, counts = [], [], []
        for row in range(count):
            text = make_text(rng, file_index, row)
            ids.append(f"<urn:uuid:{file_index:02d}-{row:04d}>")
            texts.append(text)
            counts.append(len(enc.encode_ordinary(text)) + PLANTED.get(name, {}).get(row, 0))
        pq.write_table(
            pa.table({
                "text": texts, "id": ids, "dump": ["CC-MAIN-2024-10"] * count,
                "url": [f"http://example.com/{i}" for i in range(count)], "language": ["en"] * count,
                "language_score": pa.array([0.99] * count, pa.float64()),
                "token_count": pa.array(counts, pa.int64()), "score": pa.array([3.1] * count, pa.float64()),
                "int_score": pa.array([3] * count, pa.int64()),
            }),
            raw / name, row_group_size=ROW_GROUP,
        )
        docs += [(name, i, t) for i, t in zip(ids, texts)]
        data = (raw / name).read_bytes()
        table[name] = (len(data), hashlib.sha256(data).hexdigest())
    stream = [(doc_id, reference_tokens(enc, text)) for _, doc_id, text in docs]
    patch = pytest.MonkeyPatch()
    patch.setattr(fineweb, "FILES", table)
    work = base / "work"
    assert fineweb.main(["tokenize", "--raw", str(raw), "--work", str(work), "--threads", "2"]) == 0
    yield SimpleNamespace(raw=raw, work=work, docs=docs, stream=stream, table=table, base=base,
                          planted_ids={f"<urn:uuid:{NAMES.index(n):02d}-{r:04d}>" for n, rows in PLANTED.items() for r in rows})
    patch.undo()


def run(*argv):
    return fineweb.main([str(a) for a in argv])


def shard_into(ds, out, shard_tokens=SHARD_TOKENS, rate=VAL_B_RATE, work=None):
    code = run("shard", "--work", work or ds.work, "--out", out, "--shard-tokens", shard_tokens,
               "--val-b-rate", rate)
    assert code == 0
    return json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))


def tokens_of(path):
    return np.array(fineweb.read_token_file(path))


def read_all_bytes(directory):
    return {p.name: p.read_bytes() for p in sorted(Path(directory).iterdir()) if p.name != "manifest.json"}


def without_timestamp(manifest):
    return {k: v for k, v in manifest.items() if k != "created_utc"}


# ---------------------------------------------------------------- token files

def test_header_round_trip(tmp_path):
    path = tmp_path / "tokens.bin"
    tokens = np.array([50256, 0, 1, 65535, 12345], dtype=np.uint16)
    sha = fineweb.write_token_file(path, tokens)
    raw = path.read_bytes()
    assert raw[:8] == b"FLINTTOK"
    assert int.from_bytes(raw[8:12], "little") == 1 and int.from_bytes(raw[12:16], "little") == 1
    assert int.from_bytes(raw[16:24], "little") == 5 and raw[24:64] == bytes(40)
    assert raw[64:] == tokens.astype("<u2").tobytes() and len(raw) == 64 + 2 * 5
    assert sha == hashlib.sha256(raw).hexdigest()
    loaded = fineweb.read_token_file(path)
    assert loaded.dtype == np.dtype("<u2") and np.array_equal(loaded, tokens)
    assert loaded.flags.writeable is False
    assert not list(tmp_path.glob("*.tmp"))
    del loaded


def test_wider_integer_arrays_are_converted_when_they_fit(tmp_path):
    fineweb.write_token_file(tmp_path / "a.bin", np.array([1, 50256], dtype=np.int64))
    assert tokens_of(tmp_path / "a.bin").tolist() == [1, 50256]
    with pytest.raises(ValueError):
        fineweb.write_token_file(tmp_path / "b.bin", np.array([70000], dtype=np.int64))
    assert not (tmp_path / "b.bin").exists() and not list(tmp_path.glob("*.tmp"))


def test_an_empty_token_file_round_trips(tmp_path):
    fineweb.write_token_file(tmp_path / "empty.bin", np.empty(0, dtype=np.uint16))
    assert (tmp_path / "empty.bin").stat().st_size == 64
    assert len(fineweb.read_token_file(tmp_path / "empty.bin")) == 0


def corrupt(mutate):
    def apply(path):
        raw = bytearray(path.read_bytes())
        result = mutate(raw)
        path.write_bytes(bytes(result if result is not None else raw))
    return apply


BAD_FILES = {
    "magic": corrupt(lambda b: b.__setitem__(0, ord("X"))),
    "version": corrupt(lambda b: b.__setitem__(8, 2)),
    "tokenizer id": corrupt(lambda b: b.__setitem__(12, 7)),
    "reserved bytes": corrupt(lambda b: b.__setitem__(40, 1)),
    "count too big": corrupt(lambda b: b.__setitem__(16, 9)),
    "count too small": corrupt(lambda b: b.__setitem__(16, 2)),
    "truncated": corrupt(lambda b: b[:-1]),
    "extra byte": corrupt(lambda b: b + b"\0"),
    "shorter than the header": corrupt(lambda b: b[:30]),
}


@pytest.mark.parametrize("case", sorted(BAD_FILES))
def test_bad_token_files_are_rejected(tmp_path, case):
    path = tmp_path / "tokens.bin"
    fineweb.write_token_file(path, np.arange(5, dtype=np.uint16))
    BAD_FILES[case](path)
    with pytest.raises(fineweb.FineWebError):
        fineweb.read_token_file(path)


def test_a_failed_write_leaves_neither_the_target_nor_a_tmp_file(tmp_path):
    def explode(f):
        f.write(b"partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fineweb.atomic_write(tmp_path / "x.bin", explode)
    assert list(tmp_path.iterdir()) == []
    fineweb.atomic_write(tmp_path / "x.bin", lambda f: f.write(b"old"))
    with pytest.raises(RuntimeError):
        fineweb.atomic_write(tmp_path / "x.bin", explode)
    assert (tmp_path / "x.bin").read_bytes() == b"old"


# ---------------------------------------------------------------- the dataset

def test_fixture_exercises_every_rule(synthetic):
    """Guards the tests below against passing vacuously."""
    expected = expected_outputs(synthetic.stream, SHARD_TOKENS, VAL_B_RATE)
    assert len(expected.shards) >= 5 and 0 < len(expected.shards[-1]) < SHARD_TOKENS
    assert expected.dropped > 0
    assert expected.val_b_documents >= 3
    # A val-B-eligible document inside shard 0 must be skipped by the rule.
    hashes = {d: int.from_bytes(hashlib.sha256(d.encode()).digest()[:8], "little") for d in expected.inside}
    assert any(h % VAL_B_RATE == 0 for h in hashes.values())
    lengths = [len(t) for _, t in synthetic.stream]
    assert max(lengths) > SHARD_TOKENS and min(lengths) == 1
    texts = [t for _, _, t in synthetic.docs]
    assert "" in texts and any(t and not t.strip() for t in texts) and any("<|endoftext|>" in t for t in texts)
    # A document longer than a shard is training data, spanning shard boundaries.
    long_ids = [d for d, t in synthetic.stream if len(t) > SHARD_TOKENS]
    assert any(d not in expected.inside
               and int.from_bytes(hashlib.sha256(d.encode()).digest()[:8], "little") % VAL_B_RATE != 0
               for d in long_ids)


def test_each_document_is_eot_plus_encode_ordinary_in_file_then_row_order(synthetic):
    position = 0
    for name in NAMES:
        done = json.loads((synthetic.work / f"done_{name[:-8]}.json").read_text(encoding="utf-8"))
        stored = np.fromfile(synthetic.work / f"tokens_{name[:-8]}.bin", dtype="<u2")
        mine = [tokens for (n, _, _), (_, tokens) in zip(synthetic.docs, synthetic.stream) if n == name]
        assert stored.tolist() == [t for tokens in mine for t in tokens]
        with np.load(synthetic.work / f"docs_{name[:-8]}.npz") as index:
            assert index["lengths"].dtype == np.uint32 and index["hashes"].dtype == np.uint64
            assert index["lengths"].tolist() == [len(t) for t in mine]
        ids = [d for n, d, _ in synthetic.docs if n == name]
        with np.load(synthetic.work / f"docs_{name[:-8]}.npz") as index:
            expected_hashes = [int.from_bytes(hashlib.sha256(d.encode()).digest()[:8], "little") for d in ids]
            assert index["hashes"].tolist() == expected_hashes
        assert done["documents"] == len(mine) and done["tokens"] == len(stored)
        assert done["tokens_sha256"] == hashlib.sha256(stored.tobytes()).hexdigest()
        position += len(stored)
    assert position == sum(len(t) for _, t in synthetic.stream)


def test_token_count_mismatches_match_the_planted_errors(synthetic):
    total = 0
    for name in NAMES:
        done = json.loads((synthetic.work / f"done_{name[:-8]}.json").read_text(encoding="utf-8"))
        planted = [f"<urn:uuid:{NAMES.index(name):02d}-{row:04d}>" for row in PLANTED.get(name, {})]
        assert done["token_count_mismatches"] == len(planted)
        assert done["mismatch_first_ids"] == planted
        total += len(planted)
    assert total == 3


def test_tokenize_prints_per_file_numbers_and_skips_done_files(synthetic, capsys):
    assert run("tokenize", "--raw", synthetic.raw, "--work", synthetic.work) == 0
    out = capsys.readouterr().out
    assert out.count("already tokenized, skipping") == 3 and "elapsed" in out
    fresh = synthetic.base / "work-printing"
    assert run("tokenize", "--raw", synthetic.raw, "--work", fresh, "--files", "2") == 0
    out = capsys.readouterr().out
    line = out.splitlines()[0]
    assert line.startswith("002_00000.parquet: 23 documents") and "tokens/s" in line
    assert "2 token_count mismatches (first ids: <urn:uuid:02-0002>, <urn:uuid:02-0011>)" in line
    assert sorted(p.name for p in fresh.iterdir()) == ["docs_002_00000.npz", "done_002_00000.json",
                                                       "tokens_002_00000.bin"]


def test_tokenizing_is_the_same_for_any_thread_count(synthetic, enc):
    texts = [t for _, _, t in synthetic.docs[:12]]
    one = fineweb.tokenize_texts(enc, texts, 1)
    three = fineweb.tokenize_texts(enc, texts, 3)
    assert np.array_equal(one[0], three[0]) and np.array_equal(one[1], three[1])
    assert one[0].dtype == np.dtype("<u2") and one[1].dtype == np.uint32
    flat, lengths = fineweb.tokenize_texts(enc, [], 2)
    assert len(flat) == 0 and len(lengths) == 0


def test_tokenize_refuses_a_raw_file_of_the_wrong_size(synthetic, capsys):
    raw = synthetic.base / "raw-short"
    raw.mkdir()
    (raw / NAMES[0]).write_bytes((synthetic.raw / NAMES[0]).read_bytes()[:-8])
    assert run("tokenize", "--raw", raw, "--work", synthetic.base / "work-short", "--files", "0") == 1
    assert "wrong size" in capsys.readouterr().err
    assert run("tokenize", "--raw", synthetic.base / "nowhere", "--work", synthetic.base / "work-short",
               "--files", "0") == 1
    assert "run download first" in capsys.readouterr().err


# ------------------------------------------------------------ shard rules

def check_outputs(synthetic, out, shard_tokens, rate):
    expected = expected_outputs(synthetic.stream, shard_tokens, rate)
    manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
    names = [f"shard_{i:03d}.bin" for i in range(1 + len(expected.shards))] + ["val_b.bin"]
    assert sorted(p.name for p in Path(out).iterdir()) == sorted(names + ["manifest.json"])
    assert tokens_of(Path(out) / "shard_000.bin").tolist() == expected.held_out
    for i, shard in enumerate(expected.shards, start=1):
        assert tokens_of(Path(out) / f"shard_{i:03d}.bin").tolist() == shard
    assert tokens_of(Path(out) / "val_b.bin").tolist() == expected.val_b
    totals = manifest["totals"]
    assert totals["held_out"] == shard_tokens == len(expected.held_out)
    assert totals["train"] == len(expected.train) and totals["val_b"] == len(expected.val_b)
    assert totals["dropped"] == manifest["dropped_straddle_tokens"] == expected.dropped
    assert totals["tokenized"] == expected.total
    assert totals["held_out"] + totals["train"] + totals["val_b"] + totals["dropped"] == totals["tokenized"]
    assert [f["name"] for f in manifest["files"]] == names
    assert [f["role"] for f in manifest["files"]] == ["held_out"] + ["train"] * len(expected.shards) + ["val_b"]
    assert manifest["files"][-1]["documents"] == expected.val_b_documents
    for entry in manifest["files"]:
        assert entry["sha256"] == hashlib.sha256((Path(out) / entry["name"]).read_bytes()).hexdigest()
    # Every training shard but the last holds exactly shard_tokens tokens.
    assert all(len(s) == shard_tokens for s in expected.shards[:-1])
    assert run("verify", "--out", out) == 0
    return expected, manifest


def boundary(synthetic, documents):
    return sum(len(t) for _, t in synthetic.stream[:documents])


@pytest.mark.parametrize("shard_tokens, rate", [
    (SHARD_TOKENS, VAL_B_RATE),
    (333, 4),
    (100, 5),
    (None, 3),  # shard 0 ends exactly at a document boundary
    (None, 1),  # the whole stream is shard 0
])
def test_shard_rules_hold_for_several_sizes_and_rates(synthetic, tmp_path, shard_tokens, rate):
    if shard_tokens is None:
        shard_tokens = boundary(synthetic, 11 if rate == 3 else len(synthetic.stream))
    shard_into(synthetic, tmp_path / "out", shard_tokens, rate)
    expected, manifest = check_outputs(synthetic, tmp_path / "out", shard_tokens, rate)
    if rate == 3 and shard_tokens == boundary(synthetic, 11):
        assert expected.dropped == 0 and manifest["dropped_straddle_tokens"] == 0
    if shard_tokens == expected.total:
        assert len(expected.shards) == 0 and expected.val_b == [] and manifest["totals"]["train_shards"] == 0
    else:
        assert manifest["totals"]["train_shards"] == len(expected.shards)


def test_shard_0_is_the_first_tokens_of_the_unfiltered_stream(synthetic, tmp_path):
    manifest = shard_into(synthetic, tmp_path / "out")
    flat = [t for _, tokens in synthetic.stream for t in tokens]
    assert tokens_of(tmp_path / "out" / "shard_000.bin").tolist() == flat[:SHARD_TOKENS]
    assert manifest["val_a"] == {"file": "shard_000.bin", "tokens": SHARD_TOKENS}


def test_the_straddling_documents_remainder_is_dropped_and_counted(synthetic, tmp_path):
    manifest = shard_into(synthetic, tmp_path / "out")
    straddler, position = None, 0
    for doc_id, tokens in synthetic.stream:
        if position < SHARD_TOKENS < position + len(tokens):
            straddler = (doc_id, tokens, position)
            break
        position += len(tokens)
    doc_id, tokens, start = straddler
    kept = SHARD_TOKENS - start
    assert manifest["dropped_straddle_tokens"] == len(tokens) - kept > 0
    train = np.concatenate([tokens_of(tmp_path / "out" / f["name"]) for f in manifest["files"] if f["role"] == "train"])
    # Training starts at the next document; the remainder isn't anywhere in it.
    following = synthetic.stream[[d for d, _ in synthetic.stream].index(doc_id) + 1:]
    first_training = next(t for d, t in following
                          if int.from_bytes(hashlib.sha256(d.encode()).digest()[:8], "little") % VAL_B_RATE)
    assert train[:len(first_training)].tolist() == first_training


def test_val_b_is_deterministic_whole_and_never_from_shard_0(synthetic, tmp_path):
    shard_into(synthetic, tmp_path / "a")
    shard_into(synthetic, tmp_path / "b")
    assert (tmp_path / "a" / "val_b.bin").read_bytes() == (tmp_path / "b" / "val_b.bin").read_bytes()
    expected = expected_outputs(synthetic.stream, SHARD_TOKENS, VAL_B_RATE)
    val_b = tokens_of(tmp_path / "a" / "val_b.bin")
    # Re-split val-B at its EOT tokens: whole documents, in stream order.
    starts = np.flatnonzero(val_b == EOT).tolist() + [len(val_b)]
    documents = [val_b[a:b].tolist() for a, b in zip(starts, starts[1:])]
    eligible = [(d, t) for d, t in synthetic.stream
                if d not in expected.inside
                and int.from_bytes(hashlib.sha256(d.encode()).digest()[:8], "little") % VAL_B_RATE == 0]
    assert documents == [t for _, t in eligible] and len(documents) == expected.val_b_documents
    # The rule had something to skip: eligible documents inside shard 0.
    assert any(int.from_bytes(hashlib.sha256(d.encode()).digest()[:8], "little") % VAL_B_RATE == 0
               for d in expected.inside)


def test_a_different_val_b_rate_picks_different_documents(synthetic, tmp_path):
    shard_into(synthetic, tmp_path / "a", rate=3)
    shard_into(synthetic, tmp_path / "b", rate=4)
    assert (tmp_path / "a" / "val_b.bin").read_bytes() != (tmp_path / "b" / "val_b.bin").read_bytes()


def test_documents_started_per_shard_add_up(synthetic, tmp_path):
    manifest = shard_into(synthetic, tmp_path / "out")
    started = sum(f["documents_started"] for f in manifest["files"] if f["role"] != "val_b")
    assert started + manifest["files"][-1]["documents"] == manifest["totals"]["documents"]
    # The straddler starts in shard 0 and is counted there; no shard counts it twice.
    expected = expected_outputs(synthetic.stream, SHARD_TOKENS, VAL_B_RATE)
    assert manifest["files"][0]["documents_started"] == len(expected.inside)


def test_shard_needs_enough_tokens_and_sane_numbers(synthetic, tmp_path, capsys):
    total = sum(len(t) for _, t in synthetic.stream)
    assert run("shard", "--work", synthetic.work, "--out", tmp_path / "o", "--shard-tokens", total + 1) == 1
    assert "fewer than one shard" in capsys.readouterr().err
    assert run("shard", "--work", synthetic.work, "--out", tmp_path / "o", "--val-b-rate", 0) == 1
    assert run("shard", "--work", synthetic.work, "--out", tmp_path / "o", "--shard-tokens", 0) == 1
    assert not (tmp_path / "o" / "manifest.json").exists()


def test_shard_refuses_unless_every_expected_file_is_done(synthetic, tmp_path, capsys):
    work = tmp_path / "work"
    shutil.copytree(synthetic.work, work)
    (work / "done_001_00000.json").unlink()
    out = tmp_path / "out"
    assert run("shard", "--work", work, "--out", out, "--shard-tokens", SHARD_TOKENS) == 1
    assert "001_00000.parquet" in capsys.readouterr().err
    assert not out.exists() or not any(out.iterdir())
    # Naming only the finished files is allowed.
    assert run("shard", "--work", work, "--out", out, "--shard-tokens", 200, "--files", "0,2") == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [f["name"] for f in manifest["dataset"]["files"]] == [NAMES[0], NAMES[2]]


def test_shard_catches_a_token_file_that_changed_since_tokenizing(synthetic, tmp_path, capsys):
    work = tmp_path / "work"
    shutil.copytree(synthetic.work, work)
    corrupt(lambda b: b.__setitem__(10, b[10] ^ 1))(work / "tokens_001_00000.bin")
    assert run("shard", "--work", work, "--out", tmp_path / "out", "--shard-tokens", SHARD_TOKENS) == 1
    assert "001_00000.parquet" in capsys.readouterr().err
    # An incomplete record (here: a truncated token file) counts as not done.
    shutil.copytree(synthetic.work, tmp_path / "work2")
    path = tmp_path / "work2" / "tokens_000_00000.bin"
    path.write_bytes(path.read_bytes()[:-2])
    assert fineweb.load_done(tmp_path / "work2", NAMES[0]) is None


def test_a_rerun_into_the_same_directory_clears_stale_outputs(synthetic, tmp_path):
    out = tmp_path / "out"
    shard_into(synthetic, out, shard_tokens=100, rate=5)
    first = len(list(out.glob("shard_*.bin")))
    (out / "shard_999.bin.tmp").write_bytes(b"junk")
    shard_into(synthetic, out)
    assert len(list(out.glob("shard_*.bin"))) < first
    check_outputs(synthetic, out, SHARD_TOKENS, VAL_B_RATE)
    assert not list(out.glob("*.tmp"))


# --------------------------------------------------------- resume and repeat

def run_tokenize(ds, work, *extra):
    return run("tokenize", "--raw", ds.raw, "--work", work, "--threads", 2, *extra)


def record_without_seconds(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("seconds")
    return record


def same_work(a, b):
    assert sorted(p.name for p in a.iterdir()) == sorted(p.name for p in b.iterdir())
    for path in sorted(a.iterdir()):
        if path.suffix == ".bin":
            assert path.read_bytes() == (b / path.name).read_bytes(), path.name
        elif path.suffix == ".json":
            assert record_without_seconds(path) == record_without_seconds(b / path.name), path.name
        else:
            with np.load(path) as x, np.load(b / path.name) as y:
                assert all(np.array_equal(x[k], y[k]) for k in x.files), path.name


def test_a_killed_tokenize_run_resumes_to_byte_identical_outputs(synthetic, tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    real_texts = fineweb.tokenize_texts
    calls = []

    def dies_in_the_second_file(enc, texts, threads):
        calls.append(len(texts))
        if len(calls) == 6:  # file 0 has four row groups, so this is file 1's second
            raise RuntimeError("killed")
        return real_texts(enc, texts, threads)

    monkeypatch.setattr(fineweb, "tokenize_texts", dies_in_the_second_file)
    with pytest.raises(RuntimeError, match="killed"):
        run_tokenize(synthetic, work)
    monkeypatch.setattr(fineweb, "tokenize_texts", real_texts)
    assert (work / "done_000_00000.json").exists() and not (work / "done_001_00000.json").exists()
    assert not list(work.glob("*.tmp"))
    # A hard kill leaves tmp files behind; a rerun must not trust them.
    (work / "tokens_001_00000.bin.tmp").write_bytes(b"half a file")

    real_file = fineweb.tokenize_file
    tokenized = []

    def spy(raw_path, work_dir, name, enc, threads):
        tokenized.append(name)
        return real_file(raw_path, work_dir, name, enc, threads)

    monkeypatch.setattr(fineweb, "tokenize_file", spy)
    capsys.readouterr()
    assert run_tokenize(synthetic, work) == 0
    assert tokenized == NAMES[1:]
    assert "000_00000.parquet: already tokenized, skipping" in capsys.readouterr().out
    # The junk tmp file was overwritten and renamed away by the rerun.
    assert not list(work.glob("*.tmp"))
    same_work(work, synthetic.work)
    shard_into(synthetic, tmp_path / "from-resumed", work=work)
    shard_into(synthetic, tmp_path / "from-clean")
    assert read_all_bytes(tmp_path / "from-resumed") == read_all_bytes(tmp_path / "from-clean")


def test_a_file_whose_outputs_changed_is_tokenized_again(synthetic, tmp_path, monkeypatch):
    work = tmp_path / "work"
    shutil.copytree(synthetic.work, work)
    (work / "tokens_002_00000.bin").unlink()
    (work / "done_000_00000.json").write_text("{not json", encoding="utf-8")
    tokenized = []
    real_file = fineweb.tokenize_file
    monkeypatch.setattr(fineweb, "tokenize_file", lambda raw, w, name, enc, threads:
                        tokenized.append(name) or real_file(raw, w, name, enc, threads))
    assert run_tokenize(synthetic, work) == 0
    assert tokenized == [NAMES[0], NAMES[2]]
    same_work(work, synthetic.work)


def test_two_full_runs_give_identical_shards_and_manifests(synthetic, tmp_path):
    work_a, work_b = tmp_path / "work-a", tmp_path / "work-b"
    assert run_tokenize(synthetic, work_a) == 0
    assert run_tokenize(synthetic, work_b) == 0
    manifest_a = shard_into(synthetic, tmp_path / "a", work=work_a)
    manifest_b = shard_into(synthetic, tmp_path / "b", work=work_b)
    assert read_all_bytes(tmp_path / "a") == read_all_bytes(tmp_path / "b")
    assert without_timestamp(manifest_a) == without_timestamp(manifest_b)
    assert (tmp_path / "a" / "manifest.json").read_bytes().count(b"created_utc") == 1
    assert "created_utc" in manifest_a and manifest_a["created_utc"].endswith("Z")


# ------------------------------------------------------------------- manifest

def test_manifest_contents(synthetic, tmp_path):
    manifest = shard_into(synthetic, tmp_path / "out")
    assert manifest["dataset"]["repo"] == "HuggingFaceFW/fineweb-edu"
    assert manifest["dataset"]["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert [(f["name"], f["size"], f["sha256"]) for f in manifest["dataset"]["files"]] == [
        (n, *synthetic.table[n]) for n in NAMES]
    tokenizer = manifest["tokenizer"]
    assert (tokenizer["encoding"], tokenizer["n_vocab"], tokenizer["eot"]) == ("gpt2", 50257, 50256)
    assert tokenizer["version"] == fineweb.importlib.metadata.version("tiktoken")
    assert manifest["format"]["header_bytes"] == 64 and manifest["format"]["version"] == 1
    assert manifest["shard_tokens"] == SHARD_TOKENS and manifest["val_b"]["val_b_rate"] == VAL_B_RATE
    assert "sha256" in manifest["val_b"]["rule"] and "divisible" in manifest["val_b"]["rule"]
    assert set(manifest["git"]) == {"sha", "dirty"}
    mismatches = manifest["token_count_mismatches"]
    assert mismatches["documents"] == 3
    assert mismatches["per_file"][NAMES[2]]["first_ids"] == ["<urn:uuid:02-0002>", "<urn:uuid:02-0011>"]
    raw = (tmp_path / "out" / "manifest.json").read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    # The real defaults: 10,506,240 tokens of val-A, bounded by a small shard.
    assert fineweb.VAL_A_TOKENS == 10_506_240 and fineweb.DEFAULT_SHARD_TOKENS == 100_000_000
    assert fineweb.DEFAULT_VAL_B_RATE == 1000


def test_nothing_printed_or_written_names_a_path_host_or_user(synthetic, tmp_path, capsys, monkeypatch):
    raw_bytes = {n: (synthetic.raw / n).read_bytes() for n in NAMES[:2]}

    def fake_downloader(**kwargs):
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw_bytes[target.name])
        return str(target)

    work, out, raw = tmp_path / "work", tmp_path / "out", tmp_path / "raw"
    capsys.readouterr()
    fineweb.main(["download", "--raw", str(raw), "--files", "0-1"], downloader=fake_downloader)
    run("tokenize", "--raw", raw, "--work", work, "--files", "0-1")
    run("shard", "--work", work, "--out", out, "--shard-tokens", 200, "--files", "0-1")
    run("verify", "--out", out)
    run("check", "--raw", raw, "--work", work, "--file", NAMES[0], "--docs", 5)
    run("shard", "--work", tmp_path / "missing", "--out", out)  # an error message
    corrupt(lambda b: b.__setitem__(70, b[70] ^ 1))(out / "shard_001.bin")
    run("verify", "--out", out)  # a failure message
    captured = capsys.readouterr()
    text = captured.out + captured.err + (out / "manifest.json").read_text(encoding="utf-8")
    for done in work.glob("done_*.json"):
        text += done.read_text(encoding="utf-8")
    assert "verify failed" in captured.err and "check ok" in captured.out
    forbidden = {str(tmp_path), str(flint_lib.ROOT), str(Path.home()), Path.home().name, socket.gethostname(),
                 getpass.getuser(), tmp_path.name}
    for item in forbidden:
        if len(item) >= 4:
            assert item.lower() not in text.lower(), item
            assert item.replace("\\", "/").lower() not in text.lower(), item


# --------------------------------------------------------------------- verify

@pytest.fixture
def good_output(synthetic, tmp_path):
    out = tmp_path / "out"
    shard_into(synthetic, out)
    return out


def test_verify_passes_on_good_output(good_output, capsys):
    assert run("verify", "--out", good_output) == 0
    out = capsys.readouterr().out
    assert "verify ok" in out and "elapsed" in out


def test_verify_fails_on_a_flipped_byte(good_output, capsys):
    corrupt(lambda b: b.__setitem__(100, b[100] ^ 4))(good_output / "shard_002.bin")
    assert run("verify", "--out", good_output) == 1
    assert "shard_002.bin: sha256" in capsys.readouterr().err


def test_verify_fails_on_a_missing_file(good_output, capsys):
    (good_output / "shard_003.bin").unlink()
    assert run("verify", "--out", good_output) == 1
    assert "shard_003.bin: missing" in capsys.readouterr().err
    (good_output / "manifest.json").unlink()
    assert run("verify", "--out", good_output) == 1


def test_verify_fails_on_a_wrong_count(good_output, capsys):
    path = good_output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"][1]["tokens"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    assert run("verify", "--out", good_output) == 1
    err = capsys.readouterr().err
    assert "header says" in err and "totals.train" in err
    manifest["files"][1]["tokens"] -= 1
    manifest["dropped_straddle_tokens"] += 1
    manifest["totals"]["dropped"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    assert run("verify", "--out", good_output) == 1
    assert "not the" in capsys.readouterr().err


def test_verify_fails_on_a_token_above_eot_even_with_a_matching_checksum(good_output, capsys):
    path = good_output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tokens = tokens_of(good_output / "shard_002.bin")
    tokens[7] = 50257
    manifest["files"][2]["sha256"] = fineweb.write_token_file(good_output / "shard_002.bin", tokens)
    path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    assert run("verify", "--out", good_output) == 1
    assert "above 50256" in capsys.readouterr().err


def test_verify_fails_on_a_bad_header_extra_files_and_a_short_shard(good_output, capsys):
    BAD_FILES["version"](good_output / "shard_001.bin")
    (good_output / "shard_077.bin").write_bytes(b"stray")
    assert run("verify", "--out", good_output) == 1
    err = capsys.readouterr().err
    assert "shard_001.bin: token file version" in err and "shard_077.bin: not in the manifest" in err


def test_verify_checks_that_a_short_shard_is_only_the_last(synthetic, tmp_path, capsys):
    out = tmp_path / "out"
    shard_into(synthetic, out)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"][1]["tokens"] -= 1
    path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    assert run("verify", "--out", out) == 1
    assert "shard_001.bin: 399 tokens, not 400" in capsys.readouterr().err


# ------------------------------------------------------------------- download

def test_download_skips_verified_files_and_fetches_the_rest(tmp_path, monkeypatch, capsys):
    contents = {NAMES[0]: b"a" * 1000, NAMES[1]: b"b" * 777}
    monkeypatch.setattr(fineweb, "FILES", {n: (len(c), hashlib.sha256(c).hexdigest()) for n, c in contents.items()})
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents[target.name])
        return str(target)

    raw = tmp_path / "raw"
    assert fineweb.main(["download", "--raw", str(raw)], downloader=downloader) == 0
    assert [(c["repo_id"], c["repo_type"], c["revision"], c["filename"]) for c in calls] == [
        ("HuggingFaceFW/fineweb-edu", "dataset", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", f"sample/10BT/{n}")
        for n in NAMES[:2]]
    assert all(Path(c["local_dir"]) == raw for c in calls)
    assert (raw / "sample" / "10BT" / NAMES[0]).read_bytes() == contents[NAMES[0]]
    # Everything is verified now, so nothing more is fetched.
    calls.clear()
    capsys.readouterr()
    assert fineweb.main(["download", "--raw", str(raw)], downloader=downloader) == 0
    assert calls == [] and capsys.readouterr().out.count("already downloaded and verified") == 2
    # A same-size corrupt file is replaced; the intact one is left alone.
    corrupt(lambda b: b.__setitem__(5, b[5] ^ 1))(raw / "sample" / "10BT" / NAMES[1])
    assert fineweb.main(["download", "--raw", str(raw)], downloader=downloader) == 0
    assert [c["filename"] for c in calls] == [f"sample/10BT/{NAMES[1]}"]
    assert (raw / "sample" / "10BT" / NAMES[1]).read_bytes() == contents[NAMES[1]]
    # --files picks by index.
    calls.clear()
    shutil.rmtree(raw)
    assert fineweb.main(["download", "--raw", str(raw), "--files", "1"], downloader=downloader) == 0
    assert [c["filename"] for c in calls] == [f"sample/10BT/{NAMES[1]}"]


def test_download_fails_loudly_on_a_sha256_mismatch(tmp_path, monkeypatch, capsys):
    good = b"c" * 500
    monkeypatch.setattr(fineweb, "FILES", {NAMES[0]: (len(good), hashlib.sha256(good).hexdigest())})

    def wrong_bytes(**kwargs):
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"d" * 500)  # the right size, the wrong content
        return str(target)

    raw = tmp_path / "raw"
    assert fineweb.main(["download", "--raw", str(raw)], downloader=wrong_bytes) == 1
    err = capsys.readouterr().err
    assert NAMES[0] in err and "doesn't match" in err
    assert not (raw / "sample" / "10BT" / NAMES[0]).exists()  # a bad file isn't left to be trusted


def test_download_default_is_the_hub_function():
    assert fineweb.main.__defaults__[1] is fineweb.hf_download


# ---------------------------------------------------------------------- check

def test_check_passes_on_good_data_and_for_more_documents_than_exist(synthetic, capsys):
    assert run("check", "--raw", synthetic.raw, "--work", synthetic.work, "--file", NAMES[1], "--docs", 12) == 0
    assert "check ok: 12 documents" in capsys.readouterr().out
    assert run("check", "--raw", synthetic.raw, "--work", synthetic.work, "--file", NAMES[1], "--docs", 1000) == 0
    assert "check ok: 17 documents" in capsys.readouterr().out


def test_check_fails_on_a_corrupted_token_file(synthetic, tmp_path, capsys):
    work = tmp_path / "work"
    shutil.copytree(synthetic.work, work)
    path = work / "tokens_000_00000.bin"
    first_docs = sum(len(t) for _, t in synthetic.stream[:3])
    corrupt(lambda b: b.__setitem__(2 * (first_docs + 1), b[2 * (first_docs + 1)] ^ 1))(path)
    assert run("check", "--raw", synthetic.raw, "--work", work, "--file", NAMES[0], "--docs", 10) == 1
    assert "document 3" in capsys.readouterr().err
    # A corruption beyond the compared documents isn't looked at.
    assert run("check", "--raw", synthetic.raw, "--work", work, "--file", NAMES[0], "--docs", 3) == 0
    # A wrong length in the index is caught too.
    shutil.copytree(synthetic.work, tmp_path / "work2")
    with np.load(tmp_path / "work2" / "docs_000_00000.npz") as index:
        arrays = {k: index[k] for k in index.files}
    arrays["lengths"][0] += 1
    np.savez(tmp_path / "work2" / "docs_000_00000.npz", **arrays)
    assert run("check", "--raw", synthetic.raw, "--work", tmp_path / "work2", "--file", NAMES[0], "--docs", 2) == 1
    assert "document 0" in capsys.readouterr().err


def test_check_reports_missing_inputs_and_bad_arguments(synthetic, tmp_path, capsys):
    assert run("check", "--raw", synthetic.raw, "--work", tmp_path / "none", "--file", NAMES[0], "--docs", 3) == 1
    assert "hasn't been tokenized" in capsys.readouterr().err
    assert run("check", "--raw", synthetic.raw, "--work", synthetic.work, "--file", "nope.parquet", "--docs", 3) == 1
    assert run("check", "--raw", synthetic.raw, "--work", synthetic.work, "--file", NAMES[0], "--docs", 0) == 1


# -------------------------------------------------------------- small helpers

def test_file_specs():
    names = [f"{i:03d}_00000.parquet" for i in range(14)]
    parse = fineweb.parse_file_spec
    assert parse(None, names) == names
    assert parse("0", names) == names[:1]
    assert parse("0-3", names) == names[:4]
    assert parse("0,5,13", names) == [names[0], names[5], names[13]]
    assert parse("5,0-1,1", names) == [names[0], names[1], names[5]]
    for bad in ("14", "a", "3-1", "", "1-", "-1", "0,,2"):
        with pytest.raises(fineweb.FineWebError):
            parse(bad, names)


def test_the_real_table_is_the_pinned_one():
    # Not the synthetic patch: read the constants in the source.
    spec2 = importlib.util.spec_from_file_location("fineweb_pristine", SCRIPTS / "fineweb.py")
    pristine = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(pristine)
    assert sorted(pristine.FILES) == [f"{i:03d}_00000.parquet" for i in range(14)]
    assert sum(size for size, _ in pristine.FILES.values()) == 28_518_193_415
    assert all(len(sha) == 64 and int(sha, 16) >= 0 for _, sha in pristine.FILES.values())
    assert pristine.FILES["013_00000.parquet"][0] == 540_632_672
    assert pristine.REVISION == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9" and pristine.SUBFOLDER == "sample/10BT"


def test_tiktoken_cache_defaults_to_the_home_cache_unless_set(monkeypatch):
    expected = str(Path.home() / ".cache" / "tiktoken")
    environ = {k: v for k, v in os.environ.items() if k != "TIKTOKEN_CACHE_DIR"}
    monkeypatch.setattr(os, "environ", environ)
    fineweb.configure_tiktoken_cache()
    assert environ["TIKTOKEN_CACHE_DIR"] == expected
    environ["TIKTOKEN_CACHE_DIR"] = "elsewhere"
    fineweb.configure_tiktoken_cache()
    assert environ["TIKTOKEN_CACHE_DIR"] == "elsewhere"


def test_the_encoder_is_gpt2(enc):
    assert enc.n_vocab == 50257 and enc.eot_token == 50256


def test_new_files_have_unix_line_endings():
    for path in (SCRIPTS / "fineweb.py", Path(__file__)):
        assert b"\r" not in path.read_bytes(), path.name
