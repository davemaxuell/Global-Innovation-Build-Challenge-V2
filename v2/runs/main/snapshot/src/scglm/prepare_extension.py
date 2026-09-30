"""Prepare a pinned continuation corpus using the existing tokenizer.

LMDB transactions checkpoint both deduplication and durable output positions.
Restart with the same config to resume. Only complete manifests allow training.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import shutil
import time
from urllib.parse import unquote, urlsplit, urlunsplit

import lmdb
import numpy as np
import pyarrow.parquet as pq
import xxhash
import zstandard
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .prepare_data import ExclusionIndex, fingerprint, normalized, sha256_file, write_json, log

_TOK = _EXCLUDE = None
_RNG = np.random.RandomState(424242)
_A = _RNG.randint(1, 2**32, size=32, dtype=np.uint64) | np.uint64(1)
_B = _RNG.randint(0, 2**32, size=32, dtype=np.uint64)


class ExclusionUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "ExclusionIndex" and module == "__main__":
            return ExclusionIndex
        return super().find_class(module, name)


def canonical_url(url):
    if not url:
        return ""
    try:
        u = urlsplit(str(url))
        if not u.netloc:
            return ""
        return urlunsplit(("https", u.netloc.casefold().removeprefix("www."), unquote(u.path).rstrip("/"), u.query, ""))
    except ValueError:
        return ""


def sketch(text):
    """32-component approximate MinHash over normalized five-word shingles.

    Only LSH candidates with >=28 matching components are rejected. This is a
    conservative approximate near-document filter, not semantic decontamination.
    """
    words = normalized(text).split()
    if len(words) < 5:
        return np.zeros(32, dtype="<u4").tobytes()
    hashes = np.fromiter((xxhash.xxh32_intdigest(" ".join(words[i:i+5])) for i in range(len(words)-4)), dtype=np.uint64)
    result = np.full(32, 2**32-1, dtype=np.uint64)
    for start in range(0, len(hashes), 2048):
        values = ((_A[:, None] * hashes[None, start:start+2048] + _B[:, None]) % np.uint64(2**61-1)) & np.uint64(2**32-1)
        result = np.minimum(result, values.min(axis=1))
    return result.astype("<u4").tobytes()


def worker_init(tokenizer, exclusions):
    global _TOK, _EXCLUDE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOK = Tokenizer.from_file(tokenizer)
    with open(exclusions, "rb") as f:
        _EXCLUDE = ExclusionUnpickler(f).load()


def analyze(task):
    rows, old = task
    result = []
    for row in rows:
        text = row.get("text", "")
        if not isinstance(text, str) or not 240 <= len(text) <= 200000:
            result.append({"reject": "length", "position": row["_position"]})
            continue
        digest = fingerprint(text)
        item = {"digest": digest, "sketch": sketch(text), "url": canonical_url(row.get("url")),
                "position": row["_position"], "id": row.get("id", row.get("url", digest)),
                "title": row.get("title", ""), "text": text}
        if not old and _EXCLUDE.matches(text, str(row.get("title", ""))):
            item["reject"] = "benchmark_exact"
        result.append(item)
    if not old:
        good = [r for r in result if "reject" not in r]
        for row, encoded in zip(good, _TOK.encode_batch([r["text"] for r in good], add_special_tokens=False)):
            row["tokens"] = encoded.ids + [2]
    return result


def band_keys(signature):
    return [b"B" + bytes([i]) + xxhash.xxh64_digest(signature[i*16:(i+1)*16]) for i in range(8)]


def matching_reason(txn, row):
    digest = bytes.fromhex(row["digest"])
    if txn.get(b"H"+digest):
        return "exact_document"
    if row["url"] and txn.get(b"U"+hashlib.sha256(row["url"].encode()).digest()):
        return "same_url"
    seen = set()
    signature = np.frombuffer(row["sketch"], dtype="<u4")
    for key in band_keys(row["sketch"]):
        candidates = txn.get(key) or b""
        for j in range(0, len(candidates), 32):
            other = candidates[j:j+32]
            if other in seen:
                continue
            seen.add(other)
            value = txn.get(b"S"+other)
            if value and np.count_nonzero(signature == np.frombuffer(value, dtype="<u4")) >= 28:
                return "approximate_near_document"
    return None


def remember(txn, row, full=True):
    digest = bytes.fromhex(row["digest"])
    txn.put(b"H"+digest, b"1")
    if row["url"]:
        txn.put(b"U"+hashlib.sha256(row["url"].encode()).digest(), b"1")
    if full:
        txn.put(b"S"+digest, row["sketch"])
        for key in band_keys(row["sketch"]):
            existing = txn.get(key) or b""
            # At most four representatives per LSH bucket: bounded memory/work.
            if digest not in [existing[i:i+32] for i in range(0, len(existing), 32)] and len(existing) < 128:
                txn.put(key, existing+digest)


def read_progress(env):
    with env.begin() as txn:
        data = txn.get(b"!progress")
        return json.loads(data) if data else None


def batches(iterator, size=512):
    while True:
        rows = []
        for _ in range(size):
            try:
                rows.append(next(iterator))
            except StopIteration:
                break
        if not rows:
            break
        yield rows


def analyzed(pool, iterator, old=False):
    pending = deque()
    tasks = iter(batches(iterator))
    for _ in range(8):
        try:
            pending.append(pool.submit(analyze, (next(tasks), old)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(pool.submit(analyze, (next(tasks), old)))
        except StopIteration:
            pass


def old_rows(path, start):
    with path.open("rb") as f:
        f.seek(start)
        while line := f.readline():
            row = json.loads(line)
            row["_position"] = f.tell()
            yield row


def remote_rows(spec, filename, start):
    local = hf_hub_download(spec["repo"], filename, repo_type="dataset", revision=spec["revision"])
    if filename.endswith(".parquet"):
        p = pq.ParquetFile(local)
        count = 0
        for batch in p.iter_batches(batch_size=1024, use_threads=False):
            for row in batch.to_pylist():
                count += 1
                if count <= start:
                    continue
                row["_position"] = count
                yield row
    else:
        with open(local, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader, io.TextIOWrapper(reader, encoding="utf-8") as f:
            for count, line in enumerate(f, 1):
                if count <= start:
                    continue
                row = json.loads(line)
                row["_position"] = count
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_extension.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    contract = json.dumps(config, sort_keys=True)
    cp = out / "preparation_config.json"
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError("Preparation contract differs; choose a new output directory")
    write_json(cp, config)
    if (out / "manifest.json").exists():
        log("already_complete", output=str(out)); return
    prior = Path(config["parent_manifest"]).resolve()
    parent = json.loads(prior.read_text())
    tokenizer = Path(parent["tokenizer"]["path"])
    if sha256_file(tokenizer) != parent["tokenizer"]["sha256"]:
        raise ValueError("Parent tokenizer hash mismatch")
    code_hash = sha256_file(Path(__file__))
    env = lmdb.open(str(out / "dedup.lmdb"), map_size=160*1024**3, subdir=True, max_dbs=1, sync=True, metasync=True)
    with env.begin(write=True) as txn:
        identity = (contract + code_hash + sha256_file(prior)).encode()
        digest = hashlib.sha256(identity).hexdigest().encode()
        if txn.get(b"!contract") not in (None, digest):
            raise ValueError("Preparation code/parent changed; restore original implementation or use new output")
        txn.put(b"!contract", digest)
    exclusions = str(prior.parent / "exclusion_index.pkl")
    state = read_progress(env) or {"old_done": False, "old_file": 0, "old_offset": 0, "old_documents": 0, "sources": {}}
    begin = time.monotonic()
    with ProcessPoolExecutor(max_workers=config["workers"], mp_context=multiprocessing.get_context("spawn"), initializer=worker_init, initargs=(str(tokenizer), exclusions)) as pool:
        old_files = sorted((prior.parent / "raw").glob("*.jsonl"))
        if not state["old_done"]:
            for fi in range(state["old_file"], len(old_files)):
                log("seed_prior_file", path=str(old_files[fi]), offset=state["old_offset"])
                for rows in analyzed(pool, old_rows(old_files[fi], state["old_offset"]), old=True):
                    with env.begin(write=True) as txn:
                        for row in rows:
                            if "reject" not in row:
                                remember(txn, row)
                        state["old_file"], state["old_offset"] = fi, rows[-1]["position"]
                        state["old_documents"] += len(rows)
                        txn.put(b"!progress", json.dumps(state).encode())
                    if state["old_documents"] % 10240 < 512:
                        log("seed_prior_progress", documents=state["old_documents"], elapsed_s=round(time.monotonic()-begin,1))
                state["old_file"], state["old_offset"] = fi+1, 0
                with env.begin(write=True) as txn:
                    txn.put(b"!progress", json.dumps(state).encode())
            state["old_done"] = True
            with env.begin(write=True) as txn:
                txn.put(b"!progress", json.dumps(state).encode())
        # Wikipedia first so newly encountered web mirrors cannot displace it.
        for source, spec in config["sources"].items():
            directory = out / source
            directory.mkdir(exist_ok=True)
            meta = json.loads(Path(spec["metadata_file"]).read_text())
            if meta["revision"] != spec["revision"]:
                raise ValueError("Pinned file metadata revision differs")
            files = sorted(f["path"] for f in meta["files"] if f["path"].startswith(spec["file_prefix"]) and f["path"].endswith((".parquet", ".jsonl.zst")))
            random.Random(config["seed"]).shuffle(files)
            s = state["sources"].setdefault(source, {"file":0, "row":0, "train_tokens":0, "train_documents":0, "counts":{}, "panels":{}, "output_bytes":{}, "complete":False})
            if s["complete"]:
                continue
            names = ["train.bin", "train.idx.jsonl", "train.raw.jsonl", *[f"{p}.jsonl" for p in config["panels"]]]
            handles = {}
            for name in names:
                path = directory / name
                f = path.open("r+b" if path.exists() else "w+b")
                f.truncate(s["output_bytes"].get(name, 0)); f.seek(0, 2)
                handles[name] = f
            try:
                for fi in range(s["file"], len(files)):
                    filename = files[fi]
                    log("extension_source_file", source=source, file=filename, row=s["row"], train_tokens=s["train_tokens"])
                    for rows in analyzed(pool, remote_rows(spec, filename, s["row"])):
                        end_position = rows[-1]["position"]
                        random.Random(f"{config['seed']}:{source}:{fi}:{end_position}").shuffle(rows)
                        counts = Counter(s["counts"])
                        with env.begin(write=True) as txn:
                            for row in rows:
                                counts["rows_seen"] += 1
                                if "reject" in row:
                                    counts[row["reject"]] += 1; continue
                                if source != "wiki" and ".wikipedia.org/" in row["url"]:
                                    counts["prefer_wikipedia_source"] += 1; continue
                                reason = matching_reason(txn, row)
                                if reason:
                                    counts[reason] += 1
                                    remember(txn, row, full=False)
                                    continue
                                # First representative fixes cluster ownership; near duplicates are rejected.
                                remember(txn, row)
                                bucket = int(row["digest"][:16], 16) % 1000
                                split = list(config["panels"])[bucket] if bucket < len(config["panels"]) else "train"
                                ids = row["tokens"]
                                record = {"id":f"{source}:{row['digest']}", "source":source, "source_id":str(row["id"]), "content_sha256":row["digest"], "url":row["url"], "title":row["title"], "source_file":filename, "source_row":row["position"]}
                                if split == "train":
                                    if s["train_tokens"] >= spec["target_train_tokens"]:
                                        counts["surplus_train"] += 1; continue
                                    # Whole documents, including EOS. Overshoot <= one retained document.
                                    handles["train.bin"].write(np.asarray(ids,dtype="<u2").tobytes())
                                    handles["train.idx.jsonl"].write((json.dumps({**record,"offset":s["train_tokens"],"length":len(ids)})+"\n").encode())
                                    handles["train.raw.jsonl"].write((json.dumps({**record,"text":row["text"]},ensure_ascii=False)+"\n").encode())
                                    s["train_tokens"] += len(ids); s["train_documents"] += 1
                                elif s["panels"].get(split,0) < config["panels"][split]:
                                    if len(ids)>4096:
                                        ids=ids[:4095]+[2]
                                    handles[f"{split}.jsonl"].write((json.dumps({**record,"tokens":ids})+"\n").encode())
                                    s["panels"][split] = s["panels"].get(split,0)+1
                                else:
                                    counts["surplus_reserved_panel"] += 1
                            # Output fsync precedes transaction commit; restart truncates ahead-of-commit bytes.
                            for name, f in handles.items():
                                f.flush(); os.fsync(f.fileno()); s["output_bytes"][name] = f.tell()
                            s["file"], s["row"], s["counts"] = fi, end_position, dict(counts)
                            s["complete"] = s["train_tokens"] >= spec["target_train_tokens"] and all(s["panels"].get(k,0)>=v for k,v in config["panels"].items())
                            txn.put(b"!progress", json.dumps(state).encode())
                        if counts["rows_seen"] % 10240 < 512 or s["complete"]:
                            log("extension_collect", source=source, tokens=s["train_tokens"], target=spec["target_train_tokens"], documents=s["train_documents"], counts=s["counts"], panels=s["panels"], elapsed_s=round(time.monotonic()-begin,1))
                            write_json(out/"status.json", {"state":"preparing", "sources":state["sources"], "updated_unix":time.time()})
                        if s["complete"]:
                            break
                    if s["complete"]:
                        break
                    s["file"], s["row"] = fi+1, 0
                    with env.begin(write=True) as txn:
                        txn.put(b"!progress",json.dumps(state).encode())
                if not s["complete"]:
                    raise RuntimeError(f"Insufficient source pool {source}: {s}")
            finally:
                for f in handles.values(): f.close()
    env.sync(); env.close()
    tok_dir = out/"tokenizer"
    shutil.copytree(tokenizer.parent,tok_dir,dirs_exist_ok=True)
    manifest = {"schema_version":2, "status":"complete", "parent_manifest_sha256":sha256_file(prior), "preparation_code_sha256":code_hash,
        "tokenizer":{**parent["tokenizer"],"path":str(tok_dir/"tokenizer.json"),"directory":str(tok_dir)}, "sources":{},
        "deduplication":{"exact":"normalized content SHA256 and canonical URL, including all prior raw documents", "near":"5-word shingles, 32-component MinHash; 8 bands x4; reject >=28/32 matching components; up to4 representatives/band; approximate with false negatives possible"},
        "configuration_sha256":sha256_file(cp), "exclusion_report":parent["exclusion_report"], "old_documents_indexed":state["old_documents"]}
    for source,spec in config["sources"].items():
        directory=out/source; s=state["sources"][source]
        train=directory/"train.bin"
        meta={"path":str(train),"tokens":s["train_tokens"],"documents":s["train_documents"],"dtype":"uint16_little_endian","sha256":sha256_file(train),"index_path":str(directory/"train.idx.jsonl"),"index_sha256":sha256_file(directory/"train.idx.jsonl")}
        splits={"train":meta}
        for panel in config["panels"]:
            file=directory/f"{panel}.jsonl"
            splits[panel]={"jsonl_path":str(file),"jsonl_sha256":sha256_file(file),"documents":s["panels"][panel]}
        manifest["sources"][source]={"provenance":spec,"splits":splits,"collection":s}
    write_json(out/"manifest.json",manifest)
    write_json(out/"status.json",{"state":"complete","sources":state["sources"],"updated_unix":time.time()})
    log("extension_preparation_complete",manifest=str(out/"manifest.json"))


if __name__ == "__main__":
    main()
