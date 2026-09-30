"""Build the V2 rich corpus: filter, deduplicate, route, tokenize and write shards.

Stages (each resumable; run from the project root with PYTHONPATH=src:v2/src:v2/.pydeps):
  reused  - V1 extension/main raw text (already deduplicated and benchmark-excluded):
            domain blocklist, targeted routing for edu/dclm, tokenize. Parallel byte ranges.
  new     - downloaded StackExchange, Gutenberg and DCLM files: length check, 13-word
            benchmark exclusion, domain blocklist, exact/URL/MinHash dedup against a copy
            of V1's dedup database, targeted routing for DCLM, tokenize. One LMDB write
            transaction per input file; its outputs are fsynced before the commit.
  manifest- trim held-out panels, hash every shard, write manifest and diversity report.

Output layout: <out>/<bucket>/parts/<part>.bin (uint16 LE, EOS-separated),
<part>.idx.jsonl, <part>.<panel>.jsonl (held-out, token lists), <part>.done.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .common import ROOT, event, load_config, sha256_file, write_json
from .sources import blocked, host, iter_books, iter_dclm_new, iter_stackexchange, reused_raw_paths, snapshot_files

PANELS = ("monitor", "development", "selection", "final")
BUCKETS = ("edu", "dclm", "targeted", "wiki", "stackexchange", "books")
RANGE_BYTES = 768 * 1024**2

_TOK = _SCORER = _CFG = None


def _init(tokenizer_path: str, classifier_path: str, config: dict, exclusions: str | None = None):
    global _TOK, _SCORER, _CFG
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from tokenizers import Tokenizer
    from .targeted import Scorer
    _TOK = Tokenizer.from_file(tokenizer_path)
    _SCORER = Scorer(classifier_path)
    _CFG = config
    if exclusions:
        from scglm import prepare_extension
        prepare_extension.worker_init(tokenizer_path, exclusions)


def panel_of(digest: str) -> str | None:
    bucket = int(digest[:16], 16) % 1000
    return PANELS[bucket] if bucket < len(PANELS) else None


def route(source: str, text: str) -> tuple[str, float | None]:
    if source in _CFG["targeted"]["pool_sources"]:
        score = _SCORER.score(text)
        return ("targeted" if score >= _CFG["_threshold"] else source), score
    return source, None


class PartWriter:
    """Per-bucket output files for one input part; ``close`` fsyncs everything."""

    def __init__(self, out: Path, part: str):
        self.out, self.part, self.handles, self.stats = out, part, {}, {}

    def _h(self, bucket: str, suffix: str):
        key = (bucket, suffix)
        if key not in self.handles:
            d = self.out / bucket / "parts"
            d.mkdir(parents=True, exist_ok=True)
            self.handles[key] = (d / f"{self.part}{suffix}").open("wb")
        return self.handles[key]

    def add(self, bucket: str, record: dict, ids: list[int], panel: str | None):
        s = self.stats.setdefault(bucket, {"tokens": 0, "documents": 0, "chars": 0, "panels": Counter(), "hosts": Counter()})
        if panel:
            self._h(bucket, f".{panel}.jsonl").write((json.dumps({**record, "tokens": ids}) + "\n").encode())
            s["panels"][panel] += 1
            return
        self._h(bucket, ".bin").write(np.asarray(ids, dtype="<u2").tobytes())
        self._h(bucket, ".idx.jsonl").write((json.dumps({**record, "offset": s["tokens"], "length": len(ids)}) + "\n").encode())
        s["tokens"] += len(ids); s["documents"] += 1; s["chars"] += record.get("chars", 0)
        s["hosts"][record.get("host") or record.get("origin") or ""] += 1

    def close(self, extra: dict | None = None):
        for f in self.handles.values():
            f.flush(); os.fsync(f.fileno()); f.close()
        stats = {b: {**s, "panels": dict(s["panels"]), "hosts": dict(s["hosts"].most_common(200)),
                     "distinct_hosts": len(s["hosts"])} for b, s in self.stats.items()}
        for b in stats:
            write_json(self.out / b / "parts" / f"{self.part}.done.json", {"part": self.part, **stats[b], **(extra or {})})
        return stats


# ------------------------------------------------------------------ reused stage
def _reused_range(task):
    source, path, start, end, part, out = task
    marker = Path(out) / "_reused_done" / f"{part}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    writer, counts, batch = PartWriter(Path(out), part), Counter(), []
    blocklist = _CFG["domain_blocklist"]

    def flush():
        encoded = _TOK.encode_batch([b[1] for b in batch], add_special_tokens=False)
        for (record, _text, bucket, panel), enc in zip(batch, encoded):
            writer.add(bucket, record, enc.ids + [2], panel)
        batch.clear()

    with open(path, "rb") as f:
        if start:
            # Skip through the newline at or after start-1: a line that begins exactly at
            # ``start`` is kept here; one straddling the boundary belongs to the previous range.
            f.seek(start - 1)
            f.readline()
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            row = json.loads(line)
            counts["rows"] += 1
            text = row["text"]
            if blocked(row.get("url"), blocklist):
                counts["domain_blocklist"] += 1
                continue
            digest = row.get("content_sha256") or hashlib.sha256(text.encode()).hexdigest()
            bucket, score = route(source, text)
            record = {"id": row.get("id"), "source": source, "url": row.get("url"), "title": row.get("title"),
                      "content_sha256": digest, "origin": Path(path).name, "host": host(row.get("url")),
                      "chars": len(text), "targeted_score": score}
            batch.append((record, text, bucket, panel_of(digest)))
            if len(batch) >= 256:
                flush()
    if batch:
        flush()
    stats = writer.close({"input": str(path), "byte_range": [start, end]})
    result = {"part": part, "source": source, "counts": dict(counts),
              "buckets": {b: {"tokens": s["tokens"], "documents": s["documents"]} for b, s in stats.items()}}
    write_json(marker, result)
    return result


def run_reused(config: dict, out: Path, tokenizer: str, classifier: str, workers: int, log: Path):
    tasks = []
    for source in ("edu", "dclm", "wiki"):
        for k, path in enumerate(reused_raw_paths(config, source)):
            size = path.stat().st_size
            for i, start in enumerate(range(0, size, RANGE_BYTES)):
                tasks.append((source, str(path), start, min(size, start + RANGE_BYTES) - 1,
                              f"reused_{source}{k}_{i:04d}", str(out)))
    event(log, "build_reused_start", ranges=len(tasks), workers=workers)
    totals, done = Counter(), 0
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init, initargs=(tokenizer, classifier, config)) as pool:
        for fut in as_completed([pool.submit(_reused_range, t) for t in tasks]):
            r = fut.result(); done += 1
            for b, s in r["buckets"].items():
                totals[b] += s["tokens"]
            if done % 10 == 0 or done == len(tasks):
                event(log, "build_reused_progress", ranges_done=done, ranges=len(tasks), tokens=dict(totals))
    write_json(out / "_reused_done" / "complete.json", {"ranges": len(tasks), "tokens": dict(totals)})


# ------------------------------------------------------------------ new stage
def _analyze_new(rows):
    """Worker: V1 length/exclusion/MinHash analysis, then blocklist, routing and tokens."""
    from scglm import prepare_extension as pe
    analyzed = pe.analyze((rows, True))  # old=True: skip V1-tokenizer encoding; checks done below
    out = []
    for row, item in zip(rows, analyzed):
        if "reject" in item:
            out.append(item); continue
        if blocked(row.get("url"), _CFG["domain_blocklist"]):
            item["reject"] = "domain_blocklist"; out.append(item); continue
        if pe._EXCLUDE.matches(item["text"], str(row.get("title", ""))):
            item["reject"] = "benchmark_exact"; out.append(item); continue
        bucket, score = route(row["source"], item["text"])
        item.update(bucket=bucket, score=score, source=row["source"], origin=row.get("origin"), chars=len(item["text"]),
                    tokens=_TOK.encode(item["text"], add_special_tokens=False).ids + [2])
        del item["text"]
        out.append(item)
    return out


def _new_inputs(config: dict, source: str) -> list[tuple[str, Path]]:
    plan = json.loads((ROOT / config["output_dir"] / "downloads" / "plan.json").read_text())
    spec = config["sources"][source]
    base = Path(os.environ.get("HF_HUB_CACHE", "/data/shared/hf_cache/hub")) / \
        f"datasets--{spec['repo'].replace('/', '--')}" / "snapshots" / spec["revision"]
    return [(name, base / name) for name in plan[source]]


def _iter_file(config: dict, source: str, path: Path):
    if source == "stackexchange":
        yield from iter_stackexchange(config, [path])
    elif source == "books":
        yield from iter_books(config, [path])
    else:
        yield from iter_dclm_new(config, [path])


def run_new(config: dict, out: Path, tokenizer: str, classifier: str, workers: int, log: Path, sources: list[str]):
    import lmdb
    from scglm import prepare_extension as pe
    db_path = out / "dedup.lmdb"
    if not db_path.exists():
        src = ROOT / config["reuse"]["extension_dedup_lmdb"]
        event(log, "dedup_copy_start", source=str(src))
        shutil.copytree(src, str(db_path) + ".tmp")
        os.replace(str(db_path) + ".tmp", db_path)
        event(log, "dedup_copy_done")
    env = lmdb.open(str(db_path), map_size=200 * 1024**3, subdir=True, max_dbs=1, sync=True, metasync=True)
    exclusions = str(ROOT / config["reuse"]["exclusion_index"])
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init, initargs=(tokenizer, classifier, config, exclusions)) as pool:
        for source in sources:
            for name, path in _new_inputs(config, source):
                part = f"new_{source}_" + hashlib.sha256(name.encode()).hexdigest()[:12]
                marker = out / "_new_done" / f"{part}.json"
                if marker.exists():
                    continue
                waited = 0
                while not path.exists():  # downloader still running
                    if waited == 0:
                        event(log, "build_new_waiting", file=name)
                    time.sleep(30); waited += 30
                    if waited > 4 * 3600:
                        raise RuntimeError(f"download never arrived: {name}")
                writer, counts = PartWriter(out, part), Counter()
                rows_iter = _iter_file(config, source, path)
                with env.begin(write=True) as txn:
                    pending, position = [], 0

                    def batches():
                        nonlocal position
                        chunk = []
                        for doc in rows_iter:
                            position += 1
                            chunk.append({**doc, "_position": position})
                            if len(chunk) >= 256:
                                yield chunk; chunk = []
                        if chunk:
                            yield chunk
                    stream = batches()
                    for _ in range(workers * 2):
                        try:
                            pending.append(pool.submit(_analyze_new, next(stream)))
                        except StopIteration:
                            break
                    while pending:
                        items = pending.pop(0).result()
                        try:
                            pending.append(pool.submit(_analyze_new, next(stream)))
                        except StopIteration:
                            pass
                        for item in items:
                            counts["rows"] += 1
                            if "reject" in item:
                                counts[item["reject"]] += 1; continue
                            reason = pe.matching_reason(txn, item)
                            if reason:
                                counts[reason] += 1; pe.remember(txn, item, full=False); continue
                            pe.remember(txn, item)
                            record = {"id": str(item["id"]), "source": item["source"], "url": item["url"] or None,
                                      "title": item["title"], "content_sha256": item["digest"], "origin": item["origin"],
                                      "host": host(item["url"]) if item["url"] else item["origin"],
                                      "chars": item["chars"], "targeted_score": item["score"]}
                            writer.add(item["bucket"], record, item["tokens"], panel_of(item["digest"]))
                    stats = writer.close({"input": name})
                    result = {"part": part, "source": source, "file": name, "counts": dict(counts),
                              "buckets": {b: {"tokens": s["tokens"], "documents": s["documents"]} for b, s in stats.items()}}
                    write_json(marker, result)
                event(log, "build_new_file", source=source, file=name, counts=dict(counts), buckets=result["buckets"])
    env.sync(); env.close()
    write_json(out / "_new_done" / f"complete_{'_'.join(sources)}.json", {"sources": sources})


# ------------------------------------------------------------------ manifest
def run_manifest(config: dict, out: Path, tokenizer: str, log: Path):
    quotas = config["panels_per_source"]
    manifest = {"schema": "v2-rich-1", "status": "complete", "phase_id": config["phase_id"],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "tokenizer": {"path": str(out / "tokenizer" / "tokenizer.json"), "directory": str(out / "tokenizer"),
                              "sha256": sha256_file(out / "tokenizer" / "tokenizer.json"),
                              "special_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3}},
                "domain_blocklist": config["domain_blocklist"], "planned_training_shares": config["planned_training_shares"],
                "sources": {}, "configuration_sha256": sha256_file(ROOT / "v2/configs/prep.json")}
    from tokenizers import Tokenizer
    manifest["tokenizer"]["vocab_size"] = Tokenizer.from_file(manifest["tokenizer"]["path"]).get_vocab_size()
    diversity = {}
    for bucket in BUCKETS:
        parts_dir = out / bucket / "parts"
        dones = sorted(parts_dir.glob("*.done.json"))
        shards, tokens, docs, hosts, chars = [], 0, 0, Counter(), 0
        for d in dones:
            info = json.loads(d.read_text())
            b = parts_dir / f"{info['part']}.bin"
            if info["tokens"] and b.exists():
                shards.append({"path": str(b), "tokens": info["tokens"], "documents": info["documents"],
                               "sha256": sha256_file(b), "index_path": str(b.with_suffix(".idx.jsonl"))})
                tokens += info["tokens"]; docs += info["documents"]; chars += info.get("chars", 0)
                hosts.update(info["hosts"])
        panels = {}
        for panel in PANELS:
            rows = []
            for f in sorted(parts_dir.glob(f"*.{panel}.jsonl")):
                rows.extend(json.loads(l) for l in f.open())
            rows.sort(key=lambda r: r["content_sha256"])
            rows = rows[:quotas[panel]]
            dest = out / bucket / f"{panel}.jsonl"
            with dest.open("w") as h:
                for r in rows:
                    h.write(json.dumps(r) + "\n")
            panels[panel] = {"jsonl_path": str(dest), "jsonl_sha256": sha256_file(dest), "documents": len(rows)}
        manifest["sources"][bucket] = {"train": {"shards": shards, "tokens": tokens, "documents": docs,
                                                  "dtype": "uint16_little_endian"}, "panels": panels,
                                       # V1-trainer-compatible view of the same files.
                                       "splits": {"train": {"paths": [s["path"] for s in shards], "tokens": tokens,
                                                            "documents": docs, "dtype": "uint16_little_endian"},
                                                  **panels}}
        top = hosts.most_common(20)
        diversity[bucket] = {"tokens": tokens, "documents": docs, "top_hosts_or_origins": top,
                             "top1_document_share": (top[0][1] / docs) if docs and top else None}
    total = sum(s["train"]["tokens"] for s in manifest["sources"].values())
    manifest["total_train_tokens"] = total
    shares = config["planned_training_shares"]
    manifest["epochs_at_budget"] = {str(budget): {b: round(budget * shares[b] / max(1, manifest["sources"][b]["train"]["tokens"]), 2)
                                                   for b in BUCKETS} for budget in (17_000_000_000, 40_000_000_000)}
    write_json(out / "diversity_report.json", diversity)
    write_json(out / "manifest.json", manifest)
    event(log, "manifest_written", total_train_tokens=total,
          tokens={b: manifest["sources"][b]["train"]["tokens"] for b in BUCKETS},
          epochs_at_40B=manifest["epochs_at_budget"][str(40_000_000_000)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="v2/configs/prep.json")
    parser.add_argument("--stage", choices=["reused", "new", "manifest"], required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sources", default="stackexchange,books,dclm")
    args = parser.parse_args()
    config = load_config(args.config)
    out = ROOT / config["output_dir"]
    log = ROOT / "v2/phases/preparation/events.jsonl"
    decision = json.loads((out / "tokenizer_study" / "decision.json").read_text())
    tok_dir = out / "tokenizer"
    if not (tok_dir / "tokenizer.json").exists():
        tok_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(decision["chosen_path"], tok_dir / "tokenizer.json")
        specials = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
        write_json(tok_dir / "tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1024,
                   "pad_token": specials[0], "bos_token": specials[1], "eos_token": specials[2], "unk_token": specials[3],
                   "clean_up_tokenization_spaces": False})
        write_json(tok_dir / "special_tokens_map.json", dict(zip(["pad_token", "bos_token", "eos_token", "unk_token"], specials)))
        write_json(tok_dir / "provenance.json", {"chosen": decision["chosen"], "sha256": decision["chosen_sha256"],
                                                 "study": str(out / "tokenizer_study" / "decision.json")})
    if sha256_file(tok_dir / "tokenizer.json") != decision["chosen_sha256"]:
        raise RuntimeError("V2 tokenizer differs from the study decision")
    classifier = json.loads((out / "targeted" / "classifier.json").read_text())
    config["_threshold"] = classifier["threshold"]
    clf_path = str(out / "targeted" / "classifier.bin")
    if sha256_file(Path(clf_path)) != classifier["classifier_sha256"]:
        raise RuntimeError("Classifier file differs from its record")
    tok_path = str(tok_dir / "tokenizer.json")
    if args.stage == "reused":
        run_reused(config, out, tok_path, clf_path, args.workers, log)
    elif args.stage == "new":
        run_new(config, out, tok_path, clf_path, args.workers, log, args.sources.split(","))
    else:
        run_manifest(config, out, tok_path, log)


if __name__ == "__main__":
    main()
