"""Freeze the unused-span pool, an independent held-out panel and both arms' shards.

Only whole documents that start strictly after the final consumed stream
cursor and end with their real EOS are eligible. Documents already used as
choice-study raw panels or historical replay, and their URL groups, are
excluded. The held-out panel is split by URL group before either arm is
selected, so no held-out group enters training. Both arms share the same
eligible pool, token targets and source strata; only the selection rule differs.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from scglm_post.common import identity
from scglm_study.common import (ROOT, config_digest, digest, event, immutable, read, read_jsonl, roots, sha256,
                                write_json, write_jsonl)
from .quality import coherence, features, hard_failures

ARMS = ("random", "curated")
DOCUMENT_ID = re.compile(r"(?:^|:)(edu|dclm|wiki|web):([0-9a-f]{64})")


def consumed_cursors(cfg):
    """Final stream positions of the completed continuation runs, checked against the checkpoint."""
    import torch
    consumed = Counter()
    for path in cfg["corpus"]["consumed_by"]:
        state = read(ROOT / path)
        if state["status"] != "completed" or state.get("probe_tokens", 0) or state.get("controller_seconds", 0):
            raise ValueError("Consumed-stream history is not auditable")
        consumed.update(state["source_tokens"])
    parent = cfg["parent"]
    checkpoint = ROOT / parent["checkpoint"]
    if sha256(checkpoint) != parent["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    for source, stream in payload["streams"].items():
        if stream["cycles"] or stream["cursor"] != consumed[source] or stream["tokens_consumed"] != consumed[source]:
            raise ValueError("Continuation streams wrapped or disagree with status totals")
    del payload
    return dict(consumed)


def exposure(cfg):
    """Content hashes and URL groups that must never enter either arm or the held-out panel."""
    spec = cfg["exclusions"]
    contents, groups, receipts = set(), set(), {}
    choice = ROOT / spec["choice_data"]
    if sha256(choice / "manifest.json") != spec["choice_manifest_sha256"]:
        raise ValueError("Choice-study manifest changed")
    files = read(choice / "manifest.json")["files"]
    for name in spec["choice_raw_panels"]:
        if sha256(choice / name) != files[name]:
            raise ValueError(f"Choice raw panel changed: {name}")
        receipts[f"{spec['choice_data']}/{name}"] = files[name]
        for row in read_jsonl(choice / name):
            contents.add(row["content_sha256"])
            groups.add(row["group_id"])
    for entry in spec["replay_files"]:
        path = ROOT / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Replay record changed: {entry['path']}")
        receipts[entry["path"]] = entry["sha256"]
        with path.open() as stream:
            for line in stream:
                match = DOCUMENT_ID.search(json.loads(line)["id"])
                if match:
                    contents.add(match.group(2))
    # Earlier studies' scored raw documents came from reserved panel files, not
    # the training stream; they are still excluded as a conservative measure.
    for path in sorted(ROOT.glob(spec["evaluation_record_glob"])):
        receipts[str(path.relative_to(ROOT))] = sha256(path)
        with path.open() as stream:
            for line in stream:
                match = DOCUMENT_ID.search(json.loads(line)["id"])
                if match:
                    contents.add(match.group(2))
    return {"contents": contents, "groups": groups, "receipts": receipts}


def first_entry_after(index_path, cursor):
    """Byte offset of the first index line whose document starts after ``cursor``.

    Offsets in the index increase monotonically, so a binary search over line
    boundaries avoids parsing millions of consumed entries.
    """
    with open(index_path, "rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()

        def line_start(position):
            # First line beginning at or after ``position``.
            if position == 0:
                return 0
            stream.seek(position - 1)
            stream.readline()
            return stream.tell()

        def after(position):
            start = line_start(position)
            if start >= size:
                return True
            stream.seek(start)
            return json.loads(stream.readline())["offset"] > cursor

        # ``after`` is monotone in byte position: a lower-bound search.
        low, high = 0, size
        while low < high:
            middle = (low + high) // 2
            if after(middle):
                high = middle
            else:
                low = middle + 1
        return line_start(low)


def _host(url):
    try:
        return (urlsplit(url).hostname or "none").lower()
    except ValueError:
        return "none"


def scan(cfg, source, cursor, excluded, tokenizer, needed_tokens, directory):
    """Stream eligible documents after ``cursor`` until the pool holds ``needed_tokens``."""
    spec = read(ROOT / cfg["corpus"]["manifest"])["sources"][source]["splits"]["train"]
    binary, index = Path(spec["path"]), Path(spec["index_path"])
    mmap = np.memmap(binary, mode="r", dtype="<u2")
    eos = cfg["tokenizer"]["eos_id"]
    rules, weights = cfg["selection"]["hard_filters"], cfg["selection"]["coherence_weights"]
    counts, pool, pending = Counter(), [], []
    total = 0

    def flush():
        nonlocal total
        texts = tokenizer.decode_batch([mmap[e["offset"]:e["offset"] + e["length"] - 1].tolist() for e in pending],
                                       skip_special_tokens=True)
        for entry, text in zip(pending, texts):
            f = features(text)
            failures = hard_failures(f, rules)
            pool.append({"id": entry["id"], "source": source, "offset": entry["offset"], "length": entry["length"],
                         "content_sha256": entry["content_sha256"], "url": entry.get("url"), "host": _host(entry.get("url")),
                         "group_id": identity(entry.get("url") or entry["content_sha256"]),
                         "source_file": entry.get("source_file"), "features": f, "failures": failures,
                         "score": coherence(f, weights) if not failures else None})
            counts["passed_hard_filters" if not failures else "failed_hard_filters"] += 1
            total += entry["length"]
        pending.clear()

    with open(index, "rb") as stream:
        stream.seek(first_entry_after(index, cursor))
        for line in stream:
            entry = json.loads(line)
            counts["seen"] += 1
            if entry["offset"] <= cursor:
                raise ValueError("Consumed document returned by the unused-span search")
            if entry["content_sha256"] in excluded["contents"]:
                counts["historical_exposure"] += 1
                continue
            if identity(entry.get("url") or entry["content_sha256"]) in excluded["groups"]:
                counts["exposed_url_group"] += 1
                continue
            if entry["length"] < cfg["selection"]["min_document_tokens"] or int(mmap[entry["offset"] + entry["length"] - 1]) != eos:
                counts["short_or_truncated"] += 1
                continue
            pending.append(entry)
            if len(pending) == 512:
                flush()
                if total >= needed_tokens:
                    break
        if pending:
            flush()
    del mmap
    event(directory, "pool_scanned", source=source, documents=len(pool), tokens=total, counts=dict(counts))
    if total < needed_tokens:
        raise ValueError(f"Unused span of {source} cannot supply the registered pool")
    return pool, dict(counts), {"path": str(binary), "sha256": spec["sha256"], "index_path": str(index),
                                "index_sha256": spec["index_sha256"], "cursor": cursor}


def _rank(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def split_heldout(cfg, pool, source, seed):
    """Whole URL groups, in hash order, become the independent held-out panel."""
    spec = cfg["selection"]
    groups = defaultdict(list)
    for doc in pool:
        groups[doc["group_id"]].append(doc)
    held, tokens = [], 0
    for group in sorted(groups, key=lambda g: _rank(seed, g)):
        if len(held) >= spec["heldout_documents_per_source"] and tokens >= spec["heldout_min_scored_tokens_per_source"]:
            break
        held.extend(groups[group])
        tokens += sum(min(d["length"], 1024) - 1 for d in groups[group])
    heldout_groups = {d["group_id"] for d in held}
    return held, [d for d in pool if d["group_id"] not in heldout_groups]


def select_random(pool, target, seed):
    chosen, tokens = [], 0
    for doc in sorted(pool, key=lambda d: _rank(seed, d["id"])):
        if tokens >= target:
            break
        chosen.append(doc)
        tokens += doc["length"]
    if tokens < target:
        raise ValueError("Random arm pool is too small")
    return chosen


def length_bin(length, edges):
    for i, edge in enumerate(edges):
        if length < edge:
            return i
    return len(edges)


def select_curated(pool, reference, target, spec, *, host_cap=True):
    """Top-coherence passing documents per length bin, matching the random arm's length mix.

    A per-host document cap keeps topical breadth. A bin short of eligible
    documents is filled afterwards from the remaining highest scores; the
    shortfall is recorded rather than hidden. Single-host sources such as
    Wikipedia pass ``host_cap=False``: a host cap there would cap the source.
    """
    edges = spec["length_bin_edges"]
    passing = [d for d in pool if d["score"] is not None]
    reference_bins = Counter()
    for d in reference:
        reference_bins[length_bin(d["length"], edges)] += d["length"]
    total_reference = sum(reference_bins.values())
    host_cap = max(1, math.floor(spec["max_host_document_fraction"] * len(reference))) if host_cap else len(pool)
    ranked = sorted(passing, key=lambda d: (-d["score"], d["id"]))
    by_bin = defaultdict(list)
    for d in ranked:
        by_bin[length_bin(d["length"], edges)].append(d)
    hosts, chosen, chosen_ids, shortfall = Counter(), [], set(), {}
    for b, share in sorted(reference_bins.items()):
        quota, tokens = target * share / total_reference, 0
        for d in by_bin[b]:
            if tokens >= quota:
                break
            if hosts[d["host"]] >= host_cap:
                continue
            chosen.append(d)
            chosen_ids.add(d["id"])
            hosts[d["host"]] += 1
            tokens += d["length"]
        if tokens < quota:
            shortfall[str(b)] = quota - tokens
    tokens = sum(d["length"] for d in chosen)
    for d in ranked:
        if tokens >= target:
            break
        if d["id"] in chosen_ids or hosts[d["host"]] >= host_cap:
            continue
        chosen.append(d)
        chosen_ids.add(d["id"])
        hosts[d["host"]] += 1
        tokens += d["length"]
    if tokens < target:
        raise ValueError("Curated arm cannot reach its registered token target")
    return chosen, {"length_bin_shortfall_tokens": shortfall, "host_cap": host_cap,
                    "passing_documents": len(passing), "pool_documents": len(pool)}


def write_shard(docs, source_spec, directory, seed):
    """Concatenate whole stored documents (EOS included) in a seeded order."""
    directory.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(source_spec["path"], mode="r", dtype="<u2")
    order = sorted(docs, key=lambda d: _rank(seed, d["id"]))
    index, offset = [], 0
    with (directory / "train.bin").open("wb") as stream:
        for d in order:
            tokens = np.asarray(mmap[d["offset"]:d["offset"] + d["length"]], dtype="<u2")
            stream.write(tokens.tobytes())
            index.append({"id": d["id"], "offset": offset, "length": d["length"], "source_offset": d["offset"],
                          "content_sha256": d["content_sha256"], "url": d["url"]})
            offset += d["length"]
    del mmap
    write_jsonl(directory / "train.idx.jsonl", index)
    return {"path": str(directory / "train.bin"), "sha256": sha256(directory / "train.bin"), "tokens": offset,
            "documents": len(order), "dtype": "uint16_little_endian", "index_path": str(directory / "train.idx.jsonl"),
            "index_sha256": sha256(directory / "train.idx.jsonl")}


def heldout_rows(docs, source_spec, stratum):
    mmap = np.memmap(source_spec["path"], mode="r", dtype="<u2")
    rows = [{"id": d["id"], "source": stratum, "group_id": d["group_id"],
             "tokens": mmap[d["offset"]:d["offset"] + min(d["length"], 1024)].astype(int).tolist(),
             "full_document_tokens": d["length"], "content_sha256": d["content_sha256"], "offset": d["offset"],
             "url": d["url"], "passes_quality_filters": d["score"] is not None} for d in docs]
    del mmap
    return rows


def summary(docs):
    passing = [d for d in docs if d["score"] is not None]
    return {"documents": len(docs), "tokens": sum(d["length"] for d in docs),
            "hard_filter_pass_fraction": len(passing) / max(1, len(docs)),
            "mean_coherence_of_passing": float(np.mean([d["score"] for d in passing])) if passing else None,
            "mean_document_tokens": float(np.mean([d["length"] for d in docs])) if docs else None,
            "distinct_hosts": len({d["host"] for d in docs})}


def prepare(cfg):
    from tokenizers import Tokenizer
    data, _ = roots(cfg)
    if (data / "manifest.json").exists():
        return load(cfg)
    data.mkdir(parents=True, exist_ok=True)
    corpus = ROOT / cfg["corpus"]["manifest"]
    if sha256(corpus) != cfg["corpus"]["sha256"]:
        raise ValueError("Continuation corpus manifest changed")
    tokenizer_path = ROOT / cfg["tokenizer"]["path"]
    if sha256(tokenizer_path) != cfg["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.token_to_id("<|eos|>") != cfg["tokenizer"]["eos_id"]:
        raise ValueError("Tokenizer EOS identity changed")
    cursors = consumed_cursors(cfg)
    excluded = exposure(cfg)
    training = cfg["training"]
    targets_per_arm = training["updates"] * training["batch_sequences"] * training["sequence_length"]
    seed, spec = training["seed"], cfg["selection"]
    arms = {arm: {} for arm in ARMS}
    heldout, audit, receipts = [], {}, dict(excluded["receipts"])
    for source, weight in cfg["source_weights"].items():
        # A margin covers binomial per-sequence source sampling; +1 lookahead token.
        target = math.ceil(targets_per_arm * weight * spec["train_token_margin"]) + 1
        pool, counts, source_spec = scan(cfg, source, cursors[source], excluded, tokenizer,
                                         math.ceil(target * spec["pool_multiple"]), data)
        receipts[source_spec["path"]] = source_spec["sha256"]
        receipts[source_spec["index_path"]] = source_spec["index_sha256"]
        (data / "pool").mkdir(exist_ok=True)
        write_jsonl(data / "pool" / f"{source}.jsonl", pool)
        held, remaining = split_heldout(cfg, pool, source, seed)
        heldout.extend(heldout_rows(held, source_spec, f"heldout_{source}"))
        random_docs = select_random(remaining, target, seed)
        curated_docs, curated_audit = select_curated(remaining, random_docs, target, spec,
                                                     host_cap=source in spec["host_cap_sources"])
        overlap = len({d["id"] for d in random_docs} & {d["id"] for d in curated_docs})
        for arm, docs in (("random", random_docs), ("curated", curated_docs)):
            arms[arm][source] = write_shard(docs, source_spec, data / arm / source, seed)
        audit[source] = {"cursor": cursors[source], "scan": counts, "pool": summary(pool), "heldout": summary(held),
                         "random": summary(random_docs), "curated": {**summary(curated_docs), **curated_audit},
                         "arm_document_overlap": overlap, "target_tokens_per_arm": target}
    write_jsonl(data / "heldout_raw.jsonl", heldout)
    corpus_manifest = read(corpus)
    arm_manifests = {}
    for arm in ARMS:
        manifest = {"schema_version": 2, "status": "complete", "study": cfg["experiment_id"], "arm": arm,
                    "parent_manifest_sha256": cfg["corpus"]["sha256"],
                    "tokenizer": {**corpus_manifest["tokenizer"], "path": str(tokenizer_path)},
                    "sources": {source: {"splits": {"train": arms[arm][source],
                                                    "monitor": corpus_manifest["sources"][source]["splits"]["monitor"]}}
                                for source in cfg["source_weights"]}}
        write_json(data / arm / "manifest.json", manifest)
        arm_manifests[arm] = sha256(data / arm / "manifest.json")
    write_json(data / "audit.json", audit)
    manifest = {"schema": "scglm-midtrain-data-v1", "status": "complete", "config_sha256": config_digest(cfg),
                "arm_manifests": arm_manifests, "heldout_sha256": sha256(data / "heldout_raw.jsonl"),
                "heldout_documents": len(heldout), "audit_sha256": sha256(data / "audit.json"),
                "pool_sha256": {s: sha256(data / "pool" / f"{s}.jsonl") for s in cfg["source_weights"]},
                "source_receipts": receipts, "consumed_cursors": cursors,
                "exposure_scope": "Unused span after final cursor; choice raw panels, historical replay and their URL groups excluded; "
                                  "corpus-preparation exact/MinHash dedup inherited; not semantic decontamination.",
                "model_scores_used_for_selection": False}
    immutable(data / "manifest.json", manifest)
    event(data, "prepared", arm_manifests=arm_manifests, heldout_documents=len(heldout))
    return load(cfg)


def load(cfg):
    data, _ = roots(cfg)
    manifest = read(data / "manifest.json")
    if manifest["status"] != "complete" or manifest["config_sha256"] != config_digest(cfg):
        raise ValueError("Prepared data belongs to a different configuration")
    if sha256(data / "heldout_raw.jsonl") != manifest["heldout_sha256"] or sha256(data / "audit.json") != manifest["audit_sha256"]:
        raise ValueError("Held-out panel or audit changed")
    for arm, expected in manifest["arm_manifests"].items():
        if sha256(data / arm / "manifest.json") != expected:
            raise ValueError(f"Arm manifest changed: {arm}")
    return manifest


def verify_shards(cfg, arm):
    """Full checksum pass over one arm's shards; separate because it reads gigabytes."""
    data, _ = roots(cfg)
    manifest = read(data / arm / "manifest.json")
    for source, entry in manifest["sources"].items():
        train = entry["splits"]["train"]
        if sha256(train["path"]) != train["sha256"] or sha256(train["index_path"]) != train["index_sha256"]:
            raise ValueError(f"Arm shard changed: {arm}/{source}")
    return manifest
