"""Bounded, resumable corpus preparation. No pretrained model weights are used.

Run ``python -m scglm.prepare_data --config configs/data.json``. Preparation writes
raw eligible JSONL, a training-only byte BPE tokenizer, uint16 shards, document
indexes, evaluation JSONL, and an integrity/provenance manifest. Incomplete stages
never publish the final manifest. Source quotas are counts of stored token IDs;
the trainer reports its actual loss-bearing target count separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import re
import sqlite3
import time
import unicodedata
import zlib

import numpy as np

SPLITS = [f"controller_{i}_{kind}" for i in range(3) for kind in ("proposal", "confirm")] + ["selection", "final", "monitor"]
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
BENCHMARK_REVISIONS = {
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "ybisk/piqa": "ba2f7a4920f4968f04c174c53cc157c899e93501",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
}
MASK64 = (1 << 64) - 1
_WORKER_INDEX = None
_WORKER_LIMITS = None


def log(event: str, **values):
    print(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": event, **values}), flush=True)


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def ngram_hashes(text: str, size: int = 13):
    """Stable rolling 64-bit hashes of normalized word n-grams."""
    words = re.findall(r"\w+", normalized(text))
    if len(words) < size:
        return
    base, factor, current = 1000003, pow(1000003, size - 1, 1 << 64), 0
    hashes = [zlib.crc32(word.encode()) + 1 for word in words]
    for value in hashes[:size]:
        current = (current * base + value) & MASK64
    yield current
    for index in range(size, len(hashes)):
        current = ((current - hashes[index - size] * factor) * base + hashes[index]) & MASK64
        yield current


class ExclusionIndex:
    def __init__(self):
        self.ngrams = set()
        self.titles = set()
        self.report = []

    def add(self, text: str):
        self.ngrams.update(ngram_hashes(text))

    def matches(self, text: str, title: str = "") -> bool:
        if title and normalized(title.replace("_", " ")) in self.titles:
            return True
        return any(value in self.ngrams for value in ngram_hashes(text))


def benchmark_exclusions(output: Path) -> ExclusionIndex:
    from datasets import load_dataset
    from huggingface_hub import HfApi
    cached = output / "exclusion_index.pkl"
    index = None
    if cached.exists():
        with cached.open("rb") as handle:
            index = pickle.load(handle)
        log("exclusion_cache_loaded", ngrams=len(index.ngrams))
        if all(item["status"] == "loaded" for item in index.report):
            return index
    if index is None:
        index = ExclusionIndex()
    specs = [
        ("Salesforce/wikitext", "wikitext-103-raw-v1", ["validation", "test"], "wiki"),
        ("Rowan/hellaswag", None, ["train", "validation"], "hellaswag"),
        ("allenai/ai2_arc", "ARC-Easy", ["train", "validation", "test"], "arc"),
        ("ybisk/piqa", None, ["train", "validation", "test"], "piqa"),
        ("allenai/winogrande", "winogrande_xl", ["train", "validation"], "winogrande"),
    ]
    for repo, config, splits, kind in specs:
        if any(item["repo"] == repo and item["status"] == "loaded" for item in index.report):
            continue
        index.report = [item for item in index.report if item["repo"] != repo]
        entry = {"repo": repo, "config": config, "requested_splits": splits, "loaded_splits": {}, "status": "unavailable"}
        try:
            revision = BENCHMARK_REVISIONS[repo]
            entry["revision"] = revision
            for split in splits:
                if kind == "piqa":
                    uri = f"hf://datasets/{repo}@{revision}/plain_text/{split}-00000-of-00001.parquet"
                    data = load_dataset("parquet", data_files={split: [uri]}, split=split)
                    entry["loader"] = "Pinned primary-repository parquet conversion; no remote dataset script"
                else:
                    data = load_dataset(repo, config, revision=revision, split=split)
                count = 0
                for row in data:
                    if kind == "wiki":
                        texts = [row["text"]]
                        match = re.match(r"^\s*=\s+([^=]+?)\s+=\s*$", row["text"])
                        if match:
                            index.titles.add(normalized(match.group(1)))
                    elif kind == "hellaswag":
                        texts = [row.get("ctx", ""), *row.get("endings", [])]
                    elif kind == "arc":
                        texts = [row.get("question", ""), *row.get("choices", {}).get("text", [])]
                    elif kind == "piqa":
                        texts = [row.get("goal", ""), row.get("sol1", ""), row.get("sol2", "")]
                    else:
                        texts = [row.get("sentence", "")]
                    for text in texts:
                        index.add(text)
                    count += 1
                entry["loaded_splits"][split] = count
            entry["status"] = "loaded"
        except Exception as exc:
            entry["status"] = "partial" if entry["loaded_splits"] else "unavailable"
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        index.report.append(entry)
        log("benchmark_exclusions", **entry)
    entry = {
        "method": "NFKC/casefold/whitespace-normalized exact document dedup; exact Wikipedia title exclusion; any matching normalized 13-word rolling hash excludes the whole document",
        "limitations": ["No semantic/paraphrase or fuzzy near-duplicate certification", "13-word matching misses shorter benchmark fragments and altered text", "64-bit hash matching can cause conservative false exclusions", "Unavailable benchmark sources remain an explicit coverage gap", "No benchmark accuracy or answer labels used in sampling or training"],
        "benchmarks": index.report, "ngram_count": len(index.ngrams), "wiki_title_count": len(index.titles),
    }
    write_json(output / "exclusion_report.json", entry)
    temporary = cached.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cached)
    return index


def choose_split(content_hash: str) -> str:
    bucket = int(content_hash[:16], 16) % 100
    return SPLITS[bucket] if bucket < len(SPLITS) else "train"


def iter_source(spec: dict, seed: int = 20260923):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(spec["repo"], revision=spec["revision"])
    files = sorted(item.rfilename for item in info.siblings
                   if item.rfilename.startswith(spec["file_prefix"]) and item.rfilename.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"No pinned parquet files for {spec['repo']}")
    random.Random(seed).shuffle(files)
    for filename in files:
        log("source_file_start", source=spec["repo"], file=filename)
        uri = f"hf://datasets/{spec['repo']}@{spec['revision']}/{filename}"
        stream = load_dataset("parquet", data_files={"train": [uri]}, split="train", streaming=True)
        stream = stream.shuffle(seed=seed, buffer_size=10000)
        for row_number, row in enumerate(stream):
            yield row, filename, row_number


def _worker_ready():
    return True


def _preprocess_row(item):
    row, filename, row_number = item
    text = row.get("text", "")
    if not isinstance(text, str) or not _WORKER_LIMITS[0] <= len(text) <= _WORKER_LIMITS[1]:
        return row, filename, row_number, None, None, "length_or_type"
    digest = fingerprint(text)
    if _WORKER_INDEX.matches(text, str(row.get("title", ""))):
        return row, filename, row_number, None, None, "benchmark_match"
    return row, filename, row_number, digest, choose_split(digest), None


def filtered_source(spec, config, index):
    """Parallel pure-text filtering, bounded batches and deterministic output order."""
    global _WORKER_INDEX, _WORKER_LIMITS
    _WORKER_INDEX = index
    _WORKER_LIMITS = (config["minimum_document_chars"], config["maximum_document_chars"])
    # Fork before opening the source iterator. Workers do pure Python text/hash
    # work only: no GPU, tokenizer, Arrow, or network calls in children.
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("fork")) as pool:
        pool.submit(_worker_ready).result()
        iterator = iter_source(spec, config["seed"])
        while True:
            batch = []
            for _ in range(1024):
                try:
                    batch.append(next(iterator))
                except StopIteration:
                    break
            if not batch:
                break
            yield from pool.map(_preprocess_row, batch, chunksize=64)


def collect_raw(config: dict, output: Path, index: ExclusionIndex):
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    complete = raw / "complete.json"
    if complete.exists():
        log("raw_stage_already_complete")
        return json.loads(complete.read_text())
    # Incomplete collection restarts safely: do not trust an unfinished dedup DB.
    for path in raw.glob("*.jsonl"):
        path.unlink()
    db_path = raw / "dedup.sqlite3"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY) WITHOUT ROWID")
    stats = {}
    try:
        for source, spec in config["sources"].items():
            handles = {split: (raw / f"{source}.{split}.jsonl").open("w", encoding="utf-8") for split in ["train", *SPLITS]}
            counts, excluded, chars, used_files = Counter(), Counter(), 0, set()
            target = int(spec["target_train_tokens"] * config["raw_chars_per_target_token"])
            quotas = {split: config.get("monitor_documents", 128) if split == "monitor" else config["development_docs_per_split"] for split in SPLITS}
            started, seen = time.monotonic(), 0
            try:
                for row, filename, row_number, digest, split, rejection in filtered_source(spec, config, index):
                    seen += 1
                    text = row.get("text", "")
                    if rejection:
                        excluded[rejection] += 1
                        continue
                    if split != "train" and counts[split] >= quotas[split]:
                        excluded["surplus_reserved_development"] += 1
                        continue
                    inserted = db.execute("INSERT OR IGNORE INTO seen(hash) VALUES (?)", (digest,)).rowcount
                    if not inserted:
                        excluded["exact_cross_source_duplicate"] += 1
                        continue
                    source_id = str(row.get("id", row.get("url", f"{filename}:{row_number}")))
                    record = {"id": f"{source}:{digest}", "source": source, "source_id": source_id,
                              "url": row.get("url"), "title": row.get("title"), "content_sha256": digest,
                              "source_file": filename, "shuffled_file_row": row_number, "text": text}
                    handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    used_files.add(filename)
                    if split == "train":
                        chars += len(text)
                    if seen % 10000 == 0:
                        db.commit()
                        log("collect_progress", source=source, rows_seen=seen, train_chars=chars,
                            target_chars=target, docs=dict(counts), excluded=dict(excluded), elapsed_s=round(time.monotonic()-started, 1))
                    if chars >= target and all(counts[s] >= quotas[s] for s in SPLITS):
                        break
                if chars < target or any(counts[s] < quotas[s] for s in SPLITS):
                    raise RuntimeError(f"Insufficient eligible corpus for {source}: chars={chars}/{target}, docs={dict(counts)}")
            finally:
                for handle in handles.values():
                    handle.close()
            db.commit()
            stats[source] = {"documents": dict(counts), "train_chars": chars, "excluded": dict(excluded),
                             "rows_seen": seen, "files_read": sorted(used_files), "elapsed_s": time.monotonic()-started}
            log("source_collection_complete", source=source, **stats[source])
    finally:
        db.close()
    write_json(complete, stats)
    return stats


def tokenizer_train(config: dict, output: Path):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    directory = output / "tokenizer"
    directory.mkdir(exist_ok=True)
    path = directory / "tokenizer.json"
    if path.exists():
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab_size() != config["vocab_size"]:
            raise RuntimeError("Cached tokenizer vocabulary differs from configuration")
        return tokenizer
    observed = {}
    def iterator():
        for source in config["sources"]:
            chars = 0
            with (output / "raw" / f"{source}.train.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    text = json.loads(line)["text"]
                    yield text
                    chars += len(text)
                    if chars >= config["tokenizer_training_chars_per_source"]:
                        break
            observed[source] = chars
            if chars < config["tokenizer_training_chars_per_source"]:
                raise RuntimeError(f"Insufficient tokenizer training text for {source}")
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[3]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=config["vocab_size"], special_tokens=SPECIAL_TOKENS,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    if tokenizer.get_vocab_size() != config["vocab_size"]:
        raise RuntimeError(f"Tokenizer has {tokenizer.get_vocab_size()} IDs, expected {config['vocab_size']}")
    tokenizer.save(str(path))
    write_json(directory / "tokenizer_config.json", {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1024,
        "pad_token": SPECIAL_TOKENS[0], "bos_token": SPECIAL_TOKENS[1], "eos_token": SPECIAL_TOKENS[2], "unk_token": SPECIAL_TOKENS[3],
        "clean_up_tokenization_spaces": False})
    write_json(directory / "special_tokens_map.json", dict(zip(["pad_token", "bos_token", "eos_token", "unk_token"], SPECIAL_TOKENS)))
    write_json(directory / "training_provenance.json", {"partition": "train_only", "observed_chars": observed,
               "source_revisions": {k: v["revision"] for k,v in config["sources"].items()}})
    log("tokenizer_complete", chars=observed, vocabulary=tokenizer.get_vocab_size())
    return tokenizer


def iter_raw_records(path: Path, shuffle_buffer: int, seed: int):
    """Deterministic bounded shuffle; does not claim uniform full-corpus sampling."""
    rng, buffer = random.Random(seed), []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if shuffle_buffer <= 1:
                yield record
            elif len(buffer) < shuffle_buffer:
                buffer.append(record)
            else:
                selected = rng.randrange(len(buffer))
                yield buffer[selected]
                buffer[selected] = record
    rng.shuffle(buffer)
    yield from buffer


def tokenize_split(config, output, tokenizer, source, split):
    directory = output / source
    directory.mkdir(exist_ok=True)
    bin_path, idx_path = directory / f"{split}.bin", directory / f"{split}.idx.jsonl"
    eval_path = directory / f"{split}.jsonl"
    metadata_path = directory / f"{split}.meta.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    target = config["sources"][source]["target_train_tokens"] if split == "train" else None
    total, docs, truncated = 0, 0, 0
    records = iter_raw_records(output / "raw" / f"{source}.{split}.jsonl",
                config.get("document_shuffle_buffer", 10000) if split == "train" else 0, config["seed"])
    with bin_path.open("wb") as binary, idx_path.open("w", encoding="utf-8") as indexes, \
            eval_path.open("w", encoding="utf-8") as evaluations:
        while True:
            batch = []
            for _ in range(512):
                try:
                    record = next(records)
                except StopIteration:
                    break
                batch.append(record)
            if not batch:
                break
            encoded = tokenizer.encode_batch([r["text"] for r in batch], add_special_tokens=False)
            for record, encoding in zip(batch, encoded):
                ids = encoding.ids + [2]
                if split != "train" and len(ids) > config["development_max_tokens"]:
                    ids = ids[:config["development_max_tokens"]-1] + [2]
                    truncated += 1
                if target is not None and total + len(ids) > target:
                    ids = ids[:target-total]
                    truncated += 1
                if not ids:
                    break
                np.asarray(ids, dtype="<u2").tofile(binary)
                record.pop("text")
                indexes.write(json.dumps({**record, "offset": total, "length": len(ids)}, ensure_ascii=False) + "\n")
                if split != "train":
                    evaluations.write(json.dumps({"id": record["id"], "source": source, "tokens": ids}) + "\n")
                total += len(ids)
                docs += 1
                if target is not None and total >= target:
                    break
            if docs and docs % 10240 == 0:
                log("tokenize_progress", source=source, split=split, documents=docs, tokens=total, target=target)
            if target is not None and total >= target:
                break
    if target is not None and total < target:
        raise RuntimeError(f"Insufficient actual {source} tokens ({total} < {target}); increase raw_chars_per_target_token and recollect. No final manifest published.")
    info = {"path": str(bin_path.resolve()), "index_path": str(idx_path.resolve()),
            "tokens": total, "documents": docs, "dtype": "uint16_little_endian", "truncated_documents": truncated,
            "sha256": sha256_file(bin_path), "index_sha256": sha256_file(idx_path)}
    if split != "train":
        info.update(jsonl_path=str(eval_path.resolve()), jsonl_sha256=sha256_file(eval_path))
    else:
        eval_path.unlink()
    write_json(metadata_path, info)
    log("tokenize_complete", source=source, split=split, tokens=total, documents=docs)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--token-scale", type=float, default=1.0, help="Scale source quotas for a separate smoke corpus")
    parser.add_argument("--stage", choices=["all", "collect", "tokenize"], default="all")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.token_scale != 1:
        if not args.output_dir:
            raise ValueError("Smoke --token-scale requires its own --output-dir")
        for spec in config["sources"].values():
            spec["target_train_tokens"] = max(100000, int(spec["target_train_tokens"] * args.token_scale))
        config["tokenizer_training_chars_per_source"] = min(config["tokenizer_training_chars_per_source"],
             int(min(s["target_train_tokens"] for s in config["sources"].values()) * 2))
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "preparation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing corpus configuration differs; use a new output directory")
    write_json(config_path, config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", "16")
    log("preparation_start", output=str(output), config=config)
    if args.stage != "tokenize":
        exclusion = benchmark_exclusions(output)
        raw_stats = collect_raw(config, output, exclusion)
        if args.stage == "collect":
            return
    else:
        raw_stats = json.loads((output / "raw" / "complete.json").read_text())
    tokenizer = tokenizer_train(config, output)
    sources = {}
    for source, spec in config["sources"].items():
        sources[source] = {"provenance": spec, "collection": raw_stats[source], "splits": {}}
        for split in ["train", *SPLITS]:
            sources[source]["splits"][split] = tokenize_split(config, output, tokenizer, source, split)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "configuration_sha256": sha256_file(config_path),
        "tokenizer": {"path": str((output / "tokenizer" / "tokenizer.json").resolve()),
            "directory": str((output / "tokenizer").resolve()), "sha256": sha256_file(output / "tokenizer" / "tokenizer.json"),
            "vocab_size": tokenizer.get_vocab_size(), "special_ids": {"pad": 0, "bos": 1, "eos": 2, "unk": 3}},
        "sources": sources, "exclusion_report": str((output / "exclusion_report.json").resolve()),
        "partition_policy": "Normalized content hash modulo 100; 0..8 reserved for separate development/monitor panels; remaining buckets training; global exact dedup before export",
        "shuffle_policy": {"seed": config["seed"], "source_file_order": "seeded permutation", "source_stream_buffer": 10000, "train_document_buffer": config.get("document_shuffle_buffer", 10000)},
        "token_accounting": "Stored source token IDs include EOS separators. Trainer separately counts loss-bearing targets; circular stream wraps and cycle counts must be reported.",
        "scope_limits": ["13-word exact substring exclusion is not comprehensive fuzzy/semantic decontamination", "Source database licensing is distinct from underlying content rights", "Sequential bounded source sample is not a claim of a uniformly random sample of the entire corpus"]}
    write_json(output / "manifest.json", manifest)
    log("preparation_complete", manifest=str(output / "manifest.json"))


if __name__ == "__main__":
    main()
