"""Bounded, CPU-only source inspection; never publishes a training manifest.

Uses previously saved Hub metadata in artifacts/corpus_research. Selects three
files per source with a fixed seed and inspects 128 records per file. Parquet
reads use byte ranges; zstd reads stream only a prefix. This is a convenience
sample for access/schema/tokenization checks, not a population quality estimate.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import random
import sqlite3
import time
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_url
from tokenizers import Tokenizer
import zstandard

from scglm.prepare_data import ExclusionIndex, fingerprint  # ExclusionIndex also resolves the original __main__ pickle.

OUT = Path("artifacts/corpus_research")
TOKENIZER = Path("data/processed/main/tokenizer/tokenizer.json")
SPECS = [
    ("edu", "HuggingFaceFW/fineweb-edu", "sample/100BT/", ".parquet"),
    ("dclm", "mlfoundations/dclm-baseline-1.0", "global-shard_", ".jsonl.zst"),
    ("wiki", "wikimedia/wikipedia", "20231101.en/", ".parquet"),
    ("web_fallback", "HuggingFaceFW/fineweb", "sample/100BT/", ".parquet"),
    ("math_optional", "HuggingFaceTB/finemath", "finemath-4plus/", ".parquet"),
]


def inspect(spec):
    name, repo, prefix, suffix = spec
    started = time.monotonic()
    metadata = json.loads((OUT / (repo.replace("/", "--") + ".json")).read_text())
    candidates = [f for f in metadata["files"] if f["path"].startswith(prefix) and f["path"].endswith(suffix)]
    rng = random.Random(20260923)
    chosen = rng.sample(candidates, min(3, len(candidates)))
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    with Path("data/processed/main/exclusion_index.pkl").open("rb") as handle:
        exclusion = pickle.load(handle)
    prior = sqlite3.connect("file:data/processed/main/raw/dedup.sqlite3?mode=ro", uri=True)
    counts = Counter()
    lengths, chars, ratios, hashes, files = [], [], [], set(), []
    token_total = kept_tokens = 0
    examples = []
    fs = HfFileSystem(token=False)
    try:
        for entry in chosen:
            path = entry["path"]
            file_note = {"path": path, "file_bytes": entry["size"]}
            if suffix == ".parquet":
                with fs.open(f"datasets/{repo}@{metadata['revision']}/{path}", "rb", block_size=1024 * 1024) as handle:
                    parquet = pq.ParquetFile(handle)
                    group = rng.randrange(parquet.num_row_groups)
                    columns = [c for c in ["text", "id", "url", "title", "token_count", "score", "int_score"] if c in parquet.schema_arrow.names]
                    batch = next(parquet.iter_batches(batch_size=128, row_groups=[group], columns=columns, use_threads=False))
                    rows = batch.to_pylist()
                    file_note.update(row_group=group, total_row_groups=parquet.num_row_groups, schema=parquet.schema_arrow.names)
            else:
                url = hf_hub_url(repo, path, repo_type="dataset", revision=metadata["revision"])
                rows = []
                with urlopen(url, timeout=45) as response:
                    with zstandard.ZstdDecompressor().stream_reader(response) as reader:
                        with io.TextIOWrapper(reader, encoding="utf-8") as text:
                            for _ in range(128):
                                line = text.readline()
                                if not line:
                                    break
                                rows.append(json.loads(line))
                file_note.update(schema=sorted(rows[0]) if rows else [], sampling="first 128 rows of selected zstd shard")
            file_note["sampled_rows"] = len(rows)
            files.append(file_note)
            for row in rows:
                text = row.get("text", "")
                counts["sampled_documents"] += 1
                digest = fingerprint(text)
                n = len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
                lengths.append(n)
                chars.append(len(text))
                token_total += n
                ref = row.get("token_count")
                if ref and ref > 0:
                    ratios.append((n - 1) / ref)
                reasons = []
                if len(text) < 240 or len(text) > 200000:
                    reasons.append("length_rejected")
                if exclusion.matches(text, row.get("title", "")):
                    reasons.append("benchmark_exact_match")
                if prior.execute("SELECT 1 FROM seen WHERE hash=?", (digest,)).fetchone():
                    reasons.append("exact_match_prior_collected_pool")
                if digest in hashes:
                    reasons.append("exact_duplicate_in_source_sample")
                hashes.add(digest)
                for reason in reasons:
                    counts[reason] += 1
                if not reasons:
                    counts["eligible_under_sample_checks"] += 1
                    kept_tokens += n
                if len(examples) < 3:
                    examples.append({"source_id": row.get("id"), "url": row.get("url"), "content_sha256": digest, "our_tokens_with_eos": n})
    finally:
        prior.close()
    result = {
        "source": name, "repo": repo, "revision": metadata["revision"],
        "files_in_prefix": len(candidates), "prefix_file_bytes": sum(f["size"] or 0 for f in candidates),
        "files_sampled": files, "counts": dict(counts),
        "sample_tokens_with_eos": token_total, "eligible_sample_tokens_with_eos": kept_tokens,
        "token_length_quantiles": dict(zip(["p50", "p90", "p99"], np.quantile(lengths, [.5, .9, .99]).tolist())),
        "chars_per_our_token": sum(chars) / token_total,
        "median_our_to_source_token_count_ratio": float(np.median(ratios)) if ratios else None,
        "example_identities": examples, "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / f"sample_{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source": name, "counts": result["counts"], "sample_tokens": token_total, "elapsed_seconds": round(result["elapsed_seconds"], 1)}), flush=True)
    return result


if __name__ == "__main__":
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(spec[0], pool.submit(inspect, spec)) for spec in SPECS]
        for source, future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                result = {"source": source, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                results.append(result)
                print(json.dumps(result), flush=True)
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "seed": 20260923,
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "scope": "Bounded convenience sample; no training, global fuzzy deduplication, or corpus-wide quality certification",
        "prior_overlap_scope": "All prior collected raw documents, including reserved panels; broader than actual trained prefixes",
        "results": results,
    }
    (OUT / "sample_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(any("error" in result for result in results))
