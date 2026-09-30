"""Audit completed continuation data without scoring any reserved evaluation set."""
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np
from huggingface_hub import hf_hub_download

from scglm.train import atomic_json, sha256_file


def check_index(path, tokens, reserved):
    offset = documents = 0
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            if row["content_sha256"] in reserved:
                raise ValueError("Training document overlaps a reserved panel")
            length = row["length"]
            if row["offset"] != offset or length < 2 or tokens[offset+length-1] != 2:
                raise ValueError("Broken document boundary/offset/EOS")
            offset += length
            documents += 1
    if offset != len(tokens):
        raise ValueError("Index does not cover the entire token stream")
    return documents


def preflight(config):
    start = time.monotonic()
    manifest_path = Path(config["data_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("Corpus preparation is incomplete")
    model = json.loads(Path(config["model_config"]).read_text())
    if sha256_file(manifest["tokenizer"]["path"]) != manifest["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer integrity failed")
    sources = list(config["source_weights"])
    steps = config["target_tokens"] // (config["sequence_length"] * config["batch_sequences"])
    # Reproduce exactly the RNG draws used by the fixed training sampler.
    rng = np.random.default_rng(config["seed"])
    counts = np.zeros(len(sources), dtype=np.int64)
    for first in range(0, steps, 1000):
        choices = rng.choice(len(sources), size=min(1000, steps-first)*config["batch_sequences"],
                             p=[config["source_weights"][s] for s in sources])
        counts += np.bincount(choices, minlength=len(sources)) * config["sequence_length"]
    reserved = set()
    for spec in manifest["sources"].values():
        for name, split in spec["splits"].items():
            if name == "train":
                continue
            path = Path(split["jsonl_path"])
            if sha256_file(path) != split["jsonl_sha256"]:
                raise ValueError("Panel checksum mismatch")
            n = 0
            for line in path.open():
                row = json.loads(line)
                digest = row["content_sha256"]
                if digest in reserved:
                    raise ValueError("Reserved panel exact overlap")
                reserved.add(digest)
                if not row["tokens"] or max(row["tokens"]) >= model["vocab_size"] or min(row["tokens"]) < 0:
                    raise ValueError("Invalid reserved token IDs")
                n += 1
            if n != split["documents"]:
                raise ValueError("Panel document count mismatch")
    results = {}
    downloaded_files = {}
    for i, source in enumerate(sources):
        split = manifest["sources"][source]["splits"]["train"]
        path = Path(split["path"])
        if sha256_file(path) != split["sha256"] or sha256_file(split["index_path"]) != split["index_sha256"]:
            raise ValueError("Training corpus checksum mismatch")
        if path.stat().st_size != 2*split["tokens"]:
            raise ValueError("Incorrect uint16 token file length")
        tokens = np.memmap(path, dtype="<u2", mode="r")
        histogram = np.zeros(model["vocab_size"], dtype=np.int64)
        for first in range(0, len(tokens), 16_000_000):
            block = tokens[first:first+16_000_000]
            if block.max() >= model["vocab_size"]:
                raise ValueError("Out-of-vocabulary token in corpus")
            histogram += np.bincount(block, minlength=len(histogram))
        documents = check_index(split["index_path"], tokens, reserved)
        if documents != split["documents"]:
            raise ValueError("Training document count differs")
        if counts[i] > (len(tokens)-1)*config["max_source_epochs"]:
            raise ValueError(f"Insufficient unique source pool: {source}")
        results[source] = {"stored_tokens": len(tokens), "documents": documents,
                           "planned_loss_targets": int(counts[i]), "coverage_fraction": float(counts[i]/(len(tokens)-1)),
                           "eos_tokens": int(histogram[2]), "unk_tokens": int(histogram[3]),
                           "reserved_exact_overlap": 0, "sha256": split["sha256"]}
        spec = manifest["sources"][source]["provenance"]
        metadata = json.loads(Path(spec["metadata_file"]).read_text())
        files = sorted(f["path"] for f in metadata["files"] if f["path"].startswith(spec["file_prefix"])
                       and f["path"].endswith((".parquet", ".jsonl.zst")))
        random.Random(config["seed"]).shuffle(files)
        used_count = manifest["sources"][source]["collection"]["file"]+1
        downloaded_files[source] = []
        for name in files[:used_count]:
            local = hf_hub_download(spec["repo"], name, repo_type="dataset", revision=spec["revision"], local_files_only=True)
            downloaded_files[source].append({"repo": spec["repo"], "revision": spec["revision"], "path": name,
                                               "size": Path(local).stat().st_size, "sha256": sha256_file(local)})
        print(json.dumps({"event": "preflight_source_passed", "source": source, **results[source]}), flush=True)
    result = {"status": "passed", "manifest_sha256": sha256_file(manifest_path), "sources": results,
              "downloaded_source_files": downloaded_files,
              "reserved_documents_checked_without_model_scoring": len(reserved),
              "planned_additional_loss_targets": int(counts.sum()), "elapsed_seconds": time.monotonic()-start,
              "limits": "Document and URL exact dedup plus approximate MinHash; paraphrase/semantic leakage is not ruled out."}
    atomic_json(Path("artifacts/extension_data_preflight.json"), result)
    return result


if __name__ == "__main__":
    preflight(json.loads(Path("configs/train_extension.json").read_text()))
