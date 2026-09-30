"""Download pinned human SFT sources, decontaminate, group, split, and tokenize."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import random
import re
import shutil

from .common import (FORMAT_VERSION, ROOT, ValidationRun, check_encoded, encode_example,
                     identity, normalized, read_jsonl, sha256, write_json, write_jsonl)
from .procedural import generate, correct
from .system_data import add_human_system, generate_system


def human_sources(config, output):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    files, rows, rejected = {}, [], Counter()
    for source, spec in config["sources"].items():
        if not re.fullmatch("[a-f0-9]{40}", spec["revision"]):
            raise ValueError("Every source requires an immutable commit")
        downloaded = {}
        for filename in ["README.md", *spec["files"].values()]:
            path = Path(hf_hub_download(repo_id=spec["repository"], repo_type="dataset",
                                      revision=spec["revision"], filename=filename))
            downloaded[filename] = path
            files[f"{source}/{filename}"] = {"cache_path": str(path), "sha256": sha256(path),
                                             "repository": spec["repository"], "revision": spec["revision"]}
        shutil.copy2(downloaded["README.md"], output / f"{source}.README.md")
        if source == "dolly":
            for i, row in enumerate(read_jsonl(downloaded[spec["files"]["train"]])):
                prompt = row["instruction"].strip()
                context = row["context"].strip()
                if context:
                    prompt += "\n\nContext:\n" + context
                rows.append({"id": f"dolly:{i}", "source": source, "group": f"dolly:{i}",
                             "category": row["category"], "context": context, "license": spec["license"],
                             "messages": [{"role": "user", "content": prompt},
                                          {"role": "assistant", "content": row["response"].strip()}],
                             "provenance": {"row": i, "revision": spec["revision"], "original_split": "train"}})
        elif source == "oasst1":
            all_messages = []
            for split, filename in spec["files"].items():
                for row in pq.read_table(downloaded[filename]).to_pylist():
                    all_messages.append({**row, "original_split": split})
            examples, counts = oasst_examples(all_messages, spec)
            rows.extend(examples)
            rejected.update(counts)
        else:
            raise ValueError(f"Unsupported source: {source}")
    return rows, files, rejected


def oasst_eligible(row):
    if not (row.get("lang") == "en" and row.get("deleted") is False
            and row.get("review_result") is True and row.get("synthetic") is False
            and not row.get("model_name") and row.get("review_count", 0) > 0):
        return False
    labels = row.get("labels") or {}
    ratings = dict(zip(labels.get("name", []), labels.get("value", [])))
    if any(ratings.get(name, 0) > 0 for name in ("spam", "pii", "lang_mismatch")):
        return False
    if ratings.get("quality", 1) < 0.5 or ratings.get("fails_task", 0) > 0.25:
        return False
    return True


def oasst_examples(messages, spec):
    by_id = {r["message_id"]: r for r in messages}
    if len(by_id) != len(messages):
        raise ValueError("Duplicate OpenAssistant message ID")
    children = defaultdict(list)
    for row in messages:
        if row["role"] == "assistant" and oasst_eligible(row):
            children[row["parent_id"]].append(row)
    rows, rejected = [], Counter()
    for replies in children.values():
        # Rank 0 is best. Missing ranks sort after rated replies.
        answer = min(replies, key=lambda r: (r["rank"] if r.get("rank") is not None else 10**6, r["message_id"]))
        path, seen, current = [], set(), answer
        while current is not None:
            key = current["message_id"]
            if key in seen:
                raise ValueError("Cyclic OpenAssistant ancestry")
            seen.add(key)
            path.append(current)
            parent = current["parent_id"]
            if parent is not None and parent not in by_id:
                path = []
                break
            current = by_id.get(parent)
        path.reverse()
        if not path or any(not oasst_eligible(r) for r in path):
            rejected["oasst_ancestry_metadata"] += 1
            continue
        if len({r["message_tree_id"] for r in path}) != 1 or len({r["original_split"] for r in path}) != 1:
            raise ValueError("Inconsistent OpenAssistant tree identity")
        rows.append({"id": f"oasst1:{answer['message_id']}", "source": "oasst1",
                     "group": f"oasst1:{answer['message_tree_id']}", "category": "conversation",
                     "license": spec["license"], "force_holdout": answer["original_split"] != "train",
                     "messages": [{"role": "user" if r["role"] == "prompter" else r["role"],
                                   "content": r["text"].strip()} for r in path],
                     "provenance": {"revision": spec["revision"], "original_split": answer["original_split"],
                                    "message_ids": [r["message_id"] for r in path], "rank": answer["rank"],
                                    "authorship": "filtered human-submitted; metadata cannot prove authorship"}})
    rejected["oasst_messages_not_exported"] = len(messages) - len(rows)
    return rows, rejected


def group_and_split(rows, seed):
    """Keep whole trees and exact/lexical duplicate clusters in one split."""
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)

    keys, postings, shingles = {}, defaultdict(list), []
    for i, row in enumerate(rows):
        prompt = "\n".join(m["content"] for m in row["messages"][:-1])
        exact_keys = [row["group"], "prompt:" + identity(prompt)]
        if len(row.get("context", "")) >= 80:
            exact_keys.append("context:" + identity(row["context"]))
        for key in exact_keys:
            if key in keys:
                union(i, keys[key])
            keys[key] = i
        words = re.findall(r"\w+", normalized(prompt))
        grams = {tuple(words[j:j+5]) for j in range(len(words) - 4)}
        shingles.append(grams)
        for gram in grams:
            postings[gram].append(i)
    fuzzy_pairs = 0
    for i, grams in enumerate(shingles):
        # Frequent boilerplate is not informative; bound candidates for long documents.
        rare = sorted((g for g in grams if len(postings[g]) <= 100), key=lambda g: (len(postings[g]), g))[:20]
        candidates = {j for g in rare for j in postings[g] if j < i}
        for j in candidates:
            other = shingles[j]
            if len(grams & other) / max(1, len(grams | other)) >= 0.8:
                union(i, j)
                fuzzy_pairs += 1
    clusters = defaultdict(list)
    for i in range(len(rows)):
        clusters[root(i)].append(i)
    seen_examples, kept = set(), []
    for members in clusters.values():
        cluster = min(rows[i]["id"] for i in members)
        bucket = int(identity(f"{seed}:{cluster}")[:16], 16) % 100
        forced = any(rows[i].get("force_holdout") for i in members)
        split = ("development" if bucket % 2 == 0 else "confirmation") if forced else (
            "development" if bucket < 5 else "confirmation" if bucket < 10 else "train")
        for i in members:
            row = rows[i]
            signature = identity(json.dumps(row["messages"], sort_keys=True, ensure_ascii=False))
            if signature in seen_examples:
                continue
            seen_examples.add(signature)
            kept.append({**row, "group": cluster, "split": split, "content_hash": signature})
    return kept, {"clusters": len(clusters), "lexical_near_duplicate_pairs": fuzzy_pairs,
                  "exact_examples_removed": len(rows) - len(kept)}


def exclusion_index(config, tokenizer):
    from scglm.prepare_data import ExclusionIndex, BENCHMARK_REVISIONS

    class LocalUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "ExclusionIndex" and module in ("__main__", "scglm.prepare_data"):
                return ExclusionIndex
            return super().find_class(module, name)

    path = ROOT / config["benchmark_index"]
    if sha256(path) != config["benchmark_index_sha256"]:
        raise ValueError("Benchmark exclusion index changed")
    with path.open("rb") as stream:
        index = LocalUnpickler(stream).load()
    if {r["repo"] for r in index.report if r["status"] == "loaded"} != set(BENCHMARK_REVISIONS):
        raise ValueError("Incomplete benchmark exclusion coverage")
    if any(r["revision"] != BENCHMARK_REVISIONS[r["repo"]] for r in index.report):
        raise ValueError("Benchmark revision mismatch")
    # Protect the reserved raw-LM panels as well as official benchmarks.
    manifest_path = ROOT / config["pretraining_manifest"]
    if sha256(manifest_path) != config["pretraining_manifest_sha256"]:
        raise ValueError("Pretraining manifest changed")
    manifest = json.loads(manifest_path.read_text())
    protected = []
    for source in manifest["sources"].values():
        for split, entry in source["splits"].items():
            if split == "train" or "jsonl_path" not in entry:
                continue
            path = Path(entry["jsonl_path"])
            if sha256(path) != entry["jsonl_sha256"]:
                raise ValueError("Protected development panel changed")
            for row in read_jsonl(path):
                index.add(tokenizer.decode(row.get("tokens", row.get("input_ids")), skip_special_tokens=True))
            protected.append({"split": split, "sha256": entry["jsonl_sha256"]})
    # Exact normalized prompt matching also protects short benchmark questions.
    from datasets import load_dataset
    exact = set()
    for report in index.report:
        if report["repo"] == "Salesforce/wikitext":
            continue
        for split in report["loaded_splits"]:
            if report["repo"] == "ybisk/piqa":
                uri = f"hf://datasets/ybisk/piqa@{report['revision']}/plain_text/{split}-00000-of-00001.parquet"
                data = load_dataset("parquet", data_files={split: uri}, split=split)
            else:
                data = load_dataset(report["repo"], report["config"], revision=report["revision"], split=split)
            for row in data:
                text = next((row[key] for key in ("question", "goal", "sentence", "ctx") if key in row), "")
                if text:
                    exact.add(identity(text))
    return index, exact, {"benchmarks": index.report, "protected_panels": protected, "exact_prompts": len(exact)}


def prepare(config, output):
    from transformers import AutoTokenizer

    tokenizer_path = ROOT / config["tokenizer"]
    if sha256(tokenizer_path / "tokenizer.json") != config["tokenizer_sha256"]:
        raise ValueError("Frozen tokenizer mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if len(tokenizer) != 16384 or tokenizer.eos_token_id != 2 or tokenizer.pad_token_id != 0:
        raise ValueError("Unexpected tokenizer contract")
    rows, files, rejected = human_sources(config, output)
    index, exact, exclusions = exclusion_index(config, tokenizer)
    grouped, grouping = group_and_split(rows, config["seed"])
    if config.get("system_training"):
        grouped = [add_human_system(row, config["seed"]) for row in grouped]
    grouped += [row for split, count in config["procedural_per_family"].items()
                for row in generate(split, count, config["seed"])]
    grouped += [row for split, count in config.get("system_per_family", {}).items()
                for row in generate_system(split, count, config["seed"])]
    splits = defaultdict(list)
    for row in grouped:
        if row["id"] in config.get("excluded_ids", {}):
            rejected[f"{row['source']}/review_exclusion"] += 1
            continue
        # Shared system policies are task definitions, not benchmark answers.
        texts = [m["content"] for m in row["messages"] if m["role"] != "system"]
        if any(re.search(r"\bas an? (?:ai |artificial intelligence )(?:language )?model\b", text, re.I)
               for text in texts):
            rejected[f"{row['source']}/model_identity_boilerplate"] += 1
            continue
        if any(index.matches(text) or identity(text) in exact for text in texts):
            rejected[f"{row['source']}/protected_overlap"] += 1
            continue
        try:
            encoded = encode_example(row["messages"], tokenizer, max_length=config["max_length"],
                                     max_response_tokens=config["max_response_tokens"])
        except ValueError as error:
            rejected[f"{row['source']}/{error}"] += 1
            continue
        if row["source"] == "procedural" and not correct(row, row["messages"][-1]["content"]):
            raise ValueError("Procedural reference failed its verifier")
        check_encoded(encoded, vocab_size=len(tokenizer), eos_token_id=2, max_length=config["max_length"])
        splits[row["split"]].append({**row, **encoded})
    # Fixed seeded subset keeps procedural targets <=20% of one mixed epoch.
    train = splits["train"]
    human = [r for r in train if r["source"] != "procedural"]
    procedural = [r for r in train if r["source"] == "procedural"]
    procedural_groups = defaultdict(list)
    for row in procedural:
        procedural_groups[row["group"]].append(row)
    grouped_procedural = list(procedural_groups.values())
    random.Random(config["seed"]).shuffle(grouped_procedural)
    ceiling = int(sum(r["target_tokens"] for r in human) * config["procedural_target_share"] / (1 - config["procedural_target_share"]))
    selected, used = [], 0
    for group in grouped_procedural:
        cost = sum(row["target_tokens"] for row in group)
        if used + cost <= ceiling:
            selected.extend(group)
            used += cost
    rejected["procedural/mixture_cap"] = len(procedural) - len(selected)
    splits["train"] = human + selected
    splits["train_human"] = human
    stats, all_groups = {}, {}
    for split in ("train", "development", "confirmation"):
        groups = {r["group"] for r in splits[split]}
        if any(groups & other for other in all_groups.values()):
            raise ValueError("Duplicate group crosses a data split")
        all_groups[split] = groups
        if len(splits[split]) < config["minimum_examples"][split]:
            raise ValueError(f"Insufficient retained {split} examples: {len(splits[split])}")
    for split, examples in splits.items():
        examples.sort(key=lambda r: r["id"])
        path = output / f"{split}.jsonl"
        write_jsonl(path, examples)
        stats[split] = {"file": path.name, "sha256": sha256(path), "examples": len(examples),
                        "target_tokens": sum(r["target_tokens"] for r in examples),
                        "processed_tokens": sum(r["processed_tokens"] for r in examples),
                        "sources": dict(Counter(r["source"] for r in examples)),
                        "system_cases": dict(Counter(r.get("system_case", "none") for r in examples)),
                        "categories": dict(Counter(r["category"] for r in examples))}
    # All SFT text can be excluded by the 10B preparer; at minimum exclude heldouts.
    for name, names in (("heldout_exclusions", ("development", "confirmation")),
                        ("all_instruction_exclusions", ("train", "development", "confirmation"))):
        write_jsonl(output / f"{name}.jsonl", [{"id": r["id"], "split": split, "source": r["source"],
                    "text": "\n".join(m["content"] for m in r["messages"] if m["role"] != "system")} for split in names for r in splits[split]])
    audit = {"rejections": dict(rejected), "grouping": grouping, "exclusions": exclusions,
             "procedural_target_share": used / max(1, stats["train"]["target_tokens"]),
             "limitations": ["Metadata does not certify human authorship or factual correctness.",
                             "Lexical grouping and 13-word exclusions do not prove semantic decontamination.",
                             "Future 10B pretraining overlap is not yet audited; consume heldout_exclusions.jsonl there.",
                             "Existing pretraining can contain related Wikipedia/public instruction material."]}
    write_json(output / "audit.json", audit)
    # A reproducible sample for inspection, never sampled from sealed confirmation.
    sample = []
    for source in config["sources"]:
        pool = [r for r in splits["train"] if r["source"] == source]
        sample.extend(random.Random(config["seed"]).sample(pool, min(50, len(pool))))
    write_jsonl(output / "review_sample.jsonl", [{k: v for k, v in r.items() if k not in ("input_ids", "labels")} for r in sample])
    manifest = {"schema_version": 1, "status": "complete", "format_version": FORMAT_VERSION,
                "config": config, "tokenizer_sha256": config["tokenizer_sha256"],
                "sources": files, "splits": stats, "audit_sha256": sha256(output / "audit.json"),
                "preparation_code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "heldout_exclusions_sha256": sha256(output / "heldout_exclusions.jsonl")}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining/data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/posttraining/instruction_v4")
    args = parser.parse_args()
    with ValidationRun(ROOT / "artifacts/posttraining/preparation", output=args.output,
                       metadata={"mode": "data-preparation", "config": str(args.config)}) as run:
        run.claim_output()
        config = json.loads(args.config.read_text())
        write_json(run.output / "config.json", config)
        manifest = prepare(config, run.output)
        run.update(metrics={"data": manifest["splits"]}, manifest_sha256=sha256(run.output / "manifest.json"))
        print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
