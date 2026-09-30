"""Derive immutable post-training inputs with two-way evaluation protection."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import tempfile

import lmdb
from transformers import AutoTokenizer

from scglm.prepare_data import ExclusionIndex
from scglm.prepare_extension import matching_reason, sketch
from scglm_post.common import identity, load_manifest, normalized
from scglm_post.overlap import screen
from scglm_post.audit import audit
from .battery import generate, correct, FAMILIES
from .common import ROOT, read, read_jsonl, resolve, sha256, write_json, write_jsonl, load_corpora, digest


def rebalance(rows, fraction):
    human = [r for r in rows if r["source"] != "procedural"]
    grouped = defaultdict(list)
    for row in rows:
        if row["source"] == "procedural":
            grouped[row["group"]].append(row)
    budget = int(sum(r["target_tokens"] for r in human) * fraction / (1-fraction))
    retained, used = list(human), 0
    for key in sorted(grouped):
        batch = grouped[key]; cost = sum(r["target_tokens"] for r in batch)
        if used+cost <= budget:
            retained.extend(batch); used += cost
    return sorted(retained, key=lambda r: r["id"]), sorted(human, key=lambda r: r["id"])


def prepare(config, destination):
    destination = resolve(destination)
    if destination.exists():
        raise ValueError("Use a new immutable dataset output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_path = resolve(config["instruction_manifest"])
    original = load_manifest(original_path)
    if sha256(original_path) != config["instruction_manifest_sha256"]:
        raise ValueError("Instruction source snapshot changed")
    corpora = load_corpora(config)
    new_corpus = resolve(config["corpora"]["continuation"]["manifest"])
    with tempfile.TemporaryDirectory(prefix=".pipeline-data-", dir=destination.parent) as temporary:
        out = Path(temporary)
        screen(original_path, new_corpus, out)
        manifest = load_manifest(out / "manifest.json")
        tokenizer = AutoTokenizer.from_pretrained(str(resolve(manifest["config"]["tokenizer"])), local_files_only=True)
        protected = ExclusionIndex(); exact = set(); panel_records = []
        for name, (data, spec) in corpora.items():
            for source, entry in data["sources"].items():
                for split, panel in entry["splits"].items():
                    if split == "train" or "jsonl_path" not in panel:
                        continue
                    path = resolve(panel["jsonl_path"])
                    if sha256(path) != panel["jsonl_sha256"]:
                        raise ValueError("Protected raw panel changed")
                    for row in read_jsonl(path):
                        text = tokenizer.decode(row["tokens"], skip_special_tokens=True)
                        protected.add(text); exact.add(identity(text))
                    panel_records.append({"corpus": name, "source": source, "split": split,
                                          "path": str(path), "sha256": panel["jsonl_sha256"]})
        battery_files = {}
        # Existing screen() verified the read-only index contract, including parent data.
        env = lmdb.open(str(new_corpus.parent / "dedup.lmdb"), readonly=True, create=False, lock=True)
        with env.begin() as txn:
            for phase in ("development", "confirmation"):
                rows = generate(phase, config["battery_per_family"], config["seed"])
                for row in rows:
                    if not correct(row, row["answer"]) or correct(row, row["answer"]+" unrelated claim"):
                        raise ValueError("Battery verifier failed its reference/adversarial check")
                    for text in (row["prompt"], row["context"]):
                        key = identity(text)
                        if matching_reason(txn, {"digest": key, "sketch": sketch(text), "url": ""}):
                            raise ValueError("Fresh grounded battery overlaps pretraining; revise before any scoring")
                        protected.add(text); exact.add(key)
                    prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
                    if len(prompt_ids) > 700:
                        raise ValueError("Battery prompt exceeds generation context allowance")
                path = out / f"battery_{phase}.jsonl"
                write_jsonl(path, rows)
                battery_files[phase] = {"file": path.name, "sha256": sha256(path), "examples": len(rows),
                    "families": dict(Counter(r["family"] for r in rows)),
                    "groups": dict(Counter(r["family"] for r in {r["group"]: r for r in rows}.values()))}
        env.close()
        rows_by_split = {split: read_jsonl(out / entry["file"]) for split, entry in manifest["splits"].items()}
        rejected = set()
        for row in rows_by_split["train"]:
            texts = [m["content"] for m in row["messages"] if m["role"] != "system"]
            if row.get("context"):
                texts.append(row["context"])
            if any(identity(text) in exact or protected.matches(text) for text in texts):
                rejected.add(row["group"])
        filtered = [r for r in rows_by_split["train"] if r["group"] not in rejected]
        rows_by_split["train"], rows_by_split["train_human"] = rebalance(filtered, manifest["config"]["procedural_target_share"])
        for split, rows in rows_by_split.items():
            if len(rows) < manifest["config"]["minimum_examples"].get(split, 1):
                raise ValueError(f"Insufficient data after protection: {split}")
            path = out / manifest["splits"][split]["file"]
            write_jsonl(path, rows)
            manifest["splits"][split].update(sha256=sha256(path), examples=len(rows),
                target_tokens=sum(r["target_tokens"] for r in rows), processed_tokens=sum(r["processed_tokens"] for r in rows),
                sources=dict(Counter(r["source"] for r in rows)), categories=dict(Counter(r["category"] for r in rows)),
                system_cases=dict(Counter(r.get("system_case", "none") for r in rows)))
        for name, splits in (("heldout_exclusions", ("development", "confirmation")),
                             ("all_instruction_exclusions", ("train", "development", "confirmation"))):
            write_jsonl(out / f"{name}.jsonl", [{"id": r["id"], "split": split, "source": r["source"],
                "text": "\n".join(m["content"] for m in r["messages"] if m["role"] != "system")}
                for split in splits for r in rows_by_split[split]])
        report = {"protected_raw_panels": panel_records, "rejected_sft_train_groups": len(rejected),
                  "battery": battery_files, "pipeline_config_sha256": digest(config),
                  "limits": "13-word exact overlap plus whole-document approximate MinHash. Semantic or arbitrary substring independence is not certified."}
        write_json(out / "pipeline_protection.json", report)
        manifest.update(pipeline_protection=report, battery=battery_files,
                        heldout_exclusions_sha256=sha256(out / "heldout_exclusions.jsonl"))
        write_json(out / "manifest.json", manifest)
        checked = audit(out / "manifest.json")
        write_json(out / "pipeline_audit.json", checked)
        # Publish all files together; an interruption never exposes a partial dataset.
        Path(temporary).rename(destination)
    return {"manifest": str(destination / "manifest.json"), "sha256": sha256(destination / "manifest.json"),
            "audit": checked, "rejected_sft_train_groups": len(rejected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(read(args.config), args.output), indent=2))


if __name__ == "__main__":
    main()
