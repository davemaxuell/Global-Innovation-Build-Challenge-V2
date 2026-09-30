"""Read-only screening against the FINISHED continuation corpus's dedup index.

Publish a derived instruction dataset with matching holdout groups removed.
This is exact-document/approximate near-document screening, not a semantic or
substring-contamination certificate. The underlying index includes prior raw
documents and reserved panels, so removals may be conservative.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from .common import ROOT, ValidationRun, identity, load_manifest, normalized, read_jsonl, sha256, write_json, write_jsonl


def screen(instruction_manifest, corpus_manifest, output):
    import lmdb
    from scglm.prepare_extension import matching_reason, sketch

    data = load_manifest(instruction_manifest)
    corpus = json.loads(corpus_manifest.read_text())
    if corpus["status"] != "complete":
        raise ValueError("Wait for corpus preparation to finish")
    if corpus["tokenizer"]["sha256"] != data["tokenizer_sha256"]:
        raise ValueError("Corpus/instruction tokenizer mismatch")
    code = ROOT / "src/scglm/prepare_extension.py"
    if sha256(code) != corpus["preparation_code_sha256"]:
        raise ValueError("Use the exact corpus preparer's recorded index implementation")
    config_path = corpus_manifest.parent / "preparation_config.json"
    if sha256(config_path) != corpus["configuration_sha256"]:
        raise ValueError("Corpus preparation configuration changed")
    contract = json.dumps(json.loads(config_path.read_text()), sort_keys=True)
    digest = hashlib.sha256((contract + corpus["preparation_code_sha256"] + corpus["parent_manifest_sha256"]).encode()).hexdigest()
    env = lmdb.open(str(corpus_manifest.parent / "dedup.lmdb"), readonly=True, create=False, lock=True)
    matches, rejected_groups = [], set()
    with env.begin() as txn:
        if txn.get(b"!contract") != digest.encode():
            raise ValueError("Corpus index identity mismatch")
        progress = json.loads(txn.get(b"!progress"))
        if not progress["old_done"] or any(not v["complete"] for v in progress["sources"].values()):
            raise ValueError("Corpus index is still being prepared")
        for split in ("development", "confirmation"):
            for row in read_jsonl(instruction_manifest.parent / data["splits"][split]["file"]):
                texts = [m["content"] for m in row["messages"] if m["role"] != "system"]
                if row.get("context"):
                    texts.append(row["context"])
                for text in texts:
                    text_hash = identity(text)
                    reason = "exact_document" if txn.get(b"H" + bytes.fromhex(text_hash)) else None
                    if reason is None and len(normalized(text).split()) >= 5:
                        reason = matching_reason(txn, {"digest": text_hash, "sketch": sketch(text), "url": ""})
                    if reason:
                        rejected_groups.add(row["group"])
                        matches.append({"id": row["id"], "group": row["group"], "split": split, "reason": reason})
                        break
    env.close()
    splits, rows_by_split = {}, {}
    for split, entry in data["splits"].items():
        rows = [r for r in read_jsonl(instruction_manifest.parent / entry["file"]) if r["group"] not in rejected_groups]
        if split in ("development", "confirmation"):
            if len(rows) < 500 or sum(r["source"] != "procedural" for r in rows) < 100:
                raise ValueError("Too few heldout examples remain; revise the instruction dataset before training")
            families = Counter(r["category"] for r in rows if r["source"] == "procedural")
            if len(families) != 5 or min(families.values()) < 100:
                raise ValueError("Insufficient independent procedural coverage")
            if data.get("config", {}).get("system_training"):
                coverage = Counter((r["category"], r.get("system_case")) for r in rows)
                from .procedural import FAMILIES
                from .system_data import SYSTEM_CASES
                if any(coverage[(family, case)] < 20 for family in FAMILIES for case in SYSTEM_CASES):
                    raise ValueError("Insufficient independent system instruction coverage")
        path = output / entry["file"]
        write_jsonl(path, rows)
        splits[split] = {**entry, "sha256": sha256(path), "examples": len(rows),
                         "target_tokens": sum(r["target_tokens"] for r in rows),
                         "processed_tokens": sum(r["processed_tokens"] for r in rows),
                         "sources": dict(Counter(r["source"] for r in rows)),
                         "system_cases": dict(Counter(r.get("system_case", "none") for r in rows)),
                         "categories": dict(Counter(r["category"] for r in rows))}
        rows_by_split[split] = rows
    report = {"corpus_manifest": str(corpus_manifest.resolve()), "corpus_manifest_sha256": sha256(corpus_manifest),
              "source_instruction_manifest_sha256": sha256(instruction_manifest), "index_contract": digest,
              "index_code_sha256": sha256(code), "rejected_groups": len(rejected_groups), "matches": matches,
              "method": corpus["deduplication"], "semantic_or_substring_decontamination_certified": False}
    write_json(output / "pretraining_overlap.json", report)
    for source in instruction_manifest.parent.glob("*.README.md"):
        shutil.copy2(source, output / source.name)
    shutil.copy2(instruction_manifest.parent / "audit.json", output / "preparation_audit.json")
    write_json(output / "audit.json", {"parent_preparation_audit_sha256": sha256(output / "preparation_audit.json"),
                                       "pretraining_overlap": report})
    for name, names in (("heldout_exclusions", ("development", "confirmation")),
                        ("all_instruction_exclusions", ("train", "development", "confirmation"))):
        write_jsonl(output / f"{name}.jsonl", [{"id": r["id"], "split": split, "source": r["source"],
                    "text": "\n".join(m["content"] for m in r["messages"] if m["role"] != "system")} for split in names for r in rows_by_split[split]])
    manifest = {**data, "splits": splits, "pretraining_overlap": report,
                "audit_sha256": sha256(output / "audit.json"),
                "heldout_exclusions_sha256": sha256(output / "heldout_exclusions.jsonl")}
    write_json(output / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/posttraining/instruction_v4/manifest.json")
    parser.add_argument("--corpus-manifest", type=Path, default=ROOT / "data/processed/extension_12b/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ValidationRun(ROOT / "artifacts/posttraining/overlap", output=args.output,
                       metadata={"mode": "pretraining-overlap-screen"}) as run:
        run.claim_output()
        result = screen(args.manifest, args.corpus_manifest, run.output)
        run.update(metrics=result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
