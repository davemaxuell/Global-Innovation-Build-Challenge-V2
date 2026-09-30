"""Independently re-encode and audit every prepared SFT example on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import (CHAT_TEMPLATE, ROOT, ValidationRun, check_encoded, encode_example,
                     identity, load_manifest, read_jsonl, sha256, write_json)
from .procedural import correct


def audit(manifest_path):
    from transformers import AutoTokenizer
    manifest = load_manifest(manifest_path)
    config = manifest["config"]
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / config["tokenizer"]), local_files_only=True)
    tokenizer.chat_template = CHAT_TEMPLATE
    if sha256(ROOT / config["tokenizer"] / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Frozen tokenizer mismatch")
    vocab_size = len(tokenizer)
    for source in manifest["sources"].values():
        if sha256(Path(source["cache_path"])) != source["sha256"]:
            raise ValueError("Downloaded source file changed")
    groups, prompts, counts = {}, {}, {}
    actual_rows = {}
    for split, entry in manifest["splits"].items():
        rows = read_jsonl(manifest_path.parent / entry["file"])
        ids, pairs = set(), defaultdict(list)
        targets = processed = 0
        for row in rows:
            if row["id"] in ids:
                raise ValueError("Repeated example ID")
            ids.add(row["id"])
            check_encoded(row, vocab_size=vocab_size, eos_token_id=2, max_length=config["max_length"])
            expected = encode_example(row["messages"], tokenizer, max_length=config["max_length"],
                                      max_response_tokens=config["max_response_tokens"])
            if any(expected[key] != row[key] for key in expected):
                raise ValueError("Retokenization or label mismatch")
            if tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=False) != row["input_ids"]:
                raise ValueError("Exported chat template differs from training")
            if row["source"] == "procedural":
                answer = row["messages"][-1]["content"]
                if not correct(row, answer) or correct(row, answer + " explanation"):
                    raise ValueError("Procedural verifier failed reference/adversarial check")
                if row.get("pair_id"):
                    pairs[row["pair_id"]].append(row)
            if row["source"] == "oasst1" and split.startswith("train") and row["provenance"]["original_split"] != "train":
                raise ValueError("Official source holdout entered training")
            targets += row["target_tokens"]
            processed += row["processed_tokens"]
        for pair in pairs.values():
            if len(pair) != 2 or len({r["group"] for r in pair}) != 1:
                raise ValueError("System policy pair is incomplete")
            left, right = pair
            if (left["messages"][-2] != right["messages"][-2]
                    or left["messages"][0] == right["messages"][0]
                    or correct(left, right["messages"][-1]["content"])
                    or correct(right, left["messages"][-1]["content"])):
                raise ValueError("System policy pair does not require different answers")
        if config.get("system_training") and split != "train_human":
            from .system_data import SYSTEM_CASES
            for mode in SYSTEM_CASES:
                labels = {r["messages"][-1]["content"] for r in rows
                          if r["category"] == "classification" and r.get("system_case") == mode}
                if labels != {"even", "odd", "EVEN", "ODD"}:
                    raise ValueError("System parity coverage is missing labels")
        if len(rows) != entry["examples"] or targets != entry["target_tokens"] or processed != entry["processed_tokens"]:
            raise ValueError("Manifest counts differ from data")
        if split != "train_human":
            groups[split] = {r["group"] for r in rows}
            prompts[split] = {identity(json.dumps([m for m in r["messages"][:-1] if m["role"] != "system"], sort_keys=True)) for r in rows}
        actual_rows[split] = rows
        counts[split] = {"examples": len(rows), "target_tokens": targets, "processed_tokens": processed,
                         "sources": dict(Counter(r["source"] for r in rows)),
                         "system_cases": dict(Counter(r.get("system_case", "none") for r in rows)),
                         "system_policy_pairs": len(pairs)}
    for left in groups:
        for right in groups:
            if left < right and (groups[left] & groups[right] or prompts[left] & prompts[right]):
                raise ValueError("Group or exact prompt leakage")
    human_ids = {r["id"] for r in actual_rows["train_human"]}
    if human_ids != {r["id"] for r in actual_rows["train"] if r["source"] != "procedural"}:
        raise ValueError("Human control differs from mixed arm's human subset")
    procedural_tokens = sum(r["target_tokens"] for r in actual_rows["train"] if r["source"] == "procedural")
    share = procedural_tokens / counts["train"]["target_tokens"]
    if share > config["procedural_target_share"]:
        raise ValueError("Procedural token share exceeds cap")
    return {"status": "passed", "splits": counts, "procedural_target_share": share,
            "manifest_sha256": sha256(manifest_path), "all_examples_reencoded": True,
            "chat_template_equivalence": True, "group_and_prompt_disjoint": True,
            "semantic_decontamination_certified": False,
            "completed_pretraining_document_screened": bool(manifest.get("pretraining_overlap"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/posttraining/instruction_v4/manifest.json")
    args = parser.parse_args()
    with ValidationRun(ROOT / "artifacts/posttraining/audits", output=None,
                       metadata={"mode": "data-audit", "manifest": str(args.manifest)}) as run:
        run.claim_output()
        result = audit(args.manifest)
        write_json(run.output / "audit.json", result)
        run.update(metrics=result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
