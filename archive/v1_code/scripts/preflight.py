"""Validate frozen corpus contracts without computing model/benchmark scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/main/manifest.json")
    parser.add_argument("--output", default="artifacts/data_preflight.json")
    args = parser.parse_args()
    path = Path(args.manifest).resolve()
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "complete", "Corpus is not frozen"
    tokenizer_path = Path(manifest["tokenizer"]["path"])
    assert digest(tokenizer_path) == manifest["tokenizer"]["sha256"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    assert tokenizer.get_vocab_size() == 16384
    for name, expected in {"pad": 0, "bos": 1, "eos": 2, "unk": 3}.items():
        assert tokenizer.token_to_id(f"<|{name}|>") == expected
    seen_development, report = set(), {}
    for source, details in manifest["sources"].items():
        report[source] = {}
        for name, split in details["splits"].items():
            assert digest(split["path"]) == split["sha256"], f"Corrupt {source}/{name}"
            assert digest(split["index_path"]) == split["index_sha256"]
            array = np.memmap(split["path"], mode="r", dtype="<u2")
            assert len(array) == split["tokens"]
            assert int(array.max()) < tokenizer.get_vocab_size()
            summary = {"stored_tokens": len(array), "documents": split["documents"]}
            if name != "train":
                assert digest(split["jsonl_path"]) == split["jsonl_sha256"]
                counts, capped = 0, 0
                with open(split["jsonl_path"]) as stream:
                    for line in stream:
                        record = json.loads(line)
                        identity = record["id"].split(":", 1)[-1]
                        assert identity not in seen_development, "Development split overlap"
                        seen_development.add(identity)
                        tokens = record.get("tokens", record.get("input_ids"))
                        assert len(tokens) > 1 and max(tokens) < 16384
                        counts += 1
                        capped += min(len(tokens)-1, 1024)
                assert counts == split["documents"]
                if name.startswith("controller_"):
                    assert counts >= 512 and capped >= 131072, f"Insufficient controller evidence: {source}/{name}"
                summary["scorable_targets_at_controller_cap"] = capped
            report[source][name] = summary
    # Normalized content identity ignores the source prefix, catching cross-source
    # exact overlaps as well as within-source split errors.
    for source, details in manifest["sources"].items():
        split = details["splits"]["train"]
        with open(split["index_path"]) as stream:
            for line in stream:
                record = json.loads(line)
                identity = record.get("content_sha256", record["id"].split(":", 1)[-1])
                assert identity not in seen_development, f"Training/development overlap: {source}"
    exclusions = json.loads(Path(manifest["exclusion_report"]).read_text())
    assert len(exclusions["benchmarks"]) == 5
    assert all(item["status"] == "loaded" for item in exclusions["benchmarks"]), "Incomplete benchmark exclusions"
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "manifest_sha256": digest(path), "tokenizer_sha256": digest(tokenizer_path),
              "sources": report, "exclusion_report": exclusions,
              "scope": "Hashes, token bounds, exact document split disjointness, controller sample size. No model scores or semantic decontamination claim."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "output": args.output, "sources": report}, indent=2))


if __name__ == "__main__":
    main()
