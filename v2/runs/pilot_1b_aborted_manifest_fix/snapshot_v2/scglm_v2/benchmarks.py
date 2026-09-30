"""Benchmark TRAIN-split texts for the targeted classifier and the V2 development panel.

Only training splits are read. Each example becomes natural text: the context or
question followed by its gold answer. A stable hash puts 20% of every set into a
held-out development panel that the classifier never sees. Official evaluation
splits are never loaded here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# (name, repo, config, split). Revisions are resolved and recorded at run time.
TRAIN_SETS = [
    ("arc_easy", "allenai/ai2_arc", "ARC-Easy", "train"),
    ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "train"),
    ("piqa", "ybisk/piqa", None, "train"),
    ("hellaswag", "Rowan/hellaswag", None, "train"),
    ("winogrande", "allenai/winogrande", "winogrande_xl", "train"),
    ("openbookqa", "allenai/openbookqa", "main", "train"),
    ("sciq", "allenai/sciq", None, "train"),
    ("commonsense_qa", "tau/commonsense_qa", None, "train"),
    ("social_i_qa", "allenai/social_i_qa", None, "train"),
]
PINNED = {  # V1's pinned revisions where they exist
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "ybisk/piqa": "ba2f7a4920f4968f04c174c53cc157c899e93501",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
}


def _choice(row: dict, key: str = "answerKey") -> str | None:
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    return texts[labels.index(row[key])] if row.get(key) in labels else None


def format_example(name: str, row: dict) -> dict | None:
    """Return {context, answer, text, choices, label} or None when unusable."""
    if name in ("arc_easy", "arc_challenge", "commonsense_qa"):
        answer = _choice(row)
        context, choices = row["question"], row["choices"]["text"]
        label = row["choices"]["label"].index(row["answerKey"]) if answer else None
    elif name == "openbookqa":
        answer = _choice(row)
        context, choices = row["question_stem"], row["choices"]["text"]
        label = row["choices"]["label"].index(row["answerKey"]) if answer else None
    elif name == "piqa":
        choices = [row["sol1"], row["sol2"]]
        label = int(row["label"]) if str(row.get("label")) in ("0", "1") else None
        context, answer = row["goal"], (choices[label] if label is not None else None)
    elif name == "hellaswag":
        choices = row["endings"]
        label = int(row["label"]) if str(row.get("label", "")).isdigit() else None
        context, answer = row["ctx"], (choices[label] if label is not None else None)
    elif name == "winogrande":
        choices = [row["option1"], row["option2"]]
        label = int(row["answer"]) - 1 if str(row.get("answer")) in ("1", "2") else None
        if label is None or "_" not in row["sentence"]:
            return None
        before, after = row["sentence"].split("_", 1)
        return {"context": before.rstrip(), "answer": choices[label] + after, "choices": choices,
                "label": label, "text": before + choices[label] + after}
    elif name == "sciq":
        choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
        label, answer = 3, row["correct_answer"]
        context = (row.get("support") or "").strip() + "\n" + row["question"] if row.get("support") else row["question"]
    elif name == "social_i_qa":
        choices = [row["answerA"], row["answerB"], row["answerC"]]
        label = int(row["label"]) - 1 if str(row.get("label")) in ("1", "2", "3") else None
        context = f"{row['context']} {row['question']}"
        answer = choices[label] if label is not None else None
    else:
        raise ValueError(name)
    if not answer or not context:
        return None
    return {"context": context.strip(), "answer": answer.strip(), "choices": choices, "label": label,
            "text": f"{context.strip()} {answer.strip()}"}


def is_development(name: str, text: str, fraction: float = 0.2) -> bool:
    bucket = int(hashlib.sha256(f"{name}\n{text}".encode()).hexdigest()[:8], 16) % 1000
    return bucket < int(fraction * 1000)


def load_all(output: Path, fraction: float = 0.2) -> dict:
    """Write positives.jsonl (classifier) and development.jsonl (held-out proxy panel)."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    output.mkdir(parents=True, exist_ok=True)
    report, api = {}, HfApi()
    with (output / "positives.jsonl").open("w", encoding="utf-8") as pos, \
            (output / "development.jsonl").open("w", encoding="utf-8") as dev:
        for name, repo, config, split in TRAIN_SETS:
            entry = {"repo": repo, "config": config, "split": split}
            try:
                revision = PINNED.get(repo) or api.dataset_info(repo).sha
                entry["revision"] = revision
                if repo == "ybisk/piqa":
                    uri = f"hf://datasets/{repo}@{revision}/plain_text/{split}-00000-of-00001.parquet"
                    data = load_dataset("parquet", data_files={split: [uri]}, split=split)
                else:
                    data = load_dataset(repo, config, revision=revision, split=split)
                counts = {"positives": 0, "development": 0, "unusable": 0}
                for row in data:
                    item = format_example(name, row)
                    if item is None:
                        counts["unusable"] += 1
                        continue
                    record = {"set": name, **item}
                    if is_development(name, item["text"], fraction):
                        dev.write(json.dumps(record, ensure_ascii=False) + "\n"); counts["development"] += 1
                    else:
                        pos.write(json.dumps(record, ensure_ascii=False) + "\n"); counts["positives"] += 1
                entry.update(status="loaded", **counts)
            except Exception as exc:  # recorded as a coverage gap, never silently ignored
                entry.update(status="unavailable", error=f"{type(exc).__name__}: {str(exc)[:400]}")
            report[name] = entry
            print(json.dumps({"benchmark_train_set": name, **entry}), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse
    from .common import ROOT, load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="v2/configs/prep.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    load_all(ROOT / cfg["output_dir"] / "benchmarks", cfg["targeted"]["development_holdout_fraction"])
