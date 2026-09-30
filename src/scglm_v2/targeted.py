"""Benchmark-targeted document scorer (BETR-inspired, fastText variant).

Positives: benchmark TRAIN-split texts (80% split; at most ``per_set_cap`` per set
so no single benchmark dominates). Negatives: random passages from the Edu+DCLM
pool with the same word-length distribution, so length cannot separate classes.
A document's score is the mean positive probability over ~60-word windows. The
threshold is the character-weighted quantile giving the configured top fraction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re

import numpy as np

from .common import ROOT, event, load_config, sha256_file, write_json
from .sources import reused_raw_paths, sample_jsonl

_WORD = re.compile(r"\w+|[^\w\s]")
WINDOW = 60
PER_SET_CAP = 8000


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def windows(text: str, size: int = WINDOW, limit: int = 64) -> list[str]:
    words = normalize(text).split()
    if not words:
        return []
    spans = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    if len(spans) > limit:  # bounded cost on very long documents; evenly spaced
        step = len(spans) / limit
        spans = [spans[int(i * step)] for i in range(limit)]
    return spans


class Scorer:
    def __init__(self, model_path: str):
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None
        self.model = fasttext.load_model(model_path)

    def score(self, text: str) -> float:
        spans = windows(text)
        if not spans:
            return 0.0
        labels, probs = self.model.predict(spans, k=2)
        values = [p[list(l).index("__label__pos")] if "__label__pos" in l else 0.0 for l, p in zip(labels, probs)]
        return float(np.mean(values))


def main() -> None:
    import fasttext
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="v2/configs/prep.json")
    args = parser.parse_args()
    config = load_config(args.config)
    out = ROOT / config["output_dir"] / "targeted"
    out.mkdir(parents=True, exist_ok=True)
    log = ROOT / "v2/phases/preparation/events.jsonl"
    rng = random.Random(config["seed"])

    by_set = defaultdict(list)
    for line in (ROOT / config["output_dir"] / "benchmarks/positives.jsonl").open():
        row = json.loads(line)
        by_set[row["set"]].append(row["text"])
    positives = []
    for name, texts in sorted(by_set.items()):
        rng.shuffle(texts)
        positives.extend(texts[:PER_SET_CAP])
    lengths = [max(8, len(normalize(t).split())) for t in positives]

    pool_paths = [p for s in config["targeted"]["pool_sources"] for p in reused_raw_paths(config, s)]
    negatives, per_path = [], len(positives) // len(pool_paths) + 1
    for k, path in enumerate(pool_paths):
        for doc in sample_jsonl(path, per_path, config["seed"] + 101 + k):
            words = normalize(doc["text"]).split()
            n = rng.choice(lengths)
            start = rng.randrange(max(1, len(words) - n))
            negatives.append(" ".join(words[start:start + n]))
    rows = [("__label__pos", normalize(t)) for t in positives] + [("__label__neg", t) for t in negatives]
    rng.shuffle(rows)
    cut = len(rows) // 10
    for name, part in (("valid", rows[:cut]), ("train", rows[cut:])):
        with (out / f"{name}.txt").open("w", encoding="utf-8") as f:
            for label, text in part:
                f.write(f"{label} {text}\n")
    model = fasttext.train_supervised(str(out / "train.txt"), wordNgrams=2, dim=64, epoch=5, lr=0.2,
                                      minCount=2, thread=32, seed=config["seed"] % 2**31, verbose=0)
    model.save_model(str(out / "classifier.bin"))

    # Validation AUC on the held-out 10%.
    labels = [l for l, _ in rows[:cut]]
    pred_labels, pred_probs = model.predict([t for _, t in rows[:cut]], k=2)
    scores = np.array([p[list(l).index("__label__pos")] for l, p in zip(pred_labels, pred_probs)])
    y = np.array([l == "__label__pos" for l in labels])
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    auc = (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())

    # Threshold: character-weighted quantile over a fresh pool sample.
    scorer, sample = Scorer(str(out / "classifier.bin")), []
    for k, path in enumerate(pool_paths):
        for doc in sample_jsonl(path, 20000, config["seed"] + 303 + k):
            sample.append((scorer.score(doc["text"]), len(doc["text"])))
    sample.sort(key=lambda x: -x[0])
    total, running, threshold = sum(c for _, c in sample), 0, sample[0][0]
    for score, chars in sample:
        running += chars
        threshold = score
        if running >= config["targeted"]["top_fraction"] * total:
            break
    result = {"positives": len(positives), "positives_per_set": {k: min(len(v), PER_SET_CAP) for k, v in by_set.items()},
              "negatives": len(negatives), "validation_auc": float(auc), "threshold": float(threshold),
              "top_fraction_chars": config["targeted"]["top_fraction"], "pool_sample_documents": len(sample),
              "classifier_sha256": sha256_file(out / "classifier.bin"),
              "window_words": WINDOW, "document_score": "mean positive probability over windows"}
    write_json(out / "classifier.json", result)
    # Examples for manual inspection of what the slice selects.
    top = []
    for k, path in enumerate(pool_paths):
        for doc in sample_jsonl(path, 400, config["seed"] + 909 + k):
            s = scorer.score(doc["text"])
            if s >= threshold:
                top.append({"score": s, "url": doc.get("url"), "text": doc["text"][:500]})
    write_json(out / "selected_examples.json", sorted(top, key=lambda r: -r["score"])[:40])
    event(log, "targeted_classifier", **{k: v for k, v in result.items() if k != "positives_per_set"})


if __name__ == "__main__":
    main()
