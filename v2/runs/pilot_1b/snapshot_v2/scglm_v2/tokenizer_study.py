"""Train candidate English byte-level BPE tokenizers and compare with V1's.

Training text follows the planned mixture (targeted share split over edu/dclm).
Held-out text is a disjoint sample of every source plus benchmark development
texts (held-out TRAIN-split items). The metric is UTF-8 bytes per token.
Decision rule (registered before running): adopt the best new candidate only if
it beats V1 by >= 2% on both held-out mixture and benchmark text; among passing
candidates prefer the smaller vocabulary unless the larger is a further >= 2%
better on the mixture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .common import ROOT, event, load_config, sha256_file, write_json
from .sources import iter_books, iter_stackexchange, reused_raw_paths, sample_jsonl

SPECIAL = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
EVAL_DOCS = 1500


def train_mixture_weights(config: dict) -> dict[str, float]:
    shares = dict(config["planned_training_shares"])
    targeted = shares.pop("targeted")
    pool = config["targeted"]["pool_sources"]
    for name in pool:
        shares[name] += targeted / len(pool)
    return shares


def gather(config: dict, source: str, chars: int, seed: int, exclude: set | None = None):
    """Return (texts, ids) totalling about ``chars`` characters."""
    texts, ids, total = [], set(), 0
    if source in ("edu", "dclm", "wiki"):
        paths = reused_raw_paths(config, source)
        per_path = chars // len(paths)
        for k, path in enumerate(paths):
            got = 0
            for doc in sample_jsonl(path, 10**9, seed + k, exclude):
                texts.append(doc["text"]); ids.add(doc["id"]); got += len(doc["text"])
                if got >= per_path:
                    break
            total += got
        return texts, ids
    iterator = iter_stackexchange(config) if source == "stackexchange" else iter_books(config)
    for n, doc in enumerate(iterator):
        # Deterministic thinning keeps both samples spread across the available files.
        if exclude is not None and doc["id"] in exclude:
            continue
        if hash((seed, n)) % 4:
            continue
        texts.append(doc["text"]); ids.add(doc["id"]); total += len(doc["text"])
        if total >= chars:
            break
    return texts, ids


def build_tokenizer(vocab: int, texts_iter) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=SPECIAL[3]))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts_iter, trainer=trainer)
    if tok.get_vocab_size() != vocab:
        raise RuntimeError(f"vocabulary {tok.get_vocab_size()} != {vocab}")
    for i, s in enumerate(SPECIAL):
        if tok.token_to_id(s) != i:
            raise RuntimeError(f"special token {s} is not id {i}")
    return tok


def bytes_per_token(tok: Tokenizer, texts: list[str]) -> float:
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    n_tokens = sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
    return n_bytes / n_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="v2/configs/prep.json")
    args = parser.parse_args()
    config = load_config(args.config)
    study = config["tokenizer_study"]
    out = ROOT / config["output_dir"] / "tokenizer_study"
    out.mkdir(parents=True, exist_ok=True)
    log = ROOT / "v2/phases/preparation/events.jsonl"
    weights = train_mixture_weights(config)
    event(log, "tokenizer_study_start", weights=weights, training_chars=study["training_chars"])

    train_texts, held = [], {}
    for source, share in weights.items():
        texts, ids = gather(config, source, int(study["training_chars"] * share), config["seed"])
        train_texts.extend(texts)
        # Held-out sample: disjoint by id; a fixed document count per source.
        eval_texts = []
        if source in ("edu", "dclm", "wiki"):
            for path in reused_raw_paths(config, source):
                for doc in sample_jsonl(path, EVAL_DOCS // len(reused_raw_paths(config, source)),
                                        config["seed"] + 7, exclude_ids=ids):
                    eval_texts.append(doc["text"])
        else:
            iterator = iter_stackexchange(config) if source == "stackexchange" else iter_books(config)
            for doc in iterator:
                if doc["id"] not in ids:
                    eval_texts.append(doc["text"])
                    if len(eval_texts) >= EVAL_DOCS:
                        break
        held[source] = eval_texts
        event(log, "tokenizer_sample", source=source, train_docs=len(texts),
              train_chars=sum(map(len, texts)), heldout_docs=len(eval_texts))
    bench = [json.loads(l)["text"] for l in (ROOT / config["output_dir"] / "benchmarks/development.jsonl").open()]

    candidates = {"v1": Tokenizer.from_file(str(ROOT / study["baseline_tokenizer"]))}
    for vocab in study["vocab_candidates"]:
        path = out / f"v{vocab}" / "tokenizer.json"
        if path.exists():
            candidates[f"v{vocab}"] = Tokenizer.from_file(str(path))
            continue
        tok = build_tokenizer(vocab, iter(train_texts))
        path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(path))
        candidates[f"v{vocab}"] = tok
        event(log, "tokenizer_trained", vocab=vocab, sha256=sha256_file(path))

    results = {}
    for name, tok in candidates.items():
        per_source = {s: bytes_per_token(tok, t) for s, t in held.items()}
        mixture = sum(weights[s] * per_source[s] for s in per_source) / sum(weights[s] for s in per_source)
        results[name] = {"per_source": per_source, "mixture": mixture, "benchmark": bytes_per_token(tok, bench),
                         "vocab": tok.get_vocab_size()}
    base = results["v1"]
    passing = []
    for name in (f"v{v}" for v in sorted(study["vocab_candidates"])):
        r = results[name]
        r["gain_mixture"] = r["mixture"] / base["mixture"] - 1
        r["gain_benchmark"] = r["benchmark"] / base["benchmark"] - 1
        if min(r["gain_mixture"], r["gain_benchmark"]) >= study["adopt_if_bytes_per_token_gain_at_least"]:
            passing.append(name)
    chosen = "v1"
    if passing:
        chosen = passing[0]
        for name in passing[1:]:
            if results[name]["mixture"] / results[chosen]["mixture"] - 1 >= study["adopt_if_bytes_per_token_gain_at_least"]:
                chosen = name
    decision = {"results": results, "passing": passing, "chosen": chosen,
                "chosen_path": str(ROOT / study["baseline_tokenizer"]) if chosen == "v1" else str(out / chosen / "tokenizer.json"),
                "benchmark_texts": len(bench), "heldout_docs": {s: len(t) for s, t in held.items()}}
    decision["chosen_sha256"] = sha256_file(Path(decision["chosen_path"]))
    write_json(out / "decision.json", decision)
    event(log, "tokenizer_decision", chosen=chosen, passing=passing,
          summary={k: {"mixture": round(v["mixture"], 4), "benchmark": round(v["benchmark"], 4)} for k, v in results.items()})


if __name__ == "__main__":
    main()
