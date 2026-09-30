"""Development-only evaluation for V2 selection (never official eval splits).

* Bits per byte on each V2 ``development`` panel, decoded to text so models with
  different tokenizers are compared on identical bytes.
* Zero-shot multiple choice on the held-out 20% of benchmark TRAIN splits, scored
  like lm-evaluation-harness: summed continuation log-likelihood after a " "
  delimiter (acc) and the same divided by the choice's character length
  (acc_norm). WinoGrande uses partial scoring: the shared suffix is scored after
  each option-filled prefix.

Run: PYTHONPATH=src:v2/src python -m scglm_v2.devevaluate --model <export dir> --out <json>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import ROOT, write_json

OFFICIAL_PROXY = ("arc_easy", "piqa", "hellaswag", "winogrande")


def load(model_dir: str, device: str):
    from transformers import PreTrainedTokenizerFast
    from scglm.model import load_model
    model = load_model(model_dir).to(device).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    return model, tokenizer


@torch.no_grad()
def continuation_logprob(model, device, context_ids: list[int], cont_ids: list[int], ctx_len: int = 1024) -> float:
    ids = (context_ids + cont_ids)[-(ctx_len + 1):]
    n = min(len(cont_ids), len(ids) - 1)
    x = torch.tensor([ids[:-1]], device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        logits = model(input_ids=x, use_cache=False).logits.float()
    logp = F.log_softmax(logits[0, -n:], dim=-1)
    target = torch.tensor(ids[-n:], device=device)
    return float(logp.gather(1, target[:, None]).sum())


def multiple_choice(model, tokenizer, device, path: Path, per_set: int) -> dict:
    eos = tokenizer.eos_token_id
    counts, results = defaultdict(int), defaultdict(lambda: {"acc": 0, "acc_norm": 0, "n": 0})
    for line in path.open():
        item = json.loads(line)
        name = item["set"]
        if counts[name] >= per_set or item["label"] is None:
            continue
        counts[name] += 1
        scores, norms = [], []
        if name == "winogrande":
            prefix, suffix = item["context"], item["answer"][len(item["choices"][item["label"]]):]
            for choice in item["choices"]:
                ctx = [eos] + tokenizer.encode(prefix + choice, add_special_tokens=False)
                cont = tokenizer.encode(suffix, add_special_tokens=False)
                s = continuation_logprob(model, device, ctx, cont)
                scores.append(s); norms.append(s)
        else:
            ctx = [eos] + tokenizer.encode(item["context"], add_special_tokens=False)
            for choice in item["choices"]:
                cont = tokenizer.encode(" " + choice, add_special_tokens=False)
                s = continuation_logprob(model, device, ctx, cont)
                scores.append(s); norms.append(s / max(1, len(choice)))
        r = results[name]
        r["n"] += 1
        r["acc"] += int(max(range(len(scores)), key=scores.__getitem__) == item["label"])
        r["acc_norm"] += int(max(range(len(norms)), key=norms.__getitem__) == item["label"])
    out = {k: {"acc": v["acc"] / v["n"], "acc_norm": v["acc_norm"] / v["n"], "n": v["n"]} for k, v in results.items()}
    proxy = [out[k]["acc"] for k in OFFICIAL_PROXY if k in out]
    out["_official_proxy_mean_acc"] = sum(proxy) / len(proxy) if proxy else None
    return out


@torch.no_grad()
def bits_per_byte(model, tokenizer, device, texts: list[str], ctx_len: int = 1024) -> float:
    total_nll, total_bytes = 0.0, 0
    eos = tokenizer.eos_token_id
    for text in texts:
        ids = [eos] + tokenizer.encode(text, add_special_tokens=False)
        # Non-overlapping windows: every token after the leading EOS is scored exactly once.
        for start in range(0, len(ids) - 1, ctx_len):
            chunk = ids[start:start + ctx_len + 1]
            x = torch.tensor([chunk[:-1]], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = model(input_ids=x, use_cache=False).logits.float()
            total_nll += float(F.cross_entropy(logits[0], torch.tensor(chunk[1:], device=device), reduction="sum"))
        total_bytes += len(text.encode("utf-8"))
    return total_nll / math.log(2) / total_bytes


def panel_texts(manifest_path: Path, docs: int) -> dict[str, list[str]]:
    from tokenizers import Tokenizer
    manifest = json.loads(manifest_path.read_text())
    tok = Tokenizer.from_file(manifest["tokenizer"]["path"])
    out = {}
    for bucket, spec in manifest["sources"].items():
        rows = [json.loads(l) for l in open(spec["panels"]["development"]["jsonl_path"])][:docs]
        out[bucket] = [tok.decode([t for t in r["tokens"] if t > 3]) for r in rows]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=str(ROOT / "v2/data/rich_v1/manifest.json"))
    p.add_argument("--benchmarks", default=str(ROOT / "v2/data/rich_v1/benchmarks/development.jsonl"))
    p.add_argument("--per-set", type=int, default=1000)
    p.add_argument("--panel-docs", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    model, tokenizer = load(args.model, args.device)
    texts = panel_texts(Path(args.manifest), args.panel_docs)
    bpb = {k: bits_per_byte(model, tokenizer, args.device, v) for k, v in texts.items()}
    weights = json.loads(Path(args.manifest).read_text())["planned_training_shares"]
    result = {"model": args.model, "bits_per_byte": bpb,
              "bits_per_byte_mixture": sum(weights[k] * v for k, v in bpb.items()) / sum(weights[k] for k in bpb),
              "multiple_choice": multiple_choice(model, tokenizer, args.device, Path(args.benchmarks), args.per_set),
              "note": "Development data only: V2 development panels and held-out benchmark TRAIN-split items."}
    write_json(Path(args.out), result)
    print(json.dumps({"bpb_mixture": result["bits_per_byte_mixture"],
                      "proxy_acc": result["multiple_choice"]["_official_proxy_mean_acc"]}))


if __name__ == "__main__":
    main()
