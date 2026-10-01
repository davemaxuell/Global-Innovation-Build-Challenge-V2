"""Freeze the task-tuning endpoint choice from development data only (v2/phases/task_tune).

The parent (V2) and every candidate export are scored on:

* the held-out 20% of benchmark TRAIN items, formatted and scored exactly as the pinned
  harness scores (``task_format``; these items were never trained on), and
* bits/byte on the corpus manifest's ``selection`` panels (``devevaluate.bits_per_byte``).

Rule, registered before training (v2/phases/task_tune/registration.json): among candidates
whose Wikipedia selection bits/byte is at most ``max_wiki_bpb_regression`` worse than the
parent's, take the highest ARC-Easy/PIQA/HellaSwag/WinoGrande mean acc; select it only if
that mean is at least ``min_proxy_gain`` above the parent's, otherwise keep the parent.
The record has the fields scripts/run_final_evaluation.py requires.

Run: PYTHONPATH=src python -m scglm_v2.task_select --config v2/phases/task_tune/config.json \
       --out v2/phases/task_tune/selection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .common import ROOT, sha256_file, write_json
from .devevaluate import bits_per_byte, load, panel_texts
from .task_format import score_items

TOLERANCE = 1e-12


def candidates(cfg: dict) -> list[Path]:
    found = []
    for arm in cfg["arms"].values():
        for epoch in range(1, cfg["epochs"] + 1):
            export = ROOT / arm["run_dir"] / f"epoch_{epoch}" / "export"
            if (export / "model.safetensors").exists():
                found.append(export)
    return found


def score(export: Path, items: list[dict], texts: dict, device: str) -> dict:
    model, tok = load(str(export), device)
    started = time.time()
    mc = score_items(model, items, device)
    bpb = {k: bits_per_byte(model, tok, device, v) for k, v in texts.items()}
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    # "run" is the directory whose export/ this is; scripts/run_final_evaluation.py matches on it.
    return {"run": str(export.parent), "export": str(export), "export_model_sha256": sha256_file(export / "model.safetensors"),
            "multiple_choice": mc, "proxy_mean_acc": mc["_proxy_mean_acc"],
            "bits_per_byte_selection_panels": bpb, "seconds": time.time() - started}


def decide(parent: dict, scored: list[dict], rule: dict) -> tuple[dict, str, dict | None]:
    wiki = parent["bits_per_byte_selection_panels"]["wiki"]
    eligible = [s for s in scored
                if s["bits_per_byte_selection_panels"]["wiki"] - wiki <= rule["max_wiki_bpb_regression"] + TOLERANCE]
    if not eligible:
        return parent, "parent: no candidate passed the Wikipedia bits/byte guard", None
    best = max(eligible, key=lambda s: s["proxy_mean_acc"])
    gain = best["proxy_mean_acc"] - parent["proxy_mean_acc"]
    change = best["bits_per_byte_selection_panels"]["wiki"] - wiki
    if gain >= rule["min_proxy_gain"] - TOLERANCE:
        return best, f"candidate: proxy gain {gain:+.4f} >= {rule['min_proxy_gain']}, wiki bpb change {change:+.4f}", best
    return parent, f"parent: best eligible proxy gain {gain:+.4f} < {rule['min_proxy_gain']}", best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; a selection record is frozen once written")
    cfg = json.loads((ROOT / args.config).read_text())
    rule = cfg["selection_rule"]
    data_dir = ROOT / cfg["data_dir"]
    report = json.loads((data_dir / "report.json").read_text())
    dev_path = data_dir / "development.jsonl"
    if sha256_file(dev_path) != report["files"]["development.jsonl"]:
        raise ValueError("Development items changed after preparation")
    items = [json.loads(line) for line in dev_path.open()]
    manifest = ROOT / cfg["replay_manifest"]
    texts = panel_texts(manifest, rule["panel_docs"], split="selection")
    parent = score(ROOT / cfg["parent_export"], items, texts, args.device)
    print(json.dumps({"scored": parent["export"], "proxy": parent["proxy_mean_acc"]}), flush=True)
    scored = []
    for export in candidates(cfg):
        scored.append(score(export, items, texts, args.device))
        print(json.dumps({"scored": str(export), "proxy": scored[-1]["proxy_mean_acc"],
                          "wiki_bpb": scored[-1]["bits_per_byte_selection_panels"]["wiki"]}), flush=True)
    chosen, reason, best = decide(parent, scored, rule)
    record = {
        "schema": "v2-task-tune-selection-1", "label": "task_tune", "panel": "selection", "official_scores_used": False,
        "selected_model": chosen["export"], "selected_model_sha256": chosen["export_model_sha256"],
        "decision_reason": reason, "best_eligible_candidate": best["export"] if best else None,
        "rule": rule, "parent": parent["export"], "results": [parent] + scored,
        "data": {"development_sha256": report["files"]["development.jsonl"], "manifest_sha256": sha256_file(manifest),
                 "development_items": len(items)},
        "frozen_at_unix": time.time(),
        "note": ("Development data only: held-out 20% of benchmark TRAIN splits (never trained on) and corpus "
                 "selection panels. Official evaluation splits were not used; official scores may not change this record."),
    }
    write_json(args.out, record)
    print(json.dumps({"selected": chosen["export"], "reason": reason}), flush=True)


if __name__ == "__main__":
    main()
