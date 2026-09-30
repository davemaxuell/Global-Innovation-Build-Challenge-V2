"""Freeze the V2 endpoint choice from development data only, before any official score.

Candidates are run directories with an ``export/``. Each is scored with
``devevaluate``: bits/byte on the manifest's ``selection`` panels (not used before)
and zero-shot accuracy on the held-out 20% of benchmark TRAIN splits. The
registered rule (v2/phases/final_selection/registration.json):

  choose the challenger only if its ARC-E/PIQA/HellaSwag/WinoGrande proxy mean is at
  least ``min_proxy_gain`` above the baseline's AND its Wikipedia bits/byte is not
  worse by more than ``max_wiki_bpb_regression``; otherwise keep the baseline.

The output is the selection record that scripts/run_final_evaluation.py requires
(panel="selection", official_scores_used=false, selected export and hash, all arms).

Run: PYTHONPATH=src:v2/src python -m scglm_v2.select_endpoint --baseline v2/runs/main \
       --challenger v2/runs/anneal_quality --out v2/phases/final_selection/selection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from .common import ROOT, sha256_file, write_json
from .devevaluate import bits_per_byte, load, multiple_choice, panel_texts

MIN_PROXY_GAIN = 0.005
MAX_WIKI_BPB_REGRESSION = 0.01
TOLERANCE = 1e-12  # keeps the registered inclusive boundaries exact under floating point


def score(run: Path, manifest: Path, bench: Path, device: str, per_set: int, panel_docs: int) -> dict:
    export = run / "export"
    model, tok = load(str(export), device)
    texts = panel_texts(manifest, panel_docs, split="selection")
    bpb = {k: bits_per_byte(model, tok, device, v) for k, v in texts.items()}
    mc = multiple_choice(model, tok, device, bench, per_set)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"run": str(run), "export": str(export), "export_model_sha256": sha256_file(export / "model.safetensors"),
            "bits_per_byte_selection_panels": bpb, "multiple_choice": mc,
            "proxy_mean_acc": mc["_official_proxy_mean_acc"]}


def decide(base: dict, challenger: dict | None) -> tuple[dict, str]:
    if challenger is None:
        return base, "only the baseline completed"
    gain = challenger["proxy_mean_acc"] - base["proxy_mean_acc"]
    wiki = challenger["bits_per_byte_selection_panels"]["wiki"] - base["bits_per_byte_selection_panels"]["wiki"]
    if gain >= MIN_PROXY_GAIN - TOLERANCE and wiki <= MAX_WIKI_BPB_REGRESSION + TOLERANCE:
        return challenger, f"challenger: proxy gain {gain:+.4f} >= {MIN_PROXY_GAIN} and wiki bpb change {wiki:+.4f} <= {MAX_WIKI_BPB_REGRESSION}"
    return base, f"baseline: proxy gain {gain:+.4f} (needs >= {MIN_PROXY_GAIN}), wiki bpb change {wiki:+.4f} (limit {MAX_WIKI_BPB_REGRESSION})"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--challenger", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "v2/data/rich_v1/manifest.json")
    p.add_argument("--benchmarks", type=Path, default=ROOT / "v2/data/rich_v1/benchmarks/development.jsonl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--per-set", type=int, default=10**9, help="Items per benchmark set (default: all)")
    p.add_argument("--panel-docs", type=int, default=512)
    p.add_argument("--label", default="final")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; a selection record is frozen once written")
    base = score(args.baseline.resolve(), args.manifest, args.benchmarks, args.device, args.per_set, args.panel_docs)
    challenger = None
    if args.challenger and (args.challenger / "export" / "model.safetensors").exists():
        challenger = score(args.challenger.resolve(), args.manifest, args.benchmarks, args.device, args.per_set, args.panel_docs)
    chosen, reason = decide(base, challenger)
    record = {
        "schema": "v2-selection-1", "label": args.label, "panel": "selection", "official_scores_used": False,
        "selected_model": chosen["export"], "selected_model_sha256": chosen["export_model_sha256"],
        "selected_run": chosen["run"], "decision_reason": reason,
        "rule": {"min_proxy_gain": MIN_PROXY_GAIN, "max_wiki_bpb_regression": MAX_WIKI_BPB_REGRESSION,
                 "proxy": "mean acc over ARC-Easy/PIQA/HellaSwag/WinoGrande held-out TRAIN items"},
        "results": [r for r in (base, challenger) if r is not None],
        "baseline_run": base["run"], "challenger_run": challenger["run"] if challenger else None,
        "data": {"manifest_sha256": sha256_file(args.manifest), "benchmark_development_sha256": sha256_file(args.benchmarks),
                 "per_set": args.per_set, "panel_docs": args.panel_docs},
        "frozen_at_unix": time.time(),
        "note": "Development data only. Official evaluation splits were not used; official scores may not change this record.",
    }
    write_json(args.out, record)
    print(json.dumps({"selected": chosen["run"], "reason": reason,
                      "proxy": {r["run"]: round(r["proxy_mean_acc"], 4) for r in record["results"]}}))


if __name__ == "__main__":
    main()
