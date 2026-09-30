"""Registered 1B pilot gate: V2 pilot vs V1 baseline_1b on development data only.

GO if the V2 pilot's mean accuracy over the ARC-Easy/PIQA/HellaSwag/WinoGrande
proxies (held-out benchmark TRAIN items) is >= V1 1B's. Bits/byte is reported too.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .common import ROOT, event, write_json
from .devevaluate import OFFICIAL_PROXY, bits_per_byte, load, multiple_choice, panel_texts


def evaluate(model_dir: str, manifest: Path, bench: Path) -> dict:
    model, tok = load(model_dir, "cuda")
    texts = panel_texts(manifest, 256)
    bpb = {k: bits_per_byte(model, tok, "cuda", v) for k, v in texts.items()}
    mc = multiple_choice(model, tok, "cuda", bench, 1000)
    del model
    torch.cuda.empty_cache()
    return {"model": model_dir, "bits_per_byte": bpb, "multiple_choice": mc}


def main():
    out = ROOT / "v2/phases/pilot_1b"
    manifest = ROOT / "v2/data/rich_v1/manifest.json"
    bench = ROOT / "v2/data/rich_v1/benchmarks/development.jsonl"
    v1 = evaluate(str(ROOT / "runs/baseline_1b/export"), manifest, bench)
    v2 = evaluate(str(ROOT / "v2/runs/pilot_1b/export"), manifest, bench)
    a, b = v1["multiple_choice"]["_official_proxy_mean_acc"], v2["multiple_choice"]["_official_proxy_mean_acc"]
    decision = {"rule": "GO if V2 pilot proxy mean acc >= V1 1B proxy mean acc", "v1_1b": v1, "v2_pilot": v2,
                "proxy_mean_acc": {"v1_1b": a, "v2_pilot": b, "difference": b - a},
                "per_task_acc": {k: {"v1_1b": v1["multiple_choice"][k]["acc"], "v2_pilot": v2["multiple_choice"][k]["acc"]}
                                 for k in v1["multiple_choice"] if not k.startswith("_")},
                "decision": "GO" if b >= a else "NO-GO",
                "note": "Development data only; official evaluation splits untouched."}
    write_json(out / "gate.json", decision)
    event(ROOT / "v2/phases/pilot_1b/events.jsonl", "pilot_gate", decision=decision["decision"],
          proxy_mean_acc=decision["proxy_mean_acc"])


if __name__ == "__main__":
    main()
