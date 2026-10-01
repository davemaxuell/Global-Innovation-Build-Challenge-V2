"""Weight-space interpolation between V2 and a fine-tuned descendant (WiSE-FT; phase v2/phases/wiseft).

Wortsman et al. (2022), "Robust fine-tuning of zero-shot models": theta(alpha) =
(1 - alpha) * theta0 + alpha * theta1, where theta0 is the pretrained model and theta1
its fine-tuned descendant. No training happens. Each alpha in the registered grid
becomes a standard export (same config and tokenizer as theta0) that the development
selector (``task_select``) and the official runner accept.

Run: PYTHONPATH=src python -m scglm_v2.wiseft --config v2/phases/wiseft/config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from .common import ROOT, event, now, sha256_file, write_json


def interpolate(model0, model1, alpha: float):
    """Return model0 with every parameter set to (1 - alpha) * p0 + alpha * p1 (tied weights once)."""
    params1 = dict(model1.named_parameters())
    with torch.no_grad():
        for name, p0 in model0.named_parameters():
            p1 = params1[name]
            if p0.shape != p1.shape:
                raise ValueError(f"Shape mismatch for {name}")
            p0.copy_(((1 - alpha) * p0.double() + alpha * p1.double()).to(p0.dtype))
    return model0


def build(cfg: dict) -> list[Path]:
    from scglm.model import audit_parameters, load_model, save_model
    theta0, theta1 = ROOT / cfg["theta0_export"], ROOT / cfg["theta1_export"]
    hashes = {k: sha256_file(p / "model.safetensors") for k, p in (("theta0", theta0), ("theta1", theta1))}
    if hashes != {"theta0": cfg["theta0_sha256"], "theta1": cfg["theta1_sha256"]}:
        raise ValueError(f"Endpoint weights differ from the registration: {hashes}")
    if json.loads((theta0 / "config.json").read_text()) != json.loads((theta1 / "config.json").read_text()):
        raise ValueError("Endpoint configs differ")
    out_root = ROOT / cfg["output_dir"]
    if out_root.exists():
        raise FileExistsError(f"{out_root} exists; use a new output_dir")
    log = ROOT / cfg["phase_dir"] / "events.jsonl"
    model1 = load_model(theta1)
    exports = []
    for alpha in cfg["alphas"]:
        model = interpolate(load_model(theta0), model1, float(alpha))
        export = out_root / f"alpha_{alpha:.2f}" / "export"
        save_model(model, export)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(theta0 / name, export / name)
        write_json(export / "training_provenance.json", {
            "initialization": "scratch", "created": now(),
            "wiseft": {"alpha": alpha, "formula": "(1 - alpha) * theta0 + alpha * theta1",
                       "theta0": {"export": str(theta0), "sha256": hashes["theta0"]},
                       "theta1": {"export": str(theta1), "sha256": hashes["theta1"]}},
            "parent": json.loads((theta0 / "training_provenance.json").read_text()),
            "fine_tuned": json.loads((theta1 / "training_provenance.json").read_text()),
            "parameter_audit": audit_parameters(model)})
        event(log, "interpolated", alpha=alpha, path=str(export), model_sha256=sha256_file(export / "model.safetensors"))
        exports.append(export)
    return exports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    build(json.loads((ROOT / args.config).read_text()))


if __name__ == "__main__":
    main()
