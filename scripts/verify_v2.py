"""Read-only integrity check of the V2 pipeline and selected checkpoint (no training, no scoring).

Checks: core-module hashes against the V2 run record; the checkpoint copy against its
inventory and the frozen selection; data manifest and tokenizer identities; the
official-evaluation outcome record. Optional ``--load`` reloads the model on CPU.

Run: PYTHONPATH=src python scripts/verify_v2.py [--load]
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ("__init__.py", "data.py", "evaluate.py", "model.py", "prepare_data.py", "prepare_extension.py", "train.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--load", action="store_true", help="Also reload the checkpoint on CPU and run a forward pass")
    args = p.parse_args()
    checks = {}
    run = json.loads((ROOT / "v2/runs/main/run.json").read_text())
    recorded = run["runtime_fingerprint"]["source_sha256"]
    checks["core modules match the V2 run record"] = all(sha(ROOT / "src/scglm" / n) == recorded[n] for n in CORE)
    protocol = json.loads((ROOT / "configs/evaluation.json").read_text())["wikitext103"]
    checks["evaluate.py matches the pinned protocol"] = sha(ROOT / protocol["source_file"]) == protocol["source_sha256"]
    inventory = json.loads((ROOT / "checkpoints/SELECTION_2026-10-01.json").read_text())["checkpoints"][0]
    ckpt = ROOT / "checkpoints/v2_best"
    checks["checkpoint files match inventory"] = all(sha(ckpt / n) == h for n, h in inventory["files"].items())
    selection = json.loads((ROOT / "v2/phases/final_selection/selection.json").read_text())
    checks["checkpoint is the frozen selection"] = (selection["selected_model_sha256"] == inventory["weights_sha256"]
                                                    and selection["official_scores_used"] is False)
    checks["checkpoint equals original export"] = sha(ckpt / "model.safetensors") == sha(ROOT / "v2/runs/main/export/model.safetensors")
    checks["data manifest matches run record"] = sha(ROOT / "v2/data/rich_v1/manifest.json") == run["manifest_sha256"]
    checks["tokenizer matches run record"] = run["tokenizer_sha256"] == sha(ROOT / "v2/data/rich_v1/tokenizer/tokenizer.json")
    outcome = json.loads((ROOT / "v2/phases/final_selection/v1_vs_v2.json").read_text())
    checks["pre-registered rule outcome is V2"] = outcome["outcome"] == "V2"
    status = json.loads((ROOT / "v2/runs/main/status.json").read_text())
    checks["run completed with 44,999,966,720 targets"] = status["status"] == "completed" and status["cumulative_tokens"] == 44_999_966_720
    if args.load:
        import torch
        from scglm.model import audit_parameters, load_model
        model = load_model(ckpt).eval()
        audit = audit_parameters(model)
        with torch.no_grad():
            finite = bool(torch.isfinite(model(input_ids=torch.tensor([[2, 464, 3290]])).logits).all())
        checks["offline load, 46,346,752 tied parameters, finite forward"] = audit["trainable_parameters"] == 46_346_752 and finite
    width = max(map(len, checks))
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}")
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
