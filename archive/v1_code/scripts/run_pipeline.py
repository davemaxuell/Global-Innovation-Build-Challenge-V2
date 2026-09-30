"""Verify or reproduce the selected full-SFT checkpoint; default is read-only."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def gpu_environment(config, device):
    if device == "cpu":
        return
    if device not in ("cuda", "cuda:0"):
        raise ValueError("The retained trainer uses one allocated device, cuda:0")
    allowed = config["allowed_gpu_uuids"]
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    mapping = dict(line.strip().split(", ", 1) for line in listing.splitlines())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", allowed[0])
    selected = [mapping.get(part.strip(), part.strip()) for part in visible.split(",")]
    admitted = [uuid for uuid in allowed if uuid in selected and uuid in mapping.values()]
    if not admitted:
        raise ValueError("The assigned H100 is not available in the visible device set")
    os.environ["CUDA_VISIBLE_DEVICES"] = admitted[0]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def checked_output(value):
    """Keep reproduction outputs away from every preserved run and model."""
    allowed = ROOT / "artifacts/current_model_reproductions"
    if allowed.is_symlink():
        raise ValueError("Reproduction root must not be a symlink")
    output = Path(value)
    output = (output if output.is_absolute() else ROOT / output).resolve()
    if output == allowed.resolve() or not output.is_relative_to(allowed.resolve()):
        raise ValueError("Choose a child of artifacts/current_model_reproductions/")
    return output


def verify(config, *, load_model=False):
    from scglm_pipeline.common import read, resolve, sha256, digest, read_jsonl
    from scglm_pipeline.sota_data import load
    from scglm_pipeline.sota_train import balanced_plan

    manifest_path = resolve(config["manifest"])
    if sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Selected training manifest changed")
    manifest = load(manifest_path)
    for path, expected in (
        (resolve(config["parent"]) / "model.safetensors", config["parent_sha256"]),
        (resolve(config["selected_model"]) / "model.safetensors", config["selected_model_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Preserved model identity changed: {path}")
    historical = read(resolve(config["historical_run"]))
    recipe = config["sft_recipe"]
    executed_recipe = {k: v for k, v in recipe.items() if k != "stop_after_updates"}
    if executed_recipe != historical["contract"]["recipe"]:
        raise ValueError("The retained pipeline supports the selected recipe only")
    if recipe["stop_after_updates"] != 105:
        raise ValueError("The selected checkpoint is update 105")
    rows = read_jsonl(manifest_path.parent / manifest["files"]["train"]["file"])
    rows = [row for row in rows if row.get("balanced", True)]
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Held-out examples cannot enter training")
    plan, budget = balanced_plan(rows, recipe, config["mixture"])
    if digest(plan) != config["expected_plan_sha256"]:
        raise ValueError("Selected deterministic example order changed")
    if (len(plan) != config["expected_planned_updates"]
            or budget["targets"] != config["expected_planned_response_targets"]):
        raise ValueError("The full learning-rate schedule budget changed")
    prefix = [rows[index] for batch in plan[:105] for index in batch]
    counters = {
        "response_targets": sum(row["target_tokens"] for row in prefix),
        "response_processed_tokens": sum(row["processed_tokens"] for row in prefix),
    }
    replay = read_jsonl(manifest_path.parent / manifest["files"]["replay"]["file"])
    pools = {s: [r for r in replay if r["source"] == s] for s in ("edu", "dclm", "wiki")}
    used, cursors = Counter(), Counter()
    weights = {"edu": .6, "dclm": .3, "wiki": .1}
    counters.update(replay_targets=0, replay_processed_tokens=0)
    for batch in plan[:105]:
        targets = sum(rows[i]["target_tokens"] for i in batch)
        needed = targets * recipe["replay_lambda"] / (1 - recipe["replay_lambda"])
        amount = 0
        while amount < needed:
            source = min(weights, key=lambda s: used[s] / weights[s])
            row = pools[source][cursors[source] % len(pools[source])]
            cursors[source] += 1
            used[source] += row["target_tokens"]
            amount += row["target_tokens"]
            counters["replay_targets"] += row["target_tokens"]
            counters["replay_processed_tokens"] += row["processed_tokens"]
    if counters != config["expected_selected_counters"]:
        raise ValueError("Selected checkpoint token exposure changed")
    for entry in read(ROOT / "checkpoints/SELECTION_2026-09-29.json")["checkpoints"]:
        for name, expected in entry["files"].items():
            if sha256(Path(entry["path"]) / name) != expected:
                raise ValueError("Preserved working checkpoint inventory changed")
    result = {"verified": True, "training_started": False,
              "selected_checkpoint": str(resolve(config["selected_model"])),
              "weights_sha256": config["selected_model_sha256"],
              "schedule_updates": len(plan), "selected_updates": 105,
              "plan_sha256": digest(plan), "counters": counters}
    if load_model:
        import torch
        from transformers import AutoTokenizer
        from scglm.model import load_model as load, audit_parameters
        path = resolve(config["selected_model"])
        model = load(path).eval()
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        with torch.inference_mode():
            if not torch.isfinite(model(**tokenizer("Checkpoint verification", return_tensors="pt")).logits).all():
                raise ValueError("Nonfinite model output")
        result.update(offline_reload=True, parameter_audit=audit_parameters(model))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "sft"), nargs="?", default="verify")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/current_model.json")
    parser.add_argument("--output", help="New/resumable child of artifacts/current_model_reproductions/")
    parser.add_argument("--load", action="store_true", help="Also reload the selected export on CPU")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    # Configure visibility before importing torch through the model/trainer modules.
    if args.command == "sft":
        gpu_environment(config, config["device"])
    report = verify(config, load_model=args.load)
    if args.command == "verify":
        print(json.dumps(report, indent=2))
        return
    from scglm_pipeline.sota_train import Trainer
    from scglm_pipeline.common import write_json
    from scglm_post.common import append_event
    output = checked_output(args.output or config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "artifacts/pipeline_gpu0.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        torch.set_num_threads(config["cpu_threads"])
        trainer = Trainer(config, config["sft_recipe"], parent=config["parent"],
                          manifest=config["manifest"], output=output, device=config["device"])
        endpoints = trainer.run()
        if trainer.step == 105:
            if dict(trainer.counters) != config["expected_selected_counters"]:
                raise ValueError("Completed reproduction token exposure differs")
            endpoint = next(e for e in endpoints if e["step"] == 105)
            result = {"status": "completed", "selected_endpoint": endpoint,
                      "matches_preserved_weight_bytes": endpoint["weights_sha256"] == config["selected_model_sha256"],
                      "counters": dict(trainer.counters), "original_checkpoint_preserved": True}
            write_json(output / "reproduction.json", result)
            append_event(output / "events.jsonl", {"event": "reproduction_verified", **result})
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
