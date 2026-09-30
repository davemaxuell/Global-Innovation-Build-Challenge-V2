"""Durable corpus -> audit -> monitored pilot -> full continuation supervisor.

Run inside tmux with CUDA_VISIBLE_DEVICES=0 and PYTHONPATH=src.
No official benchmark is scored. Resume only after the declared pilot passed.
"""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

try:
    from .preflight_extension import preflight
except ImportError:
    from preflight_extension import preflight
from scglm.evaluate import evaluate_documents, fixed_mixture_metrics
from scglm.train import Trainer, atomic_json, runtime_fingerprint, sha256_file


STATE = Path("artifacts/extension_workflow.json")


def pilot_failures(before, after):
    failures = []
    for group, allowance in (("new", .25), ("legacy", .35)):
        delta = after[group]["fixed_q_nll"] - before[group]["fixed_q_nll"]
        if not math.isfinite(delta) or delta > allowance:
            failures.append(f"{group} monitor NLL change {delta} > {allowance}")
        for source, metrics in after[group]["sources"].items():
            delta = metrics["nll"] - before[group]["sources"][source]["nll"]
            if not math.isfinite(delta) or delta > .5:
                failures.append(f"{group}/{source} NLL change {delta} > 0.5")
    return failures


def stage(name, **fields):
    record = {"stage": name, "updated_unix": time.time(), "pid": os.getpid(), **fields}
    atomic_json(STATE, record)
    print(json.dumps(record), flush=True)


def monitor(trainer):
    rows = trainer.validate(limit=32)
    new = fixed_mixture_metrics(rows, weights=trainer.source_weights)
    manifest = json.loads(Path("data/processed/main/manifest.json").read_text())
    documents = []
    for source in ("web", "wiki"):
        split = manifest["sources"][source]["splits"]["monitor"]
        path = Path(split["jsonl_path"])
        if sha256_file(path) != split["jsonl_sha256"]:
            raise ValueError("Legacy monitor integrity failed")
        with path.open() as f:
            for _, line in zip(range(32), f):
                row = json.loads(line)
                row.setdefault("source", source)
                documents.append(row)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        scores = evaluate_documents(trainer.model, documents, context_length=trainer.sequence_length,
                                    stride=512, max_predictable_tokens=1024, device=trainer.device)
    trainer.model.train()
    return {"new": new, "legacy": fixed_mixture_metrics(scores, weights={"web": .8, "wiki": .2})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path("configs/train_extension.json")
    config = json.loads(config_path.read_text())
    config_hash, frozen_runtime = sha256_file(config_path), runtime_fingerprint()
    lock = Path("artifacts/extension_workflow.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_dir = Path(config["run_dir"])
    try:
        if not args.resume:
            stage("waiting_for_completed_corpus")
            while not Path(config["data_manifest"]).exists():
                time.sleep(30)
            if config_hash != sha256_file(config_path) or frozen_runtime != runtime_fingerprint():
                raise ValueError("Code/config changed while waiting: review and restart supervisor")
            stage("data_preflight")
            preflight(config)
        recovery = json.loads(Path("artifacts/extension_gpu_recovery.json").read_text())
        if recovery["status"] != "passed" or not recovery["checked_three_source_continuation"]:
            raise ValueError("Real H100 continuation recovery gate has not passed")
        # Respect the single assigned GPU; never terminate another process.
        occupied = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if occupied:
            raise RuntimeError(f"GPU 0 is occupied by compute PID(s): {occupied}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("Supervisor requires the assigned GPU 0 only")
        gate_path = Path("artifacts/extension_pilot_gate.json")
        if args.resume:
            gate = json.loads(gate_path.read_text())
            if gate["status"] != "passed" or gate["config_sha256"] != config_hash:
                raise ValueError("Pilot gate must pass before full-run resume")
        else:
            stage("pilot_initializing")
            trainer = Trainer(config)
            destination = run_dir / "snapshot/scripts"
            destination.mkdir(parents=True, exist_ok=True)
            for script in (Path(__file__), Path("scripts/preflight_extension.py"), Path("scripts/check_gpu_recovery.py")):
                shutil.copy2(script, destination / script.name)
            before = monitor(trainer)
            atomic_json(Path("artifacts/extension_monitor_before.json"), before)
            stage("pilot_training", pilot_updates=1000, targets=65_536_000)
            trainer.run(max_steps=1000)
            if trainer.step != 1000:
                raise RuntimeError("Pilot stopped before its required 1,000 updates")
            after = monitor(trainer)
            # Operational divergence gates, not statistical superiority claims.
            failures = pilot_failures(before, after)
            gate = {"status": "failed" if failures else "passed", "config_sha256": config_hash,
                    "pilot_updates": trainer.step, "pilot_tokens": trainer.main_tokens,
                    "before": before, "after": after, "failures": failures,
                    "criterion": "Finite updates; weighted monitor NLL increase <=0.25 new / <=0.35 legacy; every source <=0.5 nats. Operational gate only.",
                    "official_benchmarks_scored": False, "updated_unix": time.time()}
            atomic_json(gate_path, gate)
            if failures:
                raise RuntimeError(f"Pilot divergence gate: {failures}")
            del trainer
            gc.collect(); torch.cuda.empty_cache()
        stage("full_training_initializing", pilot_gate=str(gate_path))
        trainer = Trainer(config, resume=run_dir / "checkpoint_latest.json")
        stage("full_training", cumulative_start_tokens=trainer.main_tokens+trainer.parent_provenance["main_tokens"],
              cumulative_target_tokens=12_000_034_816)
        trainer.run()
        stage("completed" if trainer.step == trainer.total_steps else "paused", status_file=str(run_dir / "status.json"))
    except Exception as exc:
        stage("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
