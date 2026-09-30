"""Registered 12B -> 13B raw-pretraining continuation with inherited data state."""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from scglm.evaluate import evaluate_documents, fixed_mixture_metrics
from scglm.model import load_model
from scglm.train import Trainer, atomic_json, configure_numerics, sha256_file, verify_scratch_lineage


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def stage(work, name, **fields):
    value = {"stage": name, "pid": os.getpid(), "updated_unix": time.time(), **fields}
    atomic_json(work / "status.json", value)
    with (work / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def freeze(work, protocol_path, protocol, config):
    paths = [*Path("src/scglm").glob("*.py"), Path(__file__), protocol_path,
             Path(protocol["training_config"]), Path(config["model_config"]), Path("PRETRAINING_13B_PLAN.md")]
    contract = {"protocol": protocol, "training": config,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "numpy", "tokenizers", "safetensors")},
                "files": {str(p.resolve()): sha256_file(p) for p in sorted(paths)}}
    contract["sha256"] = digest(contract)
    path = work / "contract.json"
    if path.exists():
        if read(path) != contract:
            raise ValueError("Registered continuation code/config changed; use a new experiment")
    else:
        atomic_json(path, contract)
        for p in paths:
            target = work / "snapshot" / p.resolve().relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    return contract["sha256"]


def checked_parent(config):
    if sha256_file(config["parent_checkpoint"]) != config["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint changed")
    payload = torch.load(config["parent_checkpoint"], map_location="cpu", weights_only=False)
    record = read(Path(config["parent_run"]) / "run.json")
    status = read(Path(config["parent_run"]) / "status.json")
    if payload["science_digest"] != record["science_digest"] or any(payload[k] != status[k] for k in ("main_tokens", "step")):
        raise ValueError("Parent payload is not the completed endpoint")
    return payload


def prepare(work, protocol, config, contract_hash):
    path = work / "preflight.json"
    if path.exists():
        result = read(path)
        if result["contract_sha256"] != contract_hash or result["status"] != "passed":
            raise ValueError("Cached preflight does not match registration")
        for phase, entry in result["panels"].items():
            if sha256_file(work / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Protected {phase} panel changed")
        return result
    lineage = verify_scratch_lineage(config["parent_run"])
    if lineage["cumulative_tokens"] != protocol["parent_cumulative_tokens"]:
        raise ValueError("Unexpected parent cumulative tokens")
    if sha256_file(Path(config["parent_run"]) / "export/model.safetensors") != protocol["parent_export_sha256"]:
        raise ValueError("Parent export identity changed")
    if sha256_file(config["data_manifest"]) != protocol["manifest_sha256"]:
        raise ValueError("Corpus manifest changed")
    manifest = read(config["data_manifest"])
    if sha256_file(protocol["prior_data_audit"]) != protocol["prior_data_audit_sha256"]:
        raise ValueError("Original corpus audit identity changed")
    audit = read(protocol["prior_data_audit"])
    if audit["status"] != "passed" or audit["manifest_sha256"] != protocol["manifest_sha256"]:
        raise ValueError("Original full corpus audit is unavailable")
    payload = checked_parent(config)
    sources = list(config["source_weights"])
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(payload["rng"])
    steps = config["target_tokens"] // (config["batch_sequences"] * config["sequence_length"])
    counts = np.zeros(len(sources), dtype=np.int64)
    for first in range(0, steps, 1000):
        draws = rng.choice(len(sources), min(1000, steps-first) * config["batch_sequences"],
                           p=[config["source_weights"][s] for s in sources])
        counts += np.bincount(draws, minlength=len(sources)) * config["sequence_length"]
    capacity, panels = {}, {"development": [], "confirmation": []}
    all_ids, all_hashes = set(), set()
    for i, source in enumerate(sources):
        split = manifest["sources"][source]["splits"]["train"]
        for file_key, hash_key in (("path", "sha256"), ("index_path", "index_sha256")):
            if sha256_file(split[file_key]) != split[hash_key]:
                raise ValueError(f"Corpus/index checksum changed: {source}")
        prior = audit["sources"][source]
        if (prior["sha256"] != split["sha256"] or prior["eos_tokens"] != prior["documents"]
                or prior["unk_tokens"] != 0 or prior["reserved_exact_overlap"] != 0):
            raise ValueError("Original EOS/vocabulary/holdout audit failed")
        stream = payload["streams"][source]
        used = stream["tokens_consumed"]
        if stream["cycles"] != 0 or stream["cursor"] != used or used != payload["main_source_tokens"][source]:
            raise ValueError("Expected unwrapped parent corpus cursors")
        if used + int(counts[i]) > split["tokens"] - 1:
            raise ValueError(f"Insufficient unread source targets: {source}")
        capacity[source] = {"parent_consumed": used, "additional_targets": int(counts[i]),
                            "unread_before": split["tokens"] - 1 - used,
                            "unread_after": split["tokens"] - 1 - used - int(counts[i]),
                            "stored_tokens": split["tokens"], "original_eos_audit_reused": True}
        reserved = manifest["sources"][source]["splits"][protocol["evaluation_split"]]
        if sha256_file(reserved["jsonl_path"]) != reserved["jsonl_sha256"]:
            raise ValueError("Held-out document panel changed")
        rows = [json.loads(line) for line in Path(reserved["jsonl_path"]).read_text().splitlines()]
        if len(rows) != 2 * protocol["documents_per_source_per_phase"]:
            raise ValueError("Unexpected development panel size")
        for row in rows:
            key = (source, row["id"])
            if key in all_ids or row["content_sha256"] in all_hashes:
                raise ValueError("Repeated evaluation document identity")
            all_ids.add(key); all_hashes.add(row["content_sha256"])
            row["source"] = source
        rows.sort(key=lambda r: digest([protocol["panel_partition_salt"], source, r["id"]]))
        size = protocol["documents_per_source_per_phase"]
        panels["development"].extend(rows[:size]); panels["confirmation"].extend(rows[size:])
        print(json.dumps({"preflight_source": source, **capacity[source]}), flush=True)
    del payload
    entries = {}
    for phase, rows in panels.items():
        p = work / "panels" / (phase + ".jsonl")
        p.parent.mkdir(exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        entries[phase] = {"path": str(p.relative_to(work)), "sha256": sha256_file(p), "documents": len(rows)}
    result = {"status": "passed", "contract_sha256": contract_hash, "lineage": lineage,
              "capacity": capacity, "panels": entries, "total_steps": steps,
              "additional_targets": int(counts.sum()), "cumulative_target": lineage["cumulative_tokens"] + int(counts.sum()),
              "prior_data_audit_sha256": sha256_file(protocol["prior_data_audit"]),
              "limitation": "Reuses original corpus decontamination and complete EOS audit after verifying identical token/index bytes; no proof of semantic independence."}
    atomic_json(path, result)
    return result


def tree_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def gpu_recovery(work, config, contract_hash):
    receipt = work / "gpu_recovery.json"
    if receipt.exists():
        previous = read(receipt)
        if previous["status"] != "passed" or previous["contract_sha256"] != contract_hash:
            raise ValueError("Recovery receipt does not match current code")
        return
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="disposable_recovery_", dir=work) as folder:
        temporary = {**config, "run_dir": folder, "validation_every": 0}
        trainer = Trainer(temporary)
        parent = checked_parent(config)
        if not (tree_equal(trainer.state()["model"], parent["model"])
                and tree_equal(trainer.optimizer.state_dict(), parent["optimizer"])
                and tree_equal(trainer.state()["streams"], parent["streams"])):
            raise ValueError("Real parent model/optimizer/streams were not inherited exactly")
        del parent
        trainer.update(None); trainer.update(None)
        trainer.save_checkpoint("disposable_recovery")
        expected_loss = trainer.update(None)
        expected = trainer.state()
        del trainer
        gc.collect(); torch.cuda.empty_cache()
        restored = Trainer(temporary, resume=Path(folder) / "checkpoint_latest.json")
        actual_loss = restored.update(None)
        if actual_loss != expected_loss or not tree_equal(restored.state(), expected):
            raise ValueError("Real GPU recovery is not exact")
        restored.export()
        exported = load_model(Path(folder) / "export")
        if not tree_equal(exported.state_dict(), expected["model"]):
            raise ValueError("Recovered export differs")
        del restored, exported, expected
        gc.collect(); torch.cuda.empty_cache()
    atomic_json(receipt, {"status": "passed", "contract_sha256": contract_hash,
                          "inherited_weights_optimizer_streams_exact": True, "resumed_state_and_loss_exact": True,
                          "export_exact": True, "discarded_optimizer_updates": 4,
                          "discarded_processed_targets": 4 * config["batch_sequences"] * config["sequence_length"],
                          "elapsed_seconds": time.monotonic() - started})


def score(model, weights_sha, phase, label, work, protocol, config, prepared, contract_hash):
    path = work / f"{label}_{phase}.json"
    panel = prepared["panels"][phase]
    identity = {"model_sha256": weights_sha, "panel_sha256": panel["sha256"], "contract_sha256": contract_hash}
    if path.exists():
        cached = read(path)
        if cached["identity"] != identity:
            raise ValueError("Cached likelihood evaluation identity differs")
        return cached
    if sha256_file(work / panel["path"]) != panel["sha256"]:
        raise ValueError("Evaluation panel changed")
    rows = [json.loads(line) for line in (work / panel["path"]).read_text().splitlines()]
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        scores = evaluate_documents(model, rows, context_length=config["sequence_length"],
                                    stride=config["sequence_length"] // 2,
                                    max_predictable_tokens=protocol["evaluation_max_predictable_tokens"], device="cuda")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    result = {"identity": identity, "phase": phase, "metrics": fixed_mixture_metrics(scores, config["source_weights"]),
              "documents": scores, "dtype": "float32", "completed_unix": time.time()}
    atomic_json(path, result)
    return result


def compare(base, candidate, weights, protocol):
    a = {(r["source"], r["id"]): r for r in base["documents"]}
    b = {(r["source"], r["id"]): r for r in candidate["documents"]}
    if len(a) != len(base["documents"]) or len(b) != len(candidate["documents"]) or a.keys() != b.keys():
        raise ValueError("Paired comparison document identities differ")
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    mixed_samples = np.zeros(protocol["bootstrap_replicates"])
    changes = {}
    for source, weight in weights.items():
        keys = sorted(k for k in a if k[0] == source)
        if not keys or any(a[k]["token_count"] != b[k]["token_count"] for k in keys):
            raise ValueError("Paired token counts differ")
        n = np.array([a[k]["token_count"] for k in keys])
        d = np.array([b[k]["nll_sum"] - a[k]["nll_sum"] for k in keys])
        if (n <= 0).any() or not np.isfinite(d).all():
            raise ValueError("Invalid paired likelihood values")
        indexes = rng.integers(0, len(keys), (protocol["bootstrap_replicates"], len(keys)))
        draws = d[indexes].sum(1) / n[indexes].sum(1)
        mixed_samples += weight * draws
        changes[source] = float(d.sum() / n.sum())
    change = sum(weights[s] * changes[s] for s in weights)
    low, high = map(float, np.quantile(mixed_samples, [.025, .975]))
    passed = (change <= -protocol["min_mixture_nll_improvement"] and high < 0
              and all(d <= protocol["max_source_nll_regression"] for d in changes.values()))
    return {"passed": passed, "candidate_minus_parent_nll": change, "lower_95": low, "upper_95": high,
            "source_nll_changes": changes, "limitation": "Document bootstrap; per-source regression guards use point estimates."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/continuation_13b_protocol.json"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--recovery-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol, config = read(args.protocol), read(read(args.protocol)["training_config"])
    work = Path(protocol["output"])
    work.mkdir(parents=True, exist_ok=True)
    with (work / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract_hash = freeze(work, args.protocol, protocol, config)
        if (work / "comparison.json").exists():
            completed = read(work / "comparison.json")
            if (completed["status"] != "completed" or completed["contract_sha256"] != contract_hash
                    or sha256_file(Path(config["run_dir"]) / "export/model.safetensors") != completed["candidate_sha256"]):
                raise ValueError("Completed experiment identity changed")
            print(json.dumps({"stage": "already_completed", "comparison": str(work / "comparison.json")}), flush=True)
            return
        stage(work, "preflight", contract_sha256=contract_hash)
        try:
            prepared = prepare(work, protocol, config, contract_hash)
        except BaseException:
            stage(work, "failed_preflight", error=traceback.format_exc())
            raise
        if args.prepare_only:
            stage(work, "prepared", cumulative_target=prepared["cumulative_target"])
            return
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("Expose only assigned GPU 0")
        with Path("artifacts/pipeline_gpu0.lock").open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            occupied = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if occupied:
                raise RuntimeError(f"GPU 0 has another compute process: {occupied}")
            configure_numerics(config, "cuda")
            try:
                stage(work, "gpu_recovery_check")
                gpu_recovery(work, config, contract_hash)
                if args.recovery_only:
                    stage(work, "recovery_passed")
                    return
                run = Path(config["run_dir"])
                exists = (run / "run.json").exists()
                if exists and not args.resume:
                    raise ValueError("Run exists; use --resume")
                stage(work, "initializing_training")
                trainer = Trainer(config, resume=run / "checkpoint_latest.json" if exists else None)
                baseline_path = work / "parent_development.json"
                before_path = work / "monitor_before.json"
                if not baseline_path.exists() or not before_path.exists():
                    if trainer.step != 0:
                        raise ValueError("Parent evidence missing after training started")
                    stage(work, "parent_evaluation")
                    score(trainer.model, protocol["parent_export_sha256"], "development", "parent", work, protocol, config, prepared, contract_hash)
                    before = fixed_mixture_metrics(trainer.validate(limit=32), config["source_weights"])
                    atomic_json(before_path, before)
                    trainer.save_checkpoint("registered_start")
                gate_path = work / "pilot_gate.json"
                if not gate_path.exists():
                    if trainer.step > protocol["pilot_steps"]:
                        raise ValueError("Training advanced without a pilot gate")
                    stage(work, "pilot_training", pilot_steps=protocol["pilot_steps"], cumulative_target=prepared["cumulative_target"])
                    trainer.run(max_steps=protocol["pilot_steps"])
                    if trainer.stop_requested or trainer.step != protocol["pilot_steps"]:
                        stage(work, "paused", training_step=trainer.step)
                        return
                    before = read(before_path)
                    after = fixed_mixture_metrics(trainer.validate(limit=32), config["source_weights"])
                    failures = []
                    delta = after["fixed_q_nll"] - before["fixed_q_nll"]
                    if not math.isfinite(delta) or delta > protocol["pilot_max_mixture_nll_increase"]:
                        failures.append("Mixture monitor loss exceeded the divergence limit")
                    for source in config["source_weights"]:
                        delta = after["sources"][source]["nll"] - before["sources"][source]["nll"]
                        if not math.isfinite(delta) or delta > protocol["pilot_max_source_nll_increase"]:
                            failures.append(f"{source} monitor loss exceeded the divergence limit")
                    atomic_json(gate_path, {"status": "failed" if failures else "passed", "before": before,
                                           "after": after, "failures": failures, "contract_sha256": contract_hash})
                gate = read(gate_path)
                if gate["status"] != "passed" or gate["contract_sha256"] != contract_hash:
                    raise ValueError("Pilot gate failed; inspect evidence before a new experiment")
                stage(work, "training", cumulative_target=prepared["cumulative_target"], training_status=str(run / "status.json"))
                trainer.run()
                if trainer.step != trainer.total_steps:
                    stage(work, "paused", training_step=trainer.step)
                    return
                candidate_sha = sha256_file(run / "export/model.safetensors")
                stage(work, "development_comparison")
                candidate = score(trainer.model, candidate_sha, "development", "candidate", work, protocol, config, prepared, contract_hash)
                development = compare(read(baseline_path), candidate, config["source_weights"], protocol)
                confirmation = None
                if development["passed"]:
                    stage(work, "confirmation_comparison")
                    final_candidate = score(trainer.model, candidate_sha, "confirmation", "candidate", work, protocol, config, prepared, contract_hash)
                    del trainer
                    gc.collect(); torch.cuda.empty_cache()
                    parent_model = load_model(Path(config["parent_run"]) / "export").to("cuda").eval()
                    final_parent = score(parent_model, protocol["parent_export_sha256"], "confirmation", "parent", work, protocol, config, prepared, contract_hash)
                    confirmation = compare(final_parent, final_candidate, config["source_weights"], protocol)
                    del parent_model
                keep_candidate = development["passed"] and confirmation is not None and confirmation["passed"]
                result = {"status": "completed", "contract_sha256": contract_hash, "development": development,
                          "confirmation": confirmation, "candidate_model": str(run / "export"),
                          "candidate_sha256": candidate_sha, "candidate_cumulative_tokens": prepared["cumulative_target"],
                          "recommended_model": str((run if keep_candidate else Path(config["parent_run"])) / "export"),
                          "recommendation": "candidate_improves_raw_likelihood" if keep_candidate else "retain_12b_parent",
                          "official_scores_previously_observed": True, "official_benchmarks_rerun": False,
                          "original_competition_selection_modified": False,
                          "limitations": "Raw-likelihood improvement does not establish better instruction following, arithmetic or official benchmark accuracy."}
                atomic_json(work / "comparison.json", result)
                stage(work, "completed", comparison=str(work / "comparison.json"), recommendation=result["recommendation"])
            except BaseException:
                stage(work, "failed", error=traceback.format_exc())
                raise


if __name__ == "__main__":
    main()
