"""Gated orchestration: prepare → preflight → base reference → random arm → curated arm → decision.

Both arms continue the preserved 13B checkpoint with its inherited AdamW state
through the retained pretraining ``Trainer``: identical seed, source-sampling
sequence, token budget and schedule. Only the document-selection rule differs.
No arm replaces a preserved checkpoint or schedules an SFT descendant.
"""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

from scglm_study.common import (ROOT, StudyStopped, append_failed_approach, digest, event, finish_phase, immutable,
                                read, read_jsonl, register, roots, runtime, sha256, sources, status, write_json)
from .prepare import ARMS, load, prepare, verify_shards


def panels(cfg):
    from scglm_study.evaluate import choice_panels
    data, _ = roots(cfg)
    result = choice_panels(cfg["evaluation"]["choice_panels"])
    result["heldout_raw"] = {"kind": "raw", "rows": read_jsonl(data / "heldout_raw.jsonl"),
                             "source": str((data / "heldout_raw.jsonl").relative_to(ROOT))}
    return result


def verify(cfg, *, load_model=False):
    parent = cfg["parent"]
    if sha256(ROOT / parent["path"] / "model.safetensors") != parent["weight_sha256"]:
        raise ValueError("Preserved base export changed")
    if sha256(ROOT / parent["checkpoint"]) != parent["checkpoint_sha256"]:
        raise ValueError("Parent training checkpoint changed")
    if sha256(ROOT / cfg["corpus"]["manifest"]) != cfg["corpus"]["sha256"]:
        raise ValueError("Continuation corpus manifest changed")
    data, output = roots(cfg)
    result = {"verified": True, "read_only": True, "training_started": False,
              "parent_weight_sha256": parent["weight_sha256"], "parent_checkpoint_sha256": parent["checkpoint_sha256"],
              "prepared": (data / "manifest.json").exists(),
              "decision": read(output / "decision.json") if (output / "decision.json").exists() else None}
    if result["prepared"]:
        manifest = load(cfg)
        result.update(manifest_sha256=sha256(data / "manifest.json"), arm_manifests=manifest["arm_manifests"],
                      heldout_documents=manifest["heldout_documents"])
    if load_model:
        result["checkpoint_matches_export"] = checkpoint_matches_export(cfg)
    return result


def checkpoint_matches_export(cfg):
    """The inherited optimizer state must belong to exactly the preserved base weights."""
    import torch
    from safetensors.torch import load_file
    payload = torch.load(ROOT / cfg["parent"]["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    exported = load_file(str(ROOT / cfg["parent"]["path"] / "model.safetensors"))
    state = payload["model"]
    tied = "lm_head.weight"
    for name, tensor in state.items():
        other = exported.get(name, exported.get("model.embed_tokens.weight") if name == tied else None)
        if other is None or not torch.equal(tensor, other):
            raise ValueError(f"Checkpoint weights differ from the preserved export: {name}")
    if "optimizer" not in payload or not payload["optimizer"]["state"]:
        raise ValueError("Parent checkpoint lacks AdamW state")
    return {"tensors": len(state), "optimizer_state_entries": len(payload["optimizer"]["state"]),
            "step": payload["step"], "main_tokens": payload["main_tokens"]}


def trainer_config(cfg, arm):
    data, output = roots(cfg)
    t, p = cfg["training"], cfg["parent"]
    return {"seed": t["seed"], "model_config": "configs/model.json",
            "data_manifest": str(data / arm / "manifest.json"), "run_dir": str(output / "runs" / arm),
            "arm": f"{cfg['experiment_id']}:{arm}", "parent_run": p["run"], "parent_checkpoint": p["checkpoint"],
            "parent_checkpoint_sha256": p["checkpoint_sha256"], "inherit_data_state": False,
            "target_tokens": t["updates"] * t["batch_sequences"] * t["sequence_length"],
            "sequence_length": t["sequence_length"], "batch_sequences": t["batch_sequences"],
            "microbatch_sequences": t["microbatch_sequences"], "source_weights": cfg["source_weights"],
            "learning_rate": t["learning_rate"], "initial_learning_rate": t["initial_learning_rate"],
            "rewarm_steps": t["rewarm_steps"], "min_lr_ratio": t["min_lr_ratio"],
            "warmup_fraction": t["warmup_fraction"], "betas": t["betas"], "adam_epsilon": t["adam_epsilon"],
            "weight_decay": t["weight_decay"], "gradient_clip": t["gradient_clip"],
            "checkpoint_every": t["checkpoint_every"], "log_every": t["log_every"], "validation_every": 0,
            "max_wall_hours": t["max_training_wall_hours"], "compile": False, "deterministic": True,
            "controller": False, "max_source_epochs": 1.0}


def preflight(cfg, manifest, budget):
    import torch
    from scglm.data import SourceStream
    from scglm.model import audit_parameters, forward_loss, load_model
    data, output = roots(cfg)
    receipt = output / "preflight.json"
    contract = {"sources": sources(cfg), "runtime": runtime(), "manifest_sha256": sha256(data / "manifest.json")}
    if receipt.exists():
        previous = read(receipt)
        if previous["contract"] != contract or previous["status"] != "completed":
            raise ValueError("Preflight contract changed; resolve before resuming")
        return previous
    phase = register(cfg, "preflight", "Run the study tests, verify both arms' shards and the inherited checkpoint, "
                     "and take one no-update forward/backward on each arm's first batch.",
                     data={"manifest_sha256": contract["manifest_sha256"]})
    attempt = phase / f"attempt_{len(list(phase.glob('attempt_*'))) + 1:03d}"
    attempt.mkdir()
    try:
        status(phase, "running")
        log = attempt / "tests.log"
        with log.open("w") as stream:
            tests = subprocess.run([sys.executable, "-m", "pytest", "-q", *cfg["preflight_tests"]], cwd=ROOT,
                                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, stdout=stream,
                                   stderr=subprocess.STDOUT)
        if tests.returncode:
            raise ValueError(f"Study tests failed: {log}")
        shards = {arm: {s: e["splits"]["train"]["sha256"] for s, e in verify_shards(cfg, arm)["sources"].items()}
                  for arm in ARMS}
        inherited = checkpoint_matches_export(cfg)
        metrics = {}
        with budget.session("shared"):
            model = load_model(ROOT / cfg["parent"]["path"]).to("cuda:0")
            audit = audit_parameters(model)
            model.train()
            for arm in ARMS:
                arm_manifest = read(data / arm / "manifest.json")
                losses = {}
                for source, entry in arm_manifest["sources"].items():
                    stream = SourceStream(entry["splits"]["train"]["path"], seq_len=cfg["training"]["sequence_length"])
                    x, y = stream.next_batch(2)
                    budget.charge(int(x.size), "preflight_forward_backward")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = forward_loss(model, torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda())
                    loss.backward()
                    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                                       for p in model.parameters()):
                        raise ValueError(f"Nonfinite preflight loss or gradient: {arm}/{source}")
                    model.zero_grad(set_to_none=True)
                    losses[source] = float(loss)
                metrics[arm] = losses
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result = {"status": "completed", "contract": contract, "parameter_audit": audit, "shards": shards,
                  "inherited_checkpoint": inherited, "no_update_losses": metrics, "optimizer_updates": 0,
                  "tests_log_sha256": sha256(log), "attempt": str(attempt)}
        if contract["sources"] != sources(cfg) or contract["runtime"] != runtime():
            raise ValueError("Implementation changed during preflight")
        immutable(receipt, result)
        finish_phase(phase, "completed", optimizer_updates=0, decision="preflight_passed",
                     next_step="Score the base reference", receipt_sha256=sha256(receipt))
        return result
    except BaseException as error:
        finish_phase(phase, "paused" if isinstance(error, InterruptedError) else "failed", error=repr(error),
                     optimizer_updates=0, attempt=str(attempt), next_step="Fix the implementation; no recipe change")
        raise


def train_arm(cfg, arm, budget):
    """Start or resume one arm with the retained trainer; return its export path."""
    from scglm.train import Trainer
    _, output = roots(cfg)
    config = trainer_config(cfg, arm)
    run = Path(config["run_dir"])
    state = read(run / "status.json") if (run / "status.json").exists() else None
    if state and state["status"] == "completed":
        return run / "export"
    if state and state["status"] == "failed":
        raise StudyStopped(f"Arm {arm} failed; it cannot be silently retrained")
    resume = run / "checkpoint_latest.json" if (run / "run.json").exists() else None
    with budget.session(arm):
        trainer = Trainer(config, device="cuda", resume=resume)
        trainer.run()
        del trainer
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    state = read(run / "status.json")
    if state["status"] != "completed":
        raise InterruptedError(f"Arm {arm} paused at step {state['step']}; resume the same study")
    return run / "export"


def run(cfg, *, resume=False):
    data, output = roots(cfg)
    verify(cfg)
    if (output / "decision.json").exists():
        return report(cfg)
    status(output, "running", phase="preparation", resume=resume)
    if not (data / "manifest.json").exists():
        phase = register(cfg, "preparation", "Scan the unused continuation span, freeze the held-out panel and write "
                         "both arms' shards at fixed 60/30/10 shares. CPU only; no model scores.",
                         data={"corpus_manifest_sha256": cfg["corpus"]["sha256"]})
        status(phase, "running")
        try:
            manifest = prepare(cfg)
        except BaseException as error:
            finish_phase(phase, "failed", error=repr(error), optimizer_updates=0)
            raise
        finish_phase(phase, "completed", optimizer_updates=0, manifest_sha256=sha256(data / "manifest.json"),
                     arm_manifests=manifest["arm_manifests"], heldout_documents=manifest["heldout_documents"])
    manifest = load(cfg)
    from scglm_study.budget import Budget
    from scglm_study.evaluate import evaluate, guards
    budget = Budget(output, cfg["budget"])
    preflight(cfg, manifest, budget)
    immutable(output / "execution_contract.json", {"sources": sources(cfg), "runtime": runtime(),
                                                   "manifest_sha256": sha256(data / "manifest.json")})
    evaluation_panels = panels(cfg)
    presentations = cfg["evaluation"]["presentations"]
    phase = register(cfg, "reference_development", "Score the preserved 13B base on the development panels.",
                     data={"manifest_sha256": sha256(data / "manifest.json")})
    try:
        status(phase, "running", optimizer_updates=0)
        with budget.session("shared"):
            base = evaluate(ROOT / cfg["parent"]["path"], output / "evaluation" / "base", evaluation_panels,
                            budget.charge, presentations=presentations)
        finish_phase(phase, "completed", optimizer_updates=0, metrics=base["metrics"], next_step="Random arm")
        results = {"base": base}
        for arm in ARMS:
            config = trainer_config(cfg, arm)
            phase = register(cfg, f"train_{arm}",
                             f"Continue the 13B base on the {arm} arm for {cfg['training']['updates']} updates "
                             f"({config['target_tokens']:,} targets) at fixed source shares, then score development.",
                             data={"arm_manifest_sha256": manifest["arm_manifests"][arm]}, recipe=config)
            status(phase, "running", run=config["run_dir"])
            status(output, "running", phase=f"train_{arm}")
            export = train_arm(cfg, arm, budget)
            with budget.session(arm):
                results[arm] = evaluate(export, output / "evaluation" / arm, evaluation_panels, budget.charge,
                                        presentations=presentations)
            state = read(Path(config["run_dir"]) / "status.json")
            failures = guards(results[arm]["metrics"], base["metrics"], cfg["evaluation"]["guards_vs_base"], "base")
            finish_phase(phase, "completed", optimizer_updates=state["step"], main_tokens=state["main_tokens"],
                         source_tokens=state["source_tokens"], export=str(export),
                         weights_sha256=sha256(export / "model.safetensors"), metrics=results[arm]["metrics"],
                         guard_failures_vs_base=failures, next_step="Next arm" if arm != ARMS[-1] else "Decision")
        return conclude(cfg, decide(cfg, results))
    except InterruptedError as error:
        finish_phase(phase, "paused", reason=str(error), next_step="Resume the same study")
        status(output, "paused", reason=str(error))
        return report(cfg)
    except BaseException as error:
        finish_phase(phase, "failed", error=repr(error), next_step="End study; no automatic follow-up")
        return conclude(cfg, {"outcome": "failed", "reason": "execution_stop", "details": repr(error)})


def decide(cfg, results):
    """Predeclared comparison of curated with random at matched exposure."""
    from scglm_study.evaluate import guards, paired_difference
    rules = cfg["evaluation"]
    base = results["base"]["metrics"]
    failures = []
    arm_guards = {arm: guards(results[arm]["metrics"], base, rules["guards_vs_base"], "base") for arm in ARMS}
    failures += [f"curated:{f}" for f in arm_guards["curated"]]
    primary = paired_difference(results, ("curated", "random"), "primary", "primary",
                                replicates=rules["bootstrap_replicates"], seed=rules["bootstrap_seed"])
    if primary["point"] + 1e-12 < rules["min_primary_gain_vs_random"] or primary["lower_95_one_sided"] <= 0:
        failures.append("primary_gain_vs_random")
    heldout = {}
    for source in cfg["source_weights"]:
        stratum = f"heldout_{source}"
        heldout[stratum] = paired_difference(results, ("curated", "random"), "heldout_raw", "raw",
                                             replicates=rules["bootstrap_replicates"], seed=rules["bootstrap_seed"],
                                             stratum=stratum)
        if heldout[stratum]["upper_95_one_sided"] > rules["max_heldout_nll_regression_vs_random"] + 1e-12:
            failures.append(f"heldout_raw:{stratum}")
    versus_base = {arm: paired_difference(results, (arm, "base"), "primary", "primary",
                                          replicates=rules["bootstrap_replicates"], seed=rules["bootstrap_seed"])
                   for arm in ARMS}
    outcome = "curated_preferred" if not failures else "not_established"
    return {"outcome": outcome, "reason": "all_registered_checks_passed" if not failures else "registered_check_failed",
            "details": {"failures": failures, "primary_curated_minus_random": primary,
                        "heldout_nll_curated_minus_random": heldout, "primary_vs_base": versus_base,
                        "guards_vs_base": arm_guards,
                        "endpoints": {arm: results[arm]["registration"]["model_sha256"] for arm in ARMS}}}


def conclude(cfg, decision):
    _, output = roots(cfg)
    immutable(output / "decision.json", {**decision, "replace_existing_checkpoint": False,
                                        "automatic_sft_descendant": False, "automatic_followup": False})
    for arm in ARMS:
        if not (output / "phases" / f"train_{arm}" / "registration.json").exists():
            phase = register(cfg, f"train_{arm}", "Registered arm; an earlier stop prevented it.",
                             data={"blocked_by": decision["reason"]})
            finish_phase(phase, "skipped", optimizer_updates=0, reason=decision["reason"], next_step="None")
    status(output, "completed" if decision["outcome"] in ("curated_preferred", "not_established") else "failed",
           outcome=decision["outcome"], reason=decision["reason"])
    # Execution failures are implementation stops, not evidence about the method.
    if decision["outcome"] == "not_established":
        marker = cfg["experiment_id"]
        append_failed_approach(marker, "A10 — Data-quality mid-training", decision["outcome"], decision["reason"],
                               f"artifacts/{marker}/REPORT.md")
    return report(cfg)


def report(cfg):
    data, output = roots(cfg)
    output.mkdir(parents=True, exist_ok=True)
    decision = read(output / "decision.json") if (output / "decision.json").exists() else {"outcome": "in_progress", "reason": "No final decision"}
    runs = {p.parent.name: read(p) for p in sorted((output / "runs").glob("*/status.json"))} if (output / "runs").exists() else {}
    measurements = {p.parent.name: read(p)["metrics"] for p in (output / "evaluation").glob("*/result.json")} if (output / "evaluation").exists() else {}
    result = {"decision": decision, "runs": runs, "measurements": measurements,
              "budget": read(output / "budget.json") if (output / "budget.json").exists() else None,
              "data_audit": read(data / "audit.json") if (data / "audit.json").exists() else None,
              "checkpoint_aliases_preserved": True,
              "disclosure": "Choice-study development panels were previously observed on the base and SFT models. "
                            "The held-out raw panel is new. No confirmation panel or competition benchmark is scored here."}
    write_json(output / "report.json", result)
    lines = [f"# {cfg['title']}", "", f"Outcome: **{decision['outcome']}**. {decision['reason']}.", "",
             "Both arms continue the preserved 13B base with its AdamW state at fixed 60/30/10 source shares, identical seed, "
             "tokens and schedule. Only document selection differs. Existing checkpoints are unchanged; no SFT descendant is scheduled.", "",
             "[Machine-readable results](report.json) · [Phase reports](phases)", ""]
    for name, st in runs.items():
        lines.append(f"- {name}: {st['status']}; step {st.get('step')}; {st.get('main_tokens', 0):,} targets; "
                     f"source tokens `{json.dumps(st.get('source_tokens', {}))}`.")
    if measurements:
        lines += ["", "| Model | Primary accuracy | Science | Commonsense | Held-out NLL (edu / dclm / wiki) |", "|---|---:|---:|---:|---|"]
        for name, m in measurements.items():
            p = m["primary"]
            held = " / ".join(f"{m['heldout_raw'][f'heldout_{s}']['nll']:.4f}" for s in cfg["source_weights"])
            lines.append(f"| {name} | {p['primary']:.3%} | {p['groups']['science']['accuracy']:.3%} | "
                         f"{p['groups']['commonsense']['accuracy']:.3%} | {held} |")
    lines += ["", "No failed or inconclusive outcome schedules another recipe.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return result
