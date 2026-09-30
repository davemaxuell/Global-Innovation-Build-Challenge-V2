"""Gated orchestration: preflight → no-update audit → references → token_mean arm → prompt_mean arm → decision.

Training starts only if the audit finds enough families with mixed-outcome
prompt groups. Both arms share the audit-frozen families, collection rule,
seed, optimizer and KL settings; only the loss normalization differs. A guard
breach stops that arm; the other arm still runs so the stop is interpretable.
No arm replaces a preserved checkpoint.
"""
import gc
import json
import os
import subprocess
import sys

from scglm_study.common import (ROOT, StudyStopped, append_failed_approach, digest, event, finish_phase, immutable,
                                read, register, roots, runtime, sha256, sources, status, write_json, write_jsonl)
from .loss import ARMS


def panels(cfg):
    from scglm_study.evaluate import choice_panels
    return choice_panels(cfg["evaluation"]["choice_panels"])


def verify(cfg, *, load_model=False):
    for key in ("parent", "base_reference"):
        if sha256(ROOT / cfg[key]["path"] / "model.safetensors") != cfg[key]["weight_sha256"]:
            raise ValueError(f"Preserved weights changed: {key}")
    if sha256(ROOT / cfg["tasks"]["data"] / "manifest.json") != cfg["tasks"]["manifest_sha256"]:
        raise ValueError("Task manifest changed")
    _, output = roots(cfg)
    result = {"verified": True, "read_only": True, "training_started": False,
              "parent_sha256": cfg["parent"]["weight_sha256"], "base_sha256": cfg["base_reference"]["weight_sha256"],
              "audit": read(output / "audit.json") if (output / "audit.json").exists() else None,
              "decision": read(output / "decision.json") if (output / "decision.json").exists() else None}
    if load_model:
        import torch
        from scglm.model import audit_parameters, load_model as load_export
        model = load_export(ROOT / cfg["parent"]["path"])
        with torch.inference_mode():
            if not torch.isfinite(model(input_ids=torch.tensor([[1, 5, 10]])).logits).all():
                raise ValueError("Nonfinite parent CPU reload")
        result["offline_reload"] = audit_parameters(model)
    return result


def preflight(cfg, budget):
    import torch
    from transformers import AutoTokenizer
    from scglm.model import load_model
    from .loss import update
    from .rollouts import groups, sample, weights_digest
    from .tasks import load_pool
    from .train import optimizer
    _, output = roots(cfg)
    receipt = output / "preflight.json"
    contract = {"sources": sources(cfg), "runtime": runtime()}
    if receipt.exists():
        previous = read(receipt)
        if previous["contract"] != contract or previous["status"] != "completed":
            raise ValueError("Preflight contract changed; resolve before resuming")
        return previous
    phase = register(cfg, "preflight", "Run the study tests, then sample a few training prompts from the parent and "
                     "check behavior/training log-probability agreement with a no-update objective pass.",
                     data={"task_manifest_sha256": cfg["tasks"]["manifest_sha256"]})
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
        pool = load_pool(cfg)
        with budget.session("shared"):
            model = load_model(ROOT / cfg["parent"]["path"]).to("cuda:0")
            reference = load_model(ROOT / cfg["parent"]["path"]).to("cuda:0").requires_grad_(False)
            tokenizer = AutoTokenizer.from_pretrained(str(ROOT / cfg["parent"]["path"]), local_files_only=True)
            tasks = [rows[0] for _, rows in sorted(pool.items())]
            rows, stats = sample(model, tokenizer, tasks, cfg["sampling"], seed=1, policy_id=weights_digest(model),
                                 charge=budget.charge)
            batch = groups(rows)
            for g in batch:
                g["informative"] = True  # Exercise every path; zero advantages contribute only KL.
            shares = {f: 1 / len(pool) for f in pool}
            opt = optimizer(model, cfg["training"])
            metrics = {arm: update(model, reference, opt, batch, shares, arm, cfg["training"], charge=budget.charge,
                                   step=False) for arm in ARMS}
            if opt.state:
                raise ValueError("Preflight unexpectedly performed an optimizer step")
            del model, reference, opt
            gc.collect()
            torch.cuda.empty_cache()
        result = {"status": "completed", "contract": contract, "sampling_stats": stats, "no_update_metrics": metrics,
                  "optimizer_updates": 0, "tests_log_sha256": sha256(log), "attempt": str(attempt)}
        immutable(receipt, result)
        finish_phase(phase, "completed", optimizer_updates=0, decision="preflight_passed", next_step="No-update audit",
                     receipt_sha256=sha256(receipt))
        return result
    except BaseException as error:
        finish_phase(phase, "paused" if isinstance(error, InterruptedError) else "failed", error=repr(error),
                     optimizer_updates=0, attempt=str(attempt), next_step="Fix the implementation; no recipe change")
        raise


def audit_phase(cfg, pool, budget):
    import torch
    from transformers import AutoTokenizer
    from scglm.model import load_model
    from .audit import run_audit
    from .rollouts import weights_digest
    _, output = roots(cfg)
    if (output / "audit.json").exists():
        return read(output / "audit.json")
    phase = register(cfg, "audit", f"No-update rollout audit: {cfg['audit']['prompts']} training prompts x "
                     f"{cfg['sampling']['samples_per_prompt']} samples x at most {cfg['sampling']['max_new_tokens']} tokens.",
                     data={"task_manifest_sha256": cfg["tasks"]["manifest_sha256"]}, recipe=cfg["audit"])
    try:
        status(phase, "running", optimizer_updates=0)
        with budget.session("audit"):
            model = load_model(ROOT / cfg["parent"]["path"]).to("cuda:0")
            tokenizer = AutoTokenizer.from_pretrained(str(ROOT / cfg["parent"]["path"]), local_files_only=True)
            result, rows, stats = run_audit(model, tokenizer, pool, cfg, policy_id=weights_digest(model),
                                            charge=budget.charge)
            del model
            torch.cuda.empty_cache()
        write_jsonl(phase / "rollouts.jsonl", rows)
        result = {**result, "sampling_stats": stats, "rollouts_sha256": sha256(phase / "rollouts.jsonl")}
        immutable(output / "audit.json", result)
        finish_phase(phase, "completed", optimizer_updates=0, ready=result["ready"],
                     eligible_families=result["eligible_families"], totals=result["totals"],
                     next_step="Train both arms" if result["ready"] else "Stop: no training")
        return result
    except BaseException as error:
        finish_phase(phase, "failed", error=repr(error), optimizer_updates=0)
        raise


def run(cfg, *, resume=False):
    from scglm_study.budget import Budget
    from scglm_study.evaluate import evaluate, guards
    from .tasks import load_pool
    from .train import ArmTrainer
    _, output = roots(cfg)
    verify(cfg)
    if (output / "decision.json").exists():
        return report(cfg)
    output.mkdir(parents=True, exist_ok=True)
    status(output, "running", phase="preflight", resume=resume)
    budget = Budget(output, cfg["budget"])
    preflight(cfg, budget)
    immutable(output / "execution_contract.json", {"sources": sources(cfg), "runtime": runtime()})
    pool = load_pool(cfg)
    audit = audit_phase(cfg, pool, budget)
    if not audit["ready"]:
        return conclude(cfg, {"outcome": "not_ready", "reason": "audit_gate", "details": {
            "eligible_families": audit["eligible_families"],
            "families": {f: {k: v[k] for k in ("mixed_fraction", "sample_accuracy", "reasons")} for f, v in audit["families"].items()}}})
    evaluation_panels, presentations = panels(cfg), cfg["evaluation"]["presentations"]
    rules = cfg["evaluation"]
    phase = register(cfg, "reference_development", "Score the selected SFT parent and the 13B base on development panels.")
    try:
        status(phase, "running", optimizer_updates=0)
        references = {}
        for name, key in (("parent", "parent"), ("base", "base_reference")):
            with budget.session("shared"):
                references[name] = evaluate(ROOT / cfg[key]["path"], output / "evaluation" / name, evaluation_panels,
                                            budget.charge, presentations=presentations)
        finish_phase(phase, "completed", optimizer_updates=0,
                     metrics={n: r["metrics"] for n, r in references.items()}, next_step="token_mean arm")
    except BaseException as error:
        finish_phase(phase, "paused" if isinstance(error, InterruptedError) else "failed", error=repr(error))
        if isinstance(error, InterruptedError):
            status(output, "paused", reason=str(error))
            return report(cfg)
        return conclude(cfg, {"outcome": "failed", "reason": "reference_scoring", "details": repr(error)})
    results = dict(references)
    for arm in ARMS:
        phase = register(cfg, f"train_{arm}",
                         f"Train the {arm} arm from the selected SFT for at most {cfg['training']['updates']} actual updates "
                         f"with deficit-based collection; development checks at {rules['development_checks_at_updates']}.",
                         data={"audit_sha256": digest(audit), "family_shares": audit["family_shares"]},
                         recipe={**cfg["training"], "arm": arm, "sampling": cfg["sampling"]})
        run_dir = output / "runs" / arm
        state = read(run_dir / "status.json")["status"] if (run_dir / "status.json").exists() else None
        if state == "completed":
            results[arm] = evaluate_final(cfg, arm)
            continue
        if state in ("stopped", "failed"):
            continue
        status(phase, "running", run=str(run_dir))
        status(output, "running", phase=f"train_{arm}")

        def development(path, trainer, arm=arm, phase=phase):
            result = evaluate(path, output / "evaluation" / f"{arm}_step_{trainer.state.step:07d}", evaluation_panels,
                              budget.charge, presentations=presentations, model=trainer.model)
            failures = guards(result["metrics"], references["parent"]["metrics"], rules["guards_vs_parent"], "parent")
            event(phase, "development_guard", step=trainer.state.step, failures=failures)
            if failures:
                raise StudyStopped("Development guard breach: " + ", ".join(failures))
        try:
            with budget.session(arm):
                trainer = ArmTrainer(cfg, arm, audit, pool, budget, development)
                export = trainer.run()
                counters, ledger = dict(trainer.state.counters), dict(trainer.ledger.value)
                del trainer
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            results[arm] = evaluate_final(cfg, arm)
            finish_phase(phase, "completed", optimizer_updates=cfg["training"]["updates"], counters=counters,
                         ledger=ledger, export=str(export), weights_sha256=sha256(export / "model.safetensors"),
                         metrics=results[arm]["metrics"], next_step="Next arm" if arm != ARMS[-1] else "Decision")
        except InterruptedError as error:
            finish_phase(phase, "paused", reason=str(error), next_step="Resume the same study")
            status(output, "paused", reason=str(error))
            return report(cfg)
        except StudyStopped as error:
            actual = read(run_dir / "status.json") if (run_dir / "status.json").exists() else {}
            finish_phase(phase, "stopped", reason=str(error), actual=actual,
                         next_step="Run the other arm; this arm is not resumed")
        except BaseException as error:
            finish_phase(phase, "failed", error=repr(error), next_step="End study; no automatic follow-up")
            return conclude(cfg, {"outcome": "failed", "reason": f"execution_stop:{arm}", "details": repr(error)})
    return conclude(cfg, decide(cfg, results))


def evaluate_final(cfg, arm):
    from scglm_study.evaluate import load_result
    _, output = roots(cfg)
    final = cfg["training"]["updates"]
    return load_result(output / "evaluation" / f"{arm}_step_{final:07d}")


def decide(cfg, results):
    from scglm_study.evaluate import paired_difference, task_bootstrap
    rules = cfg["evaluation"]
    families = sorted(read(roots(cfg)[1] / "audit.json")["family_shares"])
    completed = [arm for arm in ARMS if arm in results]
    details = {"completed_arms": completed, "families": families}
    kwargs = {"replicates": rules["bootstrap_replicates"], "seed": rules["bootstrap_seed"]}
    for arm in completed:
        details[f"{arm}_task_vs_parent"] = task_bootstrap(results, (arm, "parent"), "instructions", families=families, **kwargs)
        details[f"{arm}_primary_vs_parent"] = paired_difference(results, (arm, "parent"), "primary", "primary", **kwargs)
    improves = [arm for arm in completed
                if details[f"{arm}_task_vs_parent"]["lower_95_one_sided"] > 0
                and details[f"{arm}_primary_vs_parent"]["point"] >= rules["guards_vs_parent"]["accuracy_delta_min"]]
    details["arms_improving_parent_task_accuracy"] = improves
    if len(completed) < 2:
        return {"outcome": "inconclusive", "reason": "an_arm_did_not_complete", "details": details}
    task = task_bootstrap(results, ("prompt_mean", "token_mean"), "instructions", families=families, **kwargs)
    primary = paired_difference(results, ("prompt_mean", "token_mean"), "primary", "primary", **kwargs)
    details.update(task_prompt_minus_token=task, primary_prompt_minus_token=primary)
    minimum = rules["min_task_gain_between_arms"]
    if task["point"] >= minimum and task["lower_95_one_sided"] > 0 and primary["point"] >= -rules["max_primary_loss_between_arms"]:
        outcome = "prompt_mean_preferred"
    elif -task["point"] >= minimum and task["upper_95_one_sided"] < 0 and -primary["point"] >= -rules["max_primary_loss_between_arms"]:
        outcome = "token_mean_preferred"
    else:
        outcome = "no_difference_established"
    return {"outcome": outcome, "reason": "matched_update_comparison", "details": details}


def conclude(cfg, decision):
    _, output = roots(cfg)
    immutable(output / "decision.json", {**decision, "replace_existing_checkpoint": False, "automatic_followup": False})
    for name in ("audit", "reference_development", *(f"train_{arm}" for arm in ARMS)):
        if not (output / "phases" / name / "registration.json").exists():
            phase = register(cfg, name, "Registered conditional phase; an earlier gate prevented it.",
                             data={"blocked_by": decision["reason"]})
            finish_phase(phase, "skipped", optimizer_updates=0, reason=decision["reason"], next_step="None")
    status(output, "failed" if decision["outcome"] == "failed" else "completed", outcome=decision["outcome"],
           reason=decision["reason"])
    details = decision["details"] if isinstance(decision.get("details"), dict) else {}
    if not details.get("arms_improving_parent_task_accuracy"):
        append_failed_approach(cfg["experiment_id"], "A11 — Revised small RL study", decision["outcome"],
                               decision["reason"], f"artifacts/{cfg['experiment_id']}/REPORT.md")
    return report(cfg)


def report(cfg):
    _, output = roots(cfg)
    output.mkdir(parents=True, exist_ok=True)
    decision = read(output / "decision.json") if (output / "decision.json").exists() else {"outcome": "in_progress", "reason": "No final decision"}
    runs = {p.parent.name: read(p) for p in sorted((output / "runs").glob("*/status.json"))} if (output / "runs").exists() else {}
    ledgers = {p.parent.name: read(p) for p in sorted((output / "runs").glob("*/ledger.json"))} if (output / "runs").exists() else {}
    measurements = {p.parent.name: read(p)["metrics"] for p in (output / "evaluation").glob("*/result.json")} if (output / "evaluation").exists() else {}
    audit = read(output / "audit.json") if (output / "audit.json").exists() else None
    result = {"decision": decision, "audit": audit, "runs": runs, "collection_ledgers": ledgers,
              "measurements": measurements, "budget": read(output / "budget.json") if (output / "budget.json").exists() else None,
              "checkpoint_aliases_preserved": True,
              "disclosure": "Choice-study development panels (including the historical instruction panel) were previously "
                            "observed on the parent and base. No confirmation panel or competition benchmark is scored here."}
    write_json(output / "report.json", result)
    lines = [f"# {cfg['title']}", "", f"Outcome: **{decision['outcome']}**. {decision['reason']}.", "",
             "Both arms start from the selected SFT with the same audit-frozen families, deficit-based collection, seed, "
             "optimizer and KL settings; only the loss normalization differs. Existing checkpoints are unchanged.", "",
             "[Machine-readable results](report.json) · [Phase reports](phases)", ""]
    if audit:
        lines += ["| Family | Mixed groups | Sample reward | Semantic accuracy | Termination | Eligible |", "|---|---:|---:|---:|---:|---|"]
        for family, v in audit["families"].items():
            lines.append(f"| {family} | {v['mixed_fraction']:.1%} | {v['sample_accuracy']:.1%} | {v['semantic_accuracy']:.1%} | "
                         f"{v['termination_rate']:.1%} | {'yes' if v['eligible'] else ', '.join(v['reasons'])} |")
        lines.append("")
    for name, st in runs.items():
        lines.append(f"- {name}: {st['status']}; {st.get('optimizer_updates', 0)} actual updates; "
                     f"collection `{json.dumps(ledgers.get(name, {}))}`.")
    if measurements:
        lines += ["", "| Evaluation | Primary accuracy | Task mean (generation) |", "|---|---:|---:|"]
        for name, m in sorted(measurements.items()):
            lines.append(f"| {name} | {m['primary']['primary']:.3%} | {m['instructions']['mean']:.3%} |")
    lines += ["", "No failed or inconclusive outcome schedules another recipe.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return result
