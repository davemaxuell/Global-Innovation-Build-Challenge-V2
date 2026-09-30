"""Fixed gated study orchestration and independent completion reporting."""
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import (ROOT, SEEDS, ARMS, roots, read, read_jsonl, write_json, sha256, digest, sources, runtime,
                     immutable, register, finish_phase, status, event, StudyStopped)


def verify(cfg, *, load_model=False):
    for ingredient in cfg["data"]["ingredients"]:
        if sha256(ROOT / ingredient["path"]) != ingredient["sha256"]:
            raise ValueError("Training ingredient changed")
    for key in ("parent", "base_reference"):
        if sha256(ROOT / cfg[key]["path"] / "model.safetensors") != cfg[key]["weight_sha256"]:
            raise ValueError("Preserved weights changed")
    data, output = roots(cfg)
    result = {"verified": True, "read_only": True, "training_started": False,
              "parent_sha256": cfg["parent"]["weight_sha256"], "base_sha256": cfg["base_reference"]["weight_sha256"],
              "prepared": (data / "manifest.json").exists()}
    if result["prepared"]:
        from .prepare import load
        manifest = load(cfg, require_ready=False)
        result.update(preparation_status=manifest["status"], blockers=manifest["blockers"],
                      manifest_sha256=sha256(data / "manifest.json"), planned_positions=manifest["planned_positions"])
    if load_model:
        import torch
        from scglm.model import load_model as load_export, audit_parameters
        from .lineage import references
        result["lineage"] = references(cfg)
        model = load_export(ROOT / cfg["parent"]["path"])
        with torch.inference_mode():
            if not torch.isfinite(model(input_ids=torch.tensor([[1, 5, 10]])).logits).all():
                raise ValueError("Nonfinite parent CPU reload")
        result["offline_reload"] = audit_parameters(model)
    return result


def preflight(cfg, manifest, budget):
    from .objective import update
    from .train import optimizer
    from .lineage import references
    import torch
    from scglm.model import load_model, save_model, audit_parameters
    from transformers import AutoTokenizer, AutoModelForCausalLM
    data, output = roots(cfg)
    phase = register(cfg, "preflight", "Verify CPU/GPU mathematics, recoverability, actual export loader and unchanged official reporting preparation.",
                     data={"manifest_sha256": sha256(data / "manifest.json")})
    contract = {"sources": sources(), "runtime": runtime(), "manifest_sha256": sha256(data / "manifest.json"),
                "tests": {p.name:sha256(p) for p in sorted((ROOT / "tests").glob("test_choice*.py"))}}
    receipt = output / "preflight.json"
    if receipt.exists():
        previous = read(receipt)
        if previous["contract"] != contract or previous["status"] != "completed":
            raise ValueError("Preflight contract changed")
        return previous
    attempt = phase / f"attempt_{len(list(phase.glob('attempt_*')))+1:03d}"
    attempt.mkdir()
    try:
        status(phase, "running")
        test_log = attempt / "tests.log"
        with test_log.open("w") as stream:
            completed = subprocess.run([sys.executable,"-m","pytest","-q","tests"], cwd=ROOT,
                                       env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"OMP_NUM_THREADS":"4"},
                                       stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise ValueError(f"Correctness/regression tests failed: {test_log}")
        refs = references(cfg)
        with budget.session("shared"):
            model = load_model(ROOT / cfg["parent"]["path"]).to("cuda:0")
            tok = AutoTokenizer.from_pretrained(str(ROOT/cfg["parent"]["path"]), local_files_only=True)
            params = audit_parameters(model)
            rows = {r["id"]:r for r in read_jsonl(data/"responses.jsonl")}
            replay = {r["id"]:r for r in read_jsonl(data/"replay.jsonl")}
            choices = {r["id"]:r for r in read_jsonl(data/"choices.jsonl")}
            b = read(data/f"plan_{SEEDS[0]}.json")["batches"][0]
            opt = optimizer(model)
            metrics = {}
            for candidate in (False, True):
                metrics[str(candidate)] = update(model,opt,[rows[i] for i in b["response_ids"]],
                    [replay[i] for i in b["replay_ids"]], [choices[i] for i in b["choice_ids"]],
                    candidate=candidate, charge=budget.charge, step=False)
            # No optimizer state should have been created by a no-update preflight.
            if opt.state:
                raise ValueError("Preflight unexpectedly performed an optimizer step")
            export = attempt / "export"
            save_model(model,export)
            tok.save_pretrained(str(export))
            del opt, model
            gc.collect()
            torch.cuda.empty_cache()
            reload = AutoModelForCausalLM.from_pretrained(str(export), local_files_only=True).to("cuda:0")
            budget.charge(3,"preflight_export_reload")
            with torch.inference_mode():
                if not torch.isfinite(reload(input_ids=torch.tensor([[1,5,10]],device="cuda:0")).logits).all():
                    raise ValueError("Export reload produced nonfinite outputs")
            if reload.get_input_embeddings().weight is not reload.get_output_embeddings().weight:
                raise ValueError("Export reload lost tied weights")
            del reload
            torch.cuda.empty_cache()
            # Tiny CUDA recovery tests use a separate disposable model/output.
            gpu_log = attempt / "gpu-tests.log"
            gpu_counts = attempt / "gpu-test-counts.json"
            with gpu_log.open("w") as stream:
                command = [sys.executable,"-m","pytest","-q","tests/test_choice_pipeline.py","-k","cuda_recovery"]
                test = subprocess.run(command,cwd=ROOT,env={**os.environ,"SCGLM_CHOICE_GPU_TEST":"1","SCGLM_CHOICE_TEST_COUNTS":str(gpu_counts)},stdout=stream,stderr=subprocess.STDOUT,
                                      timeout=max(1,budget.remaining("shared")))
            if gpu_counts.exists():
                budget.charge(read(gpu_counts)["positions"],"disposable_gpu_recovery")
            if test.returncode:
                raise ValueError("Assigned-GPU recovery check failed")
            # The actual unchanged scorer validates datasets and reporting metadata
            # without opening candidate scores or the reserved study panels.
            official = attempt / "official_preparation"
            with (attempt/"official-preparation.log").open("w") as stream:
                result = subprocess.run([sys.executable,str(ROOT/"scripts/run_final_evaluation.py"),"--prepare-only",
                    "--output",str(official),"--history-dir",str(output/"official_preparation_history")],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,budget.remaining("shared")))
            if result.returncode:
                raise ValueError("Unchanged official scorer preparation failed")
        result = {"status":"completed", "contract":contract, "references":refs, "parameter_audit":params,
                  "training_only_no_update_metrics":metrics, "optimizer_updates":0,
                  "tests_log_sha256":sha256(test_log), "gpu_tests_log_sha256":sha256(gpu_log), "attempt":str(attempt),
                  "budget":budget.value}
        if contract["sources"] != sources() or contract["runtime"] != runtime():
            raise ValueError("Implementation changed during preflight")
        immutable(receipt,result)
        finish_phase(phase,"completed",optimizer_updates=0,decision="preflight_passed",next_step="Pilot control then candidate",receipt_sha256=sha256(receipt))
        return result
    except BaseException as error:
        finish_phase(phase,"paused" if isinstance(error,InterruptedError) else "failed",error=repr(error),optimizer_updates=0,
                     attempt=str(attempt),next_step="Fix implementation before model evaluation; no recipe change")
        raise


def run(cfg, *, resume=False):
    import torch
    from .prepare import load, prepare
    from .budget import Budget
    from .train import Trainer
    from .evaluate import evaluate, load_result
    from .selection import retention, pilot, replication, claim_confirmation, confirmation, official_selection
    from .lineage import endpoint
    data, output = roots(cfg)
    verify(cfg)
    if (output/"decision.json").exists():
        return report(cfg)
    status(output,"running",phase="preparation_and_preflight",resume=resume)
    manifest = load(cfg,require_ready=False) if (data/"manifest.json").exists() else prepare(cfg)
    if manifest["status"] != "complete":
        conclude(cfg,"failed","preparation_requirements",details=manifest["blockers"])
        return report(cfg)
    budget = Budget(output,cfg)
    preflight(cfg,manifest,budget)
    frozen_contract = {"sources":sources(),"runtime":runtime(),"manifest_sha256":sha256(data/"manifest.json")}
    immutable(output/"execution_contract.json",frozen_contract)
    phase = register(cfg,"reference_development","Score the frozen parent and scratch base on development only.",data=frozen_contract)
    try:
        status(phase,"running",optimizer_updates=0)
        status(output,"running",phase="reference_development")
        refs = {}
        for name,key in (("parent","parent"),("base","base_reference")):
            with budget.session("shared"):
                refs[name] = evaluate(cfg,manifest,ROOT/cfg[key]["path"],"development",output/"evaluation"/"development"/name,budget.charge)
        finish_phase(phase,"completed",optimizer_updates=0,results={n:r["metrics"] for n,r in refs.items()},next_step="Pilot")
        members, models, hashes = [], {}, {}
        for seed in SEEDS:
            pair = {}
            for arm in ARMS:
                category = f"{arm}_seed{seed}"
                phase = register(cfg,category,f"Train {arm} at seed {seed} for exactly 32 eligible updates, with development guards at 8/16/24/32.",
                                 data={"manifest_sha256":sha256(data/"manifest.json"),"plan_sha256":manifest["plans"][str(seed)]},
                                 recipe={**cfg["optimizer"],"arm":arm,"seed":seed,"objectives":cfg["objectives"]})
                status(phase,"running",run=str(output/"runs"/category))
                status(output,"running",phase=category)
                def development(path, trainer):
                    result = evaluate(cfg,manifest,path,"development",output/"evaluation"/"development"/category/f"step_{trainer.state.step:07d}",
                                      budget.charge,model=trainer.model)
                    failures = retention(result["metrics"],refs["parent"]["metrics"],refs["base"]["metrics"],cfg=cfg)
                    event(phase,"development_guard",step=trainer.state.step,failures=failures,result_sha256=sha256(Path(result["directory"])/"result.json"))
                    if failures:
                        raise StudyStopped("Development retention breach: " + ", ".join(failures))
                with budget.session(category):
                    trainer = Trainer(cfg,seed,arm,manifest,budget,development)
                    path = trainer.run()
                    counts = dict(trainer.state.counters)
                    del trainer
                    gc.collect()
                    torch.cuda.empty_cache()
                identity = endpoint(path,cfg,eligible=True)
                pair["control" if arm == "ce_control" else "candidate"] = load_result(output/"evaluation"/"development"/category/"step_0000032")["metrics"]
                models[category], hashes[category] = path, identity["weights_sha256"]
                finish_phase(phase,"completed",optimizer_updates=32,counters=counts,checkpoint=str(path),weights_sha256=hashes[category],
                             evaluation=pair["control" if arm == "ce_control" else "candidate"],next_step="Registered paired gate")
            # The shared plan hash alone is not enough: require both final runs
            # to attest the exact same complete ordered batches and schedule.
            a,b = [read(output/"runs"/f"{arm}_seed{seed}"/"run.json")["contract"] for arm in ARMS]
            if any(a[k]!=b[k] for k in ("parent_sha256","manifest_sha256","plan_sha256","seed","sources","runtime")):
                raise StudyStopped("Invalid paired training exposure")
            members.append({"seed":seed,**pair})
            if seed == SEEDS[0]:
                gate = pilot(pair["candidate"],pair["control"],refs["parent"]["metrics"],refs["base"]["metrics"],cfg)
                immutable(output/"pilot_gate.json",gate)
                if not gate["passed"]:
                    conclude(cfg,"inconclusive","pilot_gate",details=gate)
                    return report(cfg)
        gate = replication(members,refs["parent"]["metrics"],refs["base"]["metrics"],cfg)
        immutable(output/"replication_gate.json",gate)
        if not gate["passed"]:
            conclude(cfg,"inconclusive","replication_gate",details=gate)
            return report(cfg)
        pilot_candidate, pilot_control = [f"{arm}_seed{SEEDS[0]}" for arm in ("choice_candidate","ce_control")]
        frozen = {"candidate":hashes[pilot_candidate],"control":hashes[pilot_control],
                  "parent":cfg["parent"]["weight_sha256"],"base":cfg["base_reference"]["weight_sha256"]}
        immutable(output/"frozen_models.json",frozen)
        phase = register(cfg,"confirmation","Claim the reserved panel once for the four frozen model hashes; no substitutions.",data={"manifest_sha256":sha256(data/"manifest.json"),"models":frozen})
        status(phase,"running",optimizer_updates=0)
        status(output,"running",phase="confirmation")
        claim = claim_confirmation(manifest,frozen,ROOT/"artifacts/choice_confirmation_claims",data)
        paths = {"candidate":models[pilot_candidate],"control":models[pilot_control],"parent":ROOT/cfg["parent"]["path"],"base":ROOT/cfg["base_reference"]["path"]}
        results = {}
        for name,path in paths.items():
            with budget.session("shared"):
                results[name] = evaluate(cfg,manifest,path,"confirmation",output/"evaluation"/"confirmation"/name,budget.charge,claim=claim)
        gate = confirmation(results,cfg)
        immutable(output/"confirmation_gate.json",gate)
        finish_phase(phase,"completed",decision=gate,optimizer_updates=0,next_step="Official reporting" if gate["passed"] else "End study")
        if not gate["passed"]:
            conclude(cfg,"inconclusive","confirmation_gate",details=gate)
            return report(cfg)
        selection = official_selection(models,hashes,pilot_candidate)
        immutable(output/"official_selection.json",selection)
        phase = register(cfg,"official_reporting","Run the unchanged competition scorer for all six completed endpoints; disclose historical base and SFT results.",data=selection)
        status(phase,"running",optimizer_updates=0)
        status(output,"running",phase="official_reporting")
        for name,path in models.items():
            dest = output/"official"/name
            if (dest/"summary.json").exists():
                summary = read(dest/"summary.json")
                if summary["artifact_sha256"]["model.safetensors"] != hashes[name]:
                    raise ValueError("Official report model identity changed")
                continue
            with budget.session("shared"):
                # Every attempt is immutable; unfinished official scoring never
                # overwrites a prior attempt. All repeated work remains charged.
                attempts = output/"official_attempts"/name
                attempts.mkdir(parents=True,exist_ok=True)
                attempt = attempts/f"attempt_{len(list(attempts.iterdir()))+1:03d}"
                command = [sys.executable,str(ROOT/"scripts/run_final_evaluation.py"),"--model",str(path),
                    "--selection-record",str(output/"official_selection.json"),"--final-evaluation","--device","cuda:0",
                    "--output",str(attempt),"--history-dir",str(output/"official_history"),"--label",name]
                execute_official(command,output/f"{name}_official.log",budget)
                summary = read(attempt/"summary.json")
                dest.mkdir(parents=True,exist_ok=True)
                immutable(dest/"summary.json",summary)
        finish_phase(phase,"completed",optimizer_updates=0,next_step="Report efficacy and every benchmark regression")
        conclude(cfg,"qualifying","all_registered_gates_passed",details={"candidate":str(models[pilot_candidate]),"weights_sha256":hashes[pilot_candidate]})
    except InterruptedError as error:
        finish_phase(phase,"paused",reason=str(error),next_step="Resume same contracts and cumulative budget")
        status(output,"paused",reason=str(error))
        return report(cfg)
    except BaseException as error:
        run_status = output / "runs" / phase.name / "status.json"
        actual = read(run_status) if run_status.exists() else {"optimizer_updates":0}
        finish_phase(phase,"failed",error=repr(error),actual=actual,next_step="End study; no automatic recipe follow-up")
        conclude(cfg,"failed","execution_or_retention_stop",details=repr(error))
        return report(cfg)
    return report(cfg)


def execute_official(command, log, budget):
    # Own process group ensures timeout/error cannot leave an unaccounted GPU child.
    with log.open("a") as stream:
        child = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = child.wait(timeout=max(1,budget.remaining("shared")))
            if code:
                raise StudyStopped(f"Official scorer failed with exit {code}; see {log}")
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()


def conclude(cfg, outcome, reason, *, details):
    _, output = roots(cfg)
    immutable(output/"decision.json",{"outcome":outcome,"reason":reason,"details":details,
              "replace_existing_checkpoint":False,"automatic_followup":False})
    for name in [*(f"{a}_seed{s}" for s in SEEDS for a in ARMS),"confirmation","official_reporting"]:
        if not (output/"phases"/name/"registration.json").exists():
            phase = register(cfg,name,"Conditional registered study phase; prerequisite gate did not pass.",data={"blocked_by":reason})
            finish_phase(phase,"skipped",optimizer_updates=0,scored_tokens=0,reason=reason,next_step="None")
    status(output,"completed" if outcome == "qualifying" else "failed",outcome=outcome,reason=reason)
    if outcome != "qualifying":
        record = ROOT/"FAILED_APPROACHES.md"
        marker = "choice_discrimination_20260929_v1"
        if marker not in record.read_text():
            with record.open("a") as stream:
                stream.write(f"\n## A09 — Choice-discrimination study ({marker})\n\n"
                    f"Outcome: **{outcome}**. Stop: `{reason}`. [Full evidence and actual counts](artifacts/{marker}/REPORT.md). "
                    "No selected checkpoint was replaced and no further recipe is scheduled. An incomplete or unlaunched arm is not evidence that the choice objective failed.\n")


def report(cfg):
    data, output = roots(cfg)
    decision = read(output/"decision.json") if (output/"decision.json").exists() else {"outcome":"in_progress","reason":"No final decision"}
    runs = {}
    for path in sorted((output/"runs").glob("*/status.json")) if (output/"runs").exists() else []:
        runs[path.parent.name] = read(path)
    historical = {}
    for name,spec in cfg["historical_official"].items():
        path = ROOT/spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError("Historical official report changed")
        historical[name] = {"previously_observed":True,"path":spec["path"],"sha256":spec["sha256"],"metrics":read(path)["metrics"]}
    benchmarks = {p.parent.name:read(p)["metrics"] for p in (output/"official").glob("*/summary.json")} if (output/"official").exists() else {}
    gates = {name:read(output/(name+"_gate.json")) for name in ("pilot","replication","confirmation") if (output/(name+"_gate.json")).exists()}
    measurements = {str(p.parent.relative_to(output)):read(p) for p in (output/"evaluation").rglob("result.json")} if (output/"evaluation").exists() else {}
    checkpoints = {}
    for p in (output/"runs").glob("*/endpoints/*/export/endpoint.json") if (output/"runs").exists() else []:
        prov=read(p.parent/"training_provenance.json")
        checkpoints[str(p.parent.relative_to(output))]={**read(p),"optimizer_updates":prov["optimizer_updates"],"counters":prov["counters"]}
    result = {"decision":decision,"runs":runs,"budget":read(output/"budget.json") if (output/"budget.json").exists() else {"gpu_seconds":0},
              "historical_official":historical,"study_official":benchmarks,
              "gates":gates,"measurements":measurements,"checkpoints":checkpoints,
              "actual_study_updates_including_discarded":sum(r.get("actual_optimizer_attempts",{}).get("completed",r.get("optimizer_updates",0)) for r in runs.values()),
              "choice_loss_helped_beyond_control":True if decision["outcome"] == "qualifying" else "not established",
              "checkpoint_aliases_preserved":True}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"report.json",result)
    lines = ["# Choice-discrimination study", "", f"Outcome: **{decision['outcome']}**. {decision['reason']}.", "",
             f"Added choice-loss efficacy beyond matched CE/replay: **{result['choice_loss_helped_beyond_control']}**.", "",
             "Only update 32 was eligible; deployment seed remained 20260929. Existing base and selected SFT exports were preserved. "
             "Historical development, confirmation and official scores were previously observed. COPA and Winograd results are diagnostic only.", "",
             "[Complete machine-readable results, counters, checksums and resource accounting](report.json) · [Phase reports](phases)", ""]
    for name,st in runs.items():
        lines += [f"- {name}: {st['status']}; {st.get('optimizer_updates',0)} updates; counters `{json.dumps(st.get('counters',{}))}`."]
    if not runs:
        lines += ["No study training endpoint was produced."]
    if gates:
        lines += ["", "Registered gate results:", ""]
        for name,gate in gates.items():
            lines += [f"- [{name.capitalize()} gate]({name}_gate.json): {'passed' if gate['passed'] else 'did not pass'}."]
    if measurements:
        lines += ["", "| Evaluation | Primary accuracy | Science | Commonsense |", "|---|---:|---:|---:|"]
        for name,r in measurements.items():
            m=r["metrics"]
            lines.append(f"| [{name}]({name}/result.json) | {m['primary']:.3%} | {m['domains']['science']['accuracy']:.3%} | {m['domains']['commonsense']['accuracy']:.3%} |")
    if not benchmarks:
        lines += ["", "No new official benchmark scores were computed; no benchmark-improvement claim is made."]
    else:
        lines += ["", "| Endpoint | Benchmark | Raw accuracy | Change vs parent | Change vs base |", "|---|---|---:|---:|---:|"]
        for name,metrics in benchmarks.items():
            for task,v in metrics["benchmarks"].items():
                a=v["acc"];p=historical["parent"]["metrics"]["benchmarks"][task]["acc"];b=historical["base"]["metrics"]["benchmarks"][task]["acc"]
                lines.append(f"| {name} | {task} | {a:.4%} | {a-p:+.4%} | {a-b:+.4%} |")
            lines.append(f"\n{name} WikiText-103 perplexity: {metrics['wikitext103']['perplexity']:.6f}. Normalized accuracies are included in report.json.")
    lines += ["", "No failed or inconclusive outcome schedules another recipe. No final benchmark scores may reopen selection.", ""]
    (output/"REPORT.md").write_text("\n".join(lines))
    return result
