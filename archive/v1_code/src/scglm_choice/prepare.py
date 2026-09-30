"""Prepare and freeze complete study data before any candidate evaluation."""
from collections import defaultdict
import shutil

from transformers import AutoTokenizer
from scglm_pipeline.sota_data import load as load_retained
from scglm_pipeline.task_registry import prompt_text
from .common import (ROOT, SEEDS, config, roots, read, read_jsonl, write_json, write_jsonl,
                     sha256, digest, immutable, register, finish_phase, event, StudyStopped)
from .plans import training_choices, make_plan
from .data import Exposure, primary_sources, primary_panels, raw_panels, transfer_sources
from .budget import evaluation_positions


def load(cfg, *, require_ready=True):
    directory, _ = roots(cfg)
    manifest = read(directory / "manifest.json")
    if manifest["config_sha256"] != digest(cfg):
        raise ValueError("Prepared configuration changed")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"Prepared study file changed: {name}")
    if require_ready and manifest["status"] != "complete":
        raise StudyStopped("Preparation did not meet launch prerequisites: " + "; ".join(manifest["blockers"]))
    return manifest


def prepare(cfg):
    directory, output = roots(cfg)
    phase = register(cfg, "preparation", "Freeze audited data, evaluation memberships and three matched batch plans; no model scoring.",
                     data=cfg["data"]["ingredients"])
    if (directory / "manifest.json").exists():
        return load(cfg, require_ready=False)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "sources"
    cache.mkdir(exist_ok=True)
    for item in cfg["data"]["ingredients"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Registered training ingredient changed")
    old = ROOT / "data/posttraining/sota_v1"
    load_retained(old / "manifest.json")
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / cfg["parent"]["path"]), local_files_only=True)
    rows, replay = read_jsonl(old / "train.jsonl"), read_jsonl(old / "replay.jsonl")
    choices = training_choices(rows, tokenizer)
    anchors = {}
    plans = {}
    for seed in SEEDS:
        plan, new = make_plan(rows, replay, choices, seed, cfg)
        plans[str(seed)] = plan
        anchors.update({r["id"]: r for r in new})
        write_json(directory / f"plan_{seed}.json", plan)
    write_jsonl(directory / "responses.jsonl", rows + list(anchors.values()))
    write_jsonl(directory / "choices.jsonl", choices)
    write_jsonl(directory / "replay.jsonl", replay)
    instructions = defaultdict(list)
    for row in read_jsonl(old / "development.jsonl"):
        if row["verifier"] != "none" and len(tokenizer.encode(prompt_text(row), add_special_tokens=False)) + 128 <= 1024:
            instructions[row["family"]].append(row)
    diagnostic = []
    for family, pool in sorted(instructions.items()):
        pool.sort(key=lambda r: digest([20260929, r["id"]]))
        diagnostic.extend({**r, "historical_retention_diagnostic": True} for r in pool[:100])
    write_jsonl(directory / "instructions.jsonl", diagnostic)
    event(phase, "plans_prepared", plans={s: {"response_targets": p["response_targets"],
          "ce_positions": p["ce_positions"], "choice_positions": p["choice_positions"]} for s,p in plans.items()})
    exposure = Exposure(tokenizer, phase)
    blockers, audits, source_inventory = [], {}, {}
    try:
        for name, build in (
            ("primary", lambda: _primary(cfg, cache, exposure, tokenizer)),
            ("raw", lambda: raw_panels(cfg, exposure, tokenizer, phase)),
            ("transfer", lambda: transfer_sources(cfg, cache, exposure, tokenizer)),
        ):
            try:
                panels, audit = build()
                for split, records in panels.items():
                    write_jsonl(directory / f"{name}_{split}.jsonl", records)
                audits[name] = audit
                if name == "primary":
                    source_inventory = audit.pop("sources")
                event(phase, "panel_prepared", kind=name, counts={s:len(r) for s,r in panels.items()})
            except (ValueError, OSError) as error:
                if name == "transfer":
                    audits[name] = {"available": False, "reason": repr(error)}
                    for s in ("development", "confirmation"):
                        write_jsonl(directory / f"transfer_{s}.jsonl", [])
                else:
                    blockers.append(f"{name}: {error}")
                    event(phase, "mandatory_panel_failed", kind=name, error=repr(error))
        audits["historical_exposure"] = {"files": exposure.files, "documents": len(exposure.doc_ids),
                                         "conservative_prepared_pool_exclusion": True}
    finally:
        exposure.close()
    planned = {}
    if not blockers:
        panels = {k: read_jsonl(directory / f"{k}_development.jsonl") for k in ("primary", "raw", "transfer")}
        eval_cost = evaluation_positions(panels["primary"], panels["raw"], diagnostic, panels["transfer"], tokenizer, cfg["presentations"])
        for seed, plan in plans.items():
            planned[seed] = {"control": plan["ce_positions"] + 4*eval_cost,
                             "candidate": plan["ce_positions"] + plan["choice_positions"] + 4*eval_cost,
                             "single_development_forward_positions_max": eval_cost}
            if planned[seed]["candidate"] > 10_000_000:
                blockers.append(f"Seed {seed}: mandatory training/development needs {planned[seed]['candidate']} positions > 10000000")
        audit_isolation(directory)
    write_json(directory / "audit.json", audits)
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    manifest = {"schema": "scglm-choice-data-v1", "status": "blocked" if blockers else "complete",
                "blockers": blockers, "config_sha256": digest(cfg), "files": files,
                "source_revisions_and_licenses": source_inventory, "tokenizer_sha256": sha256(ROOT / cfg["parent"]["path"] / "tokenizer.json"),
                "plans": {s: digest(p) for s,p in plans.items()}, "planned_positions": planned,
                "historical_panels_previously_observed": True,
                "confirmation_membership_sha256": digest([files.get("primary_confirmation.jsonl"), files.get("raw_confirmation.jsonl"), files.get("transfer_confirmation.jsonl")]),
                "instruction_diagnostics": {f: min(100,len(r)) for f,r in instructions.items()}}
    immutable(directory / "manifest.json", manifest)
    finish_phase(phase, "failed" if blockers else "completed", decision="unlaunchable" if blockers else "prepared",
                 blockers=blockers, manifest_sha256=sha256(directory / "manifest.json"), optimizer_updates=0,
                 scored_tokens=0, gpu_seconds=0, next_step="End study" if blockers else "Correctness tests and GPU preflight")
    return manifest


def _primary(cfg, cache, exposure, tokenizer):
    rows, inventory = primary_sources(cfg, cache)
    panels, audit = primary_panels(cfg, rows, exposure, tokenizer)
    return panels, {**audit, "sources": inventory}


def audit_isolation(directory):
    seen = {}
    for kind in ("primary", "raw", "transfer"):
        for split in ("development", "confirmation"):
            records = read_jsonl(directory / f"{kind}_{split}.jsonl")
            if len({r["id"] for r in records}) != len(records):
                raise ValueError("Duplicate evaluation membership")
            for r in records:
                key = r["group_id"]
                if key in seen and seen[key] != split:
                    raise ValueError("Source group crossed evaluation panels")
                seen[key] = split
    raw_ids = {r["id"] for s in ("development", "confirmation") for r in read_jsonl(directory / f"raw_{s}.jsonl")}
    if any(any(doc in r["id"] for doc in raw_ids) for r in read_jsonl(directory / "replay.jsonl")):
        raise ValueError("Fresh raw panel leaked into replay")
