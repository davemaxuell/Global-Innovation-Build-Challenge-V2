"""Round 2 of improving V2.1 (phase v2/phases/round2): soups, stronger arms, selection, 100% retrain.

The held-out development items (20% of each benchmark train split, never trained on) are
split by a fixed hash into half A, used to build greedy soups, and half B, used only to select.

  soup --name N   Greedy soup (Wortsman et al., 2022, "Model soups"): rank ingredients by
                  half-A accuracy, then add each one to a uniform weight average only if the
                  half-A accuracy of the average does not fall. Writes an export and a record.
  select          Score the candidates and the V2.1 reference on half B and on the corpus
                  Wikipedia selection panel. Rule: among candidates whose Wikipedia bits/byte is
                  no worse than V2.1's, take the highest half-B four-task mean; it replaces V2.1
                  only if that mean is at least ``min_gain`` above V2.1's.
  plan-final      For the selected recipe (a single model, a soup of models, or V2.1 itself),
                  write retrain configs on 100% of the train splits (held-out items included).
  final           Build the final model from the retrained exports (single, or the same uniform
                  soup), check its Wikipedia bits/byte against V2.1's, and write final.json. If the
                  check fails, the 80%-trained winner from ``select`` is used instead.

Run: PYTHONPATH=src python -m scglm_v2.round2 <command> --config v2/phases/round2/config.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

from .common import ROOT, event, now, sha256_file, write_json

TOLERANCE = 1e-12


# ---------------------------------------------------------------- data and scoring
def half_of(item: dict) -> int:
    """0 = half A (soup building), 1 = half B (selection). Fixed by the item's content."""
    key = json.dumps([item["task"], item["label"], item["encoded"]], separators=(",", ":"))
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2


def halves(cfg: dict) -> tuple[list[dict], list[dict]]:
    data_dir = ROOT / cfg["data_dir"]
    report = json.loads((data_dir / "report.json").read_text())
    path = data_dir / "development.jsonl"
    if sha256_file(path) != report["files"]["development.jsonl"]:
        raise ValueError("Development items changed after preparation")
    items = [json.loads(line) for line in path.open()]
    return [i for i in items if half_of(i) == 0], [i for i in items if half_of(i) == 1]


def ingredients(cfg: dict, which: str) -> list[dict]:
    """Ingredient records {export, recipe: {config, arm, epoch}} for 'existing', 'new' or 'existing+new'."""
    out = []
    if "existing" in which.split("+"):
        out += cfg["existing_candidates"]
    if "new" in which.split("+"):
        for arm, spec in cfg["arms"].items():
            for epoch in range(1, int(spec.get("epochs", cfg["epochs"])) + 1):
                out.append({"export": f"{spec['run_dir']}/epoch_{epoch}/export",
                            "recipe": {"config": cfg["config_path"], "arm": arm, "epoch": epoch}})
    missing = [i["export"] for i in out if not (ROOT / i["export"] / "model.safetensors").exists()]
    if missing:
        raise FileNotFoundError(f"Missing ingredient exports: {missing}")
    return out


def _param_names(model) -> list[str]:
    return [n for n, _ in model.named_parameters()]


def _load_weights(export: Path, names: list[str]) -> dict:
    from safetensors.torch import load_file
    sd = load_file(str(export / "model.safetensors"))
    missing = [n for n in names if n not in sd]
    if missing:
        raise KeyError(f"{export}: missing parameters {missing[:3]}")
    return {n: sd[n].double() for n in names}


def _set_params(model, weights: dict) -> None:
    import torch
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(weights[name].to(device=p.device, dtype=p.dtype))


def _export(model, out: Path, base: Path, provenance: dict) -> str:
    from scglm.model import audit_parameters, save_model
    save_model(model, out)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(base / name, out / name)
    write_json(out / "training_provenance.json", {"initialization": "scratch", "created": now(), **provenance,
                                                  "parameter_audit": audit_parameters(model)})
    return sha256_file(out / "model.safetensors")


def average_exports(exports: list[Path], names: list[str]) -> dict:
    total = None
    for e in exports:
        w = _load_weights(e, names)
        total = w if total is None else {n: total[n] + w[n] for n in names}
    return {n: total[n] / len(exports) for n in names}


# ---------------------------------------------------------------- commands
def soup(cfg: dict, name: str, device: str) -> dict:
    from scglm.model import load_model
    from .task_format import score_items
    spec = cfg["soups"][name]
    out = ROOT / spec["output"]
    if out.exists():
        raise FileExistsError(f"{out} exists")
    log = ROOT / cfg["phase_dir"] / "events.jsonl"
    half_a, _ = halves(cfg)
    base_dir = ROOT / cfg["parent_export"]
    model = load_model(base_dir).to(device).eval()
    names = _param_names(model)
    ranked = []
    for ing in ingredients(cfg, spec["ingredients"]):
        _set_params(model, _load_weights(ROOT / ing["export"], names))
        acc = score_items(model, half_a, device)["_proxy_mean_acc"]
        ranked.append({**ing, "half_a_acc": acc, "sha256": sha256_file(ROOT / ing["export"] / "model.safetensors")})
        event(log, "soup_ingredient_scored", soup=name, export=ing["export"], half_a_acc=acc)
    ranked.sort(key=lambda r: -r["half_a_acc"])
    members, total, best, steps = [], None, -1.0, []
    for ing in ranked:
        w = _load_weights(ROOT / ing["export"], names)
        trial_total = w if total is None else {n: total[n] + w[n] for n in names}
        _set_params(model, {n: trial_total[n] / (len(members) + 1) for n in names})
        acc = score_items(model, half_a, device)["_proxy_mean_acc"]
        keep = acc >= best - TOLERANCE
        steps.append({"export": ing["export"], "trial_half_a_acc": acc, "kept": keep})
        event(log, "soup_step", soup=name, export=ing["export"], trial_half_a_acc=acc, kept=keep)
        if keep:
            members, total, best = members + [ing], trial_total, acc
    _set_params(model, {n: total[n] / len(members) for n in names})
    digest = _export(model, out / "export", base_dir, {
        "soup": {"name": name, "method": "greedy soup on half A (uniform average of kept ingredients)",
                 "members": [{k: m[k] for k in ("export", "sha256", "recipe", "half_a_acc")} for m in members]},
        "parent": json.loads((base_dir / "training_provenance.json").read_text())})
    record = {"name": name, "export": f"{spec['output']}/export", "sha256": digest, "half_a_acc": best,
              "members": [m["export"] for m in members], "ranked": ranked, "steps": steps, "created": now()}
    write_json(out / "soup.json", record)
    write_json(ROOT / cfg["phase_dir"] / f"soup_{name}.json", record)
    event(log, "soup_built", soup=name, members=len(members), half_a_acc=best, sha256=digest)
    return record


def _score_full(export: Path, items: list[dict], texts: dict, device: str) -> dict:
    import torch
    from .devevaluate import bits_per_byte, load
    from .task_format import score_items
    model, tok = load(str(export), device)
    mc = score_items(model, items, device)
    bpb = {k: bits_per_byte(model, tok, device, v) for k, v in texts.items()}
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"run": str(export.parent), "export": str(export), "export_model_sha256": sha256_file(export / "model.safetensors"),
            "multiple_choice_half_b": mc, "proxy_mean_acc": mc["_proxy_mean_acc"], "bits_per_byte_selection_panels": bpb}


def decide(reference: dict, scored: list[dict], min_gain: float) -> tuple[dict, str]:
    ref_wiki = reference["bits_per_byte_selection_panels"]["wiki"]
    eligible = [s for s in scored if s["bits_per_byte_selection_panels"]["wiki"] <= ref_wiki + TOLERANCE]
    if not eligible:
        return reference, "reference: no candidate had Wikipedia bits/byte no worse than V2.1's"
    best = max(eligible, key=lambda s: s["proxy_mean_acc"])
    gain = best["proxy_mean_acc"] - reference["proxy_mean_acc"]
    if gain >= min_gain - TOLERANCE:
        return best, f"candidate: half-B gain {gain:+.4f} >= {min_gain} over V2.1 with Wikipedia bits/byte no worse"
    return reference, f"reference: best eligible half-B gain {gain:+.4f} < {min_gain}"


def select(cfg: dict, device: str) -> dict:
    from .devevaluate import panel_texts
    out = ROOT / cfg["phase_dir"] / "selection.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; frozen")
    _, half_b = halves(cfg)
    texts = panel_texts(ROOT / cfg["replay_manifest"], cfg["selection_rule"]["panel_docs"], split="selection")
    reference = _score_full(ROOT / cfg["reference_export"], half_b, texts, device)
    if reference["export_model_sha256"] != cfg["reference_sha256"]:
        raise ValueError("Reference (V2.1) weights differ from the registration")
    candidates = [i["export"] for i in ingredients(cfg, "new")] + [f"{s['output']}/export" for s in cfg["soups"].values()]
    scored = []
    for c in candidates:
        scored.append(_score_full(ROOT / c, half_b, texts, device))
        print(json.dumps({"scored": c, "half_b": scored[-1]["proxy_mean_acc"],
                          "wiki_bpb": scored[-1]["bits_per_byte_selection_panels"]["wiki"]}), flush=True)
    chosen, reason = decide(reference, scored, cfg["selection_rule"]["min_gain"])
    record = {"schema": "v2-round2-selection-1", "label": "round2", "panel": "selection", "official_scores_used": False,
              "selected_model": chosen["export"], "selected_model_sha256": chosen["export_model_sha256"],
              "decision_reason": reason, "reference": reference["export"], "rule": cfg["selection_rule"],
              "results": [reference] + scored, "half_b_items": len(half_b), "frozen_at_unix": time.time(),
              "note": "Development data only (half B of the held-out benchmark train items; corpus selection panels)."}
    write_json(out, record)
    print(json.dumps({"selected": chosen["export"], "reason": reason}), flush=True)
    return record


def _recipes_for(cfg: dict, export: str) -> list[dict]:
    """Recipe(s) that produced an export: itself, or a soup's members."""
    rel = str(Path(export).resolve().relative_to(ROOT))
    for s in cfg["soups"].values():
        if rel == f"{s['output']}/export":
            members = json.loads((ROOT / s["output"] / "soup.json").read_text())["members"]
            return [_recipes_for(cfg, m)[0] for m in members]
    if rel == cfg["reference_export"]:
        return [cfg["reference_recipe"]]
    for ing in ingredients(cfg, "existing+new"):
        if ing["export"] == rel:
            return [ing["recipe"]]
    raise KeyError(f"No recipe recorded for {rel}")


def plan_final(cfg: dict) -> dict:
    sel = json.loads((ROOT / cfg["phase_dir"] / "selection.json").read_text())
    recipes = _recipes_for(cfg, sel["selected_model"])
    groups = {}
    for r in recipes:
        g = groups.setdefault((r["config"], r["arm"]), {"config": r["config"], "arm": r["arm"], "epochs_needed": set()})
        g["epochs_needed"].add(int(r["epoch"]))
    plan = {"selected_model": sel["selected_model"], "combination": "single" if len(recipes) == 1 else "uniform_soup",
            "retrain": [], "final_exports": []}
    for (config, arm), g in groups.items():
        source = json.loads((ROOT / config).read_text())
        new = copy.deepcopy(source)
        new["data_dir"] = cfg["full_data_dir"]
        run_dir = f"v2/runs/round2_full_{Path(config).parent.name}_{arm}"
        new["arms"] = {arm: {**source["arms"][arm], "run_dir": run_dir, "stop_after_epoch": max(g["epochs_needed"])}}
        path = f"{cfg['phase_dir']}/retrain_{Path(config).parent.name}_{arm}.json"
        write_json(ROOT / path, new)
        plan["retrain"].append({"config": path, "arm": arm, "run_dir": run_dir,
                                "epochs_needed": sorted(g["epochs_needed"]), "source_config": config})
        plan["final_exports"] += [f"{run_dir}/epoch_{e}/export" for e in sorted(g["epochs_needed"])]
    write_json(ROOT / cfg["phase_dir"] / "final_plan.json", plan)
    print(json.dumps(plan), flush=True)
    return plan


def final(cfg: dict, device: str) -> dict:
    from scglm.model import load_model
    from .devevaluate import panel_texts
    phase = ROOT / cfg["phase_dir"]
    out = phase / "final.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; frozen")
    plan = json.loads((phase / "final_plan.json").read_text())
    sel = json.loads((phase / "selection.json").read_text())
    base_dir = ROOT / cfg["parent_export"]
    exports = [ROOT / e for e in plan["final_exports"]]
    if plan["combination"] == "single":
        final_export = exports[0]
    else:
        model = load_model(base_dir).eval()
        names = _param_names(model)
        _set_params(model, average_exports(exports, names))
        final_export = ROOT / cfg["final_soup_output"] / "export"
        _export(model, final_export, base_dir, {"soup": {"name": "final_full_data", "method": "uniform average",
                                                         "members": [str(e) for e in exports]},
                                                "parent": json.loads((base_dir / "training_provenance.json").read_text())})
    _, half_b = halves(cfg)
    texts = panel_texts(ROOT / cfg["replay_manifest"], cfg["selection_rule"]["panel_docs"], split="selection")
    ref = next(r for r in sel["results"] if r["export"] == sel["reference"])
    candidate = _score_full(final_export, half_b, texts, device)
    candidate["note"] = "half-B accuracy is not held out for this model (trained on 100% of the train splits)"
    passed = candidate["bits_per_byte_selection_panels"]["wiki"] <= ref["bits_per_byte_selection_panels"]["wiki"] + TOLERANCE
    if passed:
        chosen, reason = candidate, "100%-retrained model passed the Wikipedia guard (bits/byte no worse than V2.1's)"
    else:
        chosen = next(r for r in sel["results"] if r["export"] == sel["selected_model"])
        reason = "100%-retrained model failed the Wikipedia guard; falling back to the development-selected model"
    record = {"schema": "v2-round2-final-1", "label": "round2_final", "panel": "selection", "official_scores_used": False,
              "selected_model": chosen["export"], "selected_model_sha256": chosen["export_model_sha256"],
              "decision_reason": reason, "plan": plan, "development_selection": sel["selected_model"],
              "results": [ref, candidate] + ([chosen] if chosen is not candidate and chosen is not ref else []),
              "is_reference": chosen["export"] == ref["export"], "frozen_at_unix": time.time()}
    write_json(out, record)
    print(json.dumps({"final": chosen["export"], "reason": reason}), flush=True)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["soup", "select", "plan-final", "final"])
    p.add_argument("--config", required=True)
    p.add_argument("--name")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    cfg["config_path"] = args.config
    if args.command == "soup":
        soup(cfg, args.name, args.device)
    elif args.command == "select":
        select(cfg, args.device)
    elif args.command == "plan-final":
        plan_final(cfg)
    else:
        final(cfg, args.device)


if __name__ == "__main__":
    main()
