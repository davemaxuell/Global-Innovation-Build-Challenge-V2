"""Resumable development scoring shared by the mid-training and RL studies.

Record formats match the choice study, so its paired grouped bootstrap applies
unchanged. Development panels from the completed choice study are reused and
disclosed as previously observed; its reserved confirmation panels are never
opened here.
"""
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from scglm.model import load_model
from scglm_choice.common import DOMAINS
from scglm_choice.evaluate import _completed_lines, instruction, score_choice
from scglm_choice.objective import logps
from scglm_pipeline.task_registry import raw_row
from .common import ROOT, immutable, read, read_jsonl, sha256, write_json

KINDS = ("choice", "raw", "generate", "transfer")


def choice_panels(spec):
    """Frozen choice-study development panels, verified against its manifest."""
    directory = ROOT / spec["data"]
    manifest = directory / "manifest.json"
    if sha256(manifest) != spec["manifest_sha256"]:
        raise ValueError("Choice-study manifest changed")
    files = read(manifest)["files"]
    result = {}
    for name, (kind, file) in spec["use"].items():
        if "confirmation" in file:
            raise ValueError("Reserved confirmation panels are outside these studies")
        if sha256(directory / file) != files[file]:
            raise ValueError(f"Choice-study panel changed: {file}")
        result[name] = {"kind": kind, "rows": read_jsonl(directory / file), "source": f"{spec['data']}/{file}"}
    return result


def panel_identity(panels):
    from .common import digest
    return {name: {"kind": p["kind"], "rows_sha256": digest([r["id"] for r in p["rows"]]),
                   "items": len(p["rows"])} for name, p in sorted(panels.items())}


def _score(kind, row, model, tokenizer, presentations, charge):
    if kind in ("choice", "transfer"):
        templates = presentations if kind == "choice" else presentations[:1]
        variants = [score_choice(model, tokenizer, row, t, charge) for t in templates]
        return {"id": row["id"], "group": row["group_id"], "domain": row["domain"], "variants": variants,
                **{k: float(np.mean([v[k] for v in variants])) for k in ("correct", "character_normalized_correct")},
                "gold_nll_sum": sum(v["gold_nll_sum"] for v in variants),
                "gold_tokens": sum(v["gold_tokens"] for v in variants)}
    if kind == "raw":
        values, lengths = logps(model, [raw_row(row["tokens"])], charge=charge, kind="evaluation_raw")
        return {"id": row["id"], "group": row["group_id"], "source": row["source"],
                "nll_sum": -float(values[0]), "token_count": int(lengths[0])}
    if kind == "generate":
        return instruction(model, tokenizer, row, charge)
    raise ValueError(f"Unknown panel kind: {kind}")


def evaluate(path, directory, panels, charge, *, presentations, model=None, contract=None, device="cuda:0"):
    """Score each panel once per model hash; interrupted panels resume by item."""
    path, directory = Path(path), Path(directory)
    weight_hash = sha256(path / "model.safetensors")
    registration = {"model_sha256": weight_hash, "panels": panel_identity(panels),
                    "presentations": presentations, **(contract or {})}
    directory.mkdir(parents=True, exist_ok=True)
    immutable(directory / "registration.json", registration)
    if (directory / "result.json").exists():
        return load_result(directory)
    own = model is None
    if own:
        model = load_model(path).to(device)
    was_training = model.training
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    try:
        for name, panel in panels.items():
            file = directory / f"{name}.jsonl"
            completed = _completed_lines(file)
            seen = {r["id"] for r in completed}
            if len(seen) != len(completed) or not seen.issubset({r["id"] for r in panel["rows"]}):
                raise ValueError("Partial evaluation membership changed")
            with file.open("a") as stream, torch.inference_mode():
                for row in panel["rows"]:
                    if row["id"] in seen:
                        continue
                    stream.write(json.dumps(_score(panel["kind"], row, model, tokenizer, presentations, charge),
                                            allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        if sha256(path / "model.safetensors") != weight_hash:
            raise ValueError("Export changed during evaluation")
        records = {name: read_jsonl(directory / f"{name}.jsonl") for name in panels}
        result = {"status": "completed", "registration": registration,
                  "metrics": summarize(records, {n: p["kind"] for n, p in panels.items()}),
                  "directory": str(directory),
                  "files": {f"{n}.jsonl": sha256(directory / f"{n}.jsonl") for n in panels}}
        write_json(directory / "result.json", result)
        return result
    finally:
        model.train(was_training)
        if own:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def summarize(records, kinds):
    result = {}
    for name, rows in records.items():
        kind = kinds[name]
        if kind == "raw":
            by = defaultdict(list)
            for r in rows:
                by[r["source"]].append(r)
            result[name] = {s: {"nll": sum(r["nll_sum"] for r in v) / sum(r["token_count"] for r in v),
                                "token_count": sum(r["token_count"] for r in v), "documents": len(v)}
                            for s, v in sorted(by.items())}
            continue
        key = "family" if kind == "generate" else "domain"
        by = defaultdict(list)
        for r in rows:
            by[r[key]].append(r)
        section = {}
        for group, v in sorted(by.items()):
            section[group] = {"accuracy": float(np.mean([r["correct"] for r in v])), "items": len(v)}
            if kind in ("choice", "transfer"):
                section[group].update(
                    character_normalized_accuracy=float(np.mean([r["character_normalized_correct"] for r in v])),
                    gold_answer_nll=sum(r["gold_nll_sum"] for r in v) / sum(r["gold_tokens"] for r in v))
            if kind == "generate":
                section[group].update(answer_correct=float(np.mean([r["answer_correct"] for r in v])),
                                      ended_eos=float(np.mean([r["ended_eos"] for r in v])))
        result[name] = {"groups": section, "mean": float(np.mean([s["accuracy"] for s in section.values()]))}
        if kind == "choice" and set(section) == set(DOMAINS):
            result[name]["primary"] = float(np.mean([section[d]["accuracy"] for d in DOMAINS]))
    return result


def load_result(directory):
    result = read(Path(directory) / "result.json")
    for name, expected in result["files"].items():
        if sha256(Path(directory) / name) != expected:
            raise ValueError("Evaluation evidence changed")
    return result


def evidence(result, panel):
    file = Path(result["directory"]) / f"{panel}.jsonl"
    if sha256(file) != result["files"][file.name]:
        raise ValueError("Evaluation evidence hash changed")
    return read_jsonl(file)


def guards(candidate, reference, rules, label):
    """Point-estimate retention checks against one reference model's metrics.

    ``rules``: {"raw_panels": [...], "raw_nll_delta": x, "choice_panels": [...],
    "accuracy_delta_min": y}. Returns failure labels; never raises on a breach.
    """
    failures = []
    for panel in rules.get("raw_panels", []):
        if set(candidate[panel]) != set(reference[panel]):
            raise ValueError(f"Unpaired raw coverage: {panel}")
        for source, value in candidate[panel].items():
            if value["token_count"] != reference[panel][source]["token_count"]:
                raise ValueError("Unpaired raw target counts")
            delta = value["nll"] - reference[panel][source]["nll"]
            if not np.isfinite(delta) or delta > rules["raw_nll_delta"] + 1e-12:
                failures.append(f"raw:{panel}:{source}:vs_{label}")
    for panel in rules.get("choice_panels", []):
        groups = candidate[panel]["groups"]
        if set(groups) != set(reference[panel]["groups"]):
            raise ValueError(f"Unpaired choice coverage: {panel}")
        for name, value in groups.items():
            delta = value["accuracy"] - reference[panel]["groups"][name]["accuracy"]
            if not np.isfinite(delta) or delta < rules["accuracy_delta_min"] - 1e-12:
                failures.append(f"accuracy:{panel}:{name}:vs_{label}")
    return failures


def paired_difference(results, names, panel, kind, *, replicates, seed, stratum="primary"):
    """Grouped paired bootstrap of ``names[0] - names[1]`` using the choice-study sampler."""
    from scglm_choice.selection import bootstrap
    evidence_rows = {n: evidence(results[n], panel) for n in names}
    order, points, draws = bootstrap(evidence_rows, kind, seed=seed, replicates=replicates)
    a, b = order.index(names[0]), order.index(names[1])
    difference = draws[stratum][a] - draws[stratum][b]
    return {"point": float(points[stratum][a] - points[stratum][b]),
            "lower_95_one_sided": float(np.quantile(difference, .05)),
            "upper_95_one_sided": float(np.quantile(difference, .95)),
            "interval_95": np.quantile(difference, [.025, .975]).tolist(),
            "replicates": replicates, "seed": seed}


def task_bootstrap(results, names, panel, *, replicates, seed, families=None):
    """Grouped paired bootstrap for generated-task accuracy, families equally weighted."""
    keep = set(families) if families is not None else None
    mapped = {n: {r["id"]: r for r in evidence(results[n], panel) if keep is None or r["family"] in keep} for n in names}
    first = mapped[names[0]]
    if not first or any(m.keys() != first.keys() for m in mapped.values()):
        raise ValueError("Missing or unpaired task evidence")
    families = defaultdict(lambda: defaultdict(list))
    for item_id in sorted(first):
        families[first[item_id]["family"]][first[item_id]["group"]].append(item_id)
    rng = np.random.default_rng(seed)
    total = np.zeros((len(names), replicates))
    points = np.zeros(len(names))
    for family, groups in sorted(families.items()):
        units = list(groups.values())
        sizes = np.array([len(u) for u in units])
        sums = np.array([[sum(mapped[n][i]["correct"] for i in u) for u in units] for n in names])
        index = rng.integers(len(units), size=(replicates, len(units)))
        total += sums[:, index].sum(2) / sizes[index].sum(1)
        points += sums.sum(1) / sizes.sum()
    total /= len(families)
    points /= len(families)
    difference = total[0] - total[1]
    return {"point": float(points[0] - points[1]), "lower_95_one_sided": float(np.quantile(difference, .05)),
            "upper_95_one_sided": float(np.quantile(difference, .95)), "interval_95": np.quantile(difference, [.025, .975]).tolist(), "families": sorted(families),
            "replicates": replicates, "seed": seed}
