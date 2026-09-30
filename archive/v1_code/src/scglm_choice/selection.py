"""Paired point gates, four-model confirmation and grouped inference."""
from collections import defaultdict
from pathlib import Path
import numpy as np

from .common import ROOT, DOMAINS, SEEDS, read, read_jsonl, digest, immutable, lock, sha256


def paired(models, kind):
    """Reject missing, duplicate, differently grouped, or nonfinite evidence."""
    mappings = {name: {r["id"]: r for r in rows} for name,rows in models.items()}
    first = next(iter(mappings.values()))
    for name, rows in models.items():
        if len(rows) != len(mappings[name]) or mappings[name].keys() != first.keys():
            raise ValueError("Unpaired evaluation identities")
    if not first:
        raise ValueError("Empty paired evidence")
    key = "domain" if kind == "primary" else "source" if kind == "raw" else "family"
    for item_id, ref in first.items():
        for values in mappings.values():
            row = values[item_id]
            if (row[key],row["group"]) != (ref[key],ref["group"]):
                raise ValueError("Paired grouping differs")
            metric = row["nll_sum"] if kind == "raw" else row["correct"]
            if not np.isfinite(metric):
                raise ValueError("Nonfinite paired metric")
            if kind == "raw" and (row["token_count"] <= 0 or row["token_count"] != ref["token_count"]):
                raise ValueError("Raw scored-target pairing changed")
    return mappings


def bootstrap(models, kind, *, seed=20260929, replicates=10000):
    """All comparators share the exact same group draws in every replicate."""
    mapped = paired(models, kind)
    names = list(mapped)
    first = mapped[names[0]]
    key = "domain" if kind == "primary" else "source"
    grouped = defaultdict(lambda: defaultdict(list))
    for item_id in sorted(first):
        row = first[item_id]
        grouped[row[key]][row["group"]].append(item_id)
    if kind == "primary" and set(grouped) != set(DOMAINS):
        raise ValueError("Missing primary domain")
    rng = np.random.default_rng(seed)
    draws, points = {}, {}
    for stratum, groups in sorted(grouped.items()):
        units = list(groups.values())
        denominator = np.array([sum(first[i]["token_count"] for i in g) if kind == "raw" else len(g) for g in units])
        totals = np.array([[sum(mapped[name][i]["nll_sum" if kind == "raw" else "correct"] for i in g)
                            for g in units] for name in names])
        values = np.empty((len(names), replicates))
        # Bounded allocation for large document/group panels.
        for start in range(0, replicates, 250):
            index = rng.integers(len(units), size=(min(250,replicates-start), len(units)))
            values[:,start:start+len(index)] = totals[:,index].sum(2)/denominator[index].sum(1)
        draws[stratum] = values
        points[stratum] = totals.sum(1)/denominator.sum()
    if kind == "primary":
        draws["primary"] = sum(draws[d] for d in DOMAINS)/2
        points["primary"] = sum(points[d] for d in DOMAINS)/2
    return names, points, draws


def retention(candidate, parent, base, *, cfg):
    failures = []
    for section in ("domains", "instructions", "raw"):
        expected = set(cfg["evaluation"]["raw_strata"]) if section == "raw" else set(parent[section])
        if not expected or set(candidate[section]) != expected or set(parent[section]) != expected:
            raise ValueError("Missing retention coverage: " + section)
        if section == "raw" and set(base[section]) != expected:
            raise ValueError("Missing base raw coverage")
        for name in expected:
            if section == "raw":
                if candidate[section][name]["token_count"] != parent[section][name]["token_count"]:
                    raise ValueError("Unpaired retention target counts")
                for label, reference, threshold in (("parent",parent,.01), ("base",base,.02)):
                    values = [candidate[section][name]["nll"], reference[section][name]["nll"]]
                    if not np.isfinite(values).all() or values[0]-values[1] > threshold+1e-12:
                        failures.append(f"raw:{name}:vs_{label}")
            else:
                if candidate[section][name]["items"] != parent[section][name]["items"]:
                    raise ValueError("Unpaired retention item counts")
                values = [candidate[section][name]["accuracy"], parent[section][name]["accuracy"]]
                if not np.isfinite(values).all() or values[0]-values[1] < -.02-1e-12:
                    failures.append(f"accuracy:{section}/{name}")
    return failures


def pilot(candidate, control, parent, base, cfg):
    failures = retention(candidate, parent, base, cfg=cfg)
    gains = {"control": candidate["primary"]-control["primary"], "parent": candidate["primary"]-parent["primary"],
             "base": candidate["primary"]-base["primary"]}
    for label, minimum in (("control", .01), ("parent", .01), ("base", 0.)):
        if not np.isfinite(gains[label]) or gains[label]+1e-12 < minimum:
            failures.append("primary_gain:"+label)
    return {"passed": not failures, "failures": failures, "gains": gains}


def replication(members, parent, base, cfg):
    if len(members) != 3 or {m["seed"] for m in members} != set(SEEDS):
        raise ValueError("All three paired seeds are required exactly once")
    failures, gains = [], []
    for m in members:
        c, t = m["control"], m["candidate"]
        failures.extend(f"{m['seed']}:{f}" for f in retention(t,parent,base,cfg=cfg))
        delta = {"control":t["primary"]-c["primary"], "parent":t["primary"]-parent["primary"], "base":t["primary"]-base["primary"]}
        gains.append(delta)
        if delta["control"] < -1e-12 or delta["parent"] < -1e-12:
            failures.append(f"{m['seed']}:primary_regression")
    mean = {k:float(np.mean([g[k] for g in gains])) for k in gains[0]}
    if any(not np.isfinite(mean[k]) or mean[k]+1e-12 < limit for k,limit in (("control",.01),("parent",.01),("base",0.))):
        failures.append("paired_mean_gain")
    return {"passed":not failures, "failures":failures, "mean_gains":mean, "seed_gains":gains, "deployment_seed":20260929}


def claim_confirmation(manifest, models, directory, data_directory):
    if set(models) != {"candidate", "control", "parent", "base"}:
        raise ValueError("Confirmation must bind all four fixed models")
    items = []
    for kind in ("primary", "raw", "transfer"):
        for row in read_jsonl(data_directory / f"{kind}_confirmation.jsonl"):
            content = row["tokens"] if kind == "raw" else [row["question"], row["choices"], row["answer_index"], row.get("suffix_probe")]
            items.append(digest([kind, content]))
    claim = {"panel_sha256": manifest["confirmation_membership_sha256"], "models": models,
             "item_sha256": sorted(items), "protocol": "choice-discrimination-v1"}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest(sorted(items)) + ".json")
    with lock(directory / ".registry.lock"):
        for other in directory.glob("*.json"):
            previous = read(other)
            if other != path and set(items).intersection(previous.get("item_sha256", [])):
                raise ValueError("Confirmation content already claimed")
        immutable(path, claim)
    return path


def verify_claim(path, manifest, weight_hash):
    claim = read(path)
    if (claim["panel_sha256"] != manifest["confirmation_membership_sha256"] or
            set(claim["models"]) != {"candidate", "control", "parent", "base"} or weight_hash not in claim["models"].values()):
        raise ValueError("Model/panel is outside the reserved four-model confirmation claim")


def confirmation(results, cfg):
    from .evaluate import evidence
    if set(results) != {"candidate", "control", "parent", "base"}:
        raise ValueError("Four completed confirmation models required")
    for r in results.values():
        if r["contract"]["split"] != "confirmation":
            raise ValueError("Confirmation requires reserved panel evidence")
    if len({r["contract"]["claim_sha256"] for r in results.values()}) != 1:
        raise ValueError("Confirmation models do not share a claim")
    metrics = {n:r["metrics"] for n,r in results.items()}
    failures = retention(metrics["candidate"], metrics["parent"], metrics["base"], cfg=cfg)
    names, point, draws = bootstrap({n:evidence(r,"primary") for n,r in results.items()}, "primary")
    c = names.index("candidate")
    intervals = {}
    for comparator in ("control", "parent", "base"):
        i = names.index(comparator)
        d = draws["primary"][c]-draws["primary"][i]
        gain = float(point["primary"][c]-point["primary"][i])
        low = float(np.quantile(d,.05/3))
        intervals[comparator] = {"gain":gain, "adjusted_lower":low, "descriptive_95":np.quantile(d,[.025,.975]).tolist()}
        if low <= 0 or gain+1e-12 < (.01 if comparator != "base" else 0):
            failures.append("confirmation_gain:"+comparator)
    p = names.index("parent")
    domain_bounds = {}
    for domain in DOMAINS:
        low = float(np.quantile(draws[domain][c]-draws[domain][p],.05))
        domain_bounds[domain] = low
        if low < -.02-1e-12:
            failures.append("confirmation_domain_retention:"+domain)
    names, point, draws = bootstrap({n:evidence(r,"raw") for n,r in results.items()}, "raw")
    c = names.index("candidate")
    raw_bounds = {}
    for source in cfg["evaluation"]["raw_strata"]:
        raw_bounds[source] = {}
        for name,limit in (("parent",.01),("base",.02)):
            upper = float(np.quantile(draws[source][c]-draws[source][names.index(name)],.95))
            raw_bounds[source][name] = upper
            if upper > limit+1e-12:
                failures.append(f"confirmation_raw_retention:{source}:{name}")
    paired({n:evidence(r,"instructions") for n,r in results.items()}, "instructions")
    return {"passed":not failures, "failures":failures, "primary":intervals,
            "domain_retention_lower_95":domain_bounds, "raw_retention_upper_95":raw_bounds,
            "replicates":10000, "bootstrap_seed":20260929, "transfer_diagnostic_only":True,
            "instruction_guards": "historically observed, point estimates only"}


def official_selection(models, hashes, candidate):
    if len(models) != 6 or set(models) != set(hashes) or candidate not in models:
        raise ValueError("Official reporting requires all six completed endpoints")
    return {"panel":"selection", "official_scores_used":False,
            "selected_model":str(models[candidate]), "selected_model_sha256":hashes[candidate],
            "results":[{"run":str(Path(p).parent), "export_model_sha256":hashes[n]} for n,p in models.items()],
            "deployment_seed":20260929, "confirmation_passed":True}
