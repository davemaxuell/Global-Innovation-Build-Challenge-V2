"""Grouped comparison, seed replication, and one-use confirmation decisions."""
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
from .common import ROOT,read,read_jsonl,write_json,sha256,digest,resolve
from .task_registry import PRIMARY_DOMAINS
from .selection import paired_raw_intervals


def evidence(result,name):
    path=Path(result["artifacts"])/(name+".jsonl")
    if sha256(path)!=result["files"][path.name]:raise ValueError("Selection evidence changed")
    return read_jsonl(path)


def grouped_gain(base,candidate,*,seed,replicates):
    a={r["id"]:r for r in base};b={r["id"]:r for r in candidate}
    if len(a)!=len(base) or len(b)!=len(candidate) or a.keys()!=b.keys():raise ValueError("Unpaired primary items")
    groups=defaultdict(lambda:defaultdict(list))
    for key in sorted(a):
        x,y=a[key],b[key]
        if (x["domain"],x["group"])!=(y["domain"],y["group"]):raise ValueError("Primary grouping changed")
        groups[x["domain"]][x["group"]].append(float(y["correct"])-float(x["correct"]))
    if set(groups)!=set(PRIMARY_DOMAINS) or replicates<1:raise ValueError("Missing primary groups")
    rng=np.random.default_rng(seed);boot=np.zeros(replicates);points=[];counts={}
    for domain in PRIMARY_DOMAINS:
        values=list(groups[domain].values());totals=np.array([sum(v) for v in values]);sizes=np.array([len(v) for v in values])
        indexes=rng.integers(len(values),size=(replicates,len(values)))
        boot+=totals[indexes].sum(1)/sizes[indexes].sum(1)/len(PRIMARY_DOMAINS)
        points.append(totals.sum()/sizes.sum());counts[domain]={"groups":len(values),"items":int(sizes.sum())}
    return {"gain":float(np.mean(points)),"lower_95":float(np.quantile(boot,.025)),
            "upper_95":float(np.quantile(boot,.975)),"counts":counts,"unit":"source_or_generated_fact_group"}


def retention(before,after,policy):
    failures=[]
    def paired(section,key,tolerance,direction=1):
        a=before.get(section,{});b=after.get(section,{})
        if not a or a.keys()!=b.keys():failures.append("missing_or_changed:"+section);return
        for name,x in a.items():
            y=b[name]
            if (key not in x or key not in y or x[key] is None or y[key] is None
                    or not np.isfinite([x[key],y[key]]).all()):failures.append("missing:"+section+"/"+name);continue
            if x.get("examples")!=y.get("examples"):failures.append("changed_counts:"+section+"/"+name)
            if direction*(x[key]-y[key])>tolerance+1e-12:failures.append("regression:"+section+"/"+name)
    a=before.get("raw_lm",{}).get("sources",{});b=after.get("raw_lm",{}).get("sources",{})
    if not a or a.keys()!=b.keys():failures.append("missing_raw_sources")
    for source,x in a.items():
        y=b.get(source,{})
        if y.get("nll") is None or not np.isfinite(y["nll"]) or y["nll"]-x["nll"]>policy["max_raw_nll_regression"]+1e-12:
            failures.append("raw_nll:"+source)
    for section in ("domains","choice_families","families","legacy_families"):
        paired(section,"accuracy",policy["max_accuracy_regression"])
    paired("domains","character_normalized_accuracy",policy["max_accuracy_regression"])
    systems_a=before.get("system",{});systems_b=after.get("system",{})
    modes_a=systems_a.get("modes",{});modes_b=systems_b.get("modes",{})
    if not modes_a or modes_a.keys()!=modes_b.keys():failures.append("missing_system_modes")
    for mode,x in {**modes_a,"policy_pairs":systems_a.get("policy_pairs",{})}.items():
        y=systems_b.get("policy_pairs",{}) if mode=="policy_pairs" else modes_b.get(mode,{})
        if not x.get("examples") or x.get("accuracy") is None or y.get("accuracy") is None or x.get("examples")!=y.get("examples"):
            failures.append("missing_system:"+mode)
        elif x["accuracy"]-y["accuracy"]>policy["max_accuracy_regression"]+1e-12:failures.append("system:"+mode)
    for key in ("repetition_rate","suffix_accuracy","primary"):
        if before.get(key) is None or after.get(key) is None or not np.isfinite([before[key],after[key]]).all():
            failures.append("missing:"+key)
    if before.get("repetition_rate") is not None and after.get("repetition_rate") is not None:
        if after["repetition_rate"]-before["repetition_rate"]>policy["max_repetition_increase"]+1e-12:failures.append("repetition")
    return failures


def assess(base,candidate,config,*,confirmation=False):
    if base["phase"]!=candidate["phase"]:raise ValueError("Evaluation phases differ")
    gain=grouped_gain(evidence(base,"primary"),evidence(candidate,"primary"),seed=config["seed"],
                      replicates=config["selection"]["bootstrap_replicates"])
    failures=retention(base["metrics"],candidate["metrics"],config["selection"])
    if confirmation and (gain["gain"]+1e-12<config["selection"]["min_gain"] or gain["lower_95"]<=0):
        failures.append("confirmation_improvement")
    return {"accepted":not failures,"failures":failures,"primary":gain,
            "raw_intervals":paired_raw_intervals(evidence(base,"raw"),evidence(candidate,"raw"),config["seed"],config["selection"]["bootstrap_replicates"]),
            "limitation":"Point-estimate preservation guards are not statistical non-inferiority claims."}


def replicated(base,members,config,parents=None):
    if len(members)!=len(config["seeds"]) or {m["seed"] for m in members}!=set(config["seeds"]):
        raise ValueError("Finalist requires every registered seed exactly once")
    deltas=[];reports=[]
    for member in members:
        result=member["result"]
        if result["phase"]!="development":raise ValueError("Only development may rank recipes")
        report=assess(base,result,config);delta=result["metrics"]["primary"]-base["metrics"]["primary"]
        if parents:
            parent=parents[member["seed"]]
            if result["provenance"]["parent_model_sha256"]!=parent["export_model_sha256"]:raise ValueError("Incorrect marginal parent")
            report["parent_gain"]=result["metrics"]["primary"]-parent["metrics"]["primary"]
            if report["parent_gain"]<=0:report["failures"].append("nonpositive_parent_gain")
        reports.append({"seed":member["seed"],"delta":delta,**report});deltas.append(delta)
    mean=float(np.mean(deltas));robust=all(not r["failures"] for r in reports) and mean>0
    return {"robust":robust,"mean_gain":mean,"range":[min(deltas),max(deltas)],"seeds":reports,
            "promotion_eligible":robust and mean+1e-12>=config["selection"]["min_gain"]}


def immutable(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if digest(read(path))!=digest(value):raise ValueError("Frozen decision changed")
    else:write_json(path,value)
    return value


def claim_confirmation(frozen,manifest,config,directory=None):
    # Actual question/answer content and base weights, not a mutable run/config name.
    root=resolve(directory) if directory else ROOT/"artifacts/posttraining_confirmation_claims_v2"
    root.mkdir(parents=True,exist_ok=True)
    panel=manifest["confirmation_content_sha256"]
    key=digest([config["parent_sha256"],panel]);path=root/(key+".json")
    claim={"base_sha256":config["parent_sha256"],"panel_sha256":panel,
           "candidate_sha256":frozen["candidate_sha256"],"frozen_sha256":digest(frozen),
           "item_sha256":manifest.get("confirmation_item_sha256",[])}
    # Serialize competing claims and recover an interrupted write without reopening the panel.
    import fcntl
    with (root/".registry.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        items=set(claim["item_sha256"])
        for other in root.glob("*.json"):
            previous=read(other)
            if (other!=path and previous["base_sha256"]==claim["base_sha256"]
                    and items.intersection(previous.get("item_sha256",[]))):
                raise ValueError("Confirmation reuses previously consumed protected content")
        immutable(path,claim)
    return path


def verify_claim(path,model_sha256,manifest,config):
    claim=read(path)
    if (claim["base_sha256"]!=config["parent_sha256"] or claim["panel_sha256"]!=manifest["confirmation_content_sha256"]
            or model_sha256 not in (claim["base_sha256"],claim["candidate_sha256"])):
        raise ValueError("Confirmation does not match the one-use claim")
