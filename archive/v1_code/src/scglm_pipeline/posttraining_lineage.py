"""Local scratch ancestry and sealed intermediate export identities."""
from pathlib import Path
from safetensors import safe_open
import torch
from .common import read, sha256, resolve
from scglm_post.common import pretraining_tokens


def identity(path, config, *, allow_stages=None, _seen=None):
    path=resolve(path).resolve()
    seen=set() if _seen is None else set(_seen)
    if str(path) in seen or len(seen)>8:raise ValueError("Cyclic or excessively deep ancestry")
    seen.add(str(path))
    provenance=read(path/"training_provenance.json"); architecture=read(path/"config.json")
    weight_hash=sha256(path/"model.safetensors")
    stage=provenance.get("stage","pretraining")
    if stage not in ("pretraining","sft"):raise ValueError("Unknown ancestry stage")
    if allow_stages is not None and stage not in allow_stages:
        raise ValueError("Parent stage is not admitted: "+stage)
    disposable=bool(config.get("disposable_smoke_test"))
    if provenance.get("disposable_smoke_test") and not disposable:
        raise ValueError("Disposable model cannot enter competition ancestry")
    if architecture.get("scglm_initialization")!="random":
        raise ValueError("Only the project's scratch lineage is admitted")
    if not disposable and (architecture["vocab_size"]!=16384 or architecture["max_position_embeddings"]!=1024):
        raise ValueError("Fixed architecture changed")
    if stage=="pretraining":
        if weight_hash!=config["parent_sha256"] or path!=resolve(config["parent"]).resolve():
            raise ValueError("Base checkpoint identity changed")
        if read(path.parent/"status.json")["status"]!="completed":
            raise ValueError("Scratch parent is incomplete")
    else:
        if provenance.get("pipeline") not in ("posttraining-sota-v1","selected-sft-reproduction-v1") or provenance.get("scratch_base_sha256")!=config["parent_sha256"]:
            raise ValueError("Unregistered post-training ancestry")
        seal=read(path/"endpoint.json")
        if seal["weights_sha256"]!=weight_hash or seal["provenance_sha256"]!=sha256(path/"training_provenance.json"):
            raise ValueError("Sealed endpoint changed")
        parent=resolve(provenance["parent_model"])
        if sha256(parent/"model.safetensors")!=provenance["parent_model_sha256"]:
            raise ValueError("Parent weights changed")
        if sha256(parent/"tokenizer.json")!=sha256(path/"tokenizer.json"):
            raise ValueError("Tokenizer changed in descendant")
        identity(parent,config,allow_stages=["pretraining"],_seen=seen)
        if provenance.get("optimizer_updates",0)<1:raise ValueError("No completed training update in endpoint")
    if pretraining_tokens(provenance)!=config["expected_pretraining_tokens"]:
        raise ValueError("Pretraining ancestry count changed")
    count=0
    with safe_open(path/"model.safetensors",framework="pt",device="cpu") as archive:
        for key in archive.keys():
            value=archive.get_tensor(key)
            if not torch.isfinite(value).all(): raise ValueError("Nonfinite exported weights")
            count+=value.numel()
    if count!=config["expected_parameters"] or count>50_000_000:
        raise ValueError("Unique parameter count changed")
    return {"model":str(path),"run":str(path.parent),"export_model_sha256":weight_hash,
            "stage":stage,"provenance":provenance,"parameter_count":count,
            "tokenizer_sha256":sha256(path/"tokenizer.json")}
