"""Study-only scratch -> selected SFT -> choice endpoint validator."""
from pathlib import Path
from safetensors import safe_open
import torch

from scglm_pipeline.posttraining_lineage import identity as retained_identity
from .common import ROOT, read, sha256, digest, sources, runtime


def references(cfg):
    retained = read(ROOT / "configs/current_model.json")
    parent = ROOT / cfg["parent"]["path"]
    if sha256(parent / "model.safetensors") != cfg["parent"]["weight_sha256"]:
        raise ValueError("Selected SFT parent weights changed")
    base = ROOT / cfg["base_reference"]["path"]
    if sha256(base / "model.safetensors") != cfg["base_reference"]["weight_sha256"]:
        raise ValueError("Preserved base weights changed")
    result = retained_identity(parent, retained, allow_stages=["sft"])
    provenance = result["provenance"]
    if provenance["step"] != 105 or provenance["contract_sha256"] != "475d1b108966dae1b71d611370ba75b415bff46189412aa462d8a0e71cfee96e":
        raise ValueError("Parent is not the original selected 209-schedule update 105")
    chain = []
    for name in ("baseline_1b", "continuation_12b", "continuation_13b"):
        path = ROOT / "runs" / name
        prov = read(path / "export/training_provenance.json")
        if read(path / "status.json")["status"] != "completed" or prov["initialization"] != "scratch":
            raise ValueError("Incomplete scratch ancestry")
        if prov.get("parent"):
            ancestor = prov["parent"]
            if (sha256(Path(ancestor["run"]) / "run.json") != ancestor["run_record_sha256"] or
                    sha256(ancestor["checkpoint"]) != ancestor["checkpoint_sha256"]):
                raise ValueError("Scratch ancestry hash changed")
        chain.append({"run": name, "run_sha256": sha256(path / "run.json"),
                      "weights_sha256": sha256(path / "export/model.safetensors"), "targets": prov["main_tokens"]})
    if sum(p["targets"] for p in chain) != 13000048640:
        raise ValueError("Scratch target count changed")
    return {"parent": result, "scratch_chain": chain, "base_weights_sha256": sha256(base / "model.safetensors")}


def endpoint(path, cfg, *, eligible=False):
    path = Path(path)
    seal = read(path / "endpoint.json")
    prov = read(path / "training_provenance.json")
    h = sha256(path / "model.safetensors")
    if seal["weights_sha256"] != h or seal["provenance_sha256"] != sha256(path / "training_provenance.json"):
        raise ValueError("Study endpoint seal changed")
    if prov["stage"] != "choice_discrimination" or prov["pipeline"] != "scglm-choice-study-v1":
        raise ValueError("Wrong study ancestry")
    if prov["parent_model_sha256"] != cfg["parent"]["weight_sha256"] or prov["scratch_base_sha256"] != cfg["base_reference"]["weight_sha256"]:
        raise ValueError("Study parent identity changed")
    for key in ("parent", "base_reference"):
        if sha256(ROOT / cfg[key]["path"] / "model.safetensors") != cfg[key]["weight_sha256"]:
            raise ValueError("Preserved ancestry weights changed")
    if prov["cumulative_pretraining_tokens"] != 13000048640 or prov["historical_sft_updates"] != 105:
        raise ValueError("Study ancestry count changed")
    if sha256(path / "tokenizer.json") != sha256(ROOT / cfg["parent"]["path"] / "tokenizer.json"):
        raise ValueError("Study tokenizer changed")
    run = Path(seal["source_run"])
    contract = read(run / "run.json")["contract"]
    if digest(contract) != prov["contract_sha256"] or contract["plan_sha256"] != prov["plan_sha256"]:
        raise ValueError("Endpoint does not match the study run contract")
    if eligible and (prov["optimizer_updates"] != 32 or read(run / "status.json")["status"] != "completed"):
        raise ValueError("Only a completed update-32 endpoint is eligible")
    count = 0
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as archive:
        for k in archive.keys():
            value = archive.get_tensor(k)
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite endpoint weights")
            count += value.numel()
    if count != cfg["parent"]["parameters"] or not read(path / "config.json")["tie_word_embeddings"]:
        raise ValueError("Study parameter count or tied weights changed")
    return {"model": str(path), "weights_sha256": h, "provenance": prov}
