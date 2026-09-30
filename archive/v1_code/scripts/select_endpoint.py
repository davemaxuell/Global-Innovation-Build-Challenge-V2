"""Select completed matched-budget endpoints on reserved development data only."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from scglm.evaluate import evaluate_documents, fixed_mixture_metrics, load_documents
from scglm.model import load_model
from scglm.train import atomic_json, configure_numerics, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="runs/baseline_1b")
    parser.add_argument("--candidates", nargs="*", default=[])
    parser.add_argument("--output", default="artifacts/selection/selection.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("Selection record already exists; do not silently revise after seeing final scores")
    configure_numerics({"deterministic": True}, args.device)
    if args.device.startswith("cuda") and torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one allocated GPU")
    torch.set_num_threads(8)
    paths = [Path(args.baseline).resolve(), *[Path(p).resolve() for p in args.candidates]]
    if len(paths) != len(set(paths)):
        raise ValueError("Each endpoint must appear once")
    runs = []
    for path in paths:
        run = json.loads((path / "run.json").read_text())
        status = json.loads((path / "status.json").read_text())
        provenance = json.loads((path / "export" / "training_provenance.json").read_text())
        if status["status"] != "completed" or status["main_tokens"] != run["exact_target_tokens"]:
            raise ValueError(f"Endpoint incomplete: {path}")
        if provenance["main_tokens"] != run["exact_target_tokens"] or provenance["science_digest"] != run["science_digest"]:
            raise ValueError(f"Export/run mismatch: {path}")
        if run["parameter_count"] != 46346752:
            raise ValueError("Unexpected architecture in matched experiment")
        runs.append({"path": path, "run": run, "status": status})
    reference = runs[0]["run"]
    if reference["config"]["arm"] != "B0":
        raise ValueError("Baseline must be the preregistered B0 arm")
    for item in runs[1:]:
        run = item["run"]
        for key in ("model_config", "exact_target_tokens", "manifest_sha256", "tokenizer_sha256", "train_shard_sha256", "versions", "runtime_fingerprint"):
            if run[key] != reference[key]:
                raise ValueError(f"Unmatched {key}: {item['path']}")
        for key in ("seed", "sequence_length", "batch_sequences", "microbatch_sequences", "learning_rate",
                    "min_lr_ratio", "warmup_fraction", "betas", "adam_epsilon", "weight_decay", "gradient_clip", "deterministic"):
            if run["config"][key] != reference["config"][key]:
                raise ValueError(f"Unmatched training setting {key}")
    manifest_path = Path(reference["config"]["data_manifest"])
    if sha256_file(manifest_path) != reference["manifest_sha256"]:
        raise ValueError("Data manifest changed after training")
    manifest = json.loads(manifest_path.read_text())
    documents = []
    for source in ("web", "wiki"):
        split = manifest["sources"][source]["splits"]["selection"]
        if sha256_file(split["jsonl_path"]) != split["jsonl_sha256"]:
            raise ValueError("Selection panel changed after freezing")
        documents.extend(load_documents(split["jsonl_path"]))
    results = []
    for item in runs:
        started = time.monotonic()
        model = load_model(item["path"] / "export").to(args.device)
        with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda")):
            records = evaluate_documents(model, documents, context_length=1024, stride=512,
                                         max_predictable_tokens=1024, device=args.device)
        metrics = fixed_mixture_metrics(records)
        result = {"run": str(item["path"]), "arm": item["run"]["config"]["arm"], "metrics": metrics,
                  "total_run_seconds": item["status"]["elapsed_seconds"],
                  "main_tokens": item["status"]["main_tokens"],
                  "probe_tokens": item["status"].get("probe_tokens", 0),
                  "selection_evaluation_seconds": time.monotonic()-started,
                  "export_model_sha256": sha256_file(item["path"] / "export" / "model.safetensors")}
        atomic_json(destination.parent / f"{item['path'].name}_selection.json", {**result, "documents": records})
        results.append(result)
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    baseline = results[0]
    for result in results:
        result["eligible"] = all(result["metrics"]["sources"][s]["nll"] <= baseline["metrics"]["sources"][s]["nll"]+0.02 for s in ("web", "wiki"))
    eligible = [r for r in results if r["eligible"]]
    best_loss = min(r["metrics"]["fixed_q_nll"] for r in eligible)
    tied = [r for r in eligible if r["metrics"]["fixed_q_nll"] <= best_loss+0.0001]
    chosen = min(tied, key=lambda r: (r["total_run_seconds"], r["run"]))
    atomic_json(destination, {"selected_run": chosen["run"], "selected_model": str(Path(chosen["run"])/"export"),
                             "selected_model_sha256": chosen["export_model_sha256"], "selected_at_unix": time.time(),
                             "panel": "selection", "official_scores_used": False,
                             "protocol": "Complete matched endpoints; fixed q 0.8/0.2 NLL, source regression <=0.02; ties <=0.0001 use total elapsed cost then lexical run ID; CUDA BF16 autocast with FP32 loss.",
                             "results": results})
    print(json.dumps({"selection_record": str(destination), "selected_run": chosen["run"]}, indent=2))


if __name__ == "__main__":
    main()
