#!/usr/bin/env python
"""Run the frozen official protocol only after checkpoint selection is complete.

--prepare-only validates task datasets/configuration and writes the exact planned
commands without loading a model, running inference, or calculating scores.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from validation_history import ValidationRun, harness_metrics, sha256, wikitext_metrics, write_json


ROOT = Path(__file__).resolve().parents[1]



def prepare(config_path: Path, output: Path) -> tuple[dict, Path, Path, dict]:
    import yaml

    config = json.loads(config_path.read_text())
    harness = config["harness"]
    repo = ROOT / harness["checkout"]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    if actual_commit != harness["commit"] or dirty:
        raise ValueError("Harness checkout must be clean and match the frozen commit")
    for package, version in config["runtime_versions"].items():
        actual = importlib.metadata.version(package)
        if actual != version:
            raise ValueError(f"Frozen runtime mismatch: {package} {actual} != {version}")
    for relative, expected in harness["support_file_sha256"].items():
        if sha256(repo / relative) != expected:
            raise ValueError(f"Pinned harness support file changed: {relative}")
    wiki = config["wikitext103"]
    if sha256(ROOT / wiki["source_file"]) != wiki["source_sha256"]:
        raise ValueError("WikiText-103 scorer changed after protocol freeze")

    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists() or (output / "harness").exists():
        raise ValueError("Use a fresh output directory; existing final scores must not be overwritten")
    overlay_dir = output / "task_overrides"
    overlay_dir.mkdir(exist_ok=True)
    overrides = {}
    for name, definition in harness["task_definitions"].items():
        base = repo / definition["base_yaml"]
        if sha256(base) != definition["base_sha256"]:
            raise ValueError(f"Pinned task definition changed: {name}")
        payload = {"include": str(base), "task": name, **definition["overrides"]}
        path = overlay_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        overrides[name] = {"path": str(path), "sha256": sha256(path)}

    # Resolve the actual frozen implementation, including the original !function
    # helpers. Loading task datasets does not instantiate or evaluate any model.
    sys.path.insert(0, str(repo))
    import lm_eval
    from lm_eval.tasks import TaskManager

    if not Path(lm_eval.__file__).resolve().is_relative_to(repo.resolve()):
        raise ValueError("Python imported a harness outside the pinned checkout")
    loaded = TaskManager(include_path=str(overlay_dir)).load(harness["tasks"])
    validated = {}
    for name, definition in harness["task_definitions"].items():
        task = loaded["tasks"][name]
        expected_split = definition["evaluation_split"]
        docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
        actual_split = task.config.test_split if task.has_test_docs() else task.config.validation_split
        if actual_split != expected_split or len(docs) != definition["expected_examples"]:
            raise ValueError(f"Frozen split/count mismatch for {name}: {actual_split}, {len(docs)}")
        if task.config.target_delimiter != " " or task.config.num_fewshot != 0:
            raise ValueError(f"Frozen prompt delimiter/few-shot mismatch for {name}")
        actual_metrics = list(task.aggregation())
        if actual_metrics != definition["metrics"]:
            raise ValueError(f"Frozen metrics mismatch for {name}: {actual_metrics}")
        validated[name] = {"split": actual_split, "examples": len(docs), "metrics": actual_metrics}
    from datasets import load_dataset

    wiki_rows = load_dataset(wiki["dataset"], wiki["configuration"], split=wiki["split"], revision=wiki["revision"])
    if len(wiki_rows) != wiki["raw_split_rows"]:
        raise ValueError("Frozen WikiText-103 row count mismatch")
    validated["wikitext103"] = {"split": wiki["split"], "raw_rows": len(wiki_rows),
                                "revision": wiki["revision"]}
    manifest = {
        "protocol": config["protocol_version"], "protocol_sha256": sha256(config_path),
        "harness_commit": actual_commit, "task_overrides": overrides,
        "task_validation": validated, "runtime_versions": config["runtime_versions"],
        "runner_sha256": sha256(Path(__file__)), "model_loaded": False,
        "scores_calculated": False,
    }
    write_json(output / "protocol_snapshot.json", config)
    write_json(output / "preparation.json", manifest)
    return config, repo, overlay_dir, manifest


def commands(config: dict, repo: Path, overlays: Path, model: Path, output: Path, device: str) -> list[dict]:
    harness, wiki = config["harness"], config["wikitext103"]
    model_args = {"pretrained": str(model), **harness["model_args"]}
    encoded_args = ",".join(f"{key}={value}" for key, value in model_args.items())
    harness_command = [
        sys.executable, "-m", "lm_eval", "run", "--model", "hf",
        "--model_args", encoded_args, "--tasks", *harness["tasks"],
        "--include_path", str(overlays), "--num_fewshot", str(harness["num_fewshot"]),
        "--batch_size", str(harness["batch_size"]), "--device", device,
        "--seed", ",".join(map(str, harness["seeds"])),
        "--output_path", str(output / "harness"), "--log_samples",
    ]
    wiki_command = [
        sys.executable, "-m", "scglm.evaluate", "wikitext103",
        "--model", str(model), "--tokenizer", str(model),
        "--dataset-revision", wiki["revision"], "--split", wiki["split"],
        "--context-length", str(wiki["context_length"]), "--stride", str(wiki["stride"]),
        "--device", device, "--output", str(output / "wikitext103.json"), "--final-evaluation",
    ]
    return [{"cwd": str(repo), "command": harness_command}, {"cwd": str(ROOT), "command": wiki_command}]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation.json")
    parser.add_argument("--output", type=Path, help="New artifact directory; defaults to a unique directory per run")
    parser.add_argument("--history-dir", type=Path, default=ROOT / "artifacts/validation")
    parser.add_argument("--label", default="", help="Human-readable reason or model label stored in history")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--selection-record", type=Path,
                        help="Frozen selection.json from scripts/select_endpoint.py")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--final-evaluation", action="store_true",
                        help="Attest that selection-development checkpoint selection is already frozen")
    args = parser.parse_args(argv)
    with ValidationRun(args.history_dir, output=args.output, metadata={
        "mode": "prepare" if args.prepare_only else "official", "label": args.label,
        "model": str(args.model.resolve()) if args.model else None,
        "config": str(args.config.resolve()), "device": args.device,
        "selection_record": str(args.selection_record.resolve()) if args.selection_record else None,
        "invocation": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }) as run:
        if not args.prepare_only and (not args.final_evaluation or args.model is None or args.selection_record is None):
            raise ValueError("Actual evaluation requires --model, --selection-record, and --final-evaluation")
        execute(args, run)


def execute(args, run: ValidationRun) -> None:
    output, config_path = run.output, args.config.resolve()
    run.claim_output()
    # Preserve the requested protocol even when preflight fails.
    config_bytes = config_path.read_bytes()
    (output / "protocol_requested.json").write_bytes(config_bytes)
    run.update(protocol_sha256=sha256(config_path), runner_sha256=sha256(Path(__file__)),
               history_implementation_sha256=sha256(Path(__file__).with_name("validation_history.py")))
    config, repo, overlays, manifest = prepare(config_path, output)
    run.update(preparation=manifest)
    model = args.model.resolve() if args.model is not None else ROOT / "MODEL_EXPORT_NOT_SELECTED"
    planned = commands(config, repo, overlays, model, output, args.device)
    write_json(output / "commands.json", planned)
    if args.prepare_only:
        print(json.dumps({"status": "prepared_without_model_or_scores", **manifest}, indent=2))
        return

    if not model.is_dir() or not (model / "training_provenance.json").is_file():
        raise ValueError("Final evaluation requires a local export with training provenance")
    selection_path = args.selection_record.resolve()
    selection = json.loads(selection_path.read_text())
    if selection.get("official_scores_used") is not False or selection.get("panel") != "selection":
        raise ValueError("Selection must use only the reserved selection-development panel")
    selected_model = Path(selection["selected_model"]).resolve()
    if model == selected_model:
        frozen_weight_hash = selection["selected_model_sha256"]
    else:
        # Report all preregistered completed arms, including non-winning arms,
        # without allowing their official scores to revise the chosen checkpoint.
        matches = [result for result in selection["results"]
                   if (Path(result["run"]) / "export").resolve() == model]
        if len(matches) != 1:
            raise ValueError("Model is not an endpoint listed in the frozen selection record")
        frozen_weight_hash = matches[0]["export_model_sha256"]
    if sha256(model / "model.safetensors") != frozen_weight_hash:
        raise ValueError("Model weights changed after development-only endpoint selection")
    model_config = json.loads((model / "config.json").read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    if len(tokenizer) != model_config["vocab_size"]:
        raise ValueError("Exported tokenizer/model vocabularies differ")
    if tokenizer.eos_token_id != config["harness"]["model_args"]["prefix_token_id"]:
        raise ValueError("Exported EOS differs from frozen empty-context prefix ID")
    if model_config["max_position_embeddings"] != config["harness"]["model_args"]["max_length"]:
        raise ValueError("Exported model context differs from protocol")
    weights = sorted(model.glob("*.safetensors"))
    if not weights:
        raise ValueError("Local export has no safetensors weights")
    artifact_files = sorted(set(weights + list(model.glob("*.json"))))
    started = time.time()
    run_record = {**manifest, "model": str(model), "artifact_sha256": {p.name: sha256(p) for p in artifact_files},
                  "selection_frozen_attested": True, "selection_record": str(selection_path),
                  "selection_record_sha256": sha256(selection_path),
                  "selected_model": str(selected_model), "is_selected_endpoint": model == selected_model,
                  "started_unix": started, "device": args.device}
    run_record["validation_run_id"] = run.run_id
    run.update(execution=run_record)
    write_json(output / "execution.json", run_record)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(repo), str(ROOT / "src")])
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    for name, specification in zip(("harness", "wikitext103"), planned):
        run.run_stage(name, specification, environment)
        # Persist each stage's scores immediately, even if a later stage fails.
        partial = harness_metrics(output, config) if name == "harness" else wikitext_metrics(output, config)
        run.update(metrics={**run.record["metrics"], **partial})
    # Do not attribute scores to an export replaced during inference.
    if any(sha256(model / name) != digest for name, digest in run_record["artifact_sha256"].items()):
        raise ValueError("Model artifacts changed during evaluation")
    summary = {"schema_version": 1, "run_id": run.run_id, "model": str(model),
               "protocol": config["protocol_version"], "protocol_sha256": manifest["protocol_sha256"],
               "artifact_sha256": run_record["artifact_sha256"], "metrics": run.record["metrics"]}
    write_json(output / "summary.json", summary)
    write_json(output / "completed.json", {
        **run_record, "model_loaded": True, "scores_calculated": True,
        "finished_unix": time.time(), "elapsed_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
