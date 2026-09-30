"""Recorded response-loss and deterministic instruction-following validation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch

from .common import (ROOT, ValidationRun, load_manifest, read_jsonl, render_prompt,
                     sha256, write_json, write_jsonl)
from .procedural import correct
from .sft import response_loss_sum


def generation_sample(examples, limit):
    """Balance system modes and keep counterfactual pairs together."""
    groups = defaultdict(list)
    for row in sorted(examples, key=lambda r: r["id"]):
        groups[row.get("pair_id", row["id"])].append(row)
    modes = defaultdict(list)
    for group in groups.values():
        modes[group[0].get("system_case", "none")].append(group)
    selected = []
    positions = {mode: 0 for mode in modes}
    while len(selected) < limit:
        added = False
        for mode in sorted(modes):
            position = positions[mode]
            if position < len(modes[mode]):
                group = modes[mode][position]
                if len(selected) + len(group) <= limit:
                    selected.extend(group)
                    positions[mode] += 1
                    added = True
        if not added:
            break
    return selected


def system_metrics(predictions):
    modes, pairs = defaultdict(list), defaultdict(list)
    for prediction in predictions:
        mode = prediction.get("system_case", "none")
        if mode != "none":
            modes[mode].append(prediction["correct"])
        if prediction.get("pair_id"):
            pairs[prediction["pair_id"]].append(prediction["correct"])
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("System evaluation requires complete policy pairs")
    return {"modes": {mode: {"accuracy": sum(values) / len(values), "examples": len(values)}
                      for mode, values in modes.items()},
            "policy_pairs": {"accuracy": sum(all(pair) for pair in pairs.values()) / len(pairs) if pairs else None,
                             "examples": len(pairs)}}


@torch.inference_mode()
def evaluate_records(model, rows, *, tokenizer=None, generate_limit=0, microbatch=8):
    was_training = model.training
    model.eval()
    total, targets = 0.0, 0
    predictions, families = [], defaultdict(list)
    try:
        for start in range(0, len(rows), microbatch):
            loss, count = response_loss_sum(model, rows[start:start + microbatch])
            total += float(loss.double())
            targets += count
        if not targets:
            raise ValueError("No validation targets")
        if generate_limit:
            if tokenizer is None:
                raise ValueError("Generation evaluation requires the frozen tokenizer")
            by_family = defaultdict(list)
            for row in rows:
                if row["source"] == "procedural":
                    by_family[row["category"]].append(row)
            if len(by_family) != 5:
                raise ValueError("Instruction validation must include all five task families")
            for family, examples in sorted(by_family.items()):
                for row in generation_sample(examples, generate_limit):
                    prompt_ids = tokenizer.encode(render_prompt(row["messages"][:-1]), add_special_tokens=False)
                    if prompt_ids != row["input_ids"][:row["prompt_length"]]:
                        raise ValueError("Generation/training format mismatch")
                    inputs = torch.tensor([prompt_ids], device=next(model.parameters()).device)
                    generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                               do_sample=False, max_new_tokens=min(256, model.config.max_position_embeddings - len(prompt_ids)),
                                               pad_token_id=0, eos_token_id=2, use_cache=True)
                    new = generated[0, len(prompt_ids):].tolist()
                    response = tokenizer.decode(new, skip_special_tokens=True)
                    passed = correct(row, response)
                    families[family].append(passed)
                    predictions.append({"id": row["id"], "family": family, "response": response,
                                        "group": row["group"], "system_case": row.get("system_case", "none"),
                                        "pair_id": row.get("pair_id"),
                                        "correct": passed, "generated_tokens": len(new), "ended_eos": bool(new and new[-1] == 2)})
        scores = {name: {"accuracy": sum(values) / len(values), "examples": len(values)}
                  for name, values in families.items()}
        nll = total / targets
        return {"response_nll": nll, "response_perplexity": math.exp(nll) if nll < 709 else None,
                "target_tokens": targets, "examples": len(rows), "families": scores,
                "system_instruction": system_metrics(predictions),
                "macro_accuracy": sum(v["accuracy"] for v in scores.values()) / len(scores) if scores else None}, predictions
    finally:
        model.train(was_training)


def recorded_validation(model, rows, *, metadata, tokenizer=None, generate_limit=0,
                        history=None):
    with ValidationRun(history or ROOT / "artifacts/validation", output=None,
                       metadata={"mode": "instruction-development", **metadata}) as run:
        run.claim_output()
        metrics, predictions = evaluate_records(model, rows, tokenizer=tokenizer, generate_limit=generate_limit)
        write_json(run.output / "metrics.json", metrics)
        if predictions:
            write_jsonl(run.output / "predictions.jsonl", predictions)
        run.update(metrics=metrics)
        return metrics, str(run.directory / "run.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/posttraining/instruction_v4/manifest.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--generate-per-family", type=int, default=200)
    args = parser.parse_args()
    from scglm.model import load_model
    from transformers import AutoTokenizer

    # Record even a missing model or invalid manifest as a failed validation.
    with ValidationRun(ROOT / "artifacts/validation", output=None, metadata={
        "mode": "instruction-development", "model": str(args.model.resolve()), "device": args.device,
    }) as run:
        run.claim_output()
        manifest = load_manifest(args.manifest)
        if sha256(args.model / "tokenizer.json") != manifest["tokenizer_sha256"]:
            raise ValueError("Model/data tokenizer mismatch")
        if args.device.startswith("cuda") and torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one assigned GPU")
        model = load_model(args.model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        rows = read_jsonl(args.manifest.parent / manifest["splits"]["development"]["file"])
        run.update(model_sha256=sha256(args.model / "model.safetensors"), manifest_sha256=sha256(args.manifest))
        metrics, predictions = evaluate_records(model, rows, tokenizer=tokenizer, generate_limit=args.generate_per_family)
        write_json(run.output / "metrics.json", metrics)
        write_jsonl(run.output / "predictions.jsonl", predictions)
        run.update(metrics=metrics)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
