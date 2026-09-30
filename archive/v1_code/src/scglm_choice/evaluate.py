"""Recoverable, contract-bound development and confirmation scoring."""
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from scglm.model import load_model
from scglm_pipeline.task_registry import prompt_text, verify
from scglm_pipeline.task_registry import raw_row
from .common import ROOT, DOMAINS, roots, read, read_jsonl, write_json, sha256, digest, immutable, sources
from .objective import choice_rows, logps


@torch.inference_mode()
def score_choice(model, tokenizer, row, template, charge):
    encoded = choice_rows(tokenizer, row, template)
    values, lengths = logps(model, encoded, charge=charge, kind="evaluation_choice")
    scores = values.cpu().tolist()
    chars = [max(1, len(c.strip())) for c in row["choices"]]
    predicted = int(values.argmax())
    normalized = int(np.argmax(np.array(scores) / chars))
    gold = row["answer_index"]
    return {"id": row["id"], "group": row["group_id"], "domain": row["domain"],
            "correct": float(predicted == gold), "character_normalized_correct": float(normalized == gold),
            "scores": scores, "gold_nll_sum": -scores[gold], "gold_tokens": int(lengths[gold]),
            "gold_characters": chars[gold], "target_lengths": lengths.tolist(), "predicted": predicted}


@torch.inference_mode()
def instruction(model, tokenizer, row, charge):
    prefix = tokenizer.encode(prompt_text(row), add_special_tokens=False)
    device = next(model.parameters()).device
    ids = torch.tensor([prefix], device=device)
    generated, past = [], None
    for _ in range(128):
        charge(ids.numel(), "evaluation_instruction")
        output = model(input_ids=ids, past_key_values=past, use_cache=True)
        if not torch.isfinite(output.logits).all():
            raise ValueError("Nonfinite generation logits")
        next_id = int(output.logits[0, -1].argmax())
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        past = output.past_key_values
        ids = torch.tensor([[next_id]], device=device)
    ended = generated[-1] == tokenizer.eos_token_id
    text = tokenizer.decode(generated[:-1] if ended else generated, skip_special_tokens=False)
    correct = verify(row, text)
    if correct is None:
        raise ValueError("Unverifiable historical retention item")
    return {"id": row["id"], "family": row["family"], "group": row["group_id"],
            "correct": float(correct and ended), "answer_correct": bool(correct), "ended_eos": ended,
            "generated_tokens": len(generated), "response": text, "historical_retention_diagnostic": True}


def _completed_lines(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    boundary = raw.rfind(b"\n") + 1
    if boundary < len(raw):
        # Incomplete final write is not a completed measurement. Its GPU work
        # remains charged in the separate write-ahead resource ledger.
        with path.open("r+b") as stream:
            stream.truncate(boundary)
    return [json.loads(line) for line in raw[:boundary].splitlines()]


def evaluate(cfg, manifest, path, split, directory, charge, *, model=None, claim=None):
    from .selection import verify_claim
    if split not in ("development", "confirmation"):
        raise ValueError("Unknown evaluation panel")
    path, directory = Path(path), Path(directory)
    data, _ = roots(cfg)
    weight_hash = sha256(path / "model.safetensors")
    if split == "confirmation":
        if claim is None:
            raise ValueError("Reserved panel requires the four-model one-use claim")
        verify_claim(claim, manifest, weight_hash)
    contract = {"model_sha256": weight_hash, "manifest_sha256": sha256(data / "manifest.json"),
                "split": split, "source_contract": sources(), "presentations": cfg["presentations"],
                "claim_sha256": sha256(claim) if claim else None}
    directory.mkdir(parents=True, exist_ok=True)
    immutable(directory / "registration.json", contract)
    if (directory / "result.json").exists():
        return load_result(directory)
    own_model = model is None
    if model is None:
        model = load_model(path).to("cuda:0")
    was_training = model.training
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    try:
        for kind in ("primary", "raw", "instructions", "transfer"):
            rows = read_jsonl(data / ("instructions.jsonl" if kind == "instructions" else f"{kind}_{split}.jsonl"))
            file = directory / f"{kind}.jsonl"
            completed = _completed_lines(file)
            seen = {r["id"] for r in completed}
            if len(seen) != len(completed) or not seen.issubset({r["id"] for r in rows}):
                raise ValueError("Partial evaluation membership changed")
            with file.open("a") as stream, torch.inference_mode():
                for row in rows:
                    if row["id"] in seen:
                        continue
                    if kind in ("primary", "transfer"):
                        templates = cfg["presentations"] if kind == "primary" else [cfg["presentations"][0]]
                        variants = [score_choice(model, tokenizer, row, t, charge) for t in templates]
                        result = {"id": row["id"], "group": row["group_id"], "domain": row["domain"], "variants": variants,
                                  **{k: float(np.mean([v[k] for v in variants])) for k in ("correct", "character_normalized_correct")},
                                  "gold_nll_sum": sum(v["gold_nll_sum"] for v in variants),
                                  "gold_tokens": sum(v["gold_tokens"] for v in variants)}
                    elif kind == "raw":
                        values, lengths = logps(model, [raw_row(row["tokens"])], charge=charge, kind="evaluation_raw")
                        result = {"id": row["id"], "group": row["group_id"], "source": row["source"],
                                  "nll_sum": -float(values[0]), "token_count": int(lengths[0])}
                    else:
                        result = instruction(model, tokenizer, row, charge)
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        if sha256(path / "model.safetensors") != weight_hash:
            raise ValueError("Export changed during evaluation")
        records = {kind: read_jsonl(directory / f"{kind}.jsonl") for kind in ("primary", "raw", "instructions", "transfer")}
        metrics = summarize(records)
        result = {"status": "completed", "contract": contract, "metrics": metrics,
                  "directory": str(directory), "files": {k+".jsonl": sha256(directory / (k+".jsonl")) for k in records}}
        write_json(directory / "result.json", result)
        return result
    finally:
        model.train(was_training)
        if own_model:
            del model
            torch.cuda.empty_cache()


def summarize(records):
    result = {"domains": {}, "instructions": {}, "raw": {}, "transfer_diagnostic_only": {}}
    for kind, group_key, dest in (("primary","domain","domains"), ("instructions","family","instructions"),
                                   ("transfer","domain","transfer_diagnostic_only")):
        groups = defaultdict(list)
        for row in records[kind]:
            groups[row[group_key]].append(row)
        for name, rows in groups.items():
            result[dest][name] = {"accuracy": float(np.mean([r["correct"] for r in rows])), "items": len(rows)}
            if kind == "primary":
                result[dest][name].update(character_normalized_accuracy=float(np.mean([r["character_normalized_correct"] for r in rows])),
                    gold_answer_nll=sum(r["gold_nll_sum"] for r in rows)/sum(r["gold_tokens"] for r in rows),
                    templates={str(i): float(np.mean([r["variants"][i]["correct"] for r in rows])) for i in range(2)},
                    length_strata={label: {"items":len(subset), "accuracy":float(np.mean([r["correct"] for r in subset])) if subset else None}
                                   for label, subset in (("gold_le_4_tokens", [r for r in rows if r["variants"][0]["gold_tokens"] <= 4]), ("gold_gt_4_tokens", [r for r in rows if r["variants"][0]["gold_tokens"] > 4]))} )
    if set(result["domains"]) != set(DOMAINS):
        raise ValueError("Incomplete primary domains")
    result["primary"] = float(np.mean([result["domains"][d]["accuracy"] for d in DOMAINS]))
    for source in sorted({r["source"] for r in records["raw"]}):
        rows = [r for r in records["raw"] if r["source"] == source]
        count = sum(r["token_count"] for r in rows)
        result["raw"][source] = {"nll": sum(r["nll_sum"] for r in rows)/count, "token_count": count, "documents": len(rows)}
    return result


def load_result(directory):
    result = read(Path(directory) / "result.json")
    for name, expected in result["files"].items():
        if sha256(Path(directory) / name) != expected:
            raise ValueError("Evaluation evidence changed")
    return result


def evidence(result, kind):
    file = Path(result["directory"]) / (kind + ".jsonl")
    if sha256(file) != result["files"][file.name]:
        raise ValueError("Evaluation evidence hash changed")
    return read_jsonl(file)
