"""Benchmark items in the exact lm-evaluation-harness v0.4.12 scoring format.

Used by the task-tuning phase (v2/phases/task_tune). Training examples and the
development scorer build the same (context, continuation) pairs that the pinned
harness scores, and encode them the same way (``TemplateLM._encode_pair`` with no
BOS). Templates follow the pinned task YAMLs:

* hellaswag: context = preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize()),
  continuation = " " + preprocess(ending)
* arc_easy / arc_challenge: context = "Question: {question}\\nAnswer:", continuation = " " + choice
* piqa: context = "Question: {goal}\\nAnswer:", continuation = " " + sol
* winogrande (partial scoring): one context per option, sentence[:idx] + option;
  shared continuation " " + sentence[idx + 1:].strip()

``acc`` is the argmax of summed continuation log-likelihood; ``acc_norm`` divides it by
``len(choice)`` in characters (WinoGrande: acc only). ``verify`` checks the formatter
and scorer against the per-sample records of an already completed official run.

Run: PYTHONPATH=src python -m scglm_v2.task_format --verify <harness samples dir> --model <export>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F

MAX_LENGTH = 1024
PROXY_TASKS = ("arc_easy", "piqa", "hellaswag", "winogrande")


def hellaswag_preprocess(text: str) -> str:
    """Copy of lm_eval/tasks/hellaswag/utils.preprocess (pinned release)."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def harness_item(task: str, doc: dict) -> dict:
    """Return {task, pairs: [(context, continuation)], label, choices} for one raw dataset row."""
    if task == "hellaswag":
        query = hellaswag_preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())
        choices = [hellaswag_preprocess(e) for e in doc["endings"]]
        return {"task": task, "pairs": [(query, " " + c) for c in choices], "label": int(doc["label"]), "choices": choices}
    if task in ("arc_easy", "arc_challenge"):
        choices = list(doc["choices"]["text"])
        context = f"Question: {doc['question']}\nAnswer:"
        return {"task": task, "pairs": [(context, " " + c) for c in choices],
                "label": doc["choices"]["label"].index(doc["answerKey"]), "choices": choices}
    if task == "piqa":
        choices = [doc["sol1"], doc["sol2"]]
        context = f"Question: {doc['goal']}\nAnswer:"
        return {"task": task, "pairs": [(context, " " + c) for c in choices], "label": int(doc["label"]), "choices": choices}
    if task == "winogrande":
        idx = doc["sentence"].index("_")
        target = doc["sentence"][idx + 1:].strip()
        contexts = [doc["sentence"][:idx] + opt for opt in (doc["option1"], doc["option2"])]
        return {"task": task, "pairs": [(c, " " + target) for c in contexts],
                "label": {"1": 0, "2": 1}[doc["answer"]], "choices": contexts}
    raise ValueError(task)


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Harness ``_encode_pair`` for causal models: trailing context spaces move to the continuation."""
    if not context:
        raise ValueError("Empty contexts are not used by these tasks")
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    ctx = tokenizer.encode(context, add_special_tokens=False)
    return ctx, whole[len(ctx):]


def model_input(ctx: list[int], cont: list[int]) -> tuple[list[int], int]:
    """Harness truncation: keep the last MAX_LENGTH + 1 tokens and drop the final one as input."""
    if len(cont) > MAX_LENGTH:
        raise ValueError("Continuation longer than the model context")
    full = (ctx + cont)[-(MAX_LENGTH + 1):]
    return full, len(cont)


def sequence_logprobs(model, sequences: list[tuple[list[int], list[int]]], device: str,
                      max_rows: int = 256, autocast: bool = False) -> list[float]:
    """Summed continuation log-likelihood for each (context_ids, continuation_ids) pair.

    Rows are right-padded; with causal attention, padding never affects real positions.
    ``autocast=False`` keeps FP32 computation, as in the official protocol.
    """
    order = sorted(range(len(sequences)), key=lambda i: -(len(sequences[i][0]) + len(sequences[i][1])))
    out = [0.0] * len(sequences)
    for start in range(0, len(order), max_rows):
        rows = order[start:start + max_rows]
        fulls = [model_input(*sequences[i]) for i in rows]
        width = max(len(f) - 1 for f, _ in fulls)
        x = torch.zeros((len(rows), width), dtype=torch.long)
        for r, (full, _) in enumerate(fulls):
            x[r, :len(full) - 1] = torch.tensor(full[:-1])
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=autocast and device.startswith("cuda")):
            logits = model(input_ids=x.to(device), use_cache=False).logits
        logits = logits.float()
        for r, (full, n) in enumerate(fulls):
            inplen = len(full) - 1
            piece = logits[r, inplen - n:inplen]
            target = torch.tensor(full[-n:], device=piece.device)
            out[rows[r]] = float((piece.gather(1, target[:, None]).squeeze(1) - torch.logsumexp(piece, dim=-1)).sum())
    return out


def score_items(model, items: list[dict], device: str, max_rows: int = 256) -> dict:
    """acc / acc_norm per task for tokenized items (keys: task, label, choices, encoded)."""
    flat, owners = [], []
    for k, item in enumerate(items):
        for j, pair in enumerate(item["encoded"]):
            flat.append((pair[0], pair[1])); owners.append((k, j))
    values = sequence_logprobs(model, flat, device, max_rows=max_rows)
    scores = [[0.0] * len(item["encoded"]) for item in items]
    for (k, j), v in zip(owners, values):
        scores[k][j] = v
    totals: dict[str, dict] = {}
    for item, s in zip(items, scores):
        t = totals.setdefault(item["task"], {"acc": 0, "acc_norm": 0, "n": 0})
        t["n"] += 1
        t["acc"] += int(max(range(len(s)), key=s.__getitem__) == item["label"])
        if item["task"] != "winogrande":
            norm = [v / len(c) for v, c in zip(s, item["choices"])]
            t["acc_norm"] += int(max(range(len(norm)), key=norm.__getitem__) == item["label"])
    result = {}
    for task, t in totals.items():
        result[task] = {"acc": t["acc"] / t["n"], "n": t["n"]}
        if task != "winogrande":
            result[task]["acc_norm"] = t["acc_norm"] / t["n"]
    proxy = [result[k]["acc"] for k in PROXY_TASKS if k in result]
    result["_proxy_mean_acc"] = sum(proxy) / len(proxy) if len(proxy) == len(PROXY_TASKS) else None
    return result


def encode_item(tokenizer, item: dict) -> dict:
    return {**item, "encoded": [list(map(list, encode_pair(tokenizer, c, k))) for c, k in item["pairs"]]}


def verify(samples_dir: Path, model_dir: str | None, device: str, loglik_items: int) -> dict:
    """Formatter equality on every recorded official request; scorer agreement on a few items.

    Uses only per-sample records already written by a completed official evaluation of
    ``model_dir``; no new model is scored on official data beyond reproducing those values.
    """
    from transformers import PreTrainedTokenizerFast
    report = {}
    model = tok = None
    if model_dir:
        from scglm.model import load_model
        model = load_model(model_dir).to(device).eval()
        tok = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    for task in PROXY_TASKS:
        path = next(samples_dir.glob(f"samples_{task}_*.jsonl"))
        mismatches, n, checked, max_diff = 0, 0, 0, 0.0
        for line in path.open():
            rec = json.loads(line)
            item = harness_item(task, rec["doc"])
            recorded = [(a["arg_0"], a["arg_1"]) for a in rec["arguments"].values()]
            n += 1
            # WinoGrande's recorded target is the shared suffix, not the gold index.
            gold_differs = task != "winogrande" and int(rec["target"]) != item["label"]
            if recorded != item["pairs"] or gold_differs:
                mismatches += 1
            if model is not None and checked < loglik_items:
                ours = sequence_logprobs(model, [encode_pair(tok, c, k) for c, k in item["pairs"]], device)
                theirs = [float(r[0]) for r in rec["filtered_resps"]]
                max_diff = max(max_diff, max(abs(a - b) for a, b in zip(ours, theirs)))
                checked += 1
        report[task] = {"requests": n, "format_mismatches": mismatches, "loglik_items_checked": checked,
                        "max_abs_loglik_difference": max_diff}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", type=Path, required=True, help="harness output directory with samples_*.jsonl")
    p.add_argument("--model", help="the export those samples were computed for")
    p.add_argument("--loglik-items", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    report = verify(args.verify, args.model, args.device, args.loglik_items)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if any(r["format_mismatches"] for r in report.values()):
        raise SystemExit("Formatter differs from the recorded harness requests")


if __name__ == "__main__":
    main()
