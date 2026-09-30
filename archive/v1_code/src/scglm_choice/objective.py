"""Literal continuation scoring and the paired full-parameter objective."""
from collections import Counter
import re

import torch
import torch.nn.functional as F

from scglm_post.sft import collate, response_loss_sum
from .common import DOMAINS


def encode(tokenizer, prompt, continuation, *, eos=False, max_length=1024, max_response=128):
    """Pinned harness trailing-whitespace rule, with an explicit boundary check.

    No independently tokenized answers, synthetic BOS, truncation, or EOS in
    choice scores. Response CE alone appends its real EOS target.
    """
    if not prompt or not continuation.strip() or re.search(r"<\|(?:pad|bos|eos|unk)\|>", prompt + continuation):
        raise ValueError("Empty or reserved continuation text")
    spaces = len(prompt) - len(prompt.rstrip())
    if spaces:
        continuation, prompt = prompt[-spaces:] + continuation, prompt[:-spaces]
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    if not prefix or ids[:len(prefix)] != prefix:
        raise ValueError("Tokenizer crosses the continuation boundary")
    if eos:
        ids.append(tokenizer.eos_token_id)
    targets = len(ids) - len(prefix)
    if not 0 < targets <= max_response or len(ids) > max_length:
        raise ValueError("Complete continuation exceeds limits")
    return {"input_ids": ids, "labels": [-100] * len(prefix) + ids[len(prefix):],
            "prompt_length": len(prefix), "target_tokens": targets, "processed_tokens": len(ids)}


def choice_rows(tokenizer, item, template="{question}\nAnswer:"):
    if item.get("suffix_probe"):
        probe = item["suffix_probe"]
        return [encode(tokenizer, probe["prefix"] + c, probe["suffix"], max_response=1024)
                for c in item["choices"]]
    prompt = item.get("causal_prompt") or template.format(question=item["question"])
    return [encode(tokenizer, prompt, " " + c.strip(), max_response=1024) for c in item["choices"]]


def positions(rows):
    return len(rows) * max(len(r["input_ids"]) for r in rows)


def logps(model, rows, *, charge=None, kind="choice"):
    if charge:
        charge(positions(rows), kind)
    ids, labels, attention = collate(rows, model.config.pad_token_id, next(model.parameters()).device)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != -100
    lengths = mask.sum(1)
    if lengths.tolist() != [r["target_tokens"] for r in rows] or not lengths.gt(0).all():
        raise ValueError("Continuation target count mismatch")
    values = -F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction="none")
    scores = (values * mask).sum(1)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite continuation scores")
    return scores, lengths


def choice_loss(model, items, *, charge=None):
    """Each item's alternatives share one forward; equally weighted domains."""
    counts = Counter(r["domain"] for r in items)
    if set(counts) != set(DOMAINS):
        raise ValueError("Both registered choice domains are required")
    loss = None
    for item in items:
        scores, _ = logps(model, item["encoded_choices"], charge=charge)
        term = F.cross_entropy(scores[None], scores.new_tensor([item["answer_index"]], dtype=torch.long))
        term = term / (2 * counts[item["domain"]])
        loss = term if loss is None else loss + term
    return loss


def update(model, optimizer, response, replay, items, *, candidate, microbatch=8,
           charge=None, step=True):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = Counter()
    device = next(model.parameters()).device.type
    for name, rows, weight in (("response", response, .7), ("replay", replay, .3)):
        total = sum(r["target_tokens"] for r in rows)
        if total <= 0:
            raise ValueError("Both CE objectives require targets")
        loss_sum = 0.
        for start in range(0, len(rows), microbatch):
            micro = rows[start:start + microbatch]
            if charge:
                charge(positions(micro), name)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                value, _ = response_loss_sum(model, micro, pad_token_id=model.config.pad_token_id)
            (weight * value / total).backward()
            loss_sum += float(value.detach())
            metrics[name + "_padded_positions"] += positions(micro)
        metrics[name + "_nll"] = loss_sum / total
        metrics[name + "_targets"] = total
        metrics[name + "_processed_tokens"] = sum(r["processed_tokens"] for r in rows)
    if candidate:
        counts = Counter(r["domain"] for r in items)
        if set(counts) != set(DOMAINS):
            raise ValueError("Missing choice domain")
        for item in items:
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                scores, _ = logps(model, item["encoded_choices"], charge=charge)
                loss = F.cross_entropy(scores[None], scores.new_tensor([item["answer_index"]], dtype=torch.long))
            (.05 * loss / (2 * counts[item["domain"]])).backward()
            metrics["choice_loss"] += float(loss.detach()) / (2 * counts[item["domain"]])
            metrics["choice_targets"] += sum(r["target_tokens"] for r in item["encoded_choices"])
            metrics["choice_padded_positions"] += positions(item["encoded_choices"])
            metrics["choice_processed_tokens"] += sum(r["processed_tokens"] for r in item["encoded_choices"])
        metrics["choice_items"] = len(items)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    metrics["gradient_norm"] = float(norm)
    if step:
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite parameter after optimizer step")
        if any(not torch.isfinite(v).all() for state in optimizer.state.values()
               for v in state.values() if isinstance(v, torch.Tensor)):
            raise ValueError("Nonfinite optimizer state")
    return dict(metrics)
