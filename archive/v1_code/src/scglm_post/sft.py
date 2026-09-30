"""Token-weighted response-only SFT primitives; causal targets shift once."""
import math

import torch
import torch.nn.functional as F


def collate(rows, pad_token_id, device):
    length = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"], device=device)
        labels[i, :n] = torch.tensor(row["labels"], device=device)
        attention[i, :n] = 1
    return ids, labels, attention


def response_loss_sum(model, rows, *, pad_token_id=0):
    device = next(model.parameters()).device
    ids, labels, attention = collate(rows, pad_token_id, device)
    # Never pass labels to the model: the shift is explicit and occurs once.
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    targets = labels[:, 1:]
    count = int((targets != -100).sum())
    if count != sum(r["target_tokens"] for r in rows) or count <= 0:
        raise ValueError("Response target count mismatch")
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-100, reduction="sum")
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite SFT loss")
    return loss, count




def learning_rate(tokens, total, config):
    warmup = max(1, int(total * config["warmup_fraction"]))
    if tokens < warmup:
        return config["learning_rate"] * max(1, tokens) / warmup
    progress = min(1, (tokens - warmup) / max(1, total - warmup))
    return config["learning_rate"] * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


