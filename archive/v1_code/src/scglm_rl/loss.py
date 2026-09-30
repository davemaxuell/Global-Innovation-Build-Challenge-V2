"""Clipped policy gradient with exact categorical KL and two registered loss normalizations.

For family f with declared share s_f (renormalized over families present in
the batch), groups g of that family and scored response tokens t:

    token_mean:  L = sum_f s_f * (sum_{g in f} sum_t l_t) / T_f
    prompt_mean: L = sum_f s_f * (1/G_f) * sum_{g in f} (sum_t l_t) / T_g

T_f and T_g count scored tokens in the family and in the prompt group; G_f
counts prompt groups. ``token_mean`` is the historical global-token objective
applied within each family; ``prompt_mean`` is MiMo's prompt-level averaging
(§5.1). Family weighting is shared by both arms, so only the within-family
normalization differs.
"""
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from scglm_post.sft import collate

ARMS = ("token_mean", "prompt_mean")


class BehaviorMismatch(ValueError):
    """Training-time policy probabilities differ from the recorded sampling probabilities."""


def advantages(groups, epsilon=1e-4):
    """Group-mean-centered rewards scaled by one batch standard deviation (historical rule)."""
    rewards = torch.tensor([r["reward"] for g in groups for r in g["rows"]], dtype=torch.float32)
    if not torch.isfinite(rewards).all():
        raise ValueError("Nonfinite reward")
    scale = float(rewards.std(unbiased=False)) + epsilon
    result = {}
    for g in groups:
        values = torch.tensor([r["reward"] for r in g["rows"]], dtype=torch.float32)
        if len(values) < 2:
            raise ValueError("Group-relative advantages need multiple samples")
        centered = (values - values.mean()) / scale
        for row, value in zip(g["rows"], centered.tolist()):
            result[row["id"]] = value
    return result


def row_weights(groups, shares, arm):
    """Per-token weight of every row; weighted scored tokens sum to exactly one."""
    if arm not in ARMS:
        raise ValueError("Unregistered loss normalization")
    present = sorted({g["family"] for g in groups})
    total_share = sum(shares[f] for f in present)
    if not present or total_share <= 0:
        raise ValueError("No registered family in batch")
    by_family = defaultdict(list)
    for g in groups:
        by_family[g["family"]].append(g)
    weights = {}
    for family, members in by_family.items():
        share = shares[family] / total_share
        family_tokens = sum(r["target_tokens"] for g in members for r in g["rows"])
        for g in members:
            group_tokens = sum(r["target_tokens"] for r in g["rows"])
            if group_tokens <= 0:
                raise ValueError("Empty scored group")
            value = share / family_tokens if arm == "token_mean" else share / (len(members) * group_tokens)
            for r in g["rows"]:
                weights[r["id"]] = value
    return weights


def token_logps(model, rows):
    ids, labels, attention = collate(rows, model.config.pad_token_id, next(model.parameters()).device)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    labels = labels[:, 1:]
    mask = labels != -100
    log_probs = F.log_softmax(logits, -1)
    selected = log_probs.gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return selected, mask, log_probs


def update(model, reference, optimizer, groups, shares, arm, recipe, *, charge=None, step=True):
    """One synchronous on-policy optimizer update; returns metrics.

    FP32 scoring matches the FP32 sampling engine. The first forward of every
    row doubles as the before-update behavior-probability check.
    """
    model.eval()  # No dropout; identical to the behavior policy.
    reference.eval()
    selected = [g for g in groups if g["informative"]]
    if not selected:
        raise ValueError("No informative groups")
    advantage = advantages(selected)
    weights = row_weights(selected, shares, arm)
    rows = [r for g in selected for r in g["rows"]]
    optimizer.zero_grad(set_to_none=True)
    low, high, beta = recipe["epsilon_low"], recipe["epsilon_high"], recipe["kl_beta"]
    metrics = Counter()
    worst = 0.
    micro = recipe["microbatch_examples"]
    for start in range(0, len(rows), micro):
        batch = rows[start:start + micro]
        if charge:
            width = max(len(r["input_ids"]) for r in batch)
            charge(2 * len(batch) * width, "policy_and_reference")
        logp, mask, full = token_logps(model, batch)
        with torch.no_grad():
            _, other, ref = token_logps(reference, batch)
        if not torch.equal(mask, other):
            raise ValueError("Policy/reference mask mismatch")
        old = torch.zeros_like(logp)
        for j, row in enumerate(batch):
            values = torch.as_tensor(row["old_logps"], device=logp.device)
            if len(values) != int(mask[j].sum()) or not torch.isfinite(values).all():
                raise ValueError("Behavior log probabilities/mask differ")
            old[j, mask[j]] = values
        gap = float(((logp.detach() - old).abs() * mask).max())
        worst = max(worst, gap)
        if gap > recipe["behavior_logp_tolerance"]:
            raise BehaviorMismatch(f"Behavior/training log-probability gap {gap:.3g} exceeds tolerance")
        a = torch.tensor([advantage[r["id"]] for r in batch], device=logp.device)[:, None]
        w = torch.tensor([weights[r["id"]] for r in batch], device=logp.device)[:, None]
        ratio = torch.exp(logp - old)
        clipped = ratio.clamp(1 - low, 1 + high)
        policy = -torch.minimum(ratio * a, clipped * a)
        kl = (full.exp() * (full - ref.detach())).sum(-1)
        loss = ((policy + beta * kl) * mask * w).sum()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite policy objective")
        loss.backward()
        metrics["loss"] += float(loss.detach())
        metrics["policy_term"] += float((policy.detach() * mask * w).sum())
        metrics["weighted_kl"] += float((kl.detach() * mask * w).sum())
        metrics["clipped_tokens"] += int((((ratio < 1 - low) | (ratio > 1 + high)) & mask).sum())
        metrics["policy_loss_targets"] += int(mask.sum())
        metrics["policy_processed_tokens"] += sum(r["processed_tokens"] for r in batch)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
    if step:
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite parameter after optimizer step")
    families = Counter(g["family"] for g in selected)
    return {**metrics, "gradient_norm": float(norm), "behavior_logp_max_gap": worst,
            "informative_groups": len(selected), "groups_by_family": dict(families),
            "reward_mean": sum(r["reward"] for r in rows) / len(rows), "arm": arm}
