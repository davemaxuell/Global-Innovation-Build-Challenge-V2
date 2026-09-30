"""No-update rollout audit: does the parent produce useful group-relative signal?

Records semantic correctness, termination, truncation, all-correct/all-wrong/
mixed groups and verifier disagreements per family. Families qualify only if
enough prompt groups are mixed and the reference answers fit the generation
contract. The qualifying families and their informative rates are frozen for
both training arms.
"""
from collections import defaultdict
import hashlib

import numpy as np

from .rollouts import groups as group_rows, sample
from .tasks import length_contract


def audit_tasks(pool, spec, seed, tokenizer, sampling, context):
    from .tasks import fits
    per_family = spec["prompts"] // len(pool)
    chosen = []
    for family, rows in sorted(pool.items()):
        eligible = [r for r in rows if fits(tokenizer, r, sampling["max_new_tokens"], context)]
        eligible.sort(key=lambda r: hashlib.sha256(f"audit:{seed}:{r['id']}".encode()).hexdigest())
        chosen.extend(eligible[:per_family])
    return chosen


def summarize(rows, contract, gates):
    families = defaultdict(lambda: {"groups": 0, "all_correct": 0, "all_wrong": 0, "mixed": 0, "samples": 0,
                                    "reward": 0., "answer_correct": 0, "ended_eos": 0, "truncated": 0,
                                    "verifier_disagreements": 0, "generated_tokens": 0, "prompt_tokens": 0})
    for group in group_rows(rows):
        f = families[group["family"]]
        rewards = [r["reward"] for r in group["rows"]]
        f["groups"] += 1
        f["all_correct"] += min(rewards) == 1.
        f["all_wrong"] += max(rewards) == 0.
        f["mixed"] += group["informative"]
        for r in group["rows"]:
            f["samples"] += 1
            f["reward"] += r["reward"]
            f["answer_correct"] += r["correct"]
            f["ended_eos"] += r["ended_eos"]
            f["truncated"] += r["truncated"]
            f["verifier_disagreements"] += r["verifier_disagreement"]
            f["generated_tokens"] += r["target_tokens"]
            f["prompt_tokens"] += r["prompt_length"]
    result = {}
    for family, f in sorted(families.items()):
        n = f["samples"]
        result[family] = {**f, "mixed_fraction": f["mixed"] / f["groups"], "sample_accuracy": f["reward"] / n,
                          "semantic_accuracy": f["answer_correct"] / n, "termination_rate": f["ended_eos"] / n,
                          "truncation_rate": f["truncated"] / n, "verifier_disagreement_rate": f["verifier_disagreements"] / n,
                          "length_contract": contract[family]}
        reasons = []
        if result[family]["mixed_fraction"] < gates["min_mixed_fraction"]:
            reasons.append("too_few_mixed_groups")
        if contract[family]["reference_exceeds_contract_fraction"] > gates["max_reference_exceeding_contract"]:
            reasons.append("reference_answers_exceed_contract")
        result[family]["eligible"] = not reasons
        result[family]["reasons"] = reasons
    eligible = sorted(f for f, v in result.items() if v["eligible"])
    ready = len(eligible) >= gates["min_ready_families"] and all(set(eligible) & set(any_of) for any_of in gates["required_any"])
    shares = {f: 1 / len(eligible) for f in eligible} if ready else {}
    rates = {f: result[f]["mixed_fraction"] for f in eligible}
    return {"ready": bool(ready), "eligible_families": eligible, "family_shares": shares, "informative_rates": rates,
            "families": result, "reason": "ready" if ready else "insufficient informative coverage",
            "totals": {"samples": len(rows), "generated_tokens": sum(r["target_tokens"] for r in rows),
                       "prefill_tokens": sum(r["prompt_length"] for r in rows),
                       "mean_reward": float(np.mean([r["reward"] for r in rows]))}}


def run_audit(model, tokenizer, pool, cfg, *, policy_id, charge):
    spec, sampling = cfg["audit"], cfg["sampling"]
    context = model.config.max_position_embeddings
    contract = length_contract(pool, tokenizer, sampling["max_new_tokens"], context)
    tasks = audit_tasks(pool, spec, spec["seed"], tokenizer, sampling, context)
    rows, stats = sample(model, tokenizer, tasks, sampling, seed=spec["seed"], policy_id=policy_id, charge=charge)
    return summarize(rows, contract, spec["gates"]), rows, stats
