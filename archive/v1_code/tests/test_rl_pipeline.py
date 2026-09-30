"""Revised RL study: loss normalizations, behavior probabilities, deficit collection, audit and recovery."""
import copy
from collections import Counter
from pathlib import Path

import pytest
import torch

from test_posttraining import tokenizer, tiny_model
from scglm.model import save_model
from scglm_study.budget import Budget, BudgetExceeded
from scglm_study.common import ROOT, load_config, read
import scglm_rl.rollouts as rollouts
import scglm_rl.collect as collect_module
from scglm_rl.audit import summarize
from scglm_rl.collect import Ledger, collect
from scglm_rl.loss import BehaviorMismatch, advantages, row_weights, token_logps, update
from scglm_rl.rollouts import groups, outcome, sample
from scglm_rl.tasks import Sampler, allocate

CFG = load_config(ROOT / "configs/rl_revised.json")
SAMPLING = {**CFG["sampling"], "max_new_tokens": 6, "batch_size": 8, "samples_per_prompt": 4}
RECIPE = {**CFG["training"], "microbatch_examples": 3}


def task(i, family="arithmetic"):
    return {"id": f"{family}-{i}", "group_id": f"{family}-g{i}", "family": family, "prompt": f"Hello {i}",
            "answer": "1234", "verifier": "exact", "prompt_style": "raw", "verifier_version": "scglm-tasks-v1",
            "split": "train"}


def synthetic_groups():
    def row(g, k, family, reward, tokens):
        return {"id": f"{g}-{k}", "family": family, "reward": reward, "target_tokens": tokens}
    specs = [("a1", "a", [1, 0, 0, 0], [2, 3, 4, 5]), ("a2", "a", [1, 1, 0, 0], [10, 10, 10, 10]),
             ("b1", "b", [0, 1, 1, 1], [1, 1, 1, 1])]
    return [{"id": g, "family": f, "informative": True, "rows": [row(g, k, f, r, t) for k, (r, t) in enumerate(zip(rs, ts))]}
            for g, f, rs, ts in specs]


@pytest.mark.parametrize("arm", ["token_mean", "prompt_mean"])
def test_weights_sum_to_one_and_follow_the_registered_normalization(arm):
    batch = synthetic_groups()
    shares = {"a": .25, "b": .25, "c": .5}  # c absent: shares renormalize over present families
    w = row_weights(batch, shares, arm)
    total = sum(w[r["id"]] * r["target_tokens"] for g in batch for r in g["rows"])
    assert total == pytest.approx(1.)
    per_group = {g["id"]: sum(w[r["id"]] * r["target_tokens"] for r in g["rows"]) for g in batch}
    per_family = {"a": per_group["a1"] + per_group["a2"], "b": per_group["b1"]}
    assert per_family == pytest.approx({"a": .5, "b": .5})
    if arm == "prompt_mean":
        assert per_group["a1"] == pytest.approx(per_group["a2"])  # every prompt counts equally
    else:
        assert per_group["a2"] == pytest.approx(per_group["a1"] * 40 / 14)  # token-proportional
    with pytest.raises(ValueError):
        row_weights(batch, shares, "sequence_mean")


def test_advantages_are_group_centered_with_one_batch_scale():
    batch = synthetic_groups()
    a = advantages(batch)
    rewards = torch.tensor([r["reward"] for g in batch for r in g["rows"]], dtype=torch.float32)
    scale = float(rewards.std(unbiased=False)) + 1e-4
    assert a["a1-0"] == pytest.approx((1 - .25) / scale)
    for g in batch:
        assert sum(a[r["id"]] for r in g["rows"]) == pytest.approx(0, abs=1e-6)


def test_outcome_requires_eos_and_flags_verifier_disagreement():
    t = task(0)
    assert outcome(t, "1234", True)["reward"] == 1.
    assert outcome(t, "1234", False)["reward"] == 0.
    flagged = outcome(t, "It is 1234", True)
    assert flagged["reward"] == 0. and flagged["verifier_disagreement"]
    with pytest.raises(ValueError):
        outcome({**t, "verifier": "none"}, "x", True)


def test_sampled_behavior_probabilities_match_training_forward(tokenizer):
    torch.manual_seed(0)
    model = tiny_model(tokenizer)
    rows, stats = sample(model, tokenizer, [task(0), task(1)], SAMPLING, seed=3, policy_id="p")
    assert stats["samples"] == 8 and stats["generated_tokens"] == sum(r["target_tokens"] for r in rows)
    logp, mask, _ = token_logps(model, rows)
    for j, row in enumerate(rows):
        assert row["labels"][:row["prompt_length"]] == [-100] * row["prompt_length"]
        assert int(mask[j].sum()) == row["target_tokens"] == len(row["old_logps"])
        assert torch.allclose(logp[j][mask[j]], torch.tensor(row["old_logps"]), atol=1e-4)
        assert row["truncated"] == (tokenizer.eos_token_id not in row["input_ids"][row["prompt_length"]:])
    again, _ = sample(model, tokenizer, [task(0), task(1)], SAMPLING, seed=3, policy_id="p")
    assert [r["input_ids"] for r in again] == [r["input_ids"] for r in rows]
    with pytest.raises(ValueError):
        sample(model, tokenizer, [task(0)], {**SAMPLING, "top_p": .9}, seed=3, policy_id="p")


def _batch(model, tokenizer):
    rows, _ = sample(model, tokenizer, [task(0), task(1, "logic"), task(2)], SAMPLING, seed=5, policy_id="p")
    for i, r in enumerate(rows):
        r["reward"] = float(i % 3 == 0)
    batch = groups(rows)
    for g in batch:
        g["informative"] = len({r["reward"] for r in g["rows"]}) > 1
    return batch


@pytest.mark.parametrize("arm", ["token_mean", "prompt_mean"])
def test_update_gradient_matches_explicit_per_token_oracle(tokenizer, arm):
    torch.manual_seed(0)
    model, reference = tiny_model(tokenizer), tiny_model(tokenizer)
    with torch.no_grad():
        for p in reference.parameters():
            p.add_(0.01 * torch.randn_like(p))
    batch = _batch(model, tokenizer)
    shares = {"arithmetic": .5, "logic": .5}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    metrics = update(model, reference, optimizer, batch, shares, arm, {**RECIPE, "gradient_clip": 1e9}, step=False)
    got = [p.grad.clone() for p in model.parameters()]
    assert metrics["behavior_logp_max_gap"] < 1e-4
    # Oracle: one row at a time, explicit sums, same weights definition.
    model.zero_grad(set_to_none=True)
    selected = [g for g in batch if g["informative"]]
    adv, weights = advantages(selected), row_weights(selected, shares, arm)
    for g in selected:
        for r in g["rows"]:
            logp, mask, full = token_logps(model, [r])
            with torch.no_grad():
                _, _, ref = token_logps(reference, [r])
            old = torch.tensor(r["old_logps"])
            lp = logp[0][mask[0]]
            ratio = torch.exp(lp - old)
            a = adv[r["id"]]
            policy = -torch.minimum(ratio * a, ratio.clamp(.8, 1.2) * a)
            kl = (full[0].exp() * (full[0] - ref[0])).sum(-1)[mask[0]]
            (weights[r["id"]] * (policy + RECIPE["kl_beta"] * kl).sum()).backward()
    for a, b in zip(got, [p.grad for p in model.parameters()]):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)


def test_behavior_mismatch_is_rejected_before_any_step(tokenizer):
    torch.manual_seed(0)
    model, reference = tiny_model(tokenizer), tiny_model(tokenizer)
    batch = _batch(model, tokenizer)
    target = next(g for g in batch if g["informative"])["rows"][0]
    target["old_logps"] = [v - 0.1 for v in target["old_logps"]]
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(BehaviorMismatch):
        update(model, reference, optimizer, batch, {"arithmetic": .5, "logic": .5}, "prompt_mean", RECIPE)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_allocation_and_sampler_are_deterministic_with_saved_cursors():
    assert allocate(32, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == {"a": 11, "b": 11, "c": 10}
    assert sum(allocate(7, {"a": .5, "b": .25, "c": .25}).values()) == 7
    pool = {"a": [task(i, "a") for i in range(5)]}
    counters = Counter()
    first = Sampler(pool, 1, counters).take("a", 7)
    assert counters["cursor:a"] == 7 and counters["wrapped:a"] == 1
    replay = Sampler(pool, 1, Counter()).take("a", 7)
    assert [t["id"] for t in first] == [t["id"] for t in replay]
    resumed = Sampler(pool, 1, Counter({"cursor:a": 3})).take("a", 4)
    assert [t["id"] for t in resumed] == [t["id"] for t in first[3:]]


def fake_sample(model, tokenizer, tasks, sampling, *, seed, policy_id, charge=None):
    """Family 'mixed' always yields mixed groups, 'flat' never does."""
    rows = []
    for t in tasks:
        for k in range(sampling["samples_per_prompt"]):
            reward = float(k == 0) if t["family"] == "mixed" else 0.
            rows.append({"id": f"{t['id']}-{k}-{seed}", "rollout_group": t["id"], "family": t["family"],
                         "reward": reward, "target_tokens": 2})
    return rows, {"generated_tokens": 2 * len(rows), "prefill_tokens": 5 * len(rows), "samples": len(rows),
                  "prompts": len(tasks), "elapsed_seconds": 0.}


def test_deficit_collection_refills_only_short_families_and_respects_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_module, "sample", fake_sample)
    pool = {f: [task(i, f) for i in range(500)] for f in ("mixed", "flat")}
    counters = Counter()
    sampler = Sampler(pool, 1, counters)
    limits = {"max_collection_attempts_per_arm": 100, "max_generated_tokens_per_arm": 10 ** 7}
    ledger = Ledger(tmp_path / "ledger.json", limits)
    recipe = {**RECIPE, "prompts_per_update": 8, "max_collection_rounds": 3, "min_assumed_informative_rate": .5,
              "max_prompts_per_family_round": 10}
    accepted, quotas, rounds = collect(None, None, sampler, {"mixed": .5, "flat": .5}, recipe, SAMPLING, seed=1,
                                       policy_id="p", ledger=ledger, charge=None, rates={"mixed": 1., "flat": 0.})
    assert quotas == {"mixed": 4, "flat": 4}
    assert len(accepted["mixed"]) == 4 and accepted["flat"] == []
    # The ready family is not resampled after round 0.
    assert rounds[0]["requested_prompts"] == {"flat": 8, "mixed": 4}
    assert all("mixed" not in r["requested_prompts"] for r in rounds[1:]) and len(rounds) == 3
    assert read(tmp_path / "ledger.json")["collection_attempts"] == 3
    assert counters["cursor:mixed"] == 4 and counters["cursor:flat"] == 24
    tight = Ledger(tmp_path / "tight.json", {"max_collection_attempts_per_arm": 1, "max_generated_tokens_per_arm": 10 ** 7})
    tight.reserve(1, SAMPLING)
    with pytest.raises(BudgetExceeded):
        tight.reserve(1, SAMPLING)
    small = Ledger(tmp_path / "small.json", {"max_collection_attempts_per_arm": 9, "max_generated_tokens_per_arm": 10})
    with pytest.raises(BudgetExceeded):
        small.reserve(1, SAMPLING)


def test_audit_readiness_requires_mixed_families_and_a_knowledge_family():
    rows = []
    for family, pattern in (("science", [1, 0]), ("logic", [1, 0]), ("sorting", [1, 1]), ("arithmetic", [0, 1])):
        for g in range(4):
            for k, reward in enumerate(pattern * 2):
                rows.append({"rollout_group": f"{family}{g}", "family": family, "reward": float(reward),
                             "correct": bool(reward), "ended_eos": True, "truncated": False, "verifier_disagreement": False,
                             "target_tokens": 2, "prompt_length": 10})
    contract = {f: {"reference_exceeds_contract_fraction": 0.} for f in ("science", "logic", "sorting", "arithmetic")}
    gates = CFG["audit"]["gates"]
    result = summarize(rows, contract, gates)
    assert result["ready"] and result["eligible_families"] == ["arithmetic", "logic", "science"]
    assert result["family_shares"] == pytest.approx({f: 1 / 3 for f in ("arithmetic", "logic", "science")})
    assert result["families"]["sorting"]["reasons"] == ["too_few_mixed_groups"]
    no_knowledge = [r for r in rows if r["family"] != "science"]
    assert not summarize(no_knowledge, contract, gates)["ready"]


def _study(tmp_path, tokenizer, updates=3):
    parent = tmp_path / "parent"
    torch.manual_seed(0)
    model = tiny_model(tokenizer)
    save_model(model, parent)
    tokenizer.save_pretrained(str(parent))
    parameters = sum(p.numel() for p in model.parameters())
    cfg = copy.deepcopy(CFG)
    cfg["parent"] = {**cfg["parent"], "path": str(parent), "parameters": parameters}
    cfg["isolation"] = {**cfg["isolation"], "data_root": str(tmp_path / "data"), "output_root": str(tmp_path / "out")}
    cfg["sampling"] = SAMPLING
    cfg["training"] = {**RECIPE, "updates": updates, "prompts_per_update": 2, "min_families_per_update": 1,
                       "checkpoint_every": 100, "learning_rate": 1e-3}
    cfg["evaluation"] = {**cfg["evaluation"], "development_checks_at_updates": [2]}
    audit = {"family_shares": {"arithmetic": .5, "logic": .5}, "informative_rates": {"arithmetic": .5, "logic": .5}}
    pool = {f: [task(i, f) for i in range(50)] for f in ("arithmetic", "logic")}
    return cfg, audit, pool


def _run(cfg, audit, pool, tmp_path, *, stop_after=None, name="run"):
    from scglm_rl.train import ArmTrainer
    budget = Budget(tmp_path / f"budget_{name}", {"gpu_seconds": {"default": 10 ** 6}, "overall_gpu_seconds": 10 ** 6})
    evaluated = []
    with budget.session("token_mean", timer=False):
        trainer = ArmTrainer(cfg, "prompt_mean", audit, pool, budget, lambda path, t: evaluated.append(t.state.step),
                             device="cpu", directory=tmp_path / name)
        if stop_after is not None:
            original = trainer.step_once

            def step_once():
                result = original()
                if trainer.state.step >= stop_after:
                    trainer.stop_requested = True
                return result
            trainer.step_once = step_once
        try:
            path = trainer.run()
        except InterruptedError:
            path = None
    return trainer, path, evaluated


def test_arm_recovery_resumes_exactly_after_clean_interruption(tmp_path, tokenizer, monkeypatch):
    # Rewards from the text keep groups mixed for a random tiny policy.
    monkeypatch.setattr(rollouts, "outcome", lambda t, text, ended: {
        "reward": float(len(text) % 2 == 0), "correct": False, "ended_eos": ended, "verifier_disagreement": False})
    cfg, audit, pool = _study(tmp_path, tokenizer)
    straight, path, evaluated = _run(cfg, audit, pool, tmp_path, name="straight")
    assert straight.state.step == 3 and evaluated == [2] and path.name == "export"
    assert read(tmp_path / "straight" / "status.json")["status"] == "completed"
    paused, none, _ = _run(cfg, audit, pool, tmp_path, stop_after=1, name="resumed")
    assert none is None and paused.state.step == 1
    assert read(tmp_path / "resumed" / "status.json")["status"] == "paused"
    resumed, path2, evaluated2 = _run(cfg, audit, pool, tmp_path, name="resumed")
    assert resumed.state.step == 3 and evaluated2 == [2]
    for (k, a), b in zip(straight.model.state_dict().items(), resumed.model.state_dict().values()):
        assert torch.equal(a, b), k
    ledger = read(tmp_path / "resumed" / "ledger.json")
    assert ledger["collection_attempts"] >= read(tmp_path / "straight" / "ledger.json")["collection_attempts"]
    # A completed arm returns its export without training again.
    again, path3, _ = _run(cfg, audit, pool, tmp_path, name="straight")
    assert again.state.step == 3 and path3 == path
