"""Deficit-based collection under one frozen policy (a synchronous Sample Mixer adaptation).

Informative groups from families that are already full are kept; later rounds
sample only for the families still short of their declared quota. Groups are
never cached across optimizer updates. A write-ahead ledger bounds collection
attempts and generated tokens across rollbacks; retries are never counted as
optimizer updates.
"""
import math

from scglm_study.budget import BudgetExceeded
from scglm_study.common import read, write_json
from .rollouts import groups as group_rows, sample
from .tasks import allocate


class Ledger:
    """Cumulative collection counters for one arm, never rolled back with model state."""

    FIELDS = ("collection_attempts", "generated_tokens", "prefill_tokens", "samples", "prompts",
              "skipped_updates", "rounds")

    def __init__(self, path, limits):
        self.path, self.limits = path, limits
        self.value = read(path) if path.exists() else dict.fromkeys(self.FIELDS, 0)

    def reserve(self, prompts, sampling):
        worst = prompts * sampling["samples_per_prompt"] * sampling["max_new_tokens"]
        if self.value["collection_attempts"] + 1 > self.limits["max_collection_attempts_per_arm"]:
            raise BudgetExceeded("Collection-attempt ceiling reached")
        if self.value["generated_tokens"] + worst > self.limits["max_generated_tokens_per_arm"]:
            raise BudgetExceeded("Generated-token ceiling would be exceeded")
        # Charged before sampling: an interrupted round still counts as an attempt.
        self.value["collection_attempts"] += 1
        self.save()

    def record(self, stats):
        for key in ("generated_tokens", "prefill_tokens", "samples", "prompts"):
            self.value[key] += stats[key]
        self.value["rounds"] += 1
        self.save()

    def skip(self):
        self.value["skipped_updates"] += 1
        self.save()

    def save(self):
        write_json(self.path, self.value)


def collect(model, tokenizer, sampler, shares, recipe, sampling, *, seed, policy_id, ledger, charge, rates):
    """Return accepted informative groups per family and a per-round record."""
    quotas = allocate(recipe["prompts_per_update"], shares)
    accepted = {f: [] for f in quotas}
    rounds = []
    for number in range(recipe["max_collection_rounds"]):
        deficits = {f: q - len(accepted[f]) for f, q in quotas.items() if q - len(accepted[f]) > 0}
        if not deficits:
            break
        requests = {}
        for family, deficit in sorted(deficits.items()):
            # Oversample from the audited informative rate, bounded per round.
            rate = max(rates.get(family, 0.), recipe["min_assumed_informative_rate"])
            requests[family] = min(math.ceil(deficit / rate), recipe["max_prompts_per_family_round"])
        tasks = [t for family, n in sorted(requests.items()) for t in sampler.take(family, n)]
        ledger.reserve(len(tasks), sampling)
        rows, stats = sample(model, tokenizer, tasks, sampling, seed=seed * 1000 + number, policy_id=policy_id,
                             charge=charge)
        ledger.record(stats)
        found = {f: 0 for f in requests}
        for group in group_rows(rows):
            if group["informative"]:
                found[group["family"]] += 1
                if len(accepted[group["family"]]) < quotas[group["family"]]:
                    accepted[group["family"]].append(group)
        rounds.append({"round": number, "deficits": deficits, "requested_prompts": requests,
                       "informative_found": found, "generated_tokens": stats["generated_tokens"],
                       "prefill_tokens": stats["prefill_tokens"], "elapsed_seconds": stats["elapsed_seconds"]})
    return accepted, quotas, rounds
