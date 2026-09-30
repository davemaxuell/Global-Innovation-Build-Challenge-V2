"""Training-only verifiable task pool, answer-length contract and per-family prompt sampler."""
from collections import defaultdict
import hashlib
import math

from scglm_pipeline.task_registry import prompt_text, verify
from scglm_study.common import ROOT, read, read_jsonl, sha256


def load_pool(cfg):
    spec = cfg["tasks"]
    directory = ROOT / spec["data"]
    if sha256(directory / "manifest.json") != spec["manifest_sha256"]:
        raise ValueError("Task manifest changed")
    entry = read(directory / "manifest.json")["files"]["train"]
    if sha256(directory / entry["file"]) != entry["sha256"]:
        raise ValueError("Training task file changed")
    families = set(spec["families"])
    pool = defaultdict(list)
    for row in read_jsonl(directory / entry["file"]):
        if row["family"] not in families or row["split"] != "train" or row.get("control") or row["verifier"] == "none":
            continue
        if row.get("formatting_only"):
            continue
        # Reference and adversarial checks, as in the retained task audit.
        if verify(row, row["answer"]) is not True or verify(row, row["answer"] + " unrelated claim") is not False:
            raise ValueError(f"Verifier reference check failed: {row['id']}")
        pool[row["family"]].append(row)
    if set(pool) != families:
        raise ValueError(f"Missing registered families: {sorted(families - set(pool))}")
    for rows in pool.values():
        rows.sort(key=lambda r: r["id"])
    return dict(pool)


def answer_tokens(tokenizer, row):
    """Generated tokens needed for the reference answer plus EOS under the actual prompt."""
    prompt = prompt_text(row)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + row["answer"], add_special_tokens=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("Tokenizer crosses the prompt/answer boundary")
    return len(full) - len(prefix) + 1, len(prefix)


def length_contract(pool, tokenizer, max_new_tokens, context):
    result = {}
    for family, rows in pool.items():
        over, prompt_max, answer_max = 0, 0, 0
        for row in rows:
            answer, prompt = answer_tokens(tokenizer, row)
            over += answer > max_new_tokens or prompt + max_new_tokens > context
            prompt_max, answer_max = max(prompt_max, prompt), max(answer_max, answer)
        result[family] = {"tasks": len(rows), "reference_exceeds_contract_fraction": over / len(rows),
                          "max_prompt_tokens": prompt_max, "max_reference_answer_tokens_with_eos": answer_max}
    return result


def fits(tokenizer, row, max_new_tokens, context):
    answer, prompt = answer_tokens(tokenizer, row)
    return answer <= max_new_tokens and prompt + max_new_tokens <= context


class Sampler:
    """Seeded per-family permutations with explicit cursors saved in recovery counters."""

    def __init__(self, pool, seed, counters):
        self.pool, self.seed, self.counters = pool, seed, counters
        self.orders = {f: sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{seed}:{f}:{rows[i]['id']}".encode()).hexdigest())
                       for f, rows in pool.items()}

    def take(self, family, count):
        rows, order = self.pool[family], self.orders[family]
        result = []
        for _ in range(count):
            key = f"cursor:{family}"
            cursor = self.counters[key]
            if cursor and cursor % len(order) == 0:
                self.counters[f"wrapped:{family}"] += 1
            result.append(rows[order[cursor % len(order)]])
            self.counters[key] = cursor + 1
        return result


def allocate(total, shares):
    """Largest-remainder integer quotas that preserve declared family shares."""
    raw = {f: total * s for f, s in shares.items()}
    quotas = {f: math.floor(v) for f, v in raw.items()}
    for f in sorted(raw, key=lambda f: (-(raw[f] - quotas[f]), f))[:total - sum(quotas.values())]:
        quotas[f] += 1
    return quotas
