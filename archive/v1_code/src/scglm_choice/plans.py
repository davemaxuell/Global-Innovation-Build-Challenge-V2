"""Complete immutable, matched-exposure batch plans for all three seeds."""
from collections import Counter, defaultdict
import random

from .common import DOMAINS, digest
from .objective import encode, choice_rows, positions


def training_choices(rows, tokenizer):
    result = []
    for row in rows:
        if row.get("source") not in ("tau/commonsense_qa", "allenai/qasc") or not row.get("choices"):
            continue
        if row["split"] != "train":
            raise ValueError("Held-out choice training item")
        question = row["split_facts"][0]
        item = {"id": row["source"] + ":" + row["provenance"]["source_id"],
                "original_response_id": row["id"], "question": question,
                "source": row["source"], "domain": row["domain"], "group_id": row["group_id"],
                "choices": row["choices"], "answer_index": row["choices"].index(row["answer"]),
                "provenance": row["provenance"], "license": row["license"]}
        try:
            item["encoded_choices"] = choice_rows(tokenizer, item)
            item["anchor"] = {"id": "anchor:" + item["id"], "bucket": "knowledge", "formatting_only": False,
                              **encode(tokenizer, question + "\nAnswer:", " " + row["answer"].strip(), eos=True)}
        except ValueError:
            continue
        result.append(item)
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Duplicate source choice item")
    return result


def select_choices(items, seed, *, count=1024, quota=4):
    selected = []
    for domain in DOMAINS:
        pool = sorted((r for r in items if r["domain"] == domain), key=lambda r: r["id"])
        random.Random(f"{seed}:{domain}").shuffle(pool)
        groups = Counter()
        chosen = []
        for row in pool:
            if groups[row["group_id"]] < quota:
                chosen.append(row)
                groups[row["group_id"]] += 1
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Insufficient group-limited {domain} choice capacity: {len(chosen)}/{count}")
        selected.append(chosen)
    return selected


def make_plan(rows, replay, items, seed, cfg):
    science, commonsense = select_choices(items, seed)
    selected = science + commonsense
    excluded = {r["original_response_id"] for r in selected}
    pools = defaultdict(list)
    order = sorted((r for r in rows if r.get("balanced", True) and r["id"] not in excluded), key=lambda r: r["id"])
    random.Random(seed).shuffle(order)
    for row in order:
        # The held-out presentation is never introduced by the new ordinary CE.
        if "\nThe answer is" not in row["prompt"]:
            pools[row["bucket"]].append(row)
    weights = cfg["data"]["response_bucket_shares"]
    cursors, used, replay_used, replay_cursor = Counter(), Counter(), Counter(), Counter()
    replay_pools = {s: sorted([r for r in replay if r["source"] == s], key=lambda r: r["id"])
                    for s in ("edu", "dclm", "wiki")}
    for s, pool in replay_pools.items():
        if not pool:
            raise ValueError("Missing replay source")
        random.Random(f"{seed}:{s}").shuffle(pool)
    batches, formatted = [], 0
    for step in range(32):
        choices = science[step*32:(step+1)*32] + commonsense[step*32:(step+1)*32]
        anchors = [r["anchor"] for r in choices]
        response = list(anchors)
        anchor_tokens = sum(r["target_tokens"] for r in anchors)
        if anchor_tokens > 8192 * weights["knowledge"]:
            raise ValueError("Gold anchors do not fit the registered knowledge bucket")
        used["knowledge"] += anchor_tokens
        while sum(r["target_tokens"] for r in response) < 8192:
            bucket = max(weights, key=lambda k: (step + 1)*8192*weights[k] - used[k])
            if cursors[bucket] >= len(pools[bucket]):
                raise ValueError("Response pool exhausted under single-exposure requirement")
            row = pools[bucket][cursors[bucket]]
            cursors[bucket] += 1
            if row.get("formatting_only") and formatted + row["target_tokens"] > .05 * (32*8192):
                continue
            response.append(row)
            used[bucket] += row["target_tokens"]
            if row.get("formatting_only"):
                formatted += row["target_tokens"]
        # Stable length sorting reduces padded work while fixing identical arms.
        response.sort(key=lambda r: (len(r["input_ids"]), r["id"]))
        targets = sum(r["target_tokens"] for r in response)
        raw = []
        while sum(r["target_tokens"] for r in raw) < targets * .3 / .7:
            source = min(replay_pools, key=lambda s: replay_used[s] / cfg["data"]["raw_replay_source_shares"][s])
            pool = replay_pools[source]
            r = pool[replay_cursor[source] % len(pool)]
            raw.append(r)
            replay_cursor[source] += 1
            replay_used[source] += r["target_tokens"]
        raw.sort(key=lambda r: (len(r["input_ids"]), r["id"]))
        batches.append({"response_ids": [r["id"] for r in response], "replay_ids": [r["id"] for r in raw],
                        "choice_ids": [r["id"] for r in choices], "response_targets": targets,
                        "replay_targets": sum(r["target_tokens"] for r in raw),
                        "ce_positions": sum(positions(pool[i:i+8]) for pool in (response, raw) for i in range(0, len(pool), 8)),
                        "choice_positions": sum(positions(r["encoded_choices"]) for r in choices)})
    total = sum(used.values())
    if formatted / total > .05 or any(abs(used[k]/total-w) > .002 for k,w in weights.items()):
        raise ValueError("Achieved response mixture outside complete-example tolerance (0.2 points)")
    all_response = [i for b in batches for i in b["response_ids"]]
    if len(set(all_response)) != len(all_response):
        raise ValueError("Repeated response record")
    plan = {"seed": seed, "batches": batches, "response_targets": total,
            "bucket_targets": dict(used), "formatting_targets": formatted, "replay_targets_by_source": dict(replay_used),
            "schedule_denominator": total, "matched_exposure": True,
            "ce_positions": sum(b["ce_positions"] for b in batches),
            "choice_positions": sum(b["choice_positions"] for b in batches)}
    plan["pairing_sha256"] = digest(batches)
    return plan, [r["anchor"] for r in selected]
