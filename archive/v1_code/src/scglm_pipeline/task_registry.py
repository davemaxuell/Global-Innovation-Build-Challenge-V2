"""Versioned post-training tasks; one answer contract, protected split identities."""
from __future__ import annotations
import json
import re
from scglm_post.common import encode_example, render_prompt, normalized, identity
from scglm_post.verifiers import exact_integer_object, integer_list
from .common import digest

VERSION = "scglm-tasks-v1"
PRIMARY_DOMAINS = ("science", "commonsense", "grounded")
BUCKETS = ("general", "knowledge", "grounded", "procedural")


def task(*, source, group, split, family, prompt, answer, verifier="exact", **fields):
    if split not in ("train", "development", "confirmation"):
        raise ValueError("Unknown protected split")
    if not all(isinstance(v, str) and v.strip() for v in (source, group, family, prompt, answer)):
        raise ValueError("Empty task field")
    result = {"schema": VERSION, "source": source, "group_id": group, "split": split,
              "family": family, "prompt": prompt, "answer": answer, "verifier": verifier,
              "prompt_style": "chat", "difficulty": 0, "license": "project-generated",
              "bucket": "procedural", "verifier_version": VERSION, **fields}
    result["id"] = task_identity(result)
    return result


def task_identity(row):
    return digest([row["source"],row["group_id"],row["split"],row["family"],prompt_text(row),row["answer"],row["prompt_style"]])


def prompt_text(row):
    if row["prompt_style"] == "raw":
        return row.get("raw_prompt", row["prompt"].rstrip() + "\nAnswer:\n")
    return render_prompt(row.get("messages", [{"role": "user", "content": row["prompt"]}]))


def encode_response(row, tokenizer, answer=None, *, eos=True, max_length=1024, max_response=128):
    answer = row["answer"] if answer is None else answer
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Empty answer")
    # A model may emit these strings; they never become an injected EOS in SFT.
    if re.search(r"<\|(?:pad|bos|eos|unk)\|>", answer):
        raise ValueError("Reserved token in answer text")
    prompt = prompt_text(row)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
    if not prefix or ids[:len(prefix)] != prefix:
        raise ValueError("Tokenizer crosses the response boundary")
    if eos:
        ids.append(tokenizer.eos_token_id)
    targets = len(ids) - len(prefix)
    if targets < 1 or targets > max_response or len(ids) > max_length:
        raise ValueError("Complete response outside context limits")
    return {"input_ids": ids, "labels": [-100] * len(prefix) + ids[len(prefix):],
            "prompt_length": len(prefix), "target_tokens": targets, "processed_tokens": len(ids)}


def raw_row(tokens):
    if len(tokens) < 2:
        raise ValueError("Replay requires a context and target")
    return {"input_ids": list(tokens), "labels": [-100] + list(tokens[1:]),
            "prompt_length": 1, "target_tokens": len(tokens)-1, "processed_tokens": len(tokens)}


def verify(row, response):
    """No substring answer matching, external judge, or format-only positive reward."""
    if not isinstance(response, str) or not response.strip():
        return False
    text = response.strip()
    if "<|" in text:
        return False
    kind = row["verifier"]
    try:
        if kind == "none":
            return None  # Human demonstrations without a deterministic correctness oracle.
        if kind == "legacy":
            from scglm_post.procedural import correct
            return correct(row["legacy"], text)
        if kind == "integer":
            return bool(re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text)) and int(text) == int(row["answer"])
        if kind == "integer_list":
            return integer_list(text) == integer_list(row["answer"])
        if kind == "json_integer":
            return exact_integer_object(text, json.loads(row["answer"]))
        if kind == "derivation":
            # Complete independently stored program trace, including the final answer.
            return text == row["answer"]
        if kind == "exact":
            return normalized(text) == normalized(row["answer"])
    except (ValueError, TypeError, RecursionError):
        return False
    raise ValueError("Unknown verifier: " + str(kind))


def audit_tasks(rows_by_split):
    groups, prompts, identities = {}, {}, set()
    counts = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            if row["schema"] != VERSION or row["split"] != split or row["id"] in identities or row["id"]!=task_identity(row):
                raise ValueError("Invalid task identity/split/schema")
            identities.add(row["id"])
            for index, key in ((groups, row["group_id"]), (prompts, identity(prompt_text(row)))):
                if key in index and index[key] != split:
                    raise ValueError("Protected group/prompt crossed splits")
                index[key] = split
            if row["verifier"] != "none":
                if verify(row, row["answer"]) is not True or verify(row, row["answer"] + " unrelated claim") is not False:
                    raise ValueError("Verifier reference/adversarial check failed")
            if row.get("choices"):
                choices = row["choices"]
                if (len(choices) != len(set(map(normalized, choices))) or row["answer"] not in choices
                        or any(verify(row, c) for c in choices if c != row["answer"])):
                    raise ValueError("Ambiguous choice labels")
        counts[split] = {"examples": len(rows), "groups": len({r["group_id"] for r in rows})}
    return counts
