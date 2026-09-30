"""Exact sampled trajectories with behavior log probabilities and outcome rewards.

Full-softmax sampling (temperature 1, top_p 1, top_k 0) means the recorded
scores are the unmodified policy distribution. Enabling top-p/top-k would
require replaying the truncated candidate sets in the loss (MiMo p. 32).
"""
from __future__ import annotations
import hashlib
import time

import torch
from transformers import GenerationConfig

from scglm_post.common import normalized
from scglm_pipeline.common import digest
from scglm_pipeline.task_registry import prompt_text, verify


def weights_digest(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def outcome(task, response, ended_eos):
    correct = verify(task, response)
    if correct is None:
        raise ValueError("Unverifiable task cannot receive an automated reward")
    gold = normalized(task["answer"])
    return {"reward": float(bool(correct) and ended_eos), "correct": bool(correct), "ended_eos": bool(ended_eos),
            # Diagnostic only: the strict verifier rejected text that still contains the gold answer.
            "verifier_disagreement": bool(not correct and gold and gold in normalized(response)),
            "verifier_version": task["verifier_version"]}


def generation_config(sampling, tokenizer):
    if sampling["temperature"] != 1.0 or sampling["top_p"] != 1.0 or sampling["top_k"] != 0:
        raise ValueError("Only full-softmax sampling matches the unmodified behavior probabilities")
    return GenerationConfig(max_new_tokens=sampling["max_new_tokens"], do_sample=True, temperature=1.0, top_p=1.0,
                            top_k=0, repetition_penalty=1.0, num_beams=1, pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, use_cache=True)


@torch.inference_mode()
def sample(model, tokenizer, tasks, sampling, *, seed, policy_id, charge=None):
    model.eval()
    device = next(model.parameters()).device
    count, maximum = sampling["samples_per_prompt"], sampling["max_new_tokens"]
    if count < 2 or not tasks:
        raise ValueError("Group-relative rewards need at least two samples per prompt")
    config = generation_config(sampling, tokenizer)
    requests = [(t, k) for t in tasks for k in range(count)]
    rows, started, prefill = [], time.monotonic(), 0
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for start in range(0, len(requests), sampling["batch_size"]):
            chunk = requests[start:start + sampling["batch_size"]]
            prefixes = [tokenizer.encode(prompt_text(t), add_special_tokens=False) for t, _ in chunk]
            width = max(map(len, prefixes))
            if min(map(len, prefixes)) < 1 or width + maximum > model.config.max_position_embeddings:
                raise ValueError("Rollout exceeds the fixed context; no silent truncation")
            if charge:
                # Padded prefill plus one position per generated step, charged before the work.
                charge(len(chunk) * (width + maximum), "rollout")
            ids = torch.full((len(chunk), width), tokenizer.pad_token_id, device=device, dtype=torch.long)
            attention = torch.zeros_like(ids)
            for j, prefix in enumerate(prefixes):
                ids[j, -len(prefix):] = torch.tensor(prefix, device=device)
                attention[j, -len(prefix):] = 1
            outputs = model.generate(input_ids=ids, attention_mask=attention, generation_config=config,
                                     return_dict_in_generate=True, output_scores=True)
            tokens = outputs.sequences[:, width:]
            logps = torch.stack([score.float().log_softmax(-1).gather(1, tokens[:, k:k + 1]).squeeze(1)
                                 for k, score in enumerate(outputs.scores)], 1)
            for j, (task, k) in enumerate(chunk):
                generated = tokens[j].tolist()
                ended = tokenizer.eos_token_id in generated
                if ended:
                    generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(generated[:-1] if ended else generated, skip_special_tokens=False)
                prefix = prefixes[j]
                full = prefix + generated
                row = {"task_id": task["id"], "group_id": task["group_id"], "family": task["family"],
                       "rollout_group": task["id"], "sample_index": k, "policy_sha256": policy_id,
                       "sampling_seed": seed, "response": text, "input_ids": full,
                       "labels": [-100] * len(prefix) + generated, "prompt_length": len(prefix),
                       "target_tokens": len(generated), "processed_tokens": len(full),
                       "truncated": not ended, "old_logps": logps[j, :len(generated)].tolist(),
                       **outcome(task, text, ended)}
                row["id"] = digest([policy_id, seed, task["id"], k, full])
                rows.append(row)
                prefill += len(prefix)
    return rows, {"policy_sha256": policy_id, "samples": len(rows), "prompts": len(tasks), "prefill_tokens": prefill,
                  "generated_tokens": sum(r["target_tokens"] for r in rows),
                  "elapsed_seconds": time.monotonic() - started}


def groups(rows):
    by = {}
    for row in rows:
        by.setdefault(row["rollout_group"], []).append(row)
    return [{"id": key, "family": members[0]["family"], "rows": members,
             "informative": len({r["reward"] for r in members}) > 1} for key, members in by.items()]
