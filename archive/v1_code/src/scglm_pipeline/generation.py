"""Generation diagnostics for the retained instruction checkpoint."""
from __future__ import annotations
import hashlib
import time
import torch
from transformers import GenerationConfig
from .task_registry import prompt_text, verify
from .common import digest


def weights_digest(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


@torch.inference_mode()
def sample(model, tokenizer, tasks, config, *, seed, policy_id=None, score_rewards=True):
    model.eval()
    device = next(model.parameters()).device
    count = config.get("samples_per_prompt", 8)
    if count < 1 or not tasks:
        raise ValueError("Empty rollout request")
    maximum = config.get("max_new_tokens", 128)
    resolved = {"max_new_tokens": maximum, "do_sample": config.get("do_sample", True),
                "temperature": config.get("temperature", 1.), "top_p": config.get("top_p", 1.),
                "top_k": config.get("top_k", 0), "repetition_penalty": 1., "num_beams": 1,
                "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id, "use_cache": True}
    if not resolved["do_sample"]:
        for key in ("temperature", "top_p", "top_k"):
            resolved.pop(key)
    gen = GenerationConfig(**resolved)
    requests = [(t, k) for t in tasks for k in range(count)]
    policy_id = policy_id or weights_digest(model)
    rows = []; started = time.monotonic(); prefill = 0
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        for start in range(0, len(requests), config.get("batch_size", 32)):
            chunk = requests[start:start+config.get("batch_size", 32)]
            prefixes = [tokenizer.encode(prompt_text(t), add_special_tokens=False) for t, _ in chunk]
            width = max(map(len, prefixes))
            if min(map(len, prefixes)) < 1 or width + maximum > model.config.max_position_embeddings:
                raise ValueError("Rollout exceeds the fixed context; no silent truncation")
            ids = torch.full((len(chunk), width), tokenizer.pad_token_id, device=device, dtype=torch.long)
            attention = torch.zeros_like(ids)
            for j, prefix in enumerate(prefixes):
                ids[j, -len(prefix):] = torch.tensor(prefix, device=device)
                attention[j, -len(prefix):] = 1
            outputs = model.generate(input_ids=ids, attention_mask=attention, generation_config=gen,
                                     return_dict_in_generate=True, output_scores=True)
            tokens = outputs.sequences[:, width:]
            logps = torch.stack([score.float().log_softmax(-1).gather(1, tokens[:, k:k+1]).squeeze(1)
                                for k, score in enumerate(outputs.scores)], 1)
            for j, (task, k) in enumerate(chunk):
                generated = tokens[j].tolist()
                ended = tokenizer.eos_token_id in generated
                if ended:
                    generated = generated[:generated.index(tokenizer.eos_token_id)+1]
                text_tokens = generated[:-1] if ended else generated
                text = tokenizer.decode(text_tokens, skip_special_tokens=False)
                prefix = prefixes[j]; full = prefix + generated
                row = {"task_id": task["id"], "group_id": task["group_id"], "family": task["family"],
                       "prompt_style": task["prompt_style"], "rollout_group": task["id"], "sample_index": k,
                       "policy_sha256": policy_id, "sampling_seed": seed, "generation": resolved,
                       "response": text, "ended_eos": ended, "input_ids": full,
                       "labels": [-100]*len(prefix)+generated, "prompt_length": len(prefix),
                       "target_tokens": len(generated), "processed_tokens": len(full),
                       "old_logps": logps[j, :len(generated)].tolist()}
                if score_rewards:
                    correct=verify(task,text)
                    if correct is None:raise ValueError("Human prose has no automatic correctness label")
                    row.update(reward=float(bool(correct) and ended),correct=bool(correct),
                               verifier_version=task["verifier_version"])
                row["id"] = digest([policy_id, seed, task["id"], k, full])
                rows.append(row)
                prefill += len(prefix)
    return rows, {"policy_sha256": policy_id, "samples": len(rows), "prefill_tokens": prefill,
                  "generated_tokens": sum(r["target_tokens"] for r in rows),
                  "elapsed_seconds": time.monotonic()-started, "generation": resolved}
