"""Likelihood and generation primitives used by selected-model evaluation."""
import torch
import torch.nn.functional as F
from scglm.evaluate import fixed_mixture_metrics
from scglm_post.common import encode_example
from scglm_post.sft import collate


@torch.inference_mode()
def generate_many(model, tokenizer, prompts, *, max_new_tokens=256, batch_size=16):
    """Left padding for causal generation; no prompt truncation, fixed greedy decode."""
    answers = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        encoded = [tokenizer.encode(p, add_special_tokens=False) for p in prompts[start:start+batch_size]]
        width = max(map(len, encoded))
        allowance = min(max_new_tokens, model.config.max_position_embeddings-width)
        if not encoded or min(map(len, encoded)) == 0 or allowance <= 0:
            raise ValueError("Empty prompt or insufficient generation context")
        ids = torch.full((len(encoded), width), tokenizer.pad_token_id, device=device, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for i, values in enumerate(encoded):
            ids[i, -len(values):] = torch.tensor(values, device=device)
            attention[i, -len(values):] = 1
        outputs = model.generate(input_ids=ids, attention_mask=attention, do_sample=False,
            max_new_tokens=allowance, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True)
        for output in outputs[:, width:].tolist():
            if tokenizer.eos_token_id in output:
                output = output[:output.index(tokenizer.eos_token_id)+1]
            answers.append({"response": tokenizer.decode(output, skip_special_tokens=True),
                            "generated_tokens": len(output), "ended_eos": bool(output and output[-1] == tokenizer.eos_token_id)})
    return answers


def response_logps(model, rows):
    """Unnormalized sequence log probabilities, final-response mask, exactly one shift."""
    ids, labels, attention = collate(rows, model.config.pad_token_id, next(model.parameters()).device)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != -100
    safe = targets.masked_fill(~mask, 0)
    logps = F.log_softmax(logits, -1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(logps).all() or not mask.sum(1).gt(0).all():
        raise ValueError("Invalid response log probability")
    return (logps * mask).sum(1), mask.sum(1)


@torch.inference_mode()
def choice_scores(model, tokenizer, row):
    choices = []
    for answer in row["choices"]:
        encoded = encode_example([{"role": "user", "content": row["prompt"]},
                                  {"role": "assistant", "content": answer}], tokenizer)
        # Score the literal answer, not its letter label or EOS preference.
        for key in ("input_ids", "labels"):
            encoded[key] = encoded[key][:-1]
        encoded["target_tokens"] -= 1
        encoded["processed_tokens"] -= 1
        choices.append(encoded)
    scores, lengths = response_logps(model, choices)
    normalized = scores / lengths
    return {"log_likelihoods": scores.tolist(), "target_lengths": lengths.tolist(),
            "predicted": int(scores.argmax()), "predicted_length_normalized": int(normalized.argmax()),
            "correct": int(scores.argmax()) == row["answer_index"],
            "correct_length_normalized": int(normalized.argmax()) == row["answer_index"]}


def raw_metrics(records, config):
    groups, sources = {}, {}
    for name, spec in config["corpora"].items():
        rows = [{**r, "source": r["source"].split("/", 1)[1]} for r in records if r["source"].startswith(name+"/")]
        groups[name] = fixed_mixture_metrics(rows, spec["weights"])
        sources.update({name+"/"+key: value for key, value in groups[name]["sources"].items()})
    return {"mixtures": groups, "sources": sources}
