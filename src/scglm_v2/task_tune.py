"""Task tuning of V2 on the four benchmarks' TRAIN splits (phase v2/phases/task_tune).

``prepare`` (CPU): reloads the pinned train splits, re-derives V2's existing 80/20
development split with ``benchmarks.is_development`` (the held-out 20% stays untouched
and becomes the selection panel), formats every item exactly as the pinned harness scores
it (``task_format``), drops any training item whose context matches an official
evaluation item (normalized exact match or a shared 13-word sequence), and tokenizes.
With ``include_development: true`` (the final 100% retrain) the held-out items also go
into train.jsonl, after the same official-overlap check; development.jsonl is still
written for reference but is then no longer held out.

``train`` (GPU): GPT-1-style multiple-choice fine-tuning adapted to log-likelihood
scoring. Each candidate's score is its summed continuation log-likelihood (the quantity
the harness's ``acc`` compares), so no parameters are added. Per update:

  loss = (1 - r) * (choice_ce + lm_weight * gold_lm) + r * replay_lm

choice_ce: softmax cross-entropy over the candidates' scores; gold_lm: next-token loss
over the gold sequence (GPT-1's auxiliary objective, weight 0.5); replay_lm: next-token
loss on V2 pretraining text (weight r = 0.3, as in V1's retained SFT recipe). LR warms up
over a fraction of updates, then decays linearly to zero at the end of the last epoch.
An export is written at the end of every epoch; earlier exports are prefixes of the
full schedule.

Run: PYTHONPATH=src python -m scglm_v2.task_tune prepare --config v2/phases/task_tune/config.json
     PYTHONPATH=src python -m scglm_v2.task_tune train --config v2/phases/task_tune/config.json --arm <name>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import shutil
import time

from .benchmarks import PINNED, format_example, is_development
from .common import ROOT, event, now, sha256_file, write_json
from .task_format import encode_pair, harness_item, model_input

TRAIN_TASKS = {  # task -> (repo, config, official evaluation split used only for decontamination)
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", None),
    "piqa": ("ybisk/piqa", None, "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
}
OFFICIAL_FOR = {"arc_challenge": "arc_easy"}  # ARC-Challenge train is checked against ARC-Easy test


def _load(repo: str, config: str | None, split: str, piqa_files: dict):
    from datasets import load_dataset
    if repo == "ybisk/piqa":
        spec = piqa_files[split]
        if sha256_file(Path(spec["path"])) != spec["sha256"]:
            raise ValueError(f"PIQA {split} parquet hash mismatch")
        return load_dataset("parquet", data_files={split: [spec["path"]]}, split=split)
    return load_dataset(repo, config, revision=PINNED[repo], split=split)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _context_text(task: str, doc: dict) -> str:
    if task == "hellaswag":
        return doc["ctx"]
    if task in ("arc_easy", "arc_challenge"):
        return doc["question"]
    if task == "piqa":
        return doc["goal"]
    return doc["sentence"]


def prepare(cfg: dict) -> dict:
    from transformers import PreTrainedTokenizerFast
    out = ROOT / cfg["data_dir"]
    if (out / "report.json").exists():
        raise FileExistsError(f"{out} is already prepared; use a new data_dir")
    out.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast.from_pretrained(str(ROOT / cfg["parent_export"]), local_files_only=True)
    expected = json.loads((ROOT / cfg["benchmark_split_report"]).read_text())
    official_keys, official_grams = {}, {}
    for task, (repo, config, eval_split) in TRAIN_TASKS.items():
        if eval_split is None:
            continue
        keys, grams = set(), set()
        for doc in _load(repo, config, eval_split, cfg["piqa_files"]):
            words = _words(_context_text(task, doc))
            keys.add(" ".join(words))
            grams.update(tuple(words[i:i + 13]) for i in range(len(words) - 12))
        official_keys[task], official_grams[task] = keys, grams
    report = {"created": now(), "tasks": {}, "tokenizer_sha256": sha256_file(ROOT / cfg["parent_export"] / "tokenizer.json")}
    with (out / "train.jsonl").open("w") as train_f, (out / "development.jsonl").open("w") as dev_f:
        for task, (repo, config, _) in TRAIN_TASKS.items():
            counts = {"train": 0, "development": 0, "unusable": 0, "dropped_official_overlap": 0}
            if cfg.get("include_development"):
                counts.update(train_from_development=0, dropped_development_overlap=0)
            ref = OFFICIAL_FOR.get(task, task)
            for doc in _load(repo, config, "train", cfg["piqa_files"]):
                legacy = format_example(task, doc)
                if legacy is None or legacy["label"] is None:
                    counts["unusable"] += 1
                    continue
                item = harness_item(task, doc)
                record = {"task": task, "label": item["label"], "choices": item["choices"],
                          "encoded": [list(map(list, encode_pair(tok, c, k))) for c, k in item["pairs"]]}
                words = _words(_context_text(task, doc))
                grams = {tuple(words[i:i + 13]) for i in range(len(words) - 12)}
                overlaps = " ".join(words) in official_keys[ref] or bool(grams & official_grams[ref])
                if is_development(task, legacy["text"], cfg["development_fraction"]):
                    dev_f.write(json.dumps(record, ensure_ascii=False) + "\n"); counts["development"] += 1
                    if cfg.get("include_development"):
                        if overlaps:
                            counts["dropped_development_overlap"] += 1
                        else:
                            train_f.write(json.dumps({**record, "from_development": True}, ensure_ascii=False) + "\n")
                            counts["train_from_development"] += 1
                    continue
                if overlaps:
                    counts["dropped_official_overlap"] += 1
                    continue
                train_f.write(json.dumps(record, ensure_ascii=False) + "\n"); counts["train"] += 1
            prior = expected[task]
            if counts["development"] != prior["development"] or \
                    counts["train"] + counts["dropped_official_overlap"] != prior["positives"]:
                raise ValueError(f"{task}: split differs from the recorded V2 split {prior}: {counts}")
            report["tasks"][task] = {"repo": repo, "config": config, "revision": PINNED[repo], **counts}
            print(json.dumps({"task": task, **counts}), flush=True)
    report["files"] = {name: sha256_file(out / name) for name in ("train.jsonl", "development.jsonl")}
    write_json(out / "report.json", report)
    return report


# ---------------------------------------------------------------- training
def _lr(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * max(0.0, 1 - (step - warmup + 1) / max(1, total - warmup))


class ReplayStreams:
    """Seeded replay batches from the V2 training shards, mixed by the pretraining weights."""

    def __init__(self, manifest_path: Path, weights: dict, seed: int, seq_len: int = 1024):
        from scglm.data import SourceStream
        manifest = json.loads(manifest_path.read_text())
        self.rng = random.Random(seed)
        self.names = list(weights)
        self.weights = [weights[n] for n in self.names]
        self.streams = {}
        for name in self.names:
            stream = SourceStream(manifest["sources"][name]["splits"]["train"]["paths"], seq_len=seq_len)
            stream.cursor = self.rng.randrange(stream.total_tokens)
            self.streams[name] = stream
        self.tokens = {n: 0 for n in self.names}

    def batch(self, n: int):
        import numpy as np
        xs, ys = [], []
        for name in self.rng.choices(self.names, weights=self.weights, k=n):
            x, y = self.streams[name].next_batch(1)
            xs.append(x); ys.append(y)
            self.tokens[name] += x.size
        return np.concatenate(xs), np.concatenate(ys)


def task_losses(model, items: list[dict], device: str, temperature: float = 1.0, gold_scope: str = "sequence"):
    """Return (choice_ce, gold_lm, batch_accuracy) for tokenized items under bf16 autocast.

    ``temperature`` divides the candidate scores inside the softmax only (the argmax, i.e.
    the harness decision, is unchanged). ``gold_scope`` is "sequence" (all gold tokens,
    GPT-1's auxiliary loss) or "continuation" (gold answer tokens only).
    """
    import torch
    import torch.nn.functional as F
    rows, owners = [], []
    for k, item in enumerate(items):
        for j, (ctx, cont) in enumerate(item["encoded"]):
            full, n = model_input(ctx, cont)
            rows.append((full, n)); owners.append((k, j))
    width = max(len(f) - 1 for f, _ in rows)
    x = torch.zeros((len(rows), width), dtype=torch.long)
    y = torch.full((len(rows), width), -100, dtype=torch.long)
    for r, (full, _) in enumerate(rows):
        x[r, :len(full) - 1] = torch.tensor(full[:-1])
        y[r, :len(full) - 1] = torch.tensor(full[1:])
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        logits = model(input_ids=x, use_cache=False).logits
    logp = F.log_softmax(logits.float(), dim=-1)
    token_lp = logp.gather(2, y.clamp(min=0)[..., None]).squeeze(2) * (y != -100)
    width_choices = max(len(item["encoded"]) for item in items)
    scores = torch.full((len(items), width_choices), float("-inf"), device=device)
    gold_rows = []
    cont_mask = torch.zeros_like(y, dtype=torch.bool)
    for r, ((full, n), (k, j)) in enumerate(zip(rows, owners)):
        inplen = len(full) - 1
        scores[k, j] = token_lp[r, inplen - n:inplen].sum()
        cont_mask[r, inplen - n:inplen] = True
        if j == items[k]["label"]:
            gold_rows.append(r)
    labels = torch.tensor([item["label"] for item in items], device=device)
    choice_ce = F.cross_entropy(scores / temperature, labels)
    gold = torch.tensor(gold_rows, device=device)
    mask = (y[gold] != -100) if gold_scope == "sequence" else cont_mask[gold]
    gold_lm = -(token_lp[gold] * mask).sum() / mask.sum()
    accuracy = float((scores.argmax(1) == labels).float().mean())
    return choice_ce, gold_lm, accuracy


def train(cfg: dict, arm_name: str, device: str = "cuda", max_updates: int | None = None) -> None:
    """Train one arm. ``max_updates`` is for smoke tests: stop early and export to ``smoke/export``."""
    import numpy as np
    import torch
    from scglm.model import audit_parameters, forward_loss, load_model, save_model

    arm = cfg["arms"][arm_name]
    run_dir = ROOT / arm["run_dir"]
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists; every arm writes a new run directory")
    run_dir.mkdir(parents=True)
    log = run_dir / "events.jsonl"
    parent = ROOT / cfg["parent_export"]
    parent_sha = sha256_file(parent / "model.safetensors")
    if parent_sha != cfg["parent_sha256"]:
        raise ValueError("Parent weights differ from the registered parent")
    data_dir = ROOT / cfg["data_dir"]
    data_report = json.loads((data_dir / "report.json").read_text())
    if sha256_file(data_dir / "train.jsonl") != data_report["files"]["train.jsonl"]:
        raise ValueError("Training data changed after preparation")
    items = [json.loads(line) for line in (data_dir / "train.jsonl").open()]
    seed = int(arm.get("seed", cfg["seed"]))
    torch.manual_seed(seed)
    model = load_model(parent).to(device)
    model.train()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=arm["learning_rate"], betas=tuple(cfg["betas"]), eps=cfg["adam_epsilon"],
                                  fused=device.startswith("cuda"))
    q = cfg["questions_per_update"]
    per_epoch = math.ceil(len(items) / q)
    epochs = int(arm.get("epochs", cfg["epochs"]))
    total = per_epoch * epochs
    warmup = max(1, round(cfg["warmup_fraction"] * total))
    replay = ReplayStreams(ROOT / cfg["replay_manifest"], cfg["replay_weights"], seed)
    r, lm_w = arm.get("replay_loss_weight", cfg["replay_loss_weight"]), cfg["gold_lm_weight"]
    record = {"arm": arm_name, "config": cfg, "arm_config": arm, "parent_export": str(parent),
              "parent_sha256": parent_sha, "data_report": data_report, "train_items": len(items),
              "updates_per_epoch": per_epoch, "total_updates": total, "warmup_updates": warmup,
              "source_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "started": now(), "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device}
    write_json(run_dir / "run.json", record)
    event(log, "started", total_updates=total, updates_per_epoch=per_epoch, train_items=len(items))
    step, started = 0, time.monotonic()
    counters = {"questions": 0, "candidate_sequences": 0, "task_input_tokens": 0, "replay_targets": 0}

    def status(state, **extra):
        write_json(run_dir / "status.json", {"status": state, "arm": arm_name, "step": step, "total_updates": total,
                                             "elapsed_seconds": time.monotonic() - started, "counters": counters,
                                             "replay_source_tokens": replay.tokens, "updated": now(), **extra})

    try:
        status("running")
        for epoch in range(1, epochs + 1):
            order = list(range(len(items)))
            random.Random(seed * 1000 + epoch).shuffle(order)
            for b in range(0, len(order), q):
                batch = [items[i] for i in order[b:b + q]]
                lr = _lr(step, total, arm["learning_rate"], warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                choice_ce, gold_lm, acc = task_losses(model, batch, device, cfg.get("choice_temperature", 1.0),
                                                      cfg.get("gold_lm_scope", "sequence"))
                rx, ry = replay.batch(cfg["replay_sequences"])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    replay_lm = forward_loss(model, torch.from_numpy(rx).to(device), torch.from_numpy(ry).to(device))
                loss = (1 - r) * (cfg.get("choice_weight", 1.0) * choice_ce + lm_w * gold_lm) + r * replay_lm
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                step += 1
                counters["questions"] += len(batch)
                counters["candidate_sequences"] += sum(len(i["encoded"]) for i in batch)
                counters["task_input_tokens"] += sum(len(model_input(c, k)[0]) - 1 for i in batch for c, k in i["encoded"])
                counters["replay_targets"] += int(rx.size)
                if step % cfg["log_every"] == 0 or step == 1:
                    event(log, "train", step=step, epoch=epoch, lr=lr, loss=float(loss),
                          choice_ce=float(choice_ce), gold_lm=float(gold_lm), replay_lm=float(replay_lm),
                          batch_accuracy=acc, gradient_norm=float(norm))
                    status("running", epoch=epoch, loss=float(loss), replay_lm=float(replay_lm))
                if max_updates is not None and step >= max_updates:
                    break
            smoke = max_updates is not None and step >= max_updates
            export = run_dir / ("smoke" if smoke else f"epoch_{epoch}") / "export"
            save_model(model, export)
            for name in ("tokenizer.json", "tokenizer_config.json"):
                shutil.copy2(parent / name, export / name)
            write_json(export / "training_provenance.json", {
                "run": str(run_dir), "arm": arm_name, "epoch": epoch, "step": step, "total_updates": total,
                "schedule_note": "epoch exports before the last are prefixes of the full linear-decay schedule",
                "initialization": "scratch", "task_tuning": {"parent_export": str(parent), "parent_sha256": parent_sha,
                                                             "data_files": data_report["files"], "counters": dict(counters)},
                "parent": json.loads((parent / "training_provenance.json").read_text()),
                "parameter_audit": audit_parameters(model)})
            event(log, "export", epoch=epoch, step=step, path=str(export),
                  model_sha256=sha256_file(export / "model.safetensors"))
            if smoke:
                break
            if arm.get("stop_after_epoch") and epoch >= int(arm["stop_after_epoch"]):
                event(log, "stopped_after_epoch", epoch=epoch, note="schedule length unchanged; later epochs not needed")
                break
        status("smoke_completed" if max_updates is not None else "completed")
        event(log, "finished", steps=step, counters=counters, replay_source_tokens=replay.tokens,
              seconds=time.monotonic() - started)
    except BaseException as exc:
        status("failed", error=repr(exc))
        event(log, "error", error=repr(exc))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "train"])
    p.add_argument("--config", required=True)
    p.add_argument("--arm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-updates", type=int, help="smoke test: stop early and export to smoke/export")
    p.add_argument("--run-dir", help="override the arm's run_dir (smoke tests)")
    args = p.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.command == "prepare":
        prepare(cfg)
    else:
        if not args.arm:
            raise SystemExit("--arm is required for train")
        if args.run_dir:
            cfg["arms"][args.arm]["run_dir"] = args.run_dir
        train(cfg, args.arm, args.device, args.max_updates)


if __name__ == "__main__":
    main()
