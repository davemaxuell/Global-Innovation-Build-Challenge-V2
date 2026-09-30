"""V2 scratch pretraining: V1 trainer plus compile, WSD schedule and a decay trigger.

Everything else (data streams, epoch guards, checkpoints, exact resume, export) is
the retained ``scglm.train.Trainer``. Differences:

* ``compile: true`` compiles the forward pass (validated by a CPU/GPU parity test).
* ``schedule: "wsd"`` uses linear warmup over ``warmup_steps``, then the constant
  peak LR. Decay starts only when a decay plan exists: either ``decay_start_step``
  in the config, or ``<run_dir>/decay.json`` = {"decay_start_step": S, "decay_steps": D},
  written by the operator. The LR then falls linearly to ``min_lr_ratio * peak`` over
  D steps, and the run completes and exports at S + D. The stable-phase checkpoint
  at S is kept as a milestone so other decay points can branch from it later.
* ``schedule: "cosine"`` keeps V1's warmup-fraction cosine (used by the 1B pilot).

Run: PYTHONPATH=src python -m scglm_v2.train --config <v2 config> [--resume <run>/checkpoint_latest.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scglm import train as v1
from scglm.model import forward_loss


def wsd_learning_rate(step: int, peak: float, warmup: int, min_ratio: float,
                      decay_start: int | None, decay_steps: int | None) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    if decay_start is None or step < decay_start:
        return peak
    progress = min(1.0, (step - decay_start + 1) / max(1, decay_steps))
    return peak * (1 - (1 - min_ratio) * progress)


class V2Trainer(v1.Trainer):
    def __init__(self, config, device="cuda", resume=None):
        self.compile_requested = bool(config.get("compile", False))
        base = {k: v for k, v in config.items() if k != "compile"}
        base["compile"] = False  # V1 guard; compilation is applied below
        super().__init__(base, device=device, resume=resume)
        self.config["compile"] = self.compile_requested
        if resume:
            # V1's science digest predates these keys; the decay plan itself may change.
            record = json.loads((self.run_dir / "run.json").read_text())
            stored = record["config"]
            for key in ("schedule", "warmup_steps"):
                if stored.get(key) != config.get(key):
                    raise ValueError(f"Resume changes {key}; start a new run instead")
            # The base record stores compile=False (V1 guard); the V2 block holds the real value.
            if record.get("v2", {}).get("compile", False) != self.compile_requested:
                raise ValueError("Resume changes compile; start a new run instead")
        self.forward_model = torch.compile(self.model) if self.compile_requested else self.model
        if config.get("schedule", "cosine") not in ("cosine", "wsd"):
            raise ValueError("schedule must be 'cosine' or 'wsd'")

    # ----------------------------------------------------------- schedule
    def decay_plan(self) -> tuple[int | None, int | None]:
        path = self.run_dir / "decay.json"
        if path.exists():
            plan = json.loads(path.read_text())
            plan = int(plan["decay_start_step"]), int(plan["decay_steps"])
        elif self.config.get("decay_start_step") is not None:
            plan = int(self.config["decay_start_step"]), int(self.config["decay_steps"])
        else:
            plan = (None, None)
        if plan != getattr(self, "_accepted_plan", None):
            # A new or changed plan may not start in the past or alter an already-running decay.
            old = getattr(self, "_accepted_plan", (None, None))
            if plan[0] is not None and (plan[0] < self.step or plan[1] < 1):
                raise ValueError(f"Decay plan {plan} starts before the current step {self.step}")
            if old[0] is not None and self.step > old[0]:
                raise ValueError("The decay has already started; its plan cannot change")
            self._accepted_plan = plan
            if hasattr(self, "event_path"):
                self.log("decay_plan", decay_start_step=plan[0], decay_steps=plan[1])
        return plan

    def lr_at(self, step: int) -> float:
        c = self.config
        if c.get("schedule", "cosine") == "cosine":
            return v1.learning_rate(step, self.total_steps, c["learning_rate"], c["warmup_fraction"], c["min_lr_ratio"])
        start, steps = self.decay_plan()
        return wsd_learning_rate(step, c["learning_rate"], int(c["warmup_steps"]), c["min_lr_ratio"], start, steps)

    def planned_end(self) -> int:
        if self.config.get("schedule", "cosine") == "wsd":
            start, steps = self.decay_plan()
            if start is not None:
                return min(self.total_steps, start + steps)
        return self.total_steps

    # ----------------------------------------------------------- validation
    def documents(self, split_name, limit=None):
        """Label each held-out document with its manifest bucket.

        V2 panel records keep their original source (the targeted bucket holds
        ``dclm``/``edu`` documents), and V1's reader keeps an existing label, so
        without this a bucket can appear to have no documents during validation.
        """
        for bucket in self.sources:
            split = self.manifest["sources"][bucket]["splits"].get(split_name)
            if split is None:
                raise ValueError(f"Missing split {bucket}/{split_name}")
            path = Path(split["jsonl_path"])
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            if split.get("jsonl_sha256") and v1.sha256_file(path) != split["jsonl_sha256"]:
                raise ValueError(f"Development document checksum mismatch: {bucket}/{split_name}")
            with open(path) as f:
                for index, line in enumerate(f):
                    if limit is not None and index >= limit:
                        break
                    record = json.loads(line)
                    record["origin_source"] = record.get("source")
                    record["source"] = bucket
                    yield record

    # ----------------------------------------------------------- update
    def update(self, web_share):
        started = time.monotonic()
        self.model.train()
        lr = self.lr_at(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        x, y, counts = self.batch(web_share)
        self.optimizer.zero_grad(set_to_none=True)
        accum = self.batch_sequences // self.microbatch_sequences
        losses = []
        for offset in range(0, self.batch_sequences, self.microbatch_sequences):
            xt = torch.from_numpy(x[offset:offset + self.microbatch_sequences]).to(self.device, non_blocking=True)
            yt = torch.from_numpy(y[offset:offset + self.microbatch_sequences]).to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                loss = forward_loss(self.forward_model, xt, yt)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; aborting before optimizer update.")
            (loss / accum).backward()
            losses.append(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        losses = [float(l) for l in losses]  # one host sync per update
        duration = time.monotonic() - started
        self.training_tokens_spent += self.tokens_per_step
        self.main_training_seconds += duration
        self.step += 1
        self.main_tokens += self.tokens_per_step
        for source in self.sources:
            self.main_source_tokens[source] += counts[source]
        self.log("optimizer_update_accounting", quiet=True, probe=False, duration_seconds=duration)
        return float(np.mean(losses)), float(norm), lr

    # ----------------------------------------------------------- loop
    def run(self, max_steps=None, max_seconds=None):
        """V1 loop with a movable end (the WSD decay plan is re-read every step)."""
        import signal
        stop_at = self.config.get("max_wall_hours", 30) * 3600
        invocation_start = previous_log_time = time.monotonic()
        previous_log_tokens, last_loss = self.main_tokens, None
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        try:
            self.write_status("running")
            while True:
                end = self.planned_end()
                limit = min(end, max_steps) if max_steps is not None else end
                if self.step >= limit or self.stop_requested or self.elapsed >= stop_at:
                    break
                if max_seconds and time.monotonic() - invocation_start >= max_seconds:
                    break
                start, _ = self.decay_plan()
                if start is not None and self.step == start:
                    self.save_checkpoint(reason=f"milestone_decay_start_{start}")
                tick = time.monotonic()
                loss, norm, lr = self.update(self.current_web_share)
                last_loss = loss
                if self.step % self.config.get("log_every", 10) == 0 or self.step == 1:
                    now = time.monotonic()
                    rate = (self.main_tokens - previous_log_tokens) / max(1e-9, now - previous_log_time)
                    self.log("train", loss=loss, gradient_norm=norm, learning_rate=lr, tokens_per_second=rate,
                             update_seconds=now - tick, source_tokens=dict(self.main_source_tokens))
                    self.write_status("running", loss=loss, recent_tokens_per_second=rate, planned_end_step=end)
                    previous_log_time, previous_log_tokens = now, self.main_tokens
                every = self.config.get("validation_every", 0)
                if every and self.step % every == 0:
                    self.validate(limit=self.config.get("validation_documents_per_source", 32))
                milestones = [m for m in self.config.get("milestone_cumulative_tokens", [])
                              if self.main_tokens - self.tokens_per_step < m <= self.main_tokens]
                if milestones:
                    self.save_checkpoint(reason=f"milestone_{int(milestones[-1])}")
                elif self.step % self.config.get("checkpoint_every", 250) == 0:
                    self.save_checkpoint()
            complete = self.step >= self.planned_end()
            self.save_checkpoint(reason="complete" if complete else "paused")
            if complete:
                self.export()
            self.write_status("completed" if complete else "paused", loss=last_loss)
            self.log("finished", complete=complete, total_training_tokens=self.main_tokens)
        except Exception as exc:
            self.log("error", error=repr(exc))
            self.write_status("failed", error=repr(exc))
            raise

    def write_run_record(self):
        super().write_run_record()
        record = json.loads((self.run_dir / "run.json").read_text())
        record["v2"] = {"trainer": "scglm_v2.train.V2Trainer", "compile": self.compile_requested,
                        "schedule": self.config.get("schedule", "cosine"),
                        "v2_source_sha256": {p.name: v1.sha256_file(p) for p in
                                             sorted(Path(__file__).resolve().parent.glob("*.py"))}}
        record["ai_assistance"] = "Claude (Anthropic) and earlier Codex/ChatGPT assisted research, implementation, tests and documentation."
        v1.atomic_json(self.run_dir / "run.json", record)
        import shutil
        snapshot = self.run_dir / "snapshot_v2"
        shutil.copytree(Path(__file__).resolve().parent, snapshot / "scglm_v2", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config = v1.read_config(args.config)
    trainer = V2Trainer(config, device=args.device, resume=args.resume)
    trainer.run(max_steps=args.max_steps, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
