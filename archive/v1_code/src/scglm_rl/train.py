"""Recoverable RL arm: collect under a frozen policy, one on-policy update, repeat.

Recovery checkpoints hold model, AdamW, RNG state, committed update count and
sampler cursors. The collection ledger and GPU budget are write-ahead and
survive rollback. A clean interruption pauses at an update boundary.
"""
from collections import Counter
import os
import shutil
import signal

import torch
from transformers import AutoTokenizer

from scglm.model import audit_parameters, load_model, save_model
from scglm_choice.train import Recovery
from scglm_post.common import publish_export
from scglm_study.common import ROOT, StudyStopped, digest, event, read, roots, runtime, sha256, sources, status, write_json
from .collect import Ledger, collect
from .loss import ARMS, update
from .rollouts import weights_digest
from .tasks import Sampler, fits


def optimizer(model, recipe):
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": recipe["weight_decay"]},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.}],
        lr=recipe["learning_rate"], betas=tuple(recipe["betas"]), eps=recipe["adam_epsilon"])


def eligible_pool(pool, families, tokenizer, sampling, context):
    return {f: [r for r in pool[f] if fits(tokenizer, r, sampling["max_new_tokens"], context)] for f in families}


class ArmTrainer:
    def __init__(self, cfg, arm, audit, pool, budget, on_evaluate, *, device="cuda:0", directory=None):
        if arm not in ARMS:
            raise ValueError("Unregistered arm")
        self.cfg, self.arm, self.audit, self.budget, self.on_evaluate = cfg, arm, audit, budget, on_evaluate
        self.recipe, self.sampling = cfg["training"], cfg["sampling"]
        _, output = roots(cfg)
        self.directory = directory or output / "runs" / arm
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parent = ROOT / cfg["parent"]["path"]
        seed = self.recipe["seed"]
        torch.manual_seed(seed)
        self.model = load_model(self.parent).to(device)
        if audit_parameters(self.model)["total_unique_parameters"] != cfg["parent"]["parameters"]:
            raise ValueError("Changed model architecture")
        self.reference = load_model(self.parent).to(device)
        self.reference.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.parent), local_files_only=True)
        self.shares = audit["family_shares"]
        self.pool = eligible_pool(pool, sorted(self.shares), self.tokenizer, self.sampling,
                                  self.model.config.max_position_embeddings)
        contract = {"pipeline": cfg["schema"], "arm": arm, "seed": seed, "parent_sha256": cfg["parent"]["weight_sha256"],
                    "audit_sha256": digest(audit), "recipe": self.recipe, "sampling": self.sampling,
                    "sources": sources(cfg), "runtime": runtime(), "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES")}
        self.state = Recovery(self.directory, contract, self.model, optimizer(self.model, self.recipe))
        self.state.resource_accounting = lambda: self.budget.value
        if not (self.directory / "checkpoint_latest.json").exists():
            self.state.save()
        self.sampler = Sampler(self.pool, seed, self.state.counters)
        self.ledger = Ledger(self.directory / "ledger.json", cfg["budget"])
        self.stop_requested = False
        event(self.directory, "restored", committed_updates=self.state.step, ledger=self.ledger.value)

    def export(self):
        path = self.directory / "endpoints" / f"step_{self.state.step:07d}" / "export"
        if path.exists():
            if read(path / "training_provenance.json")["contract_sha256"] != digest(self.state.contract):
                raise ValueError("Existing export contract changed")
            return path
        temporary = path.with_name(f"export.tmp.{os.getpid()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        save_model(self.model, temporary)
        self.tokenizer.save_pretrained(str(temporary))
        shutil.copy2(self.parent / "tokenizer.json", temporary / "tokenizer.json")
        write_json(temporary / "training_provenance.json", {
            "stage": "rl_revised", "pipeline": self.cfg["schema"], "arm": self.arm, "seed": self.recipe["seed"],
            "parent_model": str(self.parent), "parent_model_sha256": self.cfg["parent"]["weight_sha256"],
            "cumulative_pretraining_tokens": self.cfg["parent"]["cumulative_pretraining_tokens"],
            "historical_sft_updates": 105, "optimizer_updates": self.state.step, "step": self.state.step,
            "counters": dict(self.state.counters), "ledger": self.ledger.value,
            "contract_sha256": digest(self.state.contract), "new_optimizer": True})
        write_json(temporary / "endpoint.json", {"weights_sha256": sha256(temporary / "model.safetensors"),
                                                 "source_run": str(self.directory)})
        publish_export(temporary, path)
        return path

    def step_once(self):
        step = self.state.step
        policy = weights_digest(self.model)
        seed = self.recipe["seed"] + 101 * step + 7919 * self.ledger.value["skipped_updates"]
        accepted, quotas, rounds = collect(self.model, self.tokenizer, self.sampler, self.shares, self.recipe,
                                           self.sampling, seed=seed, policy_id=policy, ledger=self.ledger,
                                           charge=self.budget.charge, rates=self.audit["informative_rates"])
        chosen = [g for groups in accepted.values() for g in groups]
        present = sum(bool(v) for v in accepted.values())
        record = {"step": step + 1, "policy_sha256": policy, "quotas": quotas,
                  "accepted": {f: len(v) for f, v in accepted.items()}, "rounds": rounds}
        if present < self.recipe["min_families_per_update"]:
            self.ledger.skip()
            event(self.directory, "update_skipped", reason="too_few_families_with_signal", **record)
            return None
        metrics = update(self.model, self.reference, self.state.optimizer, chosen, self.shares, self.arm,
                         self.recipe, charge=self.budget.charge)
        self.state.step += 1
        self.state.counters.update({k: v for k, v in metrics.items() if k in ("policy_loss_targets", "policy_processed_tokens",
                                                                               "informative_groups", "clipped_tokens")})
        self.state.counters["collection_rounds"] += len(rounds)
        event(self.directory, "update_completed", metrics=metrics, **record)
        return metrics

    def run(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for s in previous:
            signal.signal(s, lambda *_: setattr(self, "stop_requested", True))
        checks, total = self.cfg["evaluation"]["development_checks_at_updates"], self.recipe["updates"]
        try:
            state_path = self.directory / "status.json"
            if state_path.exists() and read(state_path)["status"] in ("failed", "stopped"):
                raise StudyStopped("A stopped arm cannot be retrained or resumed")
            if state_path.exists() and read(state_path)["status"] == "completed":
                return self.export()
            if self.state.step in checks:
                self.on_evaluate(self.export(), self)
            status(self.directory, "running", optimizer_updates=self.state.step, ledger=self.ledger.value)
            while self.state.step < total and not self.stop_requested:
                if self.step_once() is None:
                    continue
                if self.state.step % self.recipe["checkpoint_every"] == 0 or self.state.step in checks:
                    self.state.save()
                if self.state.step in checks:
                    self.on_evaluate(self.export(), self)
                status(self.directory, "running", optimizer_updates=self.state.step, ledger=self.ledger.value,
                       counters=dict(self.state.counters))
            self.state.save()
            complete = self.state.step == total
            status(self.directory, "completed" if complete else "paused", optimizer_updates=self.state.step,
                   ledger=self.ledger.value, counters=dict(self.state.counters), cumulative_budget=self.budget.value)
            if not complete:
                raise InterruptedError("Clean interruption saved; resume the same study")
            return self.export()
        except InterruptedError:
            raise
        except StudyStopped as error:
            status(self.directory, "stopped", optimizer_updates=self.state.step, reason=str(error),
                   ledger=self.ledger.value, counters=dict(self.state.counters), cumulative_budget=self.budget.value)
            raise
        except BaseException as error:
            status(self.directory, "failed", optimizer_updates=self.state.step, error=repr(error),
                   ledger=self.ledger.value, recoverable_checkpoint=read(self.directory / "checkpoint_latest.json"),
                   cumulative_budget=self.budget.value)
            raise
        finally:
            for s, h in previous.items():
                signal.signal(s, h)
