"""Recoverable paired trainer, independent of the retained SFT trainer."""
from collections import Counter
import os
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from scglm.model import load_model, save_model, audit_parameters
from scglm_post.sft import learning_rate
from scglm_post.common import publish_export
from .common import (ROOT, SEEDS, ARMS, roots, read, read_jsonl, write_json, sha256, digest,
                     sources, runtime, immutable, event, status, StudyStopped)
from .objective import update


def optimizer(model):
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": .1},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.}],
        lr=3e-6, betas=(.9,.95), eps=1e-8)


class Recovery:
    """Small independent state machine, also exercised by disposable CPU tests."""
    def __init__(self, directory, contract, model, optim):
        self.directory, self.contract = Path(directory), contract
        self.model, self.optimizer = model, optim
        self.step, self.counters = 0, Counter()
        self.resource_accounting = None
        self.directory.mkdir(parents=True, exist_ok=True)
        immutable(self.directory / "run.json", {"contract": contract, "contract_sha256": digest(contract)})
        if (self.directory / "checkpoint_latest.json").exists():
            self.restore()

    def save(self):
        path = self.directory / f"checkpoint_{self.step:07d}_{time.time_ns()}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({"contract_sha256": digest(self.contract), "step": self.step, "counters": dict(self.counters),
                    "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if next(self.model.parameters()).is_cuda else None,
                    "resource_accounting": self.resource_accounting() if self.resource_accounting else None,
                    "batch_position": self.step, "replay_position": self.step}, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        write_json(self.directory / "checkpoint_latest.json", {"file": path.name, "sha256": sha256(path), "step": self.step})
        # Keep previous checkpoints as experiment evidence; no existing exports removed.
        return path

    def restore(self):
        pointer = read(self.directory / "checkpoint_latest.json")
        path = self.directory / pointer["file"]
        if not path.resolve().is_relative_to(self.directory.resolve()) or sha256(path) != pointer["sha256"]:
            raise ValueError("Recovery checkpoint changed")
        # AdamW's non-capturable step scalars live on CPU. Loading the whole
        # pickle onto CUDA moves them unnecessarily and changes recovery state.
        # load_state_dict places parameter moments on their parameter devices.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["contract_sha256"] != digest(self.contract) or state["batch_position"] != state["step"] or state["replay_position"] != state["step"]:
            raise ValueError("Resume contract or batch/replay position changed")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.counters = state["step"], Counter(state["counters"])
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])


class Trainer:
    def __init__(self, cfg, seed, arm, manifest, budget, on_evaluate):
        if seed not in SEEDS or arm not in ARMS:
            raise ValueError("Unregistered seed or arm")
        self.cfg, self.seed, self.arm, self.budget, self.on_evaluate = cfg, seed, arm, budget, on_evaluate
        data, output = roots(cfg)
        self.directory = output / "runs" / f"{arm}_seed{seed}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parent = ROOT / cfg["parent"]["path"]
        self.manifest = manifest
        self.plan = read(data / f"plan_{seed}.json")
        if digest(self.plan) != manifest["plans"][str(seed)]:
            raise ValueError("Paired plan changed")
        self.response = {r["id"]: r for r in read_jsonl(data / "responses.jsonl")}
        self.replay = {r["id"]: r for r in read_jsonl(data / "replay.jsonl")}
        self.choices = {r["id"]: r for r in read_jsonl(data / "choices.jsonl")}
        contract = {"pipeline": "scglm-choice-study-v1", "config_sha256": digest(cfg), "seed": seed, "arm": arm,
                    "parent_sha256": cfg["parent"]["weight_sha256"], "manifest_sha256": sha256(data / "manifest.json"),
                    "plan_sha256": digest(self.plan), "sources": sources(), "runtime": runtime(),
                    "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "cpu_threads": torch.get_num_threads()}
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.model = load_model(self.parent).to("cuda:0")
        if audit_parameters(self.model)["total_unique_parameters"] != cfg["parent"]["parameters"]:
            raise ValueError("Changed model architecture")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.parent), local_files_only=True)
        self.state = Recovery(self.directory, contract, self.model, optimizer(self.model))
        self.state.resource_accounting = lambda: self.budget.value
        if not (self.directory / "checkpoint_latest.json").exists():
            for relative in contract["sources"]:
                destination = self.directory / "snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            self.state.save()
        self.stop_requested = False
        self.attempts = read(self.directory / "attempts.json") if (self.directory / "attempts.json").exists() else {"started": 0, "completed": 0}
        event(self.directory, "restored", committed_updates=self.state.step, actual_completed_attempts=self.attempts["completed"],
              discarded_completed_updates=max(0,self.attempts["completed"]-self.state.step),
              cumulative_budget=self.budget.value)

    def export(self):
        if sources() != self.state.contract["sources"]:
            raise ValueError("Runtime source contract changed")
        path = self.directory / "endpoints" / f"step_{self.state.step:07d}" / "export"
        if path.exists():
            from .lineage import endpoint
            endpoint(path, self.cfg)
            prov = read(path / "training_provenance.json")
            if prov["contract_sha256"] != digest(self.state.contract):
                raise ValueError("Existing export contract changed")
            return path
        temporary = path.with_name(f"export.tmp.{os.getpid()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        save_model(self.model, temporary)
        self.tokenizer.save_pretrained(str(temporary))
        shutil.copy2(self.parent / "tokenizer.json", temporary / "tokenizer.json")
        provenance = {"stage": "choice_discrimination", "pipeline": "scglm-choice-study-v1", "arm": self.arm, "seed": self.seed,
                      "parent_model": str(self.parent), "parent_model_sha256": self.cfg["parent"]["weight_sha256"],
                      "scratch_base_sha256": self.cfg["base_reference"]["weight_sha256"],
                      "cumulative_pretraining_tokens": 13000048640, "historical_sft_updates": 105,
                      "historical_sft_schedule_updates": 209, "optimizer_updates": self.state.step,
                      "step": self.state.step, "counters": dict(self.state.counters),
                      "plan_sha256": digest(self.plan), "manifest_sha256": self.state.contract["manifest_sha256"],
                      "contract_sha256": digest(self.state.contract), "new_optimizer": True,
                      "cumulative_resource_accounting": self.budget.value}
        write_json(temporary / "training_provenance.json", provenance)
        write_json(temporary / "endpoint.json", {"weights_sha256": sha256(temporary / "model.safetensors"),
                   "provenance_sha256": sha256(temporary / "training_provenance.json"), "source_run": str(self.directory)})
        publish_export(temporary, path)
        return path

    def run(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for s in previous:
            signal.signal(s, lambda *_: setattr(self, "stop_requested", True))
        try:
            state_path = self.directory / "status.json"
            if state_path.exists() and read(state_path)["status"] == "failed":
                raise StudyStopped("Failed arm cannot be retrained or resumed")
            if state_path.exists() and read(state_path)["status"] == "completed":
                return self.export()
            # A saved update boundary may have an interrupted development check.
            if self.state.step in (8,16,24,32):
                self.on_evaluate(self.export(), self)
            status(self.directory, "running", optimizer_updates=self.state.step, counters=dict(self.state.counters))
            while self.state.step < 32 and not self.stop_requested:
                batch = self.plan["batches"][self.state.step]
                rows = [self.response[i] for i in batch["response_ids"]]
                raw = [self.replay[i] for i in batch["replay_ids"]]
                items = [self.choices[i] for i in batch["choice_ids"]]
                rate = learning_rate(self.state.counters["response_targets"] + batch["response_targets"], self.plan["schedule_denominator"],
                                     {"learning_rate": 3e-6, "warmup_fraction": .03})
                for group in self.state.optimizer.param_groups:
                    group["lr"] = rate
                self.attempts["started"] += 1
                write_json(self.directory / "attempts.json", self.attempts)
                event(self.directory, "update_started", step=self.state.step + 1, cumulative_budget=self.budget.value)
                metrics = update(self.model, self.state.optimizer, rows, raw, items, candidate=self.arm == "choice_candidate", charge=self.budget.charge)
                self.attempts["completed"] += 1
                write_json(self.directory / "attempts.json", self.attempts)
                self.state.step += 1
                self.state.counters.update({k:v for k,v in metrics.items() if k.endswith(("tokens", "targets", "positions", "items"))})
                event(self.directory, "update_completed", step=self.state.step, learning_rate=rate, metrics=metrics)
                if self.state.step % 8 == 0:
                    self.state.save()
                    self.on_evaluate(self.export(), self)
                status(self.directory, "running", optimizer_updates=self.state.step, counters=dict(self.state.counters),
                       cumulative_budget=self.budget.value)
            self.state.save()
            complete = self.state.step == 32
            status(self.directory, "completed" if complete else "paused", optimizer_updates=self.state.step,
                   actual_optimizer_attempts=self.attempts, counters=dict(self.state.counters), cumulative_budget=self.budget.value)
            if not complete:
                raise InterruptedError("Clean interruption saved; resume the same study")
            return self.export()
        except InterruptedError:
            raise
        except BaseException as error:
            # Never save a partially accumulated update or half-applied optimizer.
            status(self.directory, "failed", optimizer_updates=self.state.step, error=repr(error),
                   counters=dict(self.state.counters), actual_optimizer_attempts=self.attempts,
                   recoverable_checkpoint=read(self.directory / "checkpoint_latest.json"), cumulative_budget=self.budget.value)
            raise
        finally:
            for s,h in previous.items():
                signal.signal(s,h)
