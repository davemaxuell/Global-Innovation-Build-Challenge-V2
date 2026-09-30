"""Single-device scratch pretraining with exact state recovery and auditable costs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
from typing import Any

import numpy as np
import torch

from .data import SourceStream
from .model import build_model, count_parameters, forward_loss

SOURCES = ("web", "wiki")
RUNTIME_PACKAGES = ("torch", "transformers", "datasets", "tokenizers", "numpy", "safetensors")




def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def read_config(path):
    config = json.loads(Path(path).read_text())
    parent = config.pop("inherits", None)
    if parent:
        config = {**read_config(parent), **config}
    return config


def learning_rate(step, total_steps, peak, warmup_fraction, min_ratio):
    warmup = max(1, int(total_steps * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup - 1))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def runtime_fingerprint():
    versions = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"versions": versions, "numerics": {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_runtime": torch.__version__, "cuda_runtime": torch.version.cuda,
    }, "source_sha256": {
        p.name: sha256_file(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))}}


def configure_numerics(config, device):
    deterministic = config.get("deterministic", True)
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic must be a boolean")
    if torch.device(device).type == "cuda" and deterministic:
        # Set before the first matrix operation creates a CUDA BLAS handle.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
            raise ValueError("This experiment requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def verify_scratch_lineage(run, seen=None):
    """Follow completed local run records back to the original random run."""
    run = Path(run).resolve()
    seen = set() if seen is None else set(seen)
    if run in seen:
        raise ValueError("Cyclic scratch lineage")
    seen.add(run)
    record = json.loads((run / "run.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if status["status"] != "completed" or status["main_tokens"] != record["exact_target_tokens"]:
        raise ValueError("Parent pretraining run must be complete.")
    edge = record.get("parent")
    if record.get("initialization") == "new random model weights; no external pretrained checkpoint":
        if edge is not None or record["config"].get("parent_checkpoint"):
            raise ValueError("Random scratch origin has an unexpected parent")
        cumulative, ancestors = status["main_tokens"], []
    elif record.get("initialization") == "continuation of own documented scratch checkpoint" and edge:
        ancestor = Path(edge["run"]).resolve()
        if sha256_file(ancestor / "run.json") != edge["run_record_sha256"]:
            raise ValueError("Ancestor run record integrity failed")
        lineage = verify_scratch_lineage(ancestor, seen)
        ancestor_record = json.loads((ancestor / "run.json").read_text())
        ancestor_status = json.loads((ancestor / "status.json").read_text())
        if any(edge[key] != ancestor_status[key] for key in ("main_tokens", "step")):
            raise ValueError("Ancestor token/step ledger differs")
        if edge["science_digest"] != ancestor_record["science_digest"]:
            raise ValueError("Ancestor scientific identity differs")
        if any(record[key] != ancestor_record[key] for key in ("model_config", "tokenizer_sha256")):
            raise ValueError("Architecture or tokenizer changes inside scratch lineage")
        checkpoint = Path(edge["checkpoint"]).resolve()
        if checkpoint.parent != ancestor / "checkpoints" or sha256_file(checkpoint) != edge["checkpoint_sha256"]:
            raise ValueError("Ancestor checkpoint integrity failed")
        if edge.get("cumulative_tokens", edge["main_tokens"]) != lineage["cumulative_tokens"]:
            raise ValueError("Ancestor cumulative token ledger differs")
        cumulative = status["main_tokens"] + lineage["cumulative_tokens"]
        ancestors = lineage["runs"]
    else:
        raise ValueError("Lineage does not reach a documented random scratch origin")
    # Original random-run records predate cumulative fields. Only that root
    # may derive its cumulative ledger from its own verified target count.
    reported = status.get("cumulative_tokens", cumulative if edge is None else None)
    registered = record.get("cumulative_target_tokens", cumulative if edge is None else None)
    if reported != cumulative or registered != cumulative:
        raise ValueError("Completed cumulative token ledger differs from scratch lineage")
    return {"cumulative_tokens": cumulative,
            "runs": ancestors + [{"run": str(run), "main_tokens": status["main_tokens"],
                                   "run_record_sha256": sha256_file(run / "run.json")} ]}


class Trainer:
    def __init__(self, config, device="cuda", resume=None):
        self.started = time.monotonic()
        self.previous_elapsed = 0.0
        self.config = dict(config)
        if config.get("controller"):
            raise ValueError("Only the retained fixed-mixture pretraining recipe is supported.")
        if not isinstance(config.get("inherit_data_state", False), bool):
            raise ValueError("inherit_data_state must be boolean")
        if config.get("inherit_data_state") and not config.get("parent_checkpoint"):
            raise ValueError("Inheriting data state requires a parent checkpoint")
        self.device = torch.device(device)
        configure_numerics(config, self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one assigned GPU using CUDA_VISIBLE_DEVICES.")
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.run_dir = Path(config["run_dir"])
        if resume is not None and not (self.run_dir / "run.json").is_file():
            raise ValueError("Resume requires original run.json; different scientific configuration or missing continuation provenance.")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "run.json").exists() and resume is None:
            raise RuntimeError(f"Run exists; resume explicitly or choose a new directory: {self.run_dir}")
        self.event_path = self.run_dir / "events.jsonl"
        self.stop_requested = False
        self.step = 0
        self.main_tokens = 0
        self.source_weights = config.get("source_weights")
        self.sources = tuple(self.source_weights) if self.source_weights else SOURCES
        if self.source_weights:
            values = list(self.source_weights.values())
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-12):
                raise ValueError("Source weights must be positive, finite, and sum to one.")
        self.main_source_tokens = dict.fromkeys(self.sources, 0)
        # Keep zero-valued legacy ledger fields for recorded checkpoint schemas.
        # Adaptive control and probe optimizer updates are no longer implemented.
        self.probe_tokens = 0
        self.probe_seconds = 0.0
        self.main_training_seconds = 0.0
        self.training_tokens_spent = 0
        self.controller_seconds = 0.0
        self.controller_probes_used = 0
        self.controller_windows_done = []
        self.controller_panels_used = []
        self.current_web_share = None if self.source_weights else float(config["web_share"])
        if not self.source_weights and (not math.isfinite(self.current_web_share) or not 0 <= self.current_web_share <= 1):
            raise ValueError("Mixture share must be finite and in [0, 1].")
        if not math.isfinite(float(config.get("max_source_epochs", 1.0))) or config.get("max_source_epochs", 1.0) <= 0:
            raise ValueError("max_source_epochs must be finite and positive.")
        self.batch_sequences = int(config["batch_sequences"])
        self.microbatch_sequences = int(config["microbatch_sequences"])
        self.sequence_length = int(config["sequence_length"])
        if min(self.batch_sequences, self.microbatch_sequences, self.sequence_length) <= 0:
            raise ValueError("Batch and sequence sizes must be positive.")
        if self.batch_sequences % self.microbatch_sequences:
            raise ValueError("Effective batch must be divisible by microbatch.")
        self.tokens_per_step = self.batch_sequences * self.sequence_length
        self.total_steps = int(config["target_tokens"]) // self.tokens_per_step
        if self.total_steps < 1:
            raise ValueError("Token budget is smaller than one optimizer update.")
        self.exact_target_tokens = self.total_steps * self.tokens_per_step
        if "rewarm_steps" in config:
            if not 1 <= int(config["rewarm_steps"]) < self.total_steps:
                raise ValueError("Rewarm steps must be inside the continuation budget.")
            if not 0 < config["initial_learning_rate"] <= config["learning_rate"] or not math.isfinite(config["learning_rate"]):
                raise ValueError("Continuation learning rates must be positive and finite, with initial <= peak.")
        self.seed = int(config["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.reset_peak_memory_stats()
        self.rng = np.random.default_rng(self.seed)

        self.manifest_path = Path(config["data_manifest"]).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if "status" in self.manifest and self.manifest["status"] != "complete":
            raise ValueError("Training requires a complete prepared-data manifest.")
        self.manifest_sha = sha256_file(self.manifest_path)
        model_config = json.loads(Path(config["model_config"]).read_text())
        if model_config.get("seed", self.seed) != self.seed:
            raise ValueError("Model initialization seed differs from the run seed.")
        model_config["seed"] = self.seed
        tok_meta = self.manifest["tokenizer"]
        tokenizer_path = Path(tok_meta["path"])
        if not tokenizer_path.is_absolute():
            tokenizer_path = self.manifest_path.parent / tokenizer_path
        self.tokenizer_path = tokenizer_path
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if tokenizer.get_vocab_size() != model_config["vocab_size"]:
            raise ValueError("Tokenizer/model vocabulary mismatch.")
        self.tokenizer_sha = sha256_file(tokenizer_path)
        if tok_meta.get("sha256") and self.tokenizer_sha != tok_meta["sha256"]:
            raise ValueError("Tokenizer checksum mismatch.")
        self.special_tokens = {}
        for kind in ("pad", "bos", "eos", "unk"):
            special = next((name for name in (f"<|{kind}|>", f"<{kind}>")
                            if tokenizer.token_to_id(name) is not None), None)
            if special is None:
                raise ValueError(f"Prepared tokenizer is missing its {kind} special token.")
            self.special_tokens[kind] = special
            if kind != "unk":
                model_config[f"{kind}_token_id"] = tokenizer.token_to_id(special)
        self.model_config = model_config
        if self.sequence_length > model_config["max_position_embeddings"]:
            raise ValueError("Training sequence length exceeds the model context.")
        self.model = build_model(model_config).to(self.device)
        self.parameter_count = count_parameters(self.model)
        self.model.train()
        if config.get("compile", False):
            raise ValueError("Compilation is disabled until separately validated; use the eager SDPA path.")

        self.streams = {}
        self.source_available_tokens = {}
        self.train_path_hashes = {}
        for source in self.sources:
            split = self.manifest["sources"][source]["splits"]["train"]
            paths = split.get("paths") or [split["path"]]
            paths = [str(Path(p) if Path(p).is_absolute() else self.manifest_path.parent / p) for p in paths]
            for p in paths:
                actual = sha256_file(p)
                if len(paths) == 1 and split.get("sha256") and split["sha256"] != actual:
                    raise ValueError(f"Training shard checksum mismatch: {source}")
                self.train_path_hashes[p] = actual
            self.streams[source] = SourceStream(paths, seq_len=self.sequence_length)
            self.source_available_tokens[source] = sum(Path(p).stat().st_size // 2 for p in paths) - 1

        decay, no_decay = [], []
        for p in self.model.parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
        optimizer_args = dict(lr=config["learning_rate"], betas=tuple(config["betas"]),
                              eps=config["adam_epsilon"])
        if self.device.type == "cuda":
            optimizer_args["fused"] = True
        self.optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": config["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ], **optimizer_args)
        self.parent_provenance = self.verify_parent() if config.get("parent_checkpoint") else None
        scientific = {key: config.get(key) for key in [
            "seed", "target_tokens", "sequence_length", "batch_sequences", "microbatch_sequences", "web_share",
            "learning_rate", "min_lr_ratio", "warmup_fraction", "betas", "adam_epsilon",
            "weight_decay", "gradient_clip", "arm", "controller", "max_source_epochs",
            "controller_settings", "controller_time_budget_fraction", "controller_time_budget_seconds",
            "max_wall_hours",
            "deterministic",
            "source_weights", "parent_checkpoint", "parent_checkpoint_sha256", "parent_run",
            "initial_learning_rate", "rewarm_steps", "milestone_cumulative_tokens",
            "inherit_data_state",
        ]}
        self.science_digest = hashlib.sha256(json.dumps({
            "training": scientific, "model": model_config,
            "data_sha256": self.manifest_sha, "tokenizer_sha256": self.tokenizer_sha,
            "train_shard_sha256": self.train_path_hashes, "parent": self.parent_provenance,
        }, sort_keys=True).encode()).hexdigest()
        if resume:
            original = json.loads((self.run_dir / "run.json").read_text())
            if original.get("science_digest") != self.science_digest:
                raise ValueError("Checkpoint belongs to a different scientific configuration/data manifest.")
            if original.get("runtime_fingerprint") != runtime_fingerprint():
                raise ValueError("Resume rejected: code or dependency fingerprint changed; create an explicit new experiment.")
            self.restore_checkpoint(resume)
        else:
            if self.parent_provenance:
                self.initialize_from_parent()
            self.write_run_record()
        self.log("initialized", parameter_count=self.parameter_count,
                 exact_target_tokens=self.exact_target_tokens, resumed=bool(resume),
                 device=str(self.device), science_digest=self.science_digest)

    def verify_parent(self):
        """Validate a completed local scratch ancestor, distinct from same-run resume."""
        parent = Path(self.config["parent_run"]).resolve()
        checkpoint = Path(self.config["parent_checkpoint"]).resolve()
        if checkpoint.parent != parent / "checkpoints" or parent == self.run_dir.resolve():
            raise ValueError("Continuation must name a checkpoint in a separate parent run.")
        record = json.loads((parent / "run.json").read_text())
        status = json.loads((parent / "status.json").read_text())
        if status["status"] != "completed" or status["main_tokens"] != record["exact_target_tokens"]:
            raise ValueError("Parent pretraining run must be complete.")
        if record["tokenizer_sha256"] != self.tokenizer_sha or record["model_config"] != self.model_config:
            raise ValueError("Continuation cannot change the parent architecture or tokenizer.")
        lineage = verify_scratch_lineage(parent)
        expected = self.config["parent_checkpoint_sha256"]
        if sha256_file(checkpoint) != expected:
            raise ValueError("Parent checkpoint integrity check failed.")
        for key in ("betas", "adam_epsilon", "weight_decay"):
            if record["config"][key] != self.config[key]:
                raise ValueError(f"Parent optimizer setting differs: {key}")
        if self.config.get("inherit_data_state"):
            if record["manifest_sha256"] != self.manifest_sha:
                raise ValueError("Inherited streams require the identical data manifest")
            for key in ("source_weights", "web_share", "sequence_length", "batch_sequences"):
                if record["config"].get(key) != self.config.get(key):
                    raise ValueError(f"Inherited data sampler setting differs: {key}")
            if list(record["config"].get("source_weights") or SOURCES) != list(self.sources):
                raise ValueError("Inherited source order differs")
        return {"run": str(parent), "checkpoint": str(checkpoint), "checkpoint_sha256": expected,
                "run_record_sha256": sha256_file(parent / "run.json"), "science_digest": record["science_digest"],
                "main_tokens": status["main_tokens"], "step": status["step"],
                "elapsed_seconds": status["elapsed_seconds"], "tokenizer_sha256": self.tokenizer_sha,
                "cumulative_tokens": lineage["cumulative_tokens"], "scratch_lineage": lineage["runs"]}

    def initialize_from_parent(self):
        parent = self.parent_provenance
        payload = torch.load(parent["checkpoint"], map_location="cpu", weights_only=False)
        if (payload["science_digest"] != parent["science_digest"] or payload["main_tokens"] != parent["main_tokens"]
                or payload["step"] != parent["step"]):
            raise ValueError("Parent payload does not match completed run provenance.")
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        # Preserve the previous behavior by default. Same-corpus continuations
        # may instead inherit stream cursors and sampler/RNG state explicitly.
        if self.config.get("inherit_data_state"):
            if set(payload["streams"]) != set(self.streams):
                raise ValueError("Inherited stream sources differ")
            for source, stream in self.streams.items():
                stream.load_state_dict(payload["streams"][source])
            self.rng.bit_generator.state = copy.deepcopy(payload["rng"])
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
            torch.set_rng_state(payload["torch_rng"])
            if self.device.type == "cuda":
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        self.model.train()
        del payload

    @property
    def parent_tokens(self):
        return self.parent_provenance.get("cumulative_tokens", self.parent_provenance["main_tokens"]) if self.parent_provenance else 0

    @property
    def elapsed(self):
        return self.previous_elapsed + time.monotonic() - self.started

    def write_run_record(self):
        versions = {}
        for name in ["torch", "transformers", "datasets", "tokenizers", "numpy", "safetensors", "lm_eval"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        source_hashes = {str(p): sha256_file(p) for p in sorted(Path("src/scglm").glob("*.py"))}
        record = {
            "config": self.config, "model_config": self.model_config,
            "parameter_count": self.parameter_count,
            "initialization": "continuation of own documented scratch checkpoint" if self.parent_provenance else "new random model weights; no external pretrained checkpoint",
            "parent": self.parent_provenance,
            "cumulative_target_tokens": self.exact_target_tokens + self.parent_tokens,
            "exact_target_tokens": self.exact_target_tokens,
            "token_unit": "loss-bearing next-token targets; separators included; no padding in packed training",
            "manifest_sha256": self.manifest_sha, "tokenizer_sha256": self.tokenizer_sha,
            "train_shard_sha256": self.train_path_hashes, "science_digest": self.science_digest,
            "source_sha256": source_hashes, "versions": versions,
            "runtime_fingerprint": runtime_fingerprint(),
            "torch_runtime": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "created_at_unix": time.time(),
            "ai_assistance": "Codex/ChatGPT assisted research, implementation, tests, and documentation.",
        }
        atomic_json(self.run_dir / "run.json", record)
        snapshot = self.run_dir / "snapshot"
        for directory in [Path("src/scglm"), Path("configs")]:
            for original in directory.glob("*"):
                if original.is_file():
                    target = snapshot / original
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, target)
        shutil.copy2("requirements.lock.txt", snapshot / "requirements.lock.txt")
        self.model.config.save_pretrained(self.run_dir / "model_config")

    def log(self, event, quiet=False, **fields):
        record = {"event": event, "time_unix": time.time(), "elapsed_seconds": self.elapsed,
                  "step": self.step, "main_tokens": self.main_tokens,
                  "probe_tokens": self.probe_tokens, "probe_seconds": self.probe_seconds,
                  "training_tokens_spent": self.training_tokens_spent,
                  "main_training_seconds": self.main_training_seconds,
                  "controller_seconds": self.controller_seconds,
                  "controller_probes_used": self.controller_probes_used,
                  "controller_windows_done": list(self.controller_windows_done),
                  "controller_panels_used": list(self.controller_panels_used), **fields}
        with open(self.event_path, "a", buffering=1) as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        if not quiet:
            print(json.dumps(record, allow_nan=False), flush=True)

    def state(self):
        return {
            "model": cpu_tree(self.model.state_dict()),
            "optimizer": cpu_tree(self.optimizer.state_dict()),
            "step": self.step, "main_tokens": self.main_tokens,
            "main_source_tokens": dict(self.main_source_tokens),
            "current_web_share": self.current_web_share,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "streams": {s: self.streams[s].state_dict() for s in self.sources},
            "science_digest": self.science_digest,
        }

    def restore_state(self, state):
        if state["science_digest"] != self.science_digest:
            raise ValueError("Checkpoint belongs to a different scientific configuration/data manifest.")
        self.model.load_state_dict(state["model"])
        # Adam may retain input tensor references (notably CPU step counters).
        # A branch must never mutate the saved state used by later branches.
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.step = state["step"]
        self.main_tokens = state["main_tokens"]
        self.main_source_tokens = dict(state["main_source_tokens"])
        self.current_web_share = state["current_web_share"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for source in self.sources:
            self.streams[source].load_state_dict(state["streams"][source])
        self.model.train()

    def save_checkpoint(self, reason="periodic"):
        start = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        checkpoints = self.run_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        # Different snapshots at one main step (before/after control) need
        # distinct paths so the previous pointer stays valid until replacement.
        path = checkpoints / f"step_{self.step:07d}_{time.time_ns()}.pt"
        temp = path.with_suffix(".tmp")
        payload = self.state()
        payload["accounting"] = {
            "elapsed_seconds": self.elapsed, "probe_tokens": self.probe_tokens,
            "probe_seconds": self.probe_seconds,
            "main_training_seconds": self.main_training_seconds,
            "training_tokens_spent": self.training_tokens_spent,
            "controller_seconds": self.controller_seconds,
            "controller_probes_used": self.controller_probes_used,
            "controller_windows_done": self.controller_windows_done,
            "controller_panels_used": self.controller_panels_used,
        }
        torch.save(payload, temp)
        with open(temp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, path)
        checksum = sha256_file(path)
        atomic_json(self.run_dir / "checkpoint_latest.json",
                    {"path": str(path.resolve()), "sha256": checksum, "step": self.step,
                     "main_tokens": self.main_tokens, "reason": reason})
        if reason.startswith("milestone_"):
            retained = self.run_dir / "milestones"
            retained.mkdir(exist_ok=True)
            destination = retained / f"{reason}.pt"
            if not destination.exists():
                os.link(path, destination)
                atomic_json(destination.with_suffix(".json"), {"path": str(destination.resolve()), "sha256": checksum,
                            "step": self.step, "main_tokens": self.main_tokens,
                            "cumulative_tokens": self.main_tokens + self.parent_tokens})
        old = sorted(checkpoints.glob("step_*.pt"))
        for obsolete in old[:-2]:
            obsolete.unlink()
        self.log("checkpoint", path=str(path), sha256=checksum, reason=reason,
                 duration_seconds=time.monotonic() - start)
        self.write_status("running")
        return path

    def restore_checkpoint(self, path):
        path = Path(path)
        if path.suffix == ".json":
            pointer = json.loads(path.read_text())
            path = Path(pointer["path"])
            if sha256_file(path) != pointer["sha256"]:
                raise ValueError("Checkpoint integrity check failed.")
        if path.resolve().parent != (self.run_dir / "checkpoints").resolve():
            raise ValueError("Resume checkpoint must belong to this original run directory.")
        # Only this project's own local, trusted checkpoint format is accepted.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.restore_state(payload)
        accounting = payload.get("accounting", {})
        self.previous_elapsed = accounting.get("elapsed_seconds", 0.0)
        self.probe_tokens = accounting.get("probe_tokens", 0)
        self.probe_seconds = accounting.get("probe_seconds", 0.0)
        self.controller_windows_done = accounting.get("controller_windows_done", [])
        self.controller_panels_used = accounting.get("controller_panels_used", [])
        for key in ("training_tokens_spent", "controller_probes_used", "main_training_seconds", "controller_seconds"):
            setattr(self, key, accounting.get(key, 0))
        # Retain the historical accounting schema while recovering elapsed cost.
        if self.event_path.exists():
            with self.event_path.open() as events:
                for line in events:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A process may die during its final append.
                    for key in ("probe_tokens", "probe_seconds", "main_training_seconds", "training_tokens_spent",
                                "controller_seconds", "controller_probes_used"):
                        setattr(self, key, max(getattr(self, key), event.get(key, 0)))
                    self.previous_elapsed = max(self.previous_elapsed, event.get("elapsed_seconds", 0.0))
                    self.controller_windows_done = sorted(set(self.controller_windows_done) |
                                                          set(event.get("controller_windows_done", [])))
                    self.controller_panels_used = sorted(set(self.controller_panels_used) |
                                                         set(event.get("controller_panels_used", [])))
        self.started = time.monotonic()

    def batch(self, web_share):
        if self.source_weights:
            choices = self.rng.choice(len(self.sources), self.batch_sequences, p=[self.source_weights[s] for s in self.sources])
            masks = [(s, choices == i) for i, s in enumerate(self.sources)]
            return self.batch_from_masks(masks)
        if not math.isfinite(web_share) or not 0 <= web_share <= 1:
            raise ValueError("Mixture share must be finite and in [0, 1].")
        choices = self.rng.random(self.batch_sequences) < web_share
        masks = [("web", choices), ("wiki", ~choices)]
        return self.batch_from_masks(masks)

    def batch_from_masks(self, masks):
        x = np.empty((self.batch_sequences, self.sequence_length), dtype=np.int64)
        y = np.empty_like(x)
        counts = {}
        # Check every source before advancing any stream.
        for source, mask in masks:
            n = int(mask.sum())
            counts[source] = n * self.sequence_length
            allowed = self.source_available_tokens[source] * self.config.get("max_source_epochs", 1.0)
            if self.streams[source].tokens_consumed + counts[source] > allowed:
                raise RuntimeError(f"Declared source epoch budget exceeded for {source}; do not silently repeat data.")
        for source, mask in masks:
            n = int(mask.sum())
            if n:
                bx, by = self.streams[source].next_batch(n)
                x[mask], y[mask] = bx, by
        if x.min() < 0 or y.min() < 0 or max(x.max(), y.max()) >= self.model_config["vocab_size"]:
            raise ValueError("Out-of-vocabulary training token.")
        return x, y, counts

    def update(self, web_share):
        started = time.monotonic()
        self.model.train()
        lr = learning_rate(self.step,
                           self.total_steps, self.config["learning_rate"],
                           self.config["warmup_fraction"], self.config["min_lr_ratio"])
        if "rewarm_steps" in self.config:
            step = self.step
            warm = int(self.config["rewarm_steps"])
            if not 1 <= warm < self.total_steps:
                raise ValueError("Rewarm steps must be inside the continuation budget.")
            peak, initial = self.config["learning_rate"], self.config["initial_learning_rate"]
            if step < warm:
                lr = initial + (peak-initial) * (step / max(1, warm-1))
            else:
                progress = (step-warm) / max(1, self.total_steps-warm-1)
                lr = peak * (self.config["min_lr_ratio"] + (1-self.config["min_lr_ratio"])*0.5*(1+math.cos(math.pi*min(1.0,progress))))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        x, y, counts = self.batch(web_share)
        self.optimizer.zero_grad(set_to_none=True)
        accum = self.batch_sequences // self.microbatch_sequences
        losses = []
        for offset in range(0, self.batch_sequences, self.microbatch_sequences):
            xt = torch.from_numpy(x[offset:offset+self.microbatch_sequences]).to(self.device)
            yt = torch.from_numpy(y[offset:offset+self.microbatch_sequences]).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                loss = forward_loss(self.model, xt, yt)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; aborting before optimizer update.")
            (loss / accum).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"],
                                             error_if_nonfinite=True)
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.monotonic() - started
        self.training_tokens_spent += self.tokens_per_step
        self.main_training_seconds += duration
        self.step += 1
        self.main_tokens += self.tokens_per_step
        for source in self.sources:
            self.main_source_tokens[source] += counts[source]
        self.log("optimizer_update_accounting", quiet=True, probe=False, duration_seconds=duration)
        return float(np.mean(losses)), float(norm), lr

    def documents(self, split_name, limit=None):
        for source in self.sources:
            split = self.manifest["sources"][source]["splits"].get(split_name)
            if split is None:
                raise ValueError(f"Missing split {source}/{split_name}")
            path = Path(split["jsonl_path"])
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            if split.get("jsonl_sha256") and sha256_file(path) != split["jsonl_sha256"]:
                raise ValueError(f"Development document checksum mismatch: {source}/{split_name}")
            with open(path) as f:
                for index, line in enumerate(f):
                    if limit is not None and index >= limit:
                        break
                    record = json.loads(line)
                    record.setdefault("source", source)
                    yield record

    def validate(self, split_name="monitor", limit=None, max_predictable_tokens=1024):
        from .evaluate import evaluate_documents, fixed_mixture_metrics
        started = time.monotonic()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            rows = evaluate_documents(self.model, self.documents(split_name, limit),
                                      context_length=self.sequence_length, stride=max(1, self.sequence_length//2),
                                      max_predictable_tokens=max_predictable_tokens, device=self.device)
        metrics = fixed_mixture_metrics(rows, weights=self.source_weights or {"web": 0.8, "wiki": 0.2})
        self.model.train()
        self.log("validation", split=split_name, metrics=metrics,
                 duration_seconds=time.monotonic()-started)
        return rows

    def write_status(self, status, **extra):
        atomic_json(self.run_dir / "status.json", {
            "status": status, "pid": os.getpid(), "step": self.step,
            "main_tokens": self.main_tokens, "target_tokens": self.exact_target_tokens,
            "probe_tokens": self.probe_tokens, "total_training_tokens": self.main_tokens+self.probe_tokens,
            "training_tokens_spent": self.training_tokens_spent,
            "controller_seconds": self.controller_seconds, "probe_seconds": self.probe_seconds,
            "controller_windows_done": self.controller_windows_done,
            "source_tokens": self.main_source_tokens, "web_share": self.current_web_share,
            "source_weights": self.source_weights,
            "parent_tokens": self.parent_tokens,
            "cumulative_tokens": self.main_tokens + self.parent_tokens,
            "cumulative_target_tokens": self.exact_target_tokens + self.parent_tokens,
            "elapsed_seconds": self.elapsed, "updated_unix": time.time(),
            "parameter_count": self.parameter_count,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if self.device.type=="cuda" else 0,
            **extra,
        })

    def export(self):
        from transformers import PreTrainedTokenizerFast
        from .model import save_model
        destination = self.run_dir / "export"
        save_model(self.model, destination)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.tokenizer_path),
                    **{f"{kind}_token": value for kind, value in self.special_tokens.items()},
                    model_max_length=self.sequence_length)
        if len(tokenizer) != self.model.config.vocab_size:
            raise ValueError("Export would change the trained tokenizer vocabulary.")
        for kind in ("pad", "bos", "eos"):
            if getattr(tokenizer, f"{kind}_token_id") != getattr(self.model.config, f"{kind}_token_id"):
                raise ValueError(f"Export {kind} token ID differs from the trained model.")
        tokenizer.save_pretrained(destination)
        atomic_json(destination / "training_provenance.json", {
            "run": str(self.run_dir.resolve()), "step": self.step,
            "main_tokens": self.main_tokens, "probe_tokens": self.probe_tokens,
            "training_tokens_spent": self.training_tokens_spent,
            "science_digest": self.science_digest, "tokenizer_sha256": self.tokenizer_sha,
            "initialization": "scratch", "source_token_counts": self.main_source_tokens,
            "parent": self.parent_provenance,
            "cumulative_tokens": self.main_tokens + self.parent_tokens,
        })
        self.log("export", path=str(destination))

    def run(self, max_steps=None, max_seconds=None):
        limit = min(self.total_steps, max_steps) if max_steps is not None else self.total_steps
        stop_at = self.config.get("max_wall_hours", 30) * 3600
        invocation_start = time.monotonic()
        previous_log_time = time.monotonic()
        previous_log_tokens = self.main_tokens
        last_loss = None
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        try:
            self.write_status("running")
            while self.step < limit:
                if self.stop_requested or self.elapsed >= stop_at:
                    break
                if max_seconds and time.monotonic() - invocation_start >= max_seconds:
                    break
                tick = time.monotonic()
                loss, norm, lr = self.update(self.current_web_share)
                last_loss = loss
                if self.step % self.config.get("log_every", 10) == 0 or self.step == 1:
                    now = time.monotonic()
                    rate = (self.main_tokens-previous_log_tokens)/max(1e-9, now-previous_log_time)
                    self.log("train", loss=loss, gradient_norm=norm, learning_rate=lr,
                             web_share=self.current_web_share, tokens_per_second=rate,
                             update_seconds=now-tick, source_tokens=dict(self.main_source_tokens))
                    self.write_status("running", loss=loss, recent_tokens_per_second=rate)
                    previous_log_time, previous_log_tokens = now, self.main_tokens
                validation_every = self.config.get("validation_every", 0)
                if validation_every and self.step % validation_every == 0:
                    # Monitoring only; no hyperparameter tuning or checkpoint selection mid-run.
                    self.validate(limit=self.config.get("validation_documents_per_source", 32))
                cumulative = self.main_tokens + self.parent_tokens
                milestones = [m for m in self.config.get("milestone_cumulative_tokens", []) if cumulative-self.tokens_per_step < m <= cumulative]
                if milestones:
                    self.save_checkpoint(reason=f"milestone_{int(milestones[-1])}")
                elif self.step % self.config.get("checkpoint_every", 250) == 0:
                    self.save_checkpoint()
            self.save_checkpoint(reason="complete" if self.step == self.total_steps else "paused")
            complete = self.step == self.total_steps
            if complete:
                self.export()
            self.write_status("completed" if complete else "paused", loss=last_loss)
            self.log("finished", complete=complete, total_training_tokens=self.main_tokens+self.probe_tokens)
        except Exception as exc:
            self.log("error", error=repr(exc))
            self.write_status("failed", error=repr(exc))
            raise



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train_baseline.json")
    p.add_argument("--run-dir")
    p.add_argument("--data-manifest")
    p.add_argument("--target-tokens", type=int)
    p.add_argument("--microbatch", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-validation", action="store_true")
    args = p.parse_args()
    config = read_config(args.config)
    for arg, key in [(args.run_dir, "run_dir"), (args.data_manifest, "data_manifest"),
                     (args.target_tokens, "target_tokens"), (args.microbatch, "microbatch_sequences")]:
        if arg is not None:
            config[key] = arg
    if args.no_validation:
        config["validation_every"] = 0
    trainer = Trainer(config, device=args.device, resume=args.resume)
    trainer.run(max_steps=args.max_steps, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
