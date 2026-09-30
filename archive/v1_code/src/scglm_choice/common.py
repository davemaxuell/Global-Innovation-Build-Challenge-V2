"""Study contracts, immutable records and phase documentation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path

from scglm_pipeline.common import ROOT, digest, read, read_jsonl, resolve, sha256, write_json, write_jsonl

CONFIG = ROOT / "configs/choice_discrimination.json"
DOMAINS = ("science", "commonsense")
ARMS = ("ce_control", "choice_candidate")
SEEDS = (20260929, 20260930, 20261001)


class StudyStopped(ValueError):
    """A registered gate prevents further study execution."""


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    value = read(CONFIG)
    protocol = ROOT / "reports/research/2026-09-29_choice_discrimination_protocol.json"
    if sha256(protocol) != value["protocol_json_sha256"]:
        raise ValueError("Registered protocol changed")
    registered = read(protocol)
    for key in ("objectives", "optimizer", "budget", "evaluation", "data", "parent", "base_reference", "isolation"):
        if value[key] != registered[key]:
            raise ValueError(f"Fixed protocol changed: {key}")
    if value["transfer_diagnostic_only"] is not True:
        raise ValueError("Transfer diagnostics cannot become selection gates")
    for name, expected in value["retained_files"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Retained implementation changed: {name}")
    return value


def roots(cfg):
    paths = [ROOT / cfg["isolation"][key] for key in ("proposed_data_root", "proposed_output_root")]
    for path in paths:
        # Resolve every ancestor, not just the final component.
        if path.resolve() != path.absolute():
            raise ValueError("Study roots cannot contain symlinks")
        if path.exists() and any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError("Study output/data descendants cannot contain symlinks")
    return tuple(paths)


def within(path, root):
    path, root = Path(path), Path(root)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output escapes isolated study directory")
    return path


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def event(directory, name, **fields):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at": now(), "event": name, **fields}, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if digest(read(path)) != digest(value):
            raise ValueError(f"Frozen record changed: {path}")
    else:
        write_json(path, value)
    return value


def sources():
    paths = [*sorted((ROOT / "src/scglm_choice").glob("*.py")),
             *sorted((ROOT / "src/scglm_pipeline").glob("*.py")),
             *sorted((ROOT / "src/scglm_post").glob("*.py")),
             *sorted((ROOT / "src/scglm").glob("*.py")),
             ROOT / "scripts/run_choice_pipeline.py", CONFIG,
             ROOT / "configs/evaluation.json", ROOT / "requirements.lock.txt",
             ROOT / "scripts/run_final_evaluation.py", ROOT / "scripts/validation_history.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def runtime():
    return {n: importlib.metadata.version(n) for n in
            ("torch", "transformers", "tokenizers", "numpy", "safetensors", "datasets")}


def register(cfg, phase, objective, *, data=None, recipe=None):
    """Always called before work, including preparation and skipped phases."""
    _, output = roots(cfg)
    directory = output / "phases" / phase
    directory.mkdir(parents=True, exist_ok=True)
    record = {"phase": phase, "objective": objective, "parent": cfg["parent"],
              "base": cfg["base_reference"], "data": data,
              "recipe": recipe or cfg["optimizer"], "budget": cfg["budget"],
              "selection": cfg["evaluation"], "transfer_diagnostic_only": True,
              "output": str(output), "config_sha256": digest(cfg),
              "launch_command": f"PYTHONPATH=src:. OMP_NUM_THREADS=4 /home/bufsgpu/yes/envs/sw/bin/python scripts/run_choice_pipeline.py {'prepare' if phase == 'preparation' else 'run'}",
              "resume_command": "PYTHONPATH=src:. OMP_NUM_THREADS=4 /home/bufsgpu/yes/envs/sw/bin/python scripts/run_choice_pipeline.py resume"}
    immutable(directory / "registration.json", record)
    report = directory / "REPORT.md"
    if not report.exists():
        report.write_text(f"# Choice study: {phase}\n\nStatus: preparation.\n\n{objective}\n\n"
                          f"Parent: `{cfg['parent']['path']}`, SHA256 `{cfg['parent']['weight_sha256']}`.\n\n"
                          "The complete frozen data identity, recipe, compute limits, selection rules, output and commands are in [registration.json](registration.json).\n\n"
                          "[Status](status.json) · [Dated events](events.jsonl)\n")
    marker = f"choice_discrimination_20260929_v1/{phase}"
    with lock(ROOT / "artifacts/choice_documentation.lock"):
        index = ROOT / "TRAINING_PHASES.md"
        if marker not in index.read_text():
            with index.open("a") as stream:
                stream.write(f"\n## Choice-discrimination phase: {marker}\n\n"
                             f"Registered {now()} before execution. {objective}\n\n"
                             f"[Phase report]({report.relative_to(ROOT)}) · "
                             f"[Registration]({(directory / 'registration.json').relative_to(ROOT)})\n")
    if not (directory / "status.json").exists():
        status(directory, "preparation", optimizer_updates=0, counters={})
    return directory


def status(directory, state, **fields):
    if state not in ("preparation", "running", "paused", "failed", "completed", "skipped"):
        raise ValueError("Unknown study state")
    write_json(Path(directory) / "status.json", {"status": state, "at": now(), **fields})
    event(directory, state, **fields)


def finish_phase(directory, state, **fields):
    status(directory, state, **fields)
    with (Path(directory) / "REPORT.md").open("a") as stream:
        stream.write(f"\n## {now()}: {state}\n\n```json\n{json.dumps(fields, indent=2)}\n```\n")
