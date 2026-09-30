"""Study configuration, isolation, phase registration and status records.

Both follow-up studies register every phase in ``TRAINING_PHASES.md`` with a
linked report before any work starts, as required by ``AGENTS.md``. Generic
persistence helpers are shared with the completed choice study.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from scglm_choice.common import StudyStopped, event, immutable, lock, now
from scglm_pipeline.common import ROOT, digest, read, read_jsonl, sha256, write_json, write_jsonl

__all__ = ["ROOT", "StudyStopped", "digest", "event", "immutable", "lock", "now", "read", "read_jsonl",
           "sha256", "write_json", "write_jsonl", "load_config", "roots", "within", "sources", "runtime",
           "register", "status", "finish_phase"]

STATES = ("preparation", "running", "paused", "failed", "completed", "skipped", "stopped")
PYTHON = "/home/bufsgpu/yes/envs/sw/bin/python"


def load_config(path):
    """Read a study configuration and check every pinned input it names."""
    path = Path(path)
    cfg = read(path)
    cfg["_config_path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    for name, expected in cfg.get("retained_files", {}).items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Retained implementation or record changed: {name}")
    return cfg


def roots(cfg):
    paths = [ROOT / cfg["isolation"][key] for key in ("data_root", "output_root")]
    for path in paths:
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


def sources(cfg):
    """Hashes of every file whose change must invalidate a resume."""
    packages = cfg["isolation"]["source_packages"]
    paths = [p for package in packages for p in sorted((ROOT / "src" / package).glob("*.py"))]
    paths += [ROOT / cfg["script"], ROOT / cfg["_config_path"], ROOT / "requirements.lock.txt"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def runtime():
    return {n: importlib.metadata.version(n) for n in
            ("torch", "transformers", "tokenizers", "numpy", "safetensors")}


def config_digest(cfg):
    return digest({k: v for k, v in cfg.items() if not k.startswith("_")})


def register(cfg, phase, objective, *, data=None, recipe=None, parent=None):
    """Create the phase record and journal entry before the phase does any work."""
    _, output = roots(cfg)
    directory = output / "phases" / phase
    directory.mkdir(parents=True, exist_ok=True)
    command = f"PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 {PYTHON} {cfg['script']}"
    parent = parent or cfg["parent"]
    record = {"experiment": cfg["experiment_id"], "phase": phase, "objective": objective,
              "parent": parent, "data": data, "recipe": recipe, "budget": cfg["budget"],
              "selection": cfg["evaluation"], "output": str(output), "config": cfg["_config_path"],
              "config_sha256": config_digest(cfg), "launch_command": f"{command} run",
              "resume_command": f"{command} resume", "registered_at": now()}
    # The registration time is informational; the frozen fields are compared.
    path = directory / "registration.json"
    if path.exists():
        previous = read(path)
        if {k: v for k, v in previous.items() if k != "registered_at"} != {k: v for k, v in record.items() if k != "registered_at"}:
            raise ValueError(f"Frozen phase registration changed: {path}")
    else:
        write_json(path, record)
    report = directory / "REPORT.md"
    if not report.exists():
        weight = parent.get("weight_sha256") or parent.get("checkpoint_sha256")
        report.write_text(f"# {cfg['title']}: {phase}\n\nStatus: preparation.\n\n{objective}\n\n"
                          f"Parent: `{parent.get('path') or parent.get('checkpoint')}`, SHA256 `{weight}`.\n\n"
                          "The frozen data identity, recipe, compute limits, selection rules, output and "
                          "launch/resume commands are in [registration.json](registration.json).\n\n"
                          "[Status](status.json) · [Dated events](events.jsonl)\n")
    marker = f"{cfg['experiment_id']}/{phase}"
    with lock(ROOT / "artifacts/study_documentation.lock"):
        index = ROOT / "TRAINING_PHASES.md"
        if marker not in index.read_text():
            with index.open("a") as stream:
                stream.write(f"\n## {cfg['title']} phase: {marker}\n\n"
                             f"Registered {now()} before execution. {objective}\n\n"
                             f"[Phase report]({report.relative_to(ROOT)}) · "
                             f"[Registration]({path.relative_to(ROOT)})\n")
    if not (directory / "status.json").exists():
        status(directory, "preparation", optimizer_updates=0)
    return directory


def status(directory, state, **fields):
    if state not in STATES:
        raise ValueError("Unknown study state")
    write_json(Path(directory) / "status.json", {"status": state, "at": now(), **fields})
    event(directory, state, **fields)


def finish_phase(directory, state, **fields):
    status(directory, state, **fields)
    with (Path(directory) / "REPORT.md").open("a") as stream:
        stream.write(f"\n## {now()}: {state}\n\n```json\n{json.dumps(fields, indent=2, default=str)}\n```\n")


def append_failed_approach(marker, title, outcome, reason, report):
    """Record a non-qualifying study once, next to earlier unsuccessful approaches."""
    record = ROOT / "FAILED_APPROACHES.md"
    # Match the generated heading exactly; other entries may mention the marker.
    if f"({marker})\n" in record.read_text():
        return
    with record.open("a") as stream:
        stream.write(f"\n## {title} ({marker})\n\n"
                     f"Outcome: **{outcome}**. Stop: `{reason}`. [Evidence and actual counts]({report}). "
                     "No selected checkpoint was replaced and no further recipe is scheduled automatically.\n")
