"""Immutable contracts, identities, and local execution helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

from scglm_post.common import ROOT, read_jsonl, sha256 as _sha256, write_json, write_jsonl


def sha256(path):
    return _sha256(Path(path))


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def check_file(path, expected):
    path = resolve(path)
    if sha256(path) != expected:
        raise ValueError(f"File identity changed: {path}")
    return path






def load_corpora(config):
    result = {}
    for name, spec in config["corpora"].items():
        path = check_file(spec["manifest"], spec["sha256"])
        data = read(path)
        if data["status"] != "complete" or set(spec["weights"]) != set(data["sources"]):
            raise ValueError("Corpus incomplete or source set changed")
        result[name] = (data, spec)
    return result


def raw_panels(config, phase):
    """Namespaces distinguish legacy/wiki from continuation/wiki."""
    rows = []
    for name, (data, spec) in load_corpora(config).items():
        split = "final" if phase == "confirmation" else "selection"
        for source in spec["weights"]:
            entry = data["sources"][source]["splits"][split]
            path = check_file(entry["jsonl_path"], entry["jsonl_sha256"])
            for row in read_jsonl(path):
                rows.append({**row, "id": f"{name}:{row['id']}", "source": f"{name}/{source}"})
    return rows
