"""Shared helpers for the V2 rich-data preparation phase."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random

from scglm.prepare_data import sha256_file, write_json  # noqa: F401  (re-exported)

ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text() if not Path(path).is_absolute() else Path(path).read_text())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(log_path: Path, name: str, **values) -> None:
    """Append one dated event line and echo it to stdout."""
    record = {"time": now(), "event": name, **values}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def pinned_files(spec: dict, seed: int) -> list[str]:
    """Replicate the V1 extension's seeded file order for a pinned repository."""
    meta = json.loads((ROOT / spec["metadata_file"]).read_text())
    if meta["revision"] != spec["revision"]:
        raise ValueError(f"Pinned metadata revision differs for {spec['repo']}")
    files = sorted(f["path"] for f in meta["files"]
                   if f["path"].startswith(spec["file_prefix"]) and f["path"].endswith((".parquet", ".jsonl.zst")))
    random.Random(seed).shuffle(files)
    return files


def new_source_files(config: dict) -> dict[str, list[str]]:
    """Files each source must download, in deterministic priority order."""
    from huggingface_hub import HfApi
    out: dict[str, list[str]] = {}
    seed = config["seed"]
    for name in ("edu", "dclm"):
        spec = config["sources"][name]
        order = pinned_files(spec, seed)
        start = spec["extension_files_consumed"]
        out[name] = order[start:start + spec["new_files"]]
    api = HfApi()
    se = config["sources"]["stackexchange"]
    info = api.dataset_info(se["repo"], revision=se["revision"])
    allowed = {f"{site}.stackexchange.com" for site in se["site_allowlist"]}
    out["stackexchange"] = sorted(s.rfilename for s in info.siblings
                                  if s.rfilename.split("/")[0] in allowed and s.rfilename.endswith(".parquet"))
    books = config["sources"]["books"]
    info = api.dataset_info(books["repo"], revision=books["revision"])
    files = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.startswith(books["file_prefix"]) and s.rfilename.endswith(".parquet"))
    random.Random(seed).shuffle(files)
    out["books"] = files[:books["new_files"]]
    return out
