"""Document readers for every V2 source. Each yields dicts with id/url/title/text.

Reused V1 extension text has already passed exact/URL/MinHash dedup and the
13-word benchmark exclusion. New sources pass the same checks in ``build``.
"""
from __future__ import annotations

import glob
import io
import json
import os
from pathlib import Path
import random
import re
from urllib.parse import urlsplit

import pyarrow.parquet as pq
import zstandard

from .common import ROOT

HF_HUB = Path(os.environ.get("HF_HUB_CACHE", "/data/shared/hf_cache/hub"))
_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", re.I)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)
_SE_ASKED = re.compile(r"^\(Asked by: username_\d+ on [^)]*\)\s*$", re.M)
_SE_ANSWER = re.compile(r"^### Answer by username_\d+ on [^\n]*$", re.M)
_SE_USER = re.compile(r"\*\*username_\d+\*\*:?")


def host(url: str | None) -> str:
    if not url:
        return ""
    try:
        return urlsplit(url).netloc.casefold().removeprefix("www.")
    except ValueError:
        return ""


def blocked(url: str | None, blocklist: list[str]) -> bool:
    h = host(url)
    return any(h == d or h.endswith("." + d) for d in blocklist)


def snapshot_files(repo: str, revision: str, pattern: str) -> list[Path]:
    base = HF_HUB / f"datasets--{repo.replace('/', '--')}" / "snapshots" / revision
    return sorted(Path(p) for p in glob.glob(str(base / pattern)))


# ---------------------------------------------------------------- reused text
def reused_raw_paths(config: dict, source: str) -> list[Path]:
    spec = config["sources"][source]
    paths = []
    if spec.get("reuse_extension_source"):
        ext = Path(json.loads((ROOT / config["reuse"]["extension_manifest"]).read_text())
                   ["sources"][spec["reuse_extension_source"]]["splits"]["train"]["path"]).with_name("train.raw.jsonl")
        paths.append(ext)
    if spec.get("reuse_main_raw"):
        paths.append(ROOT / spec["reuse_main_raw"])
    return paths


def iter_jsonl(path: Path, source: str):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            yield {"id": row.get("id") or row.get("source_id"), "url": row.get("url"), "title": row.get("title") or "",
                   "text": row["text"], "source": source, "origin": str(path.name)}


def sample_jsonl(path: Path, count: int, seed: int, exclude_ids: set | None = None):
    """Random byte-offset sampling (length-weighted, i.e. roughly token-weighted)."""
    size, rng, seen = path.stat().st_size, random.Random(seed), set()
    with path.open("rb") as handle:
        while len(seen) < count:
            handle.seek(rng.randrange(size))
            handle.readline()
            line = handle.readline()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("id") or row.get("content_sha256")
            if key in seen or (exclude_ids and key in exclude_ids):
                continue
            seen.add(key)
            yield {"id": key, "url": row.get("url"), "title": row.get("title") or "", "text": row["text"]}


# ---------------------------------------------------------------- new sources
def clean_stackexchange(text: str) -> str:
    text = _SE_ASKED.sub("", text)
    text = _SE_ANSWER.sub("### Answer", text)
    text = _SE_USER.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def iter_stackexchange(config: dict, files: list[Path] | None = None):
    spec = config["sources"]["stackexchange"]
    files = files if files is not None else snapshot_files(spec["repo"], spec["revision"], "*/*.parquet")
    for path in files:
        site = path.parent.name
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024, columns=["Id", "Score", "AnswerCount", "Title", "ThreadText"]):
            for row in batch.to_pylist():
                if (row["Score"] or 0) < 1 or (row["AnswerCount"] or 0) < 1 or not row["ThreadText"]:
                    continue
                yield {"id": f"{site}:{row['Id']}", "url": f"https://{site}/questions/{row['Id']}",
                       "title": row["Title"] or "", "text": clean_stackexchange(row["ThreadText"]),
                       "source": "stackexchange", "origin": site}


def clean_gutenberg(text: str) -> str:
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[:end.start()]
    return text.replace("\r\n", "\n").strip()


def chunk_text(text: str, size: int) -> list[str]:
    """Split at paragraph boundaries into pieces of roughly ``size`` characters."""
    pieces, current, length = [], [], 0
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if length + len(para) > size and current:
            pieces.append("\n\n".join(current)); current, length = [], 0
        current.append(para); length += len(para) + 2
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def iter_books(config: dict, files: list[Path] | None = None):
    spec = config["sources"]["books"]
    files = files if files is not None else snapshot_files(spec["repo"], spec["revision"], "data/en-*.parquet")
    for path in files:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        text_col = "text" if "text" in names else names[-1]
        id_col = "id" if "id" in names else None
        for batch in pf.iter_batches(batch_size=16, columns=[c for c in (id_col, text_col) if c]):
            for i, row in enumerate(batch.to_pylist()):
                book_id = str(row.get(id_col)) if id_col else f"{path.name}:{i}"
                body = clean_gutenberg(row[text_col] or "")
                for j, piece in enumerate(chunk_text(body, spec["chunk_chars"])):
                    if len(piece) >= 1000:
                        yield {"id": f"gutenberg:{book_id}:{j}", "url": None, "title": "", "text": piece,
                               "source": "books", "origin": path.name}


def iter_dclm_new(config: dict, files: list[Path]):
    for path in files:
        with open(path, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader, \
                io.TextIOWrapper(reader, encoding="utf-8") as f:
            for n, line in enumerate(f):
                row = json.loads(line)
                yield {"id": row.get("id") or f"{path.name}:{n}", "url": row.get("url"), "title": "",
                       "text": row.get("text", ""), "source": "dclm", "origin": path.name}
