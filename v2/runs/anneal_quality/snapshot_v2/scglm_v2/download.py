"""Download the pinned new-source files into the shared Hugging Face cache.

Resumable: already-cached files return immediately. Order is StackExchange and
books first (needed by the tokenizer study), then FineWeb-Edu and DCLM.
Run: PYTHONPATH=src:v2/src python -m scglm_v2.download --config v2/configs/prep.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

from huggingface_hub import hf_hub_download

from .common import ROOT, event, load_config, new_source_files, write_json

PRIORITY = ("stackexchange", "books", "edu", "dclm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="v2/configs/prep.json")
    args = parser.parse_args()
    config = load_config(args.config)
    out = ROOT / config["output_dir"] / "downloads"
    log = out / "events.jsonl"
    plan = new_source_files(config)
    write_json(out / "plan.json", {name: plan[name] for name in PRIORITY})
    jobs = [(name, f) for name in PRIORITY for f in plan[name]]
    event(log, "download_start", files={k: len(v) for k, v in plan.items()}, hf_home=os.environ.get("HF_HOME"))
    done: dict[str, dict] = {}
    started = time.monotonic()

    def fetch(name: str, filename: str) -> tuple[str, str, str, int]:
        spec = config["sources"][name]
        for attempt in range(5):
            try:
                path = hf_hub_download(spec["repo"], filename, repo_type="dataset", revision=spec["revision"])
                return name, filename, path, os.path.getsize(path)
            except Exception as exc:  # network retries; final failure is recorded
                if attempt == 4:
                    raise
                event(log, "download_retry", source=name, file=filename, error=str(exc)[:300])
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("unreachable")

    failures = []
    with ThreadPoolExecutor(max_workers=config["download_workers"]) as pool:
        futures = {pool.submit(fetch, name, f): (name, f) for name, f in jobs}
        for future in as_completed(futures):
            name, filename = futures[future]
            try:
                _, _, path, size = future.result()
                done[f"{name}:{filename}"] = {"source": name, "file": filename, "path": path, "bytes": size}
            except Exception as exc:
                failures.append({"source": name, "file": filename, "error": str(exc)[:500]})
            write_json(out / "status.json", {"state": "downloading", "completed": len(done), "total": len(jobs),
                                              "failures": failures, "elapsed_s": round(time.monotonic() - started),
                                              "bytes": sum(d["bytes"] for d in done.values())})
    state = "complete" if not failures else "failed"
    write_json(out / "status.json", {"state": state, "completed": len(done), "total": len(jobs), "failures": failures,
                                      "elapsed_s": round(time.monotonic() - started),
                                      "bytes": sum(d["bytes"] for d in done.values())})
    write_json(out / "files.json", sorted(done.values(), key=lambda d: (d["source"], d["file"])))
    event(log, "download_" + state, completed=len(done), failures=len(failures))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
