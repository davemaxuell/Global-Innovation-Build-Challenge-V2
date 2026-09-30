"""Durable local audit records and metric checks for the validation runner."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import uuid


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    """Publish a complete JSON file atomically, including a durable file flush."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_event(path: Path, record: dict) -> None:
    """Serialize concurrent writers; never truncate earlier validation events."""
    line = json.dumps(record, allow_nan=False) + "\n"
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)


class ValidationRun:
    """Record an attempt before configuration checks, downloads, or inference.

    SIGINT/SIGTERM produce an interrupted record. An uncatchable SIGKILL or
    machine failure leaves a durable running record, never a false success.
    """

    def __init__(self, history: Path, *, output: Path | None, metadata: dict):
        self.history = history.resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = f"{stamp}-{uuid.uuid4().hex[:12]}"
        self.directory = self.history / "runs" / self.run_id
        self.output = output.resolve() if output is not None else self.directory / "results"
        self.record = {
            "schema_version": 1, **metadata, "run_id": self.run_id,
            "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "started_unix": time.time(), "hostname": socket.gethostname(), "pid": os.getpid(),
            "record_path": str(self.directory / "run.json"), "output": str(self.output),
            "stages": [], "metrics": {},
        }

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.update()
        self.event("started", **self.record)
        self.old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._terminate)
        return self

    @staticmethod
    def _terminate(signum, frame):
        raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")

    def update(self, **fields):
        self.record.update(fields)
        write_json(self.directory / "run.json", self.record)

    def event(self, event: str, **fields):
        append_event(self.history / "history.jsonl", {
            **fields, "event": event, "run_id": self.run_id, "time_unix": time.time(),
        })

    def claim_output(self):
        # A new directory is mandatory even after an unsuccessful earlier run.
        self.output.mkdir(parents=True, exist_ok=False)

    def run_stage(self, name: str, specification: dict, environment: dict):
        log_path = self.directory / f"{name}.log"
        stage = {"name": name, **specification, "status": "running",
                 "started_unix": time.time(), "log": str(log_path)}
        self.record["stages"].append(stage)
        self.update()
        self.event("stage_started", stage=stage)
        print(f"[{self.run_id}] {name}: {log_path}", flush=True)
        process = None
        started = time.monotonic()
        try:
            with log_path.open("x") as log:
                process = subprocess.Popen(
                    specification["command"], cwd=specification["cwd"], env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                stage["pid"] = process.pid
                self.update()
                code = process.wait()
                stage["returncode"] = code
                if code:
                    raise subprocess.CalledProcessError(code, specification["command"])
            stage["status"] = "completed"
        except BaseException as error:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stage["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            stage["returncode"] = process.returncode if process is not None else None
            raise
        finally:
            stage.update(finished_unix=time.time(), elapsed_seconds=time.monotonic() - started)
            self.update()
            self.event("stage_finished", stage=stage)

    def __exit__(self, error_type, error, tb):
        signal.signal(signal.SIGTERM, self.old_sigterm)
        status = "prepared" if self.record.get("mode") == "prepare" else "completed"
        if error is not None:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            self.record["error"] = {"type": type(error).__name__, "message": str(error)}
            (self.directory / "error.log").write_text("".join(traceback.format_exception(error_type, error, tb)))
        self.update(status=status, finished_at=datetime.now(timezone.utc).isoformat(),
                    finished_unix=time.time(), elapsed_seconds=time.monotonic() - self.started)
        self.event("finished", **self.record)
        print(f"Validation {status}: {self.directory / 'run.json'}", flush=True)
        return False


def _number(value, name: str, *, minimum: float = 0, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite metric: {name}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Out-of-range metric: {name}")
    return float(value)


def harness_metrics(output: Path, config: dict) -> dict:
    paths = sorted((output / "harness").rglob("results_*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one harness results file, found {len(paths)}")
    result = json.loads(paths[0].read_text())
    metrics = {}
    for task, definition in config["harness"]["task_definitions"].items():
        counts = result["n-samples"][task]
        if counts["effective"] != definition["expected_examples"]:
            raise ValueError(f"Incomplete harness evaluation: {task}: {counts}")
        if result["n-shot"][task] != config["harness"]["num_fewshot"]:
            raise ValueError(f"Unexpected few-shot setting: {task}")
        row = {"split": definition["evaluation_split"], "examples": counts["effective"]}
        for metric in definition["metrics"]:
            row[metric] = _number(result["results"][task].get(f"{metric},none"),
                                  f"{task}/{metric}", maximum=1)
            stderr = result["results"][task].get(f"{metric}_stderr,none")
            if stderr is not None and stderr != "N/A":
                row[f"{metric}_stderr"] = _number(stderr, f"{task}/{metric}_stderr")
        metrics[task] = row
    return {"benchmarks": metrics, "harness_results": str(paths[0]),
            "harness_results_sha256": sha256(paths[0])}


def wikitext_metrics(output: Path, config: dict) -> dict:
    path = output / "wikitext103.json"
    report = json.loads(path.read_text())
    expected = config["wikitext103"]
    metadata = report["metadata"]
    for key in ("revision", "split", "context_length", "stride"):
        if metadata[key] != expected[key]:
            raise ValueError(f"WikiText-103 metadata mismatch: {key}")
    if metadata["configuration"] != "wikitext-103-raw-v1" or metadata["dataset"] != expected["dataset"]:
        raise ValueError("Wrong WikiText dataset")
    source = report["metrics"]["sources"]["wikitext103"]
    tokens = _number(source["token_count"], "wikitext103/token_count", minimum=1)
    documents = _number(source["documents"], "wikitext103/documents", minimum=1)
    if int(tokens) != tokens or int(documents) != documents:
        raise ValueError("WikiText counts must be integers")
    nll = _number(source["nll"], "wikitext103/nll")
    total = _number(source["nll_sum"], "wikitext103/nll_sum")
    perplexity = _number(source["perplexity"], "wikitext103/perplexity", minimum=1)
    if not math.isclose(nll, total / tokens) or not math.isclose(math.log(perplexity), nll):
        raise ValueError("WikiText-103 NLL/perplexity aggregation mismatch")
    return {"wikitext103": {**source, "split": metadata["split"],
                            "context_length": metadata["context_length"], "stride": metadata["stride"]},
            "wikitext103_results": str(path), "wikitext103_results_sha256": sha256(path)}


def main():
    """Display one current record per attempt, including unfinished attempts."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the local validation history")
    parser.add_argument("--history-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts/validation")
    parser.add_argument("--json", action="store_true", help="Print complete current records as JSON")
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in sorted((args.history_dir / "runs").glob("*/run.json"))]
    if args.json:
        print(json.dumps(records, indent=2, allow_nan=False))
        return
    print("run_id\tstatus\tHellaSwag acc_norm\tARC acc_norm\tPIQA acc_norm\tWinoGrande acc\tWikiText PPL\tlabel")
    for record in records:
        metrics = record.get("metrics", {})
        tasks = metrics.get("benchmarks", {})
        values = [tasks.get(name, {}).get(metric) for name, metric in (
            ("hellaswag", "acc_norm"), ("arc_easy", "acc_norm"), ("piqa", "acc_norm"), ("winogrande", "acc"))]
        values.append(metrics.get("wikitext103", {}).get("perplexity"))
        label = str(record.get("label", "")).replace("\t", " ").replace("\n", " ")
        print("\t".join([record["run_id"], record["status"],
                         *("-" if v is None else f"{v:.6f}" for v in values), label]))


if __name__ == "__main__":
    main()
