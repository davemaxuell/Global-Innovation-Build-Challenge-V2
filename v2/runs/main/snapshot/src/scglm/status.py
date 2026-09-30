"""Read a local run without importing PyTorch or occupying a GPU."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="runs/baseline_1b")
    args = parser.parse_args()
    directory = Path(args.run_dir)
    path = directory / "status.json"
    if not path.exists():
        raise SystemExit(f"No training status yet: {path}")
    state = json.loads(path.read_text())
    pid = state.get("pid")
    try:
        os.kill(pid, 0)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        alive = "scglm.train" in cmdline
    except (OSError, TypeError):
        alive = False
    age = time.time() - state["updated_unix"]
    state.update(process_alive=alive, status_age_seconds=round(age, 1),
                 updated_utc=datetime.fromtimestamp(state["updated_unix"], timezone.utc).isoformat())
    if state["status"] == "running" and not alive:
        state["warning"] = "Status says running but the trainer process is absent. Inspect console.log before resuming."
    state["progress_percent"] = round(100*state["main_tokens"]/state["target_tokens"], 2)
    rate = state.get("recent_tokens_per_second")
    if alive and rate and rate > 0:
        state["estimated_remaining_train_hours"] = round((state["target_tokens"]-state["main_tokens"])/rate/3600, 2)
        state["eta_caveat"] = "Recent rate; future validation, checkpoint, and contention costs can change this."
    checkpoint = directory / "checkpoint_latest.json"
    if checkpoint.exists():
        state["checkpoint"] = json.loads(checkpoint.read_text())
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
