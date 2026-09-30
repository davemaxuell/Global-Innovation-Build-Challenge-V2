#!/usr/bin/env python
"""Isolated data-quality mid-training study. The default verifies identities without writes."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CONFIG = ROOT / "configs/midtrain_quality.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="verify",
                        choices=("verify", "prepare", "preflight", "run", "resume", "status", "report"))
    parser.add_argument("--load", action="store_true", help="Also check the inherited checkpoint against the base export")
    args = parser.parse_args(argv)
    # The retained trainer snapshots sources by project-relative paths.
    os.chdir(ROOT)
    from scglm_study.gpu import process_environment, select_assigned_gpu
    from scglm_study.common import load_config, lock, read, roots
    cfg = load_config(CONFIG)
    process_environment(cfg["budget"]["cpu_threads"])
    data, output = roots(cfg)
    gpu = args.command in ("preflight", "run", "resume")
    if gpu:
        select_assigned_gpu(cfg["budget"]["assigned_gpu_uuids"][0])
    import torch
    torch.set_num_threads(cfg["budget"]["cpu_threads"])
    if gpu:
        def interrupted(*_):
            raise InterruptedError("Clean process interruption")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
    from scglm_midtrain import runner
    if args.command == "verify":
        result = runner.verify(cfg, load_model=args.load)
    elif args.command == "status":
        result = {"status": read(output / "status.json") if (output / "status.json").exists() else {"status": "not_started"},
                  "phases": {p.parent.name: read(p) for p in (output / "phases").glob("*/status.json")},
                  "runs": {p.parent.name: read(p) for p in (output / "runs").glob("*/status.json")},
                  "budget": read(output / "budget.json") if (output / "budget.json").exists() else None}
    else:
        output.mkdir(parents=True, exist_ok=True)
        with lock(output / ".study.lock"):
            if args.command == "prepare":
                from scglm_midtrain.prepare import prepare
                result = prepare(cfg)
            elif args.command == "report":
                result = runner.report(cfg)
            else:
                with lock(ROOT / "artifacts/pipeline_gpu0.lock"):
                    if args.command == "preflight":
                        from scglm_midtrain.prepare import load
                        from scglm_study.budget import Budget
                        result = runner.preflight(cfg, load(cfg), Budget(output, cfg["budget"]))
                    else:
                        result = runner.run(cfg, resume=args.command == "resume")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
