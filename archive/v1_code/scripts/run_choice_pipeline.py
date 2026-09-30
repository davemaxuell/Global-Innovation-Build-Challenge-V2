#!/usr/bin/env python
"""Isolated choice study. The default verifies local identities without writes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",nargs="?",default="verify",choices=("verify","prepare","preflight","run","resume","status","report"))
    parser.add_argument("--load",action="store_true",help="Also verify full ancestry and the local CPU export loader")
    args = parser.parse_args(argv)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from scglm_choice.common import config,roots,lock,read
    cfg = config()
    data,output = roots(cfg)
    gpu = args.command in ("preflight","run","resume")
    if gpu:
        allowed = cfg["budget"]["assigned_gpu_uuids"][0]
        listing = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,name","--format=csv,noheader"],text=True)
        rows = [line.split(", ",2) for line in listing.splitlines()]
        if not any(uuid == allowed and "H100" in name for _,uuid,name in rows):
            raise ValueError("Registered H100 UUID is unavailable")
        mapping = {index:uuid for index,uuid,_ in rows}
        visible = os.environ.get("CUDA_VISIBLE_DEVICES",allowed).split(",")
        if allowed not in [mapping.get(i.strip(),i.strip()) for i in visible]:
            raise ValueError("Assigned H100 excluded from current allocation")
        os.environ["CUDA_VISIBLE_DEVICES"] = allowed
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    if gpu:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        def interrupted(*_):
            raise InterruptedError("Clean process interruption")
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,interrupted)
    from scglm_choice.runner import verify,preflight,run,report
    if args.command == "verify":
        result = verify(cfg,load_model=args.load)
    elif args.command == "status":
        result = {"status":read(output/"status.json") if (output/"status.json").exists() else {"status":"not_started"},
                  "phases":{p.parent.name:read(p) for p in (output/"phases").glob("*/status.json")},
                  "budget":read(output/"budget.json") if (output/"budget.json").exists() else None}
    else:
        with lock(output/".study.lock"):
            if args.command == "prepare":
                from scglm_choice.prepare import prepare
                result = prepare(cfg)
            elif args.command == "report":
                result = report(cfg)
            else:
                with lock(ROOT/"artifacts/pipeline_gpu0.lock"):
                    if args.command == "preflight":
                        from scglm_choice.prepare import load
                        from scglm_choice.budget import Budget
                        result = preflight(cfg,load(cfg),Budget(output,cfg))
                    else:
                        result = run(cfg,resume=args.command == "resume")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
