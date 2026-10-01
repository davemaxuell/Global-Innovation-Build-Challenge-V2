"""Pre-launch smoke comparison of loss variants (scratch only; never touches the selection panel).

Diagnostic items = training items at epoch-1 order positions >= 20,000, which a 600-update
smoke run (19,200 questions) never sees. Raw-LM drift = bits/byte on 64 monitor-panel docs per source.
"""
import json, random, subprocess, sys, os
from pathlib import Path
ROOT = Path("/data/team_a/dave-workspace/Global Innovation Build Challenge V2")
SCR = Path(sys.argv[1]); variants = json.loads(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
import torch
from scglm_v2.devevaluate import load, bits_per_byte, panel_texts
from scglm_v2.task_format import score_items
cfg = json.loads((ROOT / "v2/phases/task_tune/config.json").read_text())
items = [json.loads(l) for l in (ROOT / cfg["data_dir"] / "train.jsonl").open()]
order = list(range(len(items))); random.Random(cfg["seed"] * 1000 + 1).shuffle(order)
diag = [items[i] for i in order[20000:]]
texts = panel_texts(ROOT / cfg["replay_manifest"], 64, split="monitor")
def evaluate(export):
    model, tok = load(str(export), "cuda")
    mc = score_items(model, diag, "cuda")
    bpb = {k: bits_per_byte(model, tok, "cuda", v) for k, v in texts.items()}
    del model; torch.cuda.empty_cache()
    accs = {k: round(v["acc"], 4) for k, v in mc.items() if not k.startswith("_")}
    return {"acc": accs, "mean4": round(mc["_proxy_mean_acc"], 4), "bpb_wiki": round(bpb["wiki"], 4),
            "bpb_mean": round(sum(bpb.values()) / len(bpb), 4)}
results = {}
if os.environ.get("SKIP_BASE") != "1": results["base"] = evaluate(ROOT / cfg["parent_export"])
print("base", results.get("base"), flush=True)
for name, over in variants.items():
    over = dict(over); lr = over.pop("lr", None)
    c = json.loads(json.dumps({**cfg, **over, "log_every": 100}))
    if lr: c["arms"]["lr6e-5"]["learning_rate"] = lr
    cpath = SCR / f"cfg_{name}.json"; cpath.write_text(json.dumps(c))
    run = SCR / f"cmp_{name}"
    subprocess.run(["rm", "-rf", str(run)])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "scglm_v2.task_tune", "train", "--config", str(cpath), "--arm", "lr6e-5",
                    "--max-updates", "600", "--run-dir", str(run)], cwd=ROOT, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results[name] = evaluate(run / "smoke" / "export")
    print(name, over, results[name], flush=True)
(SCR / os.environ.get("OUT", "compare_results.json")).write_text(json.dumps(results, indent=2))
