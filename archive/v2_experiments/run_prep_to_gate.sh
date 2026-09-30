#!/usr/bin/env bash
# After the new-source build finishes: write the manifest, run the registered 1B
# pilot, then the pilot gate. Stops at the gate; the main run is launched separately.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/bufsgpu/yes/envs/sw/bin/python
export PYTHONPATH=src:v2/src:v2/.pydeps HF_HUB_CACHE=/data/shared/hf_cache/hub OMP_NUM_THREADS=4
GPU=GPU-78815613-815b-f56b-27c0-6bc239e093fc

until ls v2/data/rich_v1/_new_done/complete_*.json >/dev/null 2>&1; do
  tmux has-session -t gibc-v2-build-new 2>/dev/null || { echo "new-source build stopped without completing"; exit 1; }
  sleep 20
done
$PY -m scglm_v2.build --config v2/configs/prep.json --stage manifest
$PY - <<'EOF'
import json
from pathlib import Path
from scglm_v2.common import event, sha256_file
m = Path("v2/data/rich_v1/manifest.json")
event(Path("v2/phases/preparation/events.jsonl"), "preparation_complete", manifest_sha256=sha256_file(m))
Path("v2/phases/preparation/status.json").write_text(json.dumps({"phase_id": "v2_rich_prep_20260929", "state": "completed",
    "manifest": str(m), "manifest_sha256": sha256_file(m)}, indent=2) + "\n")
Path("v2/phases/pilot_1b/status.json").write_text(json.dumps({"phase_id": "v2_pilot_1b_20260929", "state": "running",
    "manifest_sha256": sha256_file(m)}, indent=2) + "\n")
EOF
CUDA_VISIBLE_DEVICES=$GPU $PY -m scglm_v2.train --config v2/configs/pilot_1b.json
CUDA_VISIBLE_DEVICES=$GPU $PY -m scglm_v2.pilot_gate
$PY - <<'EOF'
import json
from pathlib import Path
g = json.loads(Path("v2/phases/pilot_1b/gate.json").read_text())
Path("v2/phases/pilot_1b/status.json").write_text(json.dumps({"phase_id": "v2_pilot_1b_20260929", "state": "completed",
    "decision": g["decision"], "proxy_mean_acc": g["proxy_mean_acc"]}, indent=2) + "\n")
EOF
echo "GATE_DONE"
