#!/usr/bin/env bash
# WiSE-FT phase chain (v2/phases/wiseft): build the registered interpolation grid, freeze the
# development-only selection, then run the pinned official evaluation once on the model the
# registration names. Run from the project root, e.g. in tmux:  GPU=0 bash scripts/run_wiseft.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/bufsgpu/yes/envs/sw/bin/python}
GPU=${GPU:-0}
PHASE=v2/phases/wiseft
CFG=$PHASE/config.json
export PYTHONPATH=src
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a $PHASE/chain.log; }
ev() { $PY -c "import sys; from pathlib import Path; from scglm_v2.common import event; event(Path('$PHASE/events.jsonl'), sys.argv[1], detail=sys.argv[2])" "$1" "$2" > /dev/null; }
state() { $PY -c "import sys; from pathlib import Path; from scglm_v2.common import write_json; write_json(Path('$PHASE/status.json'), {'phase_id': 'v2_wiseft_20261001', 'state': sys.argv[1], 'detail': sys.argv[2]})" "$1" "$2"; }

state running "building interpolation grid"; say "build"
HF_HUB_OFFLINE=1 $PY -m scglm_v2.wiseft --config $CFG > $PHASE/build.log 2>&1 || { state failed "build failed; see build.log"; exit 1; }
ev built "$(grep -c interpolated $PHASE/events.jsonl) exports"

state running "development-only selection"; say "selection"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU $PY -m scglm_v2.task_select --config $CFG \
    --out $PHASE/selection.json --device cuda > $PHASE/selection.log 2>&1 || { state failed "selection failed; see selection.log"; exit 1; }
ev selection_frozen "$(tail -1 $PHASE/selection.log)"; say "$(tail -1 $PHASE/selection.log)"

mapfile -t PICK < <($PY - <<'PYEOF'
import json
s = json.load(open("v2/phases/wiseft/selection.json"))
if s["selected_model"] != s["parent"]:
    model, role = s["selected_model"], "selected"
elif s["best_eligible_candidate"]:
    model, role = s["best_eligible_candidate"], "best_eligible_not_selected"
else:
    model, role = "", "none_eligible"
print(model); print(model.rstrip("/").split("/")[-2] if model else ""); print(role)
PYEOF
)
MODEL=${PICK[0]}; NAME=${PICK[1]}; ROLE=${PICK[2]}
if [ -z "$MODEL" ]; then
  state completed "no alpha passed the Wikipedia guard; no new official evaluation (registered)"; say "CHAIN_DONE (none eligible)"; exit 0
fi
base=$PHASE/official/$NAME; n=1; while [ -e $base/attempt_$n ]; do n=$((n + 1)); done
state running "official evaluation: $NAME ($ROLE)"; say "official evaluation: $NAME ($ROLE), attempt $n"
# The official runner resolves pinned dataset revisions online (no offline mode).
HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_final_evaluation.py --final-evaluation --model "$MODEL" \
    --selection-record $PHASE/selection.json --device cuda:0 --output $base/attempt_$n \
    --history-dir v2/reports/validation_history --label "V2 WiSE-FT official: $NAME ($ROLE)" >> $PHASE/official.log 2>&1 \
  || { ev official_failed "$NAME attempt $n"; state failed "official evaluation failed; see official.log"; exit 1; }
ev official_completed "$NAME ($ROLE) attempt $n"
state completed "official evaluation done: $NAME ($ROLE); see $base/attempt_$n/summary.json"
say "CHAIN_DONE"
