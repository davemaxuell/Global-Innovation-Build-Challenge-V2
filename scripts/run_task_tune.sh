#!/usr/bin/env bash
# Task-tuning phase chain (v2/phases/task_tune): train the registered arms concurrently on
# one GPU, freeze the development-only selection, then run the pinned official evaluation
# once on the model the registration names. Run from the project root, e.g. in tmux:
#   GPU=0 bash scripts/run_task_tune.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/bufsgpu/yes/envs/sw/bin/python}
GPU=${GPU:-0}
PHASE=v2/phases/task_tune
CFG=$PHASE/config.json
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a $PHASE/chain.log; }
ev() { $PY -c "import sys; from pathlib import Path; from scglm_v2.common import event; event(Path('$PHASE/events.jsonl'), sys.argv[1], detail=sys.argv[2])" "$1" "$2" > /dev/null; }
state() { $PY -c "import sys, json; from pathlib import Path; from scglm_v2.common import write_json; write_json(Path('$PHASE/status.json'), {'phase_id': 'v2_task_tune_20261001', 'state': sys.argv[1], 'detail': sys.argv[2]})" "$1" "$2"; }

# 1. Training arms, concurrently.
mapfile -t ARMS < <($PY -c "import json; print('\n'.join(json.load(open('$CFG'))['arms']))")
state running "training arms: ${ARMS[*]}"; ev training_started "${ARMS[*]}"; say "training: ${ARMS[*]}"
pids=()
for arm in "${ARMS[@]}"; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -m scglm_v2.task_tune train --config $CFG --arm "$arm" > "$PHASE/train_$arm.log" 2>&1 &
  pids+=($!)
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then ev arm_completed "${ARMS[$i]}"; say "arm completed: ${ARMS[$i]}"
  else failed=1; ev arm_failed "${ARMS[$i]}"; say "arm FAILED: ${ARMS[$i]}"; fi
done
[ $failed -eq 0 ] || { state failed "a training arm failed; see train_*.log"; exit 1; }

# 2. Development-only selection (frozen before any official score of a task-tuned model).
state running "development-only selection"; say "selection"
CUDA_VISIBLE_DEVICES=$GPU $PY -m scglm_v2.task_select --config $CFG --out $PHASE/selection.json --device cuda \
    > $PHASE/selection.log 2>&1 || { state failed "selection failed; see selection.log"; exit 1; }
ev selection_frozen "$(tail -1 $PHASE/selection.log)"; say "$(tail -1 $PHASE/selection.log)"

# 3. Official evaluation of the registered model: the selected candidate; if V2 is kept, the best
#    eligible candidate; if none passed the Wikipedia guard, the highest-proxy candidate (reported only).
mapfile -t PICK < <($PY - <<'EOF'
import json
s = json.load(open("v2/phases/task_tune/selection.json"))
parent = s["parent"]
if s["selected_model"] != parent:
    model, role = s["selected_model"], "selected"
elif s["best_eligible_candidate"]:
    model, role = s["best_eligible_candidate"], "best_eligible_not_selected"
else:
    model = max((r for r in s["results"] if r["export"] != parent), key=lambda r: r["proxy_mean_acc"])["export"]
    role = "highest_proxy_guard_failed"
parts = model.rstrip("/").split("/")
print(model); print(f"{parts[-3]}_{parts[-2]}"); print(role)
EOF
)
MODEL=${PICK[0]}; NAME=${PICK[1]}; ROLE=${PICK[2]}
base=$PHASE/official/$NAME; n=1; while [ -e $base/attempt_$n ]; do n=$((n + 1)); done
state running "official evaluation: $NAME ($ROLE)"; say "official evaluation: $NAME ($ROLE), attempt $n"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_final_evaluation.py --final-evaluation --model "$MODEL" \
    --selection-record $PHASE/selection.json --device cuda:0 --output $base/attempt_$n \
    --history-dir v2/reports/validation_history --label "V2 task-tune official: $NAME ($ROLE)" >> $PHASE/official.log 2>&1 \
  || { ev official_failed "$NAME attempt $n"; state failed "official evaluation failed; see official.log"; exit 1; }
ev official_completed "$NAME ($ROLE) attempt $n"
state completed "official evaluation done: $NAME ($ROLE); see $base/attempt_$n/summary.json"
say "CHAIN_DONE"
