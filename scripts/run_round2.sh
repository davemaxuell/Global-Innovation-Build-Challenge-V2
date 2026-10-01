#!/usr/bin/env bash
# Round-2 chain (v2/phases/round2): 100% data prep -> soup of existing models -> stronger arms ->
# soup of all -> development-only selection (half B) -> 100% retrain of the winning recipe ->
# Wikipedia guard -> one official evaluation. Every step is skipped if its output exists, so the
# chain can be rerun after a failure. Run from the project root:  GPU=0 bash scripts/run_round2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/bufsgpu/yes/envs/sw/bin/python}
GPU=${GPU:-0}
PHASE=v2/phases/round2
CFG=$PHASE/config.json
export PYTHONPATH=src PYTORCH_ALLOC_CONF=expandable_segments:True
OFF="env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
say() { echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a $PHASE/chain.log; }
ev() { $PY -c "import sys; from pathlib import Path; from scglm_v2.common import event; event(Path('$PHASE/events.jsonl'), sys.argv[1], detail=sys.argv[2])" "$1" "$2" > /dev/null; }
state() { $PY -c "import sys; from pathlib import Path; from scglm_v2.common import write_json; write_json(Path('$PHASE/status.json'), {'phase_id': 'v2_round2_20261001', 'state': sys.argv[1], 'detail': sys.argv[2]})" "$1" "$2"; }
fail() { ev failed "$1"; state failed "$1"; say "FAILED: $1"; exit 1; }
done_status() { $PY -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['status'] == 'completed' else 1)" "$1/status.json" 2>/dev/null; }

train_arms() {  # $1 = config; remaining = arm names; at most 2 concurrently
  local cfg=$1; shift; local arms=("$@") pids=() names=()
  for arm in "${arms[@]}"; do
    rd=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['arms'][sys.argv[2]]['run_dir'])" "$cfg" "$arm")
    if done_status "$rd"; then say "skip completed arm $arm"; continue; fi
    CUDA_VISIBLE_DEVICES=$GPU $OFF $PY -m scglm_v2.task_tune train --config "$cfg" --arm "$arm" >> "$PHASE/train_$arm.log" 2>&1 &
    pids+=($!); names+=("$arm")
    if [ ${#pids[@]} -ge 2 ]; then
      for i in "${!pids[@]}"; do wait "${pids[$i]}" || fail "arm ${names[$i]}"; ev arm_completed "${names[$i]}"; done
      pids=(); names=()
    fi
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || fail "arm ${names[$i]}"; ev arm_completed "${names[$i]}"; done
}

state running "100% data preparation"; say "prepare 100% data"
[ -f v2/data/task_tune_full_v1/report.json ] || $OFF $PY -m scglm_v2.task_tune prepare --config $PHASE/full_data_config.json > $PHASE/prepare_full.log 2>&1 || fail "prepare_full"

state running "soup of existing models"; say "soup: existing"
[ -f v2/runs/round2_soup_existing/soup.json ] || CUDA_VISIBLE_DEVICES=$GPU $OFF $PY -m scglm_v2.round2 soup --config $CFG --name existing > $PHASE/soup_existing.log 2>&1 || fail "soup existing"

state running "training stronger arms"; say "train: lr2e-5_e6 lr4e-5_e3"
train_arms $CFG lr2e-5_e6 lr4e-5_e3

state running "soup of all models"; say "soup: all"
[ -f v2/runs/round2_soup_all/soup.json ] || CUDA_VISIBLE_DEVICES=$GPU $OFF $PY -m scglm_v2.round2 soup --config $CFG --name all > $PHASE/soup_all.log 2>&1 || fail "soup all"

state running "development-only selection (half B)"; say "selection"
[ -f $PHASE/selection.json ] || CUDA_VISIBLE_DEVICES=$GPU $OFF $PY -m scglm_v2.round2 select --config $CFG > $PHASE/selection.log 2>&1 || fail "selection"
say "$(tail -1 $PHASE/selection.log)"; ev selection_frozen "$(tail -1 $PHASE/selection.log)"

[ -f $PHASE/final_plan.json ] || $PY -m scglm_v2.round2 plan-final --config $CFG > $PHASE/plan_final.log 2>&1 || fail "plan-final"
state running "100% retrain"; say "retrain: $(tail -1 $PHASE/plan_final.log | cut -c1-300)"
mapfile -t RETRAIN < <($PY -c "import json; [print(r['config'], r['arm']) for r in json.load(open('$PHASE/final_plan.json'))['retrain']]")
for line in "${RETRAIN[@]}"; do
  read -r rcfg rarm <<< "$line"
  train_arms "$rcfg" "$rarm"
done

state running "final model and Wikipedia guard"; say "final"
[ -f $PHASE/final.json ] || CUDA_VISIBLE_DEVICES=$GPU $OFF $PY -m scglm_v2.round2 final --config $CFG > $PHASE/final.log 2>&1 || fail "final"
say "$(tail -1 $PHASE/final.log)"; ev final_frozen "$(tail -1 $PHASE/final.log)"

mapfile -t PICK < <($PY -c "import json; f = json.load(open('$PHASE/final.json')); print(f['selected_model']); print('yes' if f['is_reference'] else 'no')")
if [ "${PICK[1]}" = "yes" ]; then state completed "V2.1 kept; no new official evaluation"; say "CHAIN_DONE (V2.1 kept)"; exit 0; fi
NAME=$(basename "$(dirname "$(dirname "${PICK[0]}")")")_$(basename "$(dirname "${PICK[0]}")")
base=$PHASE/official/$NAME; n=1; while [ -e $base/attempt_$n ]; do n=$((n + 1)); done
state running "official evaluation: $NAME"; say "official evaluation: $NAME (attempt $n)"
# The official runner resolves pinned dataset revisions online (no offline mode).
HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_final_evaluation.py --final-evaluation --model "${PICK[0]}" \
    --selection-record $PHASE/final.json --device cuda:0 --output $base/attempt_$n \
    --history-dir v2/reports/validation_history --label "V2 round-2 official: $NAME" >> $PHASE/official.log 2>&1 || fail "official $NAME"
ev official_completed "$NAME attempt $n"
state completed "official evaluation done: $NAME; see $base/attempt_$n/summary.json"
say "CHAIN_DONE"
