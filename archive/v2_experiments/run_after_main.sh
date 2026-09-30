#!/usr/bin/env bash
# After the V2 main run completes:
#   1. decay A/B branch (v2_anneal_quality_20260929), one automatic resume on failure
#   2. development-only endpoint selection (frozen selection.json)
#   3. official evaluation (pinned protocol): the selected export first, then the other arm
#   4. the pre-registered V1-vs-V2 rule   5. figures
# Each step is skipped when its output exists, so rerunning after an interruption is safe.
# Arm only after the user confirms the rules in v2/phases/final_selection/registration.json.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/bufsgpu/yes/envs/sw/bin/python
GPU=GPU-78815613-815b-f56b-27c0-6bc239e093fc
export PYTHONPATH=src:v2/src:v2/.pydeps OMP_NUM_THREADS=8 HF_HUB_CACHE=/data/shared/hf_cache/hub
SEL=v2/phases/final_selection
status() { $PY -c "import json, sys
try: print(json.load(open(sys.argv[1]))['status'])
except Exception: print('unreadable')" "$1/status.json"; }
ev() { $PY -c "import sys; from pathlib import Path; from scglm_v2.common import event
event(Path(sys.argv[1]), sys.argv[2], detail=sys.argv[3])" "$1" "$2" "$3" > /dev/null; }
say() { echo "[$(date '+%F %T')] $*"; }

# 0. Wait for the main run to finish and export.
say "waiting for v2/runs/main to complete"
until [ "$(status v2/runs/main)" = completed ] && [ -f v2/runs/main/export/model.safetensors ]; do
  s=$(status v2/runs/main)
  if { [ "$s" = failed ] || [ "$s" = paused ]; } && ! tmux has-session -t gibc-v2-main 2>/dev/null \
     && ! tmux has-session -t gibc-v2-supervisor 2>/dev/null; then
    say "main run is $s and nothing will resume it; stopping"; ev v2/phases/main/events.jsonl chain_stopped "main $s"; exit 1
  fi
  sleep 60
done
say "main run complete"

# 1. Decay A/B branch.
if [ ! -f v2/runs/anneal_quality/export/model.safetensors ]; then
  [ -f v2/phases/anneal_quality/config.json ] || $PY -m scglm_v2.branch \
      --template v2/configs/anneal_quality.json --out v2/phases/anneal_quality/config.json || exit 1
  for attempt in 1 2; do
    resume=(); [ -f v2/runs/anneal_quality/checkpoint_latest.json ] && resume=(--resume v2/runs/anneal_quality/checkpoint_latest.json)
    if [ ${#resume[@]} -eq 0 ] && [ -e v2/runs/anneal_quality/run.json ]; then
      # Failed before its first checkpoint: keep the record, start clean.
      mv v2/runs/anneal_quality "v2/runs/anneal_quality_failed_before_checkpoint_$attempt"
    fi
    ev v2/phases/anneal_quality/events.jsonl train_attempt "attempt $attempt ${resume[*]}"
    say "anneal branch attempt $attempt"
    CUDA_VISIBLE_DEVICES=$GPU $PY -m scglm_v2.train --config v2/phases/anneal_quality/config.json "${resume[@]}" \
        >> v2/phases/anneal_quality/train.log 2>&1 && break
  done
  if [ -f v2/runs/anneal_quality/export/model.safetensors ]; then
    ev v2/phases/anneal_quality/events.jsonl completed "export written"
  else
    ev v2/phases/anneal_quality/events.jsonl failed "no export after 2 attempts; selection proceeds with the baseline only"
    say "anneal branch failed; continuing with the baseline only"
  fi
fi

# 2. Development-only selection, frozen before any official V2 score.
if [ ! -f $SEL/selection.json ]; then
  say "selection"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m scglm_v2.select_endpoint --baseline v2/runs/main \
      --challenger v2/runs/anneal_quality --out $SEL/selection.json --device cuda >> $SEL/selection.log 2>&1 || exit 1
  ev $SEL/events.jsonl selection_frozen "$(tail -1 $SEL/selection.log)"
fi

# 3. Official evaluation: the selected export first, then every other listed arm.
mapfile -t MODELS < <($PY -c "import json
s = json.load(open('$SEL/selection.json'))
print(s['selected_model'])
for r in s['results']:
    if r['export'] != s['selected_model']: print(r['export'])")
for model in "${MODELS[@]}"; do
  name=$(basename "$(dirname "$model")")
  base=$SEL/official/$name
  if ls $base/attempt_*/completed.json > /dev/null 2>&1; then continue; fi
  n=1; while [ -e $base/attempt_$n ]; do n=$((n + 1)); done
  say "official evaluation: $name (attempt $n)"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_final_evaluation.py --final-evaluation --model "$model" \
      --selection-record $SEL/selection.json --device cuda:0 --output $base/attempt_$n \
      --history-dir v2/reports/validation_history --label "V2 official: $name" >> $SEL/official.log 2>&1 \
    || { ev $SEL/events.jsonl official_failed "$name attempt $n"; say "official evaluation failed for $name"; exit 1; }
  ev $SEL/events.jsonl official_completed "$name attempt $n"
done

# 4. Pre-registered V1-vs-V2 rule on the selected export's official summary.
selected_name=$(basename "$(dirname "${MODELS[0]}")")
summary=$(ls $SEL/official/$selected_name/attempt_*/summary.json | tail -1)
[ -f $SEL/v1_vs_v2.json ] || $PY -m scglm_v2.v1_vs_v2 --v2-summary "$summary" --out $SEL/v1_vs_v2.json >> $SEL/official.log 2>&1
ev $SEL/events.jsonl v1_vs_v2 "$($PY -c "import json; print(json.load(open('$SEL/v1_vs_v2.json'))['outcome'])")"

# 5. Figures.
$PY -m scglm_v2.figures >> $SEL/official.log 2>&1
say "CHAIN_DONE"
