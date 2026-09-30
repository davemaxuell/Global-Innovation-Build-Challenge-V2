#!/usr/bin/env bash
# Auto-resume the V2 main run after a crash: status "failed", or the process gone while
# status still says "running" (e.g. killed by the OS). A deliberate stop or the wall-time
# limit (status "paused") is never resumed. At most MAX automatic resumes, each recorded
# in the phase event log. Run inside tmux session gibc-v2-supervisor.
# SESSION / RUN_DIR / LOG_DIR / EVENTS / TRAIN_CMD / PAUSE are overridable for testing only.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/bufsgpu/yes/envs/sw/bin/python
GPU=GPU-78815613-815b-f56b-27c0-6bc239e093fc
SESSION=${SESSION:-gibc-v2-main}
RUN_DIR=${RUN_DIR:-v2/runs/main}
LOG_DIR=${LOG_DIR:-v2/phases/main}
EVENTS=${EVENTS:-$LOG_DIR/events.jsonl}
PAUSE=${PAUSE:-60}
TRAIN_CMD=${TRAIN_CMD:-"CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src OMP_NUM_THREADS=4 $PY -m scglm_v2.train --config v2/configs/main.json --resume $RUN_DIR/checkpoint_latest.json"}
MAX=3
n=0
status() { $PY -c "import json, sys
try: print(json.load(open(sys.argv[1]))['status'])
except Exception: print('unreadable')" "$RUN_DIR/status.json"; }
record() {
  PYTHONPATH=src $PY -c "
import sys; from pathlib import Path; from scglm_v2.common import event
event(Path(sys.argv[3]), sys.argv[1], detail=sys.argv[2])" "$1" "$2" "$EVENTS" > /dev/null
}
while true; do
  while tmux has-session -t "$SESSION" 2>/dev/null; do sleep "$PAUSE"; done
  s=$(status)
  case "$s" in
    completed) record supervisor_done "run completed"; exit 0 ;;
    paused) record supervisor_done "run paused (deliberate stop or wall limit); not resuming"; exit 0 ;;
  esac
  n=$((n + 1))
  if [ "$n" -gt "$MAX" ]; then record supervisor_gave_up "status=$s after $MAX automatic resumes"; exit 1; fi
  record auto_resume "attempt $n of $MAX; status was $s; resuming from checkpoint_latest.json"
  sleep "$PAUSE"
  tmux new-session -d -c "$PWD" -s "$SESSION" "( $TRAIN_CMD ) >> $LOG_DIR/train_auto_resume${n}.log 2>&1"
  sleep "$PAUSE"
done
