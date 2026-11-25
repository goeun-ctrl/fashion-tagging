#!/bin/bash
# Redesign B short experiments, part 4 (2026-11-23: kd0 + region-consistency loss mu 0.1 / 0.3 / 1.0, upper + lower groups, thr 0.5, min 16 cells; user decisions; replaces part 3). One change vs base per run. A job starts on the first free GPU
# (free = < 2000 MiB used, so teammates' jobs are never displaced).
W=$W
cd $W/fashion-tagging
export HF_HOME=$W/cache/hf TORCH_HOME=$W/data/torch_cache XDG_CACHE_HOME=$W/cache
L=$W/logs; R=$W/runs_segb; PY=$W/segb/.venv/bin/python
C="--data $W/data --prep $W/data/segb_prep --out_dir $R --epochs 6 --workers 10"
tr() { echo "$PY -I scripts/train_segb.py --exp_id $1 $C ${@:2} > $L/$1.log 2>&1"; }
P=S6
JOBS=(
  "${P}_kd0|$(tr ${P}_kd0 --distill_alpha 0)"
  "${P}_rc03|$(tr ${P}_rc03 --rc_w 0.3 --rc_groups 0 1 --rc_thr 0.5 --rc_min 16)"
  "${P}_rc01|$(tr ${P}_rc01 --rc_w 0.1 --rc_groups 0 1 --rc_thr 0.5 --rc_min 16)"
  "${P}_rc1|$(tr ${P}_rc1 --rc_w 1.0 --rc_groups 0 1 --rc_thr 0.5 --rc_min 16)"
)
declare -A BUSY=()
log() { echo "$(date '+%F %T') $*" >> $L/queue_segb_short.log; }
log "queue start: ${#JOBS[@]} jobs"
while [ ${#JOBS[@]} -gt 0 ]; do
  for g in 0 1 2 3; do
    [ ${#JOBS[@]} -eq 0 ] && break
    p=${BUSY[$g]}
    [ -n "$p" ] && pgrep -f "$p" >/dev/null && continue
    [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ] && continue
    IFS='|' read -r name cmd <<< "${JOBS[0]}"
    log "GPU $g -> $name"
    CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "exec -a qjob_$name bash -c '$cmd'" > /dev/null 2>&1 < /dev/null &
    BUSY[$g]="qjob_$name"; JOBS=("${JOBS[@]:1}"); sleep 90
  done
  sleep 120
done
log "queue empty"
