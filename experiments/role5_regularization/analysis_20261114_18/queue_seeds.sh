#!/bin/bash
# Start R4-recipe seed runs as soon as GPU 0 or 1 is freed by the current runs (ext3458 on 0, qms on 1).
cd /data/eunjin/DL/fassion_tagging
L=/data/eunjin/DL/logs; R=/data/eunjin/DL/runs_hr
C="--model convnext_tiny.fb_in22k_ft_in1k_384 --head linear_ms --pool gapgmp --loss bce --distill_npz /data/eunjin/DL/fashionformer/train_scores/ff_r50_train.npz --distill_alpha 0.5 --distill_shift 0.3 --max_long 1333 --flip 0.5 --test_short 800 --epochs 18 --batch_size 8 --accum 4 --lr_backbone 1e-4 --lr_head 1e-3 --weight_decay 0.05 --clip 1 --warmup_iters 1000 --ema 0.9995 --drop_path 0.1 --out_dir $R --short_min 400 --short_max 800"
declare -A RUNNING=([0]=R5_bce_kd_ext3458_s42 [1]=R5_bce_kd_qms_s42)
QUEUE=("R4re_bce_kd_s43 43" "R4re_bce_kd_s44 44")
while [ ${#QUEUE[@]} -gt 0 ]; do
  for g in 0 1; do
    [ ${#QUEUE[@]} -eq 0 ] && break
    e=${RUNNING[$g]}
    if [ -n "$e" ] && ! pgrep -f "exp_id $e " >/dev/null; then
      set -- ${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}")
      echo "$(date '+%F %T') GPU $g free ($e ended) -> start $1 (seed $2)" >> $L/queue_seeds.log
      CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python scripts/train_hires.py --exp_id $1 $C --seed $2 > $L/$1.log 2>&1 < /dev/null &
      RUNNING[$g]=$1
    fi
  done
  sleep 120
done
echo "$(date '+%F %T') queue empty" >> $L/queue_seeds.log
