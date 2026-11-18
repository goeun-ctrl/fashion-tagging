#!/bin/bash
# start CNN-DINOv3 lr 3e-5, 24 epochs on GPU 0 once R5_bce_kd_cnxtdv3_lr1e5_s42 ends
cd /data/eunjin/DL/fassion_tagging
while pgrep -f "exp_id R5_bce_kd_cnxtdv3_lr1e5_s42 " >/dev/null; do sleep 120; done
e=R5_bce_kd_cnxtdv3_lr3e5_ep24_s42
echo "$(date "+%F %T") start $e" >> /data/eunjin/DL/logs/queue_cnxt24.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_hires.py --exp_id $e --model convnext_tiny.dinov3_lvd1689m --head linear_ms --pool gapgmp --loss bce --distill_npz /data/eunjin/DL/fashionformer/train_scores/ff_r50_train.npz --distill_alpha 0.5 --distill_shift 0.3 --max_long 1333 --flip 0.5 --test_short 800 --epochs 24 --batch_size 8 --accum 4 --lr_backbone 3e-5 --lr_head 1e-3 --weight_decay 0.05 --clip 1 --warmup_iters 1000 --ema 0.9995 --drop_path 0.1 --out_dir /data/eunjin/DL/runs_hr --short_min 400 --short_max 800 --seed 42 > /data/eunjin/DL/logs/$e.log 2>&1
