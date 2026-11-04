# 5번 담당 실험 (Optimizer + Head)

## 새로 생긴 옵션 (`scripts/train.py`)
아무 옵션도 주지 않으면 팀 기준선과 같다(GAP + Linear, BCE, AdamW, warmup + cosine).

| 옵션 | 값 | 파일 |
|---|---|---|
| `--pooling` | `gap`(기본) / `gmp` / `gapgmp` | `src/fashion_tagging/models/heads.py` |
| `--head` | `linear`(기본) / `mlp` / `query`(Q2L 계열, 통합안 C2) / `query_ms`(C3, stride 16+32) | `src/fashion_tagging/models/heads.py` |
| `--loss` | `bce`(기본) / `asl` (`--asl_gamma_neg 4 --asl_gamma_pos 0 --asl_clip 0.05`) | `src/fashion_tagging/training/losses.py` |
| `--optimizer` | `adamw`(기본) / `muon`(Muon+AdamW) / `soap` | `src/fashion_tagging/training/optim.py` |
| `--scheduler` | `cosine`(기본) / `constant` | `src/fashion_tagging/training/optim.py` |
| `--muon_lr_scale`, `--muon_momentum`, `--muon_head` | Muon 그룹만 조정 | `src/fashion_tagging/training/optim.py` |
| `--layer_decay` | 예: `0.75` (ViT용, 0 = 끔) | `src/fashion_tagging/training/optim.py` |
| `--drop_path` | 예: `0.1` (0 = 끔) | `src/fashion_tagging/models/heads.py` |

Muon+AdamW 분할: 백본의 2D 행렬(ViT qkv·proj·mlp, ConvNeXt pwconv)은 Muon, 나머지(4D conv, bias, norm,
위치 임베딩, 분류기)는 AdamW. Muon lr은 `adjust_lr_fn="match_rms_adamw"`라 AdamW와 같은 눈금이다.

결과 CSV에 `pooling, head, scheduler, optimizer_args, layer_decay, drop_path` 열이 추가됐다.
예전 양식 CSV는 `aggregate_results.py`가 기준선 값으로 채워서 읽는다.

## 실행 순서 (GPU 서버)
```
# 1) 실험 목록 확인 (아무것도 실행하지 않음)
python scripts/run_queue.py --jobs experiments/role5_jobs.txt --only P1 --train_dir data/train --val_dir data/val --dry_run

# 2) 실행: 빈 GPU마다 한 개씩. SSH가 끊겨도 돌도록 nohup으로
nohup python scripts/run_queue.py --jobs experiments/role5_jobs.txt --gpus 0 1 2 3 --only P1 \
    --train_dir data/train --val_dir data/val --out_dir runs --results_csv results/results.csv > queue_P1.log 2>&1 &

# 3) 두 그룹 비교 (seed가 여러 개면 같은 순서로 나열)
python scripts/compare_runs.py --a R5_cnxt_adamw_s42 --b R5_cnxt_query_s42 --labels linear query \
    --runs_dirs runs --csv results/results.csv --out report/linear_vs_query
```
- 끝난 실험(`runs/<exp_id>/preds_val.npz`가 있음)은 다시 돌리지 않는다. 중간에 끊긴 실험은 처음부터 다시 돈다.
- 실험별 콘솔 출력은 `runs/<exp_id>.log`.
- `compare_runs.py` 결과: 세 F1(평균 ± seed 표준편차), 짝별 부트스트랩 구간, 목표 점수 도달 epoch·시간,
  비용(파라미터·학습 시간·추론 속도·VRAM), 클래스별 F1, `curves.png`.

## 단계 (`role5_jobs.txt`)
| 태그 | 내용 | 시작 조건 |
|---|---|---|
| P0 | ConvNeXt-T 기준선 | 1·2번의 S1 결과가 없을 때만 |
| P1 | Head·Pooling 예비 비교 | 지금 가능 (확정 비교는 백본·손실·해상도가 정해진 뒤) |
| P2 | Muon lr 탐색 (10 epoch) | 지금 가능 |
| P3 | ConvNeXt-T AdamW vs Muon, seed 3개 | P2에서 `--muon_lr_scale`을 고른 뒤 |
| P4 | EVA02-S AdamW vs Muon | 3번의 EVA02 설정(해상도·layer decay·drop path)과 맞춘 뒤 |
| P5 | SOAP | 선택. 시간·GPU 여유가 있을 때 |

해석 원칙(통합안): 개선 폭이 seed 변동보다 작으면 더 복잡한 쪽을 채택하지 않는다. optimizer를 바꾸면 lr·weight
decay도 다시 맞춰야 하므로, 결과는 "그 설정에서의 차이"로 보고하고 optimizer 단독 효과로 단정하지 않는다.
