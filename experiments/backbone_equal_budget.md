# 백본 비교 제안: 같은 튜닝 예산으로 다시 맞추기

- 받는 사람: 2번(ConvNeXt-T), 3번(EVA02-S)
- 보내는 사람: 5번 (Optimizer·Head)
- 작성일: 2026-11-04
- 관련 파일: `experiments/backbone_budget_jobs.txt` (실험 목록), `common/compare_runs.py` (비교 보고서)

---

## 1. 왜 다시 맞추자는가

지금 백본 비교는 **두 모델에 같은 기본 설정(또는 각 모델 권장 시작값)을 한 번씩 적용한 결과**를 비교하는 구조다.
이 방식에서는 그 기본값이 더 잘 맞는 모델이 유리하다. 예를 들어 lr 1e-4가 ConvNeXt에는 적당하고 EVA02에는
크거나 작다면, EVA02가 실제로 더 나은 백본이어도 비교에서 질 수 있다(반대도 마찬가지).

3번 EVA02 제안서도 같은 위험을 적어 두었다:
> ViT는 학습률 등 설정에 민감해서, 같은 학습 예산에서 CNN보다 불리하게 보일 수 있음

그래서 **두 백본에 같은 수의 짧은 튜닝 실험을 주고, 각자 가장 좋은 설정끼리 seed 여러 개로 비교**하자는 제안이다.

---

## 2. 지금까지 각자 돌린 실험

### 2-1. 2번: ConvNeXt-T (S2) — 보고서로 확인한 내용

| 항목 | 값 |
|---|---|
| 실험 ID | `S2_jeongeun_cnxt_t_bce_384_pad` |
| 모델 | `convnext_tiny.fb_in22k_ft_in1k_384`, GAP → LayerNorm → Linear(768→27) |
| 손실 / Optimizer | BCE / AdamW, lr backbone 1e-4 · head 1e-3, weight decay 0.05 |
| 스케줄 | warmup 5% + cosine, **최대 30 epoch 기준**, holdout Macro F1 patience 5 |
| 배치 | 16 × accum 2 (실효 32), seed 42 |
| 입력 | 384px, 종횡비 유지 + 검정 padding, 좌우 반전만 |
| 결과 | best epoch 10 (epoch 15 조기 종료) / holdout Micro 0.805 · **Macro 0.707** · Image 0.809 / 공식 val Micro 0.826 · Macro 0.661 · Image 0.828 |
| 비용 | 학습 구간 59분, val forward 2.7ms/장, VRAM 3.65GB (공유 GPU, 0↔3번 이동) |
| 코드·환경 | 2번 개인 trainer (공통 코드 c5827df 재사용), torch 2.8.0 / timm 1.0.22 |
| 튜닝 | **없음.** lr·weight decay·학습 길이 비교는 "후속 후보, 이번에는 실행하지 않음"으로 명시 |

### 2-2. 3번: EVA02-S — **실행 결과는 아직 받지 못함 (확인 필요)**

제안서에 적힌 **계획값**은 아래와 같다. 실제로 돌린 설정·결과를 채워 주면 비교에 그대로 쓴다.

| 항목 | 제안서 계획값 | 실제 실행값 (3번 기입) |
|---|---|---|
| 모델 | `eva02_small_patch14_336.mim_in22k_ft_in1k` | |
| 입력 크기 | 378px (팀 384에 대응, 14의 배수) | |
| layer-wise lr decay | 0.65 ~ 0.75 | |
| drop path | 0.1 | |
| lr backbone / head | (명시 없음) | |
| weight decay | (명시 없음) | |
| 스케줄 · epoch | AdamW + warmup + cosine, 팀 공통 epoch | |
| 배치 · seed | (명시 없음) | |
| 시도한 설정 수 | — | |
| 결과 (holdout / val 세 F1) | — | |
| 비용 (학습 시간 · VRAM · 추론 ms) | — | |
| 코드 · 환경 | — | |

---

## 3. 기준을 맞춰 다시 돌릴 실험

### 3-1. 두 백본 공통 조건 (모든 실험에 같게)

| 항목 | 값 |
|---|---|
| 코드 | 공통 `common/train_eval.py` (커밋 8a2de2c 이상) |
| 환경 | gpu-server, Python 3.11 / torch 2.14.1 / timm 1.0.30 (같은 GPU 종류: RTX A5000) |
| 분할 | `common/splits/train_inner.txt` / `holdout.txt`, 공식 val은 보고용만 |
| 손실 / Optimizer | BCE / AdamW, weight decay 0.05 |
| head lr | **backbone lr × 10** (S2의 1e-4 : 1e-3 비율 유지) |
| 배치 | 32 (accum 1) |
| 스케줄 | warmup 5% + cosine |
| 증강 · 전처리 | 좌우 반전 p=0.5, 종횡비 유지 + 검정 padding |
| 임계값 | 0.5 고정 |

### 3-2. 모델별 기본값 (통합안 시작값, 튜닝 대상 아님)

| 항목 | ConvNeXt-T | EVA02-S |
|---|---|---|
| 입력 | 384px | 378px (`--set_img_size`) |
| drop path | 0 | 0.1 |
| layer decay | 없음 | 0.75 |

### 3-3. 1단계: 예비 탐색 — 백본당 4개, 같은 예산 (10 epoch, seed 42)

| # | ConvNeXt-T (2번) | EVA02-S (3번) |
|---|---|---|
| 1 | lr 5e-5 | lr 5e-5 |
| 2 | lr 1e-4 | lr 1e-4 |
| 3 | lr 2e-4 | lr 2e-4 |
| 4 | lr 1e-4 + drop path 0.1 | lr 1e-4 + layer decay 0.65 |

- lr을 3개 값으로: fine-tuning에서 가장 영향이 큰 값. 기준(1e-4)의 절반·두 배로, 기본값이 한쪽에만 유리했는지 본다.
- 4번째 칸은 모델별로 가장 영향이 클 하나: ConvNeXt는 drop path(S2에서 epoch 3 이후 holdout BCE 상승 = 과적합 신호),
  EVA02는 layer decay 범위(0.65~0.75)의 반대쪽 끝. 칸 수가 같아 예산은 공평하다.
- 10 epoch: S2의 best가 epoch 10이어서 설정 간 순위를 보기에 충분할 것으로 본다.
- **1단계 2번 칸은 S2와 다른 실험이다.** S2는 30 epoch 일정의 cosine에서 epoch 10이 best였고(그 시점 lr이 아직 높음),
  1단계는 10 epoch 일정이라 epoch 10에서 lr이 거의 0이다. 배치 분할·코드·환경도 다르다. 그래서 S2로 대신할 수 없다.

### 3-4. 2단계: 본 비교 — 백본당 최고 설정 1개 × seed 3개

- 1단계에서 백본마다 최고 설정을 골라 **30 epoch (holdout Macro F1 patience 5)**, seed **42 / 43 / 44**.
- `compare_runs.py`로 비교: 세 F1 평균 ± seed 표준편차, 짝별 부트스트랩 구간, 목표 도달 epoch·시간, 비용, 클래스별 F1.

### 3-5. 미리 정하는 선택 규칙

1. **1단계 설정 선택**: holdout Macro F1 최고. 차이 0.005 이내면 holdout Micro F1로 결정.
2. **2단계 백본 선택**: 세 F1 평균 차이가 seed 표준편차보다 크면 높은 쪽 채택. 그 이내면 학습 시간 · VRAM ·
   추론 속도가 나은(더 단순한) 쪽 채택.
3. 공식 val은 어느 단계에서도 선택에 쓰지 않는다.

### 3-6. 예상 시간 (gpu-server, GPU 4장)

| 단계 | 실험 수 | 예상 |
|---|---|---|
| 1단계 | 8 (4장 × 2회차) | 약 2시간 |
| 2단계 | 6 | 약 3~4시간 (조기 종료에 따라) |

S2의 epoch당 약 4분(학습 구간, batch 16, 공유 GPU)으로 추정. EVA02 378px 전체 데이터 epoch 시간은 아직 재지 않았다.
작은 표본 시험에서 VRAM은 ConvNeXt-T 6.4GB, EVA02-S(378, batch 32) 5.7GB로 24GB GPU에 여유가 있다.

---

## 4. 기존 결과는 어떻게 쓰나

- **S2 (2번)**: 버리지 않는다. 2단계에서 ConvNeXt를 lr 1e-4로 30 epoch 돌린 결과가 S2(holdout Macro 0.707)와 비슷하면
  **코드·환경이 바뀌어도 재현된다는 확인**이 된다. 크게 다르면 trainer 차이(배치 분할, AMP overflow step 처리 등)를 점검한다.
- **3번 기존 EVA02 결과**: 같은 방식의 재현 확인용. 2-2 표를 채워 주면 비교 보고서에 함께 적는다.

---

## 5. 실행 방법

```
cd /data/eunjin/ML/fassion_tagging        # 또는 각자 공통 코드 최신본(8a2de2c 이상) 위치
python common/run_queue.py --jobs experiments/backbone_budget_jobs.txt --only B1 --dry_run \
    --train_dir data/train --val_dir data/val           # 실행할 명령 확인만

nohup python common/run_queue.py --jobs experiments/backbone_budget_jobs.txt --gpus 0 1 2 3 --only B1 \
    --train_dir data/train --val_dir data/val --out_dir runs --results_csv results/results.csv > queue_B1.log 2>&1 &
```

- 1단계 결과를 보고 `backbone_budget_jobs.txt`의 B2 줄에 고른 lr(또는 drop path/layer decay)을 채운 뒤 `--only B2`로 실행.
- 비교:
  ```
  python common/compare_runs.py --a B2_cnxt_best_s42 B2_cnxt_best_s43 B2_cnxt_best_s44 \
      --b B2_eva_best_s42 B2_eva_best_s43 B2_eva_best_s44 --labels ConvNeXt-T EVA02-S \
      --runs_dirs runs --csv results/results.csv --out report/backbone
  ```
- 누가 돌릴지는 정하기 나름: 각자 자기 백본 4+3개를 돌려도 되고, 같은 GPU 종류에서 시간을 재기 위해 한 서버에서
  한꺼번에 돌려도 된다(5번이 gpu-server에서 돌릴 수 있음).

---

## 6. 확인 요청

- **2번**: S2의 예측 파일(`preds_holdout.npz`, `preds_val.npz`)과 epoch별 기록을 공유 폴더에 올려 줄 수 있는지.
- **3번**: 2-2 표의 "실제 실행값" 채우기. 시도한 설정이 여러 개였다면 그 목록도.
- **둘 다**: 3-3의 값(lr 3개 + 4번째 칸)과 3-5의 선택 규칙에 동의하는지, 바꾸고 싶은 값이 있는지.
- **일정**: 통합안상 백본 비교는 11/4~11/6. 1단계를 오늘 시작하면 11/5 안에 2단계까지 끝낼 수 있다.

### 한계 (같이 알고 가자)
- 10 epoch에서의 순위가 30 epoch에서도 유지된다는 보장은 없다(학습 길이가 바뀌면 최적 lr도 조금 바뀔 수 있음).
- 설정 8개면 holdout 과적합 위험은 작지만, 희귀 클래스는 holdout 양성이 8~20장이라 1단계 Macro F1 차이가
  작을 때는 너무 믿지 않는다.
- BCE로 고른 최고 백본이 ASL에서도 최고라는 보장은 없다. 두 백본 차이가 작으면 마지막에 상위 둘을 다시 확인한다.
