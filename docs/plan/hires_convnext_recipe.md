# 백본 결정: EVA02-S 대신 ConvNeXt-T + Fashionformer식 고해상도 학습

작성: 5번 (Optimizer·Head) · 2026-11-06 · 학습 코드: `scripts/train_hires.py`

## 요약

- **다음 단계의 주력 백본은 ConvNeXt-T**(`convnext_tiny.fb_in22k_ft_in1k_384`, 27.8M)로 한다. EVA02-S는 고해상도 실험에서 뺀다.
- 학습 방식은 **Fashionformer R50의 입력·최적화 설정**을 가져온다. Fashionformer는 Fashionpedia 공개 체크포인트로, 우리 val에서 지금까지 가장 높았다. 다만 우리는 이미지 단위 라벨만 있으므로 그에 맞게 일부를 바꾼다.
- 세 가지를 적용한다: ① 고해상도·비율 유지 입력, ② ASL 손실, ③ 백본/헤드 lr 분리 + gradient clipping. EMA 0.9995와 drop path 0.1은 그대로 유지한다.

## 1. 왜 이 방향인가: 근거가 된 측정

| 모델 (공식 val 1,158장) | 파라미터 | 입력 | micro | macro | image |
|---|---|---|---|---|---|
| ConvNeXt-T 기준선 (2번) | 27.8M | 384 정사각형 | 0.826 | 0.661 | 0.828 |
| EVA02-S 기준선 (3번) | 21.8M | 378 정사각형 | 0.811 | 0.632 | 0.816 |
| EVA02-S + wd 제외 + drop path + EMA (5번) | 21.8M | 378 정사각형 | 0.840 | 0.659 | 0.841 |
| **Fashionformer R50 체크포인트, 추론만** (5번) | 44.1M | 1333×800 비율 유지 | **0.858** | **0.705** | **0.861** |

자세한 내용은 `experiments/fashionformer_eval/README.md`와 `experiments/role5_regularization/results_eva02.md`에 있다.

- Fashionformer는 EVA+EMA보다 micro +0.018 [+0.007, +0.029], macro +0.046 [+0.002, +0.090], image +0.020 [+0.008, +0.032] 높다 (쌍 부트스트랩 95% 구간).
- **이긴 곳은 작은 물체다.** watch 0.42 → 0.74, sock 0.46 → 0.68, belt 0.61 → 0.75, scarf 0.52 → 0.68, glasses 0.85 → 0.95.
- 분류기의 오류 진단과 맞는다.
  - 원본은 83%가 세로 사진이고 긴 변 중앙값이 1024px이다. 378/384 정사각형에 넣으면 0.37배로 줄고, 실제 사진은 캔버스의 67%만 차지한다.
  - 그래서 작은 물체가 몇 픽셀만 남는다.
- **백본 사전학습은 오히려 우리가 더 좋다** (ImageNet-22k vs Fashionformer의 ImageNet-1k). 차이는 백본이 아니라 해상도와 학습 방식에서 온다.

## 2. EVA02-S를 고해상도 실험에서 빼는 이유

EVA02-S가 나쁜 모델이라서가 아니다. 378px에서 EMA 조합은 ConvNeXt 기준선과 비슷한 수준이었다. 빼는 이유는 **다음 단계(고해상도·가변 크기 입력)와 맞지 않기 때문**이다.

1. **고해상도에서 비용이 감당되지 않는다.**
   - EVA02-S는 전역 self-attention을 쓰는 ViT라서, 토큰 수가 늘면 attention 계산이 그 제곱으로 늘어난다.
   - Fashionformer 추론 크기(세로 사진 800×1200)를 patch 14로 자르면 토큰이 약 4,900개다. 지금(378², 729개)의 약 6.7배이고, attention 계산은 약 45배다.
   - ConvNeXt는 계산량이 픽셀 수에 비례해서 약 6.5배에 그친다. A5000 한 장으로 학습할 수 있는 건 ConvNeXt 쪽이다.
2. **가변 크기·직사각형 입력과 잘 맞지 않는다.**
   - EVA는 입력 크기가 바뀔 때마다 위치 임베딩을 보간해야 한다. 사전학습 크기(336)에서 멀어질수록 손해가 커질 수 있다.
   - 배치 안에서 크기가 다른 이미지를 패딩하면, attention에 마스크를 따로 걸어야 한다.
   - ConvNeXt는 직사각형 입력을 그대로 받고, 패딩은 마스크 평균 풀링으로 간단히 제외할 수 있다.
3. **같은 조건에서 ConvNeXt가 더 강했다.** 기준선끼리 비교하면 val macro 0.661 vs 0.632다. 이긴 Fashionformer도 CNN(ResNet-50) + 고해상도 구성이다.
4. **기존 실험과 이어진다.** 1차 ConvNeXt 실험 13개(drop path 0.1이 1위, EMA, Muon 배율)와 2번 기준선이 같은 백본이라 비교가 쉽다.

EVA02-S 결과(정규화 효과, EMA 유의성)는 참고 자료로 남긴다. ConvNeXt 고해상도가 기대만큼 나오지 않으면, EVA02-S 518 정사각형 하나를 비교용으로 다시 검토한다.

## 3. Fashionformer R50 설정 → 우리 학습 방식

Fashionformer 값은 저장소 설정 파일 `configs/fashionformer/kfashion_r50_separated_query_base.py`, `configs/_base_/datasets/fashionpedia.py`와 논문 구현 설명에서 가져왔다.

| 항목 | Fashionformer R50 | `train_hires.py` 기본값 | 바꾼 이유 |
|---|---|---|---|
| 추론 입력 | 비율 유지, 짧은 변 ≤800 / 긴 변 ≤1333, bilinear | **그대로** (`--test_short 800 --max_long 1333`) | |
| 학습 입력 | 1024 기준 0.1~2.0배 무작위 리사이즈 | **짧은 변 400~800 무작위**, 긴 변 ≤1333 | 0.1배까지 줄이면 작은 물체가 사라진다. 이를 메워 줄 객체 단위 정답이 우리에게는 없다 |
| 무작위 crop | 1024×1024 | **안 씀** | 이미지 라벨만 있으면, 잘려 나간 물체가 "있음" 정답으로 남아 라벨 노이즈가 된다 |
| 반전 | 0.5 | 0.5 | |
| 패딩 | 32의 배수 | 배치 최대 크기로 패딩(32의 배수) + **마스크 GAP** | 패딩 영역이 특징 평균에 섞이지 않게 한다 (테스트: 패딩 유무 출력 차이 7e-7) |
| 손실 | 분류 Focal loss | **ASL** (γ−=4, γ+=0, clip 0.05) | 같은 목적(쉬운 음성의 손실 줄이기)의 멀티라벨용 손실이고, 이미 구현돼 있다 |
| 옵티마이저 | AdamW, wd 0.05 | 그대로 | |
| lr | 1e-4, 백본 ×0.25 (=2.5e-5) | 기본: 백본 1e-4 / 헤드 1e-3. **Fashionformer 값은 비교 실험으로** | Fashionformer 헤드는 크고 무작위 초기화였지만, 우리 헤드는 선형층 하나다 |
| gradient clipping | max_norm 1.0 | 그대로 (`--clip 1.0`) | |
| warmup | linear, 1,000 iter, 시작 배율 0.001 | 그대로 (optimizer update 기준) | |
| 스케줄 | step ×0.1, 전체의 8/12·11/12 지점 | 그대로 (`--milestones 0.667 0.917`) | |
| 학습 길이 | 36 epoch (12 × 3) | **18 epoch**, 조기 종료 없음 | 시간 때문에 줄였다 (4장 참고). 강한 크기 증강이 있어서 끝까지 돌린다 |
| 배치 | 16 (GPU 8장) | 8 × 누적 4 = 32 | 고해상도 메모리 |
| EMA | 없음 | **0.9995** | 우리 실험에서 유일하게 유의했던 개선이다 |
| drop path | – | 0.1 | 1차 ConvNeXt 실험의 1위 설정 |

선택과 보고는 이렇게 한다.
- best epoch은 holdout macro F1(임계값 0.5)로 고른다. 저장되는 가중치는 EMA다.
- val은 마지막 보고에만 쓴다.
- `preds_holdout.npz`, `preds_val.npz`를 저장해서 부트스트랩과 공통 임계값 분석에 쓴다.

## 4. 1차 실험 계획 (연구실 서버 GPU 4장 동시)

| GPU | 실험 ID | 바꾸는 것 | 목적 |
|---|---|---|---|
| 0 | `R5_cnxt_hr_low_s42` | `--short_min 384 --short_max 384 --test_short 384 --max_long 640` | 같은 레시피에서 해상도만 낮춘 기준 (해상도 효과 분리) |
| 1 | `R5_cnxt_hr_s42` | 기본값 | Fashionformer 해상도 + 우리 lr |
| 2 | `R5_cnxt_hr_fflr_s42` | `--lr_backbone 2.5e-5 --lr_head 1e-4` | Fashionformer lr 그대로 |
| 3 | `R5_cnxt_hr_wide_s42` | `--short_min 80` | Fashionformer 원래 증강 범위에 가깝게. 범위를 좁힌 판단이 맞는지 확인 |

- **판단 기준**
  1. watch, belt, sock, scarf의 재현율
  2. val과 holdout에서 쌍 부트스트랩 95% 구간이 0을 넘는지
  3. Fashionformer 체크포인트(0.858 / 0.705 / 0.861)와의 차이
- **비용 추정** [확인 필요]: 384² 기준으로 epoch당 약 4분이었다. 짧은 변 400~800이면 평균 픽셀이 약 3.5~4배라서 epoch당 15~20분, 18 epoch이면 실행당 5~6시간으로 본다. 첫 epoch 시간을 재서 다시 맞춘다.

```bash
python scripts/train_hires.py --exp_id R5_cnxt_hr_s42 --train_dir data/train --val_dir data/val
python scripts/train_hires.py --exp_id R5_cnxt_hr_fflr_s42 --lr_backbone 2.5e-5 --lr_head 1e-4 ...
# 소형 점검 (CPU, 사전학습 없이)
python scripts/train_hires.py --exp_id smoke --no_pretrained --device cpu --model convnext_tiny \
    --limit_train 24 --limit_eval 12 --epochs 2 --batch_size 4 --accum 2 \
    --short_min 96 --short_max 160 --test_short 160 --max_long 256 --warmup_iters 2
```

## 5. 남은 결정과 위험

- **임계값**: ASL은 확률 분포를 바꿔서 0.5가 최적이 아닐 수 있다. holdout에서 공통 임계값 하나를 고르는 것까지 본다 (클래스별 임계값은 효과가 작았다).
- **추론 시간**: 800×1200 입력에서 5,000장을 처리하는 시간을 잰다 (Fashionformer는 장당 0.061초).
- **Fashionformer와의 관계**: 이 분류기가 Fashionformer 체크포인트(44.1M)를 넘지 못하면 최종 후보는 Fashionformer가 된다. 두 모델을 합치면 72M이라 예산을 넘는다.
- **원본 주석 사용 여부**(조교 질문): 허용되면 Fashionformer식 객체 단위 학습이나 crop 증강도 쓸 수 있다.
