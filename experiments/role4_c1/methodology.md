# Role4: 고해상도 C1 BCE / ASL 비교

2026-11-07 점검. GPU 본학습·프로세스 종료·push는 실행하지 않았다.
기존 미커밋 `scripts/role4_loss.py`, `experiments/role4_loss.md`, `experiments/role4_convnext_confirmed.json`은 수정하지 않았다.

## 근거와 확인의 한계

- 로컬: `main`, HEAD `7eb438903abf909a7684984fcc1e6176b5fdcfeb`. 저장소와 상위 경로에서 적용할 AGENTS.md를 찾지 못했다. 다른 worktree의 작업도 수정하지 않았다.
- 로컬에는 `scripts/train_hires.py`가 없지만, GitHub Release 대상 커밋 `183aedc5b41f408ed6959c7bc1c6e3a26099f78f`에는 존재한다. 따라서 미푸시 코드로 단정하지 않는다. 그 파일을 **그대로** `scripts/role4_c1_reference.py`로 보관했다. 현재 원격 main은 후속 head/gate/증강 등이 추가됐으므로 사용하지 않는다.
- Release: https://github.com/goeun-ctrl/fashion-tagging/releases/tag/r5-cnxt-hr-s42
- 원본 레시피와 결과 문서는 `reference/`에 보관했다. Release API 메타데이터도 `reference/release.json`에 보관했다.
- 사용자가 지정한 중간점검 문서는 작업 중 `scripts/reference/` 아래 한글 분해형 파일명으로 추가되었다. 해당 `중간점검_20261107.md` 전체를 읽고 **3-5 C1, 3-6 추론 조건, 5장 최종 조합과의 차이, 9장 STEP 5, 11장 미푸시 파일**을 대조했다. 문서는 그대로 보존한다.
- Release asset 목록·크기·SHA256은 확인했지만, 연결된 GitHub 도구가 다운로드 URL에 404를 반환한다. 설정 JSON·history CSV·summary JSON **실제 파일 내용은 아직 확인하지 못했다**. 결과 문서의 설명과 실제 실행 설정의 일치 여부는 미확정이다. 로컬 사본도 발견하지 못했다.
- 필요한 추가 자료: `R5_cnxt_hr_s42_config.json`, `R5_cnxt_hr_s42_history.csv`, `R5_cnxt_hr_s42_summary.json`, 선택적으로 `_tta_summary.json`. 기존 ASL 재사용을 검증하려면 당시 초기 모델·헤드 상태 및 RNG·loader/sampler 기록도 필요하다. `best.pt`는 ASL 학습 후 EMA 상태여서 초기 상태를 대체할 수 없다.

### 중간점검 문서와 실제 코드 대조

문서 3-5의 기본 C1은 이 태그의 단일 stride32 마스크 GAP·선형 27출력 모델이다. 5장의 stride16+32 GAP·GMP/CSRA·28번째 gate·교사 BCE 혼합 모델과 다르므로 가져오지 않는다. 9장 STEP 5의 “BCE 1회면 됨”은 기존 ASL 초기화·데이터 순서와 일치를 입증할 수 있을 때의 비용 절감안이다. 이번에는 증거가 없고 loader RNG 기록 방식도 개선하므로 두 Loss 모두 재실행한다.

문서의 무작위 400~800은 실제 코드에서 네 단계로 양자화되며, EMA에는 warmup이 켜져 있다. 문서 5장의 logits 평균 TTA는 태그 `infer_hires.py`와 일치한다. 새 비교도 `sigmoid((원본 logits + 반전 logits)/2)`를 양쪽에 적용한다.

문서 3-6의 C1(전역 .60) val Macro 약 .746과 Release의 .7442(세 클래스 .5 예외)는 평가 조건이 다르다. 원본 결과 문서의 기본 .5/no-TTA val .7277과도 직접 섞지 않는다. 실제 asset 원문이 없어서 이 수치들의 독립 재계산은 하지 않았다. 문서의 seed 부족 지적에 따라 차이 구간도 seed 변동을 대체하지 않는다고 명시한다.

문서 11장의 `docs/plan/계획서_v2.md`, `experiments/role5_regularization/loss_experiment_guide.md`, 일부 후속 코드는 로컬에 없다. 이들을 비슷하게 재구현하지 않았다. 이번에 필요한 C1 자체는 Release 태그에서 확보했으며 그 원문과 공통 Loss·RGB/정규화 모듈의 일치를 확인했다.

## 기존 ASL 재사용 판정

직접 대조군으로 재사용하지 않고 **BCE와 ASL 모두 새로 실행**한다. 문서에 기록된 ASL은 ep13 EMA 선택, 18 epoch 완료, holdout Macro 0.7790이다. 이것은 새 BCE와 초기화까지 일치한다는 증거가 아니다. 태그 코드의 체크포인트에는 model/epoch/config/categories만 있고 초기 헤드 hash·초기 backbone hash·RNG·코드 환경·매 epoch 데이터 순서 증거가 없다. 새 비교는 명시적인 공통 초기 파일과 독립 loader generator를 사용하므로 기존 실행과 augmentation RNG 경로도 다르다.

기존 문서의 @0.5 기본 결과와 @0.60·클래스 예외·TTA 결과를 섞지 않는다. 과거 val 결과는 참고 기록이며 새 Loss 선택에 사용하지 않는다. 새 비교에서 train loss의 절대 크기는 우열 판단 기준이 아니다.

## C1 대조 및 고정 조건

| 항목 | 공개된 태그 코드와 role4 설정 |
|---|---|
| 모델 | `convnext_tiny.fb_in22k_ft_in1k_384`, 27,840,891 parameters |
| head | 실제 영역 Masked GAP → timm head의 norm/flatten/pre_logits/drop → Linear(27). head LR 대상은 fc만이며 norm은 backbone 그룹. 원본 구현 보존 |
| 학습 입력 | EXIF 방향 반영·RGB·비율 유지·bilinear, 좌우 반전 0.5, timm pretrained mean/std |
| 크기 증강의 실제 동작 | 연속 400~800이 아닌 **[416, 544, 672, 800]** 4단계. 단계별 긴 변 상한 `min(1333, 2*level)`, 방향×단계 버킷, 정규화 후 0 padding·stride32 canvas. 문서의 간략 설명과 구별 |
| 데이터 | 원본 Fashionpedia train_inner 43,197 / holdout 2,425 / 공식 val 1,158 고정. 클래스 순서·전체 정답 보존. 파일·라벨·split SHA256 기록 |
| Loss | BCE raw logits + `binary_cross_entropy_with_logits`; ASL gamma_neg=4, gamma_pos=0, clip=.05. 둘 다 알려진 항목(target>=0) 평균, -1 제외·gradient 0, 전체 unknown이면 0 |
| Optimizer | AdamW, backbone 1e-4 / fc 1e-3, wd .05, grad clip 1.0 |
| 배치 | GPU 1장에 batch 8 × accum 4, 명목 effective 32. 버킷 마지막 소량 배치는 원본대로 보존 |
| Scheduler | update 기준 warmup 1,000, 시작 배율 .001; 전체 updates의 8/12·11/12에서 ×.1 |
| 기타 | 18 epochs, early stop 없음, drop path .1, EMA .9995 **use_warmup=True**, AMP·channels_last |
| 원본의 경계 처리 | accumulation 마지막 묶음도 loss/accum 유지; AMP overflow면 update·schedule·EMA 건너뜀. 두 Loss 모두 동일 |

외부 506/3458장, 빈 사진·사람 사진, 증류, gate, 새 head, 모자이크·zoom-out·옛날 사진 증강은 사용하지 않는다. 기존 384px·cosine·30 epoch role4 JSON은 이번 실험용으로 사용하지 않는다.

## 구현

`settings.json`은 두 실험의 공통 설정이다. `scripts/role4_c1.py init`은 CPU에서 timm 사전학습 backbone과 새로운 27차원 head를 한 번 생성해 저장한다. **ASL best.pt를 초기 파일로 사용하지 않는다.** 두 학습은 그 파일을 strict load하고 모델·head·파일 hash를 기록한다. 초기 파일의 pretrained config에 HF repo/URL·모델 tag·라이브러리 정보를 남긴다. CPU 테스트만 random init을 사용한다.

`role4_c1_train.py`는 태그 학습 루프에서 초기화·기록·출력 보호·공식 val 분리 부분을 수정했다. 모델·sampler·resize·optimizer·Loss·EMA는 원본 구현을 사용한다. train/eval loader에 서로 독립적인 generator를 부여하고 매 epoch worker를 다시 생성한다. 두 Loss에 동일하게 적용하며 매 epoch sampler plan SHA256, seed, source SHA256, git commit/dirty status, 라이브러리 버전, data hashes, 실행 인자(config)를 기록한다. `training_state.pt`에 raw model/optimizer/scheduler/scaler/EMA 및 Python·NumPy·torch·CUDA·loader RNG를 저장한다. 자동 resume 기능은 제공하지 않는다. GPU 커널의 서로 다른 환경 간 bitwise 일치와 단일 seed의 일반화를 보장하지 않는다.

기존 초기 파일·실험 디렉터리·평가 디렉터리가 있으면 거부한다. 실패한 실험도 다른 output 경로로 재시도해야 한다. 새 산출물은 `runs/role4_c1/`, 보고서는 `report/role4_c1_*`로 분리한다.

## 정확한 실행 명령 (이번에는 실행하지 않음)

저장소 루트에서 아래 Python 환경을 동일하게 사용한다. 다른 머신에서는 settings.json의 데이터·split 경로만 양쪽 공통으로 수정한다. 학습 사이에는 코드·라이브러리·settings를 변경하지 않는다. 가용 GPU 번호는 작업자가 배정받아 `CUDA_VISIBLE_DEVICES`로 지정한다.

```bash
PY=/home/frostpine/fashion_tagging/.venv/bin/python

# CPU에서 사전학습 초기 상태를 한 번 생성 (캐시에 없으면 timm/HF 다운로드 필요)
CUDA_VISIBLE_DEVICES='' $PY scripts/role4_c1.py init

# 동일 초기 파일에서 각각 처음부터 학습. 기본 device=cuda, 현재 보이는 GPU 1장 사용
$PY scripts/role4_c1.py train --loss bce
$PY scripts/role4_c1.py train --loss asl

# holdout: 원본 및 좌우 반전 logits를 모두 저장, 추론 800/1333
for LOSS in bce asl; do
  $PY scripts/role4_c1_evaluate.py \
    --run runs/role4_c1/R4_C1_${LOSS}_s42 --split holdout \
    --out runs/role4_c1/eval_holdout_${LOSS}
done

# holdout만으로 Loss 및 각 Loss의 threshold 고정
$PY scripts/role4_c1_compare.py \
  --bce_run runs/role4_c1/R4_C1_bce_s42 --asl_run runs/role4_c1/R4_C1_asl_s42 \
  --bce_preds runs/role4_c1/eval_holdout_bce --asl_preds runs/role4_c1/eval_holdout_asl \
  --out report/role4_c1_holdout_s42

# 고정 후 공식 val 평가. selection lock 없이는 거부
for LOSS in bce asl; do
  $PY scripts/role4_c1_evaluate.py \
    --run runs/role4_c1/R4_C1_${LOSS}_s42 --split val \
    --lock report/role4_c1_holdout_s42/selection.json \
    --out runs/role4_c1/eval_val_${LOSS}
done
$PY scripts/role4_c1_compare.py \
  --bce_run runs/role4_c1/R4_C1_bce_s42 --asl_run runs/role4_c1/R4_C1_asl_s42 \
  --bce_preds runs/role4_c1/eval_val_bce --asl_preds runs/role4_c1/eval_val_asl \
  --lock report/role4_c1_holdout_s42/selection.json --out report/role4_c1_val_s42
```

## 평가 및 자동 보고

- 매 epoch: EMA, holdout, 800/1333, threshold .5, TTA 없음. Macro F1 최대(동률이면 먼저 나온 epoch)를 best.pt로 선택한다.
- 기본 Loss 선택: holdout의 **baseline Macro F1**. 동률이면 BCE. 단일 seed 잠정 선택임을 기록한다.
- 네 행을 따로 보고: baseline(.5/no TTA), tuned(no TTA), tta_05(.5/flip TTA), tta_tuned.
- 양쪽 모두 threshold **.05~.95, 간격 .05**의 동일 전역 grid를 사용한다. holdout Macro F1 최대, 동률이면 낮은 threshold. no-TTA와 TTA 각각 독립적으로 선택한다. 기존 .60이나 클래스 예외를 이식하지 않는다.
- flip TTA는 padding 전에 이미지를 뒤집어 각각 추론한 **raw logits 평균 후 sigmoid**이다. 태그 `infer_hires.py`와 동일하다.
- `selection.json`이 Loss·threshold와 checkpoint/config SHA256을 고정한다. 공식 val에서는 재탐색하지 않는다. 두 Loss의 최종 val을 함께 보고하더라도 승자를 val로 다시 선택하지 않는다.
- `summary.csv`: Macro/Micro/Image-wise F1, Macro/Micro Precision·Recall.
- `comparison.csv`: 모드별 두 Loss Macro F1·차이(ASL−BCE)·95% 구간.
- `per_class.csv`: 27개 클래스 × 2 Loss × 4 모드의 F1·Precision·Recall·양성 수.
- `rare_classes.csv`: cardigan, vest, jumpsuit, cape, glove, leg warmer, umbrella 7개를 같은 형식으로 추출.
- `bootstrap.json`: 이미지 단위 쌍 bootstrap 2,000회, seed42, Macro/Micro/Image F1 차이의 percentile 95% 구간. threshold는 재선택하지 않으며 모델/선택을 고정한 조건부 구간이다. 학습 seed 변동이나 holdout 선택 낙관성까지 포함하지 않는다. 작은 차이를 확정적 우열로 표현하지 않는다.
- 비교기는 설정·라이브러리·source hash·초기 상태·sampler plan·원본 정답·이미지 순서·27개 클래스 순서·해상도·checkpoint 출처 불일치를 거부한다.

## CPU 검증 및 자원

```bash
CUDA_VISIBLE_DEVICES='' /home/frostpine/fashion_tagging/.venv/bin/python tests/test_role4_c1_cpu.py
CUDA_VISIBLE_DEVICES='' /home/frostpine/fashion_tagging/.venv/bin/python tests/test_losses.py
```

실제 분할 전체의 개수·중복·상호 배제·원본 정답·클래스 순서·이미지 파일 존재를 검사한다. 실제 원본 train 2장/holdout 2장, 64px, 1 epoch로 두 Loss의 학습·backward·AdamW·scheduler·EMA·checkpoint 저장/로딩을 CPU에서 확인한다. masking과 known-only reduction, 극단 logits의 유한성, 27차원 출력, 같은 초기 state hash, overwrite 거부를 확인한다. 별도 worker 2개로 데이터 순서·flip 재현도 점검한다. 합성 logits + 실제 holdout 정답으로 8개 요약행·216개 클래스행·56개 희소행, 2,000회 bootstrap 동일 예측 차이 구간 [0,0], seed 불일치 거부·val lock 거부를 확인한다.

검증 기록은 `report/role4_c1_cpu/checks.json`에 저장한다. 테스트 수치는 실제 성능 비교로 사용하지 않는다. CPU smoke는 random initialization이고 GPU AMP·실제 800/1333 메모리·사전학습 다운로드·18 epoch 성능을 검증하지 않는다.

현재 CPU 환경: Python 3.12.3, torch 2.8.0+cu126, timm 1.0.22 (나머지 정확한 버전은 checks.json). 기존 requirements.txt의 Muon용 torch>=2.9와 달리 이번 AdamW 경로는 이 환경에서 통과한다. 동일 환경을 양쪽에 사용하고 GPU 환경은 실행 시 다시 기록한다.

원본 결과 문서의 RTX A5000 24GB 실측은 **실험당 5.4시간 / 18 epoch**다. 새 두 실험은 약 **11~13 GPU-hours**를 계획값으로 잡는다. 한 장 순차 약 11~13시간, 독립 GPU 두 장이면 약 5.5~6.5시간 + 후처리다. 새 worker·checkpoint 기록 오버헤드와 하드웨어/IO에 따라 달라진다. 처음 epoch로 재산정한다. batch8×accum4 유지, 가급적 24GB GPU 1장/실험, CPU worker6/실험, RAM 16GB 이상/실험을 계획한다. 원본 960/1600 확장 실측 15.6GB는 참고치이며 이번 코드에서 GPU 메모리를 측정한 값은 아니다. 디스크는 데이터 외에 초기 모델·각 실험 best/last·학습 상태·평가 결과용 총 3~5GB 여유를 권장한다. TTA는 평가 forward 비용 약 2배다. 이 추정 때문에 GPU를 확보하거나 다른 작업을 종료하지 않는다.
