# MAT4091 Fashion Category Tagging

한 사진에 보이는 사람들의 의류·액세서리를 **27개 카테고리 다중 라벨**로 예측하는 팀 프로젝트.
평가는 Micro / Macro / Image-wise F1, 최종 모델은 파라미터 50M 이하·오프라인 실행.

---

## 0. 구조가 바뀌었어요 (2026-11-04)

모든 코드가 `common/` 한 폴더에 섞여 있던 것을 **라이브러리 / 실행 스크립트 / 설정 / 테스트**로 나눴다.
**기존 코드와 명령은 그대로 동작한다.** 다만 새로 쓰는 코드와 명령은 새 경로를 쓴다.

| 예전 | 새 위치 | 예전 방식도 되나 |
|---|---|---|
| `common/labels.py`, `dataset.py`, `preprocess.py` | `src/fashion_tagging/data/` | 된다 (`common/`의 연결 파일) |
| `common/metrics.py` | `src/fashion_tagging/evaluation/metrics.py` | 된다 |
| `common/heads.py` | `src/fashion_tagging/models/heads.py` | — |
| `common/losses.py`, `common/optim.py` | `src/fashion_tagging/training/` | — |
| `common/train_eval.py` | `scripts/train.py` | 된다 (`python common/train_eval.py ...`) |
| `common/tools/*.py` (aggregate_results 등) | `scripts/*.py` | 안 된다 → 새 경로로 실행 |
| `common/tests/*.py` | `tests/*.py` | 안 된다 → 새 경로로 실행 |
| `common/configs/` | `configs/` | — |
| `common/splits/` | `data/splits/` | — |
| `common/README_팀공유.md` | `docs/guide/README_팀공유.md` | — |
| 최상위의 문서·PDF, `팀원 공유파일/` | `docs/` 아래 (plan, data, proposals) | — |

**개인 학습 코드가 있는 사람 (2번·3번 등)**
- `sys.path`에 `common/`을 넣고 `from labels import ...`, `from dataset import ...`, `from preprocess import ...`,
  `from metrics import ...` 하던 코드는 **수정 없이** 그대로 돈다. 같은 모듈이 불려 온다.
- 분할 목록 경로를 `common/splits/...`로 직접 적어 두었다면 `data/splits/...`로 바꾼다.
- 새로 쓴다면 아래처럼 import한다.
  ```python
  import sys; sys.path.insert(0, "src")          # 저장소 최상위 기준. pip install -e . 했다면 생략
  from fashion_tagging.data.labels import load_split
  from fashion_tagging.data.dataset import FashionDataset, read_names
  from fashion_tagging.data.preprocess import timm_mean_std
  from fashion_tagging.evaluation.metrics import compute_metrics, save_predictions
  ```

**1번 (결과 통합)**: 통합 명령은 `python scripts/aggregate_results.py ...` (아래 6장).

---

## 1. 폴더 구조

```
fashion-tagging/
├── README.md                     이 파일
├── requirements.txt              필요한 패키지
├── pyproject.toml                pip install -e . 설정 (선택)
│
├── src/fashion_tagging/          ── 라이브러리: import해서 쓰는 코드 ──
│   ├── paths.py                  저장소 경로(configs, data/splits, scripts)를 한 곳에서 관리
│   ├── data/
│   │   ├── labels.py             categories.txt·labels.csv 읽기 (클래스명은 한 줄 = 한 클래스, 쉼표로 안 쪼갬)
│   │   ├── preprocess.py         RGB → 종횡비 유지 resize → 검정 padding(또는 강제 resize) → 정규화
│   │   └── dataset.py            FashionDataset (폴더 또는 ZIP, 학습 증강은 좌우 반전만)
│   ├── models/
│   │   └── heads.py              pooling(gap/gmp/gapgmp) + head(linear/mlp/query/query_ms) + build_model
│   ├── training/
│   │   ├── losses.py             BCE, ASL (라벨 -1 = 모름은 손실에서 제외)
│   │   └── optim.py              AdamW / Muon+AdamW / SOAP, layer decay, cosine·constant 스케줄
│   └── evaluation/
│       ├── metrics.py            Micro·Macro·Image-wise F1, 클래스별 TP/FP/FN, 예측 저장·불러오기
│       └── bootstrap.py          같은 이미지에서 두 예측의 짝지은 부트스트랩 구간
│
├── scripts/                      ── 실행하는 코드 ──
│   ├── train.py                  학습 + holdout으로 epoch 선택 + holdout/val 평가 + 결과 CSV 한 줄
│   ├── run_queue.py              실험 목록 파일을 빈 GPU에 하나씩 실행
│   ├── compare_runs.py           두 실험 그룹 비교 보고서 (seed 평균±표준편차, 부트스트랩, 수렴, 비용, 클래스별)
│   ├── compare_preprocess.py     pad vs resize를 같은 설정으로 학습해 비교
│   ├── aggregate_results.py      팀원 results CSV를 합친 통합 결과표
│   ├── visualize_errors.py       틀린 이미지 시트 (선정 규칙 고정·기록)
│   ├── make_split.py             train → train_inner / holdout 고정 분할 (이미 만들어져 있음)
│   └── external/                 외부 데이터 만들기: build_external_labels → download_external_images → prepare_external
│
├── configs/
│   ├── baseline_convnext_tiny.yaml   기준선 설정 기록 (train.py가 읽지는 않음. 옵션은 명령줄로)
│   └── results_template.csv          결과 CSV 열 양식
├── data/                         데이터 (git에는 splits/만 올라감)
│   └── splits/                   train_inner.txt(43,197장), holdout.txt(2,425장), split_summary.csv(클래스별 분포)
├── tests/                        test_metrics, test_data, test_aggregate, test_visualize
├── common/                       호환용 연결 파일만 (새 코드에서는 쓰지 않음, common/README.md 참고)
├── experiments/                  실험 계획과 실행 목록
│   ├── README.md                 5번(Optimizer·Head) 실험 사용법
│   ├── role5_jobs.txt            5번 실험 목록 (P0–P5)
│   ├── backbone_equal_budget.md  백본 비교 제안: 같은 튜닝 예산 (2·3번)
│   └── backbone_budget_jobs.txt  위 제안의 실행 목록 (B1, B2)
└── docs/
    ├── plan/                     모델제안_통합안.pdf (최종 계획서), PROJECT_PROMPTS_KO.md
    ├── data/                     데이터분석_요약.pdf, DATA_INSPECTION_KO.md, IMAGE_REVIEW_KO.md, EDA_Report.md
    ├── guide/                    README_팀공유.md (공통 코드 상세 사용법·지표 정의·규칙)
    └── proposals/                팀원 제안서 (EVA02, 모델 구조 후보, 달파 EDA+모델 제안)
```

---

## 2. 설치

```
# torch는 CUDA 버전에 맞게 먼저 설치 (https://pytorch.org). Muon을 쓰려면 torch 2.9 이상
pip install -r requirements.txt
pip install -e .          # 선택. 하면 어디서나 import fashion_tagging 가능. 안 해도 scripts/는 바로 돈다
```

## 3. 데이터 배치

배포 ZIP을 풀어 저장소 아래 `data/`에 둔다 (git에는 올라가지 않는다).
```
data/
├── splits/               (저장소에 포함)
├── train/  categories.txt  labels.csv  labels.npz  images/*.jpg
└── val/    categories.txt  labels.csv  labels.npz  images/*.jpg
```
ZIP을 풀지 않고 `--train_dir MAT4091_fashion_train.zip`처럼 ZIP 경로를 직접 줘도 된다.

---

## 4. 학습

```
# 팀 기준선 (ConvNeXt-T, GAP+Linear, BCE, AdamW, 384px pad)
python scripts/train.py --exp_id S1_cnxt_t_pad --owner 이름 --train_dir data/train --val_dir data/val

# EVA02-S (ViT는 --set_img_size, 크기는 14의 배수)
python scripts/train.py --exp_id S3_eva_378 --model eva02_small_patch14_336.mim_in22k_ft_in1k \
    --set_img_size --size 378 --lr_backbone 5e-5 --train_dir data/train --val_dir data/val

# 옵션 조합 예: query head + Muon
python scripts/train.py --exp_id R5_cnxt_query_muon --head query --lr_head 1e-4 --optimizer muon \
    --train_dir data/train --val_dir data/val

# 코드가 끝까지 도는지만 짧게 확인 (점수는 의미 없음)
python scripts/train.py --exp_id smoke --max_train 256 --max_eval 128 --epochs 2 --train_dir data/train --val_dir data/val
```

**옵션** (적지 않으면 기본값 = 팀 기준선). 한 실험에서는 한 가지만 바꾼다.

| 구분 | 옵션 | 기본값 | 선택지·설명 |
|---|---|---|---|
| 모델 | `--model` | `convnext_tiny.fb_in22k_ft_in1k_384` | timm 모델 이름. ViT는 `--set_img_size` 추가 |
| 입력 | `--size`, `--preprocess_mode` | 384, `pad` | `pad` / `resize` |
| Pooling | `--pooling` | `gap` | `gap` / `gmp` / `gapgmp` (`--head query`에서는 안 씀) |
| Head | `--head` | `linear` | `linear` / `mlp`(`--mlp_hidden`) / `query`(C2, `--query_dim`, `--query_heads`) / `query_ms`(C3, stride 16+32, CNN만) |
| Loss | `--loss` | `bce` | `bce` / `asl` (`--asl_gamma_neg 4 --asl_gamma_pos 0 --asl_clip 0.05`) |
| Optimizer | `--optimizer` | `adamw` | `adamw` / `muon`(`--muon_lr_scale`, `--muon_momentum`, `--muon_head`) / `soap` |
| 학습률 | `--lr_backbone`, `--lr_head` | 1e-4, 1e-3 | |
| 정규화 | `--weight_decay`, `--layer_decay`, `--drop_path` | 0.05, 0(끔), 0(끔) | ViT 통합안 시작값: layer decay 0.75, drop path 0.1 |
| 스케줄 | `--scheduler`, `--warmup_ratio` | `cosine`, 0.05 | `cosine` / `constant` |
| 길이 | `--epochs`, `--patience` | 30, 5 | holdout Macro F1이 patience번 연속 안 오르면 종료 |
| 배치 | `--batch_size`, `--accum` | 32, 1 | 실효 배치 = batch × accum |
| 외부 데이터 | `--extra_train`, `--extra_weight`, `--extra_max` | 없음, 1, 0 | 라벨 1/0/-1 폴더를 학습에만 추가. 평가는 원본 그대로 (8장 참고) |
| 기타 | `--seed`, `--threshold` | 42, 0.5 | |

**결과물**
- `runs/<exp_id>/`: `best.pt`, `last.pt`, `preds_holdout.npz`, `preds_val.npz`, `log.json`(epoch별 기록)
- `results/results.csv`: 실험 1개 = 1줄 (점수, 설정, 학습 시간, 추론 ms, 최대 VRAM, 라이브러리 버전)

여러 실험을 GPU 여러 장에 돌릴 때는 `scripts/run_queue.py` (사용법: `experiments/README.md`).

---

## 5. 비교·분석

```
# 두 그룹 비교 (seed가 여러 개면 같은 순서로 나열)
python scripts/compare_runs.py --a S1_a_s42 S1_a_s43 --b S1_b_s42 S1_b_s43 --labels A B \
    --runs_dirs runs --csv results/results.csv --out report/a_vs_b

# pad vs resize
python scripts/compare_preprocess.py --prefix S4_cnxt_t --train_dir data/train --val_dir data/val

# 오류 이미지 (worst / fn / fp / random / ok)
python scripts/visualize_errors.py --pred runs/S1_cnxt_t_pad/preds_val.npz --images data/val --mode worst --out vis/s1
```

## 6. 결과 모으기 (11/9 통합)

각자 `results.csv`와 `runs/<exp_id>/preds_val.npz`, `preds_holdout.npz`를 공유 폴더에 올리고, 1번이 합친다.
```
python scripts/aggregate_results.py --csv 공유폴더/*/results.csv --runs_dirs 공유폴더/*/runs --baseline S1_cnxt_t_pad --out report
```
예측 파일이 있으면 점수를 다시 계산해 검증하고, exp_id 중복·50M 초과·seed/임계값 불일치를 맨 위에 경고한다.
**exp_id는 팀 전체에서 겹치지 않게** 정한다. 예전 양식 CSV(새 열 없음)도 그대로 읽는다.

## 7. 테스트

```
python tests/test_metrics.py       # 데이터 없이 돈다
python tests/test_data.py          # 아래 셋은 저장소 최상위에 MAT4091_fashion_train.zip, MAT4091_fashion_val.zip 필요
python tests/test_aggregate.py
python tests/test_visualize.py
```

---

## 7-1. 외부 데이터 (희귀 클래스 보강)

jumpsuit·glove·vest·cape·umbrella 보강용으로 LVIS·Open Images 이미지를 우리 27개 클래스 라벨로 바꾼 폴더를 만든다.
외부 데이터셋은 27개를 모두 주석하지 않으므로 라벨은 **1(있음) / 0(없음 확인) / -1(모름)**이고, -1은 손실에서 빠진다.
```
python scripts/external/build_external_labels.py --root ~/fashion_tagging/external --categories data/train/categories.txt
python scripts/external/download_external_images.py --root ~/fashion_tagging/external
python scripts/external/prepare_external.py --root ~/fashion_tagging/external \
    --ours data/train data/val data/test_public data/examples_labeled --out data/external/rare_extra_clean
python scripts/train.py --exp_id R5_cnxt_extra --extra_train data/external/rare_extra_clean --train_dir data/train --val_dir data/val
```
- `prepare_external.py`는 깨진·작은 이미지, **우리 train/val/test와 겹치는 이미지**(지각 해시), 외부끼리 중복을 뺀다.
  `report.md`(제외 이유·클래스별 라벨 수)와 `review/<클래스>.jpg`(눈으로 확인용)를 만든다.
- 외부 이미지는 학습에만 쓴다. holdout·공식 val은 원본 그대로여서 외부 데이터 없이 돌린 실험과 바로 비교된다.
- 데이터는 git에 올리지 않는다(`data/external/`). leg warmer는 외부에서 찾지 못했다.

## 8. 꼭 지킬 것 (자세히: `docs/guide/README_팀공유.md`)

- **체크포인트·임계값은 holdout으로만 고른다.** 공식 val은 마지막 보고에만 쓴다.
- 출력은 sigmoid + 임계값. **softmax·argmax·강제 top-1 금지.** 아무것도 안 넘으면 빈 예측 `[]`이 정상이다.
- 클래스 순서는 `categories.txt` 그대로(0~26). 이름을 쉼표로 나누지 않는다.
- mean/std는 모델마다 다르다 (`timm_mean_std(model_name)`로 읽는다).
- 손실에 BCEWithLogits를 쓰면 모델 출력에 sigmoid를 미리 적용하지 않는다.
- 최종 추론 모델 파라미터 합계 50,000,000개 이하 (`train.py`가 넘으면 멈춘다).

---

## 9. 지금까지 결과 (2026-11-04, 각 보고서 기준)

| 실험 | 담당 | holdout Micro / Macro / Image | 공식 val Micro / Macro / Image | 비고 |
|---|---|---|---|---|
| S2 ConvNeXt-T, BCE, AdamW | 2번 | 0.805 / **0.707** / 0.809 | 0.826 / **0.661** / 0.828 | lr 1e-4, best epoch 10, RTX A5000 |
| EVA02-S 378px, BCE, AdamW | 3번 | 0.804 / **0.708** / 0.805 | 0.811 / **0.632** / 0.816 | lr 5e-5, best epoch 10, RTX 4060 |

- 둘 다 **seed 1개, 튜닝 없음**, lr·장비·학습 코드가 다르다. 이 표만으로 백본 우열을 정하지 않는다
  (같은 튜닝 예산 비교 제안: `experiments/backbone_equal_budget.md`).
- 공통 약점 클래스: cardigan, leg warmer, jumpsuit, glove, sweater (val 양성이 적어 점수가 크게 흔들림).

## 10. 역할

| 담당 | 내용 |
|---|---|
| 1번 | 데이터·평가 체계, 공통 코드, 결과 통합 |
| 2번 | ConvNeXt-Tiny CNN 기준선 (S2) |
| 3번 | EVA02-Small Transformer 비교 |
| 4번 | Loss 비교 (BCE vs ASL) |
| 5번 | Optimizer 비교 (AdamW / Muon+AdamW / SOAP), Head·Pooling |

## 11. 저장소에 올리지 않는 것 (`.gitignore`)

데이터(`data/` 중 `splits/` 외 전부, `*.zip`), 학습 산출물(`runs/`, `report/`, `vis/`, `*.pt`, `*.npz`),
가상환경(`.venv/`). 결과는 공유 폴더로 주고받는다.
