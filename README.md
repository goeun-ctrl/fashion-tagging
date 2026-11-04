# MAT4091 Fashion Category Tagging

한 사진에 보이는 27개 의류·액세서리 카테고리를 다중 라벨로 예측하는 팀 프로젝트.

## 폴더 구조

```
.
├── README.md                    이 파일
├── requirements.txt  pyproject.toml          의존성 / pip install -e . 설정
├── src/fashion_tagging/         라이브러리 (import해서 쓰는 코드)
│   ├── paths.py                 저장소 경로 (configs, data/splits, scripts)
│   ├── data/                    labels.py  preprocess.py  dataset.py
│   ├── models/                  heads.py  (pooling gap/gmp/gapgmp, head linear/mlp/query, build_model)
│   ├── training/                losses.py (BCE/ASL)  optim.py (AdamW/Muon/SOAP, scheduler)
│   └── evaluation/              metrics.py (세 F1, 클래스별)  bootstrap.py (짝지은 부트스트랩)
├── scripts/                     실행하는 코드
│   ├── train.py                 학습·평가 (옵션 기본값 = 팀 기준선)
│   ├── make_split.py            train → train_inner / holdout 고정 분할
│   ├── compare_preprocess.py    pad vs resize 비교
│   ├── compare_runs.py          두 실험 그룹 비교 보고서 (seed 평균, 부트스트랩, 수렴, 비용)
│   ├── aggregate_results.py     팀 결과 CSV 통합표
│   ├── visualize_errors.py      오류 이미지 시트
│   └── run_queue.py             실험 목록을 빈 GPU에 순서대로 실행
├── configs/                     기준선 YAML, 결과 CSV 양식
├── data/                        데이터 (git에는 splits/만 올라감)
│   └── splits/                  train_inner.txt, holdout.txt, split_summary.csv (클래스별 분포)
├── tests/                       검증 테스트 (test_metrics, test_data, test_aggregate, test_visualize)
├── common/                      호환용 연결 파일 (예전 import·명령이 깨지지 않게. 새 코드는 쓰지 않음)
├── experiments/                 실험 계획과 실행 목록
│   ├── README.md                5번(Optimizer·Head) 실험 사용법
│   ├── role5_jobs.txt           5번 실험 목록 (P0–P5)
│   ├── backbone_equal_budget.md 백본 비교 제안 (2·3번용, 같은 튜닝 예산)
│   └── backbone_budget_jobs.txt 위 제안의 실행 목록 (B1, B2)
└── docs/                        문서 (코드 아님)
    ├── plan/                    모델제안_통합안.pdf (최종 계획서), PROJECT_PROMPTS_KO.md
    ├── data/                    데이터분석_요약.pdf, DATA_INSPECTION_KO.md, IMAGE_REVIEW_KO.md, EDA_Report.md
    ├── guide/                   README_팀공유.md (공통 코드 사용법·지표 정의·지켜야 할 규칙)
    └── proposals/               팀원 제안서 (EVA02, 모델 구조 후보, 달파 EDA+모델 제안)
```

## 역할

| 담당 | 내용 |
|---|---|
| 1번 | 데이터·평가 체계, 공통 코드, 결과 통합 |
| 2번 | ConvNeXt-Tiny CNN 기준선 (S2) |
| 3번 | EVA02-Small Transformer 비교 |
| 4번 | Loss 비교 (BCE vs ASL) |
| 5번 | Optimizer 비교 (AdamW / Muon+AdamW / SOAP), Head·Pooling |

## 빠른 시작

```
# 설치 (torch는 CUDA에 맞게 먼저 설치)
pip install -r requirements.txt
pip install -e .          # 선택: 어디서나 import fashion_tagging 가능. 안 해도 scripts/는 바로 돈다

# 기준선 1회 (데이터: labels.csv와 images/가 있는 폴더 또는 ZIP)
python scripts/train.py --exp_id S1_cnxt_t_pad --owner 이름 --train_dir data/train --val_dir data/val

# 옵션 예: query head + Muon
python scripts/train.py --exp_id R5_cnxt_query_muon --head query --lr_head 1e-4 --optimizer muon \
    --train_dir data/train --val_dir data/val
```

자세한 규칙(mean/std, holdout으로만 선택, 결과 CSV 형식)은 `docs/guide/README_팀공유.md`를 먼저 읽는다.

## 저장소에 올리지 않는 것 (`.gitignore`)
데이터(`data/` 아래 `splits/` 외 전부, `*.zip`), 학습 산출물(`runs/`, `report/`, `*.pt`, `*.npz`), 가상환경(`.venv/`).
결과는 공유 폴더로 주고받는다.
