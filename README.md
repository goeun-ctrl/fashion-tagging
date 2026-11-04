# MAT4091 Fashion Category Tagging

한 사진에 보이는 27개 의류·액세서리 카테고리를 다중 라벨로 예측하는 팀 프로젝트.

## 폴더 구조

```
.
├── README.md                    이 파일
├── common/                      팀 공통 코드 (모든 실험이 같이 쓴다. 경로를 바꾸지 않는다)
│   ├── README_팀공유.md          공통 코드 사용법·지표 정의·지켜야 할 규칙
│   ├── labels.py  dataset.py  preprocess.py  metrics.py      데이터·전처리·평가
│   ├── train_eval.py            학습·평가 (pooling/head/loss/optimizer 옵션, 기본값 = 팀 기준선)
│   ├── heads.py  losses.py  optim.py                          train_eval.py의 교체 가능한 부품
│   ├── make_split.py            train → train_inner / holdout 고정 분할
│   ├── compare_preprocess.py    pad vs resize 비교
│   ├── compare_runs.py          두 실험 그룹 비교 보고서 (seed 평균, 부트스트랩, 수렴, 비용)
│   ├── aggregate_results.py     팀 결과 CSV 통합표
│   ├── visualize_errors.py      오류 이미지 시트
│   ├── run_queue.py             실험 목록을 빈 GPU에 순서대로 실행
│   ├── test_*.py                검증 테스트
│   ├── configs/                 기준선 YAML, 결과 CSV 양식
│   └── splits/                  train_inner.txt, holdout.txt, split_summary.csv (클래스별 분포)
├── experiments/                 실험 계획과 실행 목록
│   ├── README.md                5번(Optimizer·Head) 실험 사용법
│   ├── role5_jobs.txt           5번 실험 목록 (P0–P5)
│   ├── backbone_equal_budget.md 백본 비교 제안 (2·3번용, 같은 튜닝 예산)
│   └── backbone_budget_jobs.txt 위 제안의 실행 목록 (B1, B2)
└── docs/                        문서 (코드 아님)
    ├── plan/                    모델제안_통합안.pdf (최종 계획서), PROJECT_PROMPTS_KO.md
    ├── data/                    데이터분석_요약.pdf, DATA_INSPECTION_KO.md, IMAGE_REVIEW_KO.md, EDA_Report.md
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
# 기준선 1회 (데이터: labels.csv와 images/가 있는 폴더 또는 ZIP)
python common/train_eval.py --exp_id S1_cnxt_t_pad --owner 이름 --train_dir data/train --val_dir data/val

# 옵션 예: query head + Muon
python common/train_eval.py --exp_id R5_cnxt_query_muon --head query --lr_head 1e-4 --optimizer muon \
    --train_dir data/train --val_dir data/val
```

자세한 규칙(mean/std, holdout으로만 선택, 결과 CSV 형식)은 `common/README_팀공유.md`를 먼저 읽는다.

## 저장소에 올리지 않는 것 (`.gitignore`)
데이터(`data/`, `*.zip`), 학습 산출물(`runs/`, `report/`, `*.pt`, `*.npz`), 가상환경(`.venv/`).
결과는 공유 폴더로 주고받는다.
