# 공통 코드 사용법 (1번 담당)

모든 팀원이 같은 라벨 읽기·전처리·평가를 쓰면 결과표를 그대로 합칠 수 있다.

## 파일
라이브러리는 `src/fashion_tagging/`, 실행 스크립트는 `scripts/`, 설정은 `configs/`, 고정 분할은 `data/splits/`에 있다.
예전 `common/` 경로(`from labels import ...`, `python common/train_eval.py`)는 `common/`의 연결 파일로 계속 동작한다.

| 파일 | 역할 |
|---|---|
| `src/fashion_tagging/data/labels.py` | categories.txt / labels.csv 읽기. 클래스 이름은 한 줄이 한 클래스(쉼표로 쪼개지 않음) |
| `src/fashion_tagging/data/preprocess.py` | RGB 변환 → 종횡비 유지 후 검정 padding(또는 강제 resize) → 정규화 |
| `src/fashion_tagging/data/dataset.py` | `FashionDataset`. 폴더 또는 ZIP에서 읽음. train은 hflip p=0.5만 |
| `src/fashion_tagging/evaluation/metrics.py` | Micro / Macro / Image-wise F1, 클래스별 TP·FP·FN, 확률 저장·불러오기 |
| `src/fashion_tagging/evaluation/bootstrap.py` | 같은 이미지에 대한 두 예측의 짝지은 부트스트랩 구간 |
| `src/fashion_tagging/models/heads.py` | pooling(gap/gmp/gapgmp)·head(linear/mlp/query/query_ms)와 모델 생성 |
| `src/fashion_tagging/training/losses.py`, `optim.py` | loss(BCE/ASL), optimizer(AdamW/Muon/SOAP)·scheduler |
| `src/fashion_tagging/paths.py` | 저장소 경로(configs, data/splits, scripts) |
| `scripts/train.py` | 학습·평가. 기본값은 BCE+AdamW+cosine 기준선. holdout으로 epoch 선택, holdout/val 예측(.npz)과 결과 CSV 한 줄 저장 |
| `scripts/make_split.py` | Train에서 내부 holdout 5%를 고정(seed 42). 결과는 `data/splits/` |
| `scripts/compare_preprocess.py` | `scripts/train.py`를 pad / resize로 두 번 돌려 지표·부트스트랩 구간·클래스별 F1을 비교 |
| `scripts/compare_runs.py` | 두 실험 그룹 비교(seed 평균·표준편차, 부트스트랩, 수렴, 비용, 클래스별 F1) |
| `scripts/aggregate_results.py` | 팀원들의 results CSV를 합쳐 통합 결과표(성능·설정·비용·클래스별 F1)와 경고 목록을 만든다 |
| `scripts/visualize_errors.py` | 틀린 이미지를 정답·예측·확률과 함께 보여 주는 시트(JPG)와 기록(JSON) 생성 |
| `scripts/run_queue.py` | 실험 목록을 빈 GPU에 하나씩 실행 |
| `scripts/external/*.py` | 외부 데이터(LVIS·Open Images) 라벨 변환·다운로드·정리(중복 제거) |
| `data/splits/train_inner.txt`, `holdout.txt` | 학습용 43,197장, 내부 검증용 2,425장 (파일명 목록) |
| `data/splits/split_summary.csv` | 클래스별 split 개수 |
| `configs/` | 기준선 YAML, 결과 CSV 양식 |
| `tests/test_*.py` | 검증 테스트 (metrics, data, aggregate, visualize) |

## 사용 순서
1. 데이터 준비: 서버에서 ZIP을 풀어 `data/train/`, `data/val/` 구조로 둔다(`images/` 폴더가 그 안에 있음).
2. 학습: 파일 목록으로 train / holdout 로더를 만든다.
   ```python
   import sys; sys.path.insert(0, "src")      # 저장소 최상위에서 실행할 때 (pip install -e . 했다면 생략)
   from fashion_tagging.data.labels import load_split_from_zip   # 또는 labels.csv를 parse_labels_csv로 읽기
   from fashion_tagging.data.dataset import FashionDataset, read_names
   from fashion_tagging.data.preprocess import timm_mean_std
   mean, std = timm_mean_std("convnext_tiny.fb_in22k_ft_in1k_384")   # 모델마다 다르다
   train_names = read_names("data/splits/train_inner.txt")
   ```
   `FashionDataset(root, names, targets, size=384, mode="pad", train=True, mean=mean, std=std)`
   (ZIP에서 바로 읽을 때는 `root="....zip", zip_prefix="train"`)
3. 평가: 모델 출력 logits를 `save_predictions`로 저장하고, `evaluate_probs(y_true, sigmoid(logits), 0.5)`로 계산한다.
   임계값을 바꿀 때는 저장한 logits를 불러와 재평가하면 되고 재학습은 필요 없다.
4. 결과는 `configs/results_template.csv` 형식으로 한 줄씩 추가한다.

## 학습·전처리 비교 실행 (GPU 서버)
```
# 기준선 1회 (pad)
python scripts/train.py --exp_id S1_cnxt_t_pad --owner 이름 --train_dir data/train --val_dir data/val

# padding vs 강제 resize 비교 (같은 설정으로 2번 학습 → comparison.md 생성)
python scripts/compare_preprocess.py --prefix S4_cnxt_t --train_dir data/train --val_dir data/val
```
- `--train_dir`/`--val_dir`에는 `labels.csv`와 `images/`가 있는 폴더 또는 ZIP 파일을 줄 수 있다.
- 모델·epoch·batch 등은 `--model --epochs --batch_size --size` 로 바꾼다(비교 스크립트는 나머지 옵션을 그대로 넘긴다). EVA02 같은 ViT는 `--set_img_size` 를 추가한다.
- 결과는 `runs/<exp_id>/`(체크포인트, 예측 .npz, log.json)와 `results/results.csv`(한 줄씩 누적)에 쌓인다.
- 사전학습 가중치는 처음 실행할 때 timm이 내려받는다(서버 인터넷 필요). 오프라인 제출용으로는 `model.pt`에서 불러오도록 따로 바꿔야 한다.
- 이어 학습(resume)은 아직 없다. 중간에 끊기면 처음부터 다시 돌린다.
- 이 스크립트는 팀 기준선과 같은 설정을 쓰는 비교용이다. 2번 담당의 기준 학습 코드가 따로 있다면 그 코드로 두 모드를 돌려도 되고, 그때는 예측 `.npz`만 `compare_preprocess.py --skip_train` 규칙대로 `runs/<prefix>_pad`, `runs/<prefix>_resize`에 두면 비교 부분을 그대로 쓸 수 있다.

## 결과 모으기 (11/9 통합용)
각자 `results.csv`(한 줄 = 실험 1개)와 `runs/<exp_id>/preds_val.npz, preds_holdout.npz`를 공유 폴더에 올린다. 1번이 아래로 합친다.
```
python scripts/aggregate_results.py --csv 공유폴더/*/results.csv --runs_dirs 공유폴더/*/runs --baseline S1_cnxt_t_pad --out report
```
- `report/results_summary.md`(보고서), `results_summary.csv`, `per_class_f1_val.csv`, `per_class_f1_holdout.csv`가 생긴다.
- 맨 위 "확인 필요"에 문제가 모인다: exp_id 중복, smoke 테스트 행(제외됨), 50M 초과 모델, CSV 점수와 예측 파일로 다시 계산한 점수가 다른 경우, 예측 파일 누락, 실험 간 seed·threshold·평가 이미지 수 불일치.
- 예측 파일이 있으면 세 F1을 **다시 계산해서** 표에 쓴다(손으로 옮기다 틀린 값 방지). 없으면 CSV 값을 그대로 쓰고 `검증 no`로 표시한다.
- 표는 holdout 기준으로 정렬한다. 공식 val은 보고용 열이다.
- **exp_id는 팀 전체에서 겹치지 않게** 정한다(예: `S1_cnxt_t_pad`, `S3_eva_asl_378`). 겹치면 마지막 행만 남는다.
- 한 요인만 다른 실험끼리 비교해야 해석이 된다. 결과표의 "설정" 표에서 실험 간에 다른 항목을 같이 보여 준다.

## 오류 이미지 보기
`save_predictions`로 저장한 `.npz`만 있으면 된다. 사례는 아래 규칙으로 자동 선정되고 선정 규칙이 이미지 위와 JSON에 기록된다(보기 좋은 것만 고르지 않기 위해).
```
python scripts/visualize_errors.py --pred preds/exp1_val.npz --images data/val --mode worst --out vis/exp1
python scripts/visualize_errors.py --pred preds/exp1_val.npz --images data/val --mode fn --class watch --out vis/exp1
```
| mode | 고르는 기준 |
|---|---|
| `worst` | 이미지 F1이 가장 낮은 순, 그다음 오류가 많은 순 |
| `fn --class C` | 클래스 C의 정답인데 확률이 가장 낮은 것 (제일 크게 놓친 것) |
| `fp --class C` | 클래스 C가 아닌데 확률이 가장 높은 것 (제일 크게 잘못 낸 것) |
| `random` | 오류가 있는 이미지 중 무작위(seed 고정) |
| `ok` | 맞게 예측한 이미지 중 무작위(성공 사례 비교용) |

ZIP에서 바로 읽을 때는 `--images 파일.zip --zip_prefix val`. 실패 사례를 보고서에 쓸 때는 `worst`/`fn`/`fp`와 `random`을 함께 써서 선택 편향이 없음을 보이자.

## 꼭 지킬 것
- **mean/std는 모델마다 다르다.** ConvNeXt-T는 ImageNet 값, EVA02-S는 (0.481, 0.458, 0.408)/(0.269, 0.261, 0.276), EfficientNetV2-S는 0.5/0.5. `timm_mean_std`로 읽어서 쓴다.
- 체크포인트(early stopping)와 임계값은 **holdout**으로 고른다. 공식 val은 마지막 보고에만 쓴다.
- 손실에 `BCEWithLogitsLoss`를 쓰면 모델 출력에 sigmoid를 미리 적용하지 않는다. sigmoid는 평가·추론 때만.
- softmax/argmax/강제 top-1 금지. 아무것도 안 넘으면 빈 예측 `[]`이 정상이다.
- 클래스 순서는 `categories.txt` 그대로(0~26). 이름을 쉼표로 나누지 않는다.

## 지표 정의 (`metrics.py` 맨 위에도 적혀 있음)
- F1 = 2TP / (2TP + FP + FN)
- Micro: 모든 이미지·클래스의 TP/FP/FN을 합쳐서 계산
- Macro: 27개 클래스 F1의 단순 평균. support가 0이고 예측도 없는 클래스는 기본 0점(`zero_division`)
- Image-wise: 이미지별 F1의 평균. 정답과 예측이 모두 비면 1, 한쪽만 비면 0

## 테스트
```
python tests/test_metrics.py
python tests/test_data.py
```
`test_data.py`, `test_aggregate.py`, `test_visualize.py`는 저장소 최상위에 두 ZIP(`MAT4091_fashion_train.zip`, `MAT4091_fashion_val.zip`)이 있어야 돈다.
