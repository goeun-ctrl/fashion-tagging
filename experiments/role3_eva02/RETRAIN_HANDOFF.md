# EVA02-Small 최종 재학습 인수인계

**정은이의 실제 데이터 패키지와 `fashion-common-v1`의 호환성은 미검증(`BLOCKED`)이다.** 서버 담당자는 아래 연결 규약에 따라 원본 패키지를 대조하고, 팀이 승인한 입력에 대해 `preflight` 전체가 `PASS`일 때 학습을 시작한다. 이번 검수는 합성 CPU 테스트와 코드·CLI 확인까지 진행했다.

## 무엇을 학습하는가

Train_inner에 최종 승인된 외부 이미지만 합친다. 준비 보고 기준 43,197 + 506 = 43,703장은 검수 전 후보다. 최종 외부 수와 클래스별 양성 수는 실행 때 집계한다. Git에 고정된 Holdout 2,425장은 모델 선택에, 공식 Val 1,158장은 학습 완료 후 보고에 사용한다.

| 설정 | 재학습 동작 |
|---|---|
| A안 (기본) | `--label-mode complete`. 사람 검수가 끝난 27개 0/1 라벨에 BCEWithLogitsLoss. 하나라도 미확인(-1)이면 차단한다. |
| B안 (팀 승인 시) | `--label-mode partial`과 부분 라벨용 승인 패키지가 모두 필요하다. -1 위치를 제외하고 **누적 구간 전체의 확인된 원소 수**로 BCE 합을 나눈다. 전부 미확인이면 optimizer와 scheduler를 건너뛴다. |
| 이미지 선택 | Seed 42, 복원추출, epoch당 43,197회. 양성 클래스에 해당하는 가중치의 최댓값을 쓴다. leg warmer 2.5, 최종 양성 1,000 미만 1.75, 지정 중간 빈도 개선 클래스 1.30, 그 밖은 1.0. 학습 코드는 두 모델의 공통 Plan을 순서대로 재생한다. |
| 증강 | EXIF 보정 → RGB → Plan의 수평 반전(p=0.5)과 색상 계수 → 종횡비 유지 bicubic → 검정 중앙 padding → EVA 정규화. 밝기·대비 0.90–1.10, 채도 0.95–1.05, hue 0. 집중/부족 클래스 0.35, 일반 보완 0.20, 유지 0 중 이미지당 최대 확률 하나를 사용한다. Holdout과 Val은 결정적 전처리만 쓴다. |
| 모델 | `eva02_small_patch14_336.mim_in22k_ft_in1k`, 378², CLS 제외 패치 평균 → LayerNorm → Linear(384, 27). Mean (0.48145466, 0.4578275, 0.40821073), Std (0.26862954, 0.26130258, 0.27577711). |
| 초기화·최적화 | 공식 해시의 backbone, 새 분류기·AdamW·scheduler·scaler. 기존 EVA의 위치 임베딩 보간·RoPE와 가중치 로더를 재사용한다. Backbone LR 5e-5, head LR 1e-3, decay 0.05, 16장 × 누적 2회. CUDA FP16, 5% warmup + cosine. AMP overflow 시 scheduler도 대기한다. |
| 모델 선택 | 최대 30 epoch, patience 5. Holdout Macro F1이 엄격히 증가할 때만 best를 바꾼다. 동점이면 앞선 epoch를 유지한다. 모든 평가에 sigmoid ≥ 0.5를 쓴다. |

## 파일을 어디에서 받는가

| 제공처 | 준비물 |
|---|---|
| Git | `common/splits/train_inner.txt`, `holdout.txt`, `split_summary.csv`, 공통 Dataset·전처리·평가 모듈, 기존 EVA 설정·코드, `retrain_common_data/`, 이 문서 |
| 과제 원본/서버 담당자 | 압축 해제한 원본 `train/`, `val/`. 각각 `categories.txt`, `labels.csv`, `labels.npz`, `images/`를 포함한다. 클래스 이름에 쉼표가 있으므로 한 줄 전체가 한 클래스다. |
| 정은이/데이터 담당자 | 최종 계획서와 준비 결과서, 원본 공통 `augmentation.py`와 정책, 고정 분할·공식 Val 기준, 승인된 외부 이미지와 최종 27라벨, Manifest, Targets, 클래스별 집계, 해당 라벨판에서 생성한 30개 Plan. 두 모델에 배포할 신뢰된 버전·SHA-256 대조표도 함께 받는다. |
| 서버 담당자 | 해시가 기존 `config.json`의 `weight_sha256`과 일치하는 공식 `model.safetensors`, CUDA PyTorch 환경, 두 모델이 합의한 Pillow 버전, 비공개 데이터와 결과를 보관할 디스크 공간 |

후보 이미지를 27클래스로 검수하기 전에는 A안을 시작할 수 없다. 검수가 끝나면 양성 수, 증강 그룹, 가중치, **30개 Plan 모두** 다시 산출한다. 기존 부분 라벨용 Plan의 헤더나 해시만 바꿔 재사용하면 실험 조건이 달라진다.

## 공통 패키지 연결 규약

아래 `fashion-common-v1`은 **이 코드가 읽는 제안 교환 형식**이다. 정은이의 원본 키, 자료형, 연산 순서는 미확인이다. 실제 패키지를 받으면 원본 생성기·증강 구현과 대조하고, 차이가 있으면 양쪽 모델에서 합의한 변환본 또는 리더를 사용한다. 선택 순서·중복 횟수·원본 ID·라벨·Flip·계수·연산 순서가 보존되는지 검사한다. 같은 가짜 RGB 입력에서 모델별 리사이즈 **직전** 픽셀도 원본 증강 코드와 대조한다. 정은이의 `masked_bce`가 제공되면 확인 원소의 합과 개수를 기준으로 현재 CPU gradient 테스트와 대조해 연결한다. 실제 패키지와 동등성이 확인된 뒤 데이터 담당자가 대조표를 승인한다.

합성으로 완성한 파일 구조와 값의 예시는 `retrain_common_data/test_retrain.py`의 `SyntheticRelease`에 있다. 이 fixture의 2개 Plan과 5회 draw는 테스트 전용이다.

| 위치 | 필수 구조와 의미 |
|---|---|
| 독립 배포 `--shared-lock` | UTF-8 JSON. `schema="fashion-common-v1"`, `approved=true`, 비어 있지 않은 `data_version`, `label_mode`, 공식 순서의 문자열 배열 `categories`, `seed=42`, `replacement=true`, `epochs=30`, `draws=43197`, `sha256` 객체. 승인은 해당 모드의 라벨 검수, 외부 목록 및 생성 조건에 대한 팀 승인이다. |
| `sha256` 키 | 패키지 루트 기준 `training_manifest.csv`, `training_targets.npz`, `augmentation_policy.json`, `class_counts.csv`, `epoch_plans/epoch_01.npz` … `epoch_30.npz` 각 파일. 또한 과제 공식 원본 `train/labels.npz`, `val/labels.npz`의 해시. 값은 파일 원바이트의 64자 소문자 SHA-256. CSV 개행·NPZ 재저장으로 바이트가 바뀌면 해시도 달라진다. |
| `training_manifest.csv` | 열 `image_id,source,relative_path,approved,image_sha256,sampling_weight,data_version`. ID는 전체에서 유일. 원본은 `source=train_inner`, ID와 상대 경로는 고정 파일명. 외부는 `source=external`, 외부 루트 기준 경로. 승인행만 `approved=1`. 양쪽 모두 실제 이미지 바이트 해시와 최종 가중치를 담는다. 경로 구분자는 파일 안에서 `/`를 사용한다. |
| `training_targets.npz` | Unicode 문자열 `image_ids[N]`, `categories[27]`, scalar 문자열 `data_version`, 숫자 `targets[N,27]`. ID 순서는 Manifest 행 순서와 일치해야 한다. 부분 라벨은 -1을 표현할 수 있는 자료형으로 저장한다. 모든 NPZ는 `allow_pickle=False`로 읽는다. |
| `class_counts.csv` | 공식 순서의 27행, 열 `class,positive`. 최종 Targets에서 값이 1인 항목을 재집계해 대조한다. |
| `augmentation_policy.json` | `data_version`, `targets_sha256`, `replay="pillow-enhance-v1"`, `pillow_version`(양쪽에서 합의한 정확한 설치 버전), `flip_p=0.5`, `hue=0`, `ranges=[[0.9,1.1],[0.9,1.1],[0.95,1.05]]`, `jitter_p=[0.35,0.2,0.0]`. `focus`는 최종 집중·부족 클래스 이름 배열, `general`은 일반 보완, `medium_weight`는 1.30 가중치 대상 배열이다. 나머지 클래스의 jitter 확률은 0이다. 클래스 이름은 공식 표기를 그대로 쓴다. |
| `epoch_plans/epoch_XX.npz` | 01부터 시작. scalar `epoch,data_version,label_mode,manifest_sha256,targets_sha256,policy_sha256`. 길이 D=43,197인 정수 `indices`(0부터 시작하는 Manifest 행), bool 또는 정수 0/1 `flip`. 실수 `factors[D,3]`의 열은 밝기·대비·채도, 미적용은 [1,1,1]. 정수 `order[D,3]`의 각 행은 이 세 열 번호 [0,1,2]의 순열이다. |

양성 수가 1,000장 **미만**인지로 1.75 가중치를 계산하고, 집중·일반 증강 그룹은 최종 라벨로 다시 검토한 정책에서 읽는다. 한 이미지의 여러 양성 클래스에는 최대 가중치와 최대 증강 확률을 각각 사용한다. Plan에서 선택과 변환이 확정되므로 학습 loader는 `shuffle=False`이고 별도 sampler나 무작위 증강을 추가하지 않는다.

데이터 담당자가 두 모델에 **같은 승인 대조표**를 준다. 학습 기계에서 받은 파일의 해시를 그대로 복사해 기준을 만들면 원본 일치를 입증할 수 없다. Preflight는 파일 해시·계수 범위·순서·고정 분할·원본 라벨·ID와 바이트 중복을 검사한다. Plan 생성 난수의 분포와 이미지의 시각적 근접 중복·27라벨의 의미상 정확성은 패키지 준비 및 사람 검수에서 확인한다. 두 모델은 실행 중 같은 읽기 전용 패키지와 이미지 버전을 사용한다.

## 서버 명령

Git 저장소 루트에서 실행한다. 예시의 `/srv/fashion`과 `D:\fashion`은 서버의 비공개 데이터 경로로 바꾼다. `--train-root`와 `--val-root`는 각각 `images/`와 `labels.npz`를 바로 포함하는 폴더다. `--external-root`는 외부 이미지의 Manifest 상대 경로가 시작되는 폴더다. `--package`는 Manifest·정책·`epoch_plans/`가 있는 폴더이며 `--shared-lock`과 `--weights`는 각각 파일을 가리킨다.

아래 설치 예시는 기존 EVA의 Python 3.12 / CUDA 12.8 환경과 승인된 정책 파일을 기준으로 한다. 서버 드라이버와 호환되는 PyTorch 빌드를 사용한다.

### Linux Bash

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pillow_version=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8-sig"))["pillow_version"])' /srv/fashion/package/augmentation_policy.json)
python -m pip install timm==1.0.22 numpy "Pillow==$pillow_version" safetensors
export HF_HUB_OFFLINE=1
module=experiments.role3_eva02.retrain_common_data.cli
opts=(--package /srv/fashion/package --shared-lock /srv/fashion/approved.lock.json
      --train-root /srv/fashion/train --val-root /srv/fashion/val
      --external-root /srv/fashion/external --weights /srv/fashion/weights/model.safetensors
      --run-id eva-final-a01 --label-mode complete)
python -m "$module" preflight "${opts[@]}"
```

서버 담당자가 `PASS`를 확인한 뒤 아래에서 상황에 맞는 명령 하나를 선택한다.

```bash
python -m "$module" train "${opts[@]}"                     # 새 실행. 완료 후 best로 두 분할 결과 저장
python -m "$module" train "${opts[@]}" --resume            # 중단된 같은 실행의 마지막 완료 epoch에서 복구
python -m "$module" evaluate "${opts[@]}" --split holdout  # 완료된 best의 Holdout 재평가
python -m "$module" evaluate "${opts[@]}" --split val      # 완료된 best의 공식 Validation 재평가
out=experiments/role3_eva02/private/retrain/eva-final-a01
python -m "$module" compare --eva-config "$out/resolved_config.json" --peer-config /srv/fashion/convnext/resolved_config.json
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
$policy = Get-Content -LiteralPath 'D:\fashion\package\augmentation_policy.json' -Raw -Encoding utf8 | ConvertFrom-Json
& $py -m pip install timm==1.0.22 numpy "Pillow==$($policy.pillow_version)" safetensors
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$module = 'experiments.role3_eva02.retrain_common_data.cli'
$opts = @(
  '--package','D:\fashion\package','--shared-lock','D:\fashion\approved.lock.json',
  '--train-root','D:\fashion\train','--val-root','D:\fashion\val',
  '--external-root','D:\fashion\external','--weights','D:\fashion\weights\model.safetensors',
  '--run-id','eva-final-a01','--label-mode','complete'
)
& $py -m $module preflight @opts
```

```powershell
& $py -m $module train @opts                      # 새 실행
& $py -m $module train @opts --resume             # 중단 복구
& $py -m $module evaluate @opts --split holdout   # 완료된 best 재평가
& $py -m $module evaluate @opts --split val
$out = 'experiments\role3_eva02\private\retrain\eva-final-a01'
& $py -m $module compare --eva-config "$out\resolved_config.json" --peer-config 'D:\fashion\convnext\resolved_config.json'
```

B안은 팀의 명시적 승인 뒤 옵션의 모드·패키지·대조표를 모두 B안으로 바꾸고 새 run ID를 사용한다. 합성 검사는 같은 가상환경에서 `python -m unittest -v experiments.role3_eva02.retrain_common_data.test_retrain`으로 재현한다.

### Preflight 결과 읽기

- **PASS (종료 코드 0)**. 명시된 검사가 통과했다. 서버 학습·평가는 CUDA 사용 가능 여부도 별도로 확인한다.
- **FAIL (1)**. 공급된 값이 기준과 충돌한다. 예를 들어 A안의 -1, 이름·바이트 중복, 금지된 분할, stale Plan, 정책·해시 불일치다.
- **BLOCKED (2)**. 파일 또는 선행 검증이 없어 판단할 수 없다. 필요한 자료를 받은 뒤 다시 검사한다. FAIL이 하나라도 있으면 전체 상태는 FAIL이다.

Preflight는 stdout에 JSON을 출력하고 모델을 생성하지 않는다. `train`과 `evaluate`도 매번 같은 검사를 통과해야 한다. 첫 실행은 새 전용 폴더만 허용한다. 첫 epoch 전에 중단되어 `last.pt`가 없으면 새 run ID로 시작한다. Resume은 **같은 run·같은 승인 데이터·같은 설정**의 `last.pt`에서 시작하고 optimizer·scheduler·AMP scaler·난수 상태를 복원한다. 중단된 epoch는 처음부터 같은 Plan을 재생한다.

## 결과와 비교

어느 epoch든 실제 optimizer 업데이트가 0회면 해당 epoch의 Holdout 평가와 체크포인트 저장 전에 중단한다. 이때 확인 라벨 수와 AMP overflow를 점검한다. 중도 실행에 최종 평가 명령을 주면 종료 조건을 먼저 확인하고 결과 파일을 쓰기 전에 중단한다.

산출물은 Git에서 제외된 `experiments/role3_eva02/private/retrain/<run-id>/`에 모인다.

| 파일 | 확인할 값 |
|---|---|
| `resolved_config.json`, `data_audit.json`, `environment.json` | 최종 설정·모드·버전, 공유 SHA-256, 고정 분할, 최종 외부 수·양성 수·가중치별 수, 실제 서버 환경 |
| `training_history.csv`, `actual_class_exposure.csv` | 매 완료 epoch의 손실·Holdout F1·optimizer step·AMP skip. 노출량은 해당 epoch에서 실제 소비한 draw 기준 클래스별 양성·확인·미확인 수다. 복원추출 중복을 포함한다. |
| `best.pt`, `last.pt` | 각각 최고 Holdout Macro F1과 마지막 완료 epoch. 이전 Baseline epoch 10은 비교 자료로만 보관한다. |
| `metrics_holdout.json`, `metrics_val.json` | Micro/Macro/Image-wise F1, BCE, best epoch, 기존 EVA 대비 세 F1의 차이 |
| `per_class_holdout.csv`, `per_class_val.csv` | 매 분할의 27클래스 support, TP, FP, FN, precision, recall, F1 |
| `preds_holdout.npz`, `preds_val.npz` | 비공개 이미지 ID, 정답, logit, 클래스 순서. 개별 예측과 체크포인트는 비공개 폴더에 둔다. |

ConvNeXt 담당자는 **자신의 입력에서 검증한** 동일 규약의 `shared` 객체와 실제 사용한 최상위 `label_mode`를 resolved config에 기록한다. `compare`는 Manifest·Targets·Policy·각 Plan의 해시와 버전, 라벨 모드, 27클래스 순서, epoch 길이 등 공통 항목을 전부 대조한다. 파일 해시가 다르면 해당 파일명도 출력한다. 두 모델의 `compare`가 PASS인 실행끼리 같은 분할의 `metrics_*.json`을 대조한다. peer 기록이 없으면 비교 상태는 BLOCKED다.

기존 **EVA Baseline** 비교 기준은 아래와 같다. 재학습 성능은 서버 학습 완료 후 생성되는 결과로 기록한다.

| 기존 분할 | Micro F1 | Macro F1 | Image-wise F1 |
|---|---:|---:|---:|
| Holdout | 0.804340 | 0.708463 | 0.804744 |
| 공식 Validation | 0.811111 | 0.632394 | 0.815633 |

## 서버 담당자 체크리스트

- [ ] A/B 모드를 양쪽 담당자가 합의했다. A안은 외부 이미지의 27개 라벨 검수가 끝났고, B안은 별도 승인을 받았다.
- [ ] 원본 Train·Val, 승인 외부 이미지, 최종 Manifest·Targets·클래스 집계와 파일별 대조표를 받았다.
- [ ] 최종 라벨에서 정책·가중치를 갱신하고 30개 Plan을 다시 생성했다. 원본 패키지와 교환 형식 간 선택·증강 동등성을 대조했다.
- [ ] 두 모델에 같은 패키지 및 대조표를 배포했고, 공식 backbone 해시와 개인 실행 경로를 확인했다.
- [ ] Preflight의 **모든 항목**이 PASS다. 최종 외부 수와 클래스별 양성 수를 검수 기록과 대조했다.
- [ ] 새 run ID로 서버 학습을 시작한다. 재개 시 해당 run의 last를 사용한다. 원본 이미지·라벨·Plan·개별 예측·가중치는 비공개로 보관한다.
