# Role 3 · EVA02-Small 백본 비교

`FashionDataset`, `load_split`, `preprocess`, `evaluate_probs`, timm 기본 분류기와 공통 AdamW·cosine 구현을 사용한다. 모든 실행은 수업 원본이 있는 비공개 Windows PC를 기준으로 한다. 아래 명령은 Git 저장소 루트에서 PowerShell로 실행한다.

## 최초 설치

Python 3.12에서 프로젝트 전용 `.venv`를 만든 뒤 다음 명령을 실행한다. 검증 환경은 `environment.json`에 기록되어 있다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install timm==1.0.22 matplotlib
New-Item -ItemType Directory -Force experiments\role3_eva02\private | Out-Null
curl.exe -fL -o experiments\role3_eva02\private\model.safetensors https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k/resolve/main/model.safetensors
```

공식 파일의 SHA-256은 `config.json`에 고정되어 있다. 실행 시 해시, classifier 외 모든 backbone 텐서의 로드, 위치 임베딩 보간 결과를 대조한다. 값이 다르면 학습 시작 전에 종료한다. [공식 모델 및 라이선스](https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k)

## 학습·평가·추론

```powershell
$env:HF_HUB_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
$py='.\.venv\Scripts\python.exe'
$run='experiments\role3_eva02\run.py'
& $py $run preflight
& $py $run train
# 마지막 완료 epoch부터 중단된 단일 실행을 재개할 경우
& $py $run train --resume
# 완료된 best를 다시 평가하고 그림과 보고서를 재생성할 경우
& $py $run evaluate
# 로컬 이미지 한 장의 카테고리 배열 출력
& $py $run infer --image 'C:\경로\image.jpg'
```

`train` 자체가 사전검증을 다시 수행한다. PC의 프로젝트 상위 폴더에 있는 `MAT4091_fashion_train/train`, `MAT4091_fashion_val/val`과 Git에 고정된 두 분할 목록만 읽는다. 무게 파일을 받은 뒤에는 학습·평가·추론을 오프라인으로 실행할 수 있다. 추론은 보관한 `private/baseline/best.pt`의 모델·클래스 순서·정규화를 복원한다. 예측이 없으면 빈 배열을 출력한다.

## 기록 위치

- `data_audit.json`은 분할별 개수, 명단 해시, CSV/NPZ 일치 여부와 분할 간 바이트 중복 검사 결과다.
- `model_parameters.json`의 synthetic BCE와 gradient는 학습 전 시스템 검증값이다. 모델 성능은 본 학습 종료 후 생성되는 `metrics_holdout.json`과 `metrics_val.json`에 기록한다.
- `training_history.csv`는 매 완료 epoch에 갱신한다. `private/baseline/best.pt`와 `last.pt`는 epoch 단위로 원자적 교체하며 재개에 필요한 optimizer·scaler·scheduler·난수 상태를 포함한다. 학습 중인 프로세스는 한 개만 실행한다.
- `private/baseline/preds_*.npz`에 로짓, 확률, 타깃과 식별자를 로컬 저장한 뒤 세 F1과 클래스 지표를 재계산한다. 외부 공유용 산출물은 집계 CSV/JSON, 그래프, 한국어 Markdown이다.
- `convnext_reference.json`은 2번 팀원 보고서의 집계값이다. 보고서에 공개된 7개 클래스는 TP/FP/FN에서 P/R/F1을 계산한다. sock·watch는 반올림 F1만 사용한다. 전체 공식 val 클래스 집계가 있으면 `private/convnext_val.csv`에 UTF-8 CSV 열 `class,tp,fp,fn`으로 넣고 `evaluate`를 실행한다. 27개가 모두 있을 때 평균 클래스 Δ와 전체 Macro Δ를 검증한다.
- 비교 해석과 두 실험의 설정 차이는 `comparison_conditions.md`에 적었다.

## 이번 실행의 검증 기록

- 로컬에 있던 PyTorch는 CPU 전용이었다. 프로젝트 전용 Python 3.12.10 환경에 CUDA 12.8 빌드를 설치해 RTX 4060으로 실행했다. WindowsApps 실행 별칭 대신 실제 Python 실행 파일을 사용했다.
- 공통 F1 테스트는 임시 폴더의 Windows 권한 문제를 해소한 뒤 모두 통과했다. 합성 검사에서는 AMP overflow 때 optimizer/scheduler가 함께 멈추고 다음 유한 gradient에서 함께 진행하는 것을 확인했다. 본 학습의 AMP skip은 전체 15 epoch에서 6회였다.
- 이번 실행의 콘솔 로그에서 epoch 행의 `macro`는 그때까지의 최고값이다. epoch별 실제 점수는 `training_history.csv`의 `holdout_macro`다. 현재 실행 코드는 콘솔에 그 회차 값과 최고값을 구분한다.
- 저장한 두 분할의 확률에서 세 F1과 클래스별 support·TP·FP·FN을 독립 계산해 결과 CSV와 대조했다. Best/last 내부 epoch 10/15를 확인했다. 소켓 연결을 차단한 별도 프로세스에서 실제 `infer` 명령으로 합성 이미지 1장을 입력해 유효한 이름 목록을 얻었다. 평가에는 CUDA FP16, 오프라인 CPU 추론에는 FP32를 사용한다.

