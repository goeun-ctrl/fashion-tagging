# ConvNeXt-Tiny와 EVA02-Small의 비교 조건

ConvNeXt 열은 2026-11-03 팀원 보고서에 기록된 실행값이다. EVA 열은 이 폴더의 `resolved_config.json`, `environment.json`, `model_parameters.json`과 실제 실행 코드 기준이다. 팀원 보고서는 공통 코드의 원본 커밋을 `c5827dfa749e276d4e5431628a06730b26c4841f`로 특정한다. 현재 로컬 `main`의 `8a2de2c`와 대조했을 때 분할 파일, 라벨, Dataset, 전처리 및 F1 코드에 변경이 없었다.

| 항목 | ConvNeXt-Tiny | EVA02-Small | 구분과 해석 |
|---|---|---|---|
| 특징 추출 | CNN, 공간 특징 | Vision Transformer, patch 14 | 구조 차이 |
| 사전학습 | fb_in22k_ft_in1k_384. ImageNet-22k → 1k | mim_in22k_ft_in1k. ImageNet-22k에서 가린 패치 특징 학습 후 ImageNet-1k 미세조정 | 출발 가중치와 학습 방법 차이 |
| 입력 | 384² = 147,456 픽셀 | 378² = 142,884 픽셀 | 체크포인트 기본 336에서 24² → 27² 격자로 위치 임베딩 보간. 비교 입력 면적은 3.1006% 작다 |
| 전처리 | EXIF → RGB → 종횡비 유지 bicubic resize → 검정 중앙 pad → 정규화 | 동일한 공통 함수, size 및 mean/std만 변경 | 전체 시야 통제. 원본 JPEG를 매 로딩 시 한 번 처리 |
| 정규화 평균 | 0.485 / 0.456 / 0.406 | 0.48145466 / 0.4578275 / 0.40821073 | 각 체크포인트 설정 |
| 정규화 표준편차 | 0.229 / 0.224 / 0.225 | 0.26862954 / 0.26130258 / 0.27577711 | 입력 분포 차이 |
| 증강 | train 수평 반전 p=0.5, 평가는 없음 | 동일 | 통제 |
| 분류기 | 공간 평균 → LayerNorm → Linear(768, 27) | CLS 제외 패치 평균 → fc_norm LayerNorm → Linear(384, 27) | 기본 집계 구조 차이. 두 모델 모두 27개 logit 출력 |
| 목적함수·optimizer | BCEWithLogits + AdamW | 동일 | 가중치 없는 이진 교차 엔트로피 |
| LR | backbone 1e-4 / 분류기 1e-3 | backbone 5e-5 / 분류기 1e-3 | 회의록의 모델별 시작값. LR 효과를 분리할 수 없음 |
| Weight decay | 0.05, 분류기 외 LayerNorm 등은 backbone 그룹 | 0.05, bias·norm 포함 모든 파라미터. 분류기 weight·bias만 head LR | 팀 보고서는 모든 그룹의 decay 예외를 개별 열거하지 않음 |
| 배치·AMP | 16장 × 누적 2회, AMP | 16장 × 누적 2회, CUDA FP16 AMP | 유효 배치 32. 마지막 29장은 실제 표본 수 기준 가중 |
| Epoch / seed | 최대 30 / 42 | 동일 | 단일 seed |
| Scheduler | 실제 optimizer 갱신마다 5% warmup + cosine | 동일. 계획 40,500 step, warmup 2,025 step. overflow 때 진전 없음 | 학습 예산 통제 |
| 조기 종료 / 선택 | holdout Macro F1, patience 5 | 동일. 동점은 먼저 나온 epoch 유지 | 공식 val은 최종 보고에만 사용 |
| 데이터 | train_inner 43,197 / holdout 2,425 / 공식 val 1,158 | 동일 명단·클래스 순서 | CSV 및 NPZ를 파일명으로 대조. 분할 간 동일 파일명·동일 바이트 0 |
| 평가 | 공통 metrics, sigmoid ≥ 0.5 | 동일 | 27개 Macro 평균. 이미지 양쪽 공집합이면 1. 클래스 분모 0이면 0 |
| 파라미터 | 27,840,891 | 21,813,147 | 차이 −6,027,744개. 양쪽 모두 50M 이내 |
| 장비 | 공유 RTX A5000, GPU 이동 | 로컬 RTX 4060 8GB, Windows WDDM | wall time·VRAM에 하드웨어, I/O, 캐시 영향 |
| SW | Python 3.12.3 / torch 2.8.0+cu126 / torchvision 0.23.0 / timm 1.0.22 | Python 3.12.10 / torch 2.10.0+cu128 / torchvision 0.25.0 / timm 1.0.22 | 라이브러리 및 커널 구현 차이 |
| 시간 범위 | 누적 train 구간. 평가 forward와 데이터 처리 포함 시간 별도 | 동기화한 train 구간 합. 평가 시 forward만 별도 동기화, 전체 시간에는 첫 배치와 로더 기동 포함 | 장비 간 절대 속도 순위의 일반화 제한 |

## 문서와 실행 코드의 차이

- 저장소의 기본 YAML은 batch 32이며, 공통 CLI 기본값은 32 × 1이다. 2번은 개인 trainer에 16 × 2를 적용했다. EVA도 16 × 2를 사용한다.
- 공통 trainer는 마지막 누적 구간에서도 loss를 고정 누적 횟수로 나누고 AMP skip에서도 scheduler를 호출한다. EVA 전용 trainer는 팀원 보고서에 기록된 방식대로 실제 창 표본 수로 나누고 optimizer가 갱신된 경우만 scheduler를 진행한다. 공통 Dataset, 라벨, 평가 함수, AdamW 그룹 및 scheduler는 직접 재사용한다.
- 2번 개인 trainer, resolved config, 예측 NPZ, best/last 원본은 팀원 보고서에서 경로를 확인했다. 이 PC에서 직접 열어 재검증한 원본 자료와 팀원이 검증해 보고한 자료를 구분한다.
- 기본 336 모델에 378 입력을 주면 정적 크기 검사에서 거부됐다. 별도 CPU 테스트에서 같은 336 가중치를 strict 로드한 동적 모델은 378 입력에 [1, 27]을 출력했다. 실제 실험은 378 고정 모델을 사용했으며, 이 모델에서 GPU 역전파와 저장·복원을 검사했다. 세부 결과는 `input_compatibility.json`에 있다.
- Git은 이미지 식별자가 들어간 두 분할 목록과 클래스별 분할 집계를 이미 추적한다. 현재 추적 파일 목록에서 원본 JPEG·라벨 CSV/NPZ·배포 ZIP을 찾지 못했다. 새 개별 예측·가중치·실행 로그는 이 실험의 `private/`에만 보관한다.

공식 가중치는 MIT 라이선스다. 공식 설정은 336px의 고정 입력, 평균 풀링, EVA 전용 정규화를 명시한다. [모델 카드](https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k) · [체크포인트 설정](https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k/blob/main/config.json) · [배포 SHA-256](https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k/blob/main/model.safetensors)

