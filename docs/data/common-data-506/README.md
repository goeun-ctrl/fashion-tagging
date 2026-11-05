# 공통 데이터 패키지 · 506장 · B안(Masked BCE)

버전: `fashion-common-506-partial-20261105-v1` · 확정일: 2026-11-05 (한국 시간)

## 먼저 알아야 할 결론

정은의 기존 준비본을 채연의 EVA 재학습 코드가 읽는 `fashion-common-v1` 형식으로 변환하고 실제 리더로 검사했다. **패키지 호환성은 PASS, 서버 전체 preflight와 GPU 학습은 미실행**이다. EVA의 실제 재학습 데이터나 성능 결과와 비교한 것이 아니라, GitHub에 구현된 데이터 입력 규약·전처리 코드와 대조했다.

사용자가 이번 채팅에서 **B안(외부 506장 부분 라벨 유지 + 두 모델 모두 Masked BCE)**을 명시 승인했다. 506장의 27개 라벨을 사람이 모두 검수한 A안으로 표시하지 않는다. 외부 라벨 13,662개 중 12,496개(91.4654%)는 미확인(-1)이고 1,166개만 확인값이다. -1을 0으로 바꾸지 않는다. 이 사용 모드 승인은 독립적인 사람 전수 라벨 검수나 출처별 법적 권리 검증을 완료했다는 뜻이 아니다.

## 공유 파일

- [비공개 팀 GitHub Release 다운로드](https://github.com/goeun-ctrl/fashion-tagging/releases/tag/fashion-common-506-partial-20261105-v1)
- `fashion-common-506-partial-20261105-v1.zip`: 아래 전체 패키지, 외부 이미지 506장 포함.
- `fashion-common-506-partial-20261105-v1.shared-lock.json`: 데이터 담당자가 배포하는 독립 대조표. ZIP 내부 `shared-lock.json`과 같은 바이트다.
- `SHA256SUMS.txt`: ZIP과 독립 대조표의 SHA-256. 다운로드 후 먼저 대조한다.

이미지·Manifest·Targets·NPZ는 Git 이력에 추가하지 않고 비공개 Release 자산으로 배포한다. Git에는 이 안내서, 생성/검증 스크립트, 호환성 검증 결과와 독립 대조표만 둔다. 원본 train/val 이미지, 모델 가중치, 개인별 예측, 가상환경은 포함하지 않는다. 기존 준비본과 기준선은 보존했다.

```text
fashion-common-506-partial-20261105-v1/
  training_manifest.csv         # image_id/source/relative_path/approved/image_sha256/sampling_weight/data_version
  training_targets.npz          # image_ids, categories, targets[43703,27], data_version; pickle 불필요
  class_counts.csv              # class,positive; 공식 27개 순서
  augmentation_policy.json      # 정확한 Pillow 버전·정책·Targets SHA
  epoch_plans/epoch_01.npz ... epoch_30.npz
  selected_external/images/     # 선별 506장
  selected_external/labels.csv  # 원래의 1/0/-1 부분 라벨
  selected_external/sources.csv # 원 출처 정보; 결측 정보도 그대로 보존
  selected_external/categories.txt
  fixed_splits/train_inner.txt, holdout.txt
  shared-lock.json              # EVA의 --shared-lock 입력
  package-files.sha256.json     # 이 파일 자신을 제외한 패키지 파일 해시
  compatibility_report.json     # 실제 파일과 EVA 코드의 CPU 대조 결과
  label_review_status.json      # B안, 사람 전수 검수 미완료 명시
  provenance.json               # 버전과 근거 파일 해시
  planned_class_exposure.csv    # 계획 노출량; 실제 학습 실적 아님
  original_augmentation.py      # 기존 준비본의 원 구현
  visual_screening.csv          # Codex 시각 선별 기록; 사람 전수 주석 아님
  verify_package.py             # 네트워크·GPU 없이 파일과 계획 재생성 검사
  eva_cpu_tests.txt             # 합성 CPU 28개 테스트 결과
  README.md
```

## 데이터와 정책: 두 모델이 공유할 것

| 항목 | 이번 확정값 |
|---|---|
| 원본 학습 | 고정 train_inner 43,197장 |
| 외부 | 선별 506장; 전체 Release 9,731장을 모두 사용하지 않음 |
| 합친 고유 이미지 | 43,703장; Manifest 순서: 원본 43,197장 다음 외부 506장 |
| 평가 | holdout 2,425장, 공식 val 1,158장; 학습에 넣지 않음 |
| 클래스/라벨 | 공식 27개 순서; 1=확인 양성, 0=확인 음성, -1=미확인 |
| 손실 | 두 모델 모두 B안 Masked BCE; 누적 창 전체의 확인 라벨 수로 BCE 합 정규화 |
| 선택 | seed 42, epoch당 43,197회 복원 추출, 최대 30 epoch |
| 가중치 | leg warmer 2.5; 양성 합계 1,000 미만 1.75; 지정 중간 빈도 1.30; 나머지 1.0. 이미지당 최대 하나 |
| Flip | 0.5; 계획에 저장된 값을 한 번만 적용 |
| ColorJitter | 밝기/대비 0.90–1.10, 채도 0.95–1.05, hue 0; 집중 그룹 0.35, 일반 보완 0.20, 나머지 0 |
| 변환 순서 | EXIF 보정 → RGB → Flip → 기록된 순서로 색 변환 → 종횡비 유지 bicubic → 검정 중앙 padding → 모델별 정규화 |
| 중복 적용 금지 | shuffle=False; 추가 sampler, ColorJitter, extra_color match 등을 겹쳐 사용하지 않음 |

집중 그룹·일반 그룹·중간 가중치 대상은 정책 JSON에 공식 클래스명으로 명시했다. 최종 부분 라벨로 양성 수와 가중치/그룹을 다시 산출해 30개 계획을 **재생성**했다. 이번에는 라벨·선별 명단이 바뀌지 않아 재생성한 선택/변환 값이 기존 30개 계획과 모두 일치했다. 새 Plan에는 버전·라벨 모드·Manifest/Targets/Policy 해시를 추가했다. 파일 바이트와 SHA는 기존 NPZ와 다르므로 **새 shared-lock 전체를 함께 사용**한다.

첫 epoch: 원본 42,397회, 외부 800회, 고유 이미지 27,186장, leg warmer 양성 237회 선택. 이는 계획값이며 모델 학습 결과가 아니다. 모든 고유 사진이 매 epoch 반드시 등장하는 방식이 아니다.

## 검증한 범위와 남은 단계

검증 기준 EVA commit: `a84528284cc8f8d2ae6177caeee67fee05b92af4` (`feature/eva02-common-data-retrain`). 원본 공통 코드 기준 commit: `c5827dfa749e276d4e5431628a06730b26c4841f`.

- EVA 원본 `package.check_lock/load_training/check_policy/load_plan`으로 43,703행·27라벨·가중치·30개 Plan 검사 PASS.
- 외부 506장 실제 파일의 SHA-256·디코딩 검사 PASS.
- 외부 506장 모두에서 원본 공통 증강과 EVA의 **모델별 리사이즈 직전 픽셀 동일**, EVA 378px 정규화 입력 텐서도 동일.
- 기존 공통 30개 Plan의 선택 ID·중복 횟수·Flip·색상 계수·연산 순서 동일.
- 원본 46,780장 해시 캐시와 Manifest·평가 분리 대조 PASS. 이 부분은 이전 서버 읽기 전용 감사 기록 재사용이며 이번에 서버를 다시 스캔한 것이 아니다. 바이트 동일 중복은 없었고, 시각적으로 비슷한 사진까지 없다는 보장은 아니다. 기존 고정 분할은 임의 변경하지 않았다.
- EVA 합성 CPU 테스트 28개 PASS. 이는 실제 CUDA/AMP 학습 통과를 뜻하지 않는다.
- `complete` 모드는 실제 -1 라벨을 올바르게 거부했다. **반드시 --label-mode partial**로 사용한다.

**남은 서버 작업:** 원본 train/val 폴더의 현재 파일·라벨, 공식 EVA backbone 해시, 실제 경로, 환경을 대상으로 EVA `preflight` 전체 PASS를 확인한다. 실제 ConvNeXt Trainer 연결·서버 미니배치·CUDA/AMP 검사는 후속 작업이다. 이 패키지 검증을 두 모델의 학습 완료 또는 성능 개선으로 해석하지 않는다.

## 다운로드 후 CPU 확인

필요한 패키지: Python 3.11 이상, numpy==2.2.6, Pillow==11.3.0. 기존 환경에 설치돼 있다면 재설치할 필요 없다. PyTorch나 GPU는 이 검증에 필요 없다.

1. GitHub 로그인 상태에서 위 Release의 ZIP·대조표·SHA256SUMS.txt를 받는다.
2. 기존 파일을 덮어쓰지 않는 새 폴더에 압축을 푼다. 파일 SHA-256을 SHA256SUMS.txt와 대조한다.
3. 다음처럼 검사한다(경로는 실제 위치로 변경).

```bash
python fashion-common-506-partial-20261105-v1/verify_package.py fashion-common-506-partial-20261105-v1
```

출력의 package가 PASS이고 regenerated_epoch_plans_equal이 30이어야 한다. 압축을 푼 뒤 NPZ/CSV를 재저장하거나 라벨을 고치면 해시가 달라지므로 새 버전으로 다시 배포해야 한다.

## EVA 담당자 연결 예시

채연의 위 commit/브랜치가 있는 저장소에서 실행한다. 공통 패키지를 쓰기 위해 그 브랜치의 기존 코드나 결과를 덮어쓸 필요 없다. 원본 train/val은 각자의 비공개 보유 데이터를 연결한다. 아래 경로는 예시다.

```bash
python -m experiments.role3_eva02.retrain_common_data.cli preflight \
  --package /srv/fashion/fashion-common-506-partial-20261105-v1 \
  --shared-lock /srv/fashion/fashion-common-506-partial-20261105-v1.shared-lock.json \
  --train-root /srv/fashion/train --val-root /srv/fashion/val \
  --external-root /srv/fashion/fashion-common-506-partial-20261105-v1/selected_external/images \
  --weights /srv/fashion/weights/model.safetensors \
  --run-id eva-common506-b01 --label-mode partial
```

특히 `--external-root`는 **selected_external/images**까지 지정한다. `--package`나 selected_external까지만 주면 이미지 경로가 맞지 않는다. Windows PowerShell에서는 한 줄로 실행하거나 Bash의 `\` 대신 PowerShell 구문을 쓴다.

공식 train/labels.npz SHA: `2558665d02fc7a2299735f8c40ef8bfb8656b2676f7bf0ec910f12e9bf0c19ac`.
공식 val/labels.npz SHA: `c937ea8e70e255892861e5efca4d007c35e6f3feb67361044dd4e346843f469d`.
원본 이미지와 라벨을 서버에 둔 상태에서 전체 preflight가 PASS여야 후속 학습을 시작할 수 있다.

## 다음 재학습에 유지할 조건

두 모델은 기존 baseline best/last를 이어 학습하지 않고, 각각 공식 사전학습 backbone과 새 head/optimizer/scheduler/scaler로 시작한다. 공통 seed 42, 실효 배치 32(16×2), AdamW, weight decay 0.05, 최대 30 epoch, optimizer 업데이트 기준 warmup 5%+cosine, patience 5, threshold 0.5를 사용한다. Holdout Macro F1 최대 모델을 고르고 동점이면 앞선 epoch를 유지하며 공식 val은 최종 보고에만 쓴다.

모델별 차이는 ConvNeXt 384px·backbone LR 1e-4·해당 체크포인트 정규화, EVA 378px·backbone LR 5e-5·EVA 정규화다. 두 모델의 계획과 증강 전후 사진 크기까지 같아야 한다는 뜻은 아니다. 같은 원본과 동일한 크기 변경 전 변환을 공유한다.

이번 배포 작업에서는 학습·추론을 실행하지 않았고 서버·GPU 작업을 변경하지 않았다. 새 성능 결과는 없다.
