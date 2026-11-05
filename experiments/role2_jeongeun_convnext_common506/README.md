# ConvNeXt-Tiny · 공통 506장 B안 재학습 코드

현재 정은이 서버에서 실행 중인 `S2_jeongeun_cnxt_t_common506_b_384_s42`의 **실제 코드·설정 7개 파일**을 공유한다. 업로드 전 서버 파일을 읽어 로컬 파일과 SHA-256이 모두 일치함을 확인했다. `snapshot/` 파일은 수정하지 않았으며 `running_code_hash_verification.json`에 대조표를 기록했다. 성능 결과·학습 로그·가중치·데이터는 이번 코드 공유에 포함하지 않는다.

## 중요한 구분

- `snapshot/trainer.py`, `augmentation.py`, `augmented_data.py`, `training_utils.py`, `config.json`: 실제 학습에 사용 중인 코드와 설정.
- `snapshot/pipeline.py`, `snapshot/preflight.py`: 정은의 원래 서버 전용 실행 코드 기록. **서버 절대 경로와 GPU 0이 고정돼 있으므로 다른 팀원은 이 두 파일을 실행하지 않는다.**
- `run_peer.py`: 위 원본 trainer를 바꾸지 않고 팀원별 경로·GPU·실험 ID·담당자를 연결하는 새 실행 도구. 학습을 자동 시작하지 않고 기본 단계는 CPU preflight다. 원본 코드 해시를 확인하고 동일 run ID의 중복 실행을 잠금으로 막는다.
- `prepare_reference.py`: 저장소의 현재 브랜치를 바꾸지 않고, 실제 실행에 사용한 과거 common 코드를 새 폴더에 준비한다.

이것은 ConvNeXt 실행 코드다. EVA 담당자는 EVA 코드를 유지하고 **같은 공통 데이터 패키지**를 연결한다. 두 아키텍처의 코드를 동일하게 바꾸라는 의미가 아니다.

## 재현 조건

| 항목 | 실행 설정 |
|---|---|
| 공통 패키지 | `fashion-common-506-partial-20261105-v1`, 승인 B안 |
| 이미지/분할 | 고유 후보 43,703장(원본 train_inner 43,197 + 외부 506), holdout 2,425, 공식 val 1,158 |
| 라벨 | 공식 27개 순서; 외부 -1은 미확인으로 유지 |
| 모델 | `convnext_tiny.fb_in22k_ft_in1k_384`, pretrained=True, GAP + LayerNorm + Linear(27), logits |
| 입력 | RGB, 종횡비 유지 bicubic, 정규화 전 검정 중앙 padding, 384×384, 체크포인트 mean/std |
| 증강/선택 | 공통 30개 Plan의 이미지 순서·Flip·색상 계수·연산 순서를 그대로 재생. shuffle=False, 추가 sampler/증강 금지 |
| 손실 | 확인 라벨만 BCE; 누적 창 전체의 확인 라벨 수로 BCE 합을 나눔. `config.loss=bce`는 기본 BCE 연산 이름이며 미확인 라벨을 음성으로 넣는다는 뜻이 아님 |
| 최적화 | AdamW, backbone 1e-4 / head 1e-3, weight decay 0.05 |
| 예산 | batch 16 × accumulation 2 = 32; 최대 30 epoch; epoch당 43,197 draw; seed 42 |
| 스케줄 | 실제 성공 optimizer 업데이트 기준 5% warmup + cosine. AMP overflow skip 시 scheduler 대기 |
| 선택/평가 | holdout Macro F1 최대, 동점은 앞 epoch, patience 5, threshold 0.5. 공식 val은 best 선택 이후 최종 보고만 |
| 시작점 | 원래 사전학습 가중치에서 새 head/optimizer/scheduler/scaler. 기존 baseline best를 resume하지 않음 |

`snapshot/config.json`의 exp_id·owner·경로 관련 메모는 원 실행 당시 기록이다. 팀원 실행 도구는 담당자와 별도 exp_id를 적용하고 실제 peer_config.json을 출력 폴더에 남긴다. 원본 설정 파일은 덮어쓰지 않는다.

## 1. 준비물

권장 실행 환경은 현재 서버와 같은 Linux + CUDA GPU다. 새 도구의 CLI 확인은 로컬 CPU에서 했지만, 다른 팀원 장비에서 GPU 학습을 대신 실행한 것은 아니다. 기존 학습 프로세스·원본 데이터·가상환경을 종료하거나 변경하지 않는다.

- 원본 `train/`, `val/`: 각각 images/, categories.txt, labels.csv, labels.npz.
- [공통 패키지 Release](https://github.com/goeun-ctrl/fashion-tagging/releases/tag/fashion-common-506-partial-20261105-v1)의 ZIP·독립 lock·SHA256SUMS.txt. [패키지 상세 안내](../../docs/data/common-data-506/README.md).
- 실제 서버는 Python 3.12.3, torch 2.8.0+cu126, torchvision 0.23.0+cu126, timm 1.0.22, numpy 2.2.6, Pillow 11.3.0, matplotlib 3.10.7. 기존 호환 CUDA 환경을 재사용한다. 드라이버·시스템 설정을 변경하지 않는다.
- **Pillow 11.3.0**을 맞춰 동일한 색상 변환을 재생한다. `requirements_reference.txt` 참고. CUDA PyTorch는 장비에 맞는 빌드를 별도로 확인한다.
- timm 공식 사전학습 가중치 캐시 또는 첫 다운로드를 위한 네트워크. 인증정보를 코드에 넣지 않는다.
- 본인에게 배정되거나 사용 허용된 GPU 번호, 다른 실험과 충돌하지 않는 run ID.

## 2. 실제 사용한 common 코드 준비

저장소 루트에서 실행한다. 출력 폴더가 이미 있으면 덮어쓰지 않는다. 기존의 검증된 같은 reference가 있다면 아래 단계 대신 그 경로를 사용해도 된다.

```bash
python experiments/role2_jeongeun_convnext_common506/prepare_reference.py \
  --repo . \
  --out experiments/role2_jeongeun_convnext_common506/private/reference/c5827dfa
```

원본 commit은 `c5827dfa749e276d4e5431628a06730b26c4841f`다. 최신 main의 경로 정리와 무관하게 **현재 실행의 평가·전처리 정의를 그대로 재현**하기 위한 것이다. 이 commit의 텍스트는 CRLF이지만 실제 서버 reference는 LF이므로, 생성기는 텍스트 개행을 LF로 맞추고 해시를 기록한다. 코드 내용과 split 순서는 바꾸지 않는다. 필요한 commit을 fetch할 수 있지만 팀원 브랜치를 checkout하거나 공통 코드를 덮어쓰지 않는다.

## 3. CPU 검사 → 작은 GPU 테스트 → 본학습

아래는 **Linux Bash 예시**다. 경로·담당자·run ID·GPU는 본인의 값으로 바꾼다. 특히 원 실행이 GPU 0이라는 이유로 같은 GPU를 임의 선택하지 않는다. 공통 패키지에는 원본 train/val 사진이 포함돼 있지 않다.

```bash
runner=experiments/role2_jeongeun_convnext_common506/run_peer.py
opts=(--package /srv/fashion/fashion-common-506-partial-20261105-v1
      --reference-dir experiments/role2_jeongeun_convnext_common506/private/reference/c5827dfa
      --train-dir /srv/fashion/train --val-dir /srv/fashion/val
      --run-id cnxt_peer_b01 --owner 팀원이름)

# GPU를 사용하지 않고 패키지·클래스 순서·라벨·고정 split·현재 이미지 바이트를 검증
python "$runner" --stage preflight "${opts[@]}"

# 반드시 실제 허용된 번호로 바꿀 것. 명시하지 않으면 GPU 학습은 거부됨
GPU=본인에게_허용된_GPU번호
python "$runner" --stage smoke "${opts[@]}" --gpu "$GPU"

# 같은 run ID의 smoke 완료를 확인한 후 본학습
python "$runner" --stage train "${opts[@]}" --gpu "$GPU"
```

Smoke는 학습 128 draw·holdout 64장·1 epoch이며 공통 Plan의 실제 외부 draw도 포함한다. 공식 val은 평가하지 않는다. smoke 점수는 최종 성능에 포함하지 않는다. 메모리 부족 시 새 run ID로 `--batch-size 8 --accum 4` 등 실효 32를 유지해 smoke부터 확인한다. 이미 만들어진 run의 설정을 조용히 바꾸지 않는다.

원격 연결이 끊겨도 유지하려면 기존 tmux/screen을 사용하거나, 위 smoke가 끝난 뒤 다음과 같이 실행할 수 있다.

```bash
nohup python "$runner" --stage train "${opts[@]}" --gpu "$GPU" > cnxt_peer_b01.launcher.log 2>&1 < /dev/null &
tail -n 30 experiments/role2_jeongeun_convnext_common506/private/cnxt_peer_b01/train.log
```

## 4. 재개와 결과 위치

```bash
# 실행 중 프로세스가 없고, 이 동일 run의 last.pt가 있을 때만 명시적으로 재개
python "$runner" --stage train "${opts[@]}" --gpu "$GPU" --resume
```

별도 출력 폴더: `experiments/role2_jeongeun_convnext_common506/private/<run-id>/`.

- `preflight.json`, `peer_config.json`, `launch_record.json`, `smoke.log`, `train.log`.
- `smoke/`: 작은 테스트 결과. `run/`: 본학습 결과.
- `run/best.pt`, `last.pt`: `model` 키에 가중치, optimizer/scheduler/scaler/RNG까지 저장. epoch 단위 재개 가능.
- `run/history.csv`, `learning_curves.png`, `summary.json`, `results.csv`.
- `run/preds_holdout.npz`, `preds_val.npz`, `per_class_holdout.csv`, `per_class_val.csv`, 오류 CSV.
- `run/actual_epoch_exposure/`: 실제 완료 epoch의 이미지·클래스 노출량.
- `run/resolved_config.json`: 공통 `shared` lock과 `label_mode=partial`, 실제 사용한 설정·라이브러리 기록.

이 폴더는 Git에서 제외된다. 실행 도구 자체가 Git commit/push를 하지 않는다. 중간에 학습 프로세스가 살아 있으면 새로 실행하지 않는다. 첫 epoch 이전 실패로 last.pt가 없으면 원인을 확인하고 새 run ID로 smoke부터 진행한다.

## 검증 범위

서버 실행본 7개 파일의 SHA-256 동일성을 확인했다. 이 원본 코드로 실제 서버 데이터의 CPU 검사, 실제 384px DataLoader, 부분 라벨 보존·누적 gradient 검사와 GPU smoke를 통과한 뒤 본학습을 실행 중이다. **이번 공유는 그 코드의 업로드이며, 학습 완료·최종 성능 공개를 뜻하지 않는다.** 경로를 일반화한 팀원용 launcher는 별도 도구이므로 각 팀원 서버에서 preflight/smoke를 다시 통과해야 한다. 원래 서버의 pipeline.py는 최종 한국어 요약까지 자동 생성하지만, 팀원용 launcher는 원본 trainer의 CSV/JSON/예측·그림 저장까지 수행한다.
