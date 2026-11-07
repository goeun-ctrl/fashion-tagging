# C1 BCE·ASL GPU 실행 기록

사용자 승인: GPU 현황·규칙 확인 → 한 장 고정 → 소량 테스트 → BCE/ASL 각18epoch 순차 → holdout 선택 → 공식 val 최종 보고. 다른 사용자 작업 종료와 GitHub push는 하지 않는다.

## 서버·GPU 확인

- 서버 `curie`, 계정 `frostpine`. 프로젝트 실행 규칙(`experiments/README.md`)에서 빈 GPU에 실행, 완료 결과 재실행 금지, nohup/로그 보존을 확인했다.
- 접근 가능한 전역/상위/저장소 AGENTS.md와 서버 MOTD에서 추가 예약·할당 규칙은 발견하지 못했다. 공용 공지 계정 `/home/board`에는 읽기 권한이 없다. 별도 예약 규칙까지 확인한 것으로 표현하지 않는다.
- 실행 전 GPU1은 다른 계정의 학습이 점유했다. GPU0/3은 연산 프로세스가 없었다. GPU3을 선택하고 시작 직전에도 compute process 부재를 재검사했다.
- 고정 UUID: `GPU-bed51d32-f978-d07a-c020-01d7a36e63ea` (물리 GPU3, RTX A5000 24GB).
- 모든 GPU 자식 프로세스에 `CUDA_VISIBLE_DEVICES`를 해당 UUID로 전달한다. 프로세스 내 논리 장치는 `cuda:0`이며 다른 GPU는 보이지 않는다. 임의의 GPU 자동 선택·타인 프로세스 신호 전송은 없다.
- 실행기 PID 1662352, 보고서 감시기 PID 1662801. PID는 실행 당시 기록이다.

## 공통 초기 상태와 GPU 테스트

캐시된 timm 사전학습 가중치에서 CPU로 생성했다(`HF_HUB_OFFLINE=1`).

- 파일: `runs/role4_c1/shared_initial_s42.pt`
- 파일 SHA256: `a83561ececb61894b28716994f5db54b9103f594c6330875dd5fae80959da653`
- 모델 상태 SHA256: `c83851550e619511e0b179edf075690c11552c45113678fb368582a0e0c51a13`
- head 상태 SHA256: `07beb7c9b780646c6058cf0a4d6795ec5f6ae76b2c85dbf0e59a2a17aa7f0b4f`

`runs/role4_c1_gpu_smoke/`에서 두 Loss 각각 원본 train64/holdout16, 실제 C1 크기 단계·batch8·accum4, EMA·AMP·checkpoint 저장/재로딩·추론800/1333을 통과했다. 각각 성공 optimizer update3회, AMP skipped0회다. 별도로 batch8×1344×800에서 optimizer state+EMA를 포함한 forward/backward 2회도 통과했다.

| Loss | train/eval 최대 allocated GiB | 최대 reserved GiB | 최대 canvas stress allocated/reserved GiB |
|---|---|---|---|
| BCE | 10.6613 | 13.7930 | 10.5682 / 11.2363 |
| ASL | 10.6607 | 13.7988 | 10.5682 / 11.2363 |

PyTorch allocator 수치이며 nvidia-smi의 프로세스 전체 메모리와 다르다. 테스트 전후 공통 초기 파일의 SHA256과 모델 상태 hash가 동일함을 확인했다. 테스트 수치는 성능 비교 결과로 사용하지 않는다.

## 실행·완료 판정

`scripts/role4_c1_run.py`가 GPU smoke 통과 후 BCE→ASL 각18epoch을 순차 실행한다. 두 완료 summary의 epochs_run==18을 확인해야 holdout 평가·threshold 선택으로 진행한다. selection lock 생성 후 공식 val을 평가한다. 각 단계는 비정상 종료 시 다음 단계로 넘어가지 않는다. 초기 파일 hash와 동결된 소스 hash를 단계마다 검사한다. 중간 학습을 자동으로 처음부터 재실행하거나 기존 출력을 덮어쓰지 않는다.

- 진행 상태: `runs/role4_c1/pipeline_status.json`
- 전체 단계 로그: `runs/role4_c1/pipeline.log`
- 학습 로그: `runs/role4_c1/train_bce.log`, `train_asl.log`
- 학습 산출물: `runs/role4_c1/R4_C1_{bce,asl}_s42/`
- 최종 보고서: `report/role4_c1_final/report.md` (`scripts/report_role4_c1.py --watch`가 진행/실패/완료 상태를 반영).
- holdout/val의 summary·comparison·per_class·rare_classes CSV와 bootstrap JSON은 각각 `report/role4_c1_holdout_s42/`, `report/role4_c1_val_s42/`.

이 파일은 시작·검증 기록이다. 학습 완료나 특정 Loss의 우열을 의미하지 않는다. 최종 완료 여부는 pipeline_status의 `complete`와 최종 보고서에서 확인한다.
