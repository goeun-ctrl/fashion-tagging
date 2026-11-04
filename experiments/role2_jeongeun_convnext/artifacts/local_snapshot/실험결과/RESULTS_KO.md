# ConvNeXt-Tiny 기준 실험 결과

- 실험: S2_jeongeun_cnxt_t_bce_384_pad
- 원본 SHA: c5827dfa749e276d4e5431628a06730b26c4841f
- 완료 epoch: 15; best epoch: 10
- 파라미터: 27,840,891
- GPU 0에서 9 epoch 체크포인트로부터 10 epoch 재개. 이전 GPU 0/3 기록 보존.
- batch 16 × accumulation 2 = effective batch 32.
- checkpoint 선택·early stopping은 내부 holdout Macro F1만 사용. 공식 val은 최종 평가에만 사용.
- smoke는 별도 폴더이며 기준 성능에 포함하지 않음.

| 분할 | Micro F1 | Macro F1 | Image-wise F1 |
|---|---:|---:|---:|
| holdout | 0.805435 | 0.707369 | 0.809035 |
| val | 0.825538 | 0.660754 | 0.827740 |

- 누적 학습 시간: 3538.2 초
- 학습 최대 allocated VRAM: 3.402 GiB
- 학습 최대 reserved VRAM: 3.479 GiB
- GPU 이동 및 데이터 다운로드 대기 시간은 순수 학습 시간과 다름.
- 실행 설정: resolved_config.json; 라이브러리 버전: library_versions.json 및 ../../prep/requirements.freeze.txt.
- 실제 재개 명령은 resolved_config.json의 argv 및 ../../prep/migrate_gpu0.sh에 기록.
- 결정성 경고: 실행 중 CUBLAS_WORKSPACE_CONFIG 미설정 경고가 있었으므로 비트 단위 재현성을 보장하지 않음.
- Git add/commit/push 및 다른 계정 프로세스 종료는 수행하지 않음.
- best.pt/last.pt는 optimizer·scheduler·scaler·RNG를 포함한 체크포인트이며 모델 파라미터는 model 키에 저장.
- 클래스별 지표·오류 사례·팀 CSV는 이 결과 폴더의 해당 CSV 및 error_examples 하위 폴더 참조.
