# V2 과적합 진단과 로그

V2 패키지의 학습 조건은 그대로 유지한다. 선택 기준은 Holdout Macro F1의 엄격한 증가이며, 동점은 이전 Best를 유지한다. 개선이 없는 5 epoch 또는 최대 30 epoch에 종료한다. 경고와 Train 진단 점수는 종료·Best 선택에 사용하지 않는다.

`training_history.csv`는 기존 Train Masked BCE, Holdout BCE와 세 F1, Backbone/Head LR에 다음 열을 추가한다.

| 열 | 의미 |
|---|---|
| `diagnostic_train_macro`, `diagnostic_train_bce` | 고정 Train_inner 부분집합의 무증강, 평가 모드, threshold 0.5 진단 |
| `train_holdout_macro_gap` | 진단 Train Macro F1 − Holdout Macro F1 |
| `best_epoch`, `best_holdout_macro`, `is_best` | 현재까지의 Best 선택 상태 |
| `epochs_without_improvement` | 마지막 Best 이후 epoch 수 |
| `early_stopping_triggered`, `training_complete`, `stop_reason` | Patience 또는 최대 epoch 종료 상태 |
| `overfitting_warnings` | 3회 연속 변화에서 관측된 진단 경고, `|`로 구분 |

진단 부분집합은 Train_inner에서 Seed 42의 독립 PCG64 생성기로 중복 없이 2,048장을 선택한다. 고정 분할 순서를 기준으로 선택하며 Holdout/Validation과의 교집합을 차단한다. `diagnostic_subset.json`에 ID, ID 해시, 클래스별 양성 수를 기록하고 resume 시 대조한다. 작은 합성 테스트에서는 이용 가능한 전체 Train_inner를 사용한다. 실제 부분집합의 점수는 전체 Train 점수가 아니며, 희소 클래스 support에 따라 변동할 수 있다.

진단은 각 epoch의 Holdout 측정 다음에 수행한다. 기존 결정적 평가 전처리와 `no_grad` 평가 함수를 재사용한다. 전용 DataLoader generator를 사용하고 Python/NumPy/PyTorch CPU·CUDA 난수 상태를 복구하므로 공통 Epoch Plan과 학습 난수 흐름을 변경하지 않는다. 공식 Validation은 학습 종료가 확인된 후 `best.pt` 최종 평가에만 사용한다.

최근 4개 epoch, 즉 3회 연속 변화에서 다음을 콘솔과 CSV에 경고한다.

- 진단 Train Macro F1 상승과 Holdout Macro F1 하락이 동시에 이어짐.
- Train − Holdout Macro F1 격차가 계속 증가함.
- Holdout BCE가 계속 증가함.

절대 격차 임계값에 따른 자동 중단은 없다. Train 배치의 Masked BCE와 Holdout의 완전 라벨 BCE는 라벨 구성·집계 방식이 다르므로 둘의 차이만으로 과적합을 확정하지 않는다.

매 epoch `last.pt`를 원자적으로 저장하고, Best가 갱신되면 동일 상태를 `best.pt`에 보존한다. 최종 평가는 종료 여부와 Best epoch/score를 검증한 후 `best.pt`를 읽는다. 진단 설정은 `resolved_config.json`에 기록한다. 기존 모니터링 필드가 없는 재학습 체크포인트는 같은 설정으로 간주하지 않으며, V2는 새 run ID로 시작한다. Baseline과 원본 데이터·패키지는 변경하지 않는다.

검증은 `test_retrain.py`와 `test_monitoring.py`의 합성 CPU 테스트로 수행한다. 실제 EVA 생성과 CUDA 연산을 막는 별도 검수 실행기도 로컬 검수 폴더에 보관한다. Patience 5로 epoch 7에 종료하고 epoch 2의 Best를 최종 선택하는 흐름, 중단·복구, 로그 저장, 경고, 난수 보존, -1 마스킹과 누적 구간 정규화를 검사한다.
