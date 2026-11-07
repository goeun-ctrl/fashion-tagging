# EVA02-Small V2 집계 결과

외부 3,458장 V2, partial Masked BCE, Seed 42, 378px, 16×2. 실제 종료 Epoch 19, 최종 평가 Best Epoch 14.

전체 해석은 [분석 보고서](../EVA02_V2_RESULTS_ANALYSIS.md)에 있다. 이 폴더는 기존 로컬 원본 결과를 복사하거나 검증 후 집계한 자료이며 원본 Baseline 결과를 대체하지 않는다.

| 파일 | 내용 / 실험 |
|---|---|
| `metrics_holdout.json`, `metrics_val.json` | V2 Best 14의 전체 F1·BCE·기존 실행시간. 원본 복사 |
| `per_class_holdout.csv`, `per_class_val.csv` | V2 27개 클래스 Support/TP/FP/FN/Precision/Recall/F1. 원본 복사 |
| `baseline_per_class_holdout.csv`, `baseline_per_class_val.csv` | 원본 Baseline의 저장 예측으로 검증한 클래스 지표. 기존 `../class_metrics_*.csv`와 수치 일치 |
| `baseline_vs_v2_per_class_holdout.csv`, `baseline_vs_v2_per_class_val.csv` | 같은 분할·클래스를 연결한 Baseline/V2 지표와 차이. 기존 비교 CSV를 검증 후 복사 |
| `training_history.csv` | V2 Epoch 1–19의 Loss·F1·LR·Best·종료·진단 경고. 원본 복사 |
| `actual_class_exposure.csv` | V2 Epoch×클래스의 전체 source 합산 draw/positive/known/unknown. 원본 복사 |
| `experiment_config_summary.json` | resolved config 기반 설정, 버전·샘플 수·추가 정규화 비활성. 로컬 경로·호스트 정보·데이터 해시 목록 제외 |
| `analysis_summary.json` | 지표 대조, Micro/Macro P/R, TP/FP/FN, 27개 클래스 변화, Epoch별/클래스별 외부 노출 집계 |
| `code_provenance.json` | 실제 실행 코드의 출처, 기존 브랜치 비교, main 통합의 경로 차이와 코드 해시 |
| `validation_summary.json` | CPU 검사·사전 검사·집계 검증 및 공개 범위 요약 |
| `training_curves.png` | 실제 이력의 Loss, Macro F1, 진단 격차, Early Stopping 곡선 |
| `class_f1_delta.png` | 27개 클래스의 V2−Baseline F1 (%p), Val과 Holdout |

F1/Precision/Recall은 0–1 범위다. Threshold는 sigmoid 확률 **>= 0.5**, Macro는 전체 27개 클래스 평균이다. Image-wise F1은 정답과 예측이 모두 빈 이미지에서 1이다. Support는 분할의 정답 양성 수다. `delta_*`는 **V2 − Baseline**, `delta_f1_pp`는 100×delta_f1이다. CSV의 클래스 순서는 원본과 같고 쉼표가 포함된 클래스명은 하나의 필드다.

진단 Train은 전체 Train이 아니라 고정 2,048장 부분집합이다. 외부 노출은 실제 실행 Plan×source×labels를 재집계해 전체 노출 로그 513행과 일치시킨 뒤 분리했다. Best 14 성능에는 ≤14 노출을 사용해야 하며 ≤19 노출과 혼동하지 않는다. 노출은 AMP skip draw를 포함한다. 외부 데이터의 known label 수 비중을 gradient 기여 비중으로 해석하지 않는다.

506장 기본/정규화·EMA 비교는 [role5 보고서](../../role5_regularization/results_eva02.md)의 소수 4자리 보고값이다. 해당 실험의 원본 예측은 이 폴더에 없으며 재계산한 결과로 표시하지 않는다.

체크포인트, 모델 가중치, 이미지, 예측 NPZ, 원본 라벨, 데이터 Manifest, ZIP, 이미지 ID/해시 목록, 진단 부분집합 ID, 로컬 환경 경로는 포함하지 않는다. 기존 원본은 로컬에 보존한다. 이번 정리는 학습/GPU 재실행 없이 수행했다.
