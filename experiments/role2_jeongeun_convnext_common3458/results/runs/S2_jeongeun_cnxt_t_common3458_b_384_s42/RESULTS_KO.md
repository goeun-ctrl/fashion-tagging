# ConvNeXt V2 공통 데이터 B안 재학습 결과

실험: S2_jeongeun_cnxt_t_common3458_b_384_s42
Best epoch 14, 종료 epoch 19. GPU 0, batch 16 × accumulation 2.

| 분할 | 실험 | Micro F1 | Macro F1 | Image-wise F1 |
|---|---|---:|---:|---:|
| holdout | 원본-only 공통증강 대조 | 0.804838 | 0.701820 | 0.805638 |
| holdout | 외부506+공통증강+Masked BCE | 0.797168 | 0.689975 | 0.798758 |
| holdout | 외부3458+공통증강+Masked BCE | 0.809277 | 0.707107 | 0.809509 |
| val | 원본-only 공통증강 대조 | 0.818619 | 0.652603 | 0.822416 |
| val | 외부506+공통증강+Masked BCE | 0.814441 | 0.653716 | 0.813430 |
| val | 외부3458+공통증강+Masked BCE | 0.825295 | 0.643132 | 0.826315 |

학습 시간: 67.65분. 최대 할당 VRAM: 3.577 GB / 3.331 GiB. 공유 GPU에서 측정.

Holdout Macro F1으로 best를 선택했고 공식 val은 최종 평가에만 사용했다. 동일 공통 규칙의 원본-only 및 506장 실험과 비교한다. V2는 풀 크기에 따른 가중치와 노출 분포도 변하므로 개별 요소의 인과 효과를 단정하지 않는다. 외부 라벨은 사람 전수 검수 자료가 아니다.

3개 실험의 동일 평가ID/정답/클래스 및 NPZ 재계산 결과: prediction_verification.json. 전체 지표 비교: comparison_three_runs.csv. 27개 클래스 비교: comparison_27_classes.csv. 클래스별 지표: per_class_holdout.csv, per_class_val.csv. 예측: preds_holdout.npz, preds_val.npz. 가중치: best.pt, last.pt의 model 키. 실제 노출량: actual_epoch_exposure/. 로그: history.csv, learning_curves.png. Git push 미실행.
