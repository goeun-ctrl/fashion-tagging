# 역할 2 · 이정은 ConvNeXt-Tiny 베이스라인

실험 ID: `S2_jeongeun_cnxt_t_bce_384_pad`. 추가 학습 없이 완료된 실험의 코드와 검증된 결과를 공유합니다.

## 먼저 읽을 자료

- [전체 27개 카테고리 결과: holdout·val 정답 수 / Precision / Recall / F1 / TP·FP·FN](artifacts/팀공유_보고서/ConvNeXt_Tiny_전체27개_클래스별결과.md)

- [한국어 팀 공유 보고서](artifacts/팀공유_보고서/ConvNeXt_Tiny_팀공유_실험보고서.md)
- [6쪽 PDF](artifacts/팀공유_보고서/ConvNeXt_Tiny_팀공유_실험보고서.pdf)
- [클래스별 약점·과적합 상세 분석](artifacts/클래스별_약점_과적합_분석.md)
- [팀 공통 결과 CSV](artifacts/local_snapshot/실험결과/results.csv)
- [실제 실행 설정](artifacts/local_snapshot/실험결과/resolved_config.json)

| 분할 | Micro F1 | Macro F1 | Image-wise F1 |
| --- | ---: | ---: | ---: |
| holdout | 0.805435 | 0.707369 | 0.809035 |
| 공식 val | 0.825538 | 0.660754 | 0.827740 |

holdout Macro F1으로 best epoch 10을 선택했으며 patience 5로 epoch 15에서 종료했습니다. 공식 val은 최종 평가에만 사용했습니다. 파라미터는 27,840,891개입니다. F1은 정확도가 아닙니다.

## 코드와 공정한 비교

`trainer.py`, `training_utils.py`, `config.json`은 실제 서버 백업본입니다. `config.json`의 기본 batch 32/accum 1과 달리 실제 실행은 CLI에서 **batch 16 × accumulation 2**를 적용했습니다. GAP → LayerNorm → Linear(27), RGB 384, bicubic 종횡비 유지·정규화 전 검정 padding, horizontal flip 0.5, BCE logits, AdamW, seed 42, threshold 0.5입니다.

원본 공통 코드 SHA는 `c5827dfa749e276d4e5431628a06730b26c4841f`입니다. 현재 main의 공통 코드가 이후 변경됐으므로 재현할 때는 저장소 루트의 `reference/fashion-tagging-c5827dfa/common`을 명시하세요. 이 폴더는 해당 원본 Git tree를 그대로 보관하며 최신 `common/`을 수정하지 않습니다. 클래스 순서와 고정 split도 원본에 포함됩니다.

아래는 저장소 루트에서 실행할 재현 명령 예시이며, 업로드 과정에서는 실행하지 않았습니다. 데이터·가상환경을 별도로 준비하고 사용 가능한 GPU를 배정한 뒤 실행합니다.

```bash
python experiments/role2_jeongeun_convnext/trainer.py \
  --config experiments/role2_jeongeun_convnext/config.json \
  --reference-dir reference/fashion-tagging-c5827dfa \
  --train-dir data/train --val-dir data/val \
  --batch-size 16 --accum 2 --device cuda:0 \
  --output-dir experiments/role2_jeongeun_convnext/runs/S2_jeongeun_cnxt_t_bce_384_pad
```

환경은 `prep/requirements.freeze.txt`, 실제 실행 인수·mean/std·라이브러리 버전은 `artifacts/local_snapshot/실험결과/resolved_config.json`에 있습니다. 원본 학습 로그는 `prep/continuation.log`, `gpu3.log`, `gpu0.log`에 나뉘어 있습니다. CSV/JSON에는 15 epoch 모두 있지만 GPU 이동 직전 epoch 8의 console 완료 문장은 누락됐습니다. 결정성 경고가 있어 비트 단위 재현을 보장하지 않습니다.

## 결과와 백업 범위

예측 NPZ, 27개 클래스별 지표·TP/FP/FN, 학습 곡선, 원본 epoch 기록, 분석 보고서, 가중치 SHA-256 및 CPU strict 로드 검증 기록을 포함했습니다. 보고서와 분석의 로컬 절대 경로·백업 위치는 작성 당시 기록입니다. 일부 오류 사례 원본 링크는 로컬 백업을 가리키며, 팀 보고서가 사용하는 대표 이미지 4장은 `artifacts/팀공유_보고서/assets`에 함께 있습니다. 원본 데이터 이미지 전체는 업로드하지 않았습니다.

대형 `best.pt`와 `last.pt`는 Git에 넣지 않았습니다. 노트북의 검증된 백업 `ConvNeXt_약점분석_가중치백업_20261103_184413/weights`와 서버 원본에 보존돼 있습니다. `artifacts/가중치_SHA256_검증.json`의 해시를 확인한 뒤 `artifacts/weights/best.pt`에 별도로 복사하면 `artifacts/load_model_cpu.py`로 CPU 로드할 수 있습니다. 가중치는 checkpoint의 `model` 키이며 `strict=True`로 확인했습니다. 이는 구조·로드 검증이며 재추론 검증이 아닙니다.

최대 allocated VRAM은 3.652 GB(십진) = 3.402 GiB(이진), 학습 구간 누적 약 58.97분입니다. 공유 GPU 측정이라 다른 실험과 비용 비교 시 조건 확인이 필요합니다. 보고서의 개선안은 제안이며 실행하지 않았습니다.
