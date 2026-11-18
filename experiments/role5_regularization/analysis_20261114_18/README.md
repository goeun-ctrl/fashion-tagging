# 기법 효과 재검증과 백본 비교 (2026-11-14 ~ 11-18, 5번)

판정과 해석은 `docs/plan/현재상황_재정리.md`(3장)와 `docs/plan/기법효과_지도.md`에 있다. 이 폴더에는 그 근거가 되는 스크립트와 원본 출력이 있다.

## 평가 방법
- holdout(2,425장)과 val(1,158장)만 사용한다. **test_public과 examples는 쓰지 않았다.**
- 예측: 각 학습의 `best.pt`를 짧은 변 800, 긴 변 1333에 flip TTA로 추론한다(`scripts/infer_hires.py`).
- 지표:
  - 임계값과 무관한 micro AP / macro AP
  - Fashionpedia 면적 구간별 양성 순위(<0.5%, ≥8%)
  - 비슷한 옷이 있는 음성 대비 순위
- 이미지 단위 짝 bootstrap 1,000회로 95% 구간을 구한다. `*`는 구간이 0을 포함하지 않는다는 뜻이다. 판정은 holdout과 val **둘 다** `*`일 때만 한다.
- seed 잡음: micro AP 0.002 이내 (R4 s43/s44, DINOv3 s42/s43).

## 파일
| 파일 | 내용 |
|---|---|
| `gen_eval.py` | 짝 비교 범용 스크립트. `python gen_eval.py A:B C:D ...` (서버 `/data/eunjin/DL/tmp/` 기준 경로) |
| `r5_eval.py` | 11-15 비교 (외부 3,458, 쿼리 헤드, DINOv3 lr 쌍, R4 seed) |
| `why_*.py`, `ap_cmp.py` | 기법별 효과 분석 (면적별, 증류, 전체 짝) |
| `prep_ext.py` | 외부 3,458장 학습 목록과 라벨 만들기 |
| `queue_*.sh` | GPU가 비면 다음 학습을 시작하는 대기 스크립트 |
| `eval_outputs/` | 위 스크립트의 원본 출력 |
| `run_records/<실험>/` | 각 학습의 `config.json`, `summary.json`, `tta/summary.json` |

## 결과 요약 (micro AP, holdout / val)
| 실험 | 결과 |
|---|---|
| R4 (ConvNeXt-T, 지도 사전학습) | 0.943 / 0.949 |
| CNN + DINOv3 사전학습, lr 3e-5 (1e-5와 같음) | 0.952 / 0.955 |
| DeiT-3 (ViT, 지도 사전학습), lr 1e-4 | 0.946 / 0.949 |
| **DINOv3 ViT-S+, lr 1e-4** (seed 2개) | **0.958 / 0.960** |
| DINOv3 ViT: 증류 없음 | −0.008 / −0.004 ★ |
| DINOv3 ViT: Swin-B 교사 α 0.5 / α 0.7 | 차이 없음 (val만 ★) / −0.002 ★ |
| DINOv3 ViT: 외부 3,458 (seed 2개) | 차이 없음 |
| DINOv3 ViT: 헤드 중간 블록 제거, EMA 제거 | 둘 다 차이 없음 |

아직 결과가 없는 실험: ViT 해상도 1024, CNN-DINOv3 24 epoch
