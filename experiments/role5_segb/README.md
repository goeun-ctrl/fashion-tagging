# role5 재설계 B 실험 기록 (2026-11-22 ~ 11-25)

정리 문서: [`docs/plan/재설계B_결과정리.md`](../../docs/plan/재설계B_결과정리.md). 가중치: GitHub Release `segb-b24-s42`.

| 폴더 / 파일 | 내용 |
|---|---|
| `logs/b24_vs_cnn.log`, `logs/b24_vs_base.log` | 24 epoch 4개: 최종 CNN 대비 / B24_base 대비 (`scripts/diag_segb.py --loc`, 서버 CUDA) |
| `logs/s6_batch{1..4}_*_vs_{cnn,S6base}.log` | 6 epoch 16개 진단 (Mac MPS 위치). batch1은 옛 판정 규칙(`_OLDRULE`) |
| `logs/analysis_layered*.log` | `scripts/diag_layered.py`: 겹쳐 입은 상의 원인 분석 (S6_base, rc01/03/1) |
| `logs/analysis_vote_*.log` | `scripts/posthoc_vote.py`: A안 사후 투표 |
| `logs/analysis_oracle.log` | `scripts/oracle_mask.py`: 완벽 마스크 상한 |
| `logs/analysis_logitpool.log` | `scripts/logit_pool.py`: 로짓 풀링 |
| `queues/` | 서버 큐 스크립트와 시작·종료 기록 (시각은 UTC) |
| `run_records/<run>/` | 런별 `config.json`, `summary.json`, `history.csv` |

경로 표기: `$W` = gpu-server-b 작업 폴더, `<repo>` = 로컬 저장소.
