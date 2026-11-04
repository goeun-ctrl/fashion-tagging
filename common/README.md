# common/ (호환용)

코드는 `src/fashion_tagging/`(라이브러리)와 `scripts/`(실행)로 옮겼다. 이 폴더에는 예전 경로를 쓰는 코드가
깨지지 않도록 연결 파일만 남아 있다.

| 예전 | 새 위치 |
|---|---|
| `common/labels.py` | `src/fashion_tagging/data/labels.py` |
| `common/dataset.py` | `src/fashion_tagging/data/dataset.py` |
| `common/preprocess.py` | `src/fashion_tagging/data/preprocess.py` |
| `common/metrics.py` | `src/fashion_tagging/evaluation/metrics.py` |
| `common/train_eval.py` | `scripts/train.py` |

`sys.path`에 `common/`을 넣고 `from labels import ...` 하던 코드, `python common/train_eval.py ...` 명령은 그대로 동작한다.
새 코드는 `from fashion_tagging.data.labels import ...`를 쓴다(`src/`를 `sys.path`에 넣거나 `pip install -e .`).
