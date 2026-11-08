# 재현 명령

저장소 루트에서 실행한다. 아래는 재현용 명령이며 이번 결과 게시 때 재학습·재평가하지 않았다. 기존 완료 출력이 있는 경로에 실행하지 않는다. 새 저장소 사본 또는 별도 out_dir를 사용한다. 데이터·교사·체크포인트는 Git에 없으므로 권한 있는 원본을 별도로 준비한다.

## 환경과 입력

실측 torch 2.8.0+cu126 / timm 1.0.22 환경은 [실제 설정](results/config.json)과 [preflight](checks/preflight.json)를 따른다. 저장소 공통 requirements의 최신 버전 문구와 구분한다. torchvision·numpy·Pillow·matplotlib 등 환경 버전은 preflight 기록을 확인한다. `data/splits/`의 고정 분할을 사용한다. [소스 manifest](source_manifest.json)로 팀 원본 코드 hash를 확인할 수 있다.

## 실제 학습과 평가 순서

아래 경로 변수는 실행 환경에 맞게 지정한다. 학습과 두 평가에 사용될 단일 GPU는 사용자가 가용성을 확인해 선택한다. 로컬 경로·GPU UUID는 게시본에서 치환했으며 실행 인자는 [launch](results/logs/launch.json) 및 [pipeline status](results/logs/pipeline_status.json)에 있다.

```bash
PY=/path/to/prepared/venv/bin/python
TRAIN=/path/to/train
VAL=/path/to/val
TEACHER=/path/to/ff_r50_train.npz
OUT=/path/to/new-empty-output
export CUDA_VISIBLE_DEVICES=YOUR_AVAILABLE_GPU

"$PY" experiments/role4_bce_kd/train_controlled.py   --exp_id R4_bce_kd_s42 --head linear_ms --pool gapgmp --loss bce   --train_dir "$TRAIN" --val_dir "$VAL" --splits_dir data/splits   --out_dir "$OUT" --distill_alpha 0.5 --distill_shift 0.3   --distill_npz "$TEACHER"

"$PY" experiments/role4_bce_kd/evaluate_controlled.py   --run "$OUT/R4_bce_kd_s42" --split holdout
"$PY" experiments/role4_bce_kd/evaluate_controlled.py   --run "$OUT/R4_bce_kd_s42" --split val
```

既定値は [planned_config.json](planned_config.json) と [学習ソース](source/scripts/train_hires.py)に対応する。`train_controlled.py` は学習末尾の公式val評価だけを除き、holdout選択の固定後に `evaluate_controlled.py` でvalを評価する。評価は既存予測の上書きを拒否する。

## 元の自動処理と報告書

`pipeline.py` は実測時の絶対パス・GPU UUID・検証hash・出力先を固定した監査用スナップショットであり、そのまま他環境で起動する汎用コマンドではない。実測時は GPU stress → GPU smoke → train → holdout TTA → val TTA → report の順に実行した。コード・教​​師hashを検査し、失敗時は後続段階を止める。

`report_results.py` は `runs/role4_bce_kd/R4_bce_kd_s42/` の summary/selection/history/metrics/CSV から README と曲線を生成する。別の OUT から再現した場合は、新しい作業コピーの同じ標準パスに必要な結果を配置して次を実行する。既存結果がある元の作業場所では実行しない。

```bash
"$PY" experiments/role4_bce_kd/report_results.py
```

公開ファイルの数値・相対リンク・ソースhash・曲線の再生成一致は `python3 experiments/role4_bce_kd/validate_results.py` で検査できる（matplotlibが必要）。
