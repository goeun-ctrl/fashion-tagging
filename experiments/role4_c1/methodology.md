# BCE·ASL 비교 재현 명령

이 문서는 이번 비교의 실행 명령만 담는다. 조건·결과·해석은 [최종 README](README.md)를 참조한다. 저장소 루트에서 실행하고 `settings.json`의 데이터 경로를 머신에 맞게 수정한다. 양쪽 학습에 동일한 환경과 설정을 사용한다. 가용 GPU 한 장을 `CUDA_VISIBLE_DEVICES`로 지정한다. 이미 완료된 실행 폴더에 재실행하지 않는다.

```bash
PY=python

# CPU에서 사전학습 초기 상태를 한 번 생성 (캐시에 없으면 timm/HF 다운로드 필요)
CUDA_VISIBLE_DEVICES='' $PY scripts/role4_c1.py init

# 동일 초기 파일에서 각각 처음부터 학습. 기본 device=cuda, 현재 보이는 GPU 1장 사용
$PY scripts/role4_c1.py train --loss bce
$PY scripts/role4_c1.py train --loss asl

# holdout: 원본 및 좌우 반전 logits를 모두 저장, 추론 800/1333
for LOSS in bce asl; do
  $PY scripts/role4_c1_evaluate.py \
    --run runs/role4_c1/R4_C1_${LOSS}_s42 --split holdout \
    --out runs/role4_c1/eval_holdout_${LOSS}
done

# holdout만으로 Loss 및 각 Loss의 threshold 고정
$PY scripts/role4_c1_compare.py \
  --bce_run runs/role4_c1/R4_C1_bce_s42 --asl_run runs/role4_c1/R4_C1_asl_s42 \
  --bce_preds runs/role4_c1/eval_holdout_bce --asl_preds runs/role4_c1/eval_holdout_asl \
  --out report/role4_c1_holdout_s42

# 고정 후 공식 val 평가. selection lock 없이는 거부
for LOSS in bce asl; do
  $PY scripts/role4_c1_evaluate.py \
    --run runs/role4_c1/R4_C1_${LOSS}_s42 --split val \
    --lock report/role4_c1_holdout_s42/selection.json \
    --out runs/role4_c1/eval_val_${LOSS}
done
$PY scripts/role4_c1_compare.py \
  --bce_run runs/role4_c1/R4_C1_bce_s42 --asl_run runs/role4_c1/R4_C1_asl_s42 \
  --bce_preds runs/role4_c1/eval_val_bce --asl_preds runs/role4_c1/eval_val_asl \
  --lock report/role4_c1_holdout_s42/selection.json --out report/role4_c1_val_s42
```
