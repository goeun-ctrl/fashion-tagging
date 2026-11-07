# 손실 실험 안내: ConvNeXt-T 고해상도 모델에서 가중치 최적화하기

작성: 5번 · 2026-11-06 · 대상: 손실 함수로 가중치를 더 좋게 만들려는 팀원

## 1. 출발점이 되는 모델

지금까지 학습한 분류기 중 holdout과 val 모두 1위인 모델이다. 자세한 내용은 `experiments/role5_regularization/results_hires_convnext.md`에 있다.

- **Release**: https://github.com/goeun-ctrl/fashion-tagging/releases/tag/r5-cnxt-hr-s42
- **모델**: `convnext_tiny.fb_in22k_ft_in1k_384` 27.84M. 마스크 GAP와 선형 헤드를 쓰고, EMA 가중치(ep13)다.
- **학습 설정**: `scripts/train_hires.py` 기본값
  - 짧은 변 400~800 무작위(비율 유지, 반전 0.5)
  - **ASL** (γ−=4, γ+=0, clip 0.05)
  - AdamW, lr 백본 1e-4 / 헤드 1e-3, wd 0.05, clip 1.0
  - warmup 1,000 update, 전체의 8/12·11/12 지점에서 lr ×0.1
  - EMA 0.9995, drop path 0.1, 18 epoch, seed 42
- **제출 설정**: 추론 800/1333, 좌우 반전 평균. 임계값 0.60이고, sweater·leg warmer·umbrella만 0.5다.

| 평가 | holdout micro / macro / image | val micro / macro / image |
|---|---|---|
| 임계값 0.5, 반전 없음 | 0.8435 / 0.7790 / 0.8466 | 0.8545 / 0.7277 / 0.8593 |
| 제출 설정 | 0.8572 / 0.7907 / 0.8578 | 0.8669 / 0.7442 / 0.8684 |

### Release 파일

| 파일 | 내용 |
|---|---|
| `R5_cnxt_hr_s42_best.pt` | 가중치. `{"model", "epoch", "config", "categories"}` 형식. SHA-256 `7a4efece…d3b71296` |
| `R5_cnxt_hr_s42_preds_{holdout,val}.npz` | 반전 없는 logits. 키는 `filenames`, `targets`, `logits`, `categories` |
| `R5_cnxt_hr_s42_preds_{holdout,val}_flip.npz` | 반전 평균 logits (제출 설정) |
| `R5_cnxt_hr_s42_config.json`, `_summary.json`, `_history.csv`, `_tta_summary.json` | 학습 설정, 결과, epoch별 기록, TTA 점검 결과 |

```bash
gh release download r5-cnxt-hr-s42 -R goeun-ctrl/fashion-tagging -D runs_hr/R5_cnxt_hr_s42
mv runs_hr/R5_cnxt_hr_s42/R5_cnxt_hr_s42_best.pt runs_hr/R5_cnxt_hr_s42/best.pt
```

## 2. 지금 가중치의 약점 (손실로 노릴 곳)

val F1 기준이다. 괄호 안은 양성 수다.

- **아직 약한 클래스**: vest 0.47 (21), sweater 0.56 (21), cardigan 0.50 (12), jumpsuit 0.48 (21), glove 0.57 (18), leg warmer 0.00 (7)
  - 공통점은 양성이 적거나, 다른 옷과 겹치거나, 모양이 비슷하다는 것이다.
  - vest·sweater·bag·headband는 FashionFormer 체크포인트가 더 높다 (vest 0.72, sweater 0.65).
- **작은 물체는 이미 좋다**: watch 재현율 0.84, belt 0.81, sock 0.71, scarf 0.69다.
- **확률 보정**: ASL 때문에 최적 임계값이 0.5보다 높다(0.60). 0.5에서는 대부분 클래스가 정답보다 많이 예측된다. 예를 들어 holdout에서 top은 정답 820개에 1,004개, coat는 243개에 320개를 예측했다.
- **과적합은 크지 않다**: holdout BCE가 ep17까지 계속 내려갔다. 대신 macro는 ep13 이후 0.772~0.779에서 정체했다.

## 3. 실험 방법 두 가지

### A. 이 가중치에서 이어서 학습 (빠름, 추천 시작점)

`--init_ckpt`로 `best.pt`를 불러와서, 낮은 lr로 몇 epoch만 새 손실로 학습한다.

- `--lr_backbone 0 --lr_head 0`으로 1 epoch 돌린 holdout 점수가 원래 모델과 똑같이 나오는 것까지 확인했다. 시작점이 정확히 이 모델이라는 뜻이다.

```bash
python scripts/train_hires.py --exp_id R5_cnxt_hr_ft_<손실이름>_s42 \
    --init_ckpt runs_hr/R5_cnxt_hr_s42/best.pt \
    --loss <손실이름> --epochs 4 --lr_backbone 1e-5 --lr_head 1e-4 \
    --warmup_iters 200 --milestones 0.75 \
    --train_dir data/train --val_dir data/val --out_dir runs_hr
```

- 시간: A5000 기준 epoch당 약 16~20분, 4 epoch에 약 1.2시간이다.
- 위 lr과 epoch은 원래 학습의 마지막 lr 구간(1e-5 / 1e-4)에 맞춘 **출발값이다. 아직 검증하지 않았다**.
  - 1 epoch 뒤 holdout macro가 원래 모델(0.779)보다 크게 떨어지면 lr을 더 낮춘다.
- 주의: 손실의 스케일이 ASL과 많이 다르면(예: sum으로 줄이는 손실), 같은 lr로 가중치가 크게 흔들린다. 아래 4장의 규칙을 지킨다.

### B. 처음부터 다시 학습 (느림, 최종 확인용)

`--init_ckpt` 없이 `--loss`만 바꾸고 나머지는 기본값 그대로 둔다.

- 실행당 약 5.4시간이다.
- A에서 좋았던 손실 1~2개만 B로 확인하는 것을 추천한다. 이어서 학습한 결과는 ASL로 이미 학습된 가중치의 영향을 받기 때문이다.
- 빨리 걸러야 하면 384 설정으로 먼저 돌린다(약 2.6시간): `--short_min 384 --short_max 384 --test_short 384 --max_long 640`
  - 이 설정의 ASL 기준값은 holdout macro 0.760이다(`R5_cnxt_hr_low_s42`).

## 4. 새 손실을 넣는 방법

1. `src/fashion_tagging/training/losses.py`에 클래스를 만든다.
2. `LOSSES`와 `build_loss`에 이름을 등록한다. 등록하면 `--loss <이름>`으로 바로 쓸 수 있다.
3. 지켜야 할 규칙
   - 입력은 **raw logits**다. 안에서 sigmoid를 걸어야 하고, 미리 걸면 안 된다.
   - 결과는 이미지 × 클래스 **평균**으로 줄인다. 그래야 기존 lr이 공정한 출발점이 된다.
   - 라벨 `-1`(외부 데이터의 모르는 클래스)은 `_masked_mean`으로 제외한다. 원본 데이터에는 `-1`이 없어서 결과에 영향은 없다.
   - 클래스별 가중치가 필요하면 `train_inner` 양성 수로 계산해서 상수로 넣는다. holdout이나 val로 계산하면 안 된다.

이미 있는 옵션

| 옵션 | 내용 |
|---|---|
| `--loss bce` | 기본 BCE |
| `--loss asl --asl_gamma_neg 4 --asl_gamma_pos 0 --asl_clip 0.05` | 지금 모델의 설정. γ−, γ+, clip을 바꿔 볼 수 있다 |
| `--distill_npz <FF 점수> --distill_alpha 0.5 --distill_shift 0.3` | 손실 = (1−α)·[선택한 손실] + α·BCE(FashionFormer 점수). 5번이 준비 중이다 |

후보 아이디어 (아직 아무도 돌려 보지 않음)
- **ASL 세기 조정**: γ− 2·6, clip 0·0.1. 지금은 음성 손실을 많이 줄여서 과하게 예측하는 쪽이다.
- **희소 클래스 가중**: 양성 가중을 클래스 빈도의 역수 제곱근 정도로 준다. vest·sweater·cardigan·jumpsuit을 노린다.
- **증류**: 5번의 FashionFormer 점수가 준비되면 위 손실들과 함께 쓸 수 있다. FF가 강한 vest·sweater·bag을 노린다.

## 5. 좋은 가중치를 고르는 규칙

- **best epoch은 holdout macro F1로만 고른다.** `train_hires.py`가 자동으로 `best.pt`에 저장한다.
- **임계값도 holdout으로만 고른다.**
  - 손실을 바꾸면 확률 분포가 달라져서 0.5가 최적이 아닐 수 있다.
  - `scripts/infer_hires.py --run_dir <폴더> --threshold <값>`을 쓰면 0.5, 지정값, 클래스별 예외를 한 번에 계산한다. 반전 TTA 결과도 같이 나온다.
- **val은 마지막에 한 번만 본다.** val을 보고 손실이나 임계값을 고르면 val 점수가 부풀려진다.
- **좋아졌는지 판단하기**
  - Release의 `preds_*.npz`와 같은 이미지 순서로 쌍 부트스트랩(2,000회)을 돌린다. 차이의 95% 구간이 0보다 큰지 본다.
  - seed 한 번의 실험이라 val macro ±0.02는 우연일 수 있다. 양성 7장 이하인 leg warmer·cape·tie·umbrella가 크게 흔들린다.

```python
import numpy as np
from fashion_tagging.evaluation.metrics import compute_metrics
a = np.load("R5_cnxt_hr_s42_preds_val.npz"); b = np.load("runs_hr/<새 실험>/preds_val.npz")
assert (a["filenames"] == b["filenames"]).all()
y = a["targets"]; pa = a["logits"] > np.log(0.6 / 0.4); pb = b["logits"] > np.log(t / (1 - t))  # t: holdout으로 고른 값
rng = np.random.default_rng(0); idx = rng.integers(0, len(y), (2000, len(y)))
d = [compute_metrics(y[i], pb[i])["macro_f1"] - compute_metrics(y[i], pa[i])["macro_f1"] for i in idx]
print(np.percentile(d, [2.5, 97.5]))
```

## 6. 서버에서 돌릴 때

- 연구실 서버 venv: `/data/eunjin/DL/venv_common506`
- 실행 전에 `nvidia-smi`로 빈 GPU를 확인한다. 5번이 FashionFormer 점수 계산과 480~960 학습에 GPU를 쓸 예정이다.
- 결과는 `--out_dir/<exp_id>/` 아래에 저장된다: `history.csv`, `summary.json`, `config.json`, `preds_*.npz`, `best.pt`, `last.pt`
- 같은 `exp_id`로 다시 돌리면 이미 끝난 실험을 덮어쓰지 않도록 멈춘다. 새 실험에는 새 이름을 쓴다.
