# FashionFormer R50 (Fashionpedia 체크포인트) → 이미지 단위 F1, 공식 val

측정: 2026-11-06 · 5번 · 연구실 서버 RTX A5000 1장 · **학습 없이 추론만**

## 결론

- 공개 체크포인트 `fashionformer_r50_3x.pth`를 그대로 썼다. 검출 점수를 클래스별 최대값으로 모아 이미지 라벨로 만들었다.
- 임계값 0.3에서 **val micro 0.858 / macro 0.705 / image 0.861**이다. 지금까지의 최고 분류기보다 세 지표 모두 높고, 쌍 부트스트랩 95% 구간이 모두 0보다 크다.
- 개선은 **작은 물체**에서 크다. watch 0.42 → 0.74, sock 0.46 → 0.68, belt 0.61 → 0.75, scarf 0.52 → 0.68, glasses 0.85 → 0.95 (EVA+EMA 대비). 분류기의 약점(해상도)이라는 진단과 맞는다.
- 파라미터 **44,072,298개** (속성 가지 포함, 50M 이하). 추론 속도 0.061초/장 (원본 해상도 1333×800 리사이즈, 후처리 제외).

## 결과 (val 1,158장)

| 모델 | 파라미터 | micro | macro | image |
|---|---|---|---|---|
| ConvNeXt-T 384 (2번 기준선) | 27.8M | 0.8255 | 0.6608 | 0.8277 |
| EVA02-S 378 + wd 제외 + drop path + EMA (5번) | 21.8M | 0.8397 | 0.6589 | 0.8414 |
| **FashionFormer R50, 임계값 0.3** | **44.1M** | **0.8575** | **0.7048** | **0.8608** |
| FashionFormer R50, 임계값 0.5 | 44.1M | 0.8208 | 0.6224 | 0.7979 |

FashionFormer @0.3에서 다른 모델을 뺀 차이 (쌍 부트스트랩 2,000회, 95% 구간):

| 대비 | micro | macro | image |
|---|---|---|---|
| ConvNeXt-T | +0.032 [+0.022, +0.043] | +0.044 [+0.012, +0.082] | +0.033 [+0.022, +0.045] |
| EVA02-S + EMA | +0.018 [+0.007, +0.029] | +0.046 [+0.002, +0.090] | +0.020 [+0.008, +0.032] |

임계값별 결과 (val):

| 임계값 | micro | macro | image | 이미지당 예측 수 |
|---|---|---|---|---|
| 0.1 | 0.600 | 0.429 | 0.636 | 7.39 |
| 0.2 | 0.792 | 0.624 | 0.809 | 4.43 |
| **0.3** | **0.858** | **0.705** | **0.861** | 3.46 |
| 0.4 | 0.860 | 0.682 | 0.853 | 2.91 |
| 0.5 | 0.821 | 0.622 | 0.798 | 2.49 |

정답의 이미지당 평균 라벨 수는 3.25개다.

## 임계값 0.3을 쓴 근거와 한계

- 0.3은 val로 고른 값이 아니다. FashionFormer 저장소의 기본 표시 임계값(`demo/image_demo.py --score-thr 0.3`)이고, 설정의 `instance_score_thr=0.3`과도 같다. 그래도 위 표처럼 val을 본 상태이므로, 최종 선택은 다른 근거로 확정해야 한다.
- **holdout으로 임계값을 고를 수 없다.** holdout 2,425장은 Fashionpedia `train2020`에 들어 있어서, 이 체크포인트의 학습 데이터다. holdout 점수는 부풀려진다.
- 그래서 임계값을 깨끗하게 정할 데이터는 val뿐이다. val로 정하면 val 점수가 낙관적이 된다. 임계값 0.3과 0.4가 비슷하니, 선택에 따른 차이는 작다.

## 클래스별 F1 (val)

| 클래스 | 양성 | FF@0.3 | ConvNeXt | EVA+EMA |
|---|---|---|---|---|
| shirt, blouse | 102 | 0.728 | 0.685 | 0.670 |
| top, t-shirt, sweatshirt | 463 | 0.851 | 0.789 | 0.828 |
| sweater | 21 | 0.652 | 0.550 | 0.595 |
| cardigan | 12 | 0.467 | 0.095 | 0.300 |
| jacket | 179 | 0.792 | 0.782 | 0.809 |
| vest | 21 | 0.722 | 0.412 | 0.564 |
| pants | 313 | 0.896 | 0.886 | 0.913 |
| shorts | 106 | 0.809 | 0.821 | 0.822 |
| skirt | 162 | 0.713 | 0.695 | 0.762 |
| coat | 103 | 0.789 | 0.781 | 0.784 |
| dress | 506 | 0.902 | 0.891 | 0.906 |
| jumpsuit | 21 | 0.340 | 0.400 | 0.444 |
| cape | 5 | 0.000 | 0.571 | 0.333 |
| glasses | 130 | 0.950 | 0.866 | 0.848 |
| hat | 74 | 0.863 | 0.865 | 0.878 |
| headband, head covering, hair accessory | 105 | 0.829 | 0.657 | 0.703 |
| tie | 3 | 0.667 | 0.857 | 0.400 |
| glove | 18 | 0.537 | 0.500 | 0.629 |
| watch | 82 | 0.744 | 0.604 | 0.415 |
| belt | 159 | 0.746 | 0.662 | 0.610 |
| leg warmer | 7 | 0.222 | 0.000 | 0.000 |
| tights, stockings | 62 | 0.818 | 0.752 | 0.872 |
| sock | 51 | 0.680 | 0.602 | 0.456 |
| shoe | 804 | 0.983 | 0.978 | 0.981 |
| bag, wallet | 205 | 0.901 | 0.831 | 0.856 |
| scarf | 48 | 0.681 | 0.557 | 0.521 |
| umbrella | 5 | 0.750 | 0.750 | 0.889 |

FashionFormer가 약한 클래스는 cape(5장 중 0), jumpsuit 0.34다. 분류기와 섞으면 보완할 여지가 있다. 다만 파라미터 여유가 약 6M뿐이다.

## 제출에 쓰려면 남은 일

1. **환경**: Python 3.8, torch 1.9.0+cu111, mmcv-full 1.3.18, mmdet 2.18.0, panopticapi가 필요하다. 그대로 `requirements.txt`에 넣을 수 있는지, 아니면 PyTorch 2.x로 이식할지 정해야 한다.
2. **파라미터 줄이기**: 속성(attribute) 가지와 19개 부속 클래스 출력을 빼면 줄어들 수 있다. 얼마나 줄어드는지 측정해야 한다.
3. **임계값 결정 방법**: holdout을 쓸 수 없으니, val의 일부를 떼어 쓸지 0.3으로 고정할지 팀이 합의해야 한다.
4. **필요하면 fine-tuning**: 27개 클래스만 남기거나, 이미지 단위 손실을 추가하는 방식이다. 학습 데이터가 Fashionpedia train과 같아서 효과는 불확실하다.

## 재현

```bash
# 서버: /data/eunjin/DL/fashionformer (FashionFormer repo @0752737, venv = 위 환경)
cd FashionFormer
PYTHONPATH=. ../venv/bin/python ../ff_infer.py configs/fashionformer/fashionpedia/fashionformer_r50_mlvl_feat_3x.py \
    ../fashionformer_r50_3x.pth /data/eunjin/DL/fassion_tagging/data/val ../ff_r50_val.npz
python ff_eval.py ff_r50_val.npz <ConvNeXt preds_val.npz> <EVA preds_val.npz>
```

체크포인트: FashionFormer README의 OneDrive 링크, `fashionformer_r50_3x.pth` (527,858,987 bytes, SHA-256 `268e806c…91b765`).
