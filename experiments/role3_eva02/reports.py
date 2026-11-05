"""Korean aggregate reports produced only after an actual completed evaluation."""
from data_check import HERE
from analysis import METRICS, REF


def score_table(results):
    return "\n".join(["| 분할 | Micro F1 | Macro F1 | Image-wise F1 |", "|---|---:|---:|---:|",
                      *["| " + name + " | " + " | ".join(f"{results[key][k]:.6f}" for k in METRICS) + " |"
                        for key, name in (("holdout", "Holdout"), ("val", "공식 Validation"))]])


def render_reports(history, results, summary, classes):
    chosen = max(history, key=lambda r: r["holdout_macro"])
    low = min(history, key=lambda r: r["holdout_bce"])
    last = history[-1]
    best_ep, end = chosen["epoch"], last["epoch"]
    stop = "5 epoch 연속 미개선" if end - best_ep >= 5 else "최대 epoch 예산"
    minutes = sum(r["train_seconds"] for r in history) / 60
    mem = max(r["allocated_bytes"] for r in history) / 1e9
    reserved = max(r["reserved_bytes"] for r in history) / 1e9
    val, hold = results["val"], results["holdout"]
    comparable = sorted((r for r in classes if r["delta_f1"] is not None), key=lambda r: r["delta_f1"])
    worst = sorted(classes, key=lambda r: r["EVA_f1"])[:5]
    leg = next(r for r in classes if r["class"] == "leg warmer")
    gaps = " / ".join(f"{val[k]-hold[k]:+.6f}" for k in METRICS)
    diff = " / ".join(f"{r['delta']:+.6f}" for r in summary if r["split"] == "val")
    poorest = ", ".join(f"{r['class']} {r['EVA_f1']:.3f}" for r in worst)
    pair = val["most_common_simultaneous_fn_fp"]
    one = f"""# EVA02-Small 실험 보고서

## 1. 실험 요약

고정 분할로 학습한 EVA02-Small의 공식 Validation Macro F1은 **{val['macro_f1']:.6f}**다. Holdout에서 선택한 epoch {best_ep} 가중치를 공식 Validation에 한 번 적용했다. 모델의 세부 지표는 아래와 같다.

{score_table(results)}

## 2. 프로젝트와 역할

한 사진에 보이는 모든 사람의 의류·액세서리를 27개 이름의 목록으로 예측한다. 3번 역할은 사전학습 특징 추출기(Backbone)를 EVA02로 바꾸고 분류기(Head), 라벨, 평가 기준을 통제해 팀 CNN 기준선과 비교하는 것이다.

## 3. 구조와 사전학습

Vision Transformer는 이미지를 작은 조각(Patch)으로 나누고 조각 간 관계를 학습한다. EVA02는 ImageNet-22k에서 가린 패치의 특징을 학습한 후 ImageNet-1k로 미세조정됐다. Fine-tuning은 이 가중치에서 시작해 새 과제에 맞게 파라미터를 갱신하는 과정이다. 이 실행은 공식 파일 해시를 확인하고 새 Linear의 두 텐서를 제외한 특징 추출기 전체를 로드했다. [공식 모델 카드](https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k)

## 4. 데이터와 공통 코드

팀의 고정 명단에 따라 43,197장으로 파라미터를 학습했다. Holdout은 학습하지 않고 모델 선택에만 사용하는 2,425장이다. 공식 Validation은 1,158장이다. categories 순서대로 27개 multi-hot 라벨을 복원하고 CSV·NPZ를 대조했다. 분할 간 동일 이름 및 동일 바이트 이미지가 없었다. 공통 Dataset·전처리·F1 코드는 2번 팀원이 사용한 Git 버전과 일치했다. 자세한 집계는 `data_audit.json`에 있다.

## 5. 입력 해상도와 전처리

378×378의 14픽셀 패치는 27×27개다. 패치 위치를 표현하는 Position Embedding은 기본 24×24에서 bicubic 보간했다. CLS 위치는 유지했다. 회전 위치 특징(RoPE)은 대상 격자에서 생성된다. CLS를 뺀 패치 특징 평균 → LayerNorm → Linear(384→27)를 사용한다. 27개 Logit은 sigmoid를 적용하기 전의 점수다. 원본 JPEG를 EXIF 보정 후 RGB로 변환하고, 비율을 유지해 축소·확대한 뒤 중앙 검정 padding과 체크포인트별 정규화를 한 번 적용했다. 학습에만 수평 반전 0.5를 적용했다.

## 6. 학습 설정

BCEWithLogitsLoss는 각 카테고리의 존재 여부를 logit에서 직접 학습하는 손실이다. AdamW는 gradient에 따라 파라미터를 갱신하며 weight decay를 분리해 적용한다. backbone LR 5e-5, head LR 1e-3, decay 0.05, seed 42다. Gradient accumulation은 16장씩 두 번의 기울기를 모아 유효 배치 32를 만드는 방식이다. AMP는 연산에 혼합 정밀도(FP16)를 사용하는 방법이며 overflow로 갱신을 건너뛰면 scheduler도 멈춘다. 40,500개 계획 갱신 중 첫 2,025개는 warmup, 이후 cosine을 적용했다.

## 7. 학습 과정

실제 {end} epoch를 실행했다. Train BCE는 첫 epoch {history[0]['train_bce']:.6f}에서 마지막 {last['train_bce']:.6f}로 변했다. Epoch별 학습시간, F1, LR, AMP skip, 메모리는 `training_history.csv`에 있다.

![Epoch별 BCE 및 Holdout Macro F1](learning_curves.png)

## 8. Best 선정

Macro F1은 27개 클래스 F1의 동일 가중 평균이다. 최고 Holdout Macro F1은 epoch {best_ep}의 {chosen['holdout_macro']:.6f}였다. 같은 점수에서는 더 이른 epoch를 유지했다. Early stopping은 선택 지표의 개선이 멈추면 학습을 끝내는 규칙이다. 종료 epoch {end}, 종료 사유는 {stop}다.

## 9. Holdout 성능

best를 strict 복원한 뒤 재평가했다. Micro/Macro/Image-wise F1은 {hold['micro_f1']:.6f} / {hold['macro_f1']:.6f} / {hold['image_f1']:.6f}다. Holdout BCE는 {hold['bce']:.6f}다. Micro는 모든 TP/FP/FN을 합산하며 Image-wise는 사진별 F1을 평균한다.

## 10. 공식 Validation

동일한 best와 임계값 0.5에서 공식 Validation BCE는 {val['bce']:.6f}였다. 빈 예측 비율은 {val['empty_pred_rate']:.2%}, 이미지당 평균 예측 라벨은 {val['mean_pred_labels']:.2f}개다. 저장한 확률로 세 F1과 클래스별 지표를 재계산해 일치를 확인했다.

## 11. 클래스별 오류

낮은 F1 다섯 클래스는 {poorest}다. leg warmer는 정답 {leg['support']}장에서 TP {leg['EVA_tp']}, FP {leg['EVA_fp']}, FN {leg['EVA_fn']}이었다. 정답 7장에서는 한 장으로도 recall이 1/7 바뀐다.

라벨이 두 개 이상인 {val['multilabel_images']}장 중 {val['multilabel_underpredicted_count']}장은 정답 수보다 적은 라벨을 예측했다. 한 사진에서 함께 발생한 FN/FP 중 가장 잦은 조합은 {pair['fn']} 누락과 {pair['fp']} 추가({pair['images']}장)였다. 이미지 수준 기록만으로 두 물체를 서로 혼동했는지 구분할 수 없다.

## 12. 과적합과 분포

Holdout BCE가 가장 낮은 시점은 epoch {low['epoch']}({low['holdout_bce']:.6f}), 마지막은 {last['holdout_bce']:.6f}였다. 최종 Holdout Macro F1은 {last['holdout_macro']:.6f}였다. BCE는 확신의 크기에 반응하고 F1은 임계값을 넘는지에 반응한다. 둘은 다른 epoch에서 최적일 수 있다.

공식 Val − Holdout의 Micro/Macro/Image-wise 차이는 {gaps}다. 원본 train과 val에서 shoe 등장률은 각각 52.51%, 69.43%였다. 클래스 구성과 표본 수가 달라지는 두 평가셋 사이의 차이를 Private Test 성능으로 환산할 수 없다.

## 13. 파라미터와 비용

전체·학습 가능 파라미터는 모두 21,813,147개, 상한은 50,000,000개다. RTX 4060에서 누적 학습 {minutes:.2f}분, 최대 allocated {mem:.3f} GB, reserved {reserved:.3f} GB였다. 공식 Val forward {val['forward_ms_per_image']:.3f} ms/장, 로더 포함 {val['end_to_end_ms_per_image']:.3f} ms/장이다. 시간은 CUDA 동기화를 포함하고 GB는 10⁹ bytes다.

## 14. 한계

Seed 한 개를 측정했다. 공식 Val에서 반복 선택하거나 threshold를 조정하지 않았다. 가림·물체 크기·주석 경계는 별도 검증이 필요한 가설이다. 장비와 체크포인트가 다른 팀 결과의 시간·성능 차이를 아키텍처의 독립 효과로 해석할 수 없다.

## 15. 결론

고정 조건에서 공식 Val의 ConvNeXt 대비 Micro/Macro/Image-wise 차이는 {diff}다. 모델 선택 판단은 [두 백본 비교 보고서](EVA02_vs_ConvNeXt_비교보고서.md)에 기록했다.

## 16. 재현

[실행 안내](README.md)에 의존성, 입력 폴더, train/evaluate/infer 명령을 기록했다. `config.json`은 계획, `resolved_config.json`은 실제 설정이다. 개인별 예측 파일은 비공개 폴더에 저장한다.
"""
    with (HERE / "EVA02_Small_실험보고서.md").open("w", encoding="utf-8") as f:
        f.write(one)
    comparison_table = "\n".join(["| 분할 | 지표 | ConvNeXt | EVA02 | EVA−ConvNeXt |", "|---|---|---:|---:|---:|",
                                *[f"| {r['split']} | {r['metric']} | {r['ConvNeXt_report']:.6f} | {r['EVA02']:.6f} | {r['delta']:+.6f} |" for r in summary]])
    details = "\n".join("| {} | {} | {:.4f} | {:.4f} | {:+.4f} | {:+.6f} |".format(
        r["class"], r["support"], r["ConvNeXt_f1"], r["EVA_f1"], r["delta_f1"], r["macro_contribution"]) for r in comparable)
    known = len(comparable)
    direction = "EVA02" if summary[4]["delta"] > 0 else "ConvNeXt" if summary[4]["delta"] < 0 else "동점"
    fp_up = [r for r in comparable if r["delta_fp"] is not None and r["delta_fp"] > 0 and r["delta_f1"] < 0]
    fp_note = ", ".join(f"{r['class']} (FP {r['delta_fp']:+d})" for r in fp_up) or "확인된 클래스 중 없음"
    val_note = "EVA의 세 지표가 모두 낮았다." if all(val[k] < REF["val"][k] for k in METRICS) else "지표별 변화는 표의 EVA−ConvNeXt 열에 표시했다."
    known_delta = sum(r["macro_contribution"] for r in comparable)
    residual = summary[4]["delta"] - known_delta
    contribution_note = (f"확인된 {known}개 클래스의 기여 합은 약 {known_delta:+.6f}다. "
                         f"전체 Macro 차이에서 이 합을 빼면 나머지 {27-known}개의 합산 기여는 약 {residual:+.6f}다. "
                         "개별 클래스에 이 잔차를 배분할 자료는 없다.") if known < 27 else ""
    accessory = [r for r in comparable if r["class"] in ("glove", "scarf", "leg warmer") and r["delta_fn"] is not None]
    accessory_note = ", ".join(f"{r['class']} {r['ConvNeXt_fn']}→{r['EVA_fn']}개" for r in accessory)
    cn_epochs = {3: (.077381, .139571, .652264), 10: (.009334, .252751, .707369), 15: (.002751, .321645, .696809)}
    matched = ["| Epoch | ConvNeXt Train / Hold BCE / Macro | EVA Train / Hold BCE / Macro |", "|---:|---|---|"]
    for r in history:
        if r["epoch"] in cn_epochs:
            cn = " / ".join(f"{v:.6f}" for v in cn_epochs[r["epoch"]])
            matched.append(f"| {r['epoch']} | {cn} | {r['train_bce']:.6f} / {r['holdout_bce']:.6f} / {r['holdout_macro']:.6f} |")
    matched_table = "\n".join(matched)
    two = f"""# EVA02-Small과 ConvNeXt-Tiny 비교 보고서

## 1. 비교 목적

같은 이미지와 F1 정의에서 두 사전학습 Backbone 및 Fine-tuning 설정을 비교했다. ConvNeXt 수치는 팀원이 보고한 실측치이며 EVA 수치는 이 PC에서 산출했다.

## 2. 통제 조건

고정 분할, 클래스 순서, 공통 전처리 원칙, 평가 코드, BCE·AdamW, 유효 배치 32, seed 42, 30 epoch 예산, warmup 5%, threshold 0.5와 holdout 선택 규칙을 맞췄다.

## 3. 모델별 차이

입력은 384 대 378, backbone LR은 1e-4 대 5e-5다. 공식 정규화, 사전학습 목적함수와 계산 장비도 다르다. [실제 설정 대조표](comparison_conditions.md)

## 4. 구조

ConvNeXt는 주변 영역의 패턴을 합성곱으로 학습하는 CNN이다. EVA02는 패치 간 관계를 학습하는 Vision Transformer다. 두 모델 모두 평균 풀링과 정규화 후 27개 점수를 낸다.

## 5–6. Holdout과 공식 Validation

{comparison_table}

공식 Validation에서 Macro F1이 더 높은 쪽은 **{direction}**다. {val_note}

## 7. 클래스별 기여

| 공식 Val 클래스 | 정답 | ConvNeXt F1 | EVA F1 | ΔF1 | ΔF1 / 27 |
|---|---:|---:|---:|---:|---:|
{details}

{known}/27개 클래스에서 기준 F1과 비교했다. 원자료의 반올림값만 있는 항목은 CSV에서 별도 표기했다. {"27개 차이의 평균과 전체 Macro 차이의 일치를 검증했다." if all(r["baseline_precision_source"] == "counts" for r in classes) else "기준 자료가 없는 항목을 제외했다. 이 표의 기여 합으로 전체 Macro 변화를 검증할 수 없다."} FP가 늘면서 F1이 하락한 항목은 {fp_note}다. FP 증가만으로 하락 전체를 설명할 수는 없다.

{contribution_note}

![공식 Val 클래스별 F1](class_f1_comparison.png)
![공식 Val 클래스별 Precision과 Recall](class_precision_recall_comparison.png)

## 8. 희귀 클래스 오류

leg warmer 정답 7장 중 ConvNeXt는 0장, EVA는 {leg['EVA_tp']}장을 찾았다. FP는 각각 0개, {leg['EVA_fp']}개다. 이 표본에서는 TP 하나의 차이도 recall을 약 14.3%p 바꾼다. cardigan, jumpsuit, vest, glove 등은 위 표와 `class_f1_comparison.csv`의 TP/FP/FN을 함께 본다.

액세서리의 실제 FN 변화(ConvNeXt→EVA)는 {accessory_note}다.

## 9. 학습 곡선과 분포

ConvNeXt의 Holdout BCE 최소는 epoch 3, Macro 최대는 10, 종료는 15였다. EVA의 대응 시점은 {low['epoch']}, {best_ep}, {end}였다. EVA의 공식 Val − Holdout F1은 {gaps}였다. 성능 격차는 적은 양성 표본과 클래스 비율에도 영향을 받는다.

{matched_table}

## 10. 계산 비용

| 측정 | ConvNeXt / RTX A5000 | EVA02 / RTX 4060 |
|---|---:|---:|
| 파라미터 | 27,840,891 | 21,813,147 |
| 누적 train 분 | 58.97 | {minutes:.2f} |
| 학습 peak allocated / reserved GB | 3.652 / 3.735 | {mem:.3f} / {reserved:.3f} |
| 공식 val forward ms/장 | 2.666 | {val['forward_ms_per_image']:.3f} |
| 로더 포함 ms/장 | 4.392 | {val['end_to_end_ms_per_image']:.3f} |

## 11. 공정성

한 seed에서 관측한 차이다. 반복 실행 변동, 크기가 작은 클래스의 불확실성, 장비 및 소프트웨어 차이가 남는다. 모델 구조만의 효과나 통계적 유의성을 판정할 근거는 없다.

## 12. 최종 Backbone 후보

이번 고정 설정에서 공식 Val Macro F1을 우선하면 {direction}이다. 두 모델 모두 50M을 만족하며 EVA는 파라미터가 6,027,744개 적다. 배포 속도 기준의 최종 선택에는 동일 장비에서의 속도 측정이 필요하다.

## 13. 후속 비교 제안

후속으로는 공식 Val을 튜닝에 사용하지 않고 같은 Holdout에서 seed 변동을 확인하는 비교가 적절하다. 모델별 속도는 동일 장비·배치에서 재측정할 수 있다. 이번 실행은 고정된 기본 실험 한 번이다.
"""
    (HERE / "EVA02_vs_ConvNeXt_비교보고서.md").write_text(two, encoding="utf-8")

