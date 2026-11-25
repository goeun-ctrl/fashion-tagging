# 재설계 B 구현 계획 (분할 지도 기반 태깅)

작성 2026-11-22 · 5번(Optimizer·Head) · **승인 전에는 구현·학습을 시작하지 않는다.**

## 0. 왜 바꾸나

지금 최종 모델(`R5_bce_kd_cnxtdv3_lr3e5_s42`, ConvNeXt-T DINOv3)은 태그 정답만 보고 학습해서, 옷이 어디 있는지 배운 적이 없다.
Grad-CAM 진단에서 놓친 태그는 에너지의 38~39%만 정답 마스크 안에 있었고, 작은 옷 재현율이 −0.075 / −0.049(holdout / val) 낮았다.
B는 **Fashionpedia 정답 마스크로 칸(4px)마다 옷 종류를 학습**하고, 태그는 **그 분할 지도에서만** 읽는다(전역 헤드 없음).

## 1. 확정한 설계 (전부 사용자가 정한 값)

| 항목 | 값 |
|---|---|
| 백본 | ConvNeXt-T DINOv3, stage 1~4 |
| 디코더 | UPerNet(PPM 1·2·3·6 + FPN + fuse), 폭 256, 출력 stride 4, **디코더 fp32**. 37.0M 파라미터, 26.5 ms/장(측정) |
| 칸 출력 | **그룹별 softmax** 4그룹(각 그룹에 "없음"): 상의·겉옷 0,1,2,3,4,5,9,12 / 하의·원피스 6,7,8,10,11 / 다리·발 20~23 / 액세서리 13~19,24,25,26 |
| 칸 정답 | **소프트**(칸을 덮은 비율) |
| 분할 손실 | 소프트 CE + Dice, 1:1. 클래스 가중치 = 빈도 역수의 제곱근, 상한 10배. 손실은 사진별 유효 칸 평균 |
| λ (분할 : 태그) | 짧은 실험으로 결정(0.5 / 1 / 2) |
| 부위 보조 출력 | sleeve, neckline, pocket, collar, zipper, lapel, hood 7개, 부위별 sigmoid, BCE + Dice 1:1, 옷 분할의 0.5배. 태그에는 안 씀 |
| 태그 점수 | 클래스별 비율 top-k 평균. 비율 = 0.5 × (train_inner에서 그 클래스 면적의 하위 10%), 최소 4칸 |
| 태그 손실 | 모은 점수에 BCE. 클래스 가중치 같이 적용. **태그 KD 유지**(FashionFormer, α 0.5, shift 0.3) |
| 학습 | 백본 lr 3e-5, 디코더 lr 1e-4, 전체의 2/3·11/12에서 ×0.1, warmup 2,000, **24 epoch**, 실효 배치 32, wd 0.05, drop path 0.1, clip 1.0, EMA 0.9995, holdout 최고 epoch 저장 |
| 배치 구성 | Fashionpedia 80% / COCO 사람 사진 10% / 외부 5% / 빈 사진 5% |
| 임계값 | holdout만. 클래스별 안정 구간 중앙(최고 F1 − 0.005 이내 구간의 가운데), holdout 양성 20 미만은 전역값, 빈 사진 가드, 목표 = Micro·Macro·Image F1 평균, 반반 교차 점검 |
| 판정 (2026-11-23 재작성) | 설계 목표: ① 사람을 찾고 올바른 위치를 본다 ② 작은 물체를 포착한다 ③ 헷갈리는 카테고리를 구분한다. 성능 목표 1순위: 헷갈리는 카테고리 분류. **자동 탈락**: 위치(loc_topk) holdout·val 중 ★ 하락, 또는 micro AP −0.03 초과 하락. 위치 미측정이면 보류. 그 밖에는 **검수 대상**: 헷갈리는 그룹 안 AP(상의 [0~5, 9], 하의·원피스 [6, 7, 8, 10, 11], 머리 hat·headband, 다리 tights·sock, 목 tie·scarf; 그룹별과 평균 confWG_AP)를 사람이 보고 결정. 작은 물체(작은 옷 재현율, 작은 옷 위치 loc_topk_small)와 군중 AP는 참고. 군중 클래스별은 양성 10개 이상만 |
| 쓰지 않음 | 생성 이미지, Copy-Paste, 회전 |

### 데이터별 칸 정답

| 데이터 | 사람 안쪽 | 그 밖 배경 | 가방·우산·넥타이 | 부위 | 태그 |
|---|---|---|---|---|---|
| Fashionpedia train_inner (43,197) | 정답 마스크 | 정답 마스크("없음") | 정답 마스크 | 정답 마스크 | 정답 + 태그 KD |
| COCO 사람 사진 (coco_person 5,000 + coco_crowd 18,521) | FashionFormer 칸 KD | "없음". 단 FashionFormer가 찾은 옷 영역, COCO에 없는데 검출기가 찾은 사람, 사람·물체 바깥쪽 3칸 띠는 제외 | COCO 마스크(handbag·backpack·suitcase→bag, umbrella, tie) | 제외 | umbrella·tie·bag: COCO에 있으면 1, 없고 FashionFormer도 못 찾으면 0, FashionFormer만 찾으면 제외. 나머지 24개 제외. 가중치 1배 |
| 외부 LVIS (4,491 − 11) | LVIS 양성 마스크 안쪽만 그 클래스, 나머지 전부 제외 | 제외 | (LVIS 마스크) | 제외 | 1 / 0 / −1 |
| 외부 Open Images · leg warmer 크롭 | 분할 제외 | 제외 | 제외 | 제외 | 1 / 0 / −1, 색 보정 유지 |
| 빈 사진 (8,003) | 해당 없음 | "없음" | "없음" | "없음" | 전부 0 |

## 2. 데이터 준비 (구현 1단계)

| # | 작업 | 산출물 | 시간(추정) |
|---|---|---|---|
| D1 | Fashionpedia train_inner 27클래스 + 부위 7개를 **원본 해상도 RLE**로 저장. 증강(크기·반전·모자이크·축소) 뒤에 칸 비율로 줄여 소프트 정답을 만든다 | `fashionpedia_ann/rle_b/<이름>.json` | CPU 30분 |
| D2 | FashionFormer(Swin-B)를 COCO 사람 사진 23,521장에 돌려 인스턴스(클래스, 점수, RLE) 저장. 여기서 ① 배경 제외 영역 ② 사람 안쪽 칸 KD ③ 가방·우산·넥타이 "없음" 판정을 만든다 | `neg_data/ff_inst/<id>.json` | GPU 4장 약 40분 |
| D3 | 검출기 사람 박스(`neg_data/crowd_boxes.json`, 20,000장 있음) 중 COCO 사람과 IoU < 0.5인 박스 = "빠진 사람" 제외 영역. 없는 장은 검출기를 다시 돌린다 | `neg_data/missing_person.json` | 30분 |
| D4 | LVIS 주석(받아 둠, `lvis_ann/`)에서 외부 LVIS 이미지의 우리 클래스 양성 마스크 추출. **LVIS→27클래스 대응표**를 만들어 사용자 확인 | `external/lvis_rle/`, 대응표 csv | 1시간 |
| D5 | 학습 목록 4종을 `crowd_eval/exclude_from_training.txt`와 대조. 하나라도 겹치면 학습 시작을 막는다(코드 검사) | 검사 함수 | — |
| D6 | **눈으로 확인**: 데이터 종류별 20장씩 칸 정답을 색으로 덮은 그림(띠·제외 영역 포함)을 만들어 보여준다 | `vis/b_targets/` | 10분 |

## 3. 코드 (구현 2단계)

기존 `scripts/train_hires.py`는 건드리지 않고 B를 새 파일로 만든다(지금 최종 모델 재현성 유지).

| 파일 | 내용 |
|---|---|
| `scripts/train_segb.py` | 학습 루프. 4가지 데이터 섞기(비율 고정 샘플러), 손실 합산, holdout 매 epoch 평가, best 저장 |
| `src/fashion_tagging/segb/model.py` | 백본 + UPerNet(fp32) + 그룹 softmax 헤드 + 부위 sigmoid 헤드 + 비율 top-k 풀링 |
| `src/fashion_tagging/segb/targets.py` | 데이터 종류별 칸 정답·제외 마스크·태그 만들기(위 표), 띠 3칸(바깥쪽만), 소프트 비율 |
| `src/fashion_tagging/segb/losses.py` | 그룹별 소프트 CE + Dice(제외 칸 무시, 클래스 가중치), 부위 BCE + Dice, 태그 BCE + KD |
| `src/fashion_tagging/segb/augment.py` | 흑백·옛날 사진·모자이크·축소를 마스크와 함께 적용(짧은 실험에서 하나씩 켬) |
| `scripts/infer_hires.py` | B 추론 + 풀링 바꿔 보기(max / top-k 여러 비율 / LSE, 재학습 없이) |
| `scripts/tune_threshold.py` | 안정 구간 중앙, 20 미만 전역값, 빈 사진 가드, 세 F1 평균 목표 추가 |
| `scripts/diag_segb.py` | 위치 정확도(top-k 칸 중 정답 마스크 안 비율), 큰 옷 그룹 내 AP, 작은 옷 재현율, 군중 AP, 빈 사진 오탐 |
| `tests/test_segb.py` | 풀링·손실(제외 칸)·띠·소프트 정답·누수 검사 단위 테스트 |

## 4. 실행 순서

1. **점검 실행**(200 update): 메모리(배치 8이 안 들어가면 4 × 누적 8), 속도, 분할·태그 손실 크기 비교. 결과를 보고하고 epoch 시간을 다시 추정한다.
2. **짧은 실험 16번**(6 epoch, GPU 4장, 약 4라운드). 기준 = 위 확정값(λ 1, COCO 가중치 1)

   | 묶음 | 기준 외 실행 |
   |---|---|
   | COCO 분할 가중치 | 0 / 0.25 / 0.5 |
   | λ | 0.5 / 2 |
   | weight decay | 0.1 |
   | drop path | 0.2 / 0 |
   | clip | 5.0 |
   | EMA | 0.999 / 끔 |
   | 증강 하나씩 | 흑백 0.2 / 옛날 사진 0.1 / 모자이크 0.1(2·4장) / 축소 0.3(최소 0.35) |
   | 기준 | 1 |

   하이퍼파라미터는 holdout·val micro AP와 큰 옷 AP로, 증강은 목표 지표(흑백 구간 / 군중 AP / 작은 옷 재현율)로 판정한다.
3. **본 학습**: 이긴 값과 증강을 모아 24 epoch, seed 42. 최종 CNN과의 차이가 seed 잡음(약 0.002) 안이면 seed 43을 추가한다.
4. **평가**: 풀링 바꿔 보기 → 임계값(holdout) → 판정표(1장 기준) → 실패 분석(Grad-CAM 기준선과 비교).

## 5. 일정과 자원 (추정)

| 단계 | 시간 | 자원 |
|---|---|---|
| 데이터 옮기기 | 약 1.5시간(맥 경유 약 2MB/s) | — |
| 데이터 준비 D1~D6 | 반나절 | D2만 LAB-server GPU, 나머지 gpu-server-b |
| 코드 + 테스트 | 1일 | — |
| 점검 실행 | 30분 | GPU 1장 |
| 짧은 실험 16번 | 약 16시간 | GPU 4장 |
| 본 학습 | 약 15~16시간(seed 43 추가 시 2배) | GPU 1장 |
| 평가·임계값 | 1시간 | GPU 1장 |

**서버: gpu-server-b**(sw213, RTX PRO 4500 32GB × 4, 작업 폴더 `/data1/workspaces/goeun_dl/eunjin`, 2026-11-22 사용자 지시).
팀 공용 계정이라 팀원 작업과 GPU를 나눠 쓴다. 대기열 스크립트가 빈 GPU를 골라 시작한다.

- 옮길 데이터(LAB-server → gpu-server-b, 약 11GB): holdout 사진 2,425장, 공식 val, neg_data(coco·coco_person·coco_crowd·places·COCO 주석),
  LVIS 주석, external_merged, crowd_eval, 태그 KD 점수(`ff_r50_train.npz`), 백본 가중치 캐시, 최종 CNN 예측(비교용). test_public·examples_labeled는 옮기지 않는다.
- 새 venv `eunjin/segb/.venv`(Python 3.12, torch cu128, timm, pycocotools, opencv). 캐시는 작업 폴더 안.
- **예외 하나**: FashionFormer 환경은 torch 1.9 + CUDA 11.1이라 Blackwell GPU에서 돌지 않는다. D2(FashionFormer 추론)만 LAB-server에서 돌리고 결과 파일만 옮기는 안을 제안한다.

## 6. 과제 규칙 확인

- 최종 파라미터 37.0M ≤ 50M.
- 출력은 클래스 이름 목록, 태그 단위 softmax·argmax·top-1 없음(README 8장 수정본).
- test_public·examples_labeled·Fashionpedia val2020은 학습·임계값·선택에 쓰지 않는다. 군중 평가 200장(COCO val2017)은 판정에만 쓴다.
- 추가 데이터(COCO 사람·빈 사진·LVIS·Open Images·웹 크롭)의 출처·라벨·사용 방식을 README와 논문에 적는다(과제 설명서 3장).

## 7. 승인 전에 정할 것

| # | 항목 | 제안 기본값 |
|---|---|---|
| Q1 | FashionFormer 점수 기준(배경 제외 영역, 가방 "있음" 판정) | 0.3 (태그 KD shift와 같은 teacher 동작점) |
| Q2 | FashionFormer 칸 KD: 소프트 확률 그대로 / 0.3 이상만 | 소프트 그대로 |
| Q3 | 빠진 사람 판정의 검출기 점수 기준 | 0.5 |
| Q4 | LVIS→27클래스 대응표 | D4에서 만들어 보여준 뒤 확인 |
| Q5 | 빈 사진의 부위 정답 | "없음"(옷이 없으면 부위도 없음) |
| Q6 | 모자이크는 Fashionpedia 사진끼리만 | 예 |
| Q7 | 분할 CE의 "없음" 가중치 | 1(클래스 가중치 최솟값과 같게) |
| Q8 | 태그 KD가 없는 데이터(COCO·외부·빈 사진) | 정답 태그만(α 0) |
