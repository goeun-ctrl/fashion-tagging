# external_merged — 학습용 외부 데이터 (희귀 클래스 보강)

원본 train에 **추가로만** 쓰는 외부 이미지 9,731장. holdout·공식 val은 원본 그대로여서, 외부 데이터 없이 돌린
실험과 바로 비교된다. 만든 사람: 5번 (2026-11-04).

## 1. 폴더 구조

```
external_merged/
├── README.md          이 파일
├── categories.txt     과제와 같은 27개 클래스, 같은 순서
├── labels.csv         file_name + 27개 클래스 열 (값 1 / 0 / -1)
├── sources.csv        이미지별 출처·그룹·라이선스·원본 주소·크롭 영역
├── report.md          클래스별 1 / 0 / -1 개수
└── images/            이미지 9,731장 — git에는 없음. 아래 3장 명령으로 받는다
```

| 이미지 이름 | 장수 | 출처 | group |
|---|---|---|---|
| `lvis_<id>.jpg` | 4,491 | COCO 사진 + LVIS 주석 | `rare_extra_clean` |
| `openimages_<id>.jpg` | 5,057 | Open Images V7 (사람이 확인한 이미지 단위 라벨) | `rare_extra_clean` |
| `openverse_<id>_crop.jpg` | 176 | Openverse(주로 Flickr) leg warmer 사진, 무릎~발 크롭 | `legwarmer_web_crop` |
| `wikimedia_<id>_crop.jpg` | 7 | Wikimedia Commons leg warmer 사진, 무릎~발 크롭 | `legwarmer_web_crop` |

## 2. 라벨 (`labels.csv`)

과제의 labels.csv와 열 구성은 같고(file_name + 27개), **값이 세 종류**다.

| 값 | 뜻 | 학습 때 |
|---|---|---|
| `1` | 있음 (외부 주석으로 확인) | 정답으로 학습 |
| `0` | 없음 (사람이 "없음"으로 확인) | 정답으로 학습 |
| `-1` | 모름 (외부 데이터셋이 이 클래스를 확인하지 않음) | **손실에서 제외** |

이미지당 확인된 라벨은 평균 1.9개(1~17개), 나머지는 -1이다. 예: 우산 사진은 `umbrella=1`이고 보이는 신발은
`shoe=-1`이라, "있는 신발을 없다고" 배우지 않는다.

외부 클래스 → 우리 클래스 대응은 뜻이 분명히 같은 것만 했다(`scripts/external/build_external_labels.py`의 MAP).
예: LVIS `jersey`(T-shirt) → `top, t-shirt, sweatshirt`. LVIS 일반 `shirt`, `legging`, overalls, parasol 등 애매한 것은
대응하지 않아 항상 -1이다.

대상 클래스별 "있음" 수 (우리 train 양성과 비교):

| 클래스 | 외부 있음 | 우리 train |
|---|---|---|
| umbrella | 5,596 | 134 |
| glove | 3,261 | 757 |
| vest | 684 | 717 |
| leg warmer | 183 | 61 |
| cape | 48 | 152 |
| jumpsuit | 31 | 922 |

다른 클래스도 일부 확인 라벨이 있다(shoe, hat, bag 등). 전체는 `report.md`.

## 3. 이미지 받기 (git에는 이미지가 없음)

```
python scripts/external/fetch_external_images.py --dataset data/external/external_merged
```
- `sources.csv`의 `url`에서 받고, `crop_box`가 있는 행(leg warmer 웹 크롭)은 같은 영역을 다시 잘라 만든다.
- LVIS·Open Images는 원본과 바이트까지 같고, 웹 크롭은 JPEG 재저장 차이(평균 픽셀 차 < 1)만 있다.
- 끊기면 다시 실행하면 이어서 받는다. 원본이 사라져 못 받은 것은 `fetch_failed.csv`에 남고, `--prune`을 붙이면
  labels.csv에서도 뺀다.
- 연구실 서버(`/data/eunjin/DL/fassion_tagging/data/external/external_merged/`)에도 같은 이미지가 있다.

## 4. 학습에 쓰기

```
python scripts/train.py --exp_id R5_cnxt_extra --train_dir data/train --val_dir data/val \
    --extra_train data/external/external_merged --extra_color match --extra_weight 0.3
```
| 옵션 | 설명 |
|---|---|
| `--extra_train` | 이 폴더를 학습 데이터에 추가 (holdout·val에는 안 들어감) |
| `--extra_color match` | `group`마다 채도·밝기를 train_inner 평균에 맞춘다 (웹 크롭 약 ×0.67, LVIS·Open Images 약 ×0.86, ±20% 무작위) |
| `--extra_weight` | 외부 이미지 한 장을 뽑는 비중 (원본 = 1). umbrella·glove가 많으니 1보다 작게 권장 |
| `--extra_max N` | 외부 이미지 N장만 사용 (seed 고정 무작위 추출) |

-1 라벨 처리는 `src/fashion_tagging/training/losses.py`(BCE·ASL 모두), 읽기는 `load_partial_labels()`.

## 5. 만든 과정과 걸러낸 것

```
build_external_labels.py   LVIS·Open Images 라벨 → 우리 27개 (1/0/-1), 후보 선정, 야구 장면 glove 제외 규칙
download_external_images.py
flag_domains.py            CLIP으로 패션과 먼 장면 표시
prepare_external.py        깨짐·작은 이미지, 우리 데이터와 중복, 외부끼리 중복, --exclude_list 제외
collect_openverse_wikimedia.py → crop_legwarmers.py   leg warmer 자유 라이선스 사진 수집 → OWLv2로 무릎~발 크롭
merge_external.py          두 폴더를 이 폴더 하나로 (sources.csv에 group 추가)
```

| 걸러낸 것 | 장수 |
|---|---|
| 야구 장면의 glove (라벨 규칙: 야구공·배트·글러브·선수 등이 함께 있음) | 563 |
| CLIP 장면 판별: 기타 경기 620, 단체 구기 357, 군대 323, 야구 227, 탈것 219, 장난감 80 | 1,826 |
| 우리 데이터와 중복 (64비트 dHash 거리 ≤ 4): train 160, **val 1, 공개 test 1** | 162 |
| 외부 이미지끼리 중복 24, 너무 작은 이미지 1 | 25 |
| leg warmer 웹 사진: 검출 실패 106, 크롭이 너무 작음 7, 우리 데이터와 중복 3 | 116 |

남은 한계:
- 파라솔·차양 장면 우산, 스키 장갑, 작업복 조끼 등은 CLIP이 표시했지만 **남겨 두었다** (`flags/` 결과 참고).
- jumpsuit·cape는 외부에서도 적다. leg warmer 183장은 크롭 사진이라 나머지 클래스는 전부 -1이다.
- 라벨은 외부 데이터셋 주석을 옮긴 것이라 우리 정의와 조금 다를 수 있다 (예: LVIS vest에 정장 조끼와 안전조끼가 섞임).

## 6. 라이선스

- LVIS 주석 CC BY 4.0, COCO 이미지는 Flickr 원작자 조건. Open Images 이미지는 CC BY 2.0(원작자 표기 필요).
- 웹 크롭은 CC0 / 퍼블릭 도메인 / BY / BY-SA / BY-NC / BY-NC-SA만 받았다(크롭하므로 ND 제외).
  원작자·라이선스·원본 페이지는 `sources.csv`의 `creator`, `license`, `page`.
- 수업 과제의 학습·연구용이다. 이미지를 다시 배포하지 말고 `sources.csv`의 출처로 받는다.
