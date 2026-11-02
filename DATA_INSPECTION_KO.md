# 제공 데이터 확인 결과

확인일: 2026-10-30. 원본 ZIP을 변경하거나 전체 압축을 해제하지 않고 내부 목록, README, categories, CSV와 NPZ 전체를 읽었다. train/val 각각 앞의 이미지 8장, 총 16장을 디코딩하고 라벨과 함께 시각 확인했다.

## 파일 구조

```text
MAT4091_fashion_train.zip       MAT4091_fashion_val.zip
└─ train/                      └─ val/
   ├─ README.txt                  ├─ README.txt
   ├─ categories.txt              ├─ categories.txt
   ├─ labels.csv                  ├─ labels.csv
   ├─ labels.npz                  ├─ labels.npz
   └─ images/*.jpg                └─ images/*.jpg
```

현재 폴더에는 과제 PDF와 이 두 ZIP이 제공되어 있다. 공개 테스트 샘플, baseline 코드, CVPR 템플릿은 발견하지 못했다. ZIP README에는 baseline 코드에서 NPZ를 사용한다고 적혀 있지만 코드 자체는 ZIP에 없다.

## 크기와 라벨

| 항목 | train | val |
|---|---:|---:|
| 이미지 수 | 45,622 | 1,158 |
| 클래스 수 | 27 | 27 |
| 이미지당 평균 라벨 수 | 2.9580 | 3.2530 |
| 이미지당 최소/최대 라벨 수 | 1 / 11 | 1 / 9 |
| 빈 정답 이미지 | 0 | 0 |
| NPZ targets shape | (45622, 27) | (1158, 27) |

- CSV의 `file_name`은 images 폴더 안의 파일명이다.
- `labels`는 JSON 배열을 담은 문자열이다. 다른 27개 열은 이진 타깃이다.
- NPZ 키는 `filenames`, `targets`, `categories`다. targets dtype은 uint8이다.
- categories.txt의 한 줄이 하나의 클래스다. `top, t-shirt, sweatshirt` 같은 이름은 분리하면 안 된다.

## 전체 메타데이터 검사 결과

- split 내부 라벨 파일명의 중복: 각각 0건.
- 라벨이 참조하지만 ZIP에 없는 이미지: 각각 0건.
- ZIP에 있지만 라벨에 없는 이미지: 각각 0건.
- JSON 라벨 집합과 이진 열의 불일치: 각각 0건.
- 모든 타깃 값이 0 또는 1이다.
- CSV와 NPZ의 파일명 순서, 카테고리 순서, 타깃 배열이 모두 일치한다.
- train/val 사이 동일 파일명: 0건. **이미지 내용 중복·유사 이미지 여부는 검사하지 않았다.**
- 모든 이미지의 디코딩/손상 검사, ZIP 전체 CRC 검사, 전체 이미지 크기 통계는 수행하지 않았다.

## 클래스별 양성 이미지 수

순서는 실제 categories.txt와 같다.

| 클래스 | train | val |
|---|---:|---:|
| shirt, blouse | 6,115 | 102 |
| top, t-shirt, sweatshirt | 16,176 | 463 |
| sweater | 1,485 | 21 |
| cardigan | 1,101 | 12 |
| jacket | 7,742 | 179 |
| vest | 717 | 21 |
| pants | 12,347 | 313 |
| shorts | 2,745 | 106 |
| skirt | 5,034 | 162 |
| coat | 3,085 | 103 |
| dress | 18,666 | 506 |
| jumpsuit | 922 | 21 |
| cape | 152 | 5 |
| glasses | 4,848 | 130 |
| hat | 2,514 | 74 |
| headband, head covering, hair accessory | 2,984 | 105 |
| tie | 1,455 | 3 |
| glove | 757 | 18 |
| watch | 3,373 | 82 |
| belt | 6,667 | 159 |
| leg warmer | 61 | 7 |
| tights, stockings | 2,202 | 62 |
| sock | 1,445 | 51 |
| shoe | 23,954 | 804 |
| bag, wallet | 6,907 | 205 |
| scarf | 1,362 | 48 |
| umbrella | 134 | 5 |

## 프로젝트에 반영할 해석

1. **클래스 불균형**: train에서 shoe는 23,954장, leg warmer는 61장이다. 전체 평균뿐 아니라 클래스별 precision/recall/F1/support를 기록해야 한다.
2. **검증 표본의 한계**: tie는 val 양성이 3장이다. 클래스별 임계값이나 개선 효과가 소수 사례에 크게 좌우될 수 있어 정밀 튜닝의 안정성을 확인해야 한다.
3. **빈 정답 부재**: 제공 train/val로는 무대상 이미지에 대한 거짓 양성 억제 성능을 직접 평가할 수 없다. 빈 예측은 구현하되 무대상 일반화가 검증됐다고 주장할 수 없다.
4. **분포 차이**: 예를 들어 shoe 양성 비율은 train 약 52.51%, val 약 69.43%다. 검증 성능을 비공개 테스트 성능으로 단정할 수 없다.
5. **사진 형식 다양성**: 확인한 16장에는 가로/세로 사진, 인물 일부만 보이는 사진, 작은 액세서리가 있는 사진이 포함된다. 전체 라벨을 유지하면서 강한 crop을 적용하면 대상이 잘릴 가능성을 고려해야 한다. 전체 데이터의 시각적 분포를 확인한 것은 아니다.

## 과제의 제출 제약

최종 추론 총 파라미터는 50,000,000개 이하이고, 오프라인 실행이 가능해야 한다. 출력은 이미지별 test_id와 정확한 클래스명 목록을 담은 JSONL이다. 모델·임계값은 train/val에서 선택하며 테스트 데이터는 학습/튜닝에 사용할 수 없다. 논문은 CVPR 형식 본문 6페이지이고 참고문헌은 제외한다. 공식 평가 가중치, 일정, 발표 세부 사항은 추후 LMS 공지 대상이다.
