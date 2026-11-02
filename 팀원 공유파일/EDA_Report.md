# MAT4091 Fashion Category Tagging — 데이터 EDA 및 라벨 분포 분석

작성 기준: 2026-10-30T21:50:22.688970+09:00 (KST). 분석 코드 실제 실행 결과만 사용. 모델 학습·추론·Threshold 튜닝 없음.

## 1. Executive Summary

- 전체 이미지: Train 45,622, Val 1,158, Examples 30, Public Test 300. 모델 설계용 라벨 통계는 Train/Val로 계산.
- Train 최다 shoe 23,954장, 최소 leg warmer 61장으로 392.69배 차이.
- 가장 큰 클래스 등장률 차이: shoe, Train 52.51%, Val 69.43% (+16.92%p).
- 이미지당 라벨 평균 Train 2.958, Val 3.253; 중앙값 3/3. 다중 라벨 비율 79.49%/92.23%.
- 빈 정답 Train 0장, Val 0장, Examples 1장. 현재 검증셋의 빈 정답 성능 평가 가능성은 제한됨.
- Val 양성 10개 미만 4클래스, 30개 미만 9클래스. 클래스별 F1·Threshold는 소수 표본에 민감할 수 있음.
- 원본 display 짧은 변 중앙값 Train 682px, Val 682px; 세로형 79.97%/79.27%.
- 실제 디코딩 실패 총 0장. Train/Val 정확한 중복 0해시 그룹, 전체 라벨 충돌 0그룹.

## 2. Dataset Overview

| 분할 | 이미지 수 | 라벨 행 | 읽기 성공 | 읽기 오류 | 확장자 | 실제 형식 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 45,622 | 45,622 | 45,622 | 0 | {'.jpg': 45622} | {'JPEG': 45622} |
| val | 1,158 | 1,158 | 1,158 | 0 | {'.jpg': 1158} | {'JPEG': 1158} |
| examples | 30 | 30 | 30 | 0 | {'.jpg': 13, '.png': 17} | {'JPEG': 13, 'PNG': 17} |
| public_test | 300 | 미사용 | 300 | 0 | {'.jpg': 195, '.png': 102, '.jpeg': 3} | {'JPEG': 198, 'PNG': 102} |

입력은 로컬 압축 해제 폴더이며 ZIP 입력도 지원한다. categories.txt는 **줄 하나가 클래스 하나**이고 순서가 0-based 출력 인덱스이다. 모든 분할의 순서가 동일함을 검사했다. 쉼표를 기준으로 클래스명을 나누지 않는다.

| 인덱스 | 카테고리 |
| --- | --- |
| 0 | shirt, blouse |
| 1 | top, t-shirt, sweatshirt |
| 2 | sweater |
| 3 | cardigan |
| 4 | jacket |
| 5 | vest |
| 6 | pants |
| 7 | shorts |
| 8 | skirt |
| 9 | coat |
| 10 | dress |
| 11 | jumpsuit |
| 12 | cape |
| 13 | glasses |
| 14 | hat |
| 15 | headband, head covering, hair accessory |
| 16 | tie |
| 17 | glove |
| 18 | watch |
| 19 | belt |
| 20 | leg warmer |
| 21 | tights, stockings |
| 22 | sock |
| 23 | shoe |
| 24 | bag, wallet |
| 25 | scarf |
| 26 | umbrella |

Train/Val CSV는 file_name, labels 및 위 27개 이름의 binary 열로 구성된다. 파일명/JSON은 object, binary 열은 int64이다. Examples는 test_id가 추가된다. labels는 카테고리명 문자열의 JSON 리스트이며 []는 정상 빈 정답, 이진 행은 모두 0이다. NumPy는 allow_pickle=False로 읽는다.

| 분할 | CSV shape | NumPy 배열 | 키 연결 후 불일치 행 | JSON-이진 불일치 행 |
| --- | --- | --- | --- | --- |
| train | [45622, 29] | {'filenames': {'shape': [45622], 'dtype': '<U36'}, 'targets': {'shape': [45622, 27], 'dtype': 'uint8'}, 'categories': {'shape': [27], 'dtype': '<U39'}} | 0 | 0 |
| val | [1158, 29] | {'filenames': {'shape': [1158], 'dtype': '<U36'}, 'targets': {'shape': [1158, 27], 'dtype': 'uint8'}, 'categories': {'shape': [27], 'dtype': '<U39'}} | 0 | 0 |
| examples | [30, 30] | 없음 | 해당 없음 | 0 |

이미지 연결은 CSV file_name 및 NPZ filenames의 명시적 키를 경로/유일한 basename으로 해석해 수행했다. 행 순서 동일성을 가정하지 않았다. 각 NPZ categories 순서도 categories.txt와 비교했다. 원본 labels.npz의 targets는 uint8이며, 행렬 곱에는 int64로 승격해 오버플로를 방지했다. Public Test의 submission_template은 입력 식별자 구조만 검사하고 라벨을 사용하지 않았다.

| 분할 | 검사 | 결과 수 |
| --- | --- | --- |
| train | duplicate_label_records | 0 |
| train | duplicate_label_image_keys | 0 |
| train | duplicate_resolved_image_keys | 0 |
| train | labels_without_images | 0 |
| train | images_without_labels | 0 |
| train | nonbinary_multihot_cells | 0 |
| train | json_multihot_mismatch_rows | 0 |
| train | csv_npz_mismatch_rows_after_key_join | 0 |
| train | npz_duplicate_keys | 0 |
| val | duplicate_label_records | 0 |
| val | duplicate_label_image_keys | 0 |
| val | duplicate_resolved_image_keys | 0 |
| val | labels_without_images | 0 |
| val | images_without_labels | 0 |
| val | nonbinary_multihot_cells | 0 |
| val | json_multihot_mismatch_rows | 0 |
| val | csv_npz_mismatch_rows_after_key_join | 0 |
| val | npz_duplicate_keys | 0 |
| examples | duplicate_label_records | 0 |
| examples | duplicate_label_image_keys | 0 |
| examples | duplicate_resolved_image_keys | 0 |
| examples | labels_without_images | 0 |
| examples | images_without_labels | 0 |
| examples | nonbinary_multihot_cells | 0 |
| examples | json_multihot_mismatch_rows | 0 |
| public_test | template_missing_images | 0 |
| public_test | template_duplicate_file_names | 0 |
| public_test | images_without_template | 0 |
| public_test | template_duplicate_test_ids | 0 |

정확한 중복: Train/Val 파일명 교집합 0개, SHA-256 교집합 0그룹, 교차 파일 쌍 0개. 같은 이름만으로 중복으로 판정하지 않았다. 전체 라벨 있는 분할에서 상이한 라벨을 가진 동일 해시 0그룹.

| 분할 | duplicate_hash_groups | extra_duplicate_files | files_in_duplicate_groups | conflicting_label_hash_groups |
| --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 |
| examples | 0 | 0 | 0 | 0 |
| public_test | 0 | 0 | 0 | 0 |

원본 이미지는 저장·출력하지 않았다. 오류·경고 전체 목록은 [eda_warnings.md](eda_warnings.md), 구조·환경·집계 보존식 검사 결과는 [EDA_Summary.json](EDA_Summary.json)에 있다.

## 3. Class Distribution

다중 라벨이므로 클래스별 양성 이미지 수의 합은 전체 이미지 수보다 클 수 있다. Train 양성 할당 합 134,950, Val 3,767. 분모는 각 분할 전체 라벨 이미지 수 45,622/1,158이다. %p는 Val−Train, 상대 변화는 (Val 비율 / Train 비율 − 1)×100이다. 0 분모는 정의 불가로 표시한다.

| 클래스 | Train 양성 | Train % | Val 양성 | Val % | Val−Train %p | 상대 변화 % |
| --- | --- | --- | --- | --- | --- | --- |
| shirt, blouse | 6,115 | 13.40 | 102 | 8.81 | -4.60 | -34.28 |
| top, t-shirt, sweatshirt | 16,176 | 35.46 | 463 | 39.98 | 4.53 | 12.77 |
| sweater | 1,485 | 3.26 | 21 | 1.81 | -1.44 | -44.29 |
| cardigan | 1,101 | 2.41 | 12 | 1.04 | -1.38 | -57.06 |
| jacket | 7,742 | 16.97 | 179 | 15.46 | -1.51 | -8.91 |
| vest | 717 | 1.57 | 21 | 1.81 | 0.24 | 15.39 |
| pants | 12,347 | 27.06 | 313 | 27.03 | -0.03 | -0.13 |
| shorts | 2,745 | 6.02 | 106 | 9.15 | 3.14 | 52.14 |
| skirt | 5,034 | 11.03 | 162 | 13.99 | 2.96 | 26.78 |
| coat | 3,085 | 6.76 | 103 | 8.89 | 2.13 | 31.54 |
| dress | 18,666 | 40.91 | 506 | 43.70 | 2.78 | 6.80 |
| jumpsuit | 922 | 2.02 | 21 | 1.81 | -0.21 | -10.27 |
| cape | 152 | 0.33 | 5 | 0.43 | 0.10 | 29.60 |
| glasses | 4,848 | 10.63 | 130 | 11.23 | 0.60 | 5.64 |
| hat | 2,514 | 5.51 | 74 | 6.39 | 0.88 | 15.97 |
| headband, head covering, hair accessory | 2,984 | 6.54 | 105 | 9.07 | 2.53 | 38.63 |
| tie | 1,455 | 3.19 | 3 | 0.26 | -2.93 | -91.88 |
| glove | 757 | 1.66 | 18 | 1.55 | -0.10 | -6.32 |
| watch | 3,373 | 7.39 | 82 | 7.08 | -0.31 | -4.22 |
| belt | 6,667 | 14.61 | 159 | 13.73 | -0.88 | -6.04 |
| leg warmer | 61 | 0.13 | 7 | 0.60 | 0.47 | 352.10 |
| tights, stockings | 2,202 | 4.83 | 62 | 5.35 | 0.53 | 10.93 |
| sock | 1,445 | 3.17 | 51 | 4.40 | 1.24 | 39.05 |
| shoe | 23,954 | 52.51 | 804 | 69.43 | 16.92 | 32.23 |
| bag, wallet | 6,907 | 15.14 | 205 | 17.70 | 2.56 | 16.93 |
| scarf | 1,362 | 2.99 | 48 | 4.15 | 1.16 | 38.84 |
| umbrella | 134 | 0.29 | 5 | 0.43 | 0.14 | 47.00 |

### 빈도 순위 및 불균형

Train 상위 5개:

| 클래스 | Train 양성 | Train % |
| --- | --- | --- |
| shoe | 23,954 | 52.51 |
| dress | 18,666 | 40.91 |
| top, t-shirt, sweatshirt | 16,176 | 35.46 |
| pants | 12,347 | 27.06 |
| jacket | 7,742 | 16.97 |

Train 하위 5개:

| 클래스 | Train 양성 | Train % |
| --- | --- | --- |
| leg warmer | 61 | 0.13 |
| umbrella | 134 | 0.29 |
| cape | 152 | 0.33 |
| vest | 717 | 1.57 |
| glove | 757 | 1.66 |

Val 상위 5개:

| 클래스 | Val 양성 | Val % |
| --- | --- | --- |
| shoe | 804 | 69.43 |
| dress | 506 | 43.70 |
| top, t-shirt, sweatshirt | 463 | 39.98 |
| pants | 313 | 27.03 |
| bag, wallet | 205 | 17.70 |

Val 하위 5개:

| 클래스 | Val 양성 | Val % |
| --- | --- | --- |
| tie | 3 | 0.26 |
| cape | 5 | 0.43 |
| umbrella | 5 | 0.43 |
| leg warmer | 7 | 0.60 |
| cardigan | 12 | 1.04 |

| 분할 | 클래스 빈도 평균 | 중앙값 | 표준편차 | 최다−최소 수 | 최다/최소 양성 배수 |
| --- | --- | --- | --- | --- | --- |
| train | 4,998.15 | 2,745.00 | 5,963.22 | 23,893 | 392.69 |
| val | 139.52 | 82.00 | 182.35 | 801 | 268.00 |

희귀성은 절대 기준으로 확정하지 않고 **관측 Train 빈도 하위 사분위(≤1,231.5개)**를 상대적 점검 대상으로 제안한다: cardigan, vest, jumpsuit, cape, glove, leg warmer, umbrella. 이 경계는 관측된 27개 클래스 빈도 분포에서 계산했으며 의미적 난도나 최적 손실 가중치를 뜻하지 않는다. 검증 안정성은 별도로 Val 양성 0 / 1–9 / 10–29 / 30 이상으로 표시하며, 30도 안정성을 보장하는 임계값은 아니다.

| 클래스 | Train 음성 | Train P/N | Train N/P | Val 음성 | Val P/N | Val N/P |
| --- | --- | --- | --- | --- | --- | --- |
| shirt, blouse | 39,507 | 0.155 | 6.461 | 1,056 | 0.097 | 10.353 |
| top, t-shirt, sweatshirt | 29,446 | 0.549 | 1.820 | 695 | 0.666 | 1.501 |
| sweater | 44,137 | 0.034 | 29.722 | 1,137 | 0.018 | 54.143 |
| cardigan | 44,521 | 0.025 | 40.437 | 1,146 | 0.010 | 95.500 |
| jacket | 37,880 | 0.204 | 4.893 | 979 | 0.183 | 5.469 |
| vest | 44,905 | 0.016 | 62.629 | 1,137 | 0.018 | 54.143 |
| pants | 33,275 | 0.371 | 2.695 | 845 | 0.370 | 2.700 |
| shorts | 42,877 | 0.064 | 15.620 | 1,052 | 0.101 | 9.925 |
| skirt | 40,588 | 0.124 | 8.063 | 996 | 0.163 | 6.148 |
| coat | 42,537 | 0.073 | 13.788 | 1,055 | 0.098 | 10.243 |
| dress | 26,956 | 0.692 | 1.444 | 652 | 0.776 | 1.289 |
| jumpsuit | 44,700 | 0.021 | 48.482 | 1,137 | 0.018 | 54.143 |
| cape | 45,470 | 0.003 | 299.145 | 1,153 | 0.004 | 230.600 |
| glasses | 40,774 | 0.119 | 8.410 | 1,028 | 0.126 | 7.908 |
| hat | 43,108 | 0.058 | 17.147 | 1,084 | 0.068 | 14.649 |
| headband, head covering, hair accessory | 42,638 | 0.070 | 14.289 | 1,053 | 0.100 | 10.029 |
| tie | 44,167 | 0.033 | 30.355 | 1,155 | 0.003 | 385.000 |
| glove | 44,865 | 0.017 | 59.267 | 1,140 | 0.016 | 63.333 |
| watch | 42,249 | 0.080 | 12.526 | 1,076 | 0.076 | 13.122 |
| belt | 38,955 | 0.171 | 5.843 | 999 | 0.159 | 6.283 |
| leg warmer | 45,561 | 0.001 | 746.902 | 1,151 | 0.006 | 164.429 |
| tights, stockings | 43,420 | 0.051 | 19.718 | 1,096 | 0.057 | 17.677 |
| sock | 44,177 | 0.033 | 30.572 | 1,107 | 0.046 | 21.706 |
| shoe | 21,668 | 1.106 | 0.905 | 354 | 2.271 | 0.440 |
| bag, wallet | 38,715 | 0.178 | 5.605 | 953 | 0.215 | 4.649 |
| scarf | 44,260 | 0.031 | 32.496 | 1,110 | 0.043 | 23.125 |
| umbrella | 45,488 | 0.003 | 339.463 | 1,153 | 0.004 | 230.600 |

Train에는 있고 Val에는 없는 클래스: 없음. Val 양성 <10: cape 5개, tie 3개, leg warmer 7개, umbrella 5개. Val 양성 <30: sweater 21개, cardigan 12개, vest 21개, jumpsuit 21개, cape 5개, tie 3개, glove 18개, leg warmer 7개, umbrella 5개. 양성 한 장이 recall에 미치는 단위 변화는 100/support %p이며 F1 변화량과 같지는 않다.

| 클래스 | Val 양성 | 등장률 95% 하한 % | 등장률 95% 상한 % | 양성 1장당 recall %p |
| --- | --- | --- | --- | --- |
| shirt, blouse | 102 | 7.31 | 10.58 | 0.98 |
| top, t-shirt, sweatshirt | 463 | 37.20 | 42.83 | 0.22 |
| sweater | 21 | 1.19 | 2.76 | 4.76 |
| cardigan | 12 | 0.59 | 1.80 | 8.33 |
| jacket | 179 | 13.49 | 17.65 | 0.56 |
| vest | 21 | 1.19 | 2.76 | 4.76 |
| pants | 313 | 24.55 | 29.66 | 0.32 |
| shorts | 106 | 7.63 | 10.95 | 0.94 |
| skirt | 162 | 12.11 | 16.11 | 0.62 |
| coat | 103 | 7.39 | 10.67 | 0.97 |
| dress | 506 | 40.86 | 46.57 | 0.20 |
| jumpsuit | 21 | 1.19 | 2.76 | 4.76 |
| cape | 5 | 0.18 | 1.01 | 20.00 |
| glasses | 130 | 9.53 | 13.17 | 0.77 |
| hat | 74 | 5.12 | 7.95 | 1.35 |
| headband, head covering, hair accessory | 105 | 7.55 | 10.86 | 0.95 |
| tie | 3 | 0.09 | 0.76 | 33.33 |
| glove | 18 | 0.99 | 2.44 | 5.56 |
| watch | 82 | 5.74 | 8.70 | 1.22 |
| belt | 159 | 11.87 | 15.83 | 0.63 |
| leg warmer | 7 | 0.29 | 1.24 | 14.29 |
| tights, stockings | 62 | 4.20 | 6.80 | 1.61 |
| sock | 51 | 3.37 | 5.74 | 1.96 |
| shoe | 804 | 66.72 | 72.02 | 0.12 |
| bag, wallet | 205 | 15.61 | 20.01 | 0.49 |
| scarf | 48 | 3.14 | 5.45 | 2.08 |
| umbrella | 5 | 0.18 | 1.01 | 20.00 |

![Train 클래스 빈도](figures/train_class_counts.png)

![Val 클래스 빈도](figures/val_class_counts.png)

![클래스 등장률 및 Wilson 구간](figures/class_prevalence_comparison.png)

![양성 음성 불균형](figures/positive_negative_imbalance.png)

## 4. Multi-label Characteristics

| 분할 | 평균 | 중앙값 | 표준편차 | 최소 | 최대 | 0개 수 | 0개 % | 1개 수 | 1개 % | 2개 수 | 2개 % | 3개 이상 수 | 3개 이상 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2.96 | 3.00 | 1.58 | 1.00 | 11.00 | 0 | 0.00 | 9,355 | 20.51 | 10,989 | 24.09 | 25,278 | 55.41 |
| val | 3.25 | 3.00 | 1.44 | 1.00 | 9.00 | 0 | 0.00 | 90 | 7.77 | 307 | 26.51 | 761 | 65.72 |
| examples | 8.97 | 9.00 | 2.37 | 0.00 | 12.00 | 1 | 3.33 | 0 | 0.00 | 0 | 0.00 | 29 | 96.67 |

전형적인 라벨 수는 Train 중앙값 3개, Val 3개이다. 다중 라벨(≥2) 비율은 Train 79.49%, Val 92.23%이며 단일 라벨 비율은 20.51%/7.77%이다. 빈 라벨은 오류가 아니다. 빈 라벨 등장률의 Wilson 95% 구간은 Train [0.0000, 0.0084]%, Val [0.0000, 0.3306]%이다. 0건 관측으로 모집단 비율이 0이라고 증명할 수 없으며, 이 구간은 IID 표집 가정에 의존한다. Examples의 1건은 출력 형식 확인 사례이며 빈 정답 평가 표본으로 일반화하지 않는다.

![라벨 수 전체 분포](figures/label_count_histogram.png)

27×27 행렬은 각 split의 class_cooccurrence_*_matrix.csv에, 모든 순서 있는 쌍의 지표는 class_cooccurrence.csv에 저장했다. C(A,B)=두 클래스가 함께 있는 이미지 수, P(A∩B)=C/N, P(A|B)=C/n(B), Jaccard=C/[n(A)+n(B)−C], Lift=C×N/[n(A)n(B)]. 대각선 C(A,A)는 클래스 빈도이다. 조건부 확률의 열 B가 분모다. 0 분모는 공란/null로 저장한다.

Train 동시 등장 수 상위 10쌍(대각선 제외):

| A | B | 동시 등장 수 | P(A∩B) | P(A\|B) | Jaccard | Lift |
| --- | --- | --- | --- | --- | --- | --- |
| dress | shoe | 10,787 | 0.236 | 0.450 | 0.339 | 1.101 |
| top, t-shirt, sweatshirt | shoe | 8,195 | 0.180 | 0.342 | 0.257 | 0.965 |
| pants | shoe | 8,065 | 0.177 | 0.337 | 0.286 | 1.244 |
| top, t-shirt, sweatshirt | pants | 6,673 | 0.146 | 0.540 | 0.305 | 1.524 |
| shoe | bag, wallet | 4,684 | 0.103 | 0.678 | 0.179 | 1.292 |
| jacket | pants | 4,608 | 0.101 | 0.373 | 0.298 | 2.199 |
| jacket | shoe | 4,601 | 0.101 | 0.192 | 0.170 | 1.132 |
| belt | shoe | 3,935 | 0.086 | 0.164 | 0.147 | 1.124 |
| shirt, blouse | pants | 3,760 | 0.082 | 0.305 | 0.256 | 2.272 |
| top, t-shirt, sweatshirt | skirt | 3,659 | 0.080 | 0.727 | 0.208 | 2.050 |

Train 동시 등장 ≥30개인 쌍 중 Lift 상위 10쌍. 30은 작은 표본의 극단적 Lift를 걸러 보기 위한 표시 기준일 뿐 통계적 신뢰성을 보장하지 않는다.

| A | B | 동시 등장 수 | P(A∩B) | P(A\|B) | Jaccard | Lift |
| --- | --- | --- | --- | --- | --- | --- |
| shirt, blouse | tie | 1,402 | 0.031 | 0.964 | 0.227 | 7.189 |
| vest | tie | 161 | 0.004 | 0.111 | 0.080 | 7.041 |
| jacket | tie | 1,263 | 0.028 | 0.868 | 0.159 | 5.115 |
| headband, head covering, hair accessory | glove | 203 | 0.004 | 0.268 | 0.057 | 4.100 |
| coat | scarf | 354 | 0.008 | 0.260 | 0.086 | 3.844 |
| glove | scarf | 86 | 0.002 | 0.063 | 0.042 | 3.805 |
| pants | tie | 1,268 | 0.028 | 0.871 | 0.101 | 3.220 |
| shirt, blouse | vest | 300 | 0.007 | 0.418 | 0.046 | 3.122 |
| glove | tights, stockings | 110 | 0.002 | 0.050 | 0.039 | 3.011 |
| tights, stockings | scarf | 185 | 0.004 | 0.136 | 0.055 | 2.814 |

지정 관심 조합(상위 의류 전체 쌍 포함):

| 분할 | A | B | 동시 수 | 동시 % | P(A\|B) % | jaccard | lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | cardigan | coat | 23 | 0.050 | 0.746 | 0.006 | 0.309 |
| val | cardigan | coat | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | cardigan | jacket | 37 | 0.081 | 0.478 | 0.004 | 0.198 |
| val | cardigan | jacket | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | dress | shoe | 10,787 | 23.644 | 45.032 | 0.339 | 1.101 |
| val | dress | shoe | 367 | 31.693 | 45.647 | 0.389 | 1.045 |
| train | glasses | hat | 551 | 1.208 | 21.917 | 0.081 | 2.063 |
| val | glasses | hat | 9 | 0.777 | 12.162 | 0.046 | 1.083 |
| train | glove | belt | 114 | 0.250 | 1.710 | 0.016 | 1.031 |
| val | glove | belt | 1 | 0.086 | 0.629 | 0.006 | 0.405 |
| train | glove | watch | 9 | 0.020 | 0.267 | 0.002 | 0.161 |
| val | glove | watch | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | jacket | bag, wallet | 1,276 | 2.797 | 18.474 | 0.095 | 1.089 |
| val | jacket | bag, wallet | 47 | 4.059 | 22.927 | 0.139 | 1.483 |
| train | jacket | coat | 67 | 0.147 | 2.172 | 0.006 | 0.128 |
| val | jacket | coat | 1 | 0.086 | 0.971 | 0.004 | 0.063 |
| train | pants | shorts | 12 | 0.026 | 0.437 | 7.96e-04 | 0.016 |
| val | pants | shorts | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | pants | skirt | 62 | 0.136 | 1.232 | 0.004 | 0.046 |
| val | pants | skirt | 1 | 0.086 | 0.617 | 0.002 | 0.023 |
| train | shirt, blouse | cardigan | 129 | 0.283 | 11.717 | 0.018 | 0.874 |
| val | shirt, blouse | cardigan | 2 | 0.173 | 16.667 | 0.018 | 1.892 |
| train | shirt, blouse | coat | 345 | 0.756 | 11.183 | 0.039 | 0.834 |
| val | shirt, blouse | coat | 10 | 0.864 | 9.709 | 0.051 | 1.102 |
| train | shirt, blouse | jacket | 2,580 | 5.655 | 33.325 | 0.229 | 2.486 |
| val | shirt, blouse | jacket | 29 | 2.504 | 16.201 | 0.115 | 1.839 |
| train | shirt, blouse | sweater | 147 | 0.322 | 9.899 | 0.020 | 0.739 |
| val | shirt, blouse | sweater | 3 | 0.259 | 14.286 | 0.025 | 1.622 |
| train | shirt, blouse | top, t-shirt, sweatshirt | 474 | 1.039 | 2.930 | 0.022 | 0.219 |
| val | shirt, blouse | top, t-shirt, sweatshirt | 18 | 1.554 | 3.888 | 0.033 | 0.441 |
| train | shorts | skirt | 8 | 0.018 | 0.159 | 0.001 | 0.026 |
| val | shorts | skirt | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | sock | shoe | 1,397 | 3.062 | 5.832 | 0.058 | 1.841 |
| val | sock | shoe | 50 | 4.318 | 6.219 | 0.062 | 1.412 |
| train | sweater | cardigan | 13 | 0.028 | 1.181 | 0.005 | 0.363 |
| val | sweater | cardigan | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | sweater | coat | 128 | 0.281 | 4.149 | 0.029 | 1.275 |
| val | sweater | coat | 5 | 0.432 | 4.854 | 0.042 | 2.677 |
| train | sweater | jacket | 152 | 0.333 | 1.963 | 0.017 | 0.603 |
| val | sweater | jacket | 5 | 0.432 | 2.793 | 0.026 | 1.540 |
| train | top, t-shirt, sweatshirt | cardigan | 439 | 0.962 | 39.873 | 0.026 | 1.125 |
| val | top, t-shirt, sweatshirt | cardigan | 6 | 0.518 | 50.000 | 0.013 | 1.251 |
| train | top, t-shirt, sweatshirt | coat | 1,079 | 2.365 | 34.976 | 0.059 | 0.986 |
| val | top, t-shirt, sweatshirt | coat | 36 | 3.109 | 34.951 | 0.068 | 0.874 |
| train | top, t-shirt, sweatshirt | jacket | 2,876 | 6.304 | 37.148 | 0.137 | 1.048 |
| val | top, t-shirt, sweatshirt | jacket | 85 | 7.340 | 47.486 | 0.153 | 1.188 |
| train | top, t-shirt, sweatshirt | sweater | 123 | 0.270 | 8.283 | 0.007 | 0.234 |
| val | top, t-shirt, sweatshirt | sweater | 1 | 0.086 | 4.762 | 0.002 | 0.119 |
| train | watch | belt | 735 | 1.611 | 11.024 | 0.079 | 1.491 |
| val | watch | belt | 14 | 1.209 | 8.805 | 0.062 | 1.243 |

**해석 한계:** 동시 등장은 시각적 혼동·오분류를 의미하지 않는다. 한 이미지에 여러 사람이 있을 수 있어 이 지표로 같은 사람의 착장 관계도 확정할 수 없다. 희귀 조합은 반드시 횟수와 함께 해석한다.

![Train Jaccard](figures/train_cooccurrence_jaccard.png)

![Val Jaccard](figures/val_cooccurrence_jaccard.png)

## 5. Image Characteristics

모든 파일을 순차 스트리밍 SHA-256 및 Pillow verify 후 실제 픽셀 decode로 확인했다. 해상도 통계는 읽기 성공 이미지 기준이다. width/height 원본 통계는 JSON에, EXIF 회전을 반영한 display 축 통계는 아래 표에 있다. 픽셀 수는 width×height이고 aspect ratio는 width/height이다.

| 분할 | 변수 | 최소 | Q25 | 중앙값 | Q75 | P95 | 최대 | 평균 | 표준편차 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | display_width | 232.00 | 682.00 | 682.00 | 769.00 | 1,024.00 | 1,024.00 | 755.24 | 146.87 |
| train | display_height | 388.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 986.52 | 106.45 |
| train | short_side | 232.00 | 681.00 | 682.00 | 722.00 | 1,024.00 | 1,024.00 | 717.76 | 112.95 |
| train | long_side | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 0.00 |
| train | pixels | 237,568.00 | 697,344.00 | 698,368.00 | 739,328.00 | 1,048,576.00 | 1,048,576.00 | 734,988.60 | 115,661.01 |
| train | aspect_ratio | 0.23 | 0.67 | 0.67 | 0.75 | 1.50 | 2.64 | 0.79 | 0.27 |
| val | display_width | 415.00 | 681.00 | 682.00 | 768.00 | 1,024.00 | 1,024.00 | 750.74 | 147.82 |
| val | display_height | 471.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 962.38 | 131.81 |
| val | short_side | 415.00 | 680.00 | 682.00 | 683.00 | 819.00 | 1,024.00 | 689.12 | 74.47 |
| val | long_side | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 1,024.00 | 0.00 |
| val | pixels | 424,960.00 | 696,320.00 | 698,368.00 | 699,392.00 | 838,656.00 | 1,048,576.00 | 705,654.49 | 76,258.75 |
| val | aspect_ratio | 0.41 | 0.67 | 0.67 | 0.75 | 1.50 | 2.17 | 0.82 | 0.33 |

| 분할 | portrait | landscape | square | 극단 비율 수 | 짧은 변<128px 수 | RGB 등 mode | EXIF 축 교환 수 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 79.97 | 11.53 | 8.50 | 750 | 0 | {'RGB': 45436, 'L': 186} | 0 |
| val | 79.27 | 18.57 | 2.16 | 16 | 0 | {'RGB': 1158} | 0 |

위 portrait/landscape/square는 % 단위이며, 정사각형은 너비=높이의 정확한 일치다. 극단 비율은 AR<0.5 또는 >2, 작은 이미지는 짧은 변<128px라는 점검 기준이다. 이러한 조건만으로 데이터 오류로 처리하지 않았다.

Train에는 그레이스케일 L 이미지 186장 (0.408%), Val에는 0장이 있다. 입력 파이프라인에서 RGB 변환과 채널 수 통일을 명시적으로 확인해야 한다. 이 차이는 실제 파일 mode에서 확인된 입력 처리 항목이며 성능 영향은 측정하지 않았다.

![너비 높이 분포](figures/image_dimensions.png)

![종횡비 분포](figures/aspect_ratio.png)

![픽셀 수 분포](figures/image_pixels.png)

| 입력 변 px | 224 대비 입력 픽셀 배수 | Train 짧은 변 미만 % | Val 짧은 변 미만 % | Train 긴 변 미만 % | Val 긴 변 미만 % |
| --- | --- | --- | --- | --- | --- |
| 224 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 320 | 2.04 | 0.06 | 0.00 | 0.00 | 0.00 |
| 384 | 2.94 | 0.26 | 0.00 | 0.00 | 0.00 |
| 448 | 4.00 | 0.70 | 0.35 | 0.00 | 0.00 |

짧은 변<입력 크기는 정사각형 강제 리사이즈나 짧은 변 기준 resize 시 적어도 한 축의 확대가 필요한 비율이다. 긴 변<입력 크기는 긴 변을 맞추는 letterbox에서 확대가 필요한 비율이다. 이 둘을 혼동하지 않는다.

Train/Val 세로형 비율은 79.97%/79.27%로, 비정사각형을 정사각형에 강제 맞추면 축별 배율이 달라진다. 축별 배율 비의 중앙값은 1.501/1.501이다. 종횡비 유지+패딩은 기하 왜곡을 줄이는 합리적 비교 후보이지만, 정사각형 안에서 실제 이미지가 차지하는 면적의 평균은 70.09%/67.30%이므로 유효 해상도 손실도 비교해야 한다.

고해상도 입력은 전체 이미지 정보를 더 많이 보존할 가능성이 있지만 액세서리의 실제 픽셀 크기는 이 라벨로 알 수 없다. 320/384/448의 224 대비 입력 픽셀 면적 배수는 2.041/2.939/4.000이며 실제 FLOPs·속도·메모리는 구조와 구현에 따라 달라진다. 이 수치로 작은 객체 성능 향상을 단정하지 않는다.

## 6. Train–Validation Comparison

클래스 등장률 차이 절댓값 상위 8개:

| 클래스 | Train % | Val % | Val−Train %p | 차이 95% 하한 %p | 차이 95% 상한 %p | BH q |
| --- | --- | --- | --- | --- | --- | --- |
| shoe | 52.5054 | 69.4301 | 16.9247 | 14.1722 | 19.5506 | 1.94e-29 |
| shirt, blouse | 13.4036 | 8.8083 | -4.5953 | -6.1274 | -2.7970 | 1.86e-05 |
| top, t-shirt, sweatshirt | 35.4566 | 39.9827 | 4.5262 | 1.7077 | 7.4096 | 0.0057 |
| shorts | 6.0168 | 9.1537 | 3.1369 | 1.5923 | 4.9483 | 0.0002 |
| skirt | 11.0342 | 13.9896 | 2.9555 | 1.0540 | 5.0917 | 0.0062 |
| tie | 3.1893 | 0.2591 | -2.9302 | -3.1679 | -2.4062 | 1.61e-11 |
| dress | 40.9145 | 43.6960 | 2.7816 | -0.0856 | 5.6897 | 0.1092 |
| bag, wallet | 15.1396 | 17.7029 | 2.5633 | 0.4466 | 4.8904 | 0.0408 |

![클래스 등장률 차이](figures/prevalence_difference.png)

| feature_type | feature | unit | train | val | val_minus_train | difference_unit | pvalue | bh_qvalue | effect_metric | effect_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| numeric_distribution | labels_per_image | mean | 2.9580 | 3.2530 | 0.2950 | original_units | 2.04e-16 | 정의 불가 | KS_D | 0.1273 |
| numeric_distribution | display_width | mean | 755.2421 | 750.7392 | -4.5029 | original_units | 3.52e-08 | 정의 불가 | KS_D | 0.0887 |
| numeric_distribution | display_height | mean | 986.5202 | 962.3765 | -24.1437 | original_units | 1.51e-05 | 정의 불가 | KS_D | 0.0721 |
| numeric_distribution | aspect_ratio | mean | 0.7926 | 0.8242 | 0.0315 | original_units | 3.52e-08 | 정의 불가 | KS_D | 0.0887 |
| numeric_distribution | pixels | mean | 734,988.6030 | 705,654.4940 | -29,334.1091 | original_units | 1.14e-24 | 정의 불가 | KS_D | 0.1567 |
| label_count | histogram_total_variation | probability | 정의 불가 | 정의 불가 | 정의 불가 | 정의 불가 | 정의 불가 | 정의 불가 | total_variation | 0.1298 |
| label_count | empty_label_pct | percent | 0.0000 | 0.0000 | 0.0000 | percentage_points | 정의 불가 | 정의 불가 | 정의 불가 | 정의 불가 |
| cooccurrence | off_diagonal_joint_probability | percentage_points | 정의 불가 | 정의 불가 | 정의 불가 | 정의 불가 | 정의 불가 | 정의 불가 | mean_absolute_difference | 0.3986 |

27개 클래스의 Fisher 검정에서 BH q<0.05인 클래스는 13개이다. 검정은 이미지 IID 및 분할 독립성을 가정한 탐색적 참고값이며 다중 라벨의 클래스 의존·동일 출처 군집을 완전히 반영하지 않는다. p-value가 작다는 사실과 실질적 영향은 구분한다. 예를 들어 가장 큰 shoe의 +16.92%p 차이는 직접적인 빈도 변화이며, 희귀 클래스의 큰 상대 변화율은 작은 분모의 영향일 수 있다. KS는 효과 크기 D를 우선 보고, 이산 라벨 수와 ties에서는 p-value를 과도하게 해석하지 않는다. 검정 비유의는 분포 동일성의 증명이 아니다.

Val은 클래스 목록과 다중 라벨 특성을 공유하지만 주요 등장률·라벨 수·크기 분포 차이가 관측되어 Train의 동일 분포 표본이라고 확정할 수 없다. Val의 희귀 양성 부족(cape 5개, tie 3개, leg warmer 7개, umbrella 5개)은 클래스별 F1와 Threshold 해석을 제한한다. 모델 선택 시 Micro F1뿐 아니라 Macro/Image-wise F1 및 클래스별 support를 함께 확인하고, 동일 Val에서 반복 선택한 결과의 선택 편향을 기록해야 한다. Public/Hidden Test 분포는 추정하지 않았다.

## 7. Bottleneck Hypothesis Analysis

### H1 — 클래스 불균형

- 관련 EDA 결과: Train 최다 shoe 23,954개, 최소 leg warmer 61개, 최다/최소 양성 392.69배. 빈도 하위 사분위 경계 1,231.5개.
- 지지 여부: 빈도 불균형은 확인됨. 일반 BCE가 실제로 다수 클래스 위주 오류를 내는지는 판단 불가.
- EDA만으로 판단 불가: 손실/그래디언트, 클래스별 precision·recall·F1, 예측 확률을 관찰하지 않음.
- 후속 실험: 동일 backbone·해상도·학습 예산에서 BCE를 기준으로 Asymmetric Loss와 Distribution-Balanced Loss를 비교. Micro/Macro/Image-wise F1 및 희귀 클래스 recall·false positive를 함께 기록.

### H2 — 작은 액세서리

- 관련 EDA 결과: glasses: Train 4,848 (10.63%), Val 130; glove: Train 757 (1.66%), Val 18; watch: Train 3,373 (7.39%), Val 82; belt: Train 6,667 (14.61%), Val 159; sock: Train 1,445 (3.17%), Val 51
- 지지 여부: 각 액세서리의 빈도는 위 수치로 확인됨. 모두 희귀하다고 일반화할 수 없음. 실제 객체 크기가 작다는 가설은 판단 불가.
- EDA만으로 판단 불가: 이미지 라벨과 전체 해상도만으로 객체 픽셀 크기·위치·가림·관측 난도를 알 수 없음.
- 후속 실험: 224/320/384/448 입력을 통제 실험하고 클래스별 성능·메모리·속도를 측정. 고해상도 및 multi-scale·attention 효과는 각각 ablation으로 분리.

### H3 — 유사 카테고리

- 관련 EDA 결과: shirt, blouse 6,115/102; top, t-shirt, sweatshirt 16,176/463; sweater 1,485/21; cardigan 1,101/12; jacket 7,742/179; coat 3,085/103 (Train/Val 양성 수). 지정 클래스 쌍의 동시 등장·조건부 확률은 별도 표 참조.
- 지지 여부: 등장 빈도와 함께 나타나는 관계만 확인됨. 시각적 혼동 여부는 판단 불가.
- EDA만으로 판단 불가: 동시 등장, 배타적 등장, 높은 Lift는 오분류의 증거가 아님. 객체·인물 수준 관계도 알 수 없음.
- 후속 실험: C1 예측의 클래스별 false positive/false negative와 조건부 오류를 분석한 뒤 C2 Class Query의 추가 효과 확인.

### H4 — Empty Label

- 관련 EDA 결과: Train 0/45,622 (0.00%), Val 0/1,158 (0.00%), Examples 1/30 (3.33%).
- 지지 여부: 빈 정답 출력 지원은 필요함. Train/Val에서의 빈 정답 성능은 해당 표본 수로 제한되며, 이 결과로 실제 배포 빈도를 추정할 수 없음.
- EDA만으로 판단 불가: Examples는 대표성 있는 평가 표본이 아니며 빈 예시만으로 Threshold를 최적화할 수 없음.
- 후속 실험: 추론 파이프라인의 [] 출력을 단위 확인. 향후 허용된 별도 검증 설계에서 empty-label false positive 및 empty/nonempty image-wise F1 정의를 확인.

### H5 — Validation 안정성

- 관련 EDA 결과: 양성 10개 미만 4클래스: cape 5개, tie 3개, leg warmer 7개, umbrella 5개. 30개 미만 9클래스: sweater 21개, cardigan 12개, vest 21개, jumpsuit 21개, cape 5개, tie 3개, glove 18개, leg warmer 7개, umbrella 5개.
- 지지 여부: 작은 양성 표본으로 인한 추정·Threshold 민감도 위험을 지지함. 실제 과적합 정도는 판단 불가.
- EDA만으로 판단 불가: Wilson 등장률 구간은 F1 구간이 아님. 모델 score 분포·검증 예측·출처 군집 정보가 없음.
- 후속 실험: 먼저 고정/공통 Threshold 기준선을 확립하고 클래스별 최적화는 별도 내부 검증 또는 OOF 안정성·반복성 확인 후 비교. 동일 검증셋에 구조와 Threshold를 반복 맞추는 선택 편향에 주의.

## 8. Implications for Model Design

| 후보 | 데이터 기반 시사점 | 검증할 사항 |
| --- | --- | --- |
| C1 ConvNeXt-T + GAP + Linear | 27개 독립 양성 출력과 빈 집합을 지원하는 기준선. 현재 EDA는 이 구조의 성능을 측정하지 않음. | 동일 split·전처리에서 평가 3종 및 클래스별 오류·score 확보. |
| C2 Class Query Cross-Attention | 평균 2.96개 라벨 및 동시 등장 관계가 관측됨. 쿼리 헤드 우월성의 증거는 아님. | C1 대비 head만 교체한 ablation; 파라미터 증가량·오류 양상. |
| C3 Multi-scale + C2 | 전체 이미지 해상도와 액세서리 빈도는 확인했으나 객체 크기 증거 없음. | 해상도 효과와 feature pyramid 효과 분리; 메모리·속도·클래스별 F1. |
| C4 Swin-T | EDA만으로 ConvNeXt 대비 backbone 우열 결정 불가. | 동일 입력·학습 예산·손실·사전학습 조건 통제. |
| C5 EfficientNetV2-S + ResNet50 | 앙상블 이득과 오류 상보성은 예측이 있어야 확인 가능. | 최종 추론에 함께 사용하는 모든 backbone/head의 실제 파라미터 합계 50M 이하인지 계산. 지연·메모리도 측정. |
| C6 Detection/Segmentation 합집합 | 현재 제공 라벨에는 박스·마스크·인물 구분이 없음. 데이터의 기원만으로 위치 라벨이 제공됐다고 간주할 수 없음. | 허용되는 위치 지도·사전학습 사용 조건 및 분류 대비 비용/효과. 모든 사람의 카테고리 합집합 출력 준수. |
| C7 Knowledge Distillation | 50M 추론 예산과 관련된 실험 후보이며 EDA로 효과를 판단할 수 없음. | teacher/student 오류와 label coverage, student 추론 파라미터 총합, 수업 규정상 teacher 사용 허용 범위. |

공통 요구사항: 27개 카테고리 순서를 유지하는 다중 라벨 출력, 여러 사람의 합집합 정답 의미, 빈 리스트 출력 지원, 최종 추론 전체 모델 합계 ≤50M. EDA는 어떤 후보가 우수한지 결정하지 않는다. 구체적인 아키텍처 구현과 head를 포함한 실제 파라미터 계산은 후속 실험에서 수행해야 한다.

## 9. Limitations

관측 빈도는 제공 파일 전체의 정확한 집계이다. 신뢰구간은 이 파일들을 더 큰 모집단의 표본으로 보는 IID 가정하의 불확실성으로, 데이터 수집·출처·인물 군집이 알려져 있지 않아 실제보다 좁을 수 있다. 라벨 누락·정답의 의미적 정확성은 구조 검사만으로 보장할 수 없다. 박스·마스크·인물 수 정보가 없어 객체 크기, 여러 사람의 합집합 정확성, 가림, 시각적 혼동을 판단할 수 없다. 정확한 바이트 해시만 검사하므로 재인코딩·리사이즈·유사 이미지 중복은 미검사이다. 모델을 학습하거나 예측하지 않아 F1·보정·Threshold·개선 폭은 측정하지 않았다. Examples는 형식 확인용이며 대표성 있는 성능 검증에 사용하지 않았다. Public Test는 입력 무결성만 확인했으며 라벨을 추정하거나 모델 선택에 활용하지 않았다. Hidden Test 분포는 알 수 없다. 입력 면적 비율은 실제 계산 비용 벤치마크를 대신하지 않는다.

빈도 하위 사분위, 동시 수≥30, extreme AR, 작은 이미지 경계는 명시적 탐색 기준이며 보편적 난도·희귀성 기준이 아니다. 전체 분포를 모두 계산했으며 랜덤 표본 통계는 사용하지 않았다. std는 ddof=0이다. 경고를 숨기거나 라벨을 자동 보정하지 않았다.

## 10. Next Experiments

1. 동일 split의 C1 기준선에서 Micro/Macro/Image-wise F1, 클래스별 support·precision·recall·F1, 예측 확률을 확보한다. 빈 정답·빈 예측의 image-wise F1 처리 규칙은 수업 평가 정의와 맞춘다.
2. 공통 Threshold 기준으로 224/320/384/448 및 종횡비 유지 패딩의 효과를 통제 비교한다. 학습률·epoch·사전학습 조건·메모리·속도·파라미터 수를 기록한다.
3. 동일 구조·해상도에서 BCE와 불균형 대응 손실을 비교하고 빈도별 성능과 false positive 변화를 확인한다.
4. 실제 오류 분석 후 C2/C3를 하나씩 추가하여 head 및 multi-scale의 독립 효과를 확인한다. C4는 동일 예산의 backbone 대안으로 비교한다.
5. 드문 클래스 Threshold는 내부 검증/OOF 또는 반복성 평가로 안정성을 먼저 확인한다. 공식 Val 반복 사용에 따른 선택 편향을 기록하고 Public Test로 조정하지 않는다.
6. C5/C7은 단일 모델의 오류·예산 기준이 마련된 뒤 검토한다. 최종 추론에 참여하는 전체 모델 파라미터 합계 ≤50M을 실제로 계산한다. C6는 추가 위치 지도와 허용 조건을 확인한 뒤 검토한다.
7. 허용된 자료로 빈 정답 검증 표본과 출처 군집·유사 이미지 중복을 점검할 수 있는지 확인한다. 현재 원본 분할은 변경하지 않는다.

재현 방법은 [README.md](README.md), 외부 분석 도구에 제공할 집계 전용 자료는 [Model_Design_Input.md](Model_Design_Input.md)를 참조한다. 원본 이미지·개별 파일명·이미지별 라벨·원본 라벨 행은 보고서에 포함하지 않았다.
