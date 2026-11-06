# 정은 · ConvNeXt-Tiny 공통 B안 외부 3,458장 결과

**학습·최종 평가·저장 예측 재계산 검증 완료.** 외부 데이터가 증가한 결과 공식 val Micro/Image-wise F1은 상승했지만 Macro F1은 하락했습니다. 전체 클래스 개선으로 해석하지 않습니다.

- [27개 카테고리 상세 비교 보고서](results/27개카테고리_상세비교보고서.md): 27개 클래스×holdout/val×세 실험, Support·Precision·Recall·F1·TP·FP·FN 및 변화 해석.
- [전체 실험 보고서](results/V2_3458장_최종실험보고서.md)
- [27개 클래스 상세 CSV](results/세실험_27개클래스_상세비교.csv)
- [전체 F1](results/세실험_전체F1.csv) / [시간·VRAM 비용](results/세실험_시간_메모리.csv)
- [best.pt 및 예측 포함 결과 묶음](https://github.com/goeun-ctrl/fashion-tagging/releases/tag/s2-cnxt-common3458-b-s42-20261106)

|공식 val|Micro F1|Macro F1|Image-wise F1|
|---|---:|---:|---:|
|원본-only 공통증강|0.818619|0.652603|0.822416|
|외부506|0.814441|0.653716|0.813430|
|외부3458|0.825295|0.643132|0.826315|

Best epoch14, 종료19. 학습67.65분, 최대 PyTorch allocated VRAM3.5766 GB /3.3310 GiB, 파라미터27,840,891개. 공유 GPU 시간으로 부하·I/O 조건을 확인해야 합니다. 공식 val은 최종 평가 전용이며 threshold0.5를 유지했습니다.

## 가중치 받기 및 검증

Release의 **best.pt는 서버 원본 그대로의 전체 체크포인트(334,360,261bytes)**이며 optimizer·scheduler 상태도 포함합니다. `model` 키에 모델 state_dict가 있습니다. last.pt는 서버에 보존했고 이번 Release에는 포함하지 않았습니다.

SHA-256: `3187431ad69eac3ce177e44e8b04bbe99c5d170743788e530c7ffb9ac94193c6`

서버→로컬 SHA-256 일치 및 CPU strict load(누락·초과 키0개, head27×768)를 검증했습니다. 추가 GPU 추론이나 성능 재현을 수행한 검증은 아닙니다. 해당 검증 JSON도 이 폴더와 Release에 포함합니다. 신뢰하는 이 실험 체크포인트에 한해 다음과 같이 로드합니다.

```python
import torch, timm
checkpoint = torch.load('best.pt', map_location='cpu', weights_only=False)
model = timm.create_model('convnext_tiny.fb_in22k_ft_in1k_384', pretrained=False,
                         num_classes=27, global_pool='avg')
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
```

## 코드·데이터와 재실행

실제 실행 코드 스냅샷은 `code/`에 있습니다. 기존 친구 코드·기준선·506장 코드와 분리했습니다. `pipeline.py`와 `preflight.py`는 실제 실행한 학교 서버 절대경로가 포함된 기록이며 다른 서버에서 그대로 실행하면 안 됩니다. 아래 trainer CLI는 입력 경로를 명시할 수 있습니다.

- 데이터: [V2 Release 원본 패키지](https://github.com/goeun-ctrl/fashion-tagging/releases/tag/fashion-common-expanded-partial-20261106-v2)
- V2 lock SHA: `e0675cde5e43fbdd2e0e06ca5b4f901a27afedec1ac0dc6e93aed031ed36efc4`
- 공통 코드 원본 SHA: `c5827dfa749e276d4e5431628a06730b26c4841f`
- 원본 train_inner43,197 + 외부3,458 = 후보46,655. epoch마다 원본 Plan43197회 복원추출, shuffle=False.
- 입력RGB384, bicubic종횡비유지·중앙검정padding, checkpoint mean/std. Pillow11.3.0 필수.
- Masked BCE(-1제외), 누적창 확인 라벨 총수 정규화, batch16×accum2, seed42, AdamW1e-4/1e-3, WD0.05, warmup5%+cosine, 최대30/patience5.
- 기존 baseline/506 best에서 재개하지 않고 공식 사전학습에서 시작. EMA·추가 Drop Path·ASL 없음. 기존 bias/norm weight decay 처리 유지.

```bash
# 공통 코드 원본을 별도 신규 폴더로 추출. 기존 체크아웃을 전환하지 않음.
python prepare_reference.py --repo /path/to/fashion-tagging --out /new/path/pinned-reference
# 승인된 GPU와 새 출력 경로 사용. 본학습 전 실제 환경·데이터 검사와 --smoke 점검 필요.
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python code/trainer.py \
  --prepared-dir /path/to/fashion-common-expanded-partial-20261106-v2 \
  --reference-dir /new/path/pinned-reference --train-dir /path/to/train \
  --val-dir /path/to/val --output-dir /new/private/output \
  --batch-size 16 --accum 2 --workers 4 --device cuda:0
```

실제 환경·전체 설정·epoch노출·로그·예측NPZ는 Release의 `convnext_common3458_results.zip`에 있습니다. 원본 이미지·가상환경·Git 인증정보는 포함하지 않았습니다. 학습은 seed42 단일실험이며 결정성 옵션 warn_only이므로 비트 단위 재현을 보장하지 않습니다. 부분 라벨을 사람 전수검수 완료로 표시하지 않습니다.
