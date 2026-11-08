"""Generate Korean result report and curves only after actual holdout/val evaluation."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;RUN=BASE.parents[1]/'runs/role4_bce_kd/R4_bce_kd_s42'
s=json.loads((RUN/'summary.json').read_text());lock=json.loads((RUN/'selection.json').read_text())
assert s['epochs_run']==18
metrics={split:json.loads((RUN/f'{split}_metrics.json').read_text())['metrics'] for split in ('holdout','val')}
with (RUN/'history.csv').open() as f:history=list(csv.DictReader(f))
fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,k in zip(axs,['train_loss','holdout_bce','holdout_macro']):
 ax.plot([int(r['epoch']) for r in history],[float(r[k]) for r in history]);ax.set(xlabel='Epoch',title=k);ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(RUN/'curves.png');plt.close(fig)
text=f'''# R4_bce_kd_s42 최종 핵심 평가

원본 Fashionpedia, BCE + FashionFormer KD(alpha .5, shift .3), linear_ms + gapgmp. 팀 기준 R5_bce_kd_s42와 동일한 역할4 설정이다. 기존 C1 BCE·ASL 비교와 별도이며 그 선택을 새 조합에서 재검증한 결과가 아니다.

best epoch: {s['best_epoch']}, 총 18epoch, 학습 {s['train_minutes']}분. EMA holdout Macro F1 기준. 반전 logits 평균 TTA, holdout 선택 threshold **{lock['threshold']}**, sweater·leg warmer·umbrella는 .5. holdout selection.json을 저장한 뒤 공식 val을 평가했다.

| ID | 임계값 | holdout micro / macro / image | val micro / macro / image | 기준 대비 CI | 예시 micro / 군중 재현율 | 테스트 유형별 예측 수 |
|---|---|---|---|---|---|---|
'''
vals={split:' / '.join(f"{metrics[split][k]:.4f}" for k in ('micro_f1','macro_f1','image_f1')) for split in metrics}
text+=f"| R4_bce_kd_s42 | {lock['threshold']} + 3개 .5 예외 | {vals['holdout']} | {vals['val']} | 해당 없음(기준 자체) | 자료 미확보 | 자료 미확보 |\n"
text+='\n![학습 곡선](curves.png)\n\n'
for split in ('holdout','val'):
 with (RUN/f'{split}_per_class.csv').open() as f:rs=list(csv.DictReader(f))
 text+=f'## {split} 클래스별 결과\n\n| category | support | precision | recall | F1 |\n|---|---|---|---|---|\n'
 for r in rs:text+='| '+r['category']+' | '+r['support']+' | '+' | '.join(f"{float(r[k]):.4f}" for k in ('precision','recall','f1'))+' |\n'
 kept=[float(r['f1']) for r in rs if r['category'] not in ('cape','tie','leg warmer','umbrella')]
 text+=f'\n희소 4개 제외 Macro F1: {sum(kept)/len(kept):.4f}\n\n'
text+='''교사 매칭·코드·분할·라이브러리 기록은 실험 준비 폴더 checks/preflight.json 및 source_manifest.json, 초기 가중치 hash는 initialization.json에 있다. 원본 데이터·교사·test 데이터는 Git에 올리지 않는다.

OOD 진단은 자료 미확보로 미완료다. val·테스트로 설정을 수정하거나 모델을 선택하지 않았다. 자기 자신과의 차이·bootstrap은 만들지 않았다. 향후 쌍 비교를 위해 tta/의 filenames·targets·categories·logits를 보존했다. 단일 seed이며 후속 차이의 신뢰구간이 0을 포함하면 “유의한 차이를 확인하지 못함”으로 표현한다.
'''
(RUN/'README.md').write_text(text)
print('Report:',RUN/'README.md')
