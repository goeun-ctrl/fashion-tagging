import pathlib,json,sys,time,os,datetime
base=pathlib.Path('/home/frostpine/fashion_tagging')
w=base/'experiments/role2_jeongeun_convnext';r=w/'runs/S2_jeongeun_cnxt_t_bce_384_pad'
while True:
 s=json.loads((r/'status.json').read_text())
 if s.get('state')=='completed':break
 if s.get('state')=='failed' and s.get('pid')==1083741:
  (w/'prep/finalization_failure.json').write_text(json.dumps(s,ensure_ascii=False,indent=2));raise SystemExit(1)
 try:os.kill(1083741,0)
 except ProcessLookupError:
  if s.get('state')!='completed':
   (w/'prep/finalization_failure.json').write_text(json.dumps({'error':'training process ended before completed status','status':s},ensure_ascii=False,indent=2));raise SystemExit(1)
 time.sleep(20)
import numpy as np
sys.path.insert(0,str(base/'reference/fashion-tagging-c5827dfa/common'))
from metrics import compute_metrics,binarize,sigmoid
summary=json.loads((r/'summary.json').read_text());checks={}
for split,count in [('holdout',2425),('val',1158)]:
 with np.load(r/f'preds_{split}.npz',allow_pickle=False) as p:
  assert p['targets'].shape==(count,27)
  assert p['logits'].shape==(count,27)
  assert len(p['categories'])==27
  actual=compute_metrics(p['targets'],binarize(sigmoid(p['logits']),.5))
  checks[split]={k:float(actual[k]) for k in ['micro_f1','macro_f1','image_f1']}
  for k,value in checks[split].items():assert abs(value-summary[split][k])<1e-10
required=['best.pt','last.pt','history.csv','history.json','learning_curves.png','learning_curves.svg','resolved_config.json','library_versions.json','preds_holdout.npz','preds_val.npz']
for f in required:assert (r/f).is_file(),f
report={'verified':True,'verified_at':datetime.datetime.now().astimezone().isoformat(),'metrics_recomputed':checks,'files':{p.name:p.stat().st_size for p in r.iterdir() if p.is_file()}}
(r/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
lines=['# ConvNeXt-Tiny 기준 실험 결과','',f"- 실험: {summary['exp_id']}",f"- 원본 SHA: {summary['source_commit']}",f"- 완료 epoch: {summary['epochs_run']}; best epoch: {summary['best_epoch']}",f"- 파라미터: {summary['params_total']:,}",'- GPU 0에서 9 epoch 체크포인트로부터 10 epoch 재개. 이전 GPU 0/3 기록 보존.','- batch 16 × accumulation 2 = effective batch 32.','- checkpoint 선택·early stopping은 내부 holdout Macro F1만 사용. 공식 val은 최종 평가에만 사용.','- smoke는 별도 폴더이며 기준 성능에 포함하지 않음.','', '| 분할 | Micro F1 | Macro F1 | Image-wise F1 |','|---|---:|---:|---:|']
for split,m in checks.items():lines.append(f"| {split} | {m['micro_f1']:.6f} | {m['macro_f1']:.6f} | {m['image_f1']:.6f} |")
lines+=['',f"- 누적 학습 시간: {summary['train_wall_seconds']:.1f} 초",f"- 학습 최대 allocated VRAM: {summary['train_peak_allocated_bytes']/2**30:.3f} GiB",f"- 학습 최대 reserved VRAM: {summary['train_peak_reserved_bytes']/2**30:.3f} GiB",'- GPU 이동 및 데이터 다운로드 대기 시간은 순수 학습 시간과 다름.','- 실행 설정: resolved_config.json; 라이브러리 버전: library_versions.json 및 ../../prep/requirements.freeze.txt.','- 실제 재개 명령은 resolved_config.json의 argv 및 ../../prep/migrate_gpu0.sh에 기록.','- 결정성 경고: 실행 중 CUBLAS_WORKSPACE_CONFIG 미설정 경고가 있었으므로 비트 단위 재현성을 보장하지 않음.','- Git add/commit/push 및 다른 계정 프로세스 종료는 수행하지 않음.','- best.pt/last.pt는 optimizer·scheduler·scaler·RNG를 포함한 체크포인트이며 모델 파라미터는 model 키에 저장.','- 클래스별 지표·오류 사례·팀 CSV는 이 결과 폴더의 해당 CSV 및 error_examples 하위 폴더 참조.']
(r/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=True),flush=True)
