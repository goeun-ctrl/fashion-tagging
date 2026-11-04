from pathlib import Path
import csv,json,hashlib,shutil,datetime,sys
import numpy as np

BASE=Path(r'C:\Users\UserK\Desktop\딥러닝팀플자료')
SOURCE=BASE/'딥러닝_ConvNeXt_실험결과'
OUT=BASE/('ConvNeXt_약점분석_가중치백업_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(exist_ok=False)
snapshot=OUT/'local_snapshot'
shutil.copytree(SOURCE,snapshot,ignore=shutil.ignore_patterns('__pycache__','.git'))
r=snapshot/'실험결과'
def dump(name,data):
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def writecsv(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def group(n):return '정답 없음' if n==0 else '희소 1~9' if n<10 else '소수 10~49' if n<50 else '다수 50 이상'
def rank(a):
 a=np.asarray(a);return np.array([np.mean(np.flatnonzero(np.sort(a)==x)) for x in a])
allrows=[];data={};groups=[];pairs=[];verified={}
for split in ['holdout','val']:
 with np.load(r/f'preds_{split}.npz',allow_pickle=False) as z:
  names=z['filenames'].astype(str);cats=z['categories'].astype(str);y=z['targets'].astype(bool);logits=z['logits'];p=logits>=0
 assert y.shape==p.shape and y.shape[1]==27 and len(set(names))==len(names)
 tp=(y&p).sum(0);fp=(~y&p).sum(0);fn=(y&~p).sum(0);support=y.sum(0)
 precision=np.divide(tp,tp+fp,out=np.zeros(27),where=tp+fp>0);recall=np.divide(tp,support,out=np.zeros(27),where=support>0)
 f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(27),where=2*tp+fp+fn>0)
 with (r/f'per_class_{split}.csv').open(encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
 for i,c in enumerate(cats):
  assert old[i]['class_name']==c
  for k,v in [('support',support[i]),('tp',tp[i]),('fp',fp[i]),('fn',fn[i]),('precision',precision[i]),('recall',recall[i]),('f1',f1[i])]:assert abs(float(old[i][k])-float(v))<1e-10,(split,c,k)
  allrows.append(dict(split=split,class_index=i,class_name=c,n_images=len(y),support=int(support[i]),positive_rate=float(support[i]/len(y)),support_group=group(support[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),tp=int(tp[i]),fp=int(fp[i]),fn=int(fn[i])))
 image_denom=2*(y&p).sum(1)+(~y&p).sum(1)+(y&~p).sum(1)
 imagef=np.divide(2*(y&p).sum(1),image_denom,out=np.ones(len(y)),where=image_denom>0)
 verified[split]={'n_images':len(y),'macro_f1':float(f1.mean()),'micro_f1':float(2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum())),'image_f1':float(imagef.mean()),'macro_positive_classes_only':float(f1[support>0].mean()),'zero_support_classes':cats[support==0].tolist()}
 data[split]={'cats':cats,'support':support,'f1':f1,'n':len(y)}
 for g in ['정답 없음','희소 1~9','소수 10~49','다수 50 이상']:
  mask=np.array([group(n)==g for n in support]);groups.append(dict(split=split,group=g,n_classes=int(mask.sum()),total_positive_labels=int(support[mask].sum()),mean_f1=float(f1[mask].mean()) if mask.any() else '',median_f1=float(np.median(f1[mask])) if mask.any() else ''))
 for i in range(27):
  for j in range(27):
   if i!=j:
    count=int(((~y[:,i]&p[:,i])&(y[:,j]&~p[:,j])).sum())
    if count:pairs.append(dict(split=split,false_positive_class=cats[i],false_negative_class=cats[j],cooccurring_images=count))
 valid=support>0
 verified[split]['support_f1_spearman_descriptive']=float(np.corrcoef(rank(support[valid]),rank(f1[valid]))[0,1])
assert np.array_equal(data['holdout']['cats'],data['val']['cats'])
comparison=[]
for i,c in enumerate(data['val']['cats']):
 h,v=data['holdout'],data['val'];delta=float(v['f1'][i]-h['f1'][i])
 comparison.append(dict(class_name=c,holdout_support=int(h['support'][i]),val_support=int(v['support'][i]),holdout_prevalence=float(h['support'][i]/h['n']),val_prevalence=float(v['support'][i]/v['n']),holdout_f1=float(h['f1'][i]),val_f1=float(v['f1'][i]),f1_delta=delta,macro_delta_contribution=delta/27))
writecsv('全クラス指標.csv' if False else '클래스별_전체지표.csv',allrows)
writecsv('정답수_그룹비교.csv',groups)
writecsv('holdout_val_클래스차이.csv',sorted(comparison,key=lambda x:x['f1_delta']))
writecsv('FP_FN_동시발생.csv',sorted(pairs,key=lambda x:(x['split'],-x['cooccurring_images'])))
dump('예측_지표_재계산검증.json',verified)
(OUT/'class_order.json').write_text(json.dumps(data['val']['cats'].tolist(),ensure_ascii=False,indent=2),encoding='utf-8')
with (r/'history.csv').open(encoding='utf-8-sig') as f:history=list(csv.DictReader(f))
best=max(history,key=lambda x:float(x['holdout_macro_f1']));bestepoch=int(best['epoch']);assert bestepoch==10 and len(history)==15
summary=json.loads((r/'summary.json').read_text())
text=['# ConvNeXt-Tiny 클래스별 약점 및 과적합 분석','', '## 분석 근거와 제한','', '- 저장된 best epoch 10의 holdout/공식 val logits와 정답을 CPU에서 재계산했다. threshold 0.5(동일하게 logits≥0)를 유지했다. 기존 클래스별 CSV의 27개 클래스 모든 수치와 일치했다.','- 공식 val은 보고용으로만 사용했다. threshold/하이퍼파라미터 변경, 재추론, 추가 학습은 실행하지 않았다.','- 클래스별 support는 해당 클래스를 가진 이미지 수다. 다중 라벨이므로 합계는 이미지 수와 다르다. Macro F1은 정답 없는 클래스도 포함한 27개 평균이다.','- 표본 구간 0/1~9/10~49/50+는 기술적 비교를 위한 기준이며 통계적 유의성 기준은 아니다. 정답 수는 평가셋 표본 수이므로 학습셋의 희소성을 직접 증명하지 않는다.','', '## 전체 성능','', '|분할|이미지|Micro F1|Macro F1|Image F1|정답 있는 클래스만 Macro|','|---|---:|---:|---:|---:|---:|']
for s,d in verified.items():text.append(f"|{s}|{d['n_images']}|{d['micro_f1']:.6f}|{d['macro_f1']:.6f}|{d['image_f1']:.6f}|{d['macro_positive_classes_only']:.6f}|")
for split in ['holdout','val']:
 rows=[x for x in allrows if x['split']==split];text+=['',f'## {split} 성능 하위 8개','', '|클래스|정답 수|Precision|Recall|F1|TP|FP|FN|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for x in sorted(rows,key=lambda x:x['f1'])[:8]:text.append(f"|{x['class_name']}|{x['support']}|{x['precision']:.4f}|{x['recall']:.4f}|{x['f1']:.4f}|{x['tp']}|{x['fp']}|{x['fn']}|")
 text+=['',f"정답 없는 클래스: {', '.join(verified[split]['zero_support_classes']) or '없음'}. 정답 0인 클래스는 recall을 0으로 표시하는 공통 정의를 따르며 성능의 일반화 추정이 불가능하다."]
text+=['','## 표본 수와 낮은 성능의 관계','', '|분할|구간|클래스 수|정답 합계|평균 F1|','|---|---|---:|---:|---:|']
for x in groups:text.append(f"|{x['split']}|{x['group']}|{x['n_classes']}|{x['total_positive_labels']}|{x['mean_f1'] if x['mean_f1']=='' else format(x['mean_f1'],'.4f')}|")
text+=['',f"support와 F1의 기술적 Spearman 상관: holdout={verified['holdout']['support_f1_spearman_descriptive']:.3f}, val={verified['val']['support_f1_spearman_descriptive']:.3f}. 27개 클래스 수준의 관찰이며 인과관계가 아니다. 희소 클래스에서는 TP 1개 변화만으로 recall/F1이 크게 바뀐다. 다수 클래스의 낮은 F1은 단순 표본 부족으로 설명할 수 없으므로 FP/FN 패턴을 별도로 봐야 한다.",'','## 공식 val Macro F1 하락의 분해','',f"val−holdout Macro F1={verified['val']['macro_f1']-verified['holdout']['macro_f1']:.6f}. 각 클래스 ΔF1/27을 합하면 정확히 이 차이가 된다. 이미지 수 차이(2425 대 1158)만으로 성능 저하를 단정할 수는 없다.",'','|클래스|Holdout 정답|Val 정답|Holdout F1|Val F1|ΔF1|Macro 기여|','|---|---:|---:|---:|---:|---:|---:|']
for x in sorted(comparison,key=lambda x:x['f1_delta'])[:10]:text.append(f"|{x['class_name']}|{x['holdout_support']}|{x['val_support']}|{x['holdout_f1']:.4f}|{x['val_f1']:.4f}|{x['f1_delta']:.4f}|{x['macro_delta_contribution']:.5f}|")
text+=['','확인된 사실: 위 클래스별 지표 차이와 정답 수 차이가 Macro 차이를 구성한다. 클래스별 prevalence도 CSV에 함께 저장했다. 가설: 희소 클래스의 추정 변동, 이미지/포즈/의복 구성의 분포 차이, 주석 경계 차이가 영향을 줬을 수 있다. 학습 데이터의 클래스별 표본 및 독립적인 주석 검토 없이 이를 원인으로 확정할 수 없다.','', '## FP·FN 동시 발생 및 기존 이미지 사례','', 'FP 클래스와 FN 클래스가 같은 이미지에서 함께 발생한 횟수다. 다중 라벨이므로 범주형 confusion matrix나 일대일 오분류 수와 같지 않다.']
for split in ['holdout','val']:
 text+=['',f'### {split} 주요 동시 오류']
 for x in sorted([x for x in pairs if x['split']==split],key=lambda x:-x['cooccurring_images'])[:5]:text.append(f"- FP `{x['false_positive_class']}` + FN `{x['false_negative_class']}`: {x['cooccurring_images']}장")
 examples=json.loads((r/f'error_examples/{split}/examples.json').read_text())['examples']
 for rank_,ex in enumerate(examples[:4],1):text.append(f"- [기존 사례 {rank_}](local_snapshot/실험결과/error_examples/{split}/example_{rank_:02d}.jpg), 원본명 `{ex['file_name']}`: FP={ex['false_positive_labels']}, FN={ex['false_negative_labels']}. 이 링크의 이미지와 저장된 정답/예측을 연결한 기록이다.")
text+=['','시각 확인: 두 contact sheet의 총 24개 오류 이미지를 확인했다. holdout 앞 3개에는 여러 사람이 함께 등장하고, val 1·2번은 앉은 자세이며 val 3번은 상체 중심으로 잘려 있다. 이러한 구도는 작은 의복·가려진 부위 판단을 어렵게 할 수 있다는 가설을 제시하지만, 모델의 실제 원인이라고 입증하지는 않는다. `dress`와 `top/skirt/pants` 등의 경계가 외관상 애매한 사례도 있어, 저장된 주석을 시각 판단만으로 오류라고 단정하거나 수정하지 않았다. 기존 12개 시트는 최저 Image F1 순으로 고른 편향된 사례이므로 클래스 전체의 대표 표본은 아니다.','', '## 과적합과 best/early stopping 해석','', '|Epoch|Train BCE|Holdout BCE|Holdout Macro F1|','|---:|---:|---:|---:|']
for row in history:text.append(f"|{row['epoch']}|{float(row['train_loss']):.6f}|{float(row['holdout_loss']):.6f}|{float(row['holdout_macro_f1']):.6f}|")
text+=['','- Train BCE는 계속 감소하지만 holdout BCE는 초반 최저점 이후 상승한다. 이는 일부 틀린 예측의 과도한 확신/확률 보정 악화를 포함한 과적합 징후와 일치한다. 확정적인 원인 판별에는 추가 검사가 필요하다.','- BCE는 모든 이미지×클래스의 연속 확률과 확신의 정도를 평가한다. F1은 고정 threshold 0.5의 이산 판정과 TP/FP/FN을 평가한다. 따라서 소수의 고확신 오답이 BCE를 크게 올리는 동안 일부 클래스가 threshold를 올바르게 넘으면 F1은 개선될 수 있다. 다중 라벨·클래스 불균형에서는 두 지표의 가중 방식도 다르다.','- best epoch 10은 holdout Macro F1=0.707369가 최대였기 때문에 선택했다. holdout BCE 최저 epoch를 선택하는 실험이 아니었다. epoch 11~15에서 해당 Macro F1을 넘지 못한 5회 연속 미개선으로 epoch 15에서 early stopping했다. 공식 val은 이 선택에 사용하지 않았다.','', '## 후속 실험 후보 — 제안만, 실행하지 않음','', '- 내부 holdout/추가 내부 CV에서 weight decay·학습률·학습 길이를 비교하고 확률 보정(BCE, calibration)을 함께 기록한다.','- 학습셋 클래스별 support를 먼저 조사한 뒤 희소 클래스 손실 가중/샘플링을 별도 대조 실험으로 검토한다.','- 혼동이 큰 의복 경계에 대한 주석 점검과 ROI/해상도 대조를 제안한다. 현재 실험의 증강·threshold·정규화는 변경하지 않는다.','- 클래스별 threshold나 calibration을 검토하더라도 내부 holdout으로만 선택하고, 공식 val은 잠근 최종 평가로 유지한다.','', '## 비용 단위와 재현성','',f"- 파라미터 {summary['params_total']:,}; 누적 학습 {summary['train_wall_seconds']/60:.2f}분.",f"- 최대 allocated: {summary['train_peak_allocated_bytes']:,} bytes = {summary['train_peak_allocated_bytes']/1e9:.6f} GB = {summary['train_peak_allocated_bytes']/2**30:.6f} GiB.",f"- 최대 reserved: {summary['train_peak_reserved_bytes']:,} bytes = {summary['train_peak_reserved_bytes']/1e9:.6f} GB = {summary['train_peak_reserved_bytes']/2**30:.6f} GiB.",'- GB=10^9 bytes, GiB=2^30 bytes. 기존 팀 CSV max_vram_gb는 allocated/10^9이며 reserved 또는 nvidia-smi 전체 점유와 혼동하지 않는다.','- CUBLAS_WORKSPACE_CONFIG 미설정 결정성 경고가 기록됐으므로 비트 단위 재현은 보장하지 않는다.','- 원본 이미지 데이터는 이번 백업에 포함하지 않고, 기존 오류 예시만 보존한다.']
(OUT/'클래스별_약점_과적합_분석.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
dump('분석_입력파일_SHA256.json',{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()})
Path(__file__).with_name('current_backup_path.txt').write_text(str(OUT),encoding='utf-8')
print(json.dumps({'output':str(OUT),'metrics':verified,'lowest_val':sorted([x for x in allrows if x['split']=='val'],key=lambda x:x['f1'])[:8],'macro_drops':sorted(comparison,key=lambda x:x['f1_delta'])[:5]},ensure_ascii=True))
