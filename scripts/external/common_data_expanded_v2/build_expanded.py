from pathlib import Path
import csv,gzip,hashlib,importlib.util,json,shutil,platform
from collections import Counter
import numpy as np
from PIL import Image,__version__ as pillow_version

W=Path(__file__).resolve().parent;BASE=W.parent
OLD=BASE/'common_data_20261105/fashion-common-506-partial-20261105-v1'
SRC=BASE/'external_augmentation_20261104'
OUT=W/'fashion-common-expanded-partial-20261106-v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def write(p,rr):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 assert not OUT.exists()
 eligible=read(SRC/'eligible_external.json');byname={r['file_name']:r for r in eligible}
 oldlabels=rows(OLD/'selected_external/labels.csv');oldnames={r['file_name'] for r in oldlabels}
 prior={r['file_name'] for r in read(SRC/'review_candidates.json')};excluded=prior-oldnames
 candidates=read(W/'new_review_candidates.json');dec=read(W/'review_decisions.json')['pages'];review=[];accepted=[]
 for r in candidates:
  rid=r['review_id'];n=int(rid[1:]);page=rid[0]+f'_{(n-1)//100+1:02}'
  if page not in dec:state='not_reviewed'
  else:state='pass' if ((n in dec[page]['ids'])==(dec[page]['mode']=='keep')) else 'excluded_visual'
  review.append(dict(file_name=r['file_name'],review_id=rid,primary_target=r['primary_target'],decision=state,reason='Target visibility/domain screening; not exhaustive 27-label annotation' if state!='not_reviewed' else 'Not inspected; never included'))
  if state=='pass':accepted.append(r)
 caps={'umbrella':1700,'glove':1700,'vest':600};selected=[byname[n] for n in sorted(oldnames)];counts=Counter(c for r in selected for c in r['positive_classes']);cap_skipped=[]
 for target in ['vest','umbrella','glove']:
  for r in accepted:
   if r['primary_target']!=target:continue
   if len(selected)>=4000 or any(counts[c]>=limit and c in r['positive_classes'] for c,limit in caps.items()):cap_skipped.append(r['file_name']);continue
   selected.append(r);counts.update(r['positive_classes'])
 selected.sort(key=lambda r:r['file_name']);names=[r['file_name'] for r in selected];assert len(set(names))==len(names)
 assert oldnames<=set(names) and not set(names)&excluded
 assert all(counts[c]<=v for c,v in caps.items())
 # All eligible rare-class candidates were included in the prior review: none silently bypass review.
 rare={'leg warmer','cape','jumpsuit'}
 assert not [r for r in eligible if rare.intersection(r['positive_classes']) and r['file_name'] not in prior]
 original=json.loads(gzip.decompress((SRC/'original_image_audit.json.gz').read_bytes()))
 with np.load(OLD/'training_targets.npz',allow_pickle=False) as z:cats=z['categories'].tolist();original_y=z['targets'][:43197].copy()
 rawlabels={r['file_name']:r for r in rows(SRC/'data/external/external_merged/labels.csv')}
 for r in oldlabels:assert r==rawlabels[r['file_name']]
 ext_y=np.array([[int(rawlabels[n][c]) for c in cats] for n in names],dtype=np.int8)
 y=np.concatenate([original_y,ext_y]);assert np.isin(y,[-1,0,1]).all()
 assert not set(names)&(set(original['holdout'])|set(original['val']))
 allsha={r['sha256'] for r in original['hashes']};assert len(allsha)==46780
 OUT.mkdir();(OUT/'epoch_plans').mkdir();(OUT/'selected_external/images').mkdir(parents=True)
 for r in selected:
  p=SRC/'data/external/external_merged/images'/r['file_name'];h=sha(p)
  assert h==r['sha256'] and h not in allsha;allsha.add(h)
  with Image.open(p) as im:im.load()
  shutil.copy2(p,OUT/'selected_external/images'/p.name)
 write(OUT/'selected_external/labels.csv',[rawlabels[n] for n in names])
 source_rows={r['file_name']:r for r in rows(SRC/'data/external/external_merged/sources.csv')}
 write(OUT/'selected_external/sources.csv',[source_rows[n] for n in names])
 shutil.copy2(SRC/'data/external/external_merged/categories.txt',OUT/'selected_external/categories.txt')
 shutil.copytree(OLD/'fixed_splits',OUT/'fixed_splits')
 spec=importlib.util.spec_from_file_location('aug',SRC/'augmentation.py');aug=importlib.util.module_from_spec(spec);spec.loader.exec_module(aug)
 version=OUT.name;totalcounts=(y==1).sum(0);cm=dict(zip(cats,totalcounts.tolist()))
 weights=np.array([aug.sampling_weight([c for c,v in zip(cats,row) if v==1],cm) for row in y])
 focus=sorted(aug.HIGH|{c for c,n in cm.items() if n<1000});general=sorted(aug.GENERAL-set(focus))
 jitter=np.array([max([.35 if c in focus else .2 if c in general else 0. for c,v in zip(cats,row) if v==1],default=0.) for row in y])
 oldmanifest=rows(OLD/'training_manifest.csv')[:43197];ids=[r['image_id'] for r in oldmanifest]+names
 hashes0=[r['image_sha256'] for r in oldmanifest]+[r['sha256'] for r in selected]
 manifest=[dict(image_id=n,source='train_inner' if i<43197 else 'external',relative_path=n,approved=1,image_sha256=h,sampling_weight=float(weights[i]),data_version=version) for i,(n,h) in enumerate(zip(ids,hashes0))]
 write(OUT/'training_manifest.csv',manifest)
 np.savez_compressed(OUT/'training_targets.npz',image_ids=np.array(ids),categories=np.array(cats),targets=y,data_version=version)
 write(OUT/'class_counts.csv',[dict(class_=c,positive=int(n)) for c,n in zip(cats,totalcounts)])
 # EVA's schema requires the column name "class".
 write(OUT/'class_counts.csv',[{'class':c,'positive':int(n)} for c,n in zip(cats,totalcounts)])
 hashes={n:sha(OUT/n) for n in ['training_manifest.csv','training_targets.npz','class_counts.csv']}
 policy=dict(data_version=version,targets_sha256=hashes['training_targets.npz'],replay='pillow-enhance-v1',pillow_version=pillow_version,flip_p=.5,hue=0,ranges=[[.9,1.1],[.9,1.1],[.95,1.05]],jitter_p=[.35,.2,0.],focus=focus,general=general,medium_weight=sorted(aug.MID_WEIGHT))
 dump(OUT/'augmentation_policy.json',policy);hashes['augmentation_policy.json']=sha(OUT/'augmentation_policy.json');exposure=[]
 for e in range(1,31):
  p=aug.make_epoch_plan(weights,jitter,e);name=f'epoch_plans/epoch_{e:02}.npz'
  np.savez_compressed(OUT/name,indices=p['image_index'],flip=p['flip'],factors=p['factors'],order=p['order'],epoch=e,data_version=version,label_mode='partial',manifest_sha256=hashes['training_manifest.csv'],targets_sha256=hashes['training_targets.npz'],policy_sha256=hashes['augmentation_policy.json'])
  hashes[name]=sha(OUT/name);ii=p['image_index'];cc=(y[ii]==1).sum(0)
  exposure.append(dict(epoch=e,original_draws=int((ii<43197).sum()),external_draws=int((ii>=43197).sum()),unique_seen=len(np.unique(ii)),**dict(zip(cats,cc.tolist()))))
 oldlock=read(OLD/'shared-lock.json')
 for split in ['train','val']:hashes[split+'/labels.npz']=oldlock['sha256'][split+'/labels.npz']
 dump(OUT/'shared-lock.json',dict(schema='fashion-common-v1',approved=True,data_version=version,label_mode='partial',categories=cats,seed=42,replacement=True,epochs=30,draws=43197,sha256=hashes))
 write(OUT/'planned_class_exposure.csv',exposure)
 with np.load(OLD/'training_targets.npz',allow_pickle=False) as z:oldext=z['targets'][43197:]
 write(OUT/'class_counts_phase1_phase2.csv',[{'class':c,'original_train_inner':int((original_y[:,i]==1).sum()),'phase1_external_positive':int((oldext[:,i]==1).sum()),'phase2_external_positive':int((ext_y[:,i]==1).sum()),'phase2_known_negative':int((ext_y[:,i]==0).sum()),'phase2_unknown':int((ext_y[:,i]==-1).sum()),'phase2_total_positive':cm[c]} for i,c in enumerate(cats)])
 write(OUT/'new_candidate_visual_screening.csv',review)
 shutil.copy2(OLD/'visual_screening.csv',OUT/'phase1_visual_screening.csv')
 shutil.copy2(W/'review_decisions.json',OUT/'visual_page_decisions.json')
 shutil.copy2(BASE/'controlled_external_effect_20261106/4000장_확장_상한정책_v1.json',OUT/'selection_cap_policy.json')
 shutil.copy2(SRC/'augmentation.py',OUT/'original_augmentation.py')
 shutil.copy2(Path(__file__),OUT/'build_expanded.py')
 report=dict(data_version=version,target_unique=4000,actual_unique=len(names),previous506_retained=len(oldnames),added_unique=len(names)-506,external_positive_counts=dict(counts),caps=caps,cap_skipped=cap_skipped,screened_new=sum(r['decision']!='not_reviewed' for r in review),new_visual_pass=len(accepted),new_visual_rejected=sum(r['decision']=='excluded_visual' for r in review),new_not_reviewed=sum(r['decision']=='not_reviewed' for r in review),prior_rejected_preserved=len(excluded),original_count=43197,holdout_count=2425,val_count=1158,unknown_fraction=float((ext_y==-1).mean()),all_selected_sha256_and_decode_checked=True,original_duplicate_screen='Reused cached SHA256+dHash<=4 audit across 46780 train/holdout/val images; exact SHA rechecked locally',human_complete_annotation=False,training_executed=False,gpu_used=False,git_changed=False,limitations='Codex contact-sheet target/domain screening only, not full human 27-class relabeling; dHash does not guarantee absence of all semantic near-duplicates; cached original audit, no fresh server image scan; source licenses/attribution preserved without rights-clearance assertion',eva_commit='a84528284cc8f8d2ae6177caeee67fee05b92af4',source_audit_sha256=sha(SRC/'original_image_audit.json.gz'),source_eligible_sha256=sha(SRC/'eligible_external.json'))
 dump(OUT/'selection_report.json',report)
 dump(OUT/'label_review_status.json',dict(label_mode='partial',human_complete_annotation=False,unknown_entries=int((ext_y==-1).sum()),unknown_fraction=report['unknown_fraction'],loss='Masked BCE over known entries; preserve -1'))
 dump(OUT/'generation_environment.json',dict(python=platform.python_version(),numpy=np.__version__,Pillow=pillow_version,device='CPU',gpu_used=False))
 dump(OUT/'package-files.sha256.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
 print(json.dumps(report,ensure_ascii=True))
if __name__=='__main__':main()
