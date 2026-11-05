"""Build a new versioned EVA-compatible package without editing preparation v1."""
from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,json,shutil,sys
import numpy as np
from PIL import __version__ as pillow_version

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_module(path):
    s=importlib.util.spec_from_file_location('original_augmentation',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def build(old,audit,baseline_audit,out,approved):
    assert not out.exists(),'Choose a new output directory; existing packages are immutable'
    original=json.loads(gzip.decompress(audit.read_bytes()))
    source=list(csv.DictReader((old/'training_manifest.csv').open(encoding='utf-8-sig',newline='')))
    with np.load(old/'training_targets.npz',allow_pickle=False) as z:
        y=z['targets'].astype(np.int8);cats=z['categories'].tolist()
    assert cats==original['categories'] and y.shape==(43703,27)
    assert np.isin(y,[-1,0,1]).all() and (y==1).any(1).all()
    assert [r['file_name'] for r in source[:43197]]==original['train_inner']
    assert all(r['dataset']=='original' for r in source[:43197]) and all(r['dataset']=='external' for r in source[43197:])
    original_targets=dict(zip(original['train_names'],original['train_targets']))
    assert np.array_equal(y[:43197],[original_targets[n] for n in original['train_inner']])
    assert len(set(r['file_name'] for r in source))==len(source)
    assert not set(r['file_name'] for r in source)&(set(original['holdout'])|set(original['val']))
    module=load_module(old/'augmentation.py')
    version='fashion-common-506-partial-20261105-v1'
    out.mkdir(parents=True);(out/'epoch_plans').mkdir()
    shutil.copytree(old/'selected_external',out/'selected_external')
    shutil.copytree(old/'fixed_splits',out/'fixed_splits')
    counts=(y==1).sum(0);count_map=dict(zip(cats,counts.tolist()))
    weights=np.array([module.sampling_weight([c for c,v in zip(cats,row) if v==1],count_map) for row in y])
    focus=sorted(module.HIGH|{c for c,n in count_map.items() if n<1000})
    general=sorted(module.GENERAL-set(focus))
    jitter=np.array([max([.35 if c in focus else .2 if c in general else 0 for c,v in zip(cats,row) if v==1],default=0) for row in y])
    ids=[r['file_name'] for r in source]
    rows=[dict(image_id=r['file_name'],source='train_inner' if r['dataset']=='original' else 'external',relative_path=r['file_name'],approved=1,image_sha256=r['sha256'],sampling_weight=float(w),data_version=version) for r,w in zip(source,weights)]
    csvwrite(out/'training_manifest.csv',rows)
    np.savez_compressed(out/'training_targets.npz',image_ids=np.array(ids),categories=np.array(cats),targets=y,data_version=version)
    csvwrite(out/'class_counts.csv',[{'class':c,'positive':int(n)} for c,n in zip(cats,counts)])
    hashes={n:sha(out/n) for n in ['training_manifest.csv','training_targets.npz','class_counts.csv']}
    policy=dict(data_version=version,targets_sha256=hashes['training_targets.npz'],replay='pillow-enhance-v1',pillow_version=pillow_version,flip_p=.5,hue=0,ranges=[[.9,1.1],[.9,1.1],[.95,1.05]],jitter_p=[.35,.2,0.],focus=focus,general=general,medium_weight=sorted(module.MID_WEIGHT))
    dump(out/'augmentation_policy.json',policy);hashes['augmentation_policy.json']=sha(out/'augmentation_policy.json')
    exposures=[];equivalent=True
    for epoch in range(1,31):
        plan=module.make_epoch_plan(weights,jitter,epoch)
        with np.load(old/f'epoch_plans/epoch_{epoch:02d}.npz',allow_pickle=False) as z:
            equivalent &= all(np.array_equal(plan[k],z[k]) for k in plan)
        name=f'epoch_plans/epoch_{epoch:02d}.npz'
        np.savez_compressed(out/name,indices=plan['image_index'],flip=plan['flip'],factors=plan['factors'],order=plan['order'],epoch=epoch,data_version=version,label_mode='partial',manifest_sha256=hashes['training_manifest.csv'],targets_sha256=hashes['training_targets.npz'],policy_sha256=hashes['augmentation_policy.json'])
        hashes[name]=sha(out/name)
        unique,ns=np.unique(plan['image_index'],return_counts=True)
        exposures.append(dict(epoch=epoch,original_draws=int((plan['image_index']<43197).sum()),external_draws=int((plan['image_index']>=43197).sum()),unique_seen=len(unique),**{c:int((y[plan['image_index']]==1).sum(0)[i]) for i,c in enumerate(cats)}))
    report=json.loads(baseline_audit.read_text(encoding='utf-8-sig'))
    for split in ['train','val']:
        hashes[split+'/labels.npz']=report['partitions'][split]['metadata_sha256']['labels.npz']
    lock=dict(schema='fashion-common-v1',approved=approved,data_version=version,label_mode='partial',categories=cats,seed=42,replacement=True,epochs=30,draws=43197,sha256=hashes)
    dump(out/'shared-lock.json',lock)
    csvwrite(out/'planned_class_exposure.csv',exposures)
    dump(out/'label_review_status.json',dict(label_mode='partial',approved_partial_mode=approved,human_complete_annotation=False,external_unknown_entries=int((y[43197:]==-1).sum()),external_total_entries=int(y[43197:].size),unknown_fraction=float((y[43197:]==-1).mean()),annotation='Preserve -1; do not turn unknowns into negatives. User explicitly approved B mode on 2026-11-05. This does not assert full human annotation or a legal clearance.'))
    dump(out/'provenance.json',dict(data_version=version,source_prepared_hashes=sha(old/'prepared_file_sha256.json'),source_image_audit_sha256=sha(audit),baseline_audit_sha256=sha(baseline_audit),original_common_commit='c5827dfa749e276d4e5431628a06730b26c4841f',eva_branch='feature/eva02-common-data-retrain',eva_commit='a84528284cc8f8d2ae6177caeee67fee05b92af4',original_count=43197,external_count=506,holdout_count=2425,val_count=1158,regenerated_plans_equal_previous_values=bool(equivalent),plans_regenerated=True,training_executed=False,original_images_included=False,normalization_is_model_specific=True,pillow_version=pillow_version))
    shutil.copy2(old/'augmentation.py',out/'original_augmentation.py')
    shutil.copy2(old/'visual_screening.csv',out/'visual_screening.csv')
    dump(out/'package-files.sha256.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({'output':str(out),'images':506,'rows':len(rows),'plans':30,'approved':approved,'old_plan_values_preserved':bool(equivalent)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--image-audit',type=Path,required=True);p.add_argument('--baseline-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--approve-partial',action='store_true');a=p.parse_args();build(a.source,a.image_audit,a.baseline_audit,a.out,a.approve_partial)
