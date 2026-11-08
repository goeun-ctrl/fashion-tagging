"""Read-only data/source audit and optional strict teacher validation for R4_bce_kd_s42."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
SOURCE=BASE/'source'
sys.path[:0]=[str(SOURCE/'scripts'),str(SOURCE/'src')]
import train_hires as th


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def audit(teacher=None):
    argv=['--exp_id','R4_bce_kd_s42','--head','linear_ms','--pool','gapgmp','--loss','bce',
          '--train_dir','/path/to/local-data/data/train',
          '--val_dir','/path/to/local-data/data/val',
          '--splits_dir',str(ROOT/'data/splits'),'--out_dir',str(ROOT/'runs/role4_bce_kd'),
          '--distill_alpha','0.5','--distill_shift','0.3']
    if teacher:argv+=['--distill_npz',str(Path(teacher).resolve())]
    cfg=th.parse_args(argv)
    for k in ('mosaic_p','gray_p','old_p','zoom_p'):assert getattr(cfg,k)==0
    for k in ('neg_list','pos_list','ext_list','ext_labels','mask_dir','init_ckpt'):assert getattr(cfg,k) is None
    assert not cfg.gate and not cfg.no_pretrained
    cats,names,y=th.load_split(cfg.train_dir,'train')
    vc,vnames,vy=th.load_split(cfg.val_dir,'val')
    inner,hold=[th.read_names(Path(cfg.splits_dir)/(s+'.txt')) for s in ('train_inner','holdout')]
    assert cats==vc and len(cats)==27
    assert (len(names),len(inner),len(hold),len(vnames))==(45622,43197,2425,1158)
    assert all(len(x)==len(set(x)) for x in (names,inner,hold,vnames))
    assert not set(inner)&set(hold) and set(inner)|set(hold)==set(names)
    assert not set(names)&set(vnames)
    assert np.isin(y,[0,1]).all() and np.isin(vy,[0,1]).all()
    missing=[str(Path(d)/'images'/n) for d,ns in [(cfg.train_dir,names),(cfg.val_dir,vnames)] for n in ns if not (Path(d)/'images'/n).is_file()]
    assert not missing,missing[:5]
    manifest=json.loads((BASE/'source_manifest.json').read_text())
    for f in manifest['files']:assert digest(SOURCE/f['path'])==f['sha256'],f['path']
    files=[Path(cfg.splits_dir)/(s+'.txt') for s in ('train_inner','holdout')]
    files += [Path(d)/f for d in (cfg.train_dir,cfg.val_dir) for f in ('categories.txt','labels.csv')]
    blockers=[]
    teacher_info={'path':str(Path(teacher).resolve()) if teacher else None,'verified':False}
    if not teacher:
        blockers.append('ff_r50_train.npz actual path unavailable; class order, exact image matching, provenance and hash not verified')
    else:
        path=Path(teacher);assert path.is_file(),path
        with np.load(path,allow_pickle=False) as z:
            assert {'filenames','scores','categories'}<=set(z.files),'teacher must include explicit categories'
            tc=[str(c) for c in z['categories'].tolist()]; tn=[str(n) for n in z['filenames'].tolist()];scores=z['scores']
            assert tc==cats,'teacher category order mismatch'
            assert len(tn)==len(set(tn)),'duplicate teacher image keys'
            assert scores.shape==(len(tn),27) and np.isfinite(scores).all()
            assert ((scores>=0)&(scores<=1)).all()
            assert set(inner)<=set(tn),'missing train_inner teacher rows'
            assert not set(tn)&set(vnames),'official val rows in teacher artifact'
            assert set(tn)<=set(names),'non-original-training image in teacher artifact'
            selected=th.load_teacher(cfg,inner)
            assert selected.shape==(43197,27) and np.isfinite(selected).all()
            teacher_info.update(verified=True,sha256=digest(path),rows=len(tn),train_inner_matched=len(inner),
                                holdout_rows_present=len(set(tn)&set(hold)),holdout_rows_consumed_by_training=0,
                                provenance='Teacher training provenance must be supplied separately; array matching alone cannot establish teacher training-data provenance.')
    blockers.append('test_public/examples_labeled and OOD image type tags not found on this server')
    record=dict(experiment_id=cfg.exp_id,team_baseline_id='R5_bce_kd_s42',source_commit=manifest['commit'],
                status='not_started_missing_required_inputs' if not teacher else 'teacher_validated_not_launched',
                config=vars(cfg),argv=argv,categories=cats,counts=dict(train_inner=len(inner),holdout=len(hold),val=len(vnames)),
                split_and_label_sha256={str(p):digest(p) for p in files},teacher=teacher_info,blockers=blockers,
                environment=dict(python=platform.python_version(),libraries={n:importlib.metadata.version(n) for n in ('torch','torchvision','timm','numpy','Pillow')}),
                estimated_gpu_hours=[9,10],gpu_assigned=None,
                evaluation=dict(selection='EMA holdout macro F1',tta='sigmoid(mean(original_logits,flip_logits))',
                                threshold_grid='0.05..0.95 step 0.05',special={'sweater':0.5,'leg warmer':0.5,'umbrella':0.5},
                                lock_before_val=True,self_comparison=False,bootstrap_for_future_pairs=2000))
    (BASE/'checks/preflight.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps({k:record[k] for k in ('experiment_id','source_commit','status','counts','blockers')},ensure_ascii=False))
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path);a=p.parse_args();audit(a.teacher)
