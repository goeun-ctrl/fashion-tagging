"""Split-by-split team inference with a holdout selection lock before official val."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'source/scripts'),str(BASE/'source/src')]
import train_hires as th
SPECIAL=('sweater','leg warmer','umbrella')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thresholds(cats,t):return np.array([.5 if c in SPECIAL else t for c in cats])
def select(y,logits,cats):
    scores=[]
    for t in np.round(np.arange(.05,.951,.05),2):
        m=th.compute_metrics(y,th.sigmoid(logits)>=thresholds(cats,t));scores.append((m['macro_f1'],m['micro_f1'],float(t)))
    return max(scores)[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--split',choices=['holdout','val'],required=True);a=p.parse_args()
    torch.set_num_threads(4)
    run=a.run;lockfile=run/'selection.json';ckpt=run/'best.pt'
    if a.split=='val':
        lock=json.loads(lockfile.read_text());assert lock['checkpoint_sha256']==sha(ckpt);assert lock['config_sha256']==sha(run/'config.json')
        assert lock['holdout_predictions_sha256']==sha(run/'tta/preds_holdout_s800_flip.npz')
    output=run/'tta'/f'preds_{a.split}_s800_flip.npz';assert not output.exists(),output
    st=torch.load(ckpt,map_location='cpu',weights_only=False);cfg=st['config'];cats=st['categories']
    root=cfg['train_dir'] if a.split=='holdout' else cfg['val_dir']
    dc,names,y=th.load_split(root,'train' if a.split=='holdout' else 'val');assert dc==cats
    if a.split=='holdout':
        idx={n:i for i,n in enumerate(names)};names=th.read_names(Path(cfg['splits_dir'])/'holdout.txt');y=y[[idx[n] for n in names]]
    assert len(names)==(2425 if a.split=='holdout' else 1158)
    model=th.build_hires_model(cfg,len(cats),False).cuda().to(memory_format=torch.channels_last).eval();model.load_state_dict(st['model'],strict=True)
    mean,std=th.timm_mean_std(cfg['model']);por=th.oriented_portrait(root,names);lg=[]
    for mirror in (False,True):
        ds=th.HiResDataset(root,names,y,mean,std,False,1333,0.,mirror=mirror)
        dl=DataLoader(ds,batch_sampler=th.BucketBatchSampler(por,[800],8,0,False),num_workers=6,pin_memory=True,collate_fn=th.pad_collate)
        logits,_=th.predict(model,dl,torch.device('cuda'),True);lg.append(logits)
    logits=(lg[0]+lg[1])/2;assert logits.shape==y.shape and np.isfinite(logits).all()
    output.parent.mkdir(exist_ok=True);th.save_predictions(str(output),names,y,logits,cats)
    if a.split=='holdout':
        t=select(y,logits,cats)
        lock=dict(experiment_id=cfg['exp_id'],team_baseline_id='R5_bce_kd_s42',threshold=t,special={c:.5 for c in SPECIAL},
                  best_epoch=st['epoch'],checkpoint_sha256=sha(ckpt),config_sha256=sha(run/'config.json'),
                  holdout_predictions_sha256=sha(output),selection='holdout TTA Macro, tie Micro then higher threshold; .05..95/.05')
        with lockfile.open('x') as f:json.dump(lock,f,indent=2)
    else:t=lock['threshold']
    pred=th.sigmoid(logits)>=thresholds(cats,t);m=th.compute_metrics(y,pred)
    (run/f'{a.split}_metrics.json').write_text(json.dumps(dict(split=a.split,threshold=t,best_epoch=st['epoch'],metrics=m),indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else float(x)))
    tp=(y.astype(bool)&pred).sum(0);fp=(~y.astype(bool)&pred).sum(0);fn=(y.astype(bool)&~pred).sum(0)
    with (run/f'{a.split}_per_class.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['category','support','precision','recall','f1'])
        for i,c in enumerate(cats):w.writerow([c,int(y[:,i].sum()),float(tp[i]/max(1,tp[i]+fp[i])),float(tp[i]/max(1,tp[i]+fn[i])),float(2*tp[i]/max(1,2*tp[i]+fp[i]+fn[i]))])
    print(json.dumps({k:v for k,v in m.items() if isinstance(v,(float,int))}))
if __name__=='__main__':main()
