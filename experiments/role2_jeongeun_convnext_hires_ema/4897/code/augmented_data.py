from pathlib import Path
import csv,hashlib,json
import numpy as np
from PIL import Image
from original_augmentation import tensor_array

LOCK_SHA='4c3405463dc9ffecb79e43539bd802bc50adb2ab57f331d80e3ee9f74e7e9236'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_inputs(p):
    assert sha(p/'shared-lock.json')==LOCK_SHA,'Different approved data lock'
    lock=json.loads((p/'shared-lock.json').read_text(encoding='utf-8'))
    assert lock['schema']=='fashion-common-v1' and lock['approved'] is True and lock['label_mode']=='partial'
    rows=list(csv.DictReader((p/'training_manifest.csv').open(encoding='utf-8-sig',newline='')))
    with np.load(p/'training_targets.npz',allow_pickle=False) as z:y=z['targets'];cats=z['categories'].tolist();ids=z['image_ids'].tolist()
    assert ids==[r['image_id'] for r in rows] and cats==lock['categories']
    return lock,rows,y,cats

def validate_prepared(p,categories,inner,all_names,all_targets):
    lock,rows,y,cats=read_inputs(p)
    assert (lock['seed'],lock['draws'],lock['epochs'],lock['replacement'])==(42,43197,30,True)
    for n,h in lock['sha256'].items():
        if n not in ['train/labels.npz','val/labels.npz']:assert sha(p/n)==h,n
    assert cats==list(categories) and y.shape==(48094,27) and np.isin(y,[-1,0,1]).all()
    assert all(r['approved']=='1' and r['data_version']==lock['data_version'] for r in rows)
    assert [r['image_id'] for r in rows if r['source']=='train_inner']==list(inner)
    lookup=dict(zip(all_names,all_targets))
    np.testing.assert_array_equal(y[:43197],[lookup[n] for n in inner])
    policy=json.loads((p/'augmentation_policy.json').read_text(encoding='utf-8'))
    return dict(combined_unique_images=48094,adopted_external_images=4897,policy=policy,fingerprint=LOCK_SHA,shared=lock)

class PlannedDataset:
    def __init__(self,p,original_root,epoch,cfg,mean,std,smoke_samples):
        self.lock,self.rows,self.all_targets,self.categories=read_inputs(p)
        self.roots={'train_inner':original_root/'images','external':p/'selected_external/images'}
        self.size=cfg['input_size'];self.mean=mean;self.std=std
        name=f'epoch_plans/epoch_{epoch:02d}.npz'
        assert sha(p/name)==self.lock['sha256'][name]
        with np.load(p/name,allow_pickle=False) as z:
            assert z['epoch'].item()==epoch and z['label_mode'].item()=='partial'
            for key,file in [('manifest_sha256','training_manifest.csv'),('targets_sha256','training_targets.npz'),('policy_sha256','augmentation_policy.json')]:assert z[key].item()==self.lock['sha256'][file]
            self.plan={k:z[n].copy() for k,n in [('image_index','indices'),('flip','flip'),('factors','factors'),('order','order')]}
        if smoke_samples:
            # Smoke uses actual published draws, including four external draws; full plans are immutable.
            positions=list(np.flatnonzero(self.plan['image_index']>=43197)[:4])
            positions+= [k for k in range(smoke_samples) if k not in positions]
            positions=np.array(positions[:smoke_samples])
            self.plan={k:v[positions] for k,v in self.plan.items()}
        self.known_per_draw=(self.all_targets[self.plan['image_index']]!=-1).sum(1)
        # Zero-known windows are skipped by the trainer, without optimizer/scheduler updates.
    def __len__(self):return len(self.plan['image_index'])
    def __getitem__(self,k):
        import torch
        i=int(self.plan['image_index'][k]);r=self.rows[i]
        with Image.open(self.roots[r['source']]/r['relative_path']) as im:
            x=tensor_array(im,self.size,self.mean,self.std,bool(self.plan['flip'][k]),self.plan['factors'][k],self.plan['order'][k])
        return torch.from_numpy(x),torch.from_numpy(self.all_targets[i].astype(np.float32))

def make_aug_dataset(p,original_root,epoch,cfg,mean,std,smoke_samples=None):
    return PlannedDataset(p,original_root,epoch,cfg,mean,std,smoke_samples)
def known_window_count(dataset,batch_size,accum,batch_index):
    start=(batch_index//accum)*batch_size*accum
    return int(dataset.known_per_draw[start:min(start+batch_size*accum,len(dataset))].sum())
def save_exposure(run_dir,ds,epoch):
    counts=np.bincount(ds.plan['image_index'],minlength=len(ds.rows))
    positive=counts@(ds.all_targets==1).astype(np.int64)
    known=counts@(ds.all_targets!=-1).astype(np.int64)
    out=Path(run_dir)/'actual_epoch_exposure';out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'epoch_{epoch:02}.npz',image_draw_counts=counts,positive_class_draws=positive,known_class_draws=known,categories=np.array(ds.categories))
    orig=np.array([r['source']=='train_inner' for r in ds.rows])
    return dict(actual_original_draws=int(counts[orig].sum()),actual_external_draws=int(counts[~orig].sum()),actual_unique_seen=int((counts>0).sum()),actual_legwarmer_positive_draws=int(positive[ds.categories.index('leg warmer')]))
