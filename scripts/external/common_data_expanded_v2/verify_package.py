"""Verify the private release on CPU. Does not train, infer, or access the network."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance,ImageOps,__version__ as pillow_version

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(root):
    checks=json.loads((root/'package-files.sha256.json').read_text(encoding='utf-8'))
    for n,h in checks.items():
        p=(root/n).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h,n
    lock=json.loads((root/'shared-lock.json').read_text(encoding='utf-8'))
    assert lock['schema']=='fashion-common-v1' and lock['approved'] is True and lock['label_mode']=='partial'
    assert (lock['seed'],lock['draws'],lock['epochs'],lock['replacement'])==(42,43197,30,True)
    for n,h in lock['sha256'].items():
        if n not in ['train/labels.npz','val/labels.npz']:assert sha(root/n)==h,n
    rows=list(csv.DictReader((root/'training_manifest.csv').open(encoding='utf-8',newline='')))
    with np.load(root/'training_targets.npz',allow_pickle=False) as z:
        y=z['targets'];cats=z['categories'].tolist()
        assert z['image_ids'].tolist()==[r['image_id'] for r in rows]
        assert z['data_version'].item()==lock['data_version'] and cats==lock['categories']
    assert y.shape==(len(rows),27) and np.isin(y,[-1,0,1]).all()
    assert len(set(r['image_id'] for r in rows))==len(rows)
    assert all(r['approved']=='1' and r['data_version']==lock['data_version'] for r in rows)
    assert len([r for r in rows if r['source']=='train_inner'])==43197
    external=[r for r in rows if r['source']=='external'];assert len(external)==3458
    for row in external:
        p=(root/'selected_external/images'/row['relative_path']).resolve()
        assert p.is_relative_to((root/'selected_external/images').resolve())
        assert sha(p)==row['image_sha256']
        with Image.open(p) as im:im.load()
    policy=json.loads((root/'augmentation_policy.json').read_text(encoding='utf-8'))
    assert policy['pillow_version']==pillow_version,'Install exact Pillow '+policy['pillow_version']
    counts=(y==1).sum(0)
    table=list(csv.DictReader((root/'class_counts.csv').open(encoding='utf-8',newline='')))
    assert [r['class'] for r in table]==cats and [int(r['positive']) for r in table]==counts.tolist()
    w=np.ones(27);w[np.isin(cats,policy['medium_weight'])]=1.3;w[counts<1000]=1.75;w[cats.index('leg warmer')]=2.5
    weights=np.maximum(np.where(y==1,w,1.).max(1),1.)
    assert np.array_equal(weights,[float(r['sampling_weight']) for r in rows])
    j=np.zeros(27);j[np.isin(cats,policy['general'])]=.2;j[np.isin(cats,policy['focus'])]=.35
    probs=np.maximum(np.where(y==1,j,0.).max(1),0.)
    for e in range(1,31):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([42,e])))
        idx=rng.choice(len(rows),size=43197,replace=True,p=weights/weights.sum()).astype(np.int32)
        flip=rng.random(43197)<.5;apply=rng.random(43197)<probs[idx]
        factors=np.ones((43197,3),dtype=np.float32)
        factors[apply]=rng.uniform([.9,.9,.95],[1.1,1.1,1.05],size=(int(apply.sum()),3))
        order=np.argsort(rng.random((43197,3)),axis=1).astype(np.uint8)
        with np.load(root/f'epoch_plans/epoch_{e:02d}.npz',allow_pickle=False) as z:
            expected={'epoch':e,'data_version':lock['data_version'],'label_mode':'partial','manifest_sha256':lock['sha256']['training_manifest.csv'],'targets_sha256':lock['sha256']['training_targets.npz'],'policy_sha256':lock['sha256']['augmentation_policy.json']}
            assert all(z[k].item()==v for k,v in expected.items())
            assert all(np.array_equal(z[k],v) for k,v in [('indices',idx),('flip',flip),('factors',factors),('order',order)])
    print(json.dumps({'package':'PASS','files_sha256':len(checks),'external_images':len(external),'targets':list(y.shape),'regenerated_epoch_plans_equal':30,'server_full_preflight':'NOT_PERFORMED','training':'NOT_PERFORMED'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('package',type=Path);verify(p.parse_args().package)
