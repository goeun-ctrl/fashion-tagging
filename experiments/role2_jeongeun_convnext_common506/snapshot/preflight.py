"""Read-only original data audit and real package CPU integration checks."""
from pathlib import Path
import concurrent.futures,csv,hashlib,json,subprocess,sys
import numpy as np
import torch
from PIL import Image
from augmented_data import validate_prepared,make_aug_dataset,known_window_count,sha
ROOT=Path('/home/frostpine/fashion_tagging')
HERE=Path(__file__).resolve().parent
P=HERE.parent/'fashion-common-506-partial-20261105-v1'
REF=ROOT/'reference/fashion-tagging-c5827dfa'
sys.path.insert(0,str(REF/'common'))
from labels import load_split
def main():
    subprocess.run([sys.executable,str(P/'verify_package.py'),str(P)],check=True)
    cats,names,y=load_split(ROOT/'data/train','train')
    vc,vn,vy=load_split(ROOT/'data/val','val')
    inner=(REF/'common/splits/train_inner.txt').read_text().splitlines()
    hold=(REF/'common/splits/holdout.txt').read_text().splitlines()
    info=validate_prepared(P,cats,inner,names,y)
    assert vc==cats and len(vn)==1158 and len(hold)==2425 and len(inner)==43197
    assert not set(inner)&set(hold) and set(inner)|set(hold)==set(names) and not set(names)&set(vn)
    for split in ['train','val']:assert sha(ROOT/f'data/{split}/labels.npz')==info['shared']['sha256'][split+'/labels.npz']
    for name in ['train_inner.txt','holdout.txt']:assert sha(P/'fixed_splits'/name)==sha(REF/'common/splits'/name)
    rows=list(csv.DictReader((P/'training_manifest.csv').open(encoding='utf-8',newline='')))
    expected={r['image_id']:r['image_sha256'] for r in rows}
    tasks=[(ROOT/'data/train/images'/n,n) for n in names]+[(ROOT/'data/val/images'/n,n) for n in vn]
    def check(task):
        path,name=task;h=sha(path)
        if name in expected:assert h==expected[name],name
        return h
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:hashes=list(ex.map(check,tasks))
    assert len(set(hashes))==len(hashes),'Duplicate original image bytes'
    for row in rows[43197:]:
        h=sha(P/'selected_external/images'/row['relative_path']);assert h not in set(hashes)
    cfg=json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    ds=make_aug_dataset(P,ROOT/'data/train',1,cfg,[.485,.456,.406],[.229,.224,.225],64)
    batches=list(torch.utils.data.DataLoader(ds,batch_size=16,shuffle=False,num_workers=0))
    assert all(x.shape[1:]==(3,384,384) and torch.isfinite(x).all() for x,t in batches)
    assert sum(int((t==-1).sum()) for x,t in batches)>0
    z=torch.randn(32,27,requires_grad=True);target=torch.cat([t for x,t in batches[:2]])
    known=target>=0;per=torch.nn.functional.binary_cross_entropy_with_logits(z,target.clamp(min=0),reduction='none')
    expected_loss=(per*known).sum()/known.sum();g=torch.autograd.grad(expected_loss,z)[0]
    z2=z.detach().clone().requires_grad_()
    for k in range(2):
        start=k*16;t=target[start:start+16];kn=t>=0
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z2[start:start+16],t.clamp(min=0),reduction='none')
        ((loss*kn).sum()/known_window_count(ds,16,2,k)).backward()
    torch.testing.assert_close(z2.grad,g)
    assert (z2.grad[target<0]==0).all()
    report=dict(passed=True,hostname=__import__('socket').gethostname(),user=__import__('getpass').getuser(),train=45622,train_inner=43197,holdout=2425,val=1158,external=506,original_image_hashes_read=len(hashes),no_duplicate_bytes=True,real_384_dataloader='PASS',partial_labels_preserved=True,accumulated_gradient_equals_full_known_mean=True,shared=info['shared'],gpu_used=False)
    (HERE.parent/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Actual server original-file preflight and real CPU DataLoader/gradient checks PASS',flush=True)
if __name__=='__main__':main()
