"""CPU-only synthetic KD smoke using the pinned team model, cached pretrained initialization and real labels."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
from timm.utils import ModelEmaV3

BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'source/scripts'),str(BASE/'source/src')]
import train_hires as th
from fashion_tagging.training.losses import build_loss


def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    cfg=json.loads((BASE/'checks/preflight.json').read_text())['config']
    assert cfg['distill_npz'] is None,'This smoke is explicitly synthetic; do not treat it as actual teacher validation'
    th.set_seed(42)
    model=th.build_hires_model(cfg,27,True)
    initial=state_hash(model)
    cats,names,targets=th.load_split(cfg['train_dir'],'train')
    index={n:i for i,n in enumerate(names)}
    inner=th.read_names(Path(cfg['splits_dir'])/'train_inner.txt')[:2]
    mean,std=th.timm_mean_std(cfg['model'])
    ds=th.HiResDataset(cfg['train_dir'],inner,targets[[index[n] for n in inner]],mean,std,False,128,0.0)
    batch=th.pad_collate([ds[(i,64)] for i in range(2)])
    model.train()
    ema=ModelEmaV3(model,decay=.9995,use_warmup=True)
    head_ids={id(p) for p in model.head_parameters()}
    opt=torch.optim.AdamW([{'params':[p for p in model.parameters() if id(p) not in head_ids],'lr':1e-4},
                           {'params':model.head_parameters(),'lr':1e-3}],weight_decay=.05)
    with tempfile.TemporaryDirectory(prefix='r4_kd_synthetic_') as td:
        teacher=Path(td)/'synthetic.npz'
        scores=np.array([[.1]*27,[.3]*27],dtype=np.float32)
        np.savez(teacher,filenames=np.array(inner[::-1]),scores=scores,categories=np.array(cats))
        a=type('Args',(),dict(distill_npz=str(teacher),distill_shift=.3,distill_alpha=.5))()
        soft=th.load_teacher(a,inner)
        assert np.allclose(soft[0],.5) and np.allclose(soft[1],7/34,atol=1e-6),'name reordering/shift wrong'
        logits=model(batch[0],batch[2]);assert logits.shape==(2,27)
        hard=build_loss('bce')(logits,batch[1])
        kd=th.soft_bce(logits,torch.from_numpy(soft))
        loss=.5*hard+.5*kd;assert torch.isfinite(loss)
        loss.backward();assert model.fc.weight.grad is not None and torch.isfinite(model.fc.weight.grad).all()
        assert any(p.grad is not None for p in model.net.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ema.update(model,step=1)
        checkpoint=Path(td)/'smoke.pt'
        torch.save({'model':ema.module.state_dict(),'categories':cats,'config':cfg,'epoch':1},checkpoint)
        st=torch.load(checkpoint,map_location='cpu',weights_only=False)
        fresh=th.build_hires_model(cfg,27,False);fresh.load_state_dict(st['model'],strict=True)
        assert state_hash(fresh)==state_hash(ema.module)
        fresh.eval();ema.module.eval()
        with torch.no_grad():
            assert torch.equal(fresh(batch[0],batch[2]),ema.module(batch[0],batch[2]))
    result=dict(passed=True,device='cpu',pretrained=True,initial_model_sha256=initial,
                initialization='pinned train_hires.set_seed(42) then build_hires_model(config,27,True); cached timm pretrained backbone; no init_ckpt',
                params=sum(p.numel() for p in model.parameters()),output_shape=list(logits.shape),
                synthetic_teacher_matching=True,synthetic_teacher_shift=True,bce_kd_backward=True,
                ema=True,checkpoint_roundtrip=True,actual_teacher_validated=False,gpu_smoke=False,
                limitation='2 real train_inner images at short64, synthetic teacher, CPU; not full-size AMP GPU validation')
    (BASE/'checks/smoke.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
