"""Full-canvas AMP forward/backward, optimizer, EMA and checkpoint smoke on assigned GPU."""
from pathlib import Path
import sys,json,tempfile
import torch
from timm.utils import ModelEmaV3
BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'source/scripts'),str(BASE/'source/src')]
import train_hires as th
cfg=json.loads((BASE/'checks/preflight.json').read_text())['config']
th.set_seed(42);torch.set_num_threads(4)
m=th.build_hires_model(cfg,27,True).cuda().to(memory_format=torch.channels_last)
ema=ModelEmaV3(m,decay=.9995,use_warmup=True);opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
scaler=torch.amp.GradScaler('cuda');x=torch.randn(8,3,1344,800,device='cuda').to(memory_format=torch.channels_last);sizes=torch.tensor([[1344,800]]*8,device='cuda')
for k in range(2):
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.float16):
  z=m(x,sizes);loss=.5*torch.nn.functional.binary_cross_entropy_with_logits(z,torch.zeros_like(z))+.5*th.soft_bce(z,torch.full_like(z,.5).float())
 assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1);scaler.step(opt);scaler.update();ema.update(m,step=k+1)
assert opt.state, 'No successful AMP optimizer update'
assert all(torch.isfinite(p).all() for p in m.parameters())
with tempfile.TemporaryDirectory(prefix='r4_kd_gpu_') as d:
 p=Path(d)/'check.pt';torch.save(ema.module.state_dict(),p);ema.module.load_state_dict(torch.load(p,weights_only=True))
(BASE/'checks/gpu_stress.json').write_text(json.dumps(dict(passed=True,batch=8,canvas=[1344,800],amp=True,ema=True,checkpoint=True,allocated_gib=torch.cuda.max_memory_allocated()/2**30,reserved_gib=torch.cuda.max_memory_reserved()/2**30),indent=2))
