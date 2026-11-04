import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,hashlib,gc,sys,importlib.metadata
ROOT=Path(Path(__file__).with_name('current_backup_path.txt').read_text(encoding='utf-8'))
import torch,timm
assert torch.version.cuda is None,'Expected CPU-only torch wheel'
metadata=json.loads((ROOT/'가중치_SHA256_검증.json').read_text(encoding='utf-8'))
verified={}
for name in ['best.pt','last.pt']:
 p=ROOT/'weights'/name
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 assert h.hexdigest()==metadata[name]['sha256']
 checkpoint=torch.load(p,map_location='cpu',weights_only=False)
 assert isinstance(checkpoint,dict) and 'model' in checkpoint
 model=timm.create_model('convnext_tiny.fb_in22k_ft_in1k_384',pretrained=False,num_classes=27,global_pool='avg').cpu().eval()
 result=model.load_state_dict(checkpoint['model'],strict=True)
 assert not result.missing_keys and not result.unexpected_keys
 assert all(p.device.type=='cpu' for p in model.parameters())
 count=sum(p.numel() for p in model.parameters());assert count==27840891
 assert model.get_classifier().out_features==27
 verified[name]={'sha256':h.hexdigest(),'strict_load':True,'device':'cpu','tensor_keys':len(checkpoint['model']),'parameters':count,'classifier_weight_shape':list(model.get_classifier().weight.shape),'head_structure':repr(model.head),'completed_epoch':checkpoint['completed_epoch'],'best_epoch':checkpoint['best_epoch'],'checkpoint_keys':list(checkpoint),'cuda_wheel':torch.version.cuda,'forward_or_inference_executed':False}
 del checkpoint,model;gc.collect()
versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','timm','numpy','Pillow']}
(ROOT/'CPU_체크포인트_로드검증.json').write_text(json.dumps({'verified':True,'checkpoints':verified,'local_cpu_library_versions':versions},ensure_ascii=False,indent=2),encoding='utf-8')
loader='''# Trusted checkpoints only: verify SHA-256 before loading this full training checkpoint.
import torch
import timm
from pathlib import Path

root = Path(__file__).resolve().parent
checkpoint = torch.load(root / "weights" / "best.pt", map_location="cpu", weights_only=False)
# The checkpoint contains model, optimizer, scheduler, scaler and RNG state.
# Model parameters are under the model key, not the whole checkpoint dictionary.
model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k_384", pretrained=False, num_classes=27, global_pool="avg")
model.load_state_dict(checkpoint["model"], strict=True)
model = model.cpu().eval()
print("Loaded on CPU; no inference executed:", sum(p.numel() for p in model.parameters()))
'''
(ROOT/'load_model_cpu.py').write_text(loader,encoding='utf-8')
(ROOT/'모델_로드방법.md').write_text('''# 모델 CPU 로드 방법

`load_model_cpu.py`는 서버에서 직접 복사하고 SHA-256을 검증한 본인 체크포인트에만 사용한다. 전체 학습 체크포인트에는 NumPy RNG 등의 상태가 있으므로 `weights_only=False`가 필요하다. 출처가 불명확한 파일에는 사용하지 않는다.

1. CPU 전용 PyTorch와 torchvision, timm 1.0.22, NumPy를 준비한다. 실제 검증 버전은 `CPU_체크포인트_로드검증.json`에 있다.
2. 이 폴더에서 `python load_model_cpu.py`를 실행한다.
3. `torch.load(..., map_location="cpu", weights_only=False)` 후 `checkpoint["model"]`을 구조에 strict 로드한다. `pretrained=False`로 새 모델 구조만 만들기 때문에 원격 사전학습 가중치를 다시 내려받지 않는다.

이 스크립트는 로드만 수행하며 재추론·추가 학습은 하지 않는다. 추론을 별도로 수행할 때는 `class_order.json`과 보관된 공통 전처리, 실제 mean/std·384 종횡비 유지 bicubic·정규화 전 검정 padding을 유지해야 한다. 출력은 logits이며 sigmoid 이후 threshold 0.5를 적용한다. 구조는 원래 timm의 GAP → LayerNorm → Linear(27) 헤드를 보존한다.
''',encoding='utf-8')
manifest=[]
for p in ROOT.rglob('*'):
 if p.is_file() and p.name!='백업_전체파일_SHA256.json':
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8388608),b''):h.update(b)
  manifest.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(ROOT/'백업_전체파일_SHA256.json').write_text(json.dumps({'files':manifest,'total_files':len(manifest),'total_bytes':sum(x['bytes'] for x in manifest)},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'verified':True,'checkpoints':verified,'file_count':len(manifest)},ensure_ascii=True))
