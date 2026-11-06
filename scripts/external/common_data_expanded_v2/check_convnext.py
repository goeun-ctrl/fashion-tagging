from pathlib import Path
import sys,json,importlib.util,hashlib
import numpy as np
w=Path('common_data_expansion_20261106').resolve();p=w/'fashion-common-expanded-partial-20261106-v2';sys.path.insert(0,str(p))
s=importlib.util.spec_from_file_location('convnext_data_v2',p/'convnext_data_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
lock,rows,y,cats=m.read_inputs(p);orig=rows[:43197];info=m.validate_prepared(p,cats,[r['image_id'] for r in orig],[r['image_id'] for r in orig],y[:43197]);tested=0
for epoch in range(1,31):
 ds=m.PlannedDataset(p,Path('original_train_not_copied'),epoch,{'input_size':384},[.485,.456,.406],[.229,.224,.225],None)
 assert len(ds)==43197
 if epoch in [1,30]:
  selected=np.flatnonzero(ds.plan['image_index']>=43197)[:32]
  for k in selected:
   x,target=ds[int(k)];assert tuple(x.shape)==(3,384,384) and x.isfinite().all();assert np.array_equal(target.numpy(),y[int(ds.plan['image_index'][k])]);tested+=1
 assert m.known_window_count(ds,16,2,0)==int(ds.known_per_draw[:32].sum())
report={'reader':'PASS','package_lock_sha256':m.LOCK_SHA,'rows':len(rows),'external':len(rows)-43197,'epochs_checked':30,'external_384_tensors_checked':tested,'model_forward_or_training':False,'note':'New separate v2 adapter; the existing 506 trainer intentionally pins the v1 lock and cannot accept this package without selecting the v2 adapter.'}
(p/'convnext_compatibility_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
