"""CPU-only saved prediction verification and matched-run comparison."""
from pathlib import Path
import json,sys,csv
import numpy as np
ROOT=Path('/home/frostpine/fashion_tagging')
sys.path.insert(0,str(ROOT/'reference/fashion-tagging-c5827dfa/common'))
from metrics import load_predictions,compute_metrics,sigmoid,binarize
out=Path(sys.argv[1]);summary=json.loads((out/'summary.json').read_text())
base=ROOT/'experiments/role2_jeongeun_convnext'
others={'원본-only':base/'controlled_effect_20261106/original_only/runs/S2_cnxt_t_originalonly_commonaug_384_s42','외부506':base/'common506_retrain_20261105/runs/S2_jeongeun_cnxt_t_common506_b_384_s42','외부3458':out}
rows=[];classes=[]
for split in ['holdout','val']:
    names,target,logits,cats=load_predictions(out/f'preds_{split}.npz')
    current=compute_metrics(target,binarize(sigmoid(logits),.5))
    for k in ['micro_f1','macro_f1','image_f1']:assert abs(current[k]-summary[split][k])<1e-12
    for label,path in others.items():
        if not path.exists():raise FileNotFoundError(path)
        n,t,l,c=load_predictions(path/f'preds_{split}.npz')
        assert np.array_equal(n,names) and np.array_equal(t,target) and c==cats,'Evaluation identities differ'
        m=compute_metrics(t,binarize(sigmoid(l),.5));s=json.loads((path/'summary.json').read_text())
        for k in ['micro_f1','macro_f1','image_f1']:assert abs(m[k]-s[split][k])<1e-12
        rows.append(dict(split=split,experiment=label,**{k:m[k] for k in ['micro_f1','macro_f1','image_f1']}))
        for i,cat in enumerate(cats):classes.append(dict(split=split,experiment=label,category=cat,**{k:float(m['per_class'][k][i]) for k in ['support','tp','fp','fn','precision','recall','f1']}))
for name,data in [('comparison_three_runs.csv',rows),('comparison_27_classes.csv',classes)]:
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(out/'prediction_verification.json').write_text(json.dumps(dict(passed=True,threshold=.5,all_three_runs_same_ids_targets_categories=True,inference_rerun=False,metrics_recalculated_from_npz=True),indent=2))
print('Saved NPZ metrics and three-run evaluation alignment PASS')
