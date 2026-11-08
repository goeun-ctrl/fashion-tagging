"""Run pinned team training unchanged except omitting its final automatic val evaluation."""
from pathlib import Path
import hashlib
import json
import sys

BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'source/scripts'),str(BASE/'source/src')]
import train_hires as th

original=Path(th.__file__).read_text()
old='finals = [("holdout", hold_loader, t_ho, hold), ("val", val_loader, vtargets, vnames)]'
assert original.count(old)==1
controlled=original.replace(old,'finals = [("holdout", hold_loader, t_ho, hold)]')
# __file__ and imports retain the exact pinned source locations.
exec(compile(controlled,str(Path(th.__file__)), 'exec'),th.__dict__)
original_build=th.build_hires_model

def build(cfg,n,pretrained):
    model=original_build(cfg,n,pretrained)
    def sha(items):
        h=hashlib.sha256()
        for k,v in sorted(items):
            h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    out=Path(cfg.out_dir)/cfg.exp_id
    record=dict(model_sha256=sha(model.state_dict().items()),pretrained=pretrained,
                pretrained_cfg=model.net.pretrained_cfg,seed=cfg.seed,init_ckpt=cfg.init_ckpt,
                source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                adapter='Only finals list changed: holdout only; all model/loss/sampler/initialization code unchanged')
    (out/'initialization.json').write_text(json.dumps(record,indent=2,default=str))
    return model

th.build_hires_model=build
if __name__=='__main__':th.main(sys.argv[1:])
