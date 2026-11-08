"""Validate published results without training, inference, or output mutation."""
import csv, hashlib, io, json, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
B=Path(__file__).resolve().parent
R=B/'results'
def read(p):return json.loads(p.read_text())
status=read(R/'logs/pipeline_status.json')
assert status['status']=='complete_core_evaluation'
assert all(s['status']=='passed' and s['returncode']==0 for s in status['stages'])
summary=read(R/'summary.json'); cfg=read(R/'config.json');lock=read(R/'selection.json')
history=list(csv.DictReader((R/'history.csv').open()))
assert [int(r['epoch']) for r in history]==list(range(1,19))
assert summary['epochs_run']==cfg['epochs']==18
assert summary['best_epoch']==int(max(history,key=lambda r:float(r['holdout_macro']))['epoch'])==lock['best_epoch']==15
assert hashlib.sha256((R/'config.json').read_bytes()).hexdigest()==lock['published_config_sha256']
assert lock['threshold']==0.4 and set(lock['special'].values())=={0.5}
main=(B/'README.md').read_text(); detail=(R/'README.md').read_text()
for split in ('holdout','val'):
 m=read(R/f'{split}_metrics.json')['metrics'];pc=m['per_class']
 rows=list(csv.DictReader((R/f'{split}_per_class.csv').open()))
 assert len(rows)==27
 assert m['n_images']==(2425 if split=='holdout' else 1158)
 assert abs(sum(pc['f1'])/27-m['macro_f1'])<1e-12
 tp,fp,fn=map(sum,(pc['tp'],pc['fp'],pc['fn']))
 assert abs(2*tp/(2*tp+fp+fn)-m['micro_f1'])<1e-12
 for key in ('micro_f1','macro_f1','image_f1'):
  assert f'{m[key]:.4f}' in main and f'{m[key]:.4f}' in detail
 for i,row in enumerate(rows):
  assert int(row['support'])==pc['support'][i]
  for key in ('precision','recall','f1'):assert abs(float(row[key])-pc[key][i])<1e-12
  line='| '+row['category']+' | '+row['support']+' | '+' | '.join(f'{float(row[k]):.4f}' for k in ('precision','recall','f1'))+' |'
  assert line in detail,line
for p in B.rglob('*.md'):
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in target or target.startswith('#'):continue
  assert (p.parent/target.split('#')[0]).exists(),(p,target)
for item in read(B/'source_manifest.json')['files']:
 assert hashlib.sha256((B/'source'/item['path']).read_bytes()).hexdigest()==item['sha256']
fig,axs=plt.subplots(1,3,figsize=(14,4))
for ax,key in zip(axs,['train_loss','holdout_bce','holdout_macro']):
 ax.plot([int(r['epoch']) for r in history],[float(r[key]) for r in history]);ax.set(xlabel='Epoch',title=key);ax.grid(alpha=.3)
fig.tight_layout(); buf=io.BytesIO();fig.savefig(buf,format='png');plt.close(fig);buf.seek(0)
assert ImageChops.difference(Image.open(buf).convert('RGB'),Image.open(R/'curves.png').convert('RGB')).getbbox() is None
assert all(p.suffix not in ('.pt','.pth','.ckpt','.npz','.pyc','.zip') for p in B.rglob('*') if p.is_file())
print('PASS: completion, 18 epochs, best epoch, config hash, aggregate/class tables, relative links, source hashes, exact graph pixels, excluded binary artifacts')
