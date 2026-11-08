"""Detached single-GPU pipeline; no overwrite, no Git operations, fail closed."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];OUT=ROOT/'runs/role4_bce_kd';RUN=OUT/'R4_bce_kd_s42'
GPU='YOUR_AVAILABLE_GPU'
PY=sys.executable

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
 lock=open('/tmp/frostpine_R4_bce_kd_gpu2.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 OUT.mkdir(parents=True,exist_ok=True);statusfile=OUT/'pipeline_status.json'
 assert not statusfile.exists() and not RUN.exists(),'existing baseline output; refusing overwrite'
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 assert GPU not in active,'GPU now occupied; refusing to overlap'
 pre=json.loads((BASE/'checks/preflight.json').read_text());assert pre['teacher']['verified']
 frozen={str(p):digest(p) for p in [*sorted((BASE/'source').rglob('*.py')),*sorted(BASE.glob('*.py')),Path(pre['teacher']['path'])]}
 state=dict(status='running',pid=os.getpid(),gpu_uuid=GPU,started=now(),experiment_id='R4_bce_kd_s42',source_commit=pre['source_commit'],stages=[],frozen_sha256=frozen,ood_status='not_available: no OOD data or image type tags')
 def save():
  t=statusfile.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(statusfile)
 def stage(name,args):
  assert all(digest(p)==h for p,h in frozen.items()),'frozen code or teacher changed'
  item=dict(name=name,command=[PY,*args],started=now(),status='running');state['stages'].append(item);state['current_stage']=name;save()
  with (OUT/(name+'.log')).open('x') as log:r=subprocess.run([PY,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  item.update(returncode=r.returncode,status='passed' if r.returncode==0 else 'failed',finished=now());save()
  if r.returncode:raise RuntimeError(name+' failed: see '+str(OUT/(name+'.log')))
 try:
  save();argv=pre['argv']
  smoke=argv.copy();smoke[smoke.index('--exp_id')+1]='R4_bce_kd_smoke_s42';smoke+=['--limit_train','64','--limit_eval','16','--epochs','1','--warmup_iters','1']
  stage('gpu_stress',[str(BASE/'gpu_stress.py')])
  stage('gpu_smoke',[str(BASE/'train_controlled.py'),*smoke])
  assert json.loads((OUT/'R4_bce_kd_smoke_s42/summary.json').read_text())['epochs_run']==1
  stage('train',[str(BASE/'train_controlled.py'),*argv])
  summary=json.loads((RUN/'summary.json').read_text());assert summary['epochs_run']==18 and 'val' not in summary
  assert json.loads((RUN/'initialization.json').read_text())['model_sha256']==json.loads((OUT/'R4_bce_kd_smoke_s42/initialization.json').read_text())['model_sha256'], 'smoke and full run initial states differ'
  stage('holdout_tta',[str(BASE/'evaluate_controlled.py'),'--run',str(RUN),'--split','holdout'])
  assert (RUN/'selection.json').exists()
  stage('val_tta',[str(BASE/'evaluate_controlled.py'),'--run',str(RUN),'--split','val'])
  stage('report',[str(BASE/'report_results.py')])
  state.update(status='complete_core_evaluation',finished=now())
 except BaseException as e:
  state.update(status='failed',error=str(e),traceback=traceback.format_exc(),finished=now());raise
 finally:save()
if __name__=='__main__':main()
