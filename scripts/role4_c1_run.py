"""Run the authorized single-GPU pipeline sequentially, fail closed on any stage failure."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from role4_c1 import ROOT, digest, settings, environment

GPU = 'GPU-bed51d32-f978-d07a-c020-01d7a36e63ea'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('MKL_NUM_THREADS','4')
    lock = open('/tmp/frostpine_role4_c1_gpu3.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base = ROOT / 'runs/role4_c1'
    base.mkdir(parents=True,exist_ok=True)
    status_path = base / 'pipeline_status.json'
    if status_path.exists():
        raise FileExistsError(status_path)
    initial = base / 'shared_initial_s42.pt'
    initial_hash = digest(initial)
    source = environment()['source_sha256']
    state = dict(gpu_uuid=GPU, pid=os.getpid(), initial_sha256=initial_hash, started=datetime.now(timezone.utc).isoformat(),
                 environment=environment(), stages=[], status='running')
    def save():
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2)); tmp.replace(status_path)
    def run(name,args):
        assert digest(initial)==initial_hash and environment()['source_sha256']==source
        item=dict(stage=name,command=[sys.executable,*args],started=datetime.now(timezone.utc).isoformat(),status='running')
        state['stages'].append(item); state['current_stage']=name; save()
        print('START',name,flush=True)
        with (base / (name+'.log')).open('x') as log:
            done=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        item.update(returncode=done.returncode,finished=datetime.now(timezone.utc).isoformat(),status='passed' if done.returncode==0 else 'failed')
        save()
        if done.returncode:
            raise RuntimeError(f'{name} failed: see {base / (name+".log")}')
        assert digest(initial)==initial_hash
        print('DONE',name,flush=True)
    try:
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        assert GPU not in procs, 'selected GPU acquired by another compute process; refusing to overlap'
        save()
        run('gpu_smoke',['scripts/role4_c1_gpu_smoke.py'])
        smoke=json.loads((ROOT / 'runs/role4_c1_gpu_smoke/checks.json').read_text())
        assert smoke['passed'] and smoke['initial_preserved']
        for loss in ('bce','asl'):
            run('train_'+loss,['scripts/role4_c1.py','train','--loss',loss,'--device','cuda:0'])
            summary=json.loads((base / f'R4_C1_{loss}_s42/summary.json').read_text())
            assert summary['epochs_run']==18
        for loss in ('bce','asl'):
            run('holdout_'+loss,['scripts/role4_c1_evaluate.py','--run',str(base / f'R4_C1_{loss}_s42'),
                 '--split','holdout','--out',str(base / f'eval_holdout_{loss}')])
        common=['--bce_run',str(base/'R4_C1_bce_s42'),'--asl_run',str(base/'R4_C1_asl_s42')]
        report=ROOT/'report/role4_c1_holdout_s42'
        run('compare_holdout',['scripts/role4_c1_compare.py',*common,'--bce_preds',str(base/'eval_holdout_bce'),
            '--asl_preds',str(base/'eval_holdout_asl'),'--out',str(report)])
        for loss in ('bce','asl'):
            run('val_'+loss,['scripts/role4_c1_evaluate.py','--run',str(base/f'R4_C1_{loss}_s42'),
                '--split','val','--out',str(base/f'eval_val_{loss}'),'--lock',str(report/'selection.json')])
        run('compare_val',['scripts/role4_c1_compare.py',*common,'--bce_preds',str(base/'eval_val_bce'),
            '--asl_preds',str(base/'eval_val_asl'),'--out',str(ROOT/'report/role4_c1_val_s42'),
            '--lock',str(report/'selection.json')])
        state.update(status='complete',initial_preserved=digest(initial)==initial_hash,
                     finished=datetime.now(timezone.utc).isoformat()); save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc)); save(); raise


if __name__ == '__main__':
    main()
