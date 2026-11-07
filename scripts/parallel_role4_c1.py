"""Adopt the existing BCE process unchanged; launch ASL on one other GPU and join before evaluation."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from role4_c1 import ROOT, digest, environment

BCE_GPU = 'GPU-bed51d32-f978-d07a-c020-01d7a36e63ea'
ASL_GPU = 'GPU-b7682b36-2ce0-802b-8bb1-e7668087ecde'
BASE = ROOT / 'runs/role4_c1'


def now():
    return datetime.now(timezone.utc).isoformat()


def process(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p / 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, ppid=int(fields[1]), start_ticks=fields[19],
                    command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def same_process(identity):
    current = process(identity['pid'])
    return current is not None and current['start_ticks'] == identity['start_ticks']


def atomic(path, data):
    tmp = path.with_suffix('.parallel.tmp')
    tmp.write_text(json.dumps(data, indent=2)); tmp.replace(path)


def validate_completed(loss, state):
    folder = BASE / f'R4_C1_{loss}_s42'
    summary = json.loads((folder / 'summary.json').read_text())
    history = list(csv.DictReader((folder / 'history.csv').open()))
    cfg = json.loads((folder / 'config.json').read_text())
    bcfg = json.loads((BASE / 'R4_C1_bce_s42/config.json').read_text())
    assert summary['epochs_run'] == 18 and [int(r['epoch']) for r in history] == list(range(1,19))
    assert 1 <= summary['best_epoch'] <= 18
    assert cfg['epochs'] == 18 and cfg['loss'] == loss and not cfg['limit_train'] and not cfg['limit_eval']
    assert cfg['initial_file_sha256'] == state['initial_sha256']
    assert cfg['initial_model_sha256'] == bcfg['initial_model_sha256']
    assert cfg['initial_head_sha256'] == bcfg['initial_head_sha256']
    for filename in ('best.pt','last.pt','training_state.pt','preds_holdout.npz'):
        assert (folder / filename).stat().st_size > 0
    return summary


def prerequisites(old_pid, bce_pid):
    state = json.loads((BASE / 'pipeline_status.json').read_text())
    assert state['status'] == 'running' and state['current_stage'] == 'train_bce'
    old, bce = process(old_pid), process(bce_pid)
    handoff_path = BASE / 'parallel_handoff.json'
    if handoff_path.exists():
        # Recover a supervisor-only handoff if its advisory lock was still closing on first attempt.
        previous = json.loads(handoff_path.read_text())
        assert not same_process(previous['old_controller'])
        assert bce and bce['start_ticks'] == previous['bce_before']['start_ticks']
        assert bce['pid'] == previous['bce_before']['pid']
        assert process(previous['new_controller']) is None, 'another handoff controller is active'
        old = previous['old_controller']
    else:
        assert old and 'scripts/role4_c1_run.py' in old['command']
        assert bce and 'scripts/role4_c1.py train --loss bce' in bce['command'] and bce['ppid'] == old_pid
        assert str(bce_pid) in (Path('/proc') / str(old_pid) / 'task' / str(old_pid) / 'children').read_text().split()
    assert digest(BASE / 'shared_initial_s42.pt') == state['initial_sha256']
    assert environment()['source_sha256'] == state['environment']['source_sha256']
    assert not (BASE / 'R4_C1_asl_s42').exists() and not (BASE / 'train_asl.log').exists()
    procs = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert ASL_GPU not in procs, 'GPU0 has a compute process; leave sequential pipeline untouched'
    assert f'{BCE_GPU}, {bce_pid}' in procs
    return state, old, bce


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-controller',type=int,default=1662352)
    ap.add_argument('--bce-pid',type=int,default=1662623)
    ap.add_argument('--check-only',action='store_true')
    a = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == BCE_GPU
    state, old, bce = prerequisites(a.old_controller,a.bce_pid)
    print('PREREQUISITES PASS',json.dumps(dict(old=old,bce=bce,asl_gpu=ASL_GPU)),flush=True)
    if a.check_only:
        return
    gpu0_lock = open('/tmp/frostpine_role4_c1_gpu0.lock','a')
    fcntl.flock(gpu0_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Recheck immediately before handoff. Only our supervisor PID is signalled; no process-group signals.
    prerequisites(a.old_controller,a.bce_pid)
    previous_path = BASE / 'parallel_handoff.json'
    handoff = (json.loads(previous_path.read_text()) if previous_path.exists() else
               dict(started=now(),old_controller=old,bce_before=bce,
                    bce_signal_sent=False,training_restart=False,asl_gpu_uuid=ASL_GPU))
    handoff.setdefault('controller_attempts',[]).append(os.getpid())
    handoff['new_controller']=os.getpid()
    atomic(BASE / 'parallel_handoff.json',handoff)
    if same_process(old):
        os.kill(old['pid'],signal.SIGTERM)
    for _ in range(50):
        if not same_process(old):
            break
        time.sleep(.1)
    assert not same_process(old), 'old supervisor did not exit; do not launch duplicate ASL'
    assert same_process(bce), 'BCE identity unexpectedly changed'
    gpu3_lock = open('/tmp/frostpine_role4_c1_gpu3.lock','a')
    fcntl.flock(gpu3_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    handoff.update(old_controller_stopped=True,bce_after=process(bce['pid']),bce_preserved=True)
    atomic(BASE / 'parallel_handoff.json',handoff)
    state.update(pid=os.getpid(),mode='parallel',gpu_assignments=dict(bce=dict(index=3,uuid=BCE_GPU),
        asl=dict(index=0,uuid=ASL_GPU)),current_stage='train_bce_and_asl',
        controller_sha256=digest(__file__),adopted_bce=bce)
    status_path=BASE/'pipeline_status.json'
    initial=BASE/'shared_initial_s42.pt'
    expected_sources=state['environment']['source_sha256']
    def save(): atomic(status_path,state)
    def verify_frozen():
        assert digest(initial)==state['initial_sha256']
        assert environment()['source_sha256']==expected_sources
    def stage_run(name,args):
        verify_frozen()
        item=dict(stage=name,started=now(),command=[sys.executable,*args],status='running')
        state['stages'].append(item); state['current_stage']=name; save()
        print('START',name,flush=True)
        with (BASE/(name+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        item.update(finished=now(),returncode=result.returncode,status='passed' if result.returncode==0 else 'failed')
        save()
        assert result.returncode==0, f'{name} failed'
    try:
        verify_frozen()
        item=dict(stage='train_asl',started=now(),status='running',gpu_uuid=ASL_GPU)
        state['stages'].append(item)
        args=[sys.executable,'-u','scripts/role4_c1.py','train','--loss','asl','--device','cuda:0']
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=ASL_GPU
        with (BASE/'train_asl.log').open('x') as log:
            asl=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        item.update(pid=asl.pid,command=args)
        state['asl_pid']=asl.pid; save()
        print('ASL STARTED',asl.pid,'BCE UNCHANGED',bce['pid'],flush=True)
        bce_item=next(s for s in state['stages'] if s['stage']=='train_bce')
        while same_process(bce) or asl.poll() is None:
            if not same_process(bce) and bce_item['status']=='running':
                validate_completed('bce',state)
                bce_item.update(status='passed',finished=now(),completion_check='adopted process ended; all18 epochs and final artifacts verified')
                save()
            code=asl.poll()
            if code is not None and item['status']=='running':
                assert code==0, f'ASL exited {code}; BCE left untouched'
                validate_completed('asl',state)
                item.update(status='passed',returncode=code,finished=now()); save()
            time.sleep(10)
        assert asl.returncode==0
        for loss in ('bce','asl'):
            validate_completed(loss,state)
            entry=next(s for s in state['stages'] if s['stage']=='train_'+loss)
            entry.update(status='passed',finished=entry.get('finished',now()))
        state['both_training_completed']=True; save()
        # No holdout selection or official val until both independent18-epoch runs passed.
        for loss in ('bce','asl'):
            stage_run('holdout_'+loss,['scripts/role4_c1_evaluate.py','--run',str(BASE/f'R4_C1_{loss}_s42'),
                '--split','holdout','--out',str(BASE/f'eval_holdout_{loss}')])
        common=['--bce_run',str(BASE/'R4_C1_bce_s42'),'--asl_run',str(BASE/'R4_C1_asl_s42')]
        report=ROOT/'report/role4_c1_holdout_s42'
        stage_run('compare_holdout',['scripts/role4_c1_compare.py',*common,
            '--bce_preds',str(BASE/'eval_holdout_bce'),'--asl_preds',str(BASE/'eval_holdout_asl'),'--out',str(report)])
        for loss in ('bce','asl'):
            stage_run('val_'+loss,['scripts/role4_c1_evaluate.py','--run',str(BASE/f'R4_C1_{loss}_s42'),
                '--split','val','--out',str(BASE/f'eval_val_{loss}'),'--lock',str(report/'selection.json')])
        stage_run('compare_val',['scripts/role4_c1_compare.py',*common,'--bce_preds',str(BASE/'eval_val_bce'),
            '--asl_preds',str(BASE/'eval_val_asl'),'--out',str(ROOT/'report/role4_c1_val_s42'),
            '--lock',str(report/'selection.json')])
        verify_frozen()
        state.update(status='complete',finished=now(),initial_preserved=True); save()
    except BaseException as exc:
        state.update(status='failed',error=str(exc)); save(); raise


if __name__ == '__main__':
    main()
