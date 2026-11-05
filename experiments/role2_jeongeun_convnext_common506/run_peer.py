"""Portable launcher around the byte-identical running trainer; never starts on import."""
import argparse,csv,hashlib,json,os,re,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SNAPSHOT=HERE/'snapshot'
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check_snapshot():
    record=json.loads((HERE/'running_code_hash_verification.json').read_text(encoding='utf-8'))
    for name,digest in record['sha256'].items():
        if sha(SNAPSHOT/name)!=digest:raise ValueError('Source snapshot changed: '+name)
def preflight(package,reference,train,val):
    import numpy as np
    sys.path.insert(0,str(reference/'common'))
    from labels import load_split
    sys.path.insert(0,str(SNAPSHOT))
    from augmented_data import validate_prepared
    subprocess.run([sys.executable,str(package/'verify_package.py'),str(package)],check=True)
    cats,names,y=load_split(train,'train');vc,vn,vy=load_split(val,'val')
    inner=(reference/'common/splits/train_inner.txt').read_text().splitlines()
    hold=(reference/'common/splits/holdout.txt').read_text().splitlines()
    assert len(inner)==43197 and len(hold)==2425 and len(vn)==1158
    assert vc==cats and set(inner)|set(hold)==set(names) and not set(inner)&set(hold) and not set(names)&set(vn)
    info=validate_prepared(package,cats,inner,names,y)
    for split,root in [('train',train),('val',val)]:assert sha(root/'labels.npz')==info['shared']['sha256'][split+'/labels.npz']
    for name in ['train_inner.txt','holdout.txt']:assert sha(package/'fixed_splits'/name)==sha(reference/'common/splits'/name)
    rows=list(csv.DictReader((package/'training_manifest.csv').open(encoding='utf-8',newline='')))
    expected={r['image_id']:r['image_sha256'] for r in rows}
    seen=set()
    for root,ids in [(train,names),(val,vn)]:
        for name in ids:
            digest=sha(root/'images'/name)
            assert digest not in seen,'Original byte duplicate: '+name
            if name in expected:assert digest==expected[name],name
            seen.add(digest)
    for row in rows[43197:]:
        digest=sha(package/'selected_external/images'/row['relative_path'])
        assert digest not in seen and digest==row['image_sha256'];seen.add(digest)
    return {'passed':True,'label_mode':'partial','data_version':info['shared']['data_version'],'train_inner':43197,'holdout':2425,'val':1158,'external':506,'original_and_external_image_hashes_checked':len(seen)}
def args_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['preflight','smoke','train'],default='preflight')
    for name in ['package','reference-dir','train-dir','val-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--owner',required=True)
    p.add_argument('--gpu',type=int);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--accum',type=int,default=2)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--output-root',type=Path,default=HERE/'private')
    p.add_argument('--resume',action='store_true');return p
def acquire_lock(path):
    f=path.open('a+b')
    if os.name=='nt':
        import msvcrt
        if path.stat().st_size==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return f
def main():
    a=args_parser().parse_args();check_snapshot()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',a.run_id) or 'smoke' in a.run_id.lower():raise ValueError('Use a unique simple run ID without the word smoke')
    if a.batch_size*a.accum!=32 or min(a.batch_size,a.accum)<1:raise ValueError('Effective batch must be 32')
    if a.stage!='preflight' and (a.gpu is None or a.gpu<0):raise ValueError('Choose your assigned GPU explicitly with --gpu; do not reuse another person\'s allocation')
    if a.stage!='train' and a.resume:raise ValueError('--resume applies only to this run\'s full training')
    a.package=a.package.resolve();a.reference_dir=a.reference_dir.resolve();a.train_dir=a.train_dir.resolve();a.val_dir=a.val_dir.resolve()
    out=a.output_root.resolve()/a.run_id;out.mkdir(parents=True,exist_ok=True)
    lock=acquire_lock(out/'.launcher.lock')
    audit=preflight(a.package,a.reference_dir,a.train_dir,a.val_dir)
    (out/'preflight.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.stage=='preflight':print(json.dumps(audit));return
    cfg=json.loads((SNAPSHOT/'config.json').read_text(encoding='utf-8'))
    cfg.update(exp_id=a.run_id,owner=a.owner,batch_size=a.batch_size,accum=a.accum,num_workers=a.workers,work_branch='peer-reproduction',notes='Shared B-mode reproduction using the verified source snapshot. Partial labels preserved. GPU and run ID supplied by peer; no Git publication performed by launcher.')
    config=out/'peer_config.json'
    if config.exists():
        assert json.loads(config.read_text(encoding='utf-8'))==cfg,'Existing peer run configuration differs'
    else:config.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    run=out/('smoke' if a.stage=='smoke' else 'run')
    cmd=[sys.executable,str(SNAPSHOT/'trainer.py'),'--config',str(config),'--prepared-dir',str(a.package),'--reference-dir',str(a.reference_dir),'--train-dir',str(a.train_dir),'--val-dir',str(a.val_dir),'--output-dir',str(run),'--batch-size',str(a.batch_size),'--accum',str(a.accum),'--workers',str(a.workers),'--device','cuda:0']
    if a.stage=='smoke':cmd+=['--smoke','--smoke-train','128','--smoke-holdout','64']
    if a.resume:
        if not (run/'last.pt').is_file():raise FileNotFoundError('No checkpoint for this peer run; do not resume the original baseline')
        cmd+=['--resume','auto']
    elif a.stage=='train':
        smoke=out/'smoke/summary.json'
        if not smoke.exists():raise RuntimeError('Run this peer ID with --stage smoke and verify it before full training')
        summary=json.loads(smoke.read_text(encoding='utf-8'))
        assert summary['state']=='completed' and summary['batch_size']==a.batch_size and summary['accum']==a.accum
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    (out/'launch_record.json').write_text(json.dumps(dict(physical_gpu=a.gpu,command=cmd,peer_config_sha256=sha(config),source_snapshot_verified=True),ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/(a.stage+'.log')).open('a',encoding='utf-8') as log:
        print('Writing log:',log.name,flush=True);result=subprocess.run(cmd,env=environment,stdout=log,stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
