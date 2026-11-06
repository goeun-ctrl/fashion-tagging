"""Exclusive detached pipeline: CPU preflight -> GPU smoke -> training -> final report."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
ROOT=Path('/home/frostpine/fashion_tagging')
EXP='S2_jeongeun_cnxt_t_common3458_b_384_s42'
def status(**d):
    p=BASE/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),**d),ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def memory():
    data=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return int(data.strip())
def command(out,batch,accum):
    return [sys.executable,str(HERE/'trainer.py'),'--prepared-dir',str(BASE/'fashion-common-expanded-partial-20261106-v2'),'--reference-dir',str(ROOT/'reference/fashion-tagging-c5827dfa'),'--train-dir',str(ROOT/'data/train'),'--val-dir',str(ROOT/'data/val'),'--output-dir',str(out),'--batch-size',str(batch),'--accum',str(accum),'--workers','4','--device','cuda:0','--log-every','100']
def execute(cmd,log):
    print('START:',str(log),flush=True)
    with log.open('a',encoding='utf-8') as f:return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ).returncode
def main():
    handle=(BASE/'.pipeline.lock').open('a+')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Existing own pipeline active; not launching a duplicate',flush=True);return
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    try:
        if (BASE/'pipeline_status.json').exists():
            previous=json.loads((BASE/'pipeline_status.json').read_text())
            if previous.get('stage')=='completed':
                print('Already completed; not repeating',flush=True);return
        status(stage='cpu_preflight',physical_gpu=0)
        assert execute([sys.executable,str(HERE/'preflight.py')],BASE/'preflight.log')==0,'CPU preflight failed'
        batch=16;accum=2;chosen=None
        for batch,accum in [(16,2)]:
            smoke=BASE/'runs'/f'smoke_b{batch}';smoke.mkdir(parents=True,exist_ok=True)
            summary=smoke/'summary.json';log=BASE/f'smoke_b{batch}.log'
            if summary.exists():
                result=json.loads(summary.read_text());chosen=(batch,accum,result);break
            if (smoke/'last.pt').exists():raise RuntimeError('Interrupted smoke checkpoint exists; inspect before repeating')
            status(stage='gpu_smoke',physical_gpu=0,batch=batch,accum=accum,free_mib=memory())
            rc=execute(command(smoke,batch,accum)+['--smoke','--smoke-train','128','--smoke-holdout','64'],log)
            if rc==0:
                result=json.loads(summary.read_text());chosen=(batch,accum,result);break
            if 'out of memory' not in log.read_text().lower():raise RuntimeError('Smoke failed; see '+str(log))
        assert chosen,'GPU smoke failed at all permitted effective-batch-32 sizes'
        batch,accum,result=chosen
        free=memory();peak=result['train_peak_allocated_bytes'];need=int(peak/1048576*1.5)+1024
        assert free>need,f'Insufficient GPU 0 headroom after smoke: {free}MiB free, {need}MiB needed'
        estimate=dict(batch=batch,accum=accum,effective_batch=32,physical_gpu=0,smoke_peak_allocated_bytes=peak,smoke_peak_GB=peak/1e9,smoke_peak_GiB=peak/2**30,free_mib_after_smoke=free,estimated_train_seconds_per_epoch=result['train_wall_seconds']/128*43197,estimate_caveat='128-image shared-GPU smoke estimate only; excludes full holdout and checkpoint I/O, and contention can change')
        (BASE/'smoke_estimate.json').write_text(json.dumps(estimate,ensure_ascii=False,indent=2),encoding='utf-8')
        out=BASE/'runs'/EXP;out.mkdir(parents=True,exist_ok=True)
        cmd=command(out,batch,accum)
        if (out/'last.pt').exists():cmd+=['--resume','auto']
        elif (out/'resolved_config.json').exists():raise RuntimeError('Partial run without checkpoint; inspect rather than overwrite')
        status(stage='full_training',physical_gpu=0,batch=batch,accum=accum,run_dir=str(out),resuming=(out/'last.pt').exists(),smoke=estimate)
        assert execute(cmd,BASE/'training.log')==0,'Training/final evaluation failed; last.pt and logs preserved'
        s=json.loads((out/'summary.json').read_text());assert s['state']=='completed' and s.get('val')
        assert execute([sys.executable,str(HERE/'verify_results.py'),str(out)],BASE/'result_verification.log')==0,'Prediction metric verification failed'
        baseline=ROOT/'experiments/role2_jeongeun_convnext/controlled_effect_20261106/original_only/runs/S2_cnxt_t_originalonly_commonaug_384_s42/summary.json'
        old=json.loads(baseline.read_text()) if baseline.exists() else None
        phase1=ROOT/'experiments/role2_jeongeun_convnext/common506_retrain_20261105/runs/S2_jeongeun_cnxt_t_common506_b_384_s42/summary.json'
        s506=json.loads(phase1.read_text()) if phase1.exists() else None
        lines=['# ConvNeXt V2 공통 데이터 B안 재학습 결과','',f'실험: {EXP}',f'Best epoch {s["best_epoch"]}, 종료 epoch {s["epochs_run"]}. GPU 0, batch {batch} × accumulation {accum}.','', '| 분할 | 실험 | Micro F1 | Macro F1 | Image-wise F1 |','|---|---|---:|---:|---:|']
        for split in ['holdout','val']:
            for label,data in [('원본-only 공통증강 대조',old),('외부506+공통증강+Masked BCE',s506),('외부3458+공통증강+Masked BCE',s)]:
                if data:lines.append(f'| {split} | {label} | '+ ' | '.join(f'{data[split][k]:.6f}' for k in ['micro_f1','macro_f1','image_f1'])+' |')
        lines+=['',f'학습 시간: {s["train_wall_seconds"]/60:.2f}분. 최대 할당 VRAM: {s["train_peak_allocated_bytes"]/1e9:.3f} GB / {s["train_peak_allocated_bytes"]/2**30:.3f} GiB. 공유 GPU에서 측정.', '', 'Holdout Macro F1으로 best를 선택했고 공식 val은 최종 평가에만 사용했다. 동일 공통 규칙의 원본-only 및 506장 실험과 비교한다. V2는 풀 크기에 따른 가중치와 노출 분포도 변하므로 개별 요소의 인과 효과를 단정하지 않는다. 외부 라벨은 사람 전수 검수 자료가 아니다.','', '3개 실험의 동일 평가ID/정답/클래스 및 NPZ 재계산 결과: prediction_verification.json. 전체 지표 비교: comparison_three_runs.csv. 27개 클래스 비교: comparison_27_classes.csv. 클래스별 지표: per_class_holdout.csv, per_class_val.csv. 예측: preds_holdout.npz, preds_val.npz. 가중치: best.pt, last.pt의 model 키. 실제 노출량: actual_epoch_exposure/. 로그: history.csv, learning_curves.png. Git push 미실행.']
        (out/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status(stage='completed',physical_gpu=0,run_dir=str(out),best_epoch=s['best_epoch'],epochs_run=s['epochs_run'],holdout=s['holdout'],val=s['val'])
        print('COMPLETED including final holdout/val evaluation and Korean result summary',flush=True)
    except BaseException as error:
        status(stage='failed',error=str(error),traceback=traceback.format_exc(),physical_gpu=0)
        traceback.print_exc();raise
if __name__=='__main__':main()
