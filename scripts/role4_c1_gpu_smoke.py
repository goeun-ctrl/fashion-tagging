"""GPU preflight isolated from full runs; preserves the shared pretrained initial file."""
import gc
import json
import os
from pathlib import Path
import torch
from timm.utils import ModelEmaV3
from role4_c1 import ROOT, ref, digest, state_digest, train_args, environment
from role4_c1_train import train


def main():
    torch.set_num_threads(4)
    assert torch.cuda.device_count() == 1, 'exactly one explicitly visible GPU required'
    assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
    out = ROOT / 'runs/role4_c1_gpu_smoke'
    out.mkdir(parents=True, exist_ok=False)
    initial = ROOT / 'runs/role4_c1/shared_initial_s42.pt'
    before = digest(initial)
    report = dict(initial_sha256_before=before, gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  device=torch.cuda.get_device_name(0), environment=environment(), tests=[])
    for loss in ('bce', 'asl'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        a = train_args(loss, initial, out)
        a.epochs, a.limit_train, a.limit_eval, a.workers = 1, 64, 16, 2
        train(a)
        ck = torch.load(out / a.exp_id / 'best.pt', map_location='cpu', weights_only=False)
        model = ref.MaskedPoolConvNeXt(a.model, 27, False, .1).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(ck['model'], strict=True)
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            z = model(torch.zeros(1,3,800,1344,device='cuda').to(memory_format=torch.channels_last),
                      torch.tensor([[800,1333]],device='cuda'))
        assert z.shape == (1,27) and torch.isfinite(z).all()
        state = torch.load(out / a.exp_id / 'training_state.pt', map_location='cpu', weights_only=False)
        assert state['step'] > 0 and state['optimizer']['state'], 'need a successful AMP optimizer update'
        item = dict(loss=loss, train_images=64, holdout_images=16, updates=state['step'],
                    amp_skipped=ck['config'].get('amp_skipped'), checkpoint_reload=True,
                    train_eval_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    train_eval_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        del model, ck, state, z
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        # Worst C1 canvas: full batch8, portrait1344x800, optimizer state + EMA + AMP backward.
        st = torch.load(initial, map_location='cpu', weights_only=False)
        model = ref.MaskedPoolConvNeXt(a.model, 27, False, .1).cuda().to(memory_format=torch.channels_last)
        model.load_state_dict(st['model'], strict=True)
        ema = ModelEmaV3(model, decay=.9995, use_warmup=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05)
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        x = torch.zeros(8,3,1344,800,device='cuda').to(memory_format=torch.channels_last)
        sizes = torch.tensor([[1333,800]]*8,device='cuda')
        y = torch.zeros(8,27,device='cuda'); y[:,0] = 1
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                z = model(x,sizes)
            value = ref.build_loss(loss)(z.float(),y)
            assert torch.isfinite(value)
            scaler.scale(value).backward(); scaler.unscale_(opt)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt); scaler.update(); ema.update(model)
        torch.cuda.synchronize()
        item.update(stress_shape=[8,3,1344,800],
                    stress_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    stress_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        report['tests'].append(item)
        del model, ema, opt, scaler, x, sizes, y, z, value, st
        gc.collect(); torch.cuda.empty_cache()
        assert digest(initial) == before
        print(json.dumps(item),flush=True)
    st = torch.load(initial,map_location='cpu',weights_only=False)
    assert st['pretrained'] and state_digest(st['model']) == st['model_sha256']
    assert digest(initial) == before
    report.update(initial_sha256_after=digest(initial), initial_preserved=True, passed=True)
    (out / 'checks.json').write_text(json.dumps(report,indent=2))
    print('GPU SMOKE PASS',flush=True)


if __name__ == '__main__':
    main()
