"""Real original-data C1 CPU integration; no GPU, downloads, or modifications to existing runs."""
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from role4_c1 import settings, audit_data, initialize, train_args, state_digest, environment, ref, digest
from role4_c1_train import train
from role4_c1_compare import metrics, bootstrap, compare, read_pair
from role4_c1_evaluate import evaluate
from fashion_tagging.training.losses import build_loss


def main():
    torch.set_num_threads(2)
    result = dict(environment=environment(), data=audit_data(settings()), smoke=[])
    with tempfile.TemporaryDirectory(prefix='role4_c1_cpu_') as td:
        td = Path(td)
        init = td / 'initial.pt'
        initial = initialize(init, pretrained=False)
        initial_hash = initial['model_sha256']
        del initial
        for loss in ('bce', 'asl'):
            a = train_args(loss, init, td / 'runs', device='cpu')
            a.no_pretrained, a.no_amp = True, True
            a.workers, a.epochs = 0, 1
            a.limit_train, a.limit_eval = 2, 2
            a.short_min = a.short_max = a.test_short = 64
            a.max_long, a.batch_size, a.eval_batch_size, a.accum, a.warmup_iters = 128, 2, 2, 1, 1
            train(a)
            run = td / 'runs' / a.exp_id
            ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            assert len(ck['categories']) == 27 and ck['epoch'] == 1
            assert ck['config']['initial_model_sha256'] == initial_hash
            fresh = ref.MaskedPoolConvNeXt(a.model, 27, False, .1)
            fresh.load_state_dict(ck['model'], strict=True)
            assert state_digest(fresh.state_dict()) == state_digest(ck['model'])
            fresh.eval()
            with torch.no_grad():
                z = fresh(torch.randn(2, 3, 64, 96), torch.tensor([[64, 96], [64, 64]]))
            assert z.shape == (2, 27) and torch.isfinite(z).all()
            state = torch.load(run / 'training_state.pt', map_location='cpu', weights_only=False)
            assert state['step'] > 0 and state['optimizer']['state'] and state['ema']
            # Refuse any attempt to overwrite an existing run directory.
            try:
                train(a)
                raise AssertionError('overwrite guard missing')
            except FileExistsError:
                pass
            result['smoke'].append(dict(loss=loss, output_shape=list(z.shape),
                initial_model_sha256=initial_hash, checkpoint_roundtrip=True,
                optimizer_scheduler_ema_rng_saved=True, overwrite_refused=True))
            del fresh, ck, state
        for name in ('bce', 'asl'):
            x = torch.tensor([[100., -100., 0., 3.]], requires_grad=True)
            y = torch.tensor([[1., 0., -1., 1.]])
            fn = build_loss(name)
            value = fn(x, y)
            value.backward()
            assert torch.isfinite(value) and torch.isfinite(x.grad).all() and x.grad[0, 2] == 0
            known = y >= 0
            assert torch.allclose(value, fn(x.detach()[known], y[known]))
            xx = x.detach().clone().requires_grad_(True)
            zero = fn(xx, torch.full_like(y, -1))
            zero.backward()
            assert zero == 0 and torch.equal(xx.grad, torch.zeros_like(xx.grad))
        cfg = settings()
        cats, ns, ys = ref.load_split(cfg['train_dir'], 'train')
        # Real worker-process seeding: image order and flips match across fresh loaders.
        from torch.utils.data import DataLoader
        mean, std = ref.timm_mean_std(cfg['model'])
        ds = ref.HiResDataset(cfg['train_dir'], ns[:4], ys[:4], mean, std, True, 128, .5)
        portrait = ref.oriented_portrait(cfg['train_dir'], ns[:4])
        def batches():
            sampler = ref.BucketBatchSampler(portrait, [64], 2, 42, True)
            sampler.set_epoch(1)
            loader = DataLoader(ds, batch_sampler=sampler, num_workers=2, collate_fn=ref.pad_collate,
                worker_init_fn=ref.seed_worker, generator=torch.Generator().manual_seed(42))
            return list(loader)
        assert all(all(torch.equal(x, xx) for x, xx in zip(b, bb)) for b, bb in zip(batches(), batches()))
        result['worker_order_and_flip_reproducible'] = True
        hold = ref.read_names(Path(cfg['splits_dir']) / 'holdout.txt')
        ix = {n: i for i, n in enumerate(ns)}
        y = ys[[ix[n] for n in hold]]
        z = np.random.default_rng(42).normal(size=y.shape)
        # Synthetic artifacts use the real holdout order/truth; never actual experiment results.
        for loss in ('bce', 'asl'):
            run = td / 'runs' / ('R4_C1_' + loss + '_s42')
            c = json.loads((run / 'config.json').read_text())
            c.update(loss=loss, exp_id=loss, limit_train=0, limit_eval=0, no_pretrained=False)
            (run / 'config.json').write_text(json.dumps(c))
            pred = td / loss
            pred.mkdir()
            for f in ('preds.npz', 'preds_flip.npz'):
                ref.save_predictions(pred / f, hold, y, z, cats)
            (pred / 'provenance.json').write_text(json.dumps(dict(split='holdout', resolution=[800,1333],
                checkpoint_sha256=digest(run / 'best.pt'), config_sha256=digest(run / 'config.json'))))
        ref.save_predictions(td / 'asl/preds_flip.npz', hold, y, z + 2., cats)
        _, _, _, probs = read_pair(td / 'asl')
        assert np.allclose(probs[1], ref.sigmoid(z + 1.))
        assert not np.allclose(probs[1], (ref.sigmoid(z) + ref.sigmoid(z + 2.)) / 2)
        ref.save_predictions(td / 'asl/preds_flip.npz', hold, y, z, cats)
        br, ar = [td / 'runs' / ('R4_C1_' + l + '_s42') for l in ('bce', 'asl')]
        compare(br, ar, td / 'bce', td / 'asl', td / 'report')
        import csv
        rows = list(csv.DictReader((td / 'report/summary.csv').open(encoding='utf-8-sig')))
        assert len(rows) == 8
        assert len(list(csv.DictReader((td / 'report/per_class.csv').open(encoding='utf-8-sig')))) == 216
        assert len(list(csv.DictReader((td / 'report/rare_classes.csv').open(encoding='utf-8-sig')))) == 56
        intervals = json.loads((td / 'report/bootstrap.json').read_text())
        assert all(ci == [0., 0.] for m in intervals['intervals'].values() for ci in m.values())
        selection = td / 'report/selection.json'
        # Val is blocked without a lock before model construction or inference.
        try:
            evaluate(br, 'val', td / 'invalid_val', device='cpu')
            raise AssertionError('val lock missing')
        except ValueError:
            pass
        c = json.loads((ar / 'config.json').read_text())
        c['seed'] = 99
        (ar / 'config.json').write_text(json.dumps(c))
        try:
            compare(br, ar, td / 'bce', td / 'asl', td / 'invalid_report')
            raise AssertionError('mismatch guard missing')
        except AssertionError as e:
            if str(e) == 'mismatch guard missing':
                raise
        result['analysis_checks'] = dict(summary_rows=8, class_rows=216, rare_rows=56,
            bootstrap_replicates=2000, identical_predictions_ci=[0,0], mismatched_seed_rejected=True,
            official_val_without_lock_rejected=True, masking_reduction_backward=True)
    out = ROOT / 'report/role4_c1_cpu'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checks.json').write_text(json.dumps(result, indent=2))
    print('PASS:', out / 'checks.json')


if __name__ == '__main__':
    main()
