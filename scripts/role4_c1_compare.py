"""Compare paired predictions; select global thresholds on holdout, freeze before official val."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from role4_c1 import digest, audit_data, settings
from fashion_tagging.evaluation.metrics import load_predictions, sigmoid

RARE = {'cardigan', 'vest', 'jumpsuit', 'cape', 'glove', 'leg warmer', 'umbrella'}
GRID = np.round(np.arange(.05, .951, .05), 2)
MODES = ('baseline', 'tuned', 'tta_05', 'tta_tuned')


def metrics(y, pred):
    known, truth = y >= 0, y == 1
    tp = (truth & pred & known).sum(0)
    fp = (~truth & pred & known).sum(0)
    fn = (truth & ~pred & known).sum(0)
    div = lambda x, d: np.divide(x, d, out=np.zeros_like(x, dtype=float), where=d != 0)
    p, r, f = div(tp, tp + fp), div(tp, tp + fn), div(2 * tp, 2 * tp + fp + fn)
    itp = (truth & pred & known).sum(1)
    den = ((truth | pred) & known).sum(1) + itp
    image = np.divide(2 * itp, den, out=np.ones_like(itp, dtype=float), where=den != 0)
    summary = dict(macro_f1=float(f.mean()), micro_f1=float(div(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum())),
        image_f1=float(image.mean()), macro_precision=float(p.mean()), macro_recall=float(r.mean()),
        micro_precision=float(div(tp.sum(), (tp + fp).sum())), micro_recall=float(div(tp.sum(), (tp + fn).sum())))
    return summary, dict(f1=f, precision=p, recall=r, support=(truth & known).sum(0))


def bootstrap(y, pb, pa, n=2000):
    # Identical image indices for BCE/ASL; thresholds stay fixed throughout resampling.
    rng = np.random.default_rng(42)
    keys = ('macro_f1', 'micro_f1', 'image_f1')
    deltas = []
    for _ in range(n):
        idx = rng.integers(len(y), size=len(y))
        b, _ = metrics(y[idx], pb[idx])
        a, _ = metrics(y[idx], pa[idx])
        deltas.append([a[k] - b[k] for k in keys])
    return {k: list(np.percentile(np.asarray(deltas)[:, i], [2.5, 97.5])) for i, k in enumerate(keys)}


def read_pair(path):
    path = Path(path)
    n, y, z, cats = load_predictions(path / 'preds.npz')
    nn, yy, zz, cc = load_predictions(path / 'preds_flip.npz')
    assert np.array_equal(n, nn) and np.array_equal(y, yy) and cats == cc
    assert y.shape == z.shape == zz.shape and y.shape[1] == 27 and len(n) == len(set(n.tolist()))
    assert np.isin(y, (-1, 0, 1)).all() and np.isfinite(z).all() and np.isfinite(zz).all()
    prob = sigmoid(z)
    return n, y, cats, (prob, sigmoid((z.astype(np.float64) + zz) / 2))


def write_csv(path, records):
    with Path(path).open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)


def compare(bce_run, asl_run, bce_preds, asl_preds, output, lock=None):
    runs = dict(bce=Path(bce_run), asl=Path(asl_run))
    configs = {k: json.loads((v / 'config.json').read_text()) for k, v in runs.items()}
    ignored = {'loss', 'exp_id', 'out_dir', 'environment', 'initial_state'}
    b, a = configs.values()
    assert b['loss'] == 'bce' and a['loss'] == 'asl'
    assert {k: v for k, v in b.items() if k not in ignored} == {k: v for k, v in a.items() if k not in ignored}
    assert b['environment']['libraries'] == a['environment']['libraries']
    assert b['environment']['source_sha256'] == a['environment']['source_sha256']
    assert b['initial_model_sha256'] == a['initial_model_sha256']
    assert b['initial_head_sha256'] == a['initial_head_sha256']
    assert not b['limit_train'] and not b['limit_eval'] and not b['no_pretrained']
    pairs = {k: read_pair(v) for k, v in dict(bce=bce_preds, asl=asl_preds).items()}
    n, y, cats, _ = pairs['bce']
    nn, yy, cc, _ = pairs['asl']
    assert np.array_equal(n, nn) and np.array_equal(y, yy) and cats == cc
    split = 'val' if lock else 'holdout'
    assert len(n) == (1158 if lock else 2425)
    # Confirm full original truth and order, not merely agreement between the two prediction files.
    from fashion_tagging.data.labels import load_split
    tc, tn, ty = load_split(b['val_dir'] if lock else b['train_dir'], 'val' if lock else 'train')
    from role4_c1 import ref
    expected = tn if lock else ref.read_names(Path(b['splits_dir']) / 'holdout.txt')
    idx = {v: i for i, v in enumerate(tn)}
    assert list(n) == expected and cats == tc and np.array_equal(y, ty[[idx[v] for v in expected]])
    for loss, path in dict(bce=bce_preds, asl=asl_preds).items():
        pr = json.loads((Path(path) / 'provenance.json').read_text())
        assert pr['split'] == split and pr['resolution'] == [800, 1333]
        assert pr['checkpoint_sha256'] == digest(runs[loss] / 'best.pt')
        assert pr['config_sha256'] == digest(runs[loss] / 'config.json')
        if lock:
            assert pr['lock_sha256'] == digest(lock)
    selection = json.loads(Path(lock).read_text()) if lock else dict(grid=GRID.tolist(), runs={})
    if lock:
        for loss, run in runs.items():
            assert selection['runs'][loss]['checkpoint_sha256'] == digest(run / 'best.pt')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, classes, predictions = [], [], {}
    for loss, (_, _, _, probabilities) in pairs.items():
        if not lock:
            thresholds = [float(max(GRID, key=lambda t: metrics(y, probs >= t)[0]['macro_f1']))
                          for probs in probabilities]  # grid ascending: ties choose lower threshold
            selection['runs'][loss] = dict(threshold=thresholds[0], tta_threshold=thresholds[1],
                checkpoint_sha256=digest(runs[loss] / 'best.pt'), config_sha256=digest(runs[loss] / 'config.json'))
        chosen = selection['runs'][loss]
        for mode, probs, t in zip(MODES, [probabilities[0], probabilities[0], probabilities[1], probabilities[1]],
                                  [.5, chosen['threshold'], .5, chosen['tta_threshold']]):
            pred = probs >= t
            predictions[loss, mode] = pred
            m, pc = metrics(y, pred)
            rows.append(dict(split=split, loss=loss, mode=mode, threshold=t, **m))
            for i, cat in enumerate(cats):
                classes.append(dict(split=split, loss=loss, mode=mode, category=cat, rare=cat in RARE,
                                    **{k: float(v[i]) if k != 'support' else int(v[i]) for k, v in pc.items()}))
    intervals = {mode: bootstrap(y, predictions['bce', mode], predictions['asl', mode]) for mode in MODES}
    comparison = []
    for mode in MODES:
        bm, am = [next(r for r in rows if r['loss'] == loss and r['mode'] == mode) for loss in ('bce', 'asl')]
        comparison.append(dict(mode=mode, bce_macro_f1=bm['macro_f1'], asl_macro_f1=am['macro_f1'],
            delta_asl_minus_bce=am['macro_f1']-bm['macro_f1'], ci_low=intervals[mode]['macro_f1'][0],
            ci_high=intervals[mode]['macro_f1'][1]))
    write_csv(output / 'summary.csv', rows)
    write_csv(output / 'comparison.csv', comparison)
    write_csv(output / 'per_class.csv', classes)
    write_csv(output / 'rare_classes.csv', [r for r in classes if r['rare']])
    (output / 'bootstrap.json').write_text(json.dumps(dict(replicates=2000, seed=42,
        delta='ASL minus BCE', intervals=intervals,
        limitation='Conditional on fixed trained models and holdout-selected thresholds; excludes seed variance and selection uncertainty.'), indent=2))
    if not lock:
        base = [r for r in rows if r['mode'] == 'baseline']
        selection['selected_loss'] = max(base, key=lambda r: r['macro_f1'])['loss']
        selection['criterion'] = 'holdout baseline macro F1, threshold .5, no TTA; single seed provisional'
        (output / 'selection.json').write_text(json.dumps(selection, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bce_run', 'asl_run', 'bce_preds', 'asl_preds', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--lock', help='holdout selection.json; when supplied, report official val without tuning')
    a = p.parse_args()
    compare(a.bce_run, a.asl_run, a.bce_preds, a.asl_preds, a.out, a.lock)
