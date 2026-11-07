"""Explicit inference stage. Official val requires a holdout selection lock."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from role4_c1 import ref, digest, state_digest


def evaluate(run, split, output, lock=None, device='cuda', workers=6):
    run, output = Path(run), Path(output)
    ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    cfg = ck['config']
    if split == 'val':
        if lock is None:
            raise ValueError('official val requires --lock from holdout comparison')
        selection = json.loads(Path(lock).read_text())
        loss = cfg['loss']
        assert selection['runs'][loss]['checkpoint_sha256'] == digest(run / 'best.pt')
        assert selection['runs'][loss]['config_sha256'] == digest(run / 'config.json')
    assert cfg == json.loads((run / 'config.json').read_text()), 'checkpoint and run configuration differ'
    root = cfg['train_dir'] if split == 'holdout' else cfg['val_dir']
    cats, names, y = ref.load_split(root, 'train' if split == 'holdout' else 'val')
    assert cats == ck['categories']
    if split == 'holdout':
        chosen = ref.read_names(Path(cfg['splits_dir']) / 'holdout.txt')
        idx = {n: i for i, n in enumerate(names)}
        y, names = y[[idx[n] for n in chosen]], chosen
    assert len(names) == (2425 if split == 'holdout' else 1158)
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model = ref.MaskedPoolConvNeXt(cfg['model'], 27, False, cfg['drop_path'])
    model.load_state_dict(ck['model'], strict=True)
    model.to(dev, memory_format=torch.channels_last)
    por = ref.oriented_portrait(root, names)
    mean, std = ref.timm_mean_std(cfg['model'])
    # Never overwrite an earlier evaluation, even a partial one.
    output.mkdir(parents=True, exist_ok=False)
    for mirror in (False, True):
        ds = ref.HiResDataset(root, names, y, mean, std, False, 1333, 0, mirror=mirror)
        sampler = ref.BucketBatchSampler(por, [800], cfg['eval_batch_size'], cfg['seed'], False)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=workers, collate_fn=ref.pad_collate,
                            worker_init_fn=ref.seed_worker, generator=torch.Generator().manual_seed(42))
        z, seconds = ref.predict(model, loader, dev, dev.type == 'cuda' and not cfg['no_amp'])
        ref.save_predictions(output / ('preds_flip.npz' if mirror else 'preds.npz'), names, y, z, cats)
    (output / 'provenance.json').write_text(json.dumps(dict(split=split, resolution=[800, 1333],
        checkpoint_sha256=digest(run / 'best.pt'), config_sha256=digest(run / 'config.json'),
        state_sha256=state_digest(ck['model']), lock_sha256=digest(lock) if lock else None), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--split', choices=['holdout', 'val'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--lock')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()
    evaluate(a.run, a.split, a.out, a.lock, a.device, a.workers)
