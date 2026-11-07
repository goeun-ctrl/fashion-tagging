"""Matched C1 experiment utilities. GPU work is only done by explicit train/evaluate commands."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import role4_c1_reference as ref
from fashion_tagging.data.labels import load_split

SETTINGS = ROOT / 'experiments/role4_c1/settings.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def environment():
    return dict(python=platform.python_version(), command=sys.argv,
                cuda_build=torch.version.cuda,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                libraries={k: importlib.metadata.version(k)
                for k in ('torch', 'torchvision', 'timm', 'numpy', 'Pillow')},
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                    [*sorted((ROOT / 'scripts').glob('role4_c1*.py')), SETTINGS,
                     ROOT / 'src/fashion_tagging/training/losses.py',
                     ROOT / 'src/fashion_tagging/data/labels.py',
                     ROOT / 'src/fashion_tagging/data/preprocess.py',
                     ROOT / 'src/fashion_tagging/evaluation/metrics.py']})


def settings():
    return json.loads(SETTINGS.read_text())


def audit_data(cfg, check_images=True):
    cats, names, y = load_split(cfg['train_dir'], 'train')
    vc, vn, vy = load_split(cfg['val_dir'], 'val')
    inner, hold = [ref.read_names(Path(cfg['splits_dir']) / (s + '.txt')) for s in ('train_inner', 'holdout')]
    assert (len(inner), len(hold), len(vn)) == (43197, 2425, 1158)
    assert len(names) == len(set(names)) == 45622 and len(vn) == len(set(vn))
    assert len(inner) == len(set(inner)) and len(hold) == len(set(hold))
    assert not set(inner) & set(hold) and set(inner) | set(hold) == set(names)
    assert not set(names) & set(vn) and cats == vc and len(cats) == 27
    assert np.isin(y, (0, 1)).all() and np.isin(vy, (0, 1)).all()
    if check_images:
        for root, ns in ((cfg['train_dir'], names), (cfg['val_dir'], vn)):
            missing = [n for n in ns if not (Path(root) / 'images' / n).is_file()]
            assert not missing, (root, missing[:3])
    paths = [Path(cfg['splits_dir']) / (s + '.txt') for s in ('train_inner', 'holdout')]
    paths += [Path(cfg[d]) / f for d in ('train_dir', 'val_dir') for f in ('categories.txt', 'labels.csv')]
    index = {n: i for i, n in enumerate(names)}
    return dict(counts=dict(train_inner=len(inner), holdout=len(hold), val=len(vn)), categories=cats,
                positive_support=dict(train_inner=y[[index[n] for n in inner]].sum(0).tolist(),
                                      holdout=y[[index[n] for n in hold]].sum(0).tolist(),
                                      val=vy.sum(0).tolist()),
                sha256={str(p): digest(p) for p in paths},
                labels_sha256=dict(train=hashlib.sha256(y.tobytes()).hexdigest(),
                                   val=hashlib.sha256(vy.tobytes()).hexdigest()))


def initialize(path, pretrained=True):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    cfg = settings()
    ref.set_seed(cfg['seed'])
    model = ref.MaskedPoolConvNeXt(cfg['model'], 27, pretrained, cfg['drop_path'])
    state = model.state_dict()
    payload = dict(model=state, model_sha256=state_digest(state),
                   head_sha256=state_digest(model.net.head.fc.state_dict()),
                   pretrained=pretrained, seed=cfg['seed'], model_name=cfg['model'],
                   source='timm pretrained config: ' + cfg['model'],
                   pretrained_cfg=model.net.pretrained_cfg, environment=environment())
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects an existing initialization against concurrent commands.
    with path.open('xb') as f:
        torch.save(payload, f)
    return payload


def train_args(loss, initial, out, device='cuda'):
    cfg = settings()
    if any(k in cfg for k in ('distill_npz', 'init_ckpt', 'extra_train', 'gate', 'head',
                             'mosaic_p', 'zoom_p', 'old_p', 'gray_p', 'mask_dir')):
        raise ValueError('role4 C1 comparison excludes external data, distillation and alternate heads/augmentation')
    argv = ['--exp_id', 'R4_C1_' + loss + '_s42', '--out_dir', str(out), '--loss', loss]
    for k, v in cfg.items():
        argv += ['--' + k] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
    argv += ['--device', device]
    a = ref.parse_args(argv)
    a.initial_state = str(initial)
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('init')
    q.add_argument('--output', default=str(ROOT / 'runs/role4_c1/shared_initial_s42.pt'))
    q = sub.add_parser('train')
    q.add_argument('--loss', choices=('bce', 'asl'), required=True)
    q.add_argument('--initial', default=str(ROOT / 'runs/role4_c1/shared_initial_s42.pt'))
    q.add_argument('--out', default=str(ROOT / 'runs/role4_c1'))
    q.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.action == 'init':
        s = initialize(a.output)
        print(json.dumps({k: s[k] for k in ('model_sha256', 'head_sha256', 'source')}))
    else:
        from role4_c1_train import train
        train(train_args(a.loss, a.initial, a.out, a.device))


if __name__ == '__main__':
    main()
