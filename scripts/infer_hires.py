"""Evaluate a train_hires.py checkpoint on holdout and val at several test scales, with and without
horizontal-flip TTA (mean of the two logits). No training. Thresholds: one global value, optionally with
per-class overrides (e.g. 0.60 everywhere, 0.5 for sweater / leg warmer / umbrella).

    python scripts/infer_hires.py --run_dir /data/eunjin/DL/runs_hr/R5_cnxt_hr_s42 \
        --scales 800:1333 896:1493 960:1600 --threshold 0.6 \
        --class_threshold "sweater=0.5" "leg warmer=0.5" "umbrella=0.5"

--scales takes test_short:max_long pairs. With max_long fixed at 1333, a 960 short side is mostly capped by
the long side for 2:3 portraits (the bulk of the data), so the long cap is scaled with the short side.
Writes <run_dir>/tta/preds_<split>_s<short>[_flip].npz and <run_dir>/tta/summary.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_hires import (BucketBatchSampler, binarize, HiResDataset, build_hires_model, gate_split, gated, compute_metrics, load_split,  # noqa: E402
                         oriented_portrait, pad_collate, predict, read_names, save_predictions, sigmoid,
                         timm_mean_std)


def thresholds(cats, a):
    t = np.full(len(cats), a.threshold, dtype=np.float64)
    for spec in a.class_threshold:
        name, v = spec.rsplit("=", 1)
        hit = [j for j, c in enumerate(cats) if c == name or c.split(",")[0].strip() == name]
        assert len(hit) == 1, f"class {name!r} not found in {cats}"
        t[hit[0]] = float(v)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--scales", nargs="+", default=["800:1333"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--class_threshold", nargs="*", default=[])
    ap.add_argument("--gate_threshold", type=float, default=0.5, help="only for checkpoints trained with --gate")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit_eval", type=int, default=0)
    for k in ("train_dir", "val_dir", "splits_dir"):
        ap.add_argument(f"--{k}", default=None, help="override the path stored in the checkpoint config")
    a = ap.parse_args()

    st = torch.load(os.path.join(a.run_dir, a.ckpt), map_location="cpu")
    cfg = dict(st["config"])
    cfg.update({k: getattr(a, k) for k in ("train_dir", "val_dir", "splits_dir") if getattr(a, k)})
    device = torch.device("cuda")
    n_cls = len(st["categories"])
    model = build_hires_model(cfg, n_cls + (1 if cfg.get("gate") else 0), False).to(device)
    model.load_state_dict(st["model"])
    model = model.to(memory_format=torch.channels_last).eval()
    mean, std = timm_mean_std(cfg["model"])

    cats, names, targets = load_split(cfg["train_dir"], "train")
    vcats, vnames, vtargets = load_split(cfg["val_dir"], "val")
    assert cats == vcats == list(st["categories"])
    index = {n: i for i, n in enumerate(names)}
    hold = read_names(os.path.join(cfg["splits_dir"], "holdout.txt"))
    if a.limit_eval:
        hold, vnames, vtargets = hold[:a.limit_eval], vnames[:a.limit_eval], vtargets[:a.limit_eval]
    splits = {"holdout": (cfg["train_dir"], hold, targets[[index[n] for n in hold]]),
              "val": (cfg["val_dir"], vnames, vtargets)}
    por = {k: oriented_portrait(r, n) for k, (r, n, _) in splits.items()}
    thr = thresholds(cats, a)
    out_dir = os.path.join(a.run_dir, "tta")
    os.makedirs(out_dir, exist_ok=True)

    summary = {"ckpt": a.ckpt, "epoch": st["epoch"], "threshold": a.threshold,
               "class_threshold": a.class_threshold, "results": []}
    for sc in a.scales:
        short, max_long = (int(v) for v in sc.split(":"))
        for split, (root, n, y) in splits.items():
            lg = {}
            for mirror in (False, True):
                ds = HiResDataset(root, n, y, mean, std, False, max_long, 0.0, mirror=mirror)
                dl = DataLoader(ds, batch_sampler=BucketBatchSampler(por[split], [short], a.batch_size, 0, False),
                                num_workers=a.workers, pin_memory=True, collate_fn=pad_collate)
                lg[mirror], fwd = predict(model, dl, device, True)
            for tag, l in (("", lg[False]), ("_flip", (lg[False] + lg[True]) / 2)):
                l, g = gate_split(l, n_cls)
                save_predictions(os.path.join(out_dir, f"preds_{split}_s{short}{tag}.npz"), n, y, l, cats)
                if g is not None:
                    np.savez(os.path.join(out_dir, f"gate_{split}_s{short}{tag}.npz"), filenames=np.array(n), gate_logits=g)
                p = sigmoid(l)
                row = {"split": split, "short": short, "max_long": max_long, "flip_tta": bool(tag),
                       "ms_per_image_one_pass": 1000 * fwd / len(n)}
                for name, t in (("t0.5", np.full(len(cats), 0.5)), ("t_global", np.full(len(cats), a.threshold)),
                                ("t_class", thr)):
                    m = compute_metrics(y, gated(binarize(p, t), g, a.gate_threshold))
                    row[name] = {k: round(float(m[k]), 4) for k in ("micro_f1", "macro_f1", "image_f1")}
                summary["results"].append(row)
                print(json.dumps(row), flush=True)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
