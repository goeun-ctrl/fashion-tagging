"""Predictions of a train_hires.py checkpoint (the final CNN) on the redesign-B judging sets that infer_hires.py does not
cover: crowd 200 (judging only) and neg holdout (empty photos). Same preprocessing as infer_hires.py at one scale,
no flip, so the B runs (800, no flip) compare like for like. Never used for training, thresholds or selection.

    python scripts/infer_hires_extra.py --ckpt D/baseline/best.pt --data D --out D/baseline/tta --device cpu
    (--check N first re-predicts N holdout images and compares them with tta/preds_holdout_s800.npz)

Writes <out>/preds_{crowd,neg_holdout}_s<short>.npz with the same file names as scripts/train_segb.py.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_hires import (BucketBatchSampler, HiResDataset, build_hires_model, oriented_portrait, pad_collate,  # noqa: E402
                         predict, save_predictions, timm_mean_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--short", type=int, default=800)
    ap.add_argument("--max_long", type=int, default=1333)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(st["config"])
    cats = list(st["categories"])
    assert not cfg.get("gate"), "gated checkpoints are not supported here"
    dev = torch.device(a.device)
    model = build_hires_model(cfg, len(cats), False)
    model.load_state_dict(st["model"])
    model = model.to(dev).to(memory_format=torch.channels_last).eval()
    mean, std = timm_mean_std(cfg["model"])
    interp = cfg.get("interp", "bilinear")
    D = a.data

    def run(paths, y):
        ds = HiResDataset("/", paths, y, mean, std, False, a.max_long, 0.0, mirror=False, interp=interp)
        dl = DataLoader(ds, batch_sampler=BucketBatchSampler(oriented_portrait("/", paths), [a.short], a.batch_size, 0,
                                                             False),
                        num_workers=a.workers, collate_fn=pad_collate)
        with torch.no_grad():
            return predict(model, dl, dev, dev.type == "cuda")[0][:, :len(cats)]

    if a.check:
        ref = np.load(os.path.join(D, "baseline", "tta", f"preds_holdout_s{a.short}.npz"))
        n = [str(x) for x in ref["filenames"][:a.check]]
        lg = run([os.path.join(D, "holdout", "images", x) for x in n], ref["targets"][:a.check])
        d = np.abs(lg - ref["logits"][:a.check, :len(cats)])
        print(f"check on {a.check} holdout images: max |logit diff| {d.max():.4f}, mean {d.mean():.5f}", flush=True)

    nd = os.path.join(D, "neg_data")
    neg = [l.strip().replace("/data/eunjin/DL/neg_data", nd) for l in open(os.path.join(nd, "neg_holdout.txt")) if l.strip()]
    crow = list(csv.DictReader(open(os.path.join(D, "crowd_eval", "labels.csv"), encoding="utf-8-sig")))
    sets = {
        "crowd": ([os.path.join(D, "crowd_eval", "images", r["file"]) for r in crow],
                  np.array([[int(r[c]) for c in cats] for r in crow], np.uint8), [r["file"] for r in crow]),
        "neg_holdout": (neg, np.zeros((len(neg), len(cats)), np.uint8), [os.path.basename(p) for p in neg]),
    }
    os.makedirs(a.out, exist_ok=True)
    for split, (paths, y, names) in sets.items():
        lg = run(paths, y)
        f = os.path.join(a.out, f"preds_{split}_s{a.short}.npz")
        save_predictions(f, names, y, lg, cats)
        print(f"wrote {f} ({len(names)} images)", flush=True)


if __name__ == "__main__":
    main()
