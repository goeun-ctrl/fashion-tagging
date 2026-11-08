"""A train_hires checkpoint on the provided OOD sets: examples_labeled (30, labels) and test_public (300, no
labels). Flip TTA at 800/1333, optional gate. Prints F1 on the examples and prediction statistics on the test
set, and writes a submission-format CSV for test_public (labels as a JSON list, like examples_labeled).

    python scripts/eval_ood.py --ckpt runs_hr/R5_cnxt_hr_s42/best.pt --data_root /data/eunjin/DL/test_data \
        --out runs_hr/R5_cnxt_hr_s42/ood --threshold 0.6 --class_threshold "sweater=0.5" "leg warmer=0.5" "umbrella=0.5"

The 30 examples are hand-picked hard cases (README: excluded from private scoring); use them as a sanity check,
never to choose thresholds.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_hires import thresholds  # noqa: E402
from train_hires import (BucketBatchSampler, HiResDataset, binarize, build_hires_model, compute_metrics,  # noqa: E402
                         gate_split, gated, oriented_portrait, pad_collate, predict, sigmoid, timm_mean_std)

SETS = (("examples", "examples_labeled", "labels.csv"), ("test", "test_public", "submission_template.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="/data/eunjin/DL/test_data")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--class_threshold", nargs="*", default=[])
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--short", type=int, default=800)
    ap.add_argument("--max_long", type=int, default=1333)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    cfg, cats = st["config"], list(st["categories"])
    n_cls = len(cats)
    dev = torch.device("cuda")
    model = build_hires_model(cfg, n_cls + (1 if cfg.get("gate") else 0), False).to(dev)
    model.load_state_dict(st["model"])
    model = model.to(memory_format=torch.channels_last).eval()
    mean, std = timm_mean_std(cfg["model"])
    thr = thresholds(cats, a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    summary = {"ckpt": a.ckpt, "epoch": st["epoch"], "threshold": a.threshold, "class_threshold": a.class_threshold,
               "gate_threshold": a.gate_threshold if cfg.get("gate") else None}

    for name, folder, csv_name in SETS:
        root = os.path.join(a.data_root, folder)
        rows = list(csv.DictReader(open(os.path.join(root, csv_name), encoding="utf-8-sig")))
        names = [r["file_name"] for r in rows]
        y = np.array([[int(r.get(c) or 0) for c in cats] for r in rows], np.float32)
        por = oriented_portrait(root, names)
        lg = []
        for mirror in (False, True):
            ds = HiResDataset(root, names, y, mean, std, False, a.max_long, 0.0, mirror=mirror)
            dl = DataLoader(ds, batch_sampler=BucketBatchSampler(por, [a.short], 8, 0, False), num_workers=4,
                            collate_fn=pad_collate)
            lg.append(predict(model, dl, dev, True)[0])
        cls, g = gate_split((lg[0] + lg[1]) / 2, n_cls)
        p = sigmoid(cls)
        pred = gated(binarize(p, thr), g, a.gate_threshold)
        np.savez(f"{a.out}_{name}.npz", filenames=np.array(names), targets=y, probs=p,
                 gate_probs=sigmoid(g) if g is not None else np.full(len(names), np.nan), categories=np.array(cats))
        s = {"n": len(names), "pred_per_image": float(pred.sum(1).mean()), "empty": int((~pred.any(1)).sum())}
        if name == "examples":
            m = compute_metrics(y, pred)
            s.update({k: float(m[k]) for k in ("micro_f1", "macro_f1", "image_f1")},
                     true_per_image=float(y.sum(1).mean()), true_empty=int((y.sum(1) == 0).sum()))
        else:
            with open(f"{a.out}_submission.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["test_id", "file_name", "labels"])
                for r, pr in zip(rows, pred):
                    w.writerow([r["test_id"], r["file_name"], json.dumps([c for c, k in zip(cats, pr) if k])])
            s["class_rate"] = {c.split(",")[0]: round(float(pred[:, j].mean()), 3) for j, c in enumerate(cats)}
        summary[name] = s
        print(name, json.dumps({k: v for k, v in s.items() if k != "class_rate"}), flush=True)
    json.dump(summary, open(f"{a.out}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
