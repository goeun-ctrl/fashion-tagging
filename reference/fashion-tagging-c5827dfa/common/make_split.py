"""Create the fixed internal holdout split from the official train set.

The official val set is left untouched (final report only). A small part of
train becomes `holdout`, used for early stopping / threshold choice, so the
official val is not reused for selection.

Rare classes are filled first so that every class gets about HOLDOUT_FRAC of
its positives in the holdout (a plain random 5% would leave ~3 leg warmers).

Usage:  python common/make_split.py --train_zip MAT4091_fashion_train.zip
Outputs (common/splits/): train_inner.txt, holdout.txt, split_summary.csv
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import load_split_from_zip  # noqa: E402

HOLDOUT_FRAC = 0.05
SEED = 42


def make_split(targets, frac=HOLDOUT_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(targets)
    in_holdout = np.zeros(n, dtype=bool)
    support = targets.sum(0)
    for c in np.argsort(support, kind="stable"):  # rarest class first
        want = math.ceil(frac * support[c])
        have = int(targets[in_holdout, c].sum())
        if have >= want:
            continue
        pool = np.flatnonzero((targets[:, c] == 1) & ~in_holdout)
        take = rng.choice(pool, size=min(want - have, len(pool)), replace=False)
        in_holdout[take] = True
    short = round(frac * n) - int(in_holdout.sum())
    if short > 0:
        pool = np.flatnonzero(~in_holdout)
        in_holdout[rng.choice(pool, size=short, replace=False)] = True
    return in_holdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_zip", required=True)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits"))
    a = ap.parse_args()

    cats, names, targets = load_split_from_zip(a.train_zip, "train")
    names = np.asarray(names)
    mask = make_split(targets)
    os.makedirs(a.out_dir, exist_ok=True)
    for fname, sel in (("train_inner.txt", ~mask), ("holdout.txt", mask)):
        with open(os.path.join(a.out_dir, fname), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(names[sel]) + "\n")
    with open(os.path.join(a.out_dir, "split_summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "train_inner", "holdout", "holdout_pct"])
        for i, c in enumerate(cats):
            tr, ho = int(targets[~mask, i].sum()), int(targets[mask, i].sum())
            w.writerow([c, tr, ho, f"{100 * ho / (tr + ho):.1f}"])
    print(f"seed={SEED} train_inner={int((~mask).sum())} holdout={int(mask.sum())} -> {a.out_dir}")


if __name__ == "__main__":
    main()
