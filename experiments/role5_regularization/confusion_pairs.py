"""Similar-garment confusions of one or more models (preds_*.npz with logits), for the C1 vs C3 head comparison.

    python experiments/role5_regularization/confusion_pairs.py <preds.npz> [<preds.npz> ...] [--thr 0.5]

A confusion A -> B is an image whose ground truth has A but not B, where the model misses A and predicts B
instead (e.g. jumpsuit -> pants). Also prints F1 / recall of the target classes. Files must share images.
"""
import argparse
import numpy as np

PAIRS = [("sweater", "top"), ("vest", "top"), ("vest", "jacket"), ("cardigan", "jacket"), ("cardigan", "coat"),
         ("jumpsuit", "pants"), ("jumpsuit", "top"), ("jumpsuit", "dress"), ("shirt", "top"),
         ("coat", "jacket"), ("jacket", "coat")]
TARGET = ["vest", "sweater", "cardigan", "jumpsuit", "cape", "shirt", "coat", "jacket"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    args, thr = a.preds, a.thr
    ref = None
    rows = []
    for path in args:
        z = np.load(path, allow_pickle=False)
        names = z["filenames"].tolist()
        if ref is None:
            ref = names
        o = [names.index(n) for n in ref] if names != ref else slice(None)
        y = z["targets"][o].astype(bool)
        p = 1 / (1 + np.exp(-z["logits"][o])) >= thr
        cats = [str(c).split(",")[0] for c in z["categories"]]
        j = {c: k for k, c in enumerate(cats)}
        conf = [int((y[:, j[a]] & ~y[:, j[b]] & ~p[:, j[a]] & p[:, j[b]]).sum()) for a, b in PAIRS]
        f1, rec = [], []
        for c in TARGET:
            k = j[c]
            tp, fp, fn = (y[:, k] & p[:, k]).sum(), (~y[:, k] & p[:, k]).sum(), (y[:, k] & ~p[:, k]).sum()
            f1.append(2 * tp / max(1, 2 * tp + fp + fn))
            rec.append(tp / max(1, tp + fn))
        rows.append((path, conf, f1, rec))
    print(f"images {len(ref)}, threshold {thr}")
    print("confusion A->B | " + " | ".join(r[0].split("/")[-2] if "/" in r[0] else r[0] for r in rows))
    for i, (a, b) in enumerate(PAIRS):
        print(f"{a}->{b} | " + " | ".join(str(r[1][i]) for r in rows))
    print("total | " + " | ".join(str(sum(r[1])) for r in rows))
    for i, c in enumerate(TARGET):
        print(f"{c} F1/recall | " + " | ".join(f"{r[2][i]:.3f}/{r[3][i]:.2f}" for r in rows))


if __name__ == "__main__":
    main()
