"""Error Analysis Sheet for the provided OOD sets + comparison of gate decision rules.

Inputs (from scripts/eval_ood.py and scripts/tag_ood_images.py):
  <dir>/<model>_examples.npz, <model>_test.npz   probs [N, 27], gate_probs [N] (NaN without a gate), targets
  <dir>/image_tags.csv                             type (crowd / group / single / small_person / bw / no_person) ...

    python experiments/role5_regularization/error_sheet.py <dir> --models c1 kd c3 v2 --out <out_dir>

Writes <out_dir>/examples_sheet.csv (one row per image: type, true/pred counts, missed and false labels per
model), test_sheet.csv (type, predictions per model and gate rule) and prints the summaries.

Gate rules (for a model with a gate):
  hard        gate < 0.5 -> empty (training default)
  conserv     gate < 0.05 -> empty
  soft        p'_c = sigmoid(z_c + log g): the gate lowers every class score, a strong class still survives
  none        class scores only
On test_public there are no labels; an empty prediction on an image where the detector sees a person covering
>= 2% of the image is counted as "likely wrong" (a proxy, checked by eye on a sample).
"""
import argparse
import csv
import json
import os

import numpy as np

SPECIAL = ("sweater", "leg warmer", "umbrella")


def thresholds(cats, base=0.6):
    t = np.full(len(cats), base)
    t[[i for i, c in enumerate(cats) if c.startswith(SPECIAL)]] = 0.5
    return t


def rules(p, g, t):
    out = {"none": p >= t}
    if g is not None and not np.isnan(g).all():
        out["hard"] = (p >= t) & (g >= 0.5)[:, None]
        out["conserv"] = (p >= t) & (g >= 0.05)[:, None]
        z = np.log(p / (1 - p + 1e-9) + 1e-9) + np.log(g + 1e-9)[:, None]
        out["soft"] = 1 / (1 + np.exp(-z)) >= t
    return out


def f1(y, P):
    tp, fp, fn = (y & P).sum(), (~y & P).sum(), (y & ~P).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn), tp / max(1, tp + fn), tp / max(1, tp + fp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--models", nargs="+", default=["c1", "kd", "c3", "v2"])
    ap.add_argument("--thr", type=json.loads, default='{"c1": 0.6, "kd": 0.55, "c3": 0.65, "v2": 0.6}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tags = {(r["set"], r["file_name"]): r for r in csv.DictReader(open(os.path.join(a.dir, "image_tags.csv")))}

    for split, setname in (("examples", "examples_labeled"), ("test", "test_public")):
        data = {m: np.load(os.path.join(a.dir, f"{m}_{split}.npz"), allow_pickle=False) for m in a.models}
        ref = data[a.models[0]]
        names = ref["filenames"].tolist()
        cats = [str(c) for c in ref["categories"]]
        short = [c.split(",")[0] for c in cats]
        y = ref["targets"].astype(bool)
        preds = {}
        for m, z in data.items():
            assert z["filenames"].tolist() == names
            g = z["gate_probs"] if "gate_probs" in z.files else None
            for r, P in rules(z["probs"], g, thresholds(cats, a.thr.get(m, 0.6))).items():
                preds[f"{m}:{r}" if r != "none" or g is not None and not np.isnan(g).all() else m] = P
        rows = []
        for i, n in enumerate(names):
            t = tags[(setname, n)]
            row = {"file_name": n, "type": t["type"], "n_person": t["n_person"], "max_person_share": t["max_person_share"],
                   "gray": t["gray"], "true_n": int(y[i].sum())}
            for k, P in preds.items():
                row[f"{k}_n"] = int(P[i].sum())
                if split == "examples":
                    row[f"{k}_missed"] = "|".join(short[j] for j in np.flatnonzero(y[i] & ~P[i]))
                    row[f"{k}_false"] = "|".join(short[j] for j in np.flatnonzero(~y[i] & P[i]))
            rows.append(row)
        with open(os.path.join(a.out, f"{split}_sheet.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

        types = np.array([tags[(setname, n)]["type"] for n in names])
        share = np.array([float(tags[(setname, n)]["max_person_share"]) for n in names])
        print(f"\n===== {split} ({len(names)}) types:", {k: int((types == k).sum()) for k in np.unique(types)})
        if split == "examples":
            print("model | F1 / recall / precision | pred per image (true %.2f) | count MAE | empty" % y.sum(1).mean())
            for k, P in preds.items():
                f, r, p = f1(y, P)
                print(f"{k:12s} | {f:.3f} / {r:.3f} / {p:.3f} | {P.sum(1).mean():.2f} | "
                      f"{np.abs(P.sum(1) - y.sum(1)).mean():.2f} | {int((~P.any(1)).sum())}")
            for ty in np.unique(types):
                sel = types == ty
                print(f"  type {ty} ({sel.sum()}): " + ", ".join(
                    f"{k} R {f1(y[sel], P[sel])[1]:.2f}" for k, P in preds.items() if ":" not in k or k.endswith("hard")))
            P = preds[a.models[0]]
            miss = (y & ~P).sum(0)
            print("most missed (%s): " % a.models[0] + ", ".join(f"{short[j]} {miss[j]}/{y[:, j].sum()}" for j in np.argsort(-miss)[:10]))
        else:
            print("model | pred/img | empty | empty on no_person (of %d) | likely-wrong empty (person >= 2%%)" % (types == "no_person").sum())
            for k, P in preds.items():
                emp = ~P.any(1)
                print(f"{k:12s} | {P.sum(1).mean():.2f} | {int(emp.sum()):3d} | {int((emp & (types == 'no_person')).sum()):2d} | "
                      f"{int((emp & (share >= 0.02)).sum()):2d}")
            for ty in np.unique(types):
                sel = types == ty
                print(f"  type {ty} ({sel.sum()}): pred/img " + ", ".join(
                    f"{k} {preds[k][sel].sum(1).mean():.1f}" for k in preds if ":" not in k or k.endswith("hard")))


if __name__ == "__main__":
    main()
