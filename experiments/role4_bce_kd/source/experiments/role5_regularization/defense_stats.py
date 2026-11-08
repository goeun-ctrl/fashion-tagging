"""Numbers for the mid-term defence sheet (docs/plan/교수지적_대비.md): where does a macro change come from, and
is a head difference larger than the seed difference?

Each run is <dir>:<threshold>; predictions are <runs_root>/<dir>/tta/preds_<split>_s800_flip.npz (infer_hires.py).
Classes listed in --special keep threshold 0.5 (same rule as the reported results).

  macro A B        per-class F1 of B - A, bootstrap CI of the macro / micro / image difference, the part of the
                   macro change carried by classes with val support <= --tiny, and macro without those classes
  pick A ...       holdout-only threshold choice: global threshold 0.05..0.95 (step 0.05, --special classes
                   stay 0.5) with the best holdout macro F1 (ties -> micro); prints the val numbers at that threshold
  seeds A1 A2 B1 B2  head gap (mean of B - mean of A) next to seed gaps (A2 - A1, B2 - B1), all with CIs

    python experiments/role5_regularization/defense_stats.py macro R5_cnxt_hr_s42:0.60 R5_cnxt_hr_kd_msmax_s42:0.60
    python experiments/role5_regularization/defense_stats.py pick R5_new_s42 R5_other_s42
    python experiments/role5_regularization/defense_stats.py seeds R5_final_msmax_s42:0.60 R5_final_msmax_s43:0.60 \
        R5_final_csra_s42:0.50 R5_final_csra_s43:0.50
"""
import argparse
import json
import os

import numpy as np


def load(root, spec, split, special):
    d, t = spec.rsplit(":", 1)
    z = np.load(os.path.join(root, d, "tta", f"preds_{split}_s800_flip.npz"), allow_pickle=False)
    cats = [str(c) for c in z["categories"]]
    thr = np.full(len(cats), float(t))
    thr[[i for i, c in enumerate(cats) if c.startswith(tuple(special))]] = 0.5
    logits = z["logits"][:, :len(cats)]                 # a gate column, if any, is dropped
    return d, z["filenames"], z["targets"].astype(bool), 1 / (1 + np.exp(-logits)) >= thr, cats


def counts(y, p):
    return (y & p).sum(0), (~y & p).sum(0), (y & ~p).sum(0)


def f1s(y, p):
    tp, fp, fn = counts(y, p)
    micro = 2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())
    d = 2 * tp + fp + fn
    per = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
    ti = (y & p).sum(1)
    di = 2 * ti + (~y & p).sum(1) + (y & ~p).sum(1)
    image = np.where(di > 0, 2 * ti / np.maximum(di, 1), 1.0).mean()
    return np.array([micro, per.mean(), image]), per


def boot_diff(y, pa, pb, B=2000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(y), (B, len(y)))
    d = np.array([f1s(y[i], pb[i])[0] - f1s(y[i], pa[i])[0] for i in idx])
    return np.percentile(d, [2.5, 97.5], axis=0)


def fmt(point, ci):
    return " / ".join(f"{p:+.4f} [{l:+.4f}, {h:+.4f}]" for p, l, h in zip(point, ci[0], ci[1]))


def macro(a, runs, split):
    (na, fa, y, pa, cats), (nb, fb, yb, pb, _) = runs
    assert (fa == fb).all() and (y == yb).all()
    (ma, per_a), (mb, per_b) = f1s(y, pa), f1s(y, pb)
    sup = y.sum(0)
    tiny = sup <= a.tiny
    delta = per_b - per_a
    rows = sorted(range(len(cats)), key=lambda k: delta[k])
    tp_a, fp_a, fn_a = counts(y, pa)
    tp_b, fp_b, fn_b = counts(y, pb)
    print(f"\n## {split}: {nb} - {na}")
    print(f"micro / macro / image diff (95% CI): {fmt(mb - ma, boot_diff(y, pa, pb))}")
    print(f"macro change {delta.mean():+.4f}; classes with support <= {a.tiny} ({', '.join(c for c, t in zip(cats, tiny) if t)}) "
          f"carry {delta[tiny].sum() / len(cats):+.4f}")
    print(f"macro without them: {per_a[~tiny].mean():.4f} -> {per_b[~tiny].mean():.4f} ({(per_b - per_a)[~tiny].mean():+.4f})")
    print("| class | support | F1 A | F1 B | dF1 | TP A->B | FP A->B | FN A->B |\n|---|---|---|---|---|---|---|---|")
    for k in rows:
        print(f"| {cats[k]} | {sup[k]} | {per_a[k]:.3f} | {per_b[k]:.3f} | {delta[k]:+.3f} | {tp_a[k]}->{tp_b[k]} | "
              f"{fp_a[k]}->{fp_b[k]} | {fn_a[k]}->{fn_b[k]} |")
    return {"a": na, "b": nb, "diff": (mb - ma).tolist(), "tiny_classes": [c for c, t in zip(cats, tiny) if t],
            "tiny_share": float(delta[tiny].sum() / len(cats)), "macro_wo_tiny": [float(per_a[~tiny].mean()), float(per_b[~tiny].mean())],
            "per_class": {cats[k]: {"support": int(sup[k]), "a": float(per_a[k]), "b": float(per_b[k])} for k in range(len(cats))}}


def seeds(a, runs, split):
    names = [r[0] for r in runs]
    y = runs[0][2]
    assert all((r[1] == runs[0][1]).all() and (r[2] == y).all() for r in runs)
    P = [r[3] for r in runs]
    M = [f1s(y, p)[0] for p in P]
    idx = np.random.default_rng(0).integers(0, len(y), (2000, len(y)))
    bs = np.array([[f1s(y[i], p[i])[0] for p in P] for i in idx])   # [B, 4, 3]
    out = {"runs": {n: m.tolist() for n, m in zip(names, M)}}
    print(f"\n## {split} (micro / macro / image)")
    for n, m in zip(names, M):
        print(f"{n:28s} " + " / ".join(f"{v:.4f}" for v in m))
    for label, d, db in (
            (f"seed gap {names[1]} - {names[0]}", M[1] - M[0], bs[:, 1] - bs[:, 0]),
            (f"seed gap {names[3]} - {names[2]}", M[3] - M[2], bs[:, 3] - bs[:, 2]),
            ("head gap mean(B) - mean(A)", (M[2] + M[3] - M[0] - M[1]) / 2, (bs[:, 2] + bs[:, 3] - bs[:, 0] - bs[:, 1]) / 2)):
        ci = np.percentile(db, [2.5, 97.5], axis=0)
        print(f"{label:50s} {fmt(d, ci)}")
        out[label] = {"diff": d.tolist(), "lo": ci[0].tolist(), "hi": ci[1].tolist()}
    return out


def pick(a):
    for d in a.runs:
        name = d.split(":")[0]
        res = []
        for t in np.round(np.arange(0.05, 0.951, 0.05), 2):
            _, _, y, p, _ = load(a.runs_root, f"{name}:{t}", "holdout", a.special)
            m, _ = f1s(y, p)
            res.append((m[1], m[0], t))
        _, _, t = max(res)
        _, _, yv, pv, _ = load(a.runs_root, f"{name}:{t}", "val", a.special)
        mh = [r for r in res if r[2] == t][0]
        mv, _ = f1s(yv, pv)
        print(f"{name}: threshold {t} (holdout macro {mh[0]:.4f} micro {mh[1]:.4f}) -> val micro / macro / image "
              + " / ".join(f"{v:.4f}" for v in mv))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["macro", "seeds", "pick"])
    ap.add_argument("runs", nargs="+", help="<run dir>:<threshold> (pick: <run dir>)")
    ap.add_argument("--runs_root", default="/data/eunjin/DL/runs_hr")
    ap.add_argument("--splits", nargs="+", default=["val", "holdout"])
    ap.add_argument("--special", nargs="*", default=["sweater", "leg warmer", "umbrella"])
    ap.add_argument("--tiny", type=int, default=7, help="support at or below this counts as a tiny class")
    ap.add_argument("--out", default=None, help="json file for the numbers")
    a = ap.parse_args()
    if a.mode == "pick":
        return pick(a)
    assert len(a.runs) == (2 if a.mode == "macro" else 4)
    res = {}
    for split in a.splits:
        runs = [load(a.runs_root, s, split, a.special) for s in a.runs]
        res[split] = (macro if a.mode == "macro" else seeds)(a, runs, split)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
