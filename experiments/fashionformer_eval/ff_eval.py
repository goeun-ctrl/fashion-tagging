"""Image-level F1 of FashionFormer scores vs our classifiers on val, with paired bootstrap.
Usage: python ff_eval.py <ff_scores.npz> <convnext preds_val.npz> <eva_ema preds_val.npz>"""
import sys, json
import numpy as np

def f1s(y, p):
    tp = (y & p).sum(0); fp = (~y & p).sum(0); fn = (y & ~p).sum(0)
    micro = 2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())
    d = 2 * tp + fp + fn
    per = np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)
    tpi = (y & p).sum(1); di = 2 * tpi + (~y & p).sum(1) + (y & ~p).sum(1)
    image = np.where(di > 0, 2 * tpi / np.maximum(di, 1), 1.0).mean()
    return np.array([micro, np.nanmean(per), image]), per

ff = np.load(sys.argv[1], allow_pickle=False)
names = ff["filenames"].tolist(); y = ff["targets"].astype(bool); S = ff["scores"]
cats = [str(c) for c in ff["categories"]]
def load(path):
    z = np.load(path, allow_pickle=False); idx = {n: i for i, n in enumerate(z["filenames"].tolist())}
    o = [idx[n] for n in names]
    assert (z["targets"][o].astype(bool) == y).all()
    return z["logits"][o] > 0
others = {"ConvNeXt-T (2번 기준선)": load(sys.argv[2]), "EVA02-S + EMA 조합": load(sys.argv[3])}
print("== FashionFormer threshold sweep (val; any choice made here is optimistic)")
for t in (0.1, 0.2, 0.3, 0.4, 0.5):
    m, _ = f1s(y, S > t); print(f"thr {t:.1f}: micro {m[0]:.4f} macro {m[1]:.4f} image {m[2]:.4f} mean_pred {(S > t).sum(1).mean():.2f}")
print("true mean labels", y.sum(1).mean())
rng = np.random.default_rng(0); B = 2000; n = len(y); idx = rng.integers(0, n, (B, n))
for t in (0.3, 0.5):
    P = S > t
    base = np.array([f1s(y[i], P[i])[0] for i in idx])
    print(f"\n== FashionFormer @ {t} vs classifiers (val, paired bootstrap, FF minus other)")
    print("FF point", np.round(f1s(y, P)[0], 4))
    for name, Q in others.items():
        b = np.array([f1s(y[i], Q[i])[0] for i in idx]); d = base - b
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        print(f"{name:24s} point {np.round(f1s(y, Q)[0], 4)}  diff " + " ".join(f"{x:+.4f}[{l:+.4f},{h:+.4f}]" for x, l, h in zip(f1s(y, P)[0] - f1s(y, Q)[0], lo, hi)))
_, pf = f1s(y, S > 0.3); _, pe = f1s(y, others["EVA02-S + EMA 조합"]); _, pc = f1s(y, others["ConvNeXt-T (2번 기준선)"])
print("\nclass | sup | FF@0.3 | ConvNeXt | EVA+EMA")
for j, c in enumerate(cats):
    print(f"{c[:26]} | {int(y[:, j].sum())} | {pf[j]:.3f} | {pc[j]:.3f} | {pe[j]:.3f}")
