"""Judge the 4 running experiments + 2 R4 seed reruns (holdout + val only, never test).
Pairs: micro AP | macro AP | rank small<0.5% | rank large>=8% | rank vs HARD negatives (x100 ranks).
Paired image bootstrap 1000, 95%. Then per-class AP table for every run."""
import numpy as np
exec(open("/data/eunjin/DL/tmp/why_area.py").read().split("PAIRS = ")[0])
exec(open("/data/eunjin/DL/tmp/why_kd.py").read().split("def ap1")[0].split("def load")[0])  # GROUPS, G
def ap1(y, s):
    if y.sum() == 0: return np.nan
    o = np.argsort(-s, kind="stable"); y = y[o]; tp = np.cumsum(y)
    return (tp / np.arange(1, len(y) + 1) * y).sum() / y.sum()
def aps(y, s): return np.array([np.nanmean([ap1(y[:, c], s[:, c]) for c in range(27)]), ap1(y.ravel(), s.ravel())])
def hard_mask(y):
    h = np.zeros_like(y)
    for c, g in G.items():
        h[:, c] = ~y[:, c] & y[:, [k for k in GROUPS[g] if k != c]].any(1)
    return h
def rank_vs(y, s, negmask):
    r = np.full(y.shape, np.nan)
    for c in G:
        neg = np.sort(s[negmask[:, c], c]); pos = np.where(y[:, c])[0]
        if len(neg) < 5 or len(pos) == 0: continue
        r[pos, c] = (np.searchsorted(neg, s[pos, c], "left") + np.searchsorted(neg, s[pos, c], "right")) / 2 / len(neg)
    return r
def L(run, split):
    z = np.load(f"{R}/{run}/tta/preds_{split}_s800_flip.npz", allow_pickle=False); o = np.argsort(z["filenames"])
    return z["filenames"][o], z["targets"][o].astype(bool), z["logits"][o][:, :27].astype(np.float64), z["categories"]
import sys
P=[(f"{a} -> {b}",a,b) for a,b in (x.split(":") for x in sys.argv[1:])]
RUNS=list(dict.fromkeys([r for _,a,b in P for r in (a,b)]))
R4=RUNS[0]
for split in ("holdout", "val"):
    ar = areas(split); D_ = {r: L(r, split) for r in RUNS}
    f, y, _, cats = D_[R4]
    for r in RUNS: assert (D_[r][0] == f).all() and (D_[r][1] == y).all()
    area = np.full(y.shape, np.nan)
    for i, fn in enumerate(f):
        for c in np.where(y[i])[0]: area[i, c] = ar.get((str(fn), c), np.nan)
    sm, lg, hm = y & (area < .005), y & (area >= .08), hard_mask(y)
    n = len(y); idx = np.random.default_rng(0).integers(0, n, (1000, n))
    print(f"\n######## {split} (n={n}): micro AP | macro AP | rank small<0.5% | rank large>=8% | rank vs hard neg")
    for r in RUNS:
        x = aps(y, D_[r][2]); print(f"  {r:32s} micro AP {x[1]:.4f}  macro AP {x[0]:.4f}")
    for name, a, b in P:
        sa, sb = D_[a][2], D_[b][2]
        dap = np.array([aps(y[i], sb[i]) - aps(y[i], sa[i]) for i in idx]); XA, XB = aps(y, sa), aps(y, sb)
        def rk(ra, rb, m):
            m = m & ~np.isnan(ra); d = np.where(m, rb - ra, 0.); c = m.astype(float)
            bs = d.sum(1)[idx].sum(1) / c.sum(1)[idx].sum(1); lo, hi = np.percentile(bs, [2.5, 97.5])
            return f"{d.sum()/c.sum()*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]{'*' if lo > 0 or hi < 0 else ' '}"
        ra, rb = ranks(y, sa), ranks(y, sb); ha, hb = rank_vs(y, sa, hm), rank_vs(y, sb, hm)
        out = []
        for j in (1, 0):
            lo, hi = np.percentile(dap[:, j], [2.5, 97.5])
            out.append(f"{XB[j]-XA[j]:+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{name}\n    {out[0]} | {out[1]} | {rk(ra, rb, sm)} | {rk(ra, rb, lg)} | {rk(ha, hb, y)}", flush=True)
    print(f"  per-class AP ({split}); npos | " + " ".join(r.replace('R5_bce_kd_', '').replace('_s42', '')[:9].rjust(9) for r in RUNS))
    for c in range(27):
        if y[:, c].sum() == 0: continue
        print(f"  {str(cats[c])[:22]:22s} {int(y[:, c].sum()):5d} | " + " ".join(f"{ap1(y[:, c], D_[r][2][:, c]):9.3f}" for r in RUNS))
