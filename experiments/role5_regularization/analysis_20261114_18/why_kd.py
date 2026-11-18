"""Why-analysis 2: what did distillation change? (C1 -> KD, ASL, same everything else; holdout + val, never test)
(a) similar-garment separation: per-positive rank against HARD negatives (same garment group has another positive)
    vs EASY negatives, paired image bootstrap.
(b) per-class AP change vs class frequency.
(c) what the teacher says on train: shifted teacher prob on positives / hard negatives / easy negatives."""
import numpy as np, math
D = "/data/eunjin/DL"; R = f"{D}/runs_hr"
GROUPS = {"upper": [0, 1, 2, 3, 4, 5, 9], "lower+dress": [6, 7, 8, 10, 11], "legwear": [20, 21, 22]}
G = {c: g for g, cs in GROUPS.items() for c in cs}
def load(run, split):
    z = np.load(f"{R}/{run}/tta/preds_{split}_s800_flip.npz", allow_pickle=False); o = np.argsort(z["filenames"])
    return z["filenames"][o], z["targets"][o].astype(bool), z["logits"][o].astype(np.float64), z["categories"]
def hard_mask(y):
    h = np.zeros_like(y)
    for c, g in G.items():
        others = [k for k in GROUPS[g] if k != c]
        h[:, c] = ~y[:, c] & y[:, others].any(1)
    return h
def rank_vs(y, s, negmask):
    r = np.full(y.shape, np.nan)
    for c in G:
        neg = np.sort(s[negmask[:, c], c]); pos = np.where(y[:, c])[0]
        if len(neg) < 5 or len(pos) == 0: continue
        r[pos, c] = (np.searchsorted(neg, s[pos, c], "left") + np.searchsorted(neg, s[pos, c], "right")) / 2 / len(neg)
    return r
def ap1(y, s):
    o = np.argsort(-s, kind="stable"); y = y[o]; tp = np.cumsum(y)
    return (tp / np.arange(1, len(y) + 1) * y).sum() / max(1, y.sum())
ztr = np.load(f"{D}/fassion_tagging/data/train/labels.npz"); ntrain = ztr["targets"].sum(0)
for split in ("holdout", "val"):
    f, y, a, cats = load("R5_cnxt_hr_s42", split); _, _, b, _ = load("R5_cnxt_hr_kd05_s42", split)
    h = hard_mask(y); e = ~y & ~h
    print(f"\n######## {split}  (rank x100 vs negatives; 'miss' = share of positives ranked below 90% of those negatives)")
    n = len(y); idx = np.random.default_rng(0).integers(0, n, (2000, n))
    for nm, mk in (("hard negatives (similar garment present)", h), ("easy negatives", e)):
        ra, rb = rank_vs(y, a, mk), rank_vs(y, b, mk)
        m = ~np.isnan(ra)
        d = np.where(m, rb - ra, 0.); cnt = m.astype(float)
        bs = d.sum(1)[idx].sum(1) / cnt.sum(1)[idx].sum(1); lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {nm}: C1 {np.nanmean(ra)*100:.2f} -> KD {np.nanmean(rb)*100:.2f}, diff {(d.sum()/cnt.sum())*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]"
              f"; miss {(ra[m] < .9).mean():.3f} -> {(rb[m] < .9).mean():.3f}")
        for g, cs in GROUPS.items():
            mm = m.copy(); mm[:, [c for c in range(27) if c not in cs]] = False
            print(f"      {g}: diff {np.nanmean((rb - ra)[mm])*100:+.2f} over {mm.sum()} positives")
    print("  per-class AP change (KD - C1), classes sorted by train positives:")
    rows = []
    for c in np.argsort(-ntrain):
        if y[:, c].sum() == 0: continue
        rows.append((cats[c], int(ntrain[c]), int(y[:, c].sum()), ap1(y[:, c], b[:, c]) - ap1(y[:, c], a[:, c])))
    for third, sl in (("frequent 9", rows[:9]), ("middle 9", rows[9:18]), ("rare", rows[18:])):
        print(f"    {third}: mean dAP {np.mean([r[3] for r in sl]):+.4f}; " + ", ".join(f"{r[0][:10]} {r[3]:+.3f}" for r in sl))
    print("    Spearman(train freq, dAP):", round(float(np.corrcoef(np.argsort(np.argsort([r[1] for r in rows])), np.argsort(np.argsort([r[3] for r in rows])))[0, 1]), 3))
# (c) teacher on train
zt = np.load(f"{D}/fashionformer/train_scores/ff_r50_train.npz")
s = np.clip(zt["scores"].astype(np.float64), 1e-4, 1 - 1e-4)
p = 1 / (1 + np.exp(-(np.log(s / (1 - s)) - math.log(.3 / .7))))
yt = zt["targets"].astype(bool); ht = hard_mask(yt); et = ~yt & ~ht
print("\n######## teacher (shifted) on train, garment groups only")
for g, cs in GROUPS.items():
    P = p[:, cs]; Y = yt[:, cs]; H = ht[:, cs]; E = et[:, cs]
    print(f"  {g}: positives mean {P[Y].mean():.3f} (share < 0.5: {(P[Y] < .5).mean():.3f}) | hard neg mean {P[H].mean():.3f} (share > 0.2: {(P[H] > .2).mean():.3f}) | easy neg mean {P[E].mean():.3f} (share > 0.2: {(P[E] > .2).mean():.3f})")
allc = np.ones(27, bool)
print(f"  all 27: positives mean {p[yt].mean():.3f}, <0.5 share {(p[yt] < .5).mean():.3f}; negatives mean {p[~yt].mean():.3f}, >0.5 share {(p[~yt] > .5).mean():.4f}")
