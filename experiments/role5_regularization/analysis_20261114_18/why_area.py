"""Why-analysis 1: effect by garment area (holdout + val, never test).
Per positive (image, class): area = largest instance area / image area (Fashionpedia 'area' field).
Per-positive rank = share of that class's negatives (same split) scoring below it (ties 1/2): threshold-free.
Paired image bootstrap (2000) of the mean rank difference per area bin."""
import json, numpy as np, sys
CATS = None
D = "/data/eunjin/DL"; R = f"{D}/runs_hr"; A = f"{D}/fashionpedia_ann"
BINS = [0, .005, .02, .08, 1.01]; BN = ["<0.5%", "0.5-2%", "2-8%", ">=8%"]
def areas(split):
    js = json.load(open(f"{A}/instances_attributes_{'train' if split == 'holdout' else 'val'}2020.json"))
    im = {i["id"]: (i["file_name"], i["width"] * i["height"]) for i in js["images"]}
    out = {}
    for a in js["annotations"]:
        c = a["category_id"]
        if c > 26: continue
        f, wh = im[a["image_id"]]
        k = (f, c); out[k] = max(out.get(k, 0), a["area"] / wh)
    return out
def load(run, split, tta=True):
    p = f"{R}/{run}/tta/preds_{split}_s800_flip.npz" if tta else f"{R}/{run}/preds_{split}.npz"
    z = np.load(p, allow_pickle=False); o = np.argsort(z["filenames"])
    return z["filenames"][o], z["targets"][o].astype(bool), z["logits"][o].astype(np.float64)
def ranks(y, s):
    r = np.full(y.shape, np.nan)
    for c in range(y.shape[1]):
        neg = np.sort(s[~y[:, c], c]); pos = np.where(y[:, c])[0]
        if len(neg) == 0 or len(pos) == 0: continue
        lo = np.searchsorted(neg, s[pos, c], "left"); hi = np.searchsorted(neg, s[pos, c], "right")
        r[pos, c] = (lo + hi) / 2 / len(neg)
    return r
PAIRS = [("resolution 384 -> 400~800 (C1, no TTA)", "R5_cnxt_hr_low_s42", "R5_cnxt_hr_s42", False),
         ("distill (C1 -> KD)", "R5_cnxt_hr_s42", "R5_cnxt_hr_kd05_s42", True),
         ("GAP only -> R4 (adds max)", "R5_bce_kd_gap_s42", "R4_bce_kd_s42", True),
         ("GMP only -> R4 (adds avg)", "R5_bce_kd_gmp_s42", "R4_bce_kd_s42", True),
         ("fixed 800 -> R4 (adds size jitter)", "R5_bce_kd_fix800_s42", "R4_bce_kd_s42", True),
         ("R4 -> pseudo-label", "R4_bce_kd_s42", "R5_bce_kd_pl_s42", True),
         ("R4 -> CSRA+mask", "R4_bce_kd_s42", "R5_bce_kd_csra_mask_s42", True),
         ("R4 -> mosaic", "R4_bce_kd_s42", "R5_bce_kd_mos9_s42", True)]
for split in ("holdout", "val"):
    ar = areas(split)
    print(f"\n######## {split}")
    first = True
    for name, a, b, tta in PAIRS:
        fa, y, sa = load(a, split, tta); fb, yb, sb = load(b, split, tta)
        assert (fa == fb).all() and (y == yb).all()
        area = np.full(y.shape, np.nan)
        for i, f in enumerate(fa):
            for c in np.where(y[i])[0]:
                area[i, c] = ar.get((str(f), c), np.nan)
        if first:
            ok = ~np.isnan(area[y]); print(f"positives {y.sum()}, with area {ok.sum()}; area quantiles 10/25/50/75/90%:",
                  np.round(np.nanpercentile(area[y], [10, 25, 50, 75, 90]) * 100, 2), "%")
            for k in range(4):
                m = y & (area >= BINS[k]) & (area < BINS[k + 1])
                cnt = m.sum(0); top = np.argsort(-cnt)[:5]
                cats = np.load(f"{R}/{a}/preds_holdout.npz" if not tta else f"{R}/{a}/tta/preds_holdout_s800_flip.npz")["categories"]
                print(f"  bin {BN[k]}: {m.sum()} positives; top:", ", ".join(f"{cats[c]}({cnt[c]})" for c in top))
            first = False
        ra, rb = ranks(y, sa), ranks(y, sb)
        bins = [y & (area >= BINS[k]) & (area < BINS[k + 1]) for k in range(4)]
        n = len(y); idx = np.random.default_rng(0).integers(0, n, (2000, n))
        line = []
        for k, m in enumerate(bins):
            d = np.where(m, rb - ra, 0.0); cnt = m.astype(float)
            pt = d.sum() / cnt.sum()
            bs = d.sum(1)[idx].sum(1) / cnt.sum(1)[idx].sum(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            miss_a = (ra[m] < .9).mean(); miss_b = (rb[m] < .9).mean()
            sig = "*" if lo > 0 or hi < 0 else " "
            line.append(f"{BN[k]} {pt*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]{sig} (rank<.9: {miss_a:.3f}->{miss_b:.3f})")
        print(f"{name}:\n   " + "\n   ".join(line), flush=True)
