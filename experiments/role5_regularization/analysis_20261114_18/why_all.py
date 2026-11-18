"""Every clean pair (one thing changed): micro AP, macro AP, small (<0.5%) and large (>=8%) garment rank, holdout + val.
Paired image bootstrap 2000, 95%. Seed pair = noise reference. Never test data."""
import os, numpy as np
exec(open("/data/eunjin/DL/tmp/why_area.py").read().split("PAIRS = ")[0])
def ap1(y, s):
    if y.sum() == 0: return np.nan
    o = np.argsort(-s, kind="stable"); y = y[o]; tp = np.cumsum(y)
    return (tp / np.arange(1, len(y) + 1) * y).sum() / y.sum()
def aps(y, s): return np.array([np.nanmean([ap1(y[:, c], s[:, c]) for c in range(27)]), ap1(y.ravel(), s.ravel())])
def loadx(run, split, kind):
    p = {"flip": f"{R}/{run}/tta/preds_{split}_s800_flip.npz", "noflip": f"{R}/{run}/tta/preds_{split}_s800.npz",
         "raw": f"{R}/{run}/preds_{split}.npz"}[kind]
    z = np.load(p, allow_pickle=False); o = np.argsort(z["filenames"])
    return z["filenames"][o], z["targets"][o].astype(bool), z["logits"][o][:, :27].astype(np.float64)
P = [("distill (C1 -> KD), reference", ("R5_cnxt_hr_s42", "flip"), ("R5_cnxt_hr_kd05_s42", "flip")),
     ("SEED noise: final_msmax s42 -> s43", ("R5_final_msmax_s42", "flip"), ("R5_final_msmax_s43", "flip")),
     ("SEED noise: final_csra s42 -> s43", ("R5_final_csra_s42", "flip"), ("R5_final_csra_s43", "flip")),
     ("loss ASL -> BCE (kd_msmax -> R4)", ("R5_cnxt_hr_kd_msmax_s42", "flip"), ("R4_bce_kd_s42", "flip")),
     ("flip TTA off -> on (GAP-only run, R4 recipe)", ("R5_bce_kd_gap_s42", "noflip"), ("R5_bce_kd_gap_s42", "flip")),
     ("stride16 + GAP/GMP head (KD -> kd_msmax)", ("R5_cnxt_hr_kd05_s42", "flip"), ("R5_cnxt_hr_kd_msmax_s42", "flip")),
     ("query head, lr_head 1e-4 (C1 -> C3)", ("R5_cnxt_hr_s42", "flip"), ("R5_cnxt_hr_c3_s42", "flip")),
     ("resolution 400~800 -> 480~960 (C1 -> 960)", ("R5_cnxt_hr_s42", "flip"), ("R5_cnxt_hr_960_s42", "flip")),
     ("size jitter 400~800 -> 80~800 (C1 -> wide)", ("R5_cnxt_hr_s42", "raw"), ("R5_cnxt_hr_wide_s42", "raw")),
     ("FashionFormer lr (C1 -> fflr)", ("R5_cnxt_hr_s42", "raw"), ("R5_cnxt_hr_fflr_s42", "raw")),
     ("C1 + 4ep fine-tune with gate/neg/person/gray (C1 -> gate_v2)", ("R5_cnxt_hr_s42", "flip"), ("R5_cnxt_hr_gate_v2_s42", "flip")),
     ("gate+neg+person+aug bundle (kd_msmax -> final)", ("R5_cnxt_hr_kd_msmax_s42", "flip"), ("R5_final_msmax_s42", "flip"))]
for split in ("holdout", "val"):
    ar = areas(split); print(f"\n######## {split}: micro AP | macro AP | rank small<0.5% | rank large>=8%  (x100 for ranks)")
    for name, (a, ka), (b, kb) in P:
        try:
            fa, y, sa = loadx(a, split, ka); fb, yb, sb = loadx(b, split, kb)
        except FileNotFoundError as e:
            print(f"{name}: missing {e.filename}"); continue
        assert (fa == fb).all() and (y == yb).all()
        area = np.full(y.shape, np.nan)
        for i, f in enumerate(fa):
            for c in np.where(y[i])[0]: area[i, c] = ar.get((str(f), c), np.nan)
        ra, rb = ranks(y, sa), ranks(y, sb); sm, lg = y & (area < .005), y & (area >= .08)
        n = len(y); idx = np.random.default_rng(0).integers(0, n, (1000, n))
        XA, XB = aps(y, sa), aps(y, sb)
        dap = np.array([aps(y[i], sb[i]) - aps(y[i], sa[i]) for i in idx])
        def rk(m):
            d = np.where(m, rb - ra, 0.); c = m.astype(float); bs = d.sum(1)[idx].sum(1) / c.sum(1)[idx].sum(1)
            lo, hi = np.percentile(bs, [2.5, 97.5]); s = "*" if lo > 0 or hi < 0 else " "
            return f"{d.sum()/c.sum()*100:+.2f} [{lo*100:+.2f},{hi*100:+.2f}]{s}"
        out = []
        for j in (1, 0):
            lo, hi = np.percentile(dap[:, j], [2.5, 97.5]); s = "*" if lo > 0 or hi < 0 else " "
            out.append(f"{XB[j]-XA[j]:+.4f} [{lo:+.4f},{hi:+.4f}]{s}")
        print(f"{name}\n    {out[0]} | {out[1]} | {rk(sm)} | {rk(lg)}", flush=True)
