"""Threshold-free comparison: macro / micro average precision, paired image bootstrap (2000), holdout and val."""
import numpy as np
R = "/data/eunjin/DL/runs_hr"
def load(run, split):
    z = np.load(f"{R}/{run}/tta/preds_{split}_s800_flip.npz", allow_pickle=False)
    o = np.argsort(z["filenames"])
    return z["filenames"][o], z["targets"][o].astype(bool), z["logits"][o].astype(np.float64)
def ap1(y, s):
    if y.sum() == 0: return np.nan
    o = np.argsort(-s, kind="stable"); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return (prec * y).sum() / y.sum()
def m(y, s):
    return np.array([np.nanmean([ap1(y[:, c], s[:, c]) for c in range(y.shape[1])]), ap1(y.ravel(), s.ravel())])
PAIRS = [("R5_cnxt_hr_s42", "R5_cnxt_hr_kd05_s42")] + [("R4_bce_kd_s42", b) for b in
         ("R5_bce_kd_gap_s42", "R5_bce_kd_gmp_s42", "R5_bce_kd_fix800_s42", "R5_bce_kd_pl_s42", "R5_bce_kd_csra_mask_s42", "R5_bce_kd_mos9_s42", "R5_bce_kd_swinb_s42")]
for split in ("holdout", "val"):
    print("==", split, "(macro AP / micro AP)")
    for a, b in PAIRS:
        fa, y, sa = load(a, split); fb, yb, sb = load(b, split)
        assert (fa == fb).all() and (y == yb).all()
        A, B = m(y, sa), m(y, sb)
        idx = np.random.default_rng(0).integers(0, len(y), (2000, len(y)))
        d = np.array([m(y[i], sb[i]) - m(y[i], sa[i]) for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5], 0)
        print(f"{b} - {a}: A {A.round(4)} B {B.round(4)} | macroAP {B[0]-A[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | microAP {B[1]-A[1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]", flush=True)
