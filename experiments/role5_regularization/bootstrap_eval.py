"""Bootstrap 95% CIs and paired differences vs R5_eva_base_s42 for the role5 EVA runs. Usage: python bootstrap_eval.py <dir with run folders>"""
import csv, json, sys, numpy as np
R = sys.argv[1]
RUNS = ["R5_eva_base_s42", "R5_eva_wdx_s42", "R5_eva_wdx_dp01_s42", "R5_eva_wdx_dp01_ema_s42"]
def f1s(y, p):
    tp = (y & p).sum(0); fp = (~y & p).sum(0); fn = (y & ~p).sum(0)
    micro = 2*tp.sum()/max(1, 2*tp.sum()+fp.sum()+fn.sum())
    d = 2*tp+fp+fn; per = np.where(d > 0, 2*tp/np.maximum(d, 1), np.nan)
    macro = np.nanmean(per)
    tpi = (y & p).sum(1); di = 2*tpi + (~y & p).sum(1) + (y & ~p).sum(1)
    image = np.where(di > 0, 2*tpi/np.maximum(di, 1), 1.0).mean()
    return np.array([micro, macro, image])
rng = np.random.default_rng(0); B = 2000
out = {}
for split in ("val", "holdout"):
    preds = {}
    for r in RUNS:
        z = np.load(f"{R}/{r}/preds_{split}.npz", allow_pickle=False)
        preds[r] = (z["targets"].astype(bool), z["logits"] > 0)   # sigmoid>0.5 <=> logit>0
    y = preds[RUNS[0]][0]
    assert all((preds[r][0] == y).all() for r in RUNS)
    n = len(y); idx = rng.integers(0, n, (B, n))
    boot = {r: np.array([f1s(y[i], preds[r][1][i]) for i in idx]) for r in RUNS}
    point = {r: f1s(y, preds[r][1]) for r in RUNS}
    res = {}
    for r in RUNS:
        lo, hi = np.percentile(boot[r], [2.5, 97.5], axis=0)
        res[r] = {"point": point[r].tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        if r != RUNS[0]:
            d = boot[r] - boot[RUNS[0]]
            res[r]["diff_vs_base"] = (point[r]-point[RUNS[0]]).tolist()
            res[r]["diff_lo"], res[r]["diff_hi"] = np.percentile(d, [2.5, 97.5], axis=0).tolist()
            res[r]["p_better"] = (d > 0).mean(0).tolist()
    out[split] = res
json.dump(out, open(f"{R}/bootstrap.json", "w"), indent=1)
for split in out:
    print("==", split, "(micro, macro, image)")
    for r, v in out[split].items():
        s = f"{r:28s} " + " ".join(f"{p:.4f}[{l:.4f},{h:.4f}]" for p, l, h in zip(v["point"], v["lo"], v["hi"]))
        if "diff_vs_base" in v:
            s += " | diff " + " ".join(f"{d:+.4f}[{l:+.4f},{h:+.4f}]" for d, l, h in zip(v["diff_vs_base"], v["diff_lo"], v["diff_hi"]))
        print(s)
