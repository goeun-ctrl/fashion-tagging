"""Option A (post-hoc, no retraining): garment-level voting inside each softmax group, same B map.

Per softmax group g (spec.GROUPS) and cell: fg = sum of the group's class probabilities (= 1 - p(none)).
Garment region R_t = cells with fg >= t. For class c of group g:
  share_c(t) = sum_R p_c / sum_R fg                (c's share of the garment mass of its group)
  pres_c     = mean of the top-k cells of fg, k = the class's own k (ratio_c x cells, >= min_k)
  vote_c(t)  = pres_c x share_c(t)                 (empty R -> 0)
  blend_c(t) = sqrt(orig_c x vote_c(t))            (geometric mean with the current top-k score)
Scores of every variant are cached once per split (<runs_dir>/<run>/vote_<split>.npz); then micro / macro AP and
the within-group AP of the 5 judging groups are printed for each variant vs the original score.
Holdout is the selection split, val only confirms. Analysis only, nothing here feeds training.

    python -I scripts/posthoc_vote.py --data D --runs_dir R --run S6_base --split holdout --device mps
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_segb import GROUPS as JUDGE, load_preds, macro_ap, micro_ap, within_group_ap  # noqa: E402
from review_confusables import group_right  # noqa: E402

TS = (0.3, 0.5, 0.7)


def compute(a, names):
    import torch
    from fashion_tagging.data.preprocess import timm_mean_std
    from fashion_tagging.segb.data import SegBDataset, collate
    from fashion_tagging.segb.model import SegB
    from fashion_tagging.segb.spec import GROUPS

    st = torch.load(os.path.join(a.runs_dir, a.run, "best.pt"), map_location="cpu", weights_only=False)
    cfg = st["config"]
    model = SegB(cfg["model"], False, 0.0, ratio=cfg["ratio"], min_k=cfg["min_k"])
    model.load_state_dict(st["model"])
    dev = torch.device(a.device)
    model = model.to(dev).eval()
    mean, std = timm_mean_std(cfg["model"])
    ds = SegBDataset([os.path.join(a.data, a.split, "images", n) for n in names], ["eval"] * len(names),
                     np.zeros((len(names), 27), np.float32), None, None, mean, std, False, cfg["max_long"], 0.0)
    ratio = np.asarray(cfg["ratio"])
    N = len(names)
    orig = np.zeros((N, 27))
    vote = np.zeros((len(TS), N, 27))
    with torch.no_grad():
        for i in range(N):
            b = collate([ds[(i, cfg["test_short"])]], (0, 0))
            o = model(b["x"].to(dev), b["sizes"].to(dev))
            H, W = b["sizes"][0].tolist()
            vh, vw = math.ceil(H / 4), math.ceil(W / 4)
            p = o["probs"][0, :, :vh, :vw].float().cpu().numpy().reshape(27, -1)
            orig[i] = o["tags"][0].float().cpu().numpy()
            nv = p.shape[1]
            for cs in GROUPS:
                fg = p[cs].sum(0)
                fs = np.sort(fg)[::-1]
                cfs = np.cumsum(fs)
                for c in cs:
                    k = int(min(max(round(ratio[c] * nv), cfg["min_k"]), nv))
                    pres = cfs[k - 1] / k
                    for ti, t in enumerate(TS):
                        r = fg >= t
                        den = fg[r].sum()
                        vote[ti, i, c] = pres * p[c][r].sum() / den if den > 0 else 0.0
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{N}", flush=True)
    return orig, vote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="holdout", choices=["holdout", "val"])
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    names, y, saved = load_preds(a.runs_dir, a.data, a.run, a.split)
    cache = os.path.join(a.runs_dir, a.run, f"vote_{a.split}.npz")
    if os.path.exists(cache) and (np.load(cache)["filenames"].astype(str) == names).all():
        z = np.load(cache)
        orig, vote = z["orig"], z["vote"]
    else:
        orig, vote = compute(a, names)
        np.savez_compressed(cache, filenames=names, orig=orig, vote=vote)
    print(f"{a.run} {a.split}: {len(names)} images; recomputed top-k vs saved preds max |diff| "
          f"{np.abs(orig - saved).max():.4f}")

    up = JUDGE["upper"]
    layered = y[:, up].sum(1) >= 2
    variants = [("orig (top-k)", orig)]
    for ti, t in enumerate(TS):
        variants += [(f"vote t={t}", vote[ti]), (f"blend t={t}", np.sqrt(orig * vote[ti]))]
    hdr = f"{'variant':14s} {'microAP':>8s} {'macroAP':>8s} {'confWG':>7s} " + \
          " ".join(f"{g[:8]:>8s}" for g in JUDGE) + f" {'up-right':>9s} {'layered':>8s} {'single':>7s}"
    print(hdr)
    for name, s in variants:
        wg = within_group_ap(y, s)
        r, _, has = group_right(y, s, up)
        print(f"{name:14s} {micro_ap(y, s):8.4f} {macro_ap(y, s):8.4f} {np.mean(list(wg.values())):7.4f} "
              + " ".join(f"{wg[g]:8.4f}" for g in JUDGE)
              + f" {r[has].mean():9.3f} {r[has & layered].mean():8.3f} {r[has & ~layered].mean():7.3f}")
    print("up-right = share of images with an upper GT positive where every upper positive outscores every upper "
          "negative; layered = >= 2 upper positives")


if __name__ == "__main__":
    main()
