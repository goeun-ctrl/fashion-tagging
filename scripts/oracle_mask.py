"""Upper bound: how much would the confusable groups gain if the model had perfect garment masks at inference?
Analysis only (holdout / val GT masks stand in for a perfect mask predictor; nothing here feeds training,
thresholds or selection).

Per image and per softmax group g in {upper (spec group 0), lower+dress (spec group 1)}: regions = the GT masks of the
group's positive classes, used class-agnostically (the mask label is never read; same-class instances are already merged
in diag_masks). Inside region R, with fg = sum of the group's class probabilities:
  share_c(R) = sum_R p_c / sum_R fg        (the map's own class split inside that garment)
  oracle_c   = max_R share_c(R)            (each garment gets the class the map gives most of its mass)
  hard_c     = 1 if c is the argmax share in some region, plus share as tie-break (one class per garment)
  blend_c    = sqrt(orig_c x oracle_c)
Classes of other groups and images without a usable region keep the original top-k score.
Caveat: the regions leak how many garments there are (an oracle), so this is a ceiling, not a deployable method.

    python -I scripts/oracle_mask.py --data D --runs_dir R --run S6_base --split holdout --device mps
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_segb import GROUPS as JUDGE, load_masks, load_preds, micro_ap, within_group_ap  # noqa: E402
from review_confusables import group_right  # noqa: E402


def compute(a, names, masks):
    import cv2
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
    N = len(names)
    orig = np.zeros((N, 27))
    share = np.full((N, 27), np.nan)      # max over regions of share_c
    hard = np.full((N, 27), np.nan)       # 1 + share if argmax of some region, else share
    used = np.zeros((N, 2), bool)
    skipped = 0
    with torch.no_grad():
        for i in range(N):
            b = collate([ds[(i, cfg["test_short"])]], (0, 0))
            o = model(b["x"].to(dev), b["sizes"].to(dev))
            H, W = b["sizes"][0].tolist()
            vh, vw = math.ceil(H / 4), math.ceil(W / 4)
            p = o["probs"][0, :, :vh, :vw].float().cpu().numpy()
            orig[i] = o["tags"][0].float().cpu().numpy()
            for gi in (0, 1):
                cs = GROUPS[gi]
                regs = []
                for c in cs:
                    m = masks[i].get(c)
                    if m is None:
                        continue
                    if abs(m.shape[0] / m.shape[1] - H / W) > 0.1 * H / W:
                        skipped += 1
                        continue
                    r = cv2.resize(m.astype(np.uint8), (vw, vh), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if r.any():
                        regs.append(r)
                if not regs:
                    continue
                used[i, gi] = True
                pg = p[cs]
                fg = pg.sum(0)
                sh = np.stack([pg[:, r].sum(1) / max(fg[r].sum(), 1e-9) for r in regs])   # [R, n]
                mx = sh.max(0)
                win = np.zeros(len(cs), bool)
                win[np.argmax(sh, 1)] = True
                share[i, cs] = mx
                hard[i, cs] = mx + win
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{N}", flush=True)
    return orig, share, hard, used, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="holdout", choices=["holdout", "val"])
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    names, y, saved = load_preds(a.runs_dir, a.data, a.run, a.split)
    y = y.astype(bool)
    cache = os.path.join(a.runs_dir, a.run, f"oracle_{a.split}.npz")
    if os.path.exists(cache) and (np.load(cache)["filenames"].astype(str) == names).all():
        z = np.load(cache)
        orig, share, hard, used, skipped = z["orig"], z["share"], z["hard"], z["used"], int(z["skipped"])
    else:
        masks, _ = load_masks(a.data, a.split, names)
        orig, share, hard, used, skipped = compute(a, names, masks)
        np.savez_compressed(cache, filenames=names, orig=orig, share=share, hard=hard, used=used, skipped=skipped)
    print(f"{a.run} {a.split}: {len(names)} images; regions used upper {used[:, 0].sum()} / lower {used[:, 1].sum()} "
          f"images; misaligned masks skipped {skipped}")

    def fill(v):
        return np.where(np.isnan(v), orig, v)
    variants = [("orig (top-k)", orig), ("oracle share", fill(share)), ("oracle hard", fill(hard) / 2),
                ("blend", np.sqrt(orig * fill(share)))]
    up = JUDGE["upper"]
    lo = JUDGE["lower+dress"]
    lay_u = y[:, up].sum(1) >= 2
    lay_l = y[:, lo].sum(1) >= 2
    print(f"{'variant':14s} {'microAP':>8s} {'confWG':>7s} {'upper':>7s} {'low+dr':>7s} "
          f"{'up-right':>9s} {'layered':>8s} {'single':>7s} {'lo-right':>9s} {'lo-lay':>7s}")
    for name, s in variants:
        wg = within_group_ap(y, s)
        ru, _, hu = group_right(y, s, up)
        rl, _, hl = group_right(y, s, lo)
        print(f"{name:14s} {micro_ap(y, s):8.4f} {np.mean(list(wg.values())):7.4f} {wg['upper']:7.4f} "
              f"{wg['lower+dress']:7.4f} {ru[hu].mean():9.3f} {ru[hu & lay_u].mean():8.3f} "
              f"{ru[hu & ~lay_u].mean():7.3f} {rl[hl].mean():9.3f} {rl[hl & lay_l].mean():7.3f}")
    print(f"layered = >= 2 positives of the group (upper n={int((hu & lay_u).sum())}, lower n={int((hl & lay_l).sum())})")


if __name__ == "__main__":
    main()
