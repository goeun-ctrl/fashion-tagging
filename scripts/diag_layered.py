"""Why does a B run lose confusable-group cases (e.g. a shirt under a jacket)? Analysis only, holdout / val.

For every image where the run gets the group wrong (some GT positive scores below some GT negative), take the
lowest-scoring GT positive c and look at the run's own probability map for c inside c's GT mask:
  (가) pooling: the map does find c inside the mask (max prob in mask >= --hi) but the top-k pooled score is low
       (k = ratio[c] x cells is larger than the visible area, so low cells are averaged in)
  (나) map: the map does not find c inside the mask (max prob in mask < --hi); we record which class of the group
       wins the cells of c's mask instead
Also a counterfactual: would the group be right if every class of the group were pooled over only the top
--alt_k cells (pooling-only change, same map)?

    python -I scripts/diag_layered.py --data D --runs_dir R --run S6_base --ref baseline --group upper --device mps
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_segb import GROUPS, load_masks, load_preds  # noqa: E402
from review_confusables import group_right  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--split", default="holdout", choices=["holdout", "val"])
    ap.add_argument("--group", default="upper")
    ap.add_argument("--which", default="only_ref", choices=["only_ref", "all_wrong"],
                    help="only_ref: run wrong & ref right; all_wrong: every image the run gets wrong")
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--alt_k", type=int, default=4)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    import cv2
    import torch
    from fashion_tagging.data.preprocess import timm_mean_std
    from fashion_tagging.segb.data import SegBDataset, collate
    from fashion_tagging.segb.model import SegB

    sp, cs = a.split, GROUPS[a.group]
    fr, y, pr = load_preds(a.runs_dir, a.data, a.ref, sp)
    fb, yb, pb = load_preds(a.runs_dir, a.data, a.run, sp)
    assert (fr == fb).all()
    cats = [str(c).split(",")[0] for c in np.load(os.path.join(a.runs_dir, a.run, f"preds_{sp}.npz"))["categories"]][:27]
    y = y.astype(bool)
    rr, _, has = group_right(y, pr, cs)
    rb, _, _ = group_right(y, pb, cs)
    sel = has & ~rb & (rr if a.which == "only_ref" else True)
    idx = np.where(sel)[0]
    masks, area = load_masks(a.data, sp, fr)

    st = torch.load(os.path.join(a.runs_dir, a.run, "best.pt"), map_location="cpu", weights_only=False)
    cfg = st["config"]
    model = SegB(cfg["model"], False, 0.0, ratio=cfg["ratio"], min_k=cfg["min_k"])
    model.load_state_dict(st["model"])
    dev = torch.device(a.device)
    model = model.to(dev).eval()
    mean, std = timm_mean_std(cfg["model"])
    ds = SegBDataset([os.path.join(a.data, sp, "images", n) for n in fr], ["eval"] * len(fr),
                     np.zeros((len(fr), 27), np.float32), None, None, mean, std, False, cfg["max_long"], 0.0)
    ratio = np.asarray(cfg["ratio"])

    rows, alt_right, skipped = [], 0, 0
    with torch.no_grad():
        for i in idx:
            b = collate([ds[(int(i), cfg["test_short"])]], (0, 0))
            o = model(b["x"].to(dev), b["sizes"].to(dev))
            H, W = b["sizes"][0].tolist()
            vh, vw = math.ceil(H / 4), math.ceil(W / 4)
            p = o["probs"][0, :, :vh, :vw].float().cpu().numpy()
            nv = vh * vw
            # counterfactual pooling: top alt_k mean for the group's classes, same map
            alt = np.array([np.sort(p[c].ravel())[-a.alt_k:].mean() for c in range(27)])
            r_alt, _, _ = group_right(y[i:i + 1], alt[None], cs)
            alt_right += int(r_alt[0])
            pos = [c for c in cs if y[i, c]]
            c = min(pos, key=lambda k: pb[i, k])                     # the losing positive
            if c not in masks[i]:
                skipped += 1
                continue
            m = masks[i][c]
            if abs(m.shape[0] / m.shape[1] - H / W) > 0.1 * H / W:
                skipped += 1
                continue
            cov = cv2.resize(m.astype(np.uint8), (vw, vh), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not cov.any():
                skipped += 1
                continue
            k = int(min(max(round(ratio[c] * nv), cfg["min_k"]), nv))
            inside = p[c][cov]
            win = [cs[j] for j in np.argmax(p[cs][:, cov], 0)]       # which group class wins each mask cell
            wv, wc = np.unique(win, return_counts=True)
            neg = [q for q in cs if not y[i, q]]
            beat = max(neg, key=lambda q: pb[i, q]) if neg else None
            from fashion_tagging.segb.spec import CLASS_TO_GROUP, GROUPS as SG
            sg = SG[CLASS_TO_GROUP[c][0]]
            fg = p[sg][:, cov].sum(0)                                    # 1 - p(none) inside c's mask
            share = p[c][cov].sum() / max(fg.sum(), 1e-9)
            rows.append(dict(fg=fg.mean(), share=share, i=int(i), c=c, k=k, cells=int(cov.sum()), score=pb[i, c], mx=inside.max(),
                             mean=inside.mean(), winner=int(wv[np.argmax(wc)]), win_frac=wc.max() / wc.sum(),
                             beat=beat, beat_s=pb[i, beat] if beat is not None else np.nan))

    n = len(rows)
    print(f"{a.run} vs {a.ref} {sp} {a.group} ({a.which}): {len(idx)} images, analysed {n}, skipped {skipped} "
          f"(no / misaligned mask)")
    if not n:
        return
    mx = np.array([r["mx"] for r in rows])
    small = np.array([r["cells"] < r["k"] for r in rows])
    ga = mx >= a.hi
    print(f"\n(가) pooling   map max in mask >= {a.hi}: {ga.sum():4d} ({ga.mean():.0%})"
          f"   of which visible cells < k: {(ga & small).sum()}")
    print(f"(나) map       map max in mask <  {a.hi}: {(~ga).sum():4d} ({(~ga).mean():.0%})")
    for t in (0.3, 0.7):
        print(f"   (threshold {t}: 가 {(mx >= t).mean():.0%} / 나 {(mx < t).mean():.0%})")
    print(f"\ncounterfactual: pooling = top-{a.alt_k} mean for every class (same map) -> group right on "
          f"{alt_right}/{len(idx)} of these images")
    print(f"\nvisible cells of the losing class vs its k: median {np.median([r['cells'] for r in rows]):.0f} vs "
          f"{np.median([r['k'] for r in rows]):.0f}")
    fg = np.array([r["fg"] for r in rows]); sh = np.array([r["share"] for r in rows])
    print(f"\ninside the losing class's GT mask: mean garment prob (1 - none) median {np.median(fg):.2f}, "
          f">= 0.7 in {(fg >= 0.7).mean():.0%};  that class's share of the garment mass median {np.median(sh):.2f}, "
          f"< 0.5 in {(sh < 0.5).mean():.0%}")
    print("\nlosing class (count) | winner inside its mask, cases where the map misses it (나):")
    from collections import Counter
    for c, cnt in Counter(r["c"] for r in rows).most_common():
        sub = [r for r in rows if r["c"] == c]
        miss = [r for r in sub if r["mx"] < a.hi]
        w = Counter(cats[r["winner"]] for r in miss).most_common(3)
        print(f"  {cats[c]:10s} n={len(sub):3d}  가 {len(sub) - len(miss):3d}  나 {len(miss):3d}  "
              f"나 winners: {', '.join(f'{k} {v}' for k, v in w) or '-'}")
    print("\nwho beats the losing positive (top negative), count:")
    for (c, q), cnt in Counter((cats[r["c"]], cats[r["beat"]] if r["beat"] is not None else "-")
                               for r in rows).most_common(8):
        print(f"  {c:10s} < {q:10s} {cnt}")


if __name__ == "__main__":
    main()
