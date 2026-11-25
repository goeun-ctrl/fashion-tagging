"""Tag pooling from logits instead of group-softmax probabilities (no retraining, same B map). Analysis only.

Per cell and class c of softmax group g (logits z, "none" channel z_0):
  orig      : p_c = softmax_g(z)_c, tag = mean of the top-k cells          (current model)
  lo_mean   : d_c = z_c - z_0 (c vs "none" only, other garment classes do not compete);
              tag = sigmoid(mean of the top-k d_c)
  sig_mean  : tag = mean of the top-k sigmoid(d_c)                        (same, averaged in probability space)
  fg_split  : tag = mean of the top-k [ (1 - p_none) x sigmoid(z_c - logsumexp(other garment classes)) ]
              (garment presence x c vs the other garments, no shared normalisation with "none")
  blend     : sqrt(orig x lo_mean)
k = the class's own k (ratio_c x valid cells, >= min_k); top-k cells are chosen on each variant's own value.
Scores are cached per split (<runs_dir>/<run>/logitpool_<split>.npz). Holdout = selection split, val confirms.

    python -I scripts/logit_pool.py --data D --runs_dir R --run S6_base --split holdout --device mps
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_segb import GROUPS as JUDGE, load_preds, macro_ap, micro_ap, within_group_ap  # noqa: E402
from review_confusables import group_right  # noqa: E402

NAMES = ("orig", "lo_mean", "sig_mean", "fg_split")


def topk_mean(v, k):
    return np.partition(v, len(v) - k)[len(v) - k:].mean()


def compute(a, names):
    import torch
    from fashion_tagging.data.preprocess import timm_mean_std
    from fashion_tagging.segb.data import SegBDataset, collate
    from fashion_tagging.segb.model import SegB
    from fashion_tagging.segb.spec import GROUPS, GROUP_SIZES

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
    out = np.zeros((len(NAMES), N, 27))
    with torch.no_grad():
        for i in range(N):
            b = collate([ds[(i, cfg["test_short"])]], (0, 0))
            o = model(b["x"].to(dev), b["sizes"].to(dev))
            H, W = b["sizes"][0].tolist()
            vh, vw = math.ceil(H / 4), math.ceil(W / 4)
            z = o["seg"][0, :, :vh, :vw].float().cpu().numpy().reshape(o["seg"].shape[1], -1)
            p = o["probs"][0, :, :vh, :vw].float().cpu().numpy().reshape(27, -1)
            nv = z.shape[1]
            s = 0
            for cs, n in zip(GROUPS, GROUP_SIZES):
                zg = z[s:s + n]
                pnone = np.exp(zg[0] - np.logaddexp.reduce(zg, 0))
                for j, c in enumerate(cs):
                    k = int(min(max(round(ratio[c] * nv), cfg["min_k"]), nv))
                    d = zg[j + 1] - zg[0]
                    others = np.delete(zg[1:], j, 0)
                    vs = zg[j + 1] - np.logaddexp.reduce(others, 0) if len(others) else np.full(nv, 30.0)
                    out[0, i, c] = topk_mean(p[c], k)
                    out[1, i, c] = 1 / (1 + np.exp(-topk_mean(d, k)))
                    out[2, i, c] = topk_mean(1 / (1 + np.exp(-d)), k)
                    out[3, i, c] = topk_mean((1 - pnone) / (1 + np.exp(-vs)), k)
                s += n
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{N}", flush=True)
    return out


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
    cache = os.path.join(a.runs_dir, a.run, f"logitpool_{a.split}.npz")
    if os.path.exists(cache) and (np.load(cache)["filenames"].astype(str) == names).all():
        sc = np.load(cache)["scores"]
    else:
        sc = compute(a, names)
        np.savez_compressed(cache, filenames=names, scores=sc)
    print(f"{a.run} {a.split}: {len(names)} images; recomputed orig vs saved preds max |diff| "
          f"{np.abs(sc[0] - saved).max():.4f}")
    variants = list(zip(NAMES, sc)) + [("blend", np.sqrt(sc[0] * sc[1]))]
    up = JUDGE["upper"]
    lay = y[:, up].sum(1) >= 2
    print(f"{'variant':10s} {'microAP':>8s} {'macroAP':>8s} {'confWG':>7s} "
          + " ".join(f"{g[:8]:>8s}" for g in JUDGE) + f" {'up-right':>9s} {'layered':>8s} {'single':>7s}")
    for name, s in variants:
        wg = within_group_ap(y, s)
        r, _, has = group_right(y, s, up)
        print(f"{name:10s} {micro_ap(y, s):8.4f} {macro_ap(y, s):8.4f} {np.mean(list(wg.values())):7.4f} "
              + " ".join(f"{wg[g]:8.4f}" for g in JUDGE)
              + f" {r[has].mean():9.3f} {r[has & lay].mean():8.3f} {r[has & ~lay].mean():7.3f}")


if __name__ == "__main__":
    main()
