"""Redesign B judging diagnostics (plan: docs/plan/재설계B_구현계획.md, 판정). holdout / val / crowd 200 / neg holdout only;
never test_public, examples_labeled or Fashionpedia val2020.

    python scripts/diag_segb.py --data D --runs_dir D/../runs_segb --ref S6_base S6_coco0 S6_lam2 ...
    python scripts/diag_segb.py --data D --runs_dir R --ref baseline S6_base          # final CNN (800, no flip) vs B
    add --loc to run the localisation pass (GPU, needs best.pt; cached in <run>/loc_{holdout,val}.npz)

Metrics (each run vs the reference, paired image bootstrap 1000, 95%; * = CI excludes 0):
Verdict (user 2026-11-23; design goals 1 location, 2 small objects, 3 confusables; performance goal 1 = confusables):
  automatic reject on a * drop of loc_topk (holdout or val) or micro AP diff < -0.03 (holdout or val); without --loc
  보류; otherwise 검수 대상: the confusable within-group AP (confWG_AP, per group wg_*) is judged by a person.
  confusable groups: upper [0-5, 9], lower+dress [6, 7, 8, 10, 11], head [14, 15], legs [21, 22], neck [16, 25]
  micro AP / macro AP (holdout, val)                    margin: diff >= -0.03 on both
  confusable within-group AP (holdout, val)             AP of a class over its positives + hard negatives (another
                                                        class of the same group present, this one absent); per group
                                                        and mean over the 5 groups (largeWG_AP = upper + lower only)
  small-garment recall (holdout, val)                   GT positives whose mask covers < 0.5% of the image, recall at
                                                        the run's own global threshold (best holdout micro F1);
                                                        also their rank among the class negatives (threshold free)
  crowd micro AP (crowd 200), per-class AP for classes with >= 10 positives
  empty-photo false positives (neg holdout)             share of images with any tag >= the run's threshold
  localisation (holdout, val; --loc)                    for each GT positive with a mask: share of the top-k cells
                                                        (same k as the tag pooling) inside the GT mask; also the
                                                        probability mass inside the mask; reported for all
                                                        positives, missed ones and small (< 0.5%) ones
The final CNN ("baseline") has no seg map: its localisation uses Grad-CAM of stages 3+4 (localisation_cam); its crowd / neg predictions come from
scripts/infer_hires_extra.py (800, no flip, like holdout / val).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

LARGE = ("upper", "lower+dress")            # largeWG_AP (kept for continuity)
# confusable groups (user 2026-11-23): confWG_AP = mean over all five; leg warmer (8 holdout positives) left out
GROUPS = {"upper": [0, 1, 2, 3, 4, 5, 9], "lower+dress": [6, 7, 8, 10, 11], "head": [14, 15], "legs": [21, 22],
          "neck": [16, 25]}
SMALL = 0.005
CROWD_MIN_POS = 10
NONINF = -0.03   # micro AP margin, user 2026-11-23 (was -0.002)
SPLITS = ("holdout", "val", "crowd", "neg_holdout")


def ap1(y, s):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    return float((tp / np.arange(1, len(y) + 1) * y).sum() / y.sum())


def micro_ap(y, s):
    return ap1(y.ravel(), s.ravel())


def macro_ap(y, s):
    return float(np.nanmean([ap1(y[:, c], s[:, c]) for c in range(y.shape[1])]))


def within_group_ap(y, s):
    out = []
    for cs in GROUPS.values():
        v = []
        for c in cs:
            others = [k for k in cs if k != c]
            keep = y[:, c] | y[:, others].any(1)
            v.append(ap1(y[keep, c], s[keep, c]))
        out.append(np.nanmean(v))
    return dict(zip(GROUPS, out))


def best_threshold(y, p):
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1 = []
    for t in grid:
        pr = p >= t
        tp = (pr & y).sum()
        f1.append(2 * tp / max(pr.sum() + y.sum(), 1))
    return float(grid[int(np.argmax(f1))])


def small_ranks(y, s, small):
    """rank (0..1) of each small positive among all negatives of its class; nan elsewhere."""
    r = np.full(y.shape, np.nan)
    for c in range(y.shape[1]):
        neg = np.sort(s[~y[:, c], c])
        pos = np.where(small[:, c])[0]
        if len(neg) < 5 or len(pos) == 0:
            continue
        r[pos, c] = (np.searchsorted(neg, s[pos, c], "left") + np.searchsorted(neg, s[pos, c], "right")) / 2 / len(neg)
    return r


# ---------------------------------------------------------------- loading
def load_preds(runs_dir, data, run, split):
    if run == "baseline":                       # crowd / neg from scripts/infer_hires_extra.py
        f = os.path.join(data, "baseline", "tta", f"preds_{split}_s800.npz")
    else:
        f = os.path.join(runs_dir, run, f"preds_{split}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=False)
    o = np.argsort(z["filenames"])
    p = 1 / (1 + np.exp(-z["logits"][o][:, :27].astype(np.float64)))
    return z["filenames"][o].astype(str), z["targets"][o][:, :27].astype(bool), p


def load_masks(data, split, names):
    """GT masks per (image, class < 27) and the area fraction table [N, 27] (nan = not positive / no mask)."""
    d = os.path.join(data, "diag_masks", split)
    area = np.full((len(names), 27), np.nan)
    masks = []
    for i, n in enumerate(names):
        f = os.path.join(d, n + ".npz")
        if not os.path.exists(f):
            masks.append({})
            continue
        z = np.load(f)
        gh, gw = z["shape"]
        k = len(z["cls"])
        arr = np.unpackbits(z["bits"])[: k * gh * gw].reshape(k, gh, gw).astype(bool)
        m = {int(c): a for c, a in zip(z["cls"].tolist(), arr) if c < 27}
        for c, a in m.items():
            area[i, c] = a.mean()
        masks.append(m)
    return masks, area


# ---------------------------------------------------------------- localisation (GPU)
def localisation(data, runs_dir, run, split, names, masks, device="cuda", limit=0):
    cache = os.path.join(runs_dir, run, f"loc_{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if (z["filenames"].astype(str) == names).all():
            return z["hit"], z["mass"]
    import cv2
    import torch
    from fashion_tagging.data.preprocess import timm_mean_std
    from fashion_tagging.segb.data import SegBDataset, collate
    from fashion_tagging.segb.model import SegB
    ck = os.path.join(runs_dir, run, "best.pt")
    if not os.path.exists(ck):
        return None
    st = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = st["config"]
    model = SegB(cfg["model"], False, 0.0, ratio=cfg["ratio"], min_k=cfg["min_k"])
    model.load_state_dict(st["model"])
    dev = torch.device(device)
    model = model.to(dev).eval().to(memory_format=torch.channels_last)
    mean, std = timm_mean_std(cfg["model"])
    root = os.path.join(data, "holdout" if split == "holdout" else "val", "images")
    ds = SegBDataset([os.path.join(root, n) for n in names], ["eval"] * len(names), np.zeros((len(names), 27), np.float32),
                     None, None, mean, std, False, cfg["max_long"], 0.0)
    ratio = np.asarray(cfg["ratio"])
    hit = np.full((len(names), 27), np.nan, np.float32)
    mass = np.full((len(names), 27), np.nan, np.float32)
    bad_aspect = 0
    with torch.no_grad():
        for i, n in enumerate(names):
            if not masks[i] or (limit and i >= limit):
                continue
            b = collate([ds[(i, cfg["test_short"])]], (0, 0))
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                o = model(b["x"].to(dev).contiguous(memory_format=torch.channels_last), b["sizes"].to(dev))
            H, W = b["sizes"][0].tolist()
            vh, vw = math.ceil(H / 4), math.ceil(W / 4)
            pr = o["probs"][0, :, :vh, :vw].float().cpu().numpy()
            nv = vh * vw
            for c, m in masks[i].items():
                if abs(m.shape[0] / m.shape[1] - H / W) > 0.1 * H / W:
                    bad_aspect += 1
                    continue
                cov = cv2.resize(m.astype(np.uint8), (vw, vh), interpolation=cv2.INTER_NEAREST).astype(bool)
                if not cov.any():
                    continue
                k = int(min(max(round(ratio[c] * nv), cfg["min_k"]), nv))
                pc = pr[c].ravel()
                top = np.argpartition(-pc, k - 1)[:k]
                hit[i, c] = cov.ravel()[top].mean()
                mass[i, c] = pc[cov.ravel()].sum() / max(pc.sum(), 1e-9)
    if bad_aspect:
        print(f"  [loc {run} {split}] skipped {bad_aspect} masks whose aspect ratio does not match the image", flush=True)
    if not limit:                                         # a partial (--loc_limit) pass must not become the cache
        np.savez_compressed(cache, filenames=names, hit=hit, mass=mass)
    return hit, mass


def localisation_cam(data, split, names, masks, ratio, min_k, device="cuda", limit=0):
    """Same metrics as localisation() for the final CNN (train_hires linear_ms: stage 3 stride 16 + stage 4
    stride 32, gapgmp). Class map = Grad-CAM of the class logit on each of the two pooled maps (ReLU of the
    gradient-weighted channel sum), bilinearly upsampled to the stride-4 grid of the padded input, summed over
    the two levels and cropped to the real image area. top-k uses the B runs' k (ratio[c] x cells, >= min_k).
    Cached in <data>/baseline/loc_{split}.npz. 800, no flip, like the holdout / val predictions."""
    cache = os.path.join(data, "baseline", f"loc_{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if (z["filenames"].astype(str) == names).all():
            return z["hit"], z["mass"]
    import cv2
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from train_hires import HiResDataset, build_hires_model, pad_collate, timm_mean_std, valid_mask, masked_pool
    st = torch.load(os.path.join(data, "baseline", "best.pt"), map_location="cpu", weights_only=False)
    cfg = dict(st["config"])
    assert cfg.get("head") == "linear_ms" and cfg.get("pool") == "gapgmp" and not cfg.get("ms_neck") \
        and not cfg.get("feat_norm"), "Grad-CAM path written for linear_ms / gapgmp without neck / feat_norm"
    model = build_hires_model(cfg, len(st["categories"]), False)
    model.load_state_dict(st["model"])
    dev = torch.device(device)
    model = model.to(dev).eval()
    mean, std = timm_mean_std(cfg["model"])
    root = os.path.join(data, "holdout" if split == "holdout" else "val", "images")
    ds = HiResDataset("/", [os.path.join(root, n) for n in names], np.zeros((len(names), 27), np.float32), mean, std,
                      False, cfg.get("max_long", 1333), 0.0, mirror=False, interp=cfg.get("interp") or "bilinear")
    short = cfg.get("test_short") or 800
    hit = np.full((len(names), 27), np.nan, np.float32)
    mass = np.full((len(names), 27), np.nan, np.float32)
    bad_aspect = 0
    for i, n in enumerate(names):
        if not masks[i] or (limit and i >= limit):
            continue
        x, _, sizes, _ = pad_collate([ds[(i, short)]])
        x, sizes = x.to(dev), sizes.to(dev)
        maps = model.net.forward_intermediates(x, indices=[-2, -1], intermediates_only=True)
        valids = [valid_mask(sizes, *m.shape[-2:], *x.shape[-2:]) for m in maps]
        logits = model.fc(model.norm(torch.cat([masked_pool(m, v, "gapgmp") for m, v in zip(maps, valids)], 1)))[0]
        H, W = sizes[0].tolist()
        vh, vw = math.ceil(H / 4), math.ceil(W / 4)
        gh, gw = x.shape[-2] // 4, x.shape[-1] // 4
        nv = vh * vw
        cls = [c for c, m in masks[i].items()]
        for j, c in enumerate(cls):
            m = masks[i][c]
            if abs(m.shape[0] / m.shape[1] - H / W) > 0.1 * H / W:
                bad_aspect += 1
                continue
            cov = cv2.resize(m.astype(np.uint8), (vw, vh), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not cov.any():
                continue
            grads = torch.autograd.grad(logits[c], maps, retain_graph=j < len(cls) - 1)
            cam = 0
            for a_, g_ in zip(maps, grads):
                w_ = g_.mean((2, 3), keepdim=True)
                cam = cam + F.interpolate(F.relu((w_ * a_).sum(1, keepdim=True)), size=(gh, gw), mode="bilinear",
                                          align_corners=False)
            pc = cam[0, 0, :vh, :vw].detach().float().cpu().numpy().ravel()
            k = int(min(max(round(ratio[c] * nv), min_k), nv))
            top = np.argpartition(-pc, k - 1)[:k]
            hit[i, c] = cov.ravel()[top].mean()
            mass[i, c] = pc[cov.ravel()].sum() / max(pc.sum(), 1e-9)
        if (i + 1) % 500 == 0:
            print(f"  [loc baseline {split}] {i + 1}/{len(names)}", flush=True)
    if bad_aspect:
        print(f"  [loc baseline {split}] skipped {bad_aspect} masks whose aspect ratio does not match the image", flush=True)
    if not limit:
        np.savez_compressed(cache, filenames=names, hit=hit, mass=mass)
    return hit, mass


# ---------------------------------------------------------------- per-run statistics on index subsets
def stats(D, idx):
    """D: dict split -> dict(y, p, thr, small, loc...). idx: dict split -> image index array. Returns a flat dict."""
    out = {}
    for sp in ("holdout", "val"):
        if sp not in D:
            continue
        d, i = D[sp], idx[sp]
        y, p = d["y"][i], d["p"][i]
        out[f"{sp}/microAP"] = micro_ap(y, p)
        out[f"{sp}/macroAP"] = macro_ap(y, p)
        wg = within_group_ap(y, p)
        out[f"{sp}/confWG_AP"] = float(np.mean(list(wg.values())))
        for g, v in wg.items():
            out[f"{sp}/wg_{g}"] = float(v)
        out[f"{sp}/largeWG_AP"] = float(np.mean([wg[g] for g in LARGE]))
        sm = d["small"][i]
        out[f"{sp}/small_recall"] = float((p >= d["thr"])[sm].sum() / max(sm.sum(), 1))
        r = d["rank"][i]
        out[f"{sp}/small_rank"] = float(np.nanmean(r)) if np.isfinite(r).any() else np.nan
        if d.get("hit") is not None:
            h, ms = d["hit"][i], d["mass"][i]
            miss = (p < d["thr"]) & np.isfinite(h)
            out[f"{sp}/loc_topk"] = float(np.nanmean(h))
            out[f"{sp}/loc_mass"] = float(np.nanmean(ms))
            sh = np.where(sm & np.isfinite(h), h, np.nan)
            out[f"{sp}/loc_topk_small"] = float(np.nanmean(sh)) if np.isfinite(sh).any() else np.nan
            out[f"{sp}/loc_topk_missed"] = float(np.nanmean(np.where(miss, h, np.nan))) if miss.any() else np.nan
    if "crowd" in D:
        d, i = D["crowd"], idx["crowd"]
        out["crowd/microAP"] = micro_ap(d["y"][i], d["p"][i])
    if "neg_holdout" in D:
        d, i = D["neg_holdout"], idx["neg_holdout"]
        out["neg/fp_image_rate"] = float((d["p"][i] >= d["thr"]).any(1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--ref", required=True, help="reference run (or 'baseline' = final CNN, 800 no flip)")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--loc", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loc_limit", type=int, default=0, help="testing only: localise the first N images")
    a = ap.parse_args()
    runs = [a.ref] + [r for r in a.runs if r != a.ref]

    P = {r: {sp: load_preds(a.runs_dir, a.data, r, sp) for sp in SPLITS} for r in runs}
    for r in runs:
        missing = [sp for sp in ("holdout", "val") if P[r][sp] is None]
        assert not missing, f"{r}: no predictions for {missing} (run not finished?)"
    names = {sp: P[a.ref][sp][0] for sp in SPLITS if P[a.ref][sp] is not None}
    masks, area = {}, {}
    for sp in ("holdout", "val"):
        masks[sp], area[sp] = load_masks(a.data, sp, names[sp])

    D = {}
    for r in runs:
        D[r] = {}
        thr = best_threshold(P[r]["holdout"][1], P[r]["holdout"][2])
        for sp in SPLITS:
            if P[r][sp] is None or sp not in names:
                continue
            f, y, p = P[r][sp]
            assert (f == names[sp]).all(), f"{r} {sp}: image list differs from the reference"
            d = {"y": y, "p": p, "thr": thr}
            if sp in ("holdout", "val"):
                d["small"] = y & (area[sp] < SMALL)
                d["rank"] = small_ranks(y, p, d["small"])
                d["hit"] = d["mass"] = None
                if a.loc and r == "baseline":
                    bc = json.load(open(os.path.join(a.runs_dir, next(q for q in runs if q != "baseline"),
                                                     "config.json")))
                    res = localisation_cam(a.data, sp, names[sp], masks[sp], np.asarray(bc["ratio"]), bc["min_k"],
                                           a.device, a.loc_limit)
                elif a.loc:
                    res = localisation(a.data, a.runs_dir, r, sp, names[sp], masks[sp], a.device, a.loc_limit)
                else:
                    res = None
                if res is not None:
                    d["hit"], d["mass"] = res
            D[r][sp] = d

    full = {sp: np.arange(len(names[sp])) for sp in names}
    S = {r: stats(D[r], full) for r in runs}
    rng = np.random.default_rng(0)
    boot = [{sp: rng.integers(0, len(names[sp]), len(names[sp])) for sp in names} for _ in range(a.boot)]
    B = {r: [stats(D[r], ix) for ix in boot] for r in runs}

    hd = D[a.ref]["holdout"]
    print(f"reference {a.ref} | holdout n={len(names['holdout'])}, val n={len(names['val'])}"
          + (f", crowd n={len(names['crowd'])}" if "crowd" in names else "")
          + f" | small positives holdout {int(hd['small'].sum())}, val {int(D[a.ref]['val']['small'].sum())}")
    keys = list(S[a.ref])
    for r in runs:
        keys += [k for k in S[r] if k not in keys]
    report = {"ref": a.ref, "runs": {}}
    for r in runs:
        print(f"\n== {r}  (threshold {D[r]['holdout']['thr']:.2f})")
        rr = {}
        for k in keys:
            if k not in S[r]:
                continue
            v = S[r][k]
            line = f"  {k:24s} {v:.4f}"
            rr[k] = {"value": v}
            if r != a.ref and k in S[a.ref] and np.isfinite(v) and np.isfinite(S[a.ref][k]):
                d = np.array([b[k] - c[k] for b, c in zip(B[r], B[a.ref]) if k in b and k in c])
                d = d[np.isfinite(d)]
                lo, hi = np.percentile(d, [2.5, 97.5])
                star = "*" if lo > 0 or hi < 0 else " "
                dv = v - S[a.ref][k]
                line += f" | {dv:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"
                rr[k].update(diff=dv, lo=lo, hi=hi, star=star == "*")
            print(line)
        if r != a.ref:
            # rule (user 2026-11-23): design goals 1 location, 2 small objects, 3 confusables; performance goal 1 =
            # confusables. Automatic: reject on a * drop of loc_topk (holdout or val) or micro AP below NONINF.
            # Otherwise the run goes to human review; confusable within-group AP is judged by a person, not here.
            lk = ("holdout/loc_topk", "val/loc_topk")
            has_loc = all("diff" in rr.get(k, {}) for k in lk)
            loc_up = [k for k in lk if rr.get(k, {}).get("star") and rr[k]["diff"] > 0]
            loc_down = [k for k in lk if rr.get(k, {}).get("star") and rr[k]["diff"] < 0]
            ni = all(rr.get(f"{sp}/microAP", {}).get("diff", 0) >= NONINF for sp in ("holdout", "val"))
            ck = [f"{sp}/{m}" for sp in ("holdout", "val") for m in ["confWG_AP"] + [f"wg_{g}" for g in GROUPS]]
            conf_up = [k for k in ck if rr.get(k, {}).get("star") and rr[k]["diff"] > 0]
            conf_down = [k for k in ck if rr.get(k, {}).get("star") and rr[k]["diff"] < 0]
            small = [k for k in ("holdout/small_recall", "val/small_recall", "holdout/loc_topk_small",
                                 "val/loc_topk_small") if rr.get(k, {}).get("star")]
            if loc_down:
                verdict = "탈락 (위치 ★ 하락)"
            elif not ni:
                verdict = f"탈락 (micro AP {NONINF:+.3f} 초과 하락)"
            elif not has_loc:
                verdict = "보류 (위치 미측정)"
            else:
                verdict = "검수 대상 (헷갈리는 분류는 사람이 판단)"
            fm = lambda ks: [f"{k}{'+' if rr[k]['diff'] > 0 else '-'}" for k in ks] or "-"
            print(f"  -> 위치 ★ 개선 {loc_up or '-'} 하락 {loc_down or '-'} | micro AP 허용({NONINF:+.3f}) {ni} | "
                  f"작은 물체 ★ {fm(small)} | 헷갈리는 그룹 ★ 개선 {conf_up or '-'} 하락 {conf_down or '-'} | 판정: {verdict}")
            rr["verdict"] = verdict
        report["runs"][r] = rr
    # crowd per class (>= 10 positives)
    if "crowd" in names:
        y = D[a.ref]["crowd"]["y"]
        cf = os.path.join(a.data, "baseline", "tta", "preds_crowd_s800.npz") if a.ref == "baseline" \
            else os.path.join(a.runs_dir, a.ref, "preds_crowd.npz")
        cats = [str(c) for c in np.load(cf)["categories"]]
        cs = [c for c in range(27) if y[:, c].sum() >= CROWD_MIN_POS]
        rs = [r for r in runs if "crowd" in D[r]]
        print(f"\ncrowd per-class AP (>= {CROWD_MIN_POS} positives): " + " | ".join(rs))
        for c in cs:
            print(f"  {cats[c].split(',')[0][:20]:20s} n={int(y[:, c].sum()):3d} " +
                  " ".join(f"{ap1(y[:, c], D[r]['crowd']['p'][:, c]):.3f}" for r in rs))
    os.makedirs(os.path.join(a.runs_dir, "diag"), exist_ok=True)
    out = os.path.join(a.runs_dir, "diag", f"diag_vs_{a.ref}.json")
    json.dump(report, open(out, "w"), indent=1, default=float)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
