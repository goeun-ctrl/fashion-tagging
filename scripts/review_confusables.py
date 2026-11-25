"""Review sheets for the confusable categories (human review step of the redesign-B verdict, scripts/diag_segb.py).

For each confusable group (upper, lower+dress, head, legs, neck) an image counts as "right" for a model when every
GT-positive class of the group scores above every GT-negative class of the group. The sheet shows the images where
exactly one of the two models is right (both directions), the largest score gaps first:
  [image + GT masks of the group]  [run: map of its top class in the group]  [ref: same]
B runs show their segmentation probability map; the final CNN ("baseline") shows Grad-CAM (stages 3 + 4), as in
diag_segb.localisation_cam. holdout / val only; never test_public, examples_labeled or Fashionpedia val2020.

    python scripts/review_confusables.py --data D --runs_dir R --ref baseline --run S6_base --split holdout \
        --per 6 --device mps --out R/review
Writes <out>/<run>_vs_<ref>_<split>_<group>.jpg and a counts table <out>/<run>_vs_<ref>_<split>.txt.
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_segb import GROUPS, ROOT, load_masks, load_preds  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "src"))
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
PH = 300                                                   # panel height in pixels


def font(size):
    for f in (FONT, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def group_right(y, p, cs):
    """bool right [N], margin [N] (min positive score - max negative score inside the group), has [N]."""
    yg, pg = y[:, cs], p[:, cs]
    has = yg.any(1)
    pos = np.where(yg, pg, np.inf).min(1)
    neg = np.where(~yg, pg, -np.inf).max(1)
    neg = np.where(np.isfinite(neg), neg, 0.0)              # every class of the group present -> compare with 0
    m = pos - neg
    return has & (m > 0), np.where(has, m, np.nan), has


class BMaps:
    def __init__(self, data, runs_dir, run, split, names, device):
        import torch
        from fashion_tagging.data.preprocess import timm_mean_std
        from fashion_tagging.segb.data import SegBDataset, collate
        from fashion_tagging.segb.model import SegB
        st = torch.load(os.path.join(runs_dir, run, "best.pt"), map_location="cpu", weights_only=False)
        self.cfg = cfg = st["config"]
        self.model = SegB(cfg["model"], False, 0.0, ratio=cfg["ratio"], min_k=cfg["min_k"])
        self.model.load_state_dict(st["model"])
        self.dev = torch.device(device)
        self.model = self.model.to(self.dev).eval()
        mean, std = timm_mean_std(cfg["model"])
        root = os.path.join(data, split, "images")
        self.ds = SegBDataset([os.path.join(root, n) for n in names], ["eval"] * len(names),
                              np.zeros((len(names), 27), np.float32), None, None, mean, std, False, cfg["max_long"], 0.0)
        self.collate, self.torch = collate, torch

    def __call__(self, i, c):
        with self.torch.no_grad():
            b = self.collate([self.ds[(i, self.cfg["test_short"])]], (0, 0))
            o = self.model(b["x"].to(self.dev), b["sizes"].to(self.dev))
        H, W = b["sizes"][0].tolist()
        return o["probs"][0, c, :math.ceil(H / 4), :math.ceil(W / 4)].float().cpu().numpy()


class CamMaps:
    def __init__(self, data, split, names, device):
        import torch
        from train_hires import HiResDataset, build_hires_model, timm_mean_std
        st = torch.load(os.path.join(data, "baseline", "best.pt"), map_location="cpu", weights_only=False)
        self.cfg = cfg = dict(st["config"])
        self.model = build_hires_model(cfg, len(st["categories"]), False)
        self.model.load_state_dict(st["model"])
        self.dev = torch.device(device)
        self.model = self.model.to(self.dev).eval()
        mean, std = timm_mean_std(cfg["model"])
        root = os.path.join(data, split, "images")
        self.ds = HiResDataset("/", [os.path.join(root, n) for n in names], np.zeros((len(names), 27), np.float32),
                               mean, std, False, cfg.get("max_long", 1333), 0.0, mirror=False,
                               interp=cfg.get("interp") or "bilinear")
        self.torch = torch

    def __call__(self, i, c):
        import torch.nn.functional as F
        from train_hires import masked_pool, pad_collate, valid_mask
        x, _, sizes, _ = pad_collate([self.ds[(i, self.cfg.get("test_short") or 800)]])
        x, sizes = x.to(self.dev), sizes.to(self.dev)
        m = self.model
        maps = m.net.forward_intermediates(x, indices=[-2, -1], intermediates_only=True)
        valids = [valid_mask(sizes, *a.shape[-2:], *x.shape[-2:]) for a in maps]
        logit = m.fc(m.norm(self.torch.cat([masked_pool(a, v, "gapgmp") for a, v in zip(maps, valids)], 1)))[0, c]
        grads = self.torch.autograd.grad(logit, maps)
        gh, gw = x.shape[-2] // 4, x.shape[-1] // 4
        cam = 0
        for a, g in zip(maps, grads):
            cam = cam + F.interpolate(F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True)), size=(gh, gw),
                                      mode="bilinear", align_corners=False)
        H, W = sizes[0].tolist()
        return cam[0, 0, :math.ceil(H / 4), :math.ceil(W / 4)].detach().float().cpu().numpy()


def overlay(img, m, color, norm):
    """heat overlay of map m (any size) on img; norm: divide by max (Grad-CAM) or keep 0..1 (probabilities)."""
    m = np.asarray(Image.fromarray(m.astype(np.float32)).resize(img.size, Image.BILINEAR))
    if norm:
        m = m / max(float(m.max()), 1e-9)
    m = np.clip(m, 0, 1)[..., None]
    a = np.asarray(img, np.float32) * (1 - 0.6 * m) + np.array(color, np.float32) * 0.6 * m
    return Image.fromarray(a.astype(np.uint8))


def gt_panel(img, masks, cs, cats, f):
    out = img.copy()
    d = ImageDraw.Draw(out)
    cols = [(0, 200, 80), (0, 150, 255), (255, 170, 0), (200, 0, 200)]
    k = 0
    for c in cs:
        if c not in masks:
            continue
        mk = Image.fromarray(masks[c].astype(np.uint8) * 255).resize(img.size, Image.NEAREST)
        edge = np.asarray(mk.filter(__import__("PIL.ImageFilter", fromlist=["FIND_EDGES"]).FIND_EDGES)) > 0
        a = np.asarray(out).copy()
        a[edge] = cols[k % len(cols)]
        out = Image.fromarray(a)
        d = ImageDraw.Draw(out)
        d.text((4, 4 + 18 * k), cats[c], fill=cols[k % len(cols)], font=f, stroke_width=2, stroke_fill=(0, 0, 0))
        k += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--ref", required=True, help="'baseline' (final CNN) or a B run")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="holdout", choices=["holdout", "val"])
    ap.add_argument("--per", type=int, default=6, help="cases per direction per group")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp = a.split
    fr, y, pr = load_preds(a.runs_dir, a.data, a.ref, sp)
    fb, yb, pb = load_preds(a.runs_dir, a.data, a.run, sp)
    assert (fr == fb).all()
    cf = os.path.join(a.runs_dir, a.run, f"preds_{sp}.npz")
    cats = [str(c).split(",")[0] for c in np.load(cf)["categories"]][:27]
    masks, _ = load_masks(a.data, sp, fr)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.run}_vs_{a.ref}_{sp}"
    lines = [f"{tag}: images with a GT positive in the group; right = all positives above all negatives in the group",
             f"{'group':12s} {'n':>5s} {'both':>6s} {'only ' + a.run:>18s} {'only ' + a.ref:>18s} {'neither':>8s}"]
    maps = {}

    def getmap(who):
        if who not in maps:
            maps[who] = CamMaps(a.data, sp, fr, a.device) if who == "baseline" else \
                BMaps(a.data, a.runs_dir, who, sp, fr, a.device)
        return maps[who]

    f, fs = font(16), font(13)
    for g in a.groups.split(","):
        cs = GROUPS[g]
        rr, mr, has = group_right(y, pr, cs)
        rb, mb, _ = group_right(y, pb, cs)
        n = int(has.sum())
        lines.append(f"{g:12s} {n:5d} {int((rr & rb).sum()):6d} {int((rb & ~rr & has).sum()):18d} "
                     f"{int((rr & ~rb & has).sum()):18d} {int((~rr & ~rb & has).sum()):8d}")
        rows = []
        for title, sel, gap in ((f"{a.run}만 맞음", rb & ~rr & has, mb - mr), (f"{a.ref}만 맞음", rr & ~rb & has, mr - mb)):
            idx = np.where(sel)[0]
            idx = idx[np.argsort(-gap[idx])][:a.per]
            rows.append((title, idx))
        tiles = []
        for title, idx in rows:
            tiles.append(("head", f"{title}  ({len(idx)}장)" if len(idx) else f"{title}  (사례 없음)"))
            for i in idx:
                img = Image.open(os.path.join(a.data, sp, "images", fr[i])).convert("RGB")
                img = img.resize((max(1, round(img.width * PH / img.height)), PH))
                panels = [gt_panel(img, masks[i], cs, cats, fs)]
                cap = []
                for who, p in ((a.run, pb), (a.ref, pr)):
                    c = cs[int(np.argmax(p[i, cs]))]
                    q = overlay(img, getmap(who)(i, c), (255, 40, 40), who == "baseline")
                    ImageDraw.Draw(q).text((4, 4), f"{who}: {cats[c]}", fill="white", font=fs, stroke_width=2,
                                           stroke_fill=(0, 0, 0))
                    panels.append(q)
                    cap.append(f"{who}: top={cats[c]}  " + " ".join(f"{cats[k]} {p[i, k]:.2f}" for k in cs))
                gt = ", ".join(cats[k] for k in cs if y[i, k]) or "-"
                w = sum(q.width for q in panels) + 8 * (len(panels) - 1)
                t = Image.new("RGB", (max(w, 900), PH + 64), "white")
                x = 0
                for q in panels:
                    t.paste(q, (x, 0))
                    x += q.width + 8
                d = ImageDraw.Draw(t)
                d.text((4, PH + 2), f"{fr[i]}  GT: {gt}", fill="black", font=fs)
                d.text((4, PH + 22), cap[0], fill=(160, 0, 0), font=fs)
                d.text((4, PH + 42), cap[1], fill=(0, 0, 160), font=fs)
                tiles.append(("img", t))
        W = max([t.width for k, t in tiles if k == "img"] + [900])
        H = sum(t.height + 6 if k == "img" else 34 for k, t in tiles) + 40
        sheet = Image.new("RGB", (W + 16, H), (235, 235, 235))
        d = ImageDraw.Draw(sheet)
        d.text((8, 8), f"{g}  ({tag})  panels: GT masks | {a.run} map of its top class | {a.ref} map of its top class",
               fill="black", font=f)
        yy = 40
        for k, t in tiles:
            if k == "head":
                d.text((8, yy + 6), t, fill="black", font=f)
                yy += 34
            else:
                sheet.paste(t, (8, yy))
                yy += t.height + 6
        out = os.path.join(a.out, f"{tag}_{g}.jpg")
        sheet.save(out, quality=85)
        print("wrote", out, flush=True)
    txt = os.path.join(a.out, f"{tag}.txt")
    open(txt, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
