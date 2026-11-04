"""Contact sheets of prediction errors, with a fixed and recorded selection rule.

Input is the .npz written by metrics.save_predictions (filenames, targets, logits,
categories). Images come from an extracted folder (root/images/*.jpg) or a ZIP.

Selection modes (deterministic; nothing is hand-picked):
  worst   images with the lowest image-wise F1, then most errors, then file name
  fn      with --class C: positives of C with the LOWEST probability (worst misses)
  fp      with --class C: negatives of C with the HIGHEST probability (worst false alarms)
  random  random images that have at least one error, seeded (--seed)
  ok      random correctly predicted images, seeded (to see what success looks like)

Examples
  python scripts/visualize_errors.py --pred preds/exp1_val.npz --images MAT4091_fashion_val.zip \
      --zip_prefix val --mode worst --out vis/exp1
  python scripts/visualize_errors.py --pred preds/exp1_val.npz --images data/val \
      --mode fn --class watch --k 12 --out vis/exp1
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # repo/src
from fashion_tagging.data.dataset import FashionDataset  # noqa: E402
from fashion_tagging.evaluation.metrics import binarize, load_predictions, sigmoid  # noqa: E402

PANEL_W, IMG_H, TEXT_H, PAD = 300, 300, 150, 10
RED, BLUE, BLACK, GRAY = (200, 0, 0), (0, 70, 200), (0, 0, 0), (110, 110, 110)


def image_f1(y_true, y_pred):
    tp = (y_true & y_pred).sum(1).astype(float)
    denom = 2 * tp + (~y_true & y_pred).sum(1) + (y_true & ~y_pred).sum(1)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0)


def select(mode, y_true, probs, y_pred, names, k=12, cls=None, seed=42):
    """Return (indices, rule_text, n_candidates)."""
    y_true, y_pred = y_true.astype(bool), y_pred.astype(bool)
    n_err = (y_true != y_pred).sum(1)
    if mode == "worst":
        f1 = image_f1(y_true, y_pred)
        cand = np.flatnonzero(n_err > 0)
        order = sorted(cand, key=lambda i: (f1[i], -n_err[i], names[i]))
        rule = "lowest image F1 -> most errors -> file name"
    elif mode in ("fn", "fp"):
        if cls is None:
            raise ValueError("--class is required for fn/fp")
        if mode == "fn":
            cand = np.flatnonzero(y_true[:, cls] & ~y_pred[:, cls])
            order = sorted(cand, key=lambda i: (probs[i, cls], names[i]))
            rule = "positives of the class with the lowest probability"
        else:
            cand = np.flatnonzero(~y_true[:, cls] & y_pred[:, cls])
            order = sorted(cand, key=lambda i: (-probs[i, cls], names[i]))
            rule = "negatives of the class with the highest probability"
    elif mode in ("random", "ok"):
        cand = np.flatnonzero(n_err > 0) if mode == "random" else np.flatnonzero(n_err == 0)
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(cand))
        rule = f"random {'images with errors' if mode == 'random' else 'correct images'} (seed {seed})"
    else:
        raise ValueError(f"unknown mode {mode}")
    return [int(i) for i in order[:k]], rule, len(cand)


def _font(size):
    for p in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\malgun.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw, text, font, width):
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur] if cur else lines


def _panel(img, name, gt, pred, probs, cats, font):
    panel = Image.new("RGB", (PANEL_W, IMG_H + TEXT_H), "white")
    thumb = img.copy()
    thumb.thumbnail((PANEL_W - 2, IMG_H - 2))
    panel.paste(thumb, ((PANEL_W - thumb.width) // 2, (IMG_H - thumb.height) // 2))
    d = ImageDraw.Draw(panel)
    d.rectangle([0, 0, PANEL_W - 1, IMG_H + TEXT_H - 1], outline=(190, 190, 190))
    y = IMG_H + 3
    miss = [c for c in np.flatnonzero(gt & ~pred)]
    extra = [c for c in np.flatnonzero(~gt & pred)]
    blocks = [(f"{name[:14]}  errors: {len(miss) + len(extra)}", GRAY),
              ("GT: " + (", ".join(cats[c] for c in np.flatnonzero(gt)) or "(none)"), BLACK),
              ("MISS: " + (", ".join(f"{cats[c]} {probs[c]:.2f}" for c in miss) or "-"), RED),
              ("EXTRA: " + (", ".join(f"{cats[c]} {probs[c]:.2f}" for c in extra) or "-"), BLUE)]
    for text, color in blocks:
        for line in _wrap(d, text, font, PANEL_W - 8)[:3]:
            d.text((4, y), line, fill=color, font=font)
            y += 13
    return panel


def render_sheet(loader, idx, names, y_true, probs, y_pred, cats, title, cols=4):
    font, tfont = _font(11), _font(15)
    rows = max(1, -(-len(idx) // cols))
    head = 52
    sheet = Image.new("RGB", (cols * (PANEL_W + PAD) + PAD, head + rows * (IMG_H + TEXT_H + PAD) + PAD), "white")
    d = ImageDraw.Draw(sheet)
    for n, line in enumerate(title.split("\n")):
        d.text((PAD, 6 + 18 * n), line, fill=BLACK, font=tfont)
    for n, i in enumerate(idx):
        p = _panel(loader(names[i]), str(names[i]), y_true[i].astype(bool), y_pred[i], probs[i], cats, font)
        sheet.paste(p, (PAD + (n % cols) * (PANEL_W + PAD), head + (n // cols) * (IMG_H + TEXT_H + PAD)))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--images", required=True, help="folder that contains images/ , or a .zip")
    ap.add_argument("--zip_prefix", default=None, help="train or val when --images is a ZIP")
    ap.add_argument("--mode", default="worst", choices=["worst", "fn", "fp", "random", "ok"])
    ap.add_argument("--class", dest="cls", default=None, help="exact class name, e.g. 'shirt, blouse'")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names, y_true, logits, cats = load_predictions(a.pred)
    names = [str(n) for n in names]
    probs = sigmoid(logits)
    y_pred = binarize(probs, a.threshold)
    cls = None
    if a.cls is not None:
        if a.cls not in cats:
            raise SystemExit(f"unknown class {a.cls!r}; use one of: {cats}")
        cls = cats.index(a.cls)

    idx, rule, n_cand = select(a.mode, y_true, probs, y_pred, names, a.k, cls, a.seed)
    if not idx:
        raise SystemExit("no images match this selection")
    ds = FashionDataset(a.images, names, np.zeros((len(names), len(cats))), zip_prefix=a.zip_prefix)
    what = a.mode + (f" / {a.cls}" if cls is not None else "")
    title = (f"{what}  |  threshold {a.threshold}  |  showing {len(idx)} of {n_cand} matching ({len(names)} images)\n"
             f"selection: {rule}.  red MISS = false negative, blue EXTRA = false positive (number = probability)")
    sheet = render_sheet(ds.load_image, idx, names, y_true, probs, y_pred, cats, title, a.cols)

    os.makedirs(a.out, exist_ok=True)
    stem = a.mode + (f"_{cls}" if cls is not None else "")
    sheet.save(os.path.join(a.out, f"{stem}.jpg"), quality=90)
    record = [{"file": names[i], "gt": [cats[c] for c in np.flatnonzero(y_true[i])],
               "pred": [cats[c] for c in np.flatnonzero(y_pred[i])],
               "probs": {cats[c]: round(float(probs[i, c]), 4) for c in range(len(cats))}} for i in idx]
    with open(os.path.join(a.out, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump({"mode": a.mode, "class": a.cls, "threshold": a.threshold, "rule": rule,
                   "candidates": n_cand, "seed": a.seed, "items": record}, f, ensure_ascii=False, indent=1)
    print(f"saved {stem}.jpg / .json in {a.out}  ({len(idx)} of {n_cand})")


if __name__ == "__main__":
    main()
