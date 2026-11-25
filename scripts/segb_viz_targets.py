"""D6: draw the per-cell training targets of redesign B for a few images of each data kind (check before training).

    python scripts/segb_viz_targets.py --data D --prep D/segb_prep --out D/../vis/b_targets --n 5

Per image: image | upper+lower groups | legs+accessories groups. Colour = class with the largest target in the
cell (black = none), grey = ignored cell, cells supervised by the FashionFormer teacher (not GT) are dimmed.
"""
import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fashion_tagging.segb.data import SegBDataset  # noqa: E402
from fashion_tagging.segb.spec import GROUP_SIZES, GROUPS  # noqa: E402

PAL = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
                [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180],
                [0, 0, 128], [128, 128, 128], [255, 99, 71], [100, 149, 237], [154, 205, 50], [255, 140, 0],
                [186, 85, 211], [32, 178, 170], [255, 20, 147]], np.uint8)
OFF = np.cumsum([0] + GROUP_SIZES)[:-1]


def paint(t, v, hard, groups):
    h, w = t.shape[1:]
    img = np.full((h, w, 3), 90, np.uint8)
    for gi in groups:
        tg = t[OFF[gi]:OFF[gi] + GROUP_SIZES[gi]]
        arg = tg.argmax(0)
        for j, c in enumerate([None] + GROUPS[gi]):
            m = v[gi] & (arg == j)
            if c is None:
                if gi == groups[0]:
                    img[m] = 0
                continue
            col = PAL[c].astype(np.float32)
            img[m] = np.where(hard[gi][m][:, None], col, col * 0.5).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rl = lambda f: [l.strip() for l in open(os.path.join(a.prep, f)) if l.strip()]
    fp = [os.path.join(a.data, "train_inner", "images", n) for n in rl("list_fp.txt")]
    E = os.path.join(a.data, "external", "external_merged")
    ext = [os.path.join(E, "images", f) for f in rl("list_ext.txt") if f.startswith("lvis_")]
    crowd = [p for p in rl("list_coco.txt") if "coco_crowd" in p]
    rnd = random.Random(1)
    picks = [("fp", rnd.sample(fp, a.n)), ("coco", rnd.sample(crowd, a.n)), ("ext", rnd.sample(ext, a.n)),
             ("empty", rnd.sample(rl("list_empty.txt"), 2))]
    paths = [p for _, ps in picks for p in ps]
    kinds = [k for k, ps in picks for _ in ps]
    ds = SegBDataset(paths, kinds, np.zeros((len(paths), 27), np.float32), None, a.prep, (0, 0, 0), (1, 1, 1), True,
                     flip=0.0)
    rows = []
    for i, (p, k) in enumerate(zip(paths, kinds)):
        it = ds[(i, 800)]
        x = (it["x"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
        t, v, hard = it["t"].numpy(), it["v"].numpy(), it["hard"].numpy()
        H, W = x.shape[:2]
        A = cv2.resize(paint(t, v, hard, [0, 1]), (W, H), interpolation=cv2.INTER_NEAREST)[..., ::-1]
        B = cv2.resize(paint(t, v, hard, [2, 3]), (W, H), interpolation=cv2.INTER_NEAREST)[..., ::-1]
        row = np.hstack([x, (0.4 * x + 0.6 * A).astype(np.uint8), (0.4 * x + 0.6 * B).astype(np.uint8)])
        s = 360 / H
        row = cv2.resize(row, (int(row.shape[1] * s), 360))
        bar = np.zeros((26, row.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{k} | {os.path.basename(p)} | mid: upper+lower groups, right: legs+accessories",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.vstack([bar, row]))
    Wm = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 6), (0, Wm - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
    cv2.imwrite(os.path.join(a.out, "targets.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("wrote", os.path.join(a.out, "targets.jpg"), len(rows))


if __name__ == "__main__":
    main()
