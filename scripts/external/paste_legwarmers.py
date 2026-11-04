"""Copy-paste augmentation for leg warmers from Fashionpedia masks, using train_inner images only.

    python scripts/external/paste_legwarmers.py --ann ~/fashion_tagging/external/fashionpedia/instances_attributes_train2020.json \
        --train_dir data/train --out data/external/legwarmer_paste --per_instance 10

Sources: every Fashionpedia "leg warmer" instance whose image is in data/splits/train_inner.txt (holdout and the
official val are never read). Backgrounds: train_inner images that have a shoe and no leg warmer. Each composite
pastes one source instance above every shoe of the background (lower leg), scaled to the shoe width, randomly
flipped, with a slightly feathered mask. Labels = the background's own 27 labels + leg warmer = 1, so every label
is known (no -1).

Output (--out): categories.txt, labels.csv (file_name + 27 columns), images/, sources.csv (source image/annotation,
background image, scale), review.jpg, overlap.txt (dHash check against holdout / val / public test).
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from fashion_tagging.data.labels import load_split  # noqa: E402
from fashion_tagging.paths import SPLITS_DIR  # noqa: E402


def rle_decode(seg, h, w):
    """COCO RLE (compressed string or uncompressed list) -> bool mask [h, w]."""
    counts = seg["counts"]
    if isinstance(counts, str):                       # COCO compressed counts (LEB128-like)
        vals, i, s = [], 0, counts.encode()
        while i < len(s):
            x, k, more = 0, 0, True
            while more:
                c = s[i] - 48
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                i, k = i + 1, k + 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(vals) > 2:
                x += vals[-2]
            vals.append(x)
        counts = vals
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in counts:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape(w, h).T                        # column-major


def poly_mask(polys, h, w):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        if len(p) >= 6:
            d.polygon(list(zip(p[0::2], p[1::2])), fill=255)
    return np.asarray(m) > 0


def cutout(img, seg, h, w):
    """RGBA crop of one instance."""
    m = poly_mask(seg, h, w) if isinstance(seg, list) else rle_decode(seg, h, w)
    ys, xs = np.nonzero(m)
    if len(ys) < 50:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = img.convert("RGBA").crop((x0, y0, x1, y1))
    alpha = Image.fromarray((m[y0:y1, x0:x1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
    rgba.putalpha(alpha)
    return rgba


def dhash(im):
    g = im.convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return int(np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view(">u8")[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, help="Fashionpedia instances_attributes_train2020.json")
    ap.add_argument("--train_dir", required=True, help="our train folder (categories.txt, labels.csv, images/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_instance", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_hashes", default="", help="optional hash_cache_ours.npz from prepare_external.py")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)

    cats, names, targets = load_split(a.train_dir, "train")
    lab = dict(zip(names, targets))
    inner = set(l.strip() for l in open(SPLITS_DIR / "train_inner.txt") if l.strip())
    d = json.load(open(os.path.expanduser(a.ann)))
    cname = {c["id"]: c["name"] for c in d["categories"]}
    LW = next(i for i, n in cname.items() if n == "leg warmer")
    SHOE = next(i for i, n in cname.items() if n == "shoe")
    im = {i["id"]: i for i in d["images"]}
    by_img = {}
    for ann in d["annotations"]:
        by_img.setdefault(ann["image_id"], []).append(ann)

    sources = [x for x in d["annotations"] if x["category_id"] == LW and im[x["image_id"]]["file_name"] in inner]
    lw_imgs = {im[x["image_id"]]["file_name"] for x in d["annotations"] if x["category_id"] == LW}
    backgrounds = [iid for iid, anns in by_img.items()
                   if im[iid]["file_name"] in inner and im[iid]["file_name"] not in lw_imgs
                   and any(x["category_id"] == SHOE and x["bbox"][2] > 20 for x in anns)]
    assert all(im[x["image_id"]]["file_name"] in inner for x in sources)
    print(f"sources {len(sources)} leg warmers (train_inner only), backgrounds {len(backgrounds)}", flush=True)

    li = cats.index("leg warmer")
    rows, prov, made = [], [], []
    for s in sources:
        si = im[s["image_id"]]
        with Image.open(os.path.join(a.train_dir, "images", si["file_name"])) as simg:
            # masks are in the stored pixel frame, so the cut is taken before any EXIF rotation
            piece = cutout(simg.convert("RGB"), s["segmentation"], si["height"], si["width"])
        if piece is None:
            continue
        for k in range(a.per_instance):
            bid = rng.choice(backgrounds)
            bi = im[bid]
            with Image.open(os.path.join(a.train_dir, "images", bi["file_name"])) as bimg:
                rotated = bimg.getexif().get(0x0112, 1) != 1   # EXIF orientation: training would rotate it
                canvas = bimg.convert("RGB")
            if rotated or canvas.size != (bi["width"], bi["height"]):   # annotation frame must match the pixels
                continue
            shoes = [x["bbox"] for x in by_img[bid] if x["category_id"] == SHOE and x["bbox"][2] > 20]
            p = piece.transpose(Image.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else piece
            scale_r = rng.uniform(0.8, 1.2)
            for (x, y, w, h) in shoes[:2]:
                nw = max(8, int(w * scale_r))
                nh = max(8, int(p.height * nw / p.width))
                q = p.resize((nw, nh), Image.BICUBIC)
                px = int(x + w / 2 - nw / 2)
                py = int(y + 0.15 * h - nh)                     # bottom edge slightly inside the shoe top
                if py < 0:
                    q = q.crop((0, -py, nw, nh))
                    py = 0
                canvas.paste(q, (px, py), q)
            fn = f"lwpaste_{s['id']}_{k}_{os.path.splitext(bi['file_name'])[0]}.jpg"
            canvas.save(os.path.join(out, "images", fn), quality=92)
            t = lab[bi["file_name"]].astype(int).copy()
            t[li] = 1
            rows.append([fn, *t.tolist()])
            prov.append([fn, si["file_name"], s["id"], bi["file_name"], f"{scale_r:.2f}", len(shoes[:2])])
            made.append(fn)
    print(f"made {len(made)} composites", flush=True)

    shutil.copyfile(os.path.join(a.train_dir, "categories.txt"), os.path.join(out, "categories.txt"))
    with open(os.path.join(out, "labels.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["file_name", *cats], *rows])
    with open(os.path.join(out, "sources.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["file_name", "source_image", "source_ann_id", "background_image", "scale", "shoes"],
                                  *prov])

    # overlap check: composites against holdout / val / public test (train backgrounds are expected to be close)
    lines = [f"composites: {len(made)} from {len(sources)} train_inner leg warmers on {len({p[3] for p in prov})} backgrounds",
             "holdout / official val images were not read (sources and backgrounds are train_inner only)."]
    if a.eval_hashes and os.path.exists(os.path.expanduser(a.eval_hashes)):
        z = np.load(os.path.expanduser(a.eval_hashes), allow_pickle=False)
        hold = set(l.strip() for l in open(SPLITS_DIR / "holdout.txt") if l.strip())
        ev = np.array([h for p, h in zip(z["paths"], z["hash"])
                       if ("/val/" in p or "/test_public/" in p or os.path.basename(p) in hold)], dtype=np.uint64)
        ch = np.array([dhash(Image.open(os.path.join(out, "images", f))) for f in made], dtype=np.uint64)
        dist = np.array([int(np.min([bin(int(c) ^ int(e)).count("1") for e in ev])) for c in ch]) if len(ev) else []
        lines.append(f"dHash vs {len(ev)} holdout/val/test images: min distance {int(dist.min()) if len(dist) else '-'}, "
                     f"within 4 bits: {int((dist <= 4).sum()) if len(dist) else 0}")
    open(os.path.join(out, "overlap.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    sel = rng.sample(made, min(24, len(made)))
    cell, cols = 200, 6
    sheet = Image.new("RGB", (cols * cell, ((len(sel) + cols - 1) // cols) * cell), "white")
    for i, f in enumerate(sel):
        with Image.open(os.path.join(out, "images", f)) as t:
            t = t.convert("RGB")
            t.thumbnail((cell - 4, cell - 4))
        sheet.paste(t, ((i % cols) * cell + 2, (i // cols) * cell + 2))
    sheet.save(os.path.join(out, "review.jpg"), quality=85)


if __name__ == "__main__":
    main()
