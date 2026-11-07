"""Low-resolution per-class garment masks from the original Fashionpedia annotations, for the mask-supervised
CSRA head (scripts/train_hires.py --mask_dir). Only train_inner images: holdout masks are never written, so
holdout stays clean for epoch / threshold selection.

For every image: the 27 classes (Fashionpedia category ids 0-26, same order as our labels) are rasterised
(polygons and RLE, pycocotools), the EXIF orientation our loader applies is applied too, and the mask is
reduced to a grid whose long side is --grid (a cell is on when >= --cover of it is garment).
Output: <out>/<file_name>.npz with cls (uint8 [k]) and bits (packbits of bool [k, gh, gw]), shape (gh, gw).

Needs pycocotools (the FashionFormer venv has it; Python 3.8 compatible):
    /data/eunjin/DL/fashionformer/venv/bin/python scripts/build_masks.py
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

TRANSPOSE = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM, 5: Image.TRANSPOSE,
             6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90}       # = PIL ImageOps.exif_transpose
G = {}


def one(job):
    name, w, h, anns = job
    out = os.path.join(G["out"], name + ".npz")
    if os.path.exists(out):
        return 0
    with Image.open(os.path.join(G["img_dir"], name)) as im:
        orient = im.getexif().get(274, 1)
        assert im.size == (w, h), (name, im.size, (w, h))
    per = {}
    for cat, seg in anns:
        if isinstance(seg, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
        elif isinstance(seg.get("counts"), list):
            rle = mask_utils.frPyObjects(seg, h, w)
        else:
            rle = seg
        m = mask_utils.decode(rle).astype(bool)
        per[cat] = per[cat] | m if cat in per else m
    cls = sorted(per)
    if orient in TRANSPOSE:
        w, h = (h, w) if orient in (5, 6, 7, 8) else (w, h)
    s = G["grid"] / max(w, h)
    gw, gh = max(1, round(w * s)), max(1, round(h * s))
    grids = []
    for c in cls:
        img = Image.fromarray(per[c].astype(np.uint8) * 255)
        if orient in TRANSPOSE:
            img = img.transpose(TRANSPOSE[orient])
        grids.append(np.asarray(img.resize((gw, gh), Image.BOX)) >= 255 * G["cover"])
    arr = np.stack(grids) if grids else np.zeros((0, gh, gw), bool)
    np.savez_compressed(out, cls=np.array(cls, np.uint8), bits=np.packbits(arr.reshape(-1)), shape=np.array([gh, gw]))
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", default="/data/eunjin/DL/fashionpedia_ann/instances_attributes_train2020.json")
    ap.add_argument("--img_dir", default="/data/eunjin/DL/fassion_tagging/data/train/images")
    ap.add_argument("--names", default="/data/eunjin/DL/fassion_tagging/data/splits/train_inner.txt")
    ap.add_argument("--holdout", default="/data/eunjin/DL/fassion_tagging/data/splits/holdout.txt")
    ap.add_argument("--out", default="/data/eunjin/DL/fashionpedia_ann/masks27")
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--cover", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    names = [l.strip() for l in open(a.names) if l.strip()]
    hold = {l.strip() for l in open(a.holdout) if l.strip()}
    assert not hold & set(names), "train_inner and holdout overlap"
    d = json.load(open(a.ann))
    by_file = {im["file_name"]: im for im in d["images"]}
    anns = {}
    for x in d["annotations"]:
        if x["category_id"] < 27:
            anns.setdefault(x["image_id"], []).append((x["category_id"], x["segmentation"]))
    os.makedirs(a.out, exist_ok=True)
    G.update(out=a.out, img_dir=a.img_dir, grid=a.grid, cover=a.cover)
    jobs = [(n, by_file[n]["width"], by_file[n]["height"], anns.get(by_file[n]["id"], [])) for n in names]
    with Pool(a.workers, initializer=G.update, initargs=(dict(G),)) as p:
        done = sum(p.imap_unordered(one, jobs, chunksize=64))
    print(f"wrote {done} masks ({len(names)} train_inner images, holdout excluded) to {a.out}")


if __name__ == "__main__":
    main()
