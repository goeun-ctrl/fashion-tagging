"""Data preparation for redesign B (segmentation-supervised tagging). Plan: docs/plan/재설계B_구현계획.md.

    python scripts/segb_prep.py fp      --data D --out D/segb_prep      # D1 Fashionpedia train_inner garment + part maps
    python scripts/segb_prep.py detect  --data D --out D/segb_prep      # D3 person detector on COCO person photos (GPU)
    python scripts/segb_prep.py coco    --data D --out D/segb_prep      # D2/D3 COCO layers (needs ff_inst + detect)
    python scripts/segb_prep.py lvis    --data D --out D/segb_prep      # D4 LVIS positive masks for external images
    python scripts/segb_prep.py lists   --data D --out D/segb_prep      # D5 training lists + leak check

Layout under --data (gpu-server-b: /data1/workspaces/goeun_dl/eunjin/data):
  train_inner/images, train_inner/fp_train_inner_ann.json, holdout/images, val/, meta/(splits, train labels),
  neg_data/(coco, coco_person, coco_crowd, places, annotations, neg_*.txt, ff_inst), lvis_ann/, external/external_merged,
  crowd_eval/(labels.csv, exclude_from_training.txt)

Maps are stored in the EXIF-transposed image frame (the frame our loader sees), long side <= 1024.
  fp/<name>.npz    g uint8 [4, H, W] per class group (0 = none, j = j-th class of the group), parts uint8 [H, W] bit k = PARTS[k]
  coco/<stem>.npz  P bool (person incl. crowd), O uint8 (0 / 17 tie / 25 bag / 27 umbrella = class id + 1),
                   E bool (excluded: FashionFormer garment outside persons, detector person missing from COCO),
                   kd uint8 [27, h4, w4] FashionFormer class probability x score (stride 4 of the original),
                   ff_found bool [27] (score >= --ff_thr), coco_has bool [27]
  lvis/<file>.npz  g uint8 [4, H, W], 255 = unknown, class index where an LVIS positive mask of a mapped class lies
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fashion_tagging.segb.spec import GROUPS, PARTS, CLASS_TO_GROUP  # noqa: E402

TRANSPOSE = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM, 5: Image.TRANSPOSE,
             6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90}
LONG = 1024
COCO_OBJ = {"handbag": 24, "backpack": 24, "suitcase": 24, "umbrella": 26, "tie": 16}


def dec(seg, h, w):
    from pycocotools import mask as mu
    if isinstance(seg, list):
        r = mu.merge(mu.frPyObjects(seg, h, w))
    elif isinstance(seg.get("counts"), list):
        r = mu.frPyObjects(seg, h, w)
    else:
        r = seg
    return mu.decode(r).astype(bool)


def orient_of(path):
    with Image.open(path) as im:
        return im.getexif().get(274, 1), im.size


def to_frame(arr, orient, nearest=True):
    """uint8 / bool [H, W] in raw file coordinates -> EXIF-transposed frame, long side <= LONG."""
    img = Image.fromarray(arr.astype(np.uint8))
    if orient in TRANSPOSE:
        img = img.transpose(TRANSPOSE[orient])
    w, h = img.size
    s = min(1.0, LONG / max(w, h))
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.NEAREST if nearest else Image.BILINEAR)
    return np.asarray(img)


# ------------------------------------------------------------------ D1 Fashionpedia
def fp_one(job):
    path, out, w, h, anns = job
    if os.path.exists(out):
        return 0
    orient, size = orient_of(path)
    assert size == (w, h), (path, size, (w, h))
    g = np.zeros((4, h, w), np.uint8)
    parts = np.zeros((h, w), np.uint8)
    for cat, seg in anns:
        m = dec(seg, h, w)
        if cat in CLASS_TO_GROUP:
            gi, j = CLASS_TO_GROUP[cat]
            g[gi][m] = j
        elif cat in PARTS:
            parts[m] |= np.uint8(1 << PARTS.index(cat))
    gf = np.stack([to_frame(x, orient) for x in g])
    np.savez_compressed(out, g=gf, parts=to_frame(parts, orient))
    return 1


def cmd_fp(a):
    d = json.load(open(os.path.join(a.data, "train_inner", "fp_train_inner_ann.json")))
    A = defaultdict(list)
    for x in d["annotations"]:
        if x["category_id"] in CLASS_TO_GROUP or x["category_id"] in PARTS:
            A[x["image_id"]].append((x["category_id"], x["segmentation"]))
    os.makedirs(os.path.join(a.out, "fp"), exist_ok=True)
    jobs = [(os.path.join(a.data, "train_inner", "images", im["file_name"]), os.path.join(a.out, "fp", im["file_name"] + ".npz"),
             im["width"], im["height"], A.get(im["id"], [])) for im in d["images"]]
    with Pool(a.workers) as p:
        n = sum(p.imap_unordered(fp_one, jobs, chunksize=16))
    print(f"fp: wrote {n} new maps ({len(jobs)} train_inner images)")


# ------------------------------------------------------------------ D3 person detector
def coco_paths(a):
    nd = os.path.join(a.data, "neg_data")
    return sorted(os.path.join(nd, s, f) for s in ("coco_person", "coco_crowd") for f in os.listdir(os.path.join(nd, s))
                  if f.endswith(".jpg"))


def cmd_detect(a):
    import torch
    import torchvision
    from torchvision.transforms.functional import to_tensor
    os.environ.setdefault("TORCH_HOME", os.path.join(a.data, "torch_cache"))
    paths = coco_paths(a)[a.shard::a.nshard]
    out = os.path.join(a.out, f"det_person_{a.shard}.json")
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=0.3).cuda().eval()
    res = {}
    with torch.no_grad():
        for k, p in enumerate(paths):
            x = to_tensor(Image.open(p).convert("RGB")).cuda()     # COCO files carry no EXIF rotation
            o = m([x])[0]
            keep = o["labels"] == 1
            res[os.path.basename(p)] = [[*map(float, b), float(s)] for b, s in zip(o["boxes"][keep].cpu(), o["scores"][keep].cpu())]
            if k % 1000 == 0:
                print(a.shard, k, len(paths), flush=True)
    json.dump(res, open(out, "w"))
    print("detect done", a.shard, len(res))


# ------------------------------------------------------------------ D2/D3 COCO layers
def iou(b, B):
    x0, y0 = np.maximum(b[0], B[:, 0]), np.maximum(b[1], B[:, 1])
    x1, y1 = np.minimum(b[2], B[:, 2]), np.minimum(b[3], B[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def coco_one(job):
    path, out, w, h, anns, dets, ff_path, ff_thr, det_thr = job
    if os.path.exists(out):
        return 0
    P = np.zeros((h, w), bool)
    O = np.zeros((h, w), np.uint8)
    pboxes, coco_has = [], np.zeros(27, bool)
    for name, seg, bbox in anns:
        m = dec(seg, h, w)
        if name == "person":
            P |= m
            pboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        else:
            c = COCO_OBJ[name]
            O[m & (O == 0)] = c + 1
            coco_has[c] = True
    E = np.zeros((h, w), bool)
    pb = np.array(pboxes, np.float64).reshape(-1, 4)
    for x0, y0, x1, y1, s in dets:                            # detector person absent from COCO -> exclude its box
        if s >= det_thr and (len(pb) == 0 or iou(np.array([x0, y0, x1, y1]), pb).max() < 0.5):
            E[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
    z = np.load(ff_path)
    lab, sc, pr = z["labels"].astype(int), z["scores"].astype(np.float32), z["probs"]
    h4, w4 = (h + 3) // 4, (w + 3) // 4
    kd = np.zeros((27, h4, w4), np.float32)
    ff_found = np.zeros(27, bool)
    for l, s, p in zip(lab, sc, pr):
        if l >= 27:
            continue
        p = p[:h4, :w4].astype(np.float32) / 255.0
        kd[l] = np.maximum(kd[l], p * s)                       # class probability x instance score
        if s >= ff_thr:
            ff_found[l] = True
            big = np.repeat(np.repeat(p >= 0.5, 4, 0), 4, 1)[:h, :w]
            E |= big & ~P                                      # FashionFormer garment outside COCO persons
    np.savez_compressed(out, P=P, O=O, E=E & ~P, kd=np.round(kd * 255).astype(np.uint8), ff_found=ff_found, coco_has=coco_has)
    return 1


def cmd_coco(a):
    d = json.load(open(os.path.join(a.data, "neg_data", "annotations", "instances_train2017.json")))
    cname = {c["id"]: c["name"] for c in d["categories"]}
    ims = {im["file_name"]: im for im in d["images"]}
    A = defaultdict(list)
    for x in d["annotations"]:
        n = cname[x["category_id"]]
        if n == "person" or n in COCO_OBJ:
            A[x["image_id"]].append((n, x["segmentation"], x["bbox"]))
    dets = {}
    for f in sorted(os.listdir(a.out)):
        if f.startswith("det_person_") and f.endswith(".json"):
            dets.update(json.load(open(os.path.join(a.out, f))))
    paths = coco_paths(a)
    miss = [p for p in paths if os.path.basename(p) not in dets]
    assert not miss, f"{len(miss)} images have no detector output (run detect), e.g. {miss[:2]}"
    ffd = os.path.join(a.data, "neg_data", "ff_inst")
    os.makedirs(os.path.join(a.out, "coco"), exist_ok=True)
    jobs = []
    for p in paths:
        f = os.path.basename(p); im = ims[f]; stem = f[:-4]
        ff = os.path.join(ffd, stem + ".npz")
        assert os.path.exists(ff), f"no FashionFormer output for {f}"
        jobs.append((p, os.path.join(a.out, "coco", stem + ".npz"), im["width"], im["height"], A.get(im["id"], []),
                     dets[f], ff, a.ff_thr, a.det_thr))
    with Pool(a.workers) as pool:
        n = sum(pool.imap_unordered(coco_one, jobs, chunksize=16))
    print(f"coco: wrote {n} new layer files ({len(jobs)} images)")


# ------------------------------------------------------------------ D4 LVIS
def lvis_map():
    sys.path.insert(0, os.path.join(ROOT, "scripts", "external"))
    src = open(os.path.join(ROOT, "scripts", "external", "build_external_labels.py"), encoding="utf-8").read()
    start, end = src.index("MAP = {"), src.index("assert set(MAP)")
    ns = {}
    exec(src[start:end], ns)                                   # the team's LVIS -> 27 class table (single source of truth)
    return ns["MAP"]


def lvis_one(job):
    out, w, h, items = job
    if os.path.exists(out):
        return 0
    g = np.full((4, h, w), 255, np.uint8)
    for cls, seg in items:
        gi, j = CLASS_TO_GROUP[cls]
        g[gi][dec(seg, h, w)] = j
    np.savez_compressed(out, g=np.stack([to_frame(x, 1) for x in g]))
    return 1


def cmd_lvis(a):
    cats = [l.strip() for l in open(os.path.join(a.data, "meta", "train", "categories.txt"), encoding="utf-8-sig") if l.strip()]
    MAP = lvis_map()
    ext = os.path.join(a.data, "external", "external_merged")
    src = {r["file_name"]: r for r in csv.DictReader(open(os.path.join(ext, "sources.csv")))}
    lab = {r["file_name"]: r for r in csv.DictReader(open(os.path.join(ext, "labels.csv")))}
    excl = {l.strip() for l in open(os.path.join(a.data, "crowd_eval", "exclude_from_training.txt")) if l.strip() and not l.startswith("#")}
    want = {f: r for f, r in src.items() if r["source"] == "lvis" and f not in excl}
    os.makedirs(os.path.join(a.out, "lvis"), exist_ok=True)
    rows = [["our class", "LVIS categories"]] + [[c, " | ".join(MAP[c][0])] for c in cats]
    with open(os.path.join(a.out, "lvis_mapping.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    jobs = []
    for split in ("train", "val"):
        d = json.load(open(os.path.join(a.data, "lvis_ann", f"lvis_v1_{split}.json")))
        lname = {c["id"]: c["name"] for c in d["categories"]}
        ims = {im["id"]: im for im in d["images"]}
        anns = defaultdict(list)
        for x in d["annotations"]:
            anns[x["image_id"]].append((lname[x["category_id"]], x["segmentation"]))
        for f, r in want.items():
            if r["split"] != split:
                continue
            iid = int(r["image_id"]); im = ims[iid]
            items = []
            for ci, c in enumerate(cats):
                if float(lab[f][c]) == 1 and MAP[c][0]:
                    items += [(ci, seg) for n, seg in anns[iid] if n in MAP[c][0]]
            jobs.append((os.path.join(a.out, "lvis", f + ".npz"), im["width"], im["height"], items))
    with Pool(a.workers) as p:
        n = sum(p.imap_unordered(lvis_one, jobs, chunksize=16))
    print(f"lvis: wrote {n} new maps ({len(jobs)} external LVIS images, {len(excl)} exclusion entries honoured)")


# ------------------------------------------------------------------ D5 lists + leak check
def cmd_lists(a):
    D, nd = a.data, os.path.join(a.data, "neg_data")
    remap = lambda p: p.replace("/data/eunjin/DL/neg_data", nd)
    excl = [l.strip() for l in open(os.path.join(D, "crowd_eval", "exclude_from_training.txt")) if l.strip() and not l.startswith("#")]
    bad_ids = {int(x.split(":")[1]) for x in excl if x.startswith("coco_id:")}
    bad_files = {x for x in excl if not x.startswith("coco_id:")}
    inner = [l.strip() for l in open(os.path.join(D, "meta", "splits", "train_inner.txt")) if l.strip()]
    hold = {l.strip() for l in open(os.path.join(D, "meta", "splits", "holdout.txt")) if l.strip()}
    assert not set(inner) & hold
    coco = coco_paths(a)
    empty = [remap(l.strip()) for l in open(os.path.join(nd, "neg_train.txt")) if l.strip()]
    ext = os.path.join(D, "external", "external_merged")
    exts = [r["file_name"] for r in csv.DictReader(open(os.path.join(ext, "sources.csv"))) if r["file_name"] not in bad_files]
    src = {r["file_name"]: r for r in csv.DictReader(open(os.path.join(ext, "sources.csv")))}
    problems = []
    for p in coco + empty:
        b = os.path.basename(p)
        if b[:-4].isdigit() and int(b[:-4]) in bad_ids:
            problems.append(p)
    for f in exts:
        r = src[f]
        if r["source"] == "lvis" and int(r["image_id"]) in bad_ids:
            problems.append(f)
    assert not problems, f"crowd-eval leak: {problems[:5]} ({len(problems)})"
    out = {"fp": inner, "coco": coco, "empty": empty, "ext": exts}
    for k, v in out.items():
        with open(os.path.join(a.out, f"list_{k}.txt"), "w") as f:
            f.write("\n".join(v) + "\n")
    print({k: len(v) for k, v in out.items()}, "leak check passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fp", "detect", "coco", "lvis", "lists"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ff_thr", type=float, default=0.3, help="Q1: FashionFormer score for 'garment found'")
    ap.add_argument("--det_thr", type=float, default=0.5, help="Q3: detector score for a missing person")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"fp": cmd_fp, "detect": cmd_detect, "coco": cmd_coco, "lvis": cmd_lvis, "lists": cmd_lists}[a.cmd](a)


if __name__ == "__main__":
    main()
