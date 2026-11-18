"""Diagnostic: does the classifier recognise garments once each person is seen at a large scale?

For every image of examples_labeled (30, labels) and test_public (300): detect people (torchvision Faster R-CNN
R50-FPN v2, COCO), crop each person (box + margin), run the train_hires checkpoint on the full image and on every
crop (flip TTA, test scale 800/1333), and combine with a per-class max.
  full          the usual prediction
  crops         max over person crops only
  full+crops    max over the full image and the crops
If full+crops recalls much more than full, the backbone can recognise the garments and the misses come from
people being small in crowds (scale); if not, the features themselves fail on these images.

    python scripts/crop_infer.py --ckpt runs_hr/R5_cnxt_hr_s42/best.pt --out runs_hr/R5_cnxt_hr_s42/ood/crop
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_hires import build_hires_model, ceil32, gate_split, target_size, timm_mean_std  # noqa: E402

SETS = {"examples": ("examples_labeled", "labels.csv"), "test": ("test_public", "submission_template.csv"), "val": (None, "labels.csv")}
DETECTORS = {   # name -> (builder, weights); parameter counts are printed at start
    "frcnn_r50_v2": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights"),
    "frcnn_mbv3": ("fasterrcnn_mobilenet_v3_large_fpn", "FasterRCNN_MobileNet_V3_Large_FPN_Weights"),
    "ssdlite": ("ssdlite320_mobilenet_v3_large", "SSDLite320_MobileNet_V3_Large_Weights"),
}


def classify(model, imgs, mean, std, dev, short=800, max_long=1333):
    """list of PIL images -> class probabilities [N, C] (flip TTA, masked pooling over each real area)."""
    xs, szs = [], []
    for im in imgs:
        im = im.resize(target_size(*im.size, short, max_long), Image.BILINEAR)
        x = (np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
        xs.append(torch.from_numpy(np.ascontiguousarray(x)))
        szs.append(x.shape[1:])
    out = []
    for k in range(0, len(xs), 8):
        chunk, sz = xs[k:k + 8], szs[k:k + 8]
        H, W = ceil32(max(s[0] for s in sz)), ceil32(max(s[1] for s in sz))
        b = torch.zeros(len(chunk), 3, H, W)
        for j, x in enumerate(chunk):
            b[j, :, :x.shape[1], :x.shape[2]] = x
        b = b.to(dev).contiguous(memory_format=torch.channels_last)
        s = torch.tensor(sz, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lg = model(b, s).float()
            # flipped batch: the real area moves to the right edge, so flip each image inside its own area instead
            fb = torch.zeros_like(b)
            for j, (h, w) in enumerate(sz):
                fb[j, :, :h, :w] = torch.flip(b[j, :, :h, :w], dims=[2])
            lg = (lg + model(fb.contiguous(memory_format=torch.channels_last), s).float()) / 2
        out.append(lg.cpu().numpy())
    return np.concatenate(out)


def f1(y, P):
    tp, fp, fn = (y & P).sum(0), (~y & P).sum(0), (y & ~P).sum(0)
    micro = 2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())
    d = 2 * tp + fp + fn
    macro = np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0))
    ti, di = (y & P).sum(1), 2 * (y & P).sum(1) + (~y & P).sum(1) + (y & ~P).sum(1)
    image = np.where(di > 0, 2 * ti / np.maximum(di, 1), 1.0).mean()
    return {"micro": round(float(micro), 4), "macro": round(float(macro), 4), "image": round(float(image), 4),
            "recall": round(float(tp.sum() / max(1, tp.sum() + fn.sum())), 4),
            "precision": round(float(tp.sum() / max(1, tp.sum() + fp.sum())), 4)}


def thresholds(cats, base, special):
    t = np.full(len(cats), base)
    t[[i for i, c in enumerate(cats) if c.startswith(tuple(special))]] = 0.5
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="/data/eunjin/DL/test_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--special", nargs="*", default=["sweater", "leg warmer", "umbrella"], help="classes kept at 0.5")
    ap.add_argument("--person_thr", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--max_crops", type=int, default=12, help="largest people first")
    ap.add_argument("--min_height", type=int, default=48, help="skip people shorter than this (pixels)")
    ap.add_argument("--detector", choices=list(DETECTORS), default="frcnn_r50_v2")
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=["examples", "test"])
    ap.add_argument("--val_dir", default="/data/eunjin/DL/fassion_tagging/data/val")
    a = ap.parse_args()
    dev = torch.device("cuda")
    st = torch.load(a.ckpt, map_location="cpu")
    cfg, cats = st["config"], list(st["categories"])
    n_cls = len(cats)
    model = build_hires_model(cfg, n_cls + (1 if cfg.get("gate") else 0), False).to(dev)
    model.load_state_dict(st["model"])
    model = model.to(memory_format=torch.channels_last).eval()
    mean, std = (np.asarray(v, np.float32).reshape(3, 1, 1) for v in timm_mean_std(cfg["model"]))
    builder, wname = DETECTORS[a.detector]
    w = getattr(torchvision.models.detection, wname).DEFAULT
    pid = w.meta["categories"].index("person")
    kw = {"score_thresh": 0.3} if a.detector == "ssdlite" else {"box_score_thresh": 0.3}
    det = getattr(torchvision.models.detection, builder)(weights=w, **kw).to(dev).eval()
    n_det, n_cls_p = sum(p.numel() for p in det.parameters()), sum(p.numel() for p in model.parameters())
    print(f"detector {a.detector} {n_det / 1e6:.2f}M + classifier {n_cls_p / 1e6:.2f}M = {(n_det + n_cls_p) / 1e6:.2f}M", flush=True)
    thr = thresholds(cats, a.threshold, a.special)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    report = {}
    for name in a.sets:
        folder, csv_name = SETS[name]
        root = a.val_dir if folder is None else os.path.join(a.data_root, folder)
        rows = list(csv.DictReader(open(os.path.join(root, csv_name), encoding="utf-8-sig")))
        y = np.array([[int(r.get(c) or 0) for c in cats] for r in rows], bool)
        full, crops, ncrop = [], [], []
        for r in rows:
            with Image.open(os.path.join(root, "images", r["file_name"])) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
            with torch.no_grad():
                o = det([to_tensor(im).to(dev)])[0]
            keep = (o["labels"] == pid) & (o["scores"] >= a.person_thr)
            boxes = o["boxes"][keep].cpu().numpy()
            boxes = boxes[(boxes[:, 3] - boxes[:, 1]) >= a.min_height] if len(boxes) else boxes
            if len(boxes):
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = boxes[np.argsort(-area)[:a.max_crops]]
            W, H = im.size
            cims = []
            for x0, y0, x1, y1 in boxes:
                mx, my = (x1 - x0) * a.margin, (y1 - y0) * a.margin
                cims.append(im.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my)))))
            lg = classify(model, [im] + cims, mean, std, dev)
            p = 1 / (1 + np.exp(-gate_split(lg, n_cls)[0]))
            full.append(p[0])
            crops.append(p[1:].max(0) if len(cims) else np.zeros(n_cls))
            ncrop.append(len(cims))
        full, crops = np.array(full), np.array(crops)
        variants = {"full": full >= thr, "crops": crops >= thr, "full+crops": np.maximum(full, crops) >= thr}
        res = {"n": len(rows), "crops_per_image": float(np.mean(ncrop))}
        for k, P in variants.items():
            res[k] = {"pred_per_image": round(float(P.sum(1).mean()), 2)}
            if name != "test":
                res[k].update(f1(y, P), count_mae=round(float(np.abs(P.sum(1) - y.sum(1)).mean()), 2))
        report[name] = res
        np.savez(f"{a.out}_{name}.npz", filenames=np.array([r["file_name"] for r in rows]), targets=y, full=full,
                 crops=crops, n_crops=np.array(ncrop), categories=np.array(cats))
        print(name, json.dumps(res), flush=True)
    json.dump(report, open(f"{a.out}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
