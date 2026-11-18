"""Pseudo-labels for multi-person COCO scenes (scripts/build_negatives.py cococrowd), for train_hires.py --ext_list.

Why: the test set is half crowds, Fashionpedia almost only single people. A person crop shown at full scale is
classified well (crop diagnosis: recall 0.54 -> 0.77), the same people seen in the whole crowd image are not.
So the teacher looks at the crops, and the student learns to say the same from the whole image.

  score   teacher = a train_hires checkpoint, flip TTA at 800/1333, on the full image and on every COCO
          ground-truth person box (tallest --max_crops, at least --min_height px, + --margin); per-class max.
          Writes <out>_probs.npz (filenames, probs [N, 27], n_crops). GPU. Several shards can run in parallel.
  label   probs -> targets: 1 if p >= --pos_thr, 0 if p <= --neg_thr, else -1 (unknown, masked in the loss).
          Writes <out>.npz (filenames, targets, categories) for --ext_labels.
  boxes   person boxes from a detector (Faster R-CNN R50 v2, offline only) for images without COCO boxes, e.g.
          the Fashionpedia holdout, in the --boxes format.
  calib   precision of p >= t and negative predictive value of p <= t on the Fashionpedia holdout, to choose
          --pos_thr / --neg_thr without touching any test data. With --out the holdout was scored with the SAME
          crop teacher (score step on the holdout list); otherwise the full-image TTA predictions (--preds) are used.
          Crops raise false positives (val precision 0.85 -> 0.50 with crops), so calibrate with --out.

    python scripts/pseudo_label.py score --ckpt runs_hr/X/best.pt --list neg_data/crowd_train.txt --out neg_data/pl_X
    python scripts/pseudo_label.py boxes --list neg_data/holdout_paths.txt --boxes neg_data/holdout_boxes.json
    python scripts/pseudo_label.py score --ckpt runs_hr/X/best.pt --list neg_data/holdout_paths.txt \
        --boxes neg_data/holdout_boxes.json --out neg_data/plho_X
    python scripts/pseudo_label.py calib --preds runs_hr/X/tta/preds_holdout_s800_flip.npz --out neg_data/plho_X
    python scripts/pseudo_label.py label --out neg_data/pl_X --pos_thr 0.8 --neg_thr 0.05
"""
import argparse
import json
import os
import sys

import numpy as np


def score(a):
    import torch
    from PIL import Image, ImageOps
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from crop_infer import classify
    from train_hires import build_hires_model, gate_split, timm_mean_std
    dev = torch.device("cuda")
    st = torch.load(a.ckpt, map_location="cpu")
    cfg, cats = st["config"], list(st["categories"])
    n_cls = len(cats)
    model = build_hires_model(cfg, n_cls + (1 if cfg.get("gate") else 0), False).to(dev)
    model.load_state_dict(st["model"])
    model = model.to(memory_format=torch.channels_last).eval()
    mean, std = (np.asarray(v, np.float32).reshape(3, 1, 1) for v in timm_mean_std(cfg["model"]))
    names = [l.strip() for l in open(a.list) if l.strip()]
    names = names[a.shard::a.n_shards]
    boxes = json.load(open(a.boxes))
    probs, ncrop = [], []
    for k, n in enumerate(names):
        with Image.open(n) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        W, H = im.size
        bs = sorted((b for b in boxes[n] if b[3] - b[1] >= a.min_height), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        crops = []
        for x0, y0, x1, y1 in bs[:a.max_crops]:
            mx, my = (x1 - x0) * a.margin, (y1 - y0) * a.margin
            crops.append(im.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my)))))
        p = 1 / (1 + np.exp(-gate_split(classify(model, [im] + crops, mean, std, dev), n_cls)[0]))
        probs.append(p.max(0))
        ncrop.append(len(crops))
        if k % 500 == 0:
            print(f"shard {a.shard}: {k}/{len(names)}", flush=True)
    np.savez(f"{a.out}_probs_{a.shard}.npz", filenames=np.array(names), probs=np.array(probs, np.float32),
             n_crops=np.array(ncrop), categories=np.array(cats))
    print(f"shard {a.shard}: wrote {len(names)}")


def merged(out):
    parts = sorted(f for f in os.listdir(os.path.dirname(out) or ".") if f.startswith(os.path.basename(out) + "_probs_"))
    zs = [np.load(os.path.join(os.path.dirname(out), f)) for f in parts]
    return (np.concatenate([z["filenames"] for z in zs]), np.concatenate([z["probs"] for z in zs]),
            np.concatenate([z["n_crops"] for z in zs]), [str(c) for c in zs[0]["categories"]])


def label(a):
    names, p, nc, cats = merged(a.out)
    t = np.full(p.shape, -1, np.int8)
    t[p >= a.pos_thr] = 1
    t[p <= a.neg_thr] = 0
    np.savez(f"{a.out}.npz", filenames=names, targets=t, categories=np.array(cats))
    print(f"{len(names)} images, {nc.mean():.2f} crops each: positives {(t == 1).sum(1).mean():.2f} / image, "
          f"negatives {(t == 0).mean():.3f}, unknown {(t == -1).mean():.3f} of all entries")
    for j in np.argsort(-(t == 1).sum(0)):
        print(f"  {cats[j][:28]:28s} pos {(t[:, j] == 1).sum():6d}  unknown {(t[:, j] == -1).sum():6d}")


def boxes(a):
    import torch
    import torchvision
    from PIL import Image, ImageOps
    from torchvision.transforms.functional import to_tensor
    w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    pid = w.meta["categories"].index("person")
    det = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.3).cuda().eval()
    out = {}
    for n in (l.strip() for l in open(a.list) if l.strip()):
        with Image.open(n) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        with torch.no_grad():
            o = det([to_tensor(im).cuda()])[0]
        keep = (o["labels"] == pid) & (o["scores"] >= a.person_thr)
        out[n] = o["boxes"][keep].cpu().numpy().round(1).tolist()
    json.dump(out, open(a.boxes, "w"))
    print(f"{len(out)} images, {np.mean([len(v) for v in out.values()]):.2f} people each -> {a.boxes}")


def calib(a):
    z = np.load(a.preds)
    y, p = z["targets"].astype(bool), 1 / (1 + np.exp(-z["logits"][:, :z["targets"].shape[1]]))
    if a.out:                                                  # same crop teacher, matched by file name
        names, pc, _, _ = merged(a.out)
        row = {os.path.basename(str(n)): k for k, n in enumerate(names)}
        keep = [k for k, f in enumerate(z["filenames"]) if os.path.basename(str(f)) in row]
        y, p = y[keep], pc[[row[os.path.basename(str(z["filenames"][k]))] for k in keep]]
        print(f"crop teacher on {len(keep)} holdout images")
    print("p >= t  precision  (share of true positives kept)")
    for t in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        m = p >= t
        print(f"  {t:.2f}   {(y & m).sum() / max(1, m.sum()):.3f}      {(y & m).sum() / y.sum():.3f}")
    print("p <= t  NPV        (share of true negatives kept)")
    for t in (0.02, 0.05, 0.1, 0.2):
        m = p <= t
        print(f"  {t:.2f}   {(~y & m).sum() / max(1, m.sum()):.4f}     {(~y & m).sum() / (~y).sum():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["score", "label", "calib", "boxes"])
    ap.add_argument("--ckpt")
    ap.add_argument("--list")
    ap.add_argument("--boxes", default="/data/eunjin/DL/neg_data/crowd_boxes.json")
    ap.add_argument("--out")
    ap.add_argument("--preds")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--max_crops", type=int, default=6, help="tallest people first (crop diagnosis: 3 keep most of it)")
    ap.add_argument("--min_height", type=float, default=64, help="smaller people get no crop (and no label of their own)")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--person_thr", type=float, default=0.5, help="boxes: detector score for a person")
    ap.add_argument("--pos_thr", type=float, default=0.8)
    ap.add_argument("--neg_thr", type=float, default=0.05)
    a = ap.parse_args()
    {"score": score, "label": label, "calib": calib, "boxes": boxes}[a.step](a)


if __name__ == "__main__":
    main()
