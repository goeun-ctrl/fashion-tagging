"""Detector-free tile inference: does looking at overlapping tiles recover what person crops recovered?

Each image is classified whole and as overlapping tiles (grids --grids, each tile = 1/g of the image side
times --overlap), every view at test scale 800/1333 with flip TTA; per-class max over the chosen views.
No extra parameters. Evaluated on examples_labeled (30), test_public (300, prediction counts only) and the
official val (1,158: does tiling add false positives on single-person Fashionpedia photos?).

    python scripts/tile_infer.py --ckpt runs_hr/R5_cnxt_hr_s42/best.pt --out runs_hr/R5_cnxt_hr_s42/ood/tile
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_infer import classify, f1, thresholds  # noqa: E402
from train_hires import build_hires_model, gate_split, timm_mean_std  # noqa: E402


def tiles(im, g, overlap):
    W, H = im.size
    tw, th = min(W, int(W / g * overlap)), min(H, int(H / g * overlap))
    xs = np.linspace(0, W - tw, g).astype(int) if g > 1 else [0]
    ys = np.linspace(0, H - th, g).astype(int) if g > 1 else [0]
    return [im.crop((x, y, x + tw, y + th)) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="/data/eunjin/DL/test_data")
    ap.add_argument("--val_dir", default="/data/eunjin/DL/fassion_tagging/data/val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--special", nargs="*", default=["sweater", "leg warmer", "umbrella"])
    ap.add_argument("--grids", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--overlap", type=float, default=1.5, help="tile side = image side / g * overlap")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tile_gate", type=float, default=None,
                    help="gate checkpoints only: drop tiles whose gate probability is below this (no dressed person in it)")
    ap.add_argument("--gate_threshold", type=float, default=0.5, help="gate checkpoints: full-image gate below this -> empty")
    a = ap.parse_args()
    dev = torch.device("cuda")
    st = torch.load(a.ckpt, map_location="cpu")
    cfg, cats = st["config"], list(st["categories"])
    n_cls = len(cats)
    has_gate = bool(cfg.get("gate"))
    assert a.tile_gate is None or has_gate, "--tile_gate needs a checkpoint trained with --gate"
    model = build_hires_model(cfg, n_cls + (1 if cfg.get("gate") else 0), False).to(dev)
    model.load_state_dict(st["model"])
    model = model.to(memory_format=torch.channels_last).eval()
    mean, std = (np.asarray(v, np.float32).reshape(3, 1, 1) for v in timm_mean_std(cfg["model"]))
    thr = thresholds(cats, a.threshold, a.special)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sets = [("examples", os.path.join(a.data_root, "examples_labeled"), "labels.csv"),
            ("test", os.path.join(a.data_root, "test_public"), "submission_template.csv"),
            ("val", a.val_dir, "labels.csv")]
    report = {}
    for name, root, csv_name in sets:
        rows = list(csv.DictReader(open(os.path.join(root, csv_name), encoding="utf-8-sig")))
        if a.limit:
            rows = rows[:a.limit]
        y = np.array([[int(r.get(c) or 0) for c in cats] for r in rows], bool)
        views, n_kept = {"full": []}, []
        views.update({f"g{g}": [] for g in a.grids})
        for r in rows:
            with Image.open(os.path.join(root, "images", r["file_name"])) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
            ims, owners = [im], ["full"]
            for g in a.grids:
                t = tiles(im, g, a.overlap)
                ims += t
                owners += [f"g{g}"] * len(t)
            lg, g = gate_split(classify(model, ims, mean, std, dev), n_cls)
            p = 1 / (1 + np.exp(-lg))
            owners = np.array(owners)
            keep = np.ones(len(ims), bool)
            if has_gate:
                gp = 1 / (1 + np.exp(-np.asarray(g).reshape(-1)))
                if gp[0] < a.gate_threshold:      # whole image judged empty: every combination predicts nothing
                    p = np.zeros_like(p)
                if a.tile_gate is not None:
                    keep = (gp >= a.tile_gate) | (owners == "full")
                n_kept.append(int(keep[owners != "full"].sum()))
            for k in views:
                m = (owners == k) & keep
                views[k].append(p[m].max(0) if m.any() else np.zeros(n_cls))
        V = {k: np.array(v) for k, v in views.items()}
        combos = {"full": V["full"]}
        for g in a.grids:
            combos[f"full+{g}x{g}"] = np.maximum(V["full"], V[f"g{g}"])
        if len(a.grids) > 1:
            combos["full+" + "+".join(f"{g}x{g}" for g in a.grids)] = np.maximum.reduce([V[k] for k in V])
        res = {"n": len(rows)}
        if n_kept:
            res["tiles_kept_per_image"] = round(float(np.mean(n_kept)), 2)
        for k, p in combos.items():
            P = p >= thr
            res[k] = {"pred_per_image": round(float(P.sum(1).mean()), 2)}
            if name != "test":
                res[k].update(f1(y, P))
                res[k]["count_mae"] = round(float(np.abs(P.sum(1) - y.sum(1)).mean()), 2)
        report[name] = res
        np.savez(f"{a.out}_{name}.npz", filenames=np.array([r["file_name"] for r in rows]), targets=y,
                 categories=np.array(cats), **{k.replace("+", "_"): v for k, v in combos.items()})
        print(name, json.dumps(res), flush=True)
    json.dump(report, open(f"{a.out}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
