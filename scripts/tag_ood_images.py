"""Image-type tags for the provided OOD sets (examples_labeled, test_public), for the error analysis sheet.

Tags come from simple measurements, not from labels:
  person count / largest person share   torchvision Faster R-CNN R50-FPN v2 (COCO), person score >= 0.5
  gray / low_sat                        mean HSV saturation < 0.05 / < 0.10 (old or B/W photos)
  landscape                             width > height (after EXIF)
Derived type (first match): no_person (0 persons), crowd (>= 5), small_person (largest person < 5% of the
image), bw (gray), group (2-4), single.

    python scripts/tag_ood_images.py --data_root /data/eunjin/DL/test_data   # -> <data_root>/image_tags.csv
"""
import argparse
import csv
import os

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="/data/eunjin/DL/test_data")
    ap.add_argument("--person_thr", type=float, default=0.5)
    a = ap.parse_args()
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    pid = weights.meta["categories"].index("person")
    det = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.3).cuda().eval()
    rows = []
    for folder, csv_name in (("examples_labeled", "labels.csv"), ("test_public", "submission_template.csv")):
        names = [r["file_name"] for r in csv.DictReader(open(os.path.join(a.data_root, folder, csv_name), encoding="utf-8-sig"))]
        for n in names:
            with Image.open(os.path.join(a.data_root, folder, "images", n)) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            small = im.copy(); small.thumbnail((128, 128))
            sat = float(np.asarray(small.convert("HSV"), np.float32)[..., 1].mean() / 255)
            with torch.no_grad():
                o = det([to_tensor(im).cuda()])[0]
            keep = (o["labels"] == pid) & (o["scores"] >= a.person_thr)
            boxes = o["boxes"][keep].cpu()
            areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (w * h)).numpy() if len(boxes) else np.zeros(0)
            n_p, big = int(keep.sum()), float(areas.max()) if len(areas) else 0.0
            gray = sat < 0.05
            kind = ("no_person" if n_p == 0 else "crowd" if n_p >= 5 else "small_person" if big < 0.05
                    else "bw" if gray else "group" if n_p >= 2 else "single")
            rows.append({"set": folder, "file_name": n, "type": kind, "n_person": n_p, "max_person_share": round(big, 4),
                         "saturation": round(sat, 3), "gray": int(gray), "low_sat": int(sat < 0.10),
                         "landscape": int(w > h), "width": w, "height": h})
    out = os.path.join(a.data_root, "image_tags.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    for s in ("examples_labeled", "test_public"):
        sub = [r for r in rows if r["set"] == s]
        kinds = {}
        for r in sub:
            kinds[r["type"]] = kinds.get(r["type"], 0) + 1
        print(s, len(sub), kinds, "gray", sum(r["gray"] for r in sub), "landscape", sum(r["landscape"] for r in sub))
    print("wrote", out)


if __name__ == "__main__":
    main()
