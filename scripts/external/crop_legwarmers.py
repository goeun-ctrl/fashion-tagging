"""Detect leg warmers in collected web photos (OWLv2 open-vocabulary detector), crop knee-to-foot, and label.

    python scripts/external/crop_legwarmers.py --raw data/external/legwarmer_web --out data/external/legwarmer_web_crop \
        --categories data/train/categories.txt

For each image in <raw>/raw/ (with <raw>/meta.csv from collect_openverse_wikimedia.py): boxes for "leg warmers"
with score >= --min_score (and >= 0.5 x the best score in that image) are merged; the crop spans those boxes,
extended 35% of their height upward (knee) and 45% downward (feet), plus 20% sideways. Images with no box are
dropped. Labels: leg warmer = 1, every other class = -1 (not checked). Output: categories.txt, labels.csv,
images/, sources.csv (license, creator, source page, box, score), review.jpg.
"""
import argparse
import csv
import os
import shutil

import torch
from PIL import Image, ImageOps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", required=True)
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--min_score", type=float, default=0.2)
    ap.add_argument("--min_side", type=int, default=96)
    a = ap.parse_args()
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    cats = [l.rstrip("\n") for l in open(a.categories, encoding="utf-8-sig") if l.strip()]
    li = cats.index("leg warmer")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained(a.model)
    model = Owlv2ForObjectDetection.from_pretrained(a.model).to(dev).eval()
    meta = [m for m in csv.DictReader(open(os.path.join(a.raw, "meta.csv"), encoding="utf-8")) if m["downloaded"] == "1"]
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    rows, prov, dropped = [], [], {"no leg warmer found": 0, "unreadable": 0, "crop too small": 0}
    for m in meta:
        try:
            img = ImageOps.exif_transpose(Image.open(os.path.join(a.raw, "raw", m["file_name"]))).convert("RGB")
        except Exception:  # noqa: BLE001
            dropped["unreadable"] += 1
            continue
        W, H = img.size
        inp = proc(text=[["a photo of leg warmers"]], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inp)
        side = max(W, H)                                   # OWLv2 pads to a square of the long side
        res = proc.image_processor.post_process_object_detection(
            o, threshold=a.min_score, target_sizes=torch.tensor([[side, side]], device=dev))[0]
        if len(res["scores"]) == 0:
            dropped["no leg warmer found"] += 1
            continue
        best = res["scores"].max().item()
        boxes = res["boxes"][res["scores"] >= 0.5 * best].clamp(min=0).tolist()
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = min(max(b[2] for b in boxes), W), min(max(b[3] for b in boxes), H)
        bh, bw = y1 - y0, x1 - x0
        cx0, cx1 = max(0, x0 - 0.2 * bw), min(W, x1 + 0.2 * bw)
        cy0, cy1 = max(0, y0 - 0.35 * bh), min(H, y1 + 0.45 * bh)
        if min(cx1 - cx0, cy1 - cy0) < a.min_side:
            dropped["crop too small"] += 1
            continue
        fn = os.path.splitext(m["file_name"])[0] + "_crop.jpg"
        img.crop((int(cx0), int(cy0), int(cx1), int(cy1))).save(os.path.join(a.out, "images", fn), quality=92)
        t = [-1] * len(cats)
        t[li] = 1
        rows.append([fn, *t])
        prov.append([fn, m["file_name"], m["source"], m["license"], m["license_version"], m["creator"], m["page"],
                     f"{best:.3f}", f"{int(cx0)},{int(cy0)},{int(cx1)},{int(cy1)}"])
    shutil.copyfile(a.categories, os.path.join(a.out, "categories.txt"))
    with open(os.path.join(a.out, "labels.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["file_name", *cats], *rows])
    with open(os.path.join(a.out, "sources.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["file_name", "raw_file", "source", "license", "license_version", "creator", "page",
                                   "score", "crop_box"], *prov])
    print(f"input {len(meta)}, kept {len(rows)}, dropped {dropped}")
    cell, cols = 160, 8
    sel = [r[0] for r in rows][:48]
    if sel:
        sheet = Image.new("RGB", (cols * cell, ((len(sel) + cols - 1) // cols) * cell), "white")
        for i, f in enumerate(sel):
            with Image.open(os.path.join(a.out, "images", f)) as t:
                t = t.convert("RGB")
                t.thumbnail((cell - 4, cell - 4))
            sheet.paste(t, ((i % cols) * cell + 2, (i // cols) * cell + 2))
        sheet.save(os.path.join(a.out, "review.jpg"), quality=85)


if __name__ == "__main__":
    main()
