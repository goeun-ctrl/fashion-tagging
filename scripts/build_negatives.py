"""Empty-label negatives: scenes without people for scripts/train_hires.py --neg_list / --gate.

Why: the provided test_public / examples_labeled sets contain images with no person (street at night, food,
buildings, cars) whose ground truth is the empty label set. Our train/val (Fashionpedia) have 0 such images,
so the classifier always predicts some garments.

Sources (downloaded to --root, default /data/eunjin/DL/neg_data):
  COCO train2017   images with no annotation of person or of objects that are our classes on their own
                   (handbag/backpack/suitcase -> bag, umbrella, tie). Fetched one by one from coco_url.
  Places365 val    val_large.tar (36,500 scene images, high resolution).
Both are then checked with torchvision Faster R-CNN R50-FPN v2 (COCO) and dropped when it sees a person
(score >= --person_thr) or one of the objects above (score >= --object_thr). The detector is used only here,
never in the submitted model.

    python scripts/build_negatives.py coco   --n 6000
    python scripts/build_negatives.py places --n 8000
    python scripts/build_negatives.py filter          # GPU; writes detector scores
    python scripts/build_negatives.py split  --n_holdout 1000
    python scripts/build_negatives.py cocopos --n 5000   # person positives (gate v2), writes pos_train/holdout.txt

Outputs in --root: neg_train.txt, neg_holdout.txt (absolute paths), negatives.csv (source, path, max person
score, max object score, kept, split).
"""
import argparse
import csv
import json
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DROP_COCO = {"person", "handbag", "backpack", "suitcase", "umbrella", "tie"}
DET_OBJECTS = {"handbag", "backpack", "suitcase", "umbrella", "tie"}


def coco(a):
    ann = json.load(open(os.path.join(a.root, "annotations", "instances_train2017.json")))
    cat = {c["id"]: c["name"] for c in ann["categories"]}
    bad = {x["image_id"] for x in ann["annotations"] if cat[x["category_id"]] in DROP_COCO}
    imgs = [im for im in ann["images"] if im["id"] not in bad and min(im["width"], im["height"]) >= 300]
    print(f"COCO train2017: {len(ann['images'])} images, {len(imgs)} without person/bag/umbrella/tie (short side >= 300)")
    random.Random(0).shuffle(imgs)
    imgs = imgs[:a.n]
    out = os.path.join(a.root, "coco")
    os.makedirs(out, exist_ok=True)

    def get(im):
        dst = os.path.join(out, im["file_name"])
        if not os.path.exists(dst):
            urllib.request.urlretrieve(im["coco_url"], dst + ".part")
            os.replace(dst + ".part", dst)
        return dst
    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(get, imgs))
    with open(os.path.join(a.root, "cand_coco.txt"), "w") as f:
        f.write("\n".join(paths) + "\n")
    print(f"downloaded {len(paths)} to {out}")


def cocopos(a):
    """Person positives from outside Fashionpedia (crowds, distant people, everyday scenes, old photos): gate = 1,
    class labels unknown. They stop the gate from learning "Fashionpedia-looking photo" instead of "person"."""
    ann = json.load(open(os.path.join(a.root, "annotations", "instances_train2017.json")))
    pid = [c["id"] for c in ann["categories"] if c["name"] == "person"][0]
    has = {x["image_id"] for x in ann["annotations"] if x["category_id"] == pid and not x.get("iscrowd") and x["area"] >= a.min_area}
    imgs = [im for im in ann["images"] if im["id"] in has and min(im["width"], im["height"]) >= 300]
    print(f"COCO train2017: {len(imgs)} images with a person (area >= {a.min_area} px, short side >= 300)")
    random.Random(2).shuffle(imgs)
    imgs = imgs[:a.n]
    out = os.path.join(a.root, "coco_person")
    os.makedirs(out, exist_ok=True)

    def get(im):
        dst = os.path.join(out, im["file_name"])
        if not os.path.exists(dst):
            urllib.request.urlretrieve(im["coco_url"], dst + ".part")
            os.replace(dst + ".part", dst)
        return dst
    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(get, imgs))
    places_pos = [r["path"] for r in csv.DictReader(open(os.path.join(a.root, "det_scores.csv")))
                  if r["source"] == "places" and float(r["person"]) >= a.pos_thr]
    rng = random.Random(3)
    rng.shuffle(paths)
    rng.shuffle(places_pos)
    h = a.n_holdout // 2
    hold, train = paths[:h] + places_pos[:h], paths[h:] + places_pos[h:]
    for name, part in (("pos_train.txt", train), ("pos_holdout.txt", hold)):
        with open(os.path.join(a.root, name), "w") as f:
            f.write("\n".join(part) + "\n")
    print(f"COCO person {len(paths)}, Places person (det >= {a.pos_thr}) {len(places_pos)} -> train {len(train)}, holdout {len(hold)}")


def places(a):
    base = os.path.join(a.root, "places")
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(base) for f in fs if f.lower().endswith(".jpg"))
    print(f"Places365 val: {len(files)} images")
    random.Random(0).shuffle(files)
    with open(os.path.join(a.root, "cand_places.txt"), "w") as f:
        f.write("\n".join(files[:a.n]) + "\n")
    print(f"{min(a.n, len(files))} candidates")


def filter_(a):
    import torch
    import torchvision
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    names = weights.meta["categories"]
    pid = names.index("person")
    oid = [names.index(n) for n in DET_OBJECTS]
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.05).cuda().eval()
    rows = []
    for src in ("coco", "places"):
        paths = [l.strip() for l in open(os.path.join(a.root, f"cand_{src}.txt")) if l.strip()]
        for k in range(0, len(paths), a.batch):
            chunk = paths[k:k + a.batch]
            ims = []
            for p in chunk:
                with Image.open(p) as im:
                    ims.append(to_tensor(im.convert("RGB")).cuda())
            with torch.no_grad():
                outs = model(ims)
            for p, o in zip(chunk, outs):
                lab, sc = o["labels"].cpu(), o["scores"].cpu()
                ps = float(sc[lab == pid].max()) if (lab == pid).any() else 0.0
                om = torch.isin(lab, torch.tensor(oid))
                os_ = float(sc[om].max()) if om.any() else 0.0
                rows.append({"source": src, "path": p, "person": round(ps, 4), "object": round(os_, 4)})
            if k % (a.batch * 50) == 0:
                print(src, k, len(paths), flush=True)
    with open(os.path.join(a.root, "det_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("scored", len(rows))


def split(a):
    rows = list(csv.DictReader(open(os.path.join(a.root, "det_scores.csv"))))
    for r in rows:
        r["kept"] = int(float(r["person"]) < a.person_thr and float(r["object"]) < a.object_thr)
    kept = {s: [r for r in rows if r["source"] == s and r["kept"]] for s in ("coco", "places")}
    for s, v in kept.items():
        tot = sum(r["source"] == s for r in rows)
        print(f"{s}: kept {len(v)}/{tot} (person < {a.person_thr}, object < {a.object_thr})")
    rng = random.Random(1)
    for s in kept:
        rng.shuffle(kept[s])
    half = a.n_holdout // 2
    hold = kept["coco"][:half] + kept["places"][:a.n_holdout - half]
    train = kept["coco"][half:] + kept["places"][a.n_holdout - half:]
    for r in rows:
        r["split"] = ""
    for r in hold:
        r["split"] = "holdout"
    for r in train:
        r["split"] = "train"
    for name, part in (("neg_train.txt", train), ("neg_holdout.txt", hold)):
        with open(os.path.join(a.root, name), "w") as f:
            f.write("\n".join(r["path"] for r in part) + "\n")
    with open(os.path.join(a.root, "negatives.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"train {len(train)} (coco {sum(r['source'] == 'coco' for r in train)}), holdout {len(hold)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["coco", "places", "filter", "split", "cocopos"])
    ap.add_argument("--root", default="/data/eunjin/DL/neg_data")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--person_thr", type=float, default=0.3)
    ap.add_argument("--object_thr", type=float, default=0.5)
    ap.add_argument("--n_holdout", type=int, default=1000)
    ap.add_argument("--min_area", type=float, default=0, help="cocopos: smallest person box area kept (0 = any)")
    ap.add_argument("--pos_thr", type=float, default=0.7, help="cocopos: Places images with person score >= this")
    a = ap.parse_args()
    {"coco": coco, "places": places, "filter": filter_, "split": split, "cocopos": cocopos}[a.step](a)


if __name__ == "__main__":
    main()
