"""Build 27-class labels for external candidate images (LVIS train+val, Open Images train+val+test).

For each image and each of our 27 classes the label is
   1  the class is present (an annotation / human-verified positive of a mapped class)
   0  the class is verified absent (EVERY mapped class is a verified negative for that image)
  -1  unknown -> excluded from the loss when training
Only clear one-to-one meanings are mapped; ambiguous external classes (generic shirt, legging, overalls,
parasol, suit, robe, helmet, ...) are left out, so they only ever produce -1.

Candidates = images positive for at least one target class (jumpsuit, glove, vest, cape, umbrella, leg warmer).

    python scripts/external/build_external_labels.py --root ~/fashion_tagging/external --categories data/train/categories.txt
Needs under --root: lvis/lvis_v1_{train,val}.json, openimages/oidv7-class-descriptions.csv,
openimages/oidv7-{val,test}-annotations-human-imagelabels.csv, openimages/train_candidate_all_labels.csv
(the train label rows of candidate images).
Output under --root: external_labels.csv (source, split, image_id, url, 27 class columns), external_labels_summary.csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--categories", default="data/train/categories.txt")
ARGS = ap.parse_args()
E = os.path.expanduser(ARGS.root)
CATS = [l.rstrip("\n") for l in open(ARGS.categories, encoding="utf-8-sig") if l.strip()]
TARGETS = ["jumpsuit", "glove", "vest", "cape", "umbrella", "leg warmer"]

# our class -> (LVIS category names, Open Images display names)
MAP = {
    "shirt, blouse": (["blouse"], ["Blouse"]),
    "top, t-shirt, sweatshirt": (["jersey", "sweatshirt"], ["T-shirt", "Sweatshirt"]),
    "sweater": (["sweater"], ["Sweater"]),
    "cardigan": (["cardigan"], ["Cardigan"]),
    "jacket": (["jacket"], ["Jacket"]),
    "vest": (["vest"], ["Vest", "Waistcoat"]),
    "pants": (["trousers", "jean", "sweat_pants"], ["Trousers", "Jeans"]),
    "shorts": (["short_pants"], ["Shorts"]),
    "skirt": (["skirt", "ballet_skirt"], ["Skirt", "Miniskirt"]),
    "coat": (["coat", "raincoat", "trench_coat"], ["Coat", "Overcoat"]),
    "dress": (["dress"], ["Dress"]),
    "jumpsuit": (["jumpsuit", "coverall"], ["Jumpsuit", "Romper suit"]),
    "cape": (["cape", "cloak", "poncho"], ["Cape", "Cloak", "Poncho"]),
    "glasses": (["sunglasses"], ["Glasses", "Sunglasses"]),
    "hat": (["hat", "baseball_cap", "beret", "bowler_hat", "cowboy_hat", "dress_hat", "sunhat", "bonnet", "skullcap"],
            ["Hat", "Baseball cap", "Beret", "Cap", "Cowboy hat", "Fedora", "Sombrero", "Sun hat"]),
    "headband, head covering, hair accessory": (["headband", "headscarf", "hairpin", "tiara", "turban", "veil"],
                                                ["Headband", "Hair accessory"]),
    "tie": (["necktie", "bow-tie", "bolo_tie"], ["Tie", "Bow tie"]),
    "glove": (["glove", "mitten"], ["Glove", "Mittens"]),
    "watch": (["watch"], ["Watch"]),
    "belt": (["belt"], ["Belt"]),
    "leg warmer": ([], []),
    "tights, stockings": (["tights_(clothing)"], ["Tights", "Stocking"]),
    "sock": (["sock"], ["Sock"]),
    "shoe": (["shoe", "boot", "sandal_(type_of_shoe)", "flip-flop_(sandal)", "slipper_(footwear)", "ski_boot",
              "arctic_(type_of_shoe)"], ["Footwear", "Shoe", "Boot", "High heels", "Sandal", "Sneakers"]),
    "bag, wallet": (["handbag", "backpack", "wallet", "clutch_bag", "shoulder_bag", "tote_bag", "satchel", "briefcase",
                     "duffel_bag"], ["Handbag", "Backpack", "Wallet", "Briefcase"]),
    "scarf": (["scarf", "shawl"], ["Scarf"]),
    "umbrella": (["umbrella"], ["Umbrella"]),
}
assert set(MAP) == set(CATS), set(CATS) ^ set(MAP)


def decide(pos_names, neg_names, mapped):
    if not mapped:
        return -1
    if any(n in pos_names for n in mapped):
        return 1
    if all(n in neg_names for n in mapped):
        return 0
    return -1


rows = []
# ---------------- LVIS ----------------
for split in ("train", "val"):
    d = json.load(open(os.path.join(E, "lvis", f"lvis_v1_{split}.json")))
    cname = {c["id"]: c["name"] for c in d["categories"]}
    pos = defaultdict(set)
    for a in d["annotations"]:
        pos[a["image_id"]].add(cname[a["category_id"]])
    for im in d["images"]:
        p = pos.get(im["id"], set())
        neg = {cname[i] for i in im.get("neg_category_ids", [])}
        lab = [decide(p, neg, MAP[c][0]) for c in CATS]
        if any(lab[CATS.index(t)] == 1 for t in TARGETS):
            rows.append(["lvis", split, im["id"], im["coco_url"], *lab])

# ---------------- Open Images ----------------
mid2name = {r[0]: r[1] for r in csv.reader(open(os.path.join(E, "openimages", "oidv7-class-descriptions.csv"))) if len(r) == 2}
files = {"train": "train_candidate_all_labels.csv", "validation": "oidv7-val-annotations-human-imagelabels.csv",
         "test": "oidv7-test-annotations-human-imagelabels.csv"}
for split, fn in files.items():
    pos, neg = defaultdict(set), defaultdict(set)
    with open(os.path.join(E, "openimages", fn)) as fh:
        for r in csv.reader(fh):
            if r[0] == "ImageID":
                continue
            name = mid2name.get(r[2])
            if name is None:
                continue
            (pos if r[3] == "1" else neg)[r[0]].add(name)
    for iid in set(pos) | set(neg):
        lab = [decide(pos[iid], neg[iid], MAP[c][1]) for c in CATS]
        if any(lab[CATS.index(t)] == 1 for t in TARGETS):
            rows.append(["openimages", split, iid, f"https://open-images-dataset.s3.amazonaws.com/{split}/{iid}.jpg", *lab])

with open(os.path.join(E, "external_labels.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["source", "split", "image_id", "url", *CATS])
    w.writerows(rows)

summ = []
for j, c in enumerate(CATS):
    col = [r[4 + j] for r in rows]
    summ.append([c, col.count(1), col.count(0), col.count(-1)])
with open(os.path.join(E, "external_labels_summary.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["class", "positive", "verified_negative", "unknown"])
    w.writerows(summ)
by = defaultdict(int)
for r in rows:
    by[r[0]] += 1
print(f"candidate images: {len(rows)} ({dict(by)})")
print(f"{'class':42s}{'pos':>7s}{'neg':>7s}{'unknown':>9s}")
for c, p, n, u in summ:
    print(f"{c:42s}{p:7d}{n:7d}{u:9d}")
