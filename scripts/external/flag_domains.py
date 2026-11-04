"""Flag external images from strongly non-fashion domains (sports games, military, workwear, labs, toys,
object-only umbrella scenes, ...) with CLIP zero-shot scene matching.

    python scripts/external/flag_domains.py --dataset data/external/rare_extra_clean --out flagged_domains.csv

Each image is compared with the scene prompts below. Probabilities are summed per domain; an image is flagged
when its most likely domain is an EXCLUDE domain with probability >= --threshold. Output:
    flagged_domains.csv   file_name, top_domain, top_prob, flagged (0/1), one column per domain
    flag_summary.txt      counts per domain at the chosen threshold and at nearby thresholds
    review_flag_<domain>.jpg / review_kept_borderline.jpg   sheets for a quick visual check
Feed the flagged list to prepare_external.py --exclude_list.
"""
import argparse
import csv
import os
import random
from collections import Counter

import torch
from PIL import Image, ImageDraw, ImageOps

EXCLUDE = {
    "baseball": ["a photo of a baseball game", "a baseball player in a baseball uniform", "a softball game",
                 "a baseball pitcher or batter"],
    "team_sports": ["a soccer match", "an american football game", "a basketball game", "an ice hockey game",
                    "a rugby match", "a cricket match", "a volleyball game"],
    "other_sports": ["a boxing or martial arts match", "a tennis match", "a cyclist in a bike race",
                     "a motorcycle race", "a golfer on a golf course", "a runner in a marathon race"],
    "winter_sports": ["a person skiing on a ski slope", "a snowboarder doing a trick", "a ski race"],
    "military": ["soldiers in military uniform", "a military parade", "an army soldier carrying a rifle",
                 "military personnel in camouflage"],
    "workwear": ["a construction worker in a safety vest", "an airport ground crew worker",
                 "a factory or warehouse worker", "a worker wearing work gloves doing manual labor",
                 "a police officer in uniform", "a firefighter in protective gear"],
    "medical_lab": ["a doctor or nurse in a hospital", "a scientist in a laboratory wearing gloves",
                    "a person wearing rubber cleaning gloves", "hands in disposable medical gloves"],
    "kitchen": ["a chef cooking in a kitchen", "a person baking food in a kitchen"],
    "toys": ["a stuffed toy or teddy bear", "a doll or action figure", "a carousel horse"],
    "equestrian": ["a person riding a horse", "a rodeo event", "a farmer with livestock"],
    "objects_scene": ["a beach with sun umbrellas", "an outdoor cafe with patio umbrellas",
                      "a market stall with umbrellas", "a cocktail with a paper umbrella",
                      "a decorated cake or cupcakes", "a pile of products for sale", "an animal"],
    "vehicles": ["an airplane at an airport", "a train at a railway station", "a car on the road"],
}
KEEP = {
    "fashion_people": ["a street fashion photo of a person", "a person wearing everyday casual clothes",
                       "a fashion runway show", "a celebrity on the red carpet", "people walking on a city street",
                       "a person holding an umbrella in the rain", "a person wearing a winter coat, gloves and a scarf",
                       "a portrait photo of a person", "a group of friends at a party", "a person shopping",
                       "a wedding photo", "a family photo", "a person wearing a cape or poncho",
                       "a person wearing a vest", "a person wearing a jumpsuit"],
}


def sheet(path, files, title, n=40, cols=8, cell=150, seed=0):
    files = list(files)
    random.Random(seed).shuffle(files)
    files = files[:n]
    if not files:
        return
    rows = (len(files) + cols - 1) // cols
    img = Image.new("RGB", (cols * cell, rows * cell + 22), "white")
    ImageDraw.Draw(img).text((4, 4), title, fill="black")
    for i, f in enumerate(files):
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((cell - 4, cell - 4))
        img.paste(im, ((i % cols) * cell + 2, 22 + (i // cols) * cell + 2))
    img.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="folder with images/ and labels.csv")
    ap.add_argument("--out", required=True, help="output CSV path (other outputs go next to it)")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    import open_clip

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained, device=dev)
    tok = open_clip.get_tokenizer(a.model)
    model.eval()
    domains = {**EXCLUDE, **KEEP}
    prompts, owner = [], []
    for d, ps in domains.items():
        prompts += ps
        owner += [d] * len(ps)
    with torch.no_grad():
        t = model.encode_text(tok(prompts).to(dev)).float()
        t = t / t.norm(dim=-1, keepdim=True)
    dnames = list(domains)
    owner_idx = torch.tensor([dnames.index(d) for d in owner], device=dev)

    names = [r["file_name"] for r in csv.DictReader(open(os.path.join(a.dataset, "labels.csv"), encoding="utf-8"))]
    probs = []
    for s in range(0, len(names), a.batch):
        batch = names[s:s + a.batch]
        x = torch.stack([prep(Image.open(os.path.join(a.dataset, "images", n)).convert("RGB")) for n in batch]).to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
            f = model.encode_image(x).float()
        f = f / f.norm(dim=-1, keepdim=True)
        p = (100.0 * f @ t.T).softmax(dim=-1)                                     # [B, prompts]
        pd = torch.zeros(len(batch), len(dnames), device=dev).index_add_(1, owner_idx, p)   # sum per domain
        probs += pd.cpu().tolist()
        if (s // a.batch) % 20 == 0:
            print(f"  {s + len(batch)}/{len(names)}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(a.out))
    flagged = []
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file_name", "top_domain", "top_prob", "flagged", *dnames])
        for n, pd in zip(names, probs):
            k = max(range(len(dnames)), key=lambda i: pd[i])
            fl = int(dnames[k] in EXCLUDE and pd[k] >= a.threshold)
            if fl:
                flagged.append((n, dnames[k], pd[k]))
            w.writerow([n, dnames[k], f"{pd[k]:.3f}", fl, *(f"{v:.3f}" for v in pd)])

    lines = [f"model {a.model}/{a.pretrained}, images {len(names)}, threshold {a.threshold}",
             f"flagged {len(flagged)} ({100 * len(flagged) / len(names):.1f}%)"]
    for d, c in Counter(d for _, d, _ in flagged).most_common():
        lines.append(f"  {d:14s} {c}")
    for thr in (0.4, 0.5, 0.6, 0.7):
        cnt = sum(1 for pd in probs if (lambda k: dnames[k] in EXCLUDE and pd[k] >= thr)(max(range(len(dnames)), key=lambda i: pd[i])))
        lines.append(f"at threshold {thr}: {cnt}")
    open(os.path.join(out_dir, "flag_summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    img = lambda n: os.path.join(a.dataset, "images", n)  # noqa: E731
    for d in EXCLUDE:
        fs = [img(n) for n, dd, _ in flagged if dd == d]
        sheet(os.path.join(out_dir, f"review_flag_{d}.jpg"), fs, f"flagged as {d}: {len(fs)}")
    border = [img(n) for n, pd in zip(names, probs)
              if (lambda k: dnames[k] in EXCLUDE and 0.3 <= pd[k] < a.threshold)(max(range(len(dnames)), key=lambda i: pd[i]))]
    sheet(os.path.join(out_dir, "review_kept_borderline.jpg"), border, f"kept, exclude-domain prob 0.3-{a.threshold}: {len(border)}")


if __name__ == "__main__":
    main()
