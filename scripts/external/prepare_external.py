"""Clean and deduplicate downloaded external images and write them in the training format.

    python scripts/external/prepare_external.py --root ~/fashion_tagging/external \
        --ours data/train data/val data/test_public data/examples_labeled --out ~/fashion_tagging/external/rare_extra_clean

Input (made by build_external_labels.py and download_external_images.py under --root):
    external_labels.csv            source, split, image_id, url, 27 class columns (1 / 0 / -1)
    rare_extra/images/<source>_<image_id>.jpg
Steps (each drop is counted in report.md):
    missing / unreadable / too small (short side < --min_side)
    near-duplicate of one of OUR images (64-bit dHash, Hamming distance <= --hash_dist), so no external image
        repeats a train / holdout / val / public-test picture
    near-duplicate of another external image (the first one, in file order, is kept)
    listed in --exclude_list (CSV with a file_name column, e.g. CLIP non-fashion scenes from flag_domains.py)
Output (--out), same layout as a split folder plus provenance:
    categories.txt, labels.csv (file_name + 27 columns, 1 / 0 / -1), images/ (hard links), sources.csv,
    report.md, review/<class>.jpg (random positives for a quick visual check), hash cache
"""
import argparse
import csv
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGETS = ["jumpsuit", "glove", "vest", "cape", "umbrella", "leg warmer"]


def dhash(path):
    """(64-bit difference hash, width, height) or None if unreadable. Uses JPEG draft decoding for speed."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            im.draft("L", (64, 64))
            g = ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.BILINEAR)
        a = np.asarray(g, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        return int(np.packbits(bits).view(">u8")[0]), w, h
    except Exception:  # noqa: BLE001
        return None


def hash_files(paths, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(dhash, paths, chunksize=64))


def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def nearest(query, ref, chunk=2048):
    """Index and Hamming distance of the nearest ref hash for each query hash."""
    best_d = np.full(len(query), 65, dtype=np.int64)
    best_i = np.full(len(query), -1, dtype=np.int64)
    for s in range(0, len(ref), chunk):
        r = ref[s:s + chunk]
        d = popcount64(query[:, None] ^ r[None, :]).astype(np.int64)
        j = d.argmin(1)
        dj = d[np.arange(len(query)), j]
        upd = dj < best_d
        best_d[upd], best_i[upd] = dj[upd], j[upd] + s
    return best_i, best_d


def cached_hashes(cache, paths, workers):
    old = {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        old = {p: (int(h), int(w), int(hh)) for p, h, w, hh in zip(z["paths"], z["hash"], z["w"], z["h"])}
    todo = [p for p in paths if p not in old]
    for p, r in zip(todo, hash_files(todo, workers) if todo else []):
        old[p] = r if r is not None else (0, -1, -1)
    np.savez(cache, paths=np.array(list(old)), hash=np.array([v[0] for v in old.values()], dtype=np.uint64),
             w=np.array([v[1] for v in old.values()]), h=np.array([v[2] for v in old.values()]))
    return [old[p] for p in paths]


def review_sheet(out_png, files, title, n=40, cols=8, cell=160, seed=0):
    random.Random(seed).shuffle(files)
    files = files[:n]
    if not files:
        return
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell + 24), "white")
    ImageDraw.Draw(sheet).text((4, 4), title, fill="black")
    for i, f in enumerate(files):
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((cell - 4, cell - 4))
        sheet.paste(im, ((i % cols) * cell + 2, 24 + (i // cols) * cell + 2))
    sheet.save(out_png, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with external_labels.csv and rare_extra/images")
    ap.add_argument("--ours", nargs="+", required=True, help="our split folders (each has images/) to dedup against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_side", type=int, default=128)
    ap.add_argument("--hash_dist", type=int, default=4, help="Hamming distance counted as a duplicate (0-64)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--exclude_list", nargs="*", default=[], help="CSVs with file_name (+ optional reason) to drop")
    a = ap.parse_args()

    root, out = os.path.expanduser(a.root), os.path.expanduser(a.out)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "review"), exist_ok=True)
    cats_path = os.path.join(a.ours[0], "categories.txt")
    cats = [l.rstrip("\n") for l in open(cats_path, encoding="utf-8-sig") if l.strip()]
    rows = list(csv.DictReader(open(os.path.join(root, "external_labels.csv"), encoding="utf-8")))
    assert [c for c in rows[0] if c in cats] == cats, "external_labels.csv class columns differ from categories.txt"

    # ---- our images ------------------------------------------------------------------------
    our_paths, our_set = [], []
    for d in a.ours:
        imgs = sorted(os.listdir(os.path.join(d, "images")))
        our_paths += [os.path.join(d, "images", f) for f in imgs]
        our_set += [os.path.basename(os.path.normpath(d))] * len(imgs)
    print(f"hashing {len(our_paths)} of our images ...", flush=True)
    our_h = cached_hashes(os.path.join(out, "hash_cache_ours.npz"), our_paths, a.workers)
    ok = [i for i, r in enumerate(our_h) if r[1] > 0]
    our_hash = np.array([our_h[i][0] for i in ok], dtype=np.uint64)
    our_set = [our_set[i] for i in ok]

    # ---- external images -------------------------------------------------------------------
    ext_paths = [os.path.join(root, "rare_extra", "images", f"{r['source']}_{r['image_id']}.jpg") for r in rows]
    present = [os.path.exists(p) for p in ext_paths]
    print(f"hashing {sum(present)} external images ...", flush=True)
    ext_h = cached_hashes(os.path.join(out, "hash_cache_external.npz"), [p for p, e in zip(ext_paths, present) if e],
                          a.workers)
    it = iter(ext_h)
    ext_h = [next(it) if e else None for e in present]

    drop = {}
    excl = {}
    for path in a.exclude_list:
        for r in csv.DictReader(open(os.path.expanduser(path), encoding="utf-8")):
            excl[r["file_name"]] = (r.get("reason") or os.path.basename(path)).split(" ")[0:2]
    for i, r in enumerate(rows):
        fn = f"{r['source']}_{r['image_id']}.jpg"
        if fn in excl:
            drop[i] = "excluded by list (" + " ".join(excl[fn]) + ")"
    for i, (p, r) in enumerate(zip(ext_paths, ext_h)):
        if i in drop:
            continue
        if r is None:
            drop[i] = "missing"
        elif r[1] <= 0:
            drop[i] = "unreadable"
        elif min(r[1], r[2]) < a.min_side:
            drop[i] = f"small (<{a.min_side}px)"
    cand = [i for i in range(len(rows)) if i not in drop]
    q = np.array([ext_h[i][0] for i in cand], dtype=np.uint64)
    j, d = nearest(q, our_hash)
    for k, i in enumerate(cand):
        if d[k] <= a.hash_dist:
            drop[i] = f"duplicate of ours ({our_set[j[k]]})"
    cand = [i for i in cand if i not in drop]
    q = np.array([ext_h[i][0] for i in cand], dtype=np.uint64)
    kept_hash, n_kept = np.empty(len(cand), dtype=np.uint64), 0
    for k, i in enumerate(cand):          # greedy in file order: keep the first of each near-duplicate group
        if n_kept and popcount64(kept_hash[:n_kept] ^ q[k]).min() <= a.hash_dist:
            drop[i] = "duplicate within external"
            continue
        kept_hash[n_kept] = q[k]
        n_kept += 1
    keep = [i for i in range(len(rows)) if i not in drop]

    # ---- write -----------------------------------------------------------------------------
    shutil.copyfile(cats_path, os.path.join(out, "categories.txt"))
    with open(os.path.join(out, "labels.csv"), "w", newline="", encoding="utf-8") as fl, \
            open(os.path.join(out, "sources.csv"), "w", newline="", encoding="utf-8") as fs:
        wl, ws = csv.writer(fl), csv.writer(fs)
        wl.writerow(["file_name", *cats])
        ws.writerow(["file_name", "source", "split", "image_id", "url"])
        for i in keep:
            r = rows[i]
            fn = f"{r['source']}_{r['image_id']}.jpg"
            dst = os.path.join(out, "images", fn)
            if not os.path.exists(dst):
                try:
                    os.link(ext_paths[i], dst)
                except OSError:
                    shutil.copyfile(ext_paths[i], dst)
            wl.writerow([fn, *(r[c] for c in cats)])
            ws.writerow([fn, r["source"], r["split"], r["image_id"], r["url"]])

    from collections import Counter
    reasons = Counter(drop.values())
    src_kept = Counter(rows[i]["source"] for i in keep)
    L = ["# 외부 데이터 정리 결과", "",
         f"- 입력 {len(rows):,}장 → **사용 {len(keep):,}장** ({', '.join(f'{k} {v:,}' for k, v in src_kept.items())})",
         f"- 중복 기준: 64비트 dHash Hamming 거리 ≤ {a.hash_dist}, 최소 짧은 변 {a.min_side}px",
         f"- 비교한 우리 이미지: {len(our_hash):,}장 ({', '.join(f'{k} {v:,}' for k, v in Counter(our_set).items())})", "",
         "## 제외", "", "| 이유 | 장수 |", "|---|---|", *[f"| {k} | {v:,} |" for k, v in reasons.most_common()], "",
         "## 클래스별 라벨 (사용 이미지 기준)", "", "| 클래스 | 있음(1) | 없음 확인(0) | 모름(-1) |", "|---|---|---|---|"]
    for c in cats:
        col = [int(rows[i][c]) for i in keep]
        L.append(f"| {c} | {col.count(1):,} | {col.count(0):,} | {col.count(-1):,} |")
    L += ["", "## 눈으로 확인할 것", "",
          "`review/<클래스>.jpg`에 대상 클래스 양성 이미지 무작위 40장. 라벨 정의와 다른 것(예: 안전조끼, 파라솔)이",
          "많으면 해당 출처·클래스를 빼거나 걸러야 한다. 이 보고서의 숫자는 그 확인 전 기준이다."]
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    for t in TARGETS:
        files = [os.path.join(out, "images", f"{rows[i]['source']}_{rows[i]['image_id']}.jpg")
                 for i in keep if rows[i][t] == "1"]
        review_sheet(os.path.join(out, "review", f"{t.replace(' ', '_')}.jpg"), files, f"{t}: {len(files)} positives")
    print("\n".join(L[:12]))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
