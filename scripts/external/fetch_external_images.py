"""Rebuild the images/ folder of an external dataset from its sources.csv (images are not stored in git).

    python scripts/external/fetch_external_images.py --dataset data/external/external_merged

For every row of labels.csv: download sources.csv `url`; if the row has a `crop_box` ("x0,y0,x1,y1"), apply EXIF
rotation and crop that box exactly as crop_legwarmers.py did. Existing files are skipped, so the command can be
rerun after an interruption. Rows that fail are listed in <dataset>/fetch_failed.csv (a source may have removed
a photo; training simply uses the images that exist only if you drop those rows, see --prune).
"""
import argparse
import csv
import io
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

UA = "fashion-tagging-research/0.1 (university course project)"


def fetch(row, out):
    dst = os.path.join(out, row["file_name"])
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    err = ""
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(urllib.request.Request(row["url"], headers={"User-Agent": UA}),
                                          timeout=30).read()
            if row.get("crop_box"):
                x0, y0, x1, y1 = map(int, row["crop_box"].split(","))
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                img.crop((x0, y0, x1, y1)).save(dst + ".part", "JPEG", quality=92)
            else:
                open(dst + ".part", "wb").write(data)
            os.replace(dst + ".part", dst)
            return "ok"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:150]
            time.sleep(1 + attempt)
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (quick test)")
    ap.add_argument("--prune", action="store_true", help="remove rows whose image could not be fetched from labels.csv")
    a = ap.parse_args()
    out = os.path.join(a.dataset, "images")
    os.makedirs(out, exist_ok=True)
    names = [r[0] for r in list(csv.reader(open(os.path.join(a.dataset, "labels.csv"), encoding="utf-8")))[1:]]
    src = {r["file_name"]: r for r in csv.DictReader(open(os.path.join(a.dataset, "sources.csv"), encoding="utf-8"))}
    rows = [src[n] for n in names][: a.limit or None]
    print(f"{len(rows)} images to check", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        status = list(ex.map(lambda r: fetch(r, out), rows))
    failed = [(r["file_name"], r["url"], s) for r, s in zip(rows, status) if s not in ("ok", "skip")]
    print(f"downloaded {status.count('ok')}, already had {status.count('skip')}, failed {len(failed)}")
    if failed:
        with open(os.path.join(a.dataset, "fetch_failed.csv"), "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([["file_name", "url", "error"], *failed])
        if a.prune:
            bad = {f[0] for f in failed}
            lab = list(csv.reader(open(os.path.join(a.dataset, "labels.csv"), encoding="utf-8")))
            with open(os.path.join(a.dataset, "labels.csv"), "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([lab[0], *[r for r in lab[1:] if r[0] not in bad]])
            print(f"pruned {len(bad)} rows from labels.csv")


if __name__ == "__main__":
    main()
