"""Download the images listed in external_labels.csv into rare_extra/images/<source>_<image_id>.jpg.

    python scripts/external/download_external_images.py --root ~/fashion_tagging/external [--source lvis|openimages]
Reads <root>/external_labels.csv. Skips files that already exist, so it can be rerun after an interruption.
Failures go to <root>/rare_extra/failed_<source>.csv.
"""
import argparse
import csv
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor



def fetch(row, out):
    dst = os.path.join(out, f"{row['source']}_{row['image_id']}.jpg")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return row, "skip"
    for attempt in range(3):
        try:
            req = urllib.request.Request(row["url"], headers={"User-Agent": "fashion-tagging-research"})
            data = urllib.request.urlopen(req, timeout=30).read()
            tmp = dst + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dst)
            return row, "ok"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            time.sleep(1 + attempt)
    return row, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--source", choices=["lvis", "openimages"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    E = os.path.expanduser(a.root)
    out = os.path.join(E, "rare_extra", "images")
    os.makedirs(out, exist_ok=True)
    rows = [r for r in csv.DictReader(open(os.path.join(E, "external_labels.csv"), encoding="utf-8"))
            if not a.source or r["source"] == a.source]
    print(f"{len(rows)} images to check", flush=True)
    done, failed = {"ok": 0, "skip": 0}, []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (row, status) in enumerate(ex.map(fetch, rows, [out] * len(rows)), 1):
            if status in done:
                done[status] += 1
            else:
                failed.append([row["source"], row["image_id"], row["url"], status])
            if i % 500 == 0:
                print(f"  {i}/{len(rows)} ok {done['ok']} skip {done['skip']} failed {len(failed)}", flush=True)
    if failed:
        with open(os.path.join(E, "rare_extra", f"failed_{a.source or 'all'}.csv"), "w", newline="") as fh:
            csv.writer(fh).writerows([["source", "image_id", "url", "error"], *failed])
    print(f"done: downloaded {done['ok']}, already had {done['skip']}, failed {len(failed)}", flush=True)


if __name__ == "__main__":
    main()
