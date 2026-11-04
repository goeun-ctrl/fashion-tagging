"""Collect openly licensed leg warmer photos from Openverse and Wikimedia Commons.

    python scripts/external/collect_openverse_wikimedia.py --out ~/fashion_tagging/external/legwarmer_web

Only licenses that allow modified copies are kept (CC0, public domain, BY, BY-SA, BY-NC, BY-NC-SA); ND licenses are
skipped because the images will be cropped. Every file's source page, license and creator go to meta.csv so the
dataset can be attributed. Images land in <out>/raw/<source>_<id>.<ext>. Already downloaded files are skipped.
"""
import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request

UA = "fashion-tagging-research/0.1 (university course project)"
OPENVERSE_QUERIES = ["leg warmers", "legwarmers", "leg warmer", "knitted leg warmers", "ballet leg warmers"]
ALLOWED = {"cc0", "pdm", "by", "by-sa", "by-nc", "by-nc-sa"}


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=30))


def openverse():
    seen = {}
    for q in OPENVERSE_QUERIES:
        for page in range(1, 13):                       # anonymous API: 20 per page, 240 results per query
            url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode({"q": q, "page_size": 20, "page": page})
            try:
                d = get_json(url)
            except Exception as e:  # noqa: BLE001
                print(f"  openverse '{q}' page {page}: {e}")
                break
            for r in d.get("results", []):
                if r["license"] in ALLOWED and r["id"] not in seen:
                    seen[r["id"]] = {"source": "openverse", "id": r["id"], "title": r.get("title") or "",
                                     "url": r["url"], "page": r.get("foreign_landing_url") or "",
                                     "license": r["license"], "license_version": r.get("license_version") or "",
                                     "creator": r.get("creator") or "", "provider": r.get("provider") or "",
                                     "query": q}
            if page >= (d.get("page_count") or 0):
                break
            time.sleep(0.5)
    return list(seen.values())


def wikimedia():
    api = "https://commons.wikimedia.org/w/api.php?"
    titles = set()
    for cat in ("Category:Legwarmers", "Category:Legwarmers in cosplay"):
        d = get_json(api + urllib.parse.urlencode({"action": "query", "list": "categorymembers", "cmtitle": cat,
                                                   "cmtype": "file", "cmlimit": 500, "format": "json"}))
        titles |= {m["title"] for m in d["query"]["categorymembers"]}
    d = get_json(api + urllib.parse.urlencode({"action": "query", "list": "search", "srnamespace": 6,
                                               "srsearch": "intitle:warmers", "srlimit": 200, "format": "json"}))
    titles |= {s["title"] for s in d["query"]["search"] if "warmer" in s["title"].lower() and "leg" in s["title"].lower()}
    out = []
    titles = sorted(titles)
    for s in range(0, len(titles), 50):
        d = get_json(api + urllib.parse.urlencode({"action": "query", "titles": "|".join(titles[s:s + 50]),
                                                   "prop": "imageinfo", "iiprop": "url|extmetadata|mime",
                                                   "iiurlwidth": 1600, "format": "json"}))
        for p in d["query"]["pages"].values():
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii.get("mime", "").startswith("image/") or ii.get("mime") == "image/svg+xml":
                continue
            md = ii.get("extmetadata", {})
            lic = (md.get("LicenseShortName", {}).get("value") or "").lower().replace(" ", "-")
            norm = ("cc0" if "cc0" in lic else "pdm" if "public-domain" in lic or lic == "pd" else
                    lic.replace("cc-", "").rsplit("-", 1)[0] if lic.startswith("cc-") else lic)
            if norm not in ALLOWED:
                continue
            out.append({"source": "wikimedia", "id": str(p["pageid"]), "title": p["title"],
                        "url": ii.get("thumburl") or ii["url"], "page": ii.get("descriptionurl", ""),
                        "license": norm, "license_version": md.get("LicenseShortName", {}).get("value", ""),
                        "creator": md.get("Artist", {}).get("value", "")[:200], "provider": "wikimedia", "query": "category/title"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    raw = os.path.join(out, "raw")
    os.makedirs(raw, exist_ok=True)
    items = openverse() + wikimedia()
    print(f"candidates: {len(items)} (openverse {sum(i['source'] == 'openverse' for i in items)}, "
          f"wikimedia {sum(i['source'] == 'wikimedia' for i in items)})", flush=True)
    ok = 0
    for it in items:
        ext = os.path.splitext(urllib.parse.urlparse(it["url"]).path)[1].lower()
        ext = ext if ext in (".jpg", ".jpeg", ".png", ".webp") else ".jpg"
        it["file_name"] = f"{it['source']}_{it['id']}{ext}"
        dst = os.path.join(raw, it["file_name"])
        it["downloaded"] = 1 if os.path.exists(dst) else 0
        if not it["downloaded"]:
            try:
                data = urllib.request.urlopen(urllib.request.Request(it["url"], headers={"User-Agent": UA}), timeout=30).read()
                open(dst, "wb").write(data)
                it["downloaded"] = 1
                time.sleep(0.2)
            except Exception as e:  # noqa: BLE001
                it["error"] = str(e)[:120]
        ok += it["downloaded"]
    keys = ["file_name", "source", "id", "title", "license", "license_version", "creator", "provider", "page", "url",
            "query", "downloaded", "error"]
    with open(os.path.join(out, "meta.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(items)
    print(f"downloaded {ok}/{len(items)} -> {raw}")


if __name__ == "__main__":
    main()
