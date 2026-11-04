"""Merge several external data folders (each: categories.txt, labels.csv with 1/0/-1, images/, sources.csv)
into one folder for --extra_train.

    python scripts/external/merge_external.py --inputs ~/fashion_tagging/external/rare_extra_clean \
        ~/fashion_tagging/external/legwarmer_web_crop --out ~/fashion_tagging/external/external_merged

sources.csv of the result gets a `group` column (the input folder name) next to every input's own columns, so
scripts/train.py --extra_color match can tone each group separately. Images are hard-linked (copied if that fails).
File names must be unique across inputs (they are: lvis_/openimages_/openverse_/wikimedia_ prefixes).
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    shutil.rmtree(os.path.join(out, "images"), ignore_errors=True)   # rebuilt from the inputs every time;
    os.makedirs(os.path.join(out, "images"))                         # README.md and other notes are kept
    cats, header, labels, sources, keys = None, None, [], [], ["file_name", "group"]
    counts = {}
    for d in map(os.path.expanduser, a.inputs):
        group = os.path.basename(os.path.normpath(d))
        c = open(os.path.join(d, "categories.txt"), encoding="utf-8-sig").read()
        if cats is None:
            cats = c
            shutil.copyfile(os.path.join(d, "categories.txt"), os.path.join(out, "categories.txt"))
        elif c != cats:
            raise SystemExit(f"{d}: categories.txt differs from the first input")
        rows = list(csv.reader(open(os.path.join(d, "labels.csv"), encoding="utf-8")))
        if header is None:
            header = rows[0]
        elif rows[0] != header:
            raise SystemExit(f"{d}: labels.csv header differs")
        src = {r["file_name"]: r for r in csv.DictReader(open(os.path.join(d, "sources.csv"), encoding="utf-8"))}
        for r in rows[1:]:
            dst = os.path.join(out, "images", r[0])
            if os.path.exists(dst):
                raise SystemExit(f"duplicate file name across inputs: {r[0]}")
            try:
                os.link(os.path.join(d, "images", r[0]), dst)
            except OSError:
                shutil.copyfile(os.path.join(d, "images", r[0]), dst)
            labels.append(r)
            s = {k: v for k, v in src.get(r[0], {}).items() if k != "file_name"}
            keys += [k for k in s if k not in keys]
            sources.append({"file_name": r[0], "group": group, **s})
        counts[group] = len(rows) - 1
    with open(os.path.join(out, "labels.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([header, *labels])
    with open(os.path.join(out, "sources.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(sources)
    cls = header[1:]
    lines = ["# 외부 데이터 통합", "", f"- 합계 {len(labels):,}장: " + ", ".join(f"{g} {n:,}" for g, n in counts.items()), "",
             "| 클래스 | 있음(1) | 없음 확인(0) | 모름(-1) |", "|---|---|---|---|"]
    for j, c in enumerate(cls, 1):
        col = [r[j] for r in labels]
        lines.append(f"| {c} | {col.count('1'):,} | {col.count('0'):,} | {col.count('-1'):,} |")
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:3]))


if __name__ == "__main__":
    main()
