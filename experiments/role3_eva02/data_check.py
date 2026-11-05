"""Verify only the two authorized local splits. Emit aggregate evidence."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO.parents[1]
# Repository layout (main): shared code in src/fashion_tagging, fixed splits in data/splits, scripts/train.py.
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from fashion_tagging.data.dataset import FashionDataset, read_names  # noqa: E402
from fashion_tagging.data.labels import load_split  # noqa: E402
from fashion_tagging.paths import SPLITS_DIR as SPLITS  # noqa: E402  (data/splits)

MANIFEST = {
    "train_inner": (43197, "04bebf0668097b138fd3394c7496169b78e8a71005c54e7e322bb4652630ec1e"),
    "holdout": (2425, "be62c24fac22d7eb2650b40d2e75b31033177d0e9d8bb1fe177a431a700baa1a"),
}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def require(test, reason):
    if not test:
        raise ValueError(reason)


class MissingLabeledImages(ValueError):
    """Keep the baseline's validation error, while distinguishing missing inputs."""


def default_root(split):
    """Original task folder for `split` (train/val): the private Windows layout
    <WORK>/MAT4091_fashion_<split>/<split> if it exists, otherwise repo data/<split> (lab server layout)."""
    legacy = WORK / f"MAT4091_fashion_{split}" / split
    return legacy if legacy.exists() else REPO / "data" / split


def source(split, root=None):
    if root is None:
        root = default_root(split)
    cats, names, labels = load_split(root, split)
    with np.load(root / "labels.npz", allow_pickle=False) as z:
        keys = z["filenames"].tolist()
        index = {k: i for i, k in enumerate(keys)}
        require(len(keys) == len(index) == len(set(names)) == len(names), "duplicate/missing label keys")
        require(cats == z["categories"].tolist() and set(names) == set(keys), "CSV/NPZ mapping differs")
        require(np.array_equal(labels, z["targets"][[index[k] for k in names]]), "CSV/NPZ targets differ")
    image_files = {p.name for p in (root / "images").iterdir() if p.is_file()}
    missing = set(names) - image_files
    if missing:
        raise MissingLabeledImages(f"{split}: missing {len(missing)} labeled image files")
    require(set(names) == image_files, "image/label mismatch")
    return root, cats, names, labels


def check_bytes(groups):
    # Only equal-length files can be byte-identical. Hash candidates across splits.
    sizes = defaultdict(list)
    for split, (root, names, _) in groups.items():
        for name in names:
            p = root / "images" / name
            sizes[p.stat().st_size].append((split, p))
    shared = defaultdict(int)
    for candidates in sizes.values():
        if len({s for s, _ in candidates}) < 2:
            continue
        hashes = defaultdict(set)
        for split, path in candidates:
            hashes[hashlib.sha256(path.read_bytes()).digest()].add(split)
        for splits in hashes.values():
            if len(splits) > 1:
                shared["/".join(sorted(splits))] += 1
    return dict(shared)


def verified_data(out=HERE):
    root, cats, names, targets = source("train")
    vroot, vcats, vnames, vtargets = source("val")
    require(cats == vcats and len(cats) == 27 and len(vnames) == 1158, "official val schema/count differs")
    require(not set(names).intersection(vnames), "official val ID also appears in train")
    groups, digests = {}, {}
    index = {k: i for i, k in enumerate(names)}
    for split, (count, digest) in MANIFEST.items():
        p = SPLITS / (split + ".txt")
        ids = read_names(p)
        digests[split] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        require(digests[split] == digest and len(ids) == len(set(ids)) == count, "team manifest changed: " + split)
        groups[split] = root, ids, targets[[index[k] for k in ids]]
    inner, hold = (set(groups[g][1]) for g in MANIFEST)
    require(not inner & hold and inner | hold == set(names), "split overlap or incomplete coverage")
    with (SPLITS / "split_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    require([r["class"] for r in rows] == cats, "team summary class order differs")
    for split in groups:
        require(np.array_equal(groups[split][2].sum(0), [int(r[split]) for r in rows]), "team support differs")
    groups["val"] = vroot, vnames, vtargets
    shared = check_bytes(groups)
    audit = {
        "manifests_lf_sha256": digests, "counts": {s: len(g[1]) for s, g in groups.items()},
        "cross_split_shared_byte_hash_groups": shared, "csv_npz_match": True, "class_count": len(cats),
        "shoe_prevalence": {"train": float(targets[:, cats.index("shoe")].mean()),
                           "val": float(vtargets[:, cats.index("shoe")].mean())},
    }
    write_json(out / "data_audit.json", audit)
    require(not shared, f"cross-split identical image bytes: {shared}")
    return cats, groups


if __name__ == "__main__":
    verified_data()
    print((HERE / "data_audit.json").read_text(encoding="utf-8"))

