"""Label utilities shared by the whole team.

The class order is the order of lines in categories.txt. Names contain commas
(e.g. "shirt, blouse"), so a line is one class and must never be split.
"""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

NUM_CLASSES = 27


def parse_categories(text):
    """One non-empty line = one class. Only the line ending is stripped."""
    names = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    if len(names) != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} categories, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate category names")
    return names


def load_categories(path):
    return parse_categories(Path(path).read_text(encoding="utf-8-sig"))


def parse_labels_csv(text, categories):
    """Return (file_names, targets[N, 27] uint8) from labels.csv text.

    The labels column is a JSON array of category names and is parsed with
    json.loads (never eval). It is cross-checked against the binary columns.
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    names, targets = [], np.zeros((len(rows), len(categories)), dtype=np.uint8)
    index = {c: i for i, c in enumerate(categories)}
    for r, row in enumerate(rows):
        names.append(row["file_name"])
        for label in json.loads(row["labels"]):
            targets[r, index[label]] = 1
        binary = np.array([int(row[c]) for c in categories], dtype=np.uint8)
        if not np.array_equal(binary, targets[r]):
            raise ValueError(f"JSON labels and binary columns disagree: {row['file_name']}")
    return names, targets


def load_split_from_zip(zip_path, split):
    """Read categories and labels of 'train' or 'val' straight from the ZIP."""
    with zipfile.ZipFile(zip_path) as zf:
        categories = parse_categories(zf.read(f"{split}/categories.txt").decode("utf-8-sig"))
        names, targets = parse_labels_csv(zf.read(f"{split}/labels.csv").decode("utf-8-sig"), categories)
    return categories, names, targets


def load_split(path, split):
    """Categories, file names and targets from a ZIP ('<split>/...' inside) or from an
    extracted folder that directly contains categories.txt, labels.csv and images/."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        return load_split_from_zip(p, split)
    categories = load_categories(p / "categories.txt")
    names, targets = parse_labels_csv((p / "labels.csv").read_text(encoding="utf-8-sig"), categories)
    return categories, names, targets


def load_partial_labels(path, categories):
    """External data whose labels do not cover every class. `path` is a folder with categories.txt
    (same 27 lines, same order), images/ and labels.csv = file_name + one column per class holding
    1 (present), 0 (verified absent) or -1 (unknown; the loss skips it).
    Returns (file_names, targets[N, C] float32 with -1 for unknown)."""
    p = Path(path)
    cats = load_categories(p / "categories.txt")
    if cats != list(categories):
        raise ValueError(f"{p}: categories.txt differs from the training categories")
    rows = list(csv.DictReader(io.StringIO((p / "labels.csv").read_text(encoding="utf-8-sig"))))
    names = [r["file_name"] for r in rows]
    targets = np.array([[int(r[c]) for c in categories] for r in rows], dtype=np.float32)
    if not np.isin(targets, (-1, 0, 1)).all():
        raise ValueError(f"{p}/labels.csv: values must be 1, 0 or -1")
    return names, targets


def to_names(row, categories):
    """multi-hot row -> list of exact category names (empty list if none)."""
    return [categories[i] for i in np.flatnonzero(row)]
