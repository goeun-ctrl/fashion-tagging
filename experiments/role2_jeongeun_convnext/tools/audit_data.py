#!/usr/bin/env python3
"""Audit actual server data with the untouched common label parser and fixed splits.

This validates labels and image integrity; it does not evaluate a model on official val.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image

from download_data import DEFAULT_MANIFEST, safe_zip_members, write_json_preserving


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_common_labels(reference: Path):
    path = reference / "common" / "labels.py"
    spec = importlib.util.spec_from_file_location("untouched_team_labels_for_audit", path)
    require(spec is not None and spec.loader is not None, "Cannot load common/labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PartitionReader:
    def __init__(self, root: Path, partition: str, spec: dict):
        self.root, self.partition, self.spec = root, partition, spec
        self.zf = zipfile.ZipFile(root) if root.suffix.lower() == ".zip" else None
        if self.zf:
            safe_zip_members(self.zf, spec)

    def read(self, relative: str) -> bytes:
        if self.zf:
            return self.zf.read(f"{self.partition}/{relative}")
        return (self.root / relative).read_bytes()

    def image_names(self) -> list[str]:
        if self.zf:
            prefix = f"{self.partition}/images/"
            return [i.filename[len(prefix):] for i in self.zf.infolist()
                    if i.filename.startswith(prefix) and not i.is_dir()]
        image_dir = self.root / "images"
        require(image_dir.is_dir(), f"Missing image directory: {image_dir}")
        return [p.relative_to(image_dir).as_posix() for p in image_dir.rglob("*") if p.is_file()]

    def decode_image(self, name: str) -> tuple[str, int, int]:
        if self.zf:
            source = io.BytesIO(self.read("images/" + name))
        else:
            source = self.root / "images" / name
        with Image.open(source) as image:
            image.load()
            require(image.width > 0 and image.height > 0, f"Invalid image size: {name}")
            # Exercise the RGB conversion used by the common preprocessing path.
            rgb = image.convert("RGB")
            rgb.load()
            return image.mode, image.width, image.height

    def close(self):
        if self.zf:
            self.zf.close()


def validate_csv_raw(text: str, categories: list[str]) -> list[str]:
    reader = csv.DictReader(io.StringIO(text))
    require(reader.fieldnames == ["file_name", "labels", *categories], "CSV columns/class order differ from the observed contract")
    names = []
    for row_number, row in enumerate(reader, start=2):
        name = row["file_name"]
        require(bool(name) and Path(name).name == name and "/" not in name and "\\" not in name,
                f"Unsafe/non-basename image name at CSV row {row_number}")
        require(name.lower().endswith(".jpg"), f"Unexpected image extension: {name}")
        labels = json.loads(row["labels"])
        require(isinstance(labels, list) and all(isinstance(v, str) for v in labels), f"labels must be a JSON string array: {name}")
        require(len(labels) == len(set(labels)), f"Duplicate label in JSON array: {name}")
        require(set(labels).issubset(categories), f"Unknown category in JSON array: {name}")
        for category in categories:
            require(row[category] in ("0", "1"), f"Non-binary raw CSV value: {name}, {category}")
        names.append(name)
    require(len(names) == len(set(names)), "Duplicate CSV image names")
    return names


def audit_partition(root: Path, partition: str, spec: dict, expected_categories: list[str], common, workers: int = 4) -> tuple[dict, list[str]]:
    reader = PartitionReader(root, partition, spec)
    try:
        category_bytes = reader.read("categories.txt")
        csv_bytes = reader.read("labels.csv")
        categories = common.parse_categories(category_bytes.decode("utf-8-sig"))
        require(categories == expected_categories, f"{partition}: class order differs from manifest")
        require(len(categories) == 27 and len(set(categories)) == 27, f"{partition}: expected 27 unique categories")
        raw_names = validate_csv_raw(csv_bytes.decode("utf-8-sig"), categories)
        # This is the same unchanged parser used by the team's datasets/training code.
        common_categories, names, targets = common.load_split(root, partition)
        require(common_categories == categories and names == raw_names, f"{partition}: common parser row order mismatch")
        require(len(names) == spec["image_count"], f"{partition}: expected {spec['image_count']} labels, got {len(names)}")
        require(targets.shape == (len(names), 27), f"{partition}: wrong target matrix shape")
        require(bool(np.isin(targets, (0, 1)).all()), f"{partition}: target matrix is not binary")
        images = reader.image_names()
        require(len(images) == len(set(images)), f"{partition}: duplicate image paths")
        missing, extra = set(names) - set(images), set(images) - set(names)
        require(not missing and not extra, f"{partition}: image/label mismatch; missing={len(missing)}, extra={len(extra)}; samples={sorted(missing)[:5]}/{sorted(extra)[:5]}")
        npz_bytes = reader.read("labels.npz")
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as packed:
            require(set(packed.files) == {"filenames", "targets", "categories"}, f"{partition}: unexpected NPZ arrays")
            npz_names, npz_targets, npz_categories = packed["filenames"], packed["targets"], packed["categories"]
            require(npz_names.dtype.kind in ("U", "S") and npz_categories.dtype.kind in ("U", "S"), f"{partition}: NPZ strings must not require pickle")
            require(npz_names.tolist() == names, f"{partition}: CSV/NPZ filenames/order differ")
            require(npz_categories.tolist() == categories, f"{partition}: CSV/NPZ categories/order differ")
            require(npz_targets.dtype == np.uint8 and np.array_equal(npz_targets, targets), f"{partition}: CSV/NPZ targets differ")
        modes: dict[str, int] = {}
        min_width = min_height = None
        max_width = max_height = 0
        errors: list[dict] = []
        error_count = 0
        started = time.monotonic()

        def decode(name):
            try:
                return name, reader.decode_image(name), None
            except Exception as error:
                return name, None, type(error).__name__ + ": " + str(error)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for count, (name, decoded, error) in enumerate(pool.map(decode, names), start=1):
                if error:
                    error_count += 1
                    if len(errors) < 20:
                        errors.append({"file_name": name, "error": error})
                else:
                    mode, width, height = decoded
                    modes[mode] = modes.get(mode, 0) + 1
                    min_width = width if min_width is None else min(min_width, width)
                    min_height = height if min_height is None else min(min_height, height)
                    max_width, max_height = max(max_width, width), max(max_height, height)
                if count % 5000 == 0:
                    print(f"{partition}: decoded {count:,}/{len(names):,}", flush=True)
        require(error_count == 0, f"{partition}: {error_count} image decode failures; samples={errors}")
        return {
            "root": str(root.resolve()), "image_count": len(images), "label_count": len(names),
            "targets_shape": list(targets.shape), "targets_dtype": str(targets.dtype),
            "categories": categories, "category_support": targets.astype(np.int64).sum(axis=0).tolist(),
            "all_images_decoded": True, "decoded_count": len(names), "decode_errors": [],
            "decode_seconds": time.monotonic() - started, "original_image_modes": modes,
            "image_dimensions": {"min_width": min_width, "max_width": max_width, "min_height": min_height, "max_height": max_height},
            "csv_npz_identical": True, "image_label_names_identical": True,
            "metadata_sha256": {"categories.txt": sha256_bytes(category_bytes), "labels.csv": sha256_bytes(csv_bytes), "labels.npz": sha256_bytes(npz_bytes)},
        }, names
    finally:
        reader.close()


def read_fixed_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(names) == len(set(names)), f"Duplicate entries in fixed split: {path.name}")
    return names


def validate_fixed_splits(reference: Path, train_names: list[str], val_names: list[str]) -> dict:
    inner_path = reference / "common" / "splits" / "train_inner.txt"
    holdout_path = reference / "common" / "splits" / "holdout.txt"
    inner, holdout = read_fixed_names(inner_path), read_fixed_names(holdout_path)
    require(bool(inner) and bool(holdout), "Fixed train_inner and holdout must both be nonempty")
    require(not (set(inner) & set(holdout)), "Fixed train_inner and holdout overlap")
    expected, covered = set(train_names), set(inner) | set(holdout)
    require(covered == expected, f"Fixed split coverage mismatch: missing={len(expected-covered)}, unknown={len(covered-expected)}")
    require(not (expected & set(val_names)), "Training source and official val image names overlap")
    require(not (set(holdout) & set(val_names)), "Holdout and official val overlap")
    return {"train_inner": len(inner), "holdout": len(holdout), "train_source": len(train_names), "official_val": len(val_names),
            "disjoint": True, "full_train_coverage": True, "official_val_disjoint": True,
            "files": {"train_inner": str(inner_path.resolve()), "holdout": str(holdout_path.resolve())},
            "sha256": {"train_inner": sha256_bytes(inner_path.read_bytes()), "holdout": sha256_bytes(holdout_path.read_bytes())}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True, help="extracted train folder or original train ZIP")
    parser.add_argument("--val-dir", type=Path, required=True, help="extracted val folder or original val ZIP")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = {"passed": False, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "reference_dir": str(args.reference_dir.resolve()),
              "selected_paths": {"train": str(args.train_dir.resolve()), "val": str(args.val_dir.resolve())},
              "official_val_use": "Structural/integrity checks only; no model selection or performance evaluation.",
              "public_test_and_examples_used": False}
    started = time.monotonic()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        common = load_common_labels(args.reference_dir)
        report["categories"] = manifest["categories"]
        report["common_labels_sha256"] = sha256_bytes((args.reference_dir / "common" / "labels.py").read_bytes())
        report["partitions"] = {}
        train_report, train_names = audit_partition(args.train_dir, "train", manifest["files"]["train"], manifest["categories"], common, args.workers)
        report["partitions"]["train"] = train_report
        val_report, val_names = audit_partition(args.val_dir, "val", manifest["files"]["val"], manifest["categories"], common, args.workers)
        report["partitions"]["val"] = val_report
        report["splits"] = validate_fixed_splits(args.reference_dir, train_names, val_names)
        report["counts"] = {"train": len(train_names), "val": len(val_names),
                            "train_inner": report["splits"]["train_inner"], "holdout": report["splits"]["holdout"], "classes": 27}
        report["passed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        print(f"DATA AUDIT FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json_preserving(args.output, report)
    print("DATA AUDIT PASSED" if report["passed"] else "DATA AUDIT FAILED", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
