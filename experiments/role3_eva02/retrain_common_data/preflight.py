"""Read-only audit. Only a complete PASS yields inputs for the trainer."""
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from . import ROLE, SPLITS, require
from . import package as pkg
from data_check import MANIFEST as FIXED, MissingLabeledImages, read_names, source
from fashion_tagging.data.preprocess import timm_mean_std

CFG = json.loads((ROLE / "config.json").read_text(encoding="utf-8"))
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
RUN_ROOT = ROLE / "private" / "retrain"
VAL_COUNT = 1158


@dataclass(frozen=True)
class Inputs:
    package: Path
    shared_lock: Path
    train_root: Path
    val_root: Path
    external_root: Path
    weights: Path


def run_directory(run_id):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id), "invalid run ID")
    root = RUN_ROOT / run_id
    require(root.resolve() == root.absolute() and not root.is_reserved(), "unsafe private output directory")
    return root


def static_config():
    require(CFG["input_size"] == 378 and CFG["model"] == "eva02_small_patch14_336.mim_in22k_ft_in1k",
            "unexpected EVA architecture")
    require((CFG["microbatch"], CFG["effective_batch"], CFG["max_epochs"], CFG["seed"], CFG["lr_backbone"],
             CFG["lr_head"], CFG["weight_decay"], CFG["warmup_ratio"], CFG["patience"], CFG["threshold"])
            == (16, 32, 30, 42, 5e-5, 1e-3, 0.05, 0.05, 5, 0.5), "baseline settings changed")
    require(timm_mean_std(CFG["model"]) == (MEAN, STD), "EVA normalization mismatch")


def read_lock(path, mode):
    lock = json.loads(path.read_text(encoding="utf-8-sig"))
    pkg.check_lock(lock, mode)
    return lock


def team_splits():
    groups = {}
    for split, (count, digest) in FIXED.items():
        path = SPLITS / (split + ".txt")
        require(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest,
                "team split hash mismatch: " + split)
        ids = read_names(path)
        require(len(set(ids)) == len(ids) == count, "team split size or uniqueness mismatch")
        groups[split] = ids
    require(not set(groups["train_inner"]) & set(groups["holdout"]), "team split overlap")
    with (SPLITS / "split_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        cats = [r["class"] for r in csv.DictReader(f)]
    require(len(cats) == len(set(cats)) == 27, "invalid team class order")
    return cats, groups


def original_splits(inputs, lock, fixed):
    cats, groups = fixed
    require(cats == lock["categories"], "release and Git class orders differ")
    result = {}
    for split, root in (("train", inputs.train_root), ("val", inputs.val_root)):
        require(pkg.sha(root / "labels.npz") == lock["sha256"][split + "/labels.npz"],
                split + " official label hash mismatch")
        _, actual_cats, names, labels = source(split, root)
        require(actual_cats == cats, "official class order mismatch")
        if split == "train":
            index = {name: i for i, name in enumerate(names)}
            require(set(index) == set(groups["train_inner"]) | set(groups["holdout"]), "original split coverage")
            for key, ids in groups.items():
                result[key] = root, ids, labels[[index[name] for name in ids]]
        else:
            require(len(names) == VAL_COUNT and not set(names) & set(index), "official val size/ID overlap")
            result["val"] = root, names, labels
    return result


def audit_images(inputs, training, groups):
    rows, targets = training
    root, inner_ids, inner_y = groups["train_inner"]
    original = dict(zip(inner_ids, inner_y))
    held_out = set(groups["holdout"][1]) | set(groups["val"][1])
    used_original, digests, paths = set(), set(), []
    for row, target in zip(rows, targets, strict=True):
        name, rel = row["image_id"], row["relative_path"]
        require(name not in held_out, "evaluation ID in manifest")
        if row["source"] == "train_inner":
            require(name == rel and name in original and np.array_equal(original[name], target),
                    "original ID, path or target differs")
            used_original.add(name)
            image_root = root / "images"
        else:
            require(name not in original, "external ID collides with original")
            image_root = inputs.external_root
        path = pkg.image_path(image_root, rel)
        digest = pkg.sha(path)
        require(digest == row["image_sha256"], "training image hash mismatch")
        require(digest not in digests, "duplicate image bytes in training candidates")
        paths.append(path)
        digests.add(digest)
    require(used_original == set(inner_ids), "manifest omits train_inner images")
    for split in ("holdout", "val"):
        root, names, _ = groups[split]
        for name in names:
            digest = pkg.sha(pkg.image_path(root / "images", name))
            require(digest not in digests, f"duplicate image bytes in/across splits: {split}")
            digests.add(digest)
    return paths


def audit_plans(root, lock, probabilities):
    plans = root / "epoch_plans"
    expected = {pkg.plan_name(i) for i in range(1, lock["epochs"] + 1)}
    found = {p.relative_to(root).as_posix() for p in plans.glob("*.npz")}
    require(not found - expected, "unexpected extra epoch plans")
    for i in range(1, lock["epochs"] + 1):
        pkg.load_plan(root, lock, i, probabilities)


def preflight(inputs, mode, run_id):
    checks, values = {}, {}

    def check(name, fn, depends=()):
        missing = [key for key in depends if checks[key]["status"] != "PASS"]
        if missing:
            checks[name] = {"status": "BLOCKED", "reason": "requires " + ", ".join(missing)}
            return
        try:
            values[name] = fn()
            checks[name] = {"status": "PASS"}
        except (FileNotFoundError, MissingLabeledImages) as exc:
            checks[name] = {"status": "BLOCKED", "reason": str(exc)}
        except (ValueError, TypeError, KeyError, IndexError, OSError, EOFError, BadZipFile, AttributeError) as exc:
            checks[name] = {"status": "FAIL", "reason": str(exc)}

    def directory(path):
        if not path.exists():
            raise FileNotFoundError(str(path))
        require(path.is_dir(), f"expected directory: {path}")

    check("eva_config", static_config)
    check("private_output", lambda: run_directory(run_id))
    check("official_backbone_sha256", lambda: require(pkg.sha(inputs.weights) == CFG["weight_sha256"],
                                                     "incorrect official backbone hash"))
    check("git_fixed_splits", team_splits)
    check("package_directory", lambda: directory(inputs.package))
    check("external_directory", lambda: directory(inputs.external_root))
    check("approved_team_lock", lambda: read_lock(inputs.shared_lock, mode))
    check("original_splits_labels", lambda: original_splits(inputs, values["approved_team_lock"],
                                                           values["git_fixed_splits"]),
          ("git_fixed_splits", "approved_team_lock"))
    check("manifest_targets_label_mode", lambda: pkg.load_training(inputs.package, values["approved_team_lock"]),
          ("package_directory", "approved_team_lock"))
    check("weights_counts_augmentation", lambda: pkg.check_policy(inputs.package, values["approved_team_lock"],
                                                                  *values["manifest_targets_label_mode"]),
          ("manifest_targets_label_mode",))
    check("image_hashes_duplicates_leakage", lambda: audit_images(inputs, values["manifest_targets_label_mode"],
                                                                values["original_splits_labels"]),
          ("manifest_targets_label_mode", "original_splits_labels", "external_directory"))
    check("epoch_plans", lambda: audit_plans(inputs.package, values["approved_team_lock"],
                                             values["weights_counts_augmentation"][0]),
          ("weights_counts_augmentation",))
    states = {c["status"] for c in checks.values()}
    status = "FAIL" if "FAIL" in states else "BLOCKED" if "BLOCKED" in states else "PASS"
    audit = {"status": status, "checks": checks, "label_mode": mode}
    if "manifest_targets_label_mode" in values:
        rows, _ = values["manifest_targets_label_mode"]
        audit["approved_external"] = sum(r["source"] == "external" for r in rows)
        audit["train_candidates"] = len(rows)
    if "weights_counts_augmentation" in values:
        _, audit["positive_counts"], audit["weight_histogram"] = values["weights_counts_augmentation"]
    if status != "PASS":
        return audit, None
    audit["split_counts"] = {s: len(g[1]) for s, g in values["original_splits_labels"].items()}
    audit["fixed_split_sha256_lf"] = {s: v[1] for s, v in FIXED.items()}
    audit["shared"] = values["approved_team_lock"]
    return audit, values
