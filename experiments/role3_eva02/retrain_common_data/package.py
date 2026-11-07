"""Reader for the proposed team interchange format. No plan or random draws are generated here."""
import csv
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import __version__ as pillow_version

from . import require

MANIFEST = "training_manifest.csv"
TARGETS = "training_targets.npz"
POLICY = "augmentation_policy.json"
COUNTS = "class_counts.csv"
HEADBAND = "headband, head covering, hair accessory"
GENERAL = {HEADBAND, "skirt", "coat"}
MEDIUM = {HEADBAND, "sweater", "sock", "scarf", "watch", "glove"}
LIMITS = np.array([[0.90, 1.10], [0.90, 1.10], [0.95, 1.05]])
DRAWS = 43197
EPOCHS = 30


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checked_bytes(path, digest):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, f"SHA-256 mismatch: {path.name}")
    return raw


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))


def image_path(root, name):
    # Canonical relative POSIX paths inside the manifest on either OS.
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name
            and not PureWindowsPath(name).anchor and all(p not in {"", ".", ".."} for p in name.split("/")),
            "invalid relative image path")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "image path escapes its root")
    return path


def check_labels(y, mode):
    require(mode in {"complete", "partial"}, "unsupported label mode")
    allowed = (0, 1) if mode == "complete" else (-1, 0, 1)
    require(y.ndim == 2 and y.shape[1] == 27 and y.dtype.kind in "biuf"
            and np.isin(y, allowed).all(), f"invalid 27-class labels for {mode} mode")


def plan_name(epoch):
    return f"epoch_plans/epoch_{epoch:02d}.npz"


def check_lock(lock, mode):
    require(lock["schema"] == "fashion-common-v1", "unsupported interchange schema")
    require(isinstance(lock["data_version"], str) and lock["data_version"].strip(), "missing data version")
    require(mode in {"complete", "partial"} and lock["label_mode"] == mode and lock["approved"] is True,
            "unapproved release or label mode mismatch")
    require(all(type(lock[k]) is int for k in ("draws", "epochs", "seed")) and lock["replacement"] is True
            and (lock["draws"], lock["epochs"], lock["seed"]) == (DRAWS, EPOCHS, 42), "unexpected sampling budget")
    files = {MANIFEST, TARGETS, POLICY, COUNTS, "train/labels.npz", "val/labels.npz"}
    files.update(plan_name(i) for i in range(1, EPOCHS + 1))
    hashes = lock["sha256"]
    require(set(hashes) == files and all(isinstance(h, str) and len(h) == 64 and
            all(c in "0123456789abcdef" for c in h) for h in hashes.values()), "invalid lock file hashes")
    cats = lock["categories"]
    require(isinstance(cats, list) and all(isinstance(c, str) and c.strip() for c in cats)
            and len(cats) == len(set(cats)) == 27, "invalid locked class order")


def load_training(root, lock):
    hashes = lock["sha256"]
    rows = csv_rows(checked_bytes(root / MANIFEST, hashes[MANIFEST]))
    ids = [r["image_id"] for r in rows]
    require(rows and all(ids) and len(set(ids)) == len(ids), "empty or duplicate manifest IDs")
    require(all(r["data_version"] == lock["data_version"] for r in rows), "manifest version mismatch")
    require(all(r["source"] in {"train_inner", "external"} and r["approved"] == "1" for r in rows),
            "manifest contains an unapproved row or forbidden split")
    require(any(r["source"] == "external" for r in rows), "no approved external images")
    with np.load(io.BytesIO(checked_bytes(root / TARGETS, hashes[TARGETS])), allow_pickle=False) as z:
        require(z["image_ids"].tolist() == ids, "target IDs/order differ from manifest")
        require(z["categories"].tolist() == lock["categories"], "target class order mismatch")
        require(z["data_version"].item() == lock["data_version"], "targets version mismatch")
        y = z["targets"]
    check_labels(y, lock["label_mode"])
    require(len(y) == len(rows), "manifest/target row count mismatch")
    return rows, y


def policy_vectors(cats, positives, focus, general, medium):
    focus, general, medium = set(focus), set(general), set(medium)
    require((focus | general | medium) <= set(cats) and not focus & general, "invalid policy groups")
    require(GENERAL <= (general | focus) and MEDIUM <= medium, "required policy classes missing")
    weights = np.ones(len(cats))
    weights[np.isin(cats, list(medium))] = 1.30
    weights[np.asarray(positives) < 1000] = 1.75
    weights[cats.index("leg warmer")] = 2.5
    jitter = np.zeros(len(cats))
    jitter[np.isin(cats, list(general))] = 0.20
    jitter[np.isin(cats, list(focus))] = 0.35
    return weights, jitter


def max_for_image(y, class_values, default):
    return np.maximum(np.where(y == 1, class_values, default).max(axis=1), default)


def check_policy(root, lock, rows, y):
    hashes, cats = lock["sha256"], lock["categories"]
    policy = json.loads(checked_bytes(root / POLICY, hashes[POLICY]))
    require(set(policy) == {"data_version", "targets_sha256", "replay", "pillow_version", "flip_p",
                            "hue", "ranges", "jitter_p", "focus", "general", "medium_weight"},
            "unsupported policy fields (verify the team adapter)")
    require(policy["replay"] == "pillow-enhance-v1" and policy["pillow_version"] == pillow_version,
            "unverified jitter implementation or Pillow version")
    require(policy["data_version"] == lock["data_version"] and policy["targets_sha256"] == hashes[TARGETS],
            "policy was prepared for different labels")
    require(policy["flip_p"] == 0.5 and policy["hue"] == 0 and policy["ranges"] == LIMITS.tolist()
            and policy["jitter_p"] == [0.35, 0.2, 0.0], "augmentation settings mismatch")
    counts = csv_rows(checked_bytes(root / COUNTS, hashes[COUNTS]))
    positives = (y == 1).sum(axis=0)
    require([r["class"] for r in counts] == cats and [int(r["positive"]) for r in counts] ==
            positives.tolist(), "class counts/order mismatch")
    w, p = policy_vectors(cats, positives, policy["focus"], policy["general"], policy["medium_weight"])
    weights = max_for_image(y, w, 1.)
    require(np.array_equal(weights, [float(r["sampling_weight"]) for r in rows]), "sampling weight mismatch")
    return max_for_image(y, p, 0.), dict(zip(cats, positives.tolist())), {
        str(weight): int(np.sum(weights == weight)) for weight in (1., 1.30, 1.75, 2.5)}


def load_plan(root, lock, epoch, probabilities):
    name = plan_name(epoch)
    hashes = lock["sha256"]
    with np.load(io.BytesIO(checked_bytes(root / name, hashes[name])), allow_pickle=False) as z:
        expected = {"epoch": epoch, "data_version": lock["data_version"], "label_mode": lock["label_mode"],
                    "manifest_sha256": hashes[MANIFEST], "targets_sha256": hashes[TARGETS],
                    "policy_sha256": hashes[POLICY]}
        require(all(z[k].item() == v for k, v in expected.items()), f"stale plan metadata: {name}")
        idx, flip, factors, order = (z[k] for k in ("indices", "flip", "factors", "order"))
    n = lock["draws"]
    require(idx.shape == (n,) and idx.dtype.kind in "iu" and (idx >= 0).all() and
            (idx < len(probabilities)).all(), f"invalid plan indices: {name}")
    require(flip.shape == (n,) and flip.dtype.kind in "biu" and np.isin(flip, (0, 1)).all(),
            f"invalid flip flags: {name}")
    # Float32 package endpoints can be one ULP away from the decimal bounds.
    require(factors.shape == (n, 3) and factors.dtype.kind in "f" and
            (factors >= LIMITS[:, 0].astype(factors.dtype)).all() and
            (factors <= LIMITS[:, 1].astype(factors.dtype)).all(), f"invalid jitter coefficients: {name}")
    require(np.all(factors[probabilities[idx] == 0] == 1), f"jitter on a maintenance-only image: {name}")
    require(order.shape == (n, 3) and order.dtype.kind in "iu" and
            np.all(np.sort(order, axis=1) == [0, 1, 2]), f"invalid jitter operation order: {name}")
    return idx, flip, factors, order
