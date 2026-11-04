"""Checks for preprocessing, split and Dataset. Run:  python common/tests/test_data.py"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))       # common/tests
COMMON = os.path.dirname(HERE)                          # common/
sys.path[:0] = [COMMON, os.path.join(COMMON, "tools")]
from dataset import FashionDataset, read_names  # noqa: E402
from labels import load_split_from_zip  # noqa: E402
from preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess  # noqa: E402

ROOT = os.path.dirname(COMMON)
M, S = np.array(IMAGENET_MEAN, dtype=np.float32), np.array(IMAGENET_STD, dtype=np.float32)
BLACK = (0 - M) / S  # normalised value of a black padding pixel

# 1. pad mode: portrait 100x200 (WxH) -> 384x384, content 192 wide, centred, black borders
img = Image.new("RGB", (100, 200), (255, 255, 255))
x = preprocess(img, 384, "pad")
assert x.shape == (3, 384, 384) and x.dtype == np.float32
assert np.allclose(x[:, 0, 0], BLACK) and np.allclose(x[:, 200, 191 - 96 - 1], BLACK)  # left padding
assert not np.allclose(x[:, 200, 192], BLACK)                                         # image content
cols = np.flatnonzero((x[0] > BLACK[0] + 1e-3).any(0))
assert cols.min() == 96 and cols.max() == 96 + 192 - 1, (cols.min(), cols.max())

# 2. resize mode fills the whole square (no black border) ; landscape pads top/bottom
x = preprocess(img, 384, "resize")
assert not np.allclose(x[:, 0, 0], BLACK)
x = preprocess(Image.new("RGB", (200, 100), (255, 255, 255)), 384, "pad")
rows = np.flatnonzero((x[0] > BLACK[0] + 1e-3).any(1))
assert rows.min() == 96 and rows.max() == 96 + 192 - 1

# 3. grayscale and RGBA/palette inputs become 3 channels; deterministic; flip mirrors
gray = Image.fromarray(np.arange(100 * 60, dtype=np.uint8).reshape(100, 60) % 255, "L")
assert preprocess(gray, 64).shape == (3, 64, 64)
assert preprocess(Image.new("RGBA", (50, 80), (10, 20, 30, 0)), 64).shape == (3, 64, 64)
a, b = preprocess(gray, 64), preprocess(gray, 64)
assert np.array_equal(a, b)
assert np.array_equal(preprocess(gray, 64, hflip=True), a[:, :, ::-1])

# 4. split: disjoint, complete, files named in labels, class coverage
cats, names, targets = load_split_from_zip(os.path.join(ROOT, "MAT4091_fashion_train.zip"), "train")
inner = read_names(os.path.join(COMMON, "splits", "train_inner.txt"))
hold = read_names(os.path.join(COMMON, "splits", "holdout.txt"))
assert not set(inner) & set(hold) and set(inner) | set(hold) == set(names)
assert len(inner) + len(hold) == len(names) == 45622 and len(set(hold)) == len(hold)
idx = {n: i for i, n in enumerate(names)}
ho_t = targets[[idx[n] for n in hold]]
assert (ho_t.sum(0) >= 1).all(), "a class is missing from the holdout"
print(f"split ok: inner={len(inner)} holdout={len(hold)} min class positives in holdout={int(ho_t.sum(0).min())}")

# 5. Dataset straight from the real val ZIP (no extraction): shape, labels, determinism
vcats, vnames, vt = load_split_from_zip(os.path.join(ROOT, "MAT4091_fashion_val.zip"), "val")
assert vcats == cats
ds = FashionDataset(os.path.join(ROOT, "MAT4091_fashion_val.zip"), vnames[:8], vt[:8], size=384, mode="pad",
                    zip_prefix="val")
x0, y0 = ds[0]
assert tuple(x0.shape) == (3, 384, 384) and tuple(y0.shape) == (27,)
assert np.array_equal(np.asarray(y0), vt[0].astype(np.float32))
assert np.array_equal(np.asarray(ds[0][0]), np.asarray(x0))  # val is deterministic
print("dataset ok (ZIP read, torch installed: %s)" % ("yes" if type(x0).__module__.startswith("torch") else "no"))
print("all data tests passed")
