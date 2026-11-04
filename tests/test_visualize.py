"""Tests for the error-visualisation tool. Run:  python tests/test_visualize.py

Uses synthetic logits (NOT model output) on real val images, only to check the tool.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]
from fashion_tagging.data.labels import load_split_from_zip  # noqa: E402
from fashion_tagging.evaluation.metrics import binarize, save_predictions, sigmoid  # noqa: E402
from visualize_errors import select  # noqa: E402

# 1. selection rules on a tiny hand-made case
names = ["a", "b", "c", "d"]
y_true = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
probs = np.array([[0.9, 0.2, 0.1],    # a: misses class1 (p .2)           -> F1 2/3
                  [0.1, 0.3, 0.8],    # b: misses class0, extra class2    -> F1 0
                  [0.1, 0.9, 0.1],    # c: exact
                  [0.1, 0.1, 0.7]])   # d: extra class2 on empty truth    -> F1 0
pred = binarize(probs, 0.5)
idx, _, n = select("worst", y_true, probs, pred, names)
assert [names[i] for i in idx] == ["b", "d", "a"] and n == 3      # F1 0 first; b has 2 errors > d has 1
idx, _, n = select("fn", y_true, probs, pred, names, cls=1)
assert [names[i] for i in idx] == ["a"] and n == 1
idx, _, n = select("fp", y_true, probs, pred, names, cls=2)
assert [names[i] for i in idx] == ["b", "d"] and n == 2             # highest prob first (.8 then .7)
idx, _, n = select("ok", y_true, probs, pred, names)
assert [names[i] for i in idx] == ["c"]
r1 = select("random", y_true, probs, pred, names, k=2, seed=7)[0]
assert r1 == select("random", y_true, probs, pred, names, k=2, seed=7)[0]   # seeded -> repeatable
assert set(r1) <= {0, 1, 3}                                                  # only images with errors

# 2. end-to-end CLI on real val ZIP images with synthetic logits
root = ROOT
vz = os.path.join(root, "MAT4091_fashion_val.zip")
cats, vnames, vt = load_split_from_zip(vz, "val")
n = 60
rng = np.random.default_rng(0)
logits = (vt[:n] * 4.0 - 2.0) + rng.normal(0, 2.5, size=(n, 27))
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "fake.npz")
    save_predictions(p, vnames[:n], vt[:n], logits, cats)
    cmd = [sys.executable, "-I", os.path.join(ROOT, "scripts", "visualize_errors.py"), "--pred", p, "--images", vz,
           "--zip_prefix", "val", "--out", d]
    for extra in (["--mode", "worst", "--k", "8"], ["--mode", "fn", "--class", "shoe", "--k", "4"],
                  ["--mode", "fp", "--class", "shirt, blouse", "--k", "4"], ["--mode", "random", "--k", "5"]):
        subprocess.run(cmd + extra, check=True, capture_output=True)
    for f in ("worst.jpg", "fn_23.jpg", "fp_0.jpg", "random.jpg"):
        with Image.open(os.path.join(d, f)) as im:
            assert im.width > 1000 and im.height > 400, (f, im.size)
    with open(os.path.join(d, "worst.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    assert rec["rule"] and len(rec["items"]) == 8
    assert all(x["gt"] != x["pred"] for x in rec["items"])          # every shown 'worst' item really has an error
    bad = subprocess.run(cmd + ["--mode", "fn", "--class", "nope"], capture_output=True)
    assert bad.returncode != 0                                       # unknown class name is rejected
    with Image.open(os.path.join(d, "worst.jpg")) as im:
        im.save(os.path.join(tempfile.gettempdir(), "vis_check_worst.jpg"))
print("all visualize tests passed")
