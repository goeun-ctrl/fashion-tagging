"""Tests for scripts/diagnose.py. Run:  python tests/test_diagnose.py

Predictions are fabricated from the real val labels with errors planted on purpose (NOT model output),
so the expected numbers are known in advance.
"""
import csv
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fashion_tagging.data.labels import load_split_from_zip  # noqa: E402
from fashion_tagging.evaluation.metrics import save_predictions  # noqa: E402

spec = importlib.util.spec_from_file_location("diagnose", os.path.join(ROOT, "scripts", "diagnose.py"))
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)

VAL_ZIP = os.path.join(ROOT, "MAT4091_fashion_val.zip")
cats, names, Y = load_split_from_zip(VAL_ZIP, "val")
Y = Y.astype(bool)
N = len(names)
ci = {c: i for i, c in enumerate(cats)}

# a perfect model, then plant two known failures
P = np.where(Y, 0.9, 0.1)
card = Y[:, ci["cardigan"]] & ~Y[:, ci["jacket"]]                  # cardigan photos without a jacket
P[card, ci["cardigan"]] = 0.1                                       # miss the cardigan ...
P[card, ci["jacket"]] = 0.9                                         # ... and report a jacket instead
watch_idx = np.flatnonzero(Y[:, ci["watch"]])[:30]
P[watch_idx, ci["watch"]] = 0.1                                     # 30 missed watches, nothing else wrong
logits = np.log(P / (1 - P))

with tempfile.TemporaryDirectory() as d:
    npz = os.path.join(d, "preds_val.npz")
    save_predictions(npz, names, Y.astype(np.uint8), logits, cats)
    out = os.path.join(d, "out")
    r = diagnose.main(["--val", npz, "--out", out, "--images", VAL_ZIP, "--zip_prefix", "val"])

    n_card = int(card.sum())
    assert n_card >= 5, "need cardigan photos for this test"
    # 1. the planted pair is the top error pair
    k, ai, bi, nfn = r["pairs"][0]
    assert (cats[ai], cats[bi], k, nfn) == ("cardigan", "jacket", n_card, n_card), r["pairs"][:3]
    # 2. watch: precision 1, recall = 52/82, recall-limited
    w = next(x for x in r["rows"] if x[0] == "watch")
    assert w[2] == 82 and w[5] == 30 and w[4] == 0 and w[9] == "놓침(recall)", w
    assert abs(w[7] - 52 / 82) < 1e-3 and w[6] == 1.0
    # 3. jacket: false positives come only from the planted cardigan photos, so precision-limited
    j = next(x for x in r["rows"] if x[0] == "jacket")
    assert j[4] == n_card and j[9] == "과다(precision)", j
    # 4. groups assigned as documented
    groups = {x[0]: x[1] for x in r["rows"]}
    assert groups["watch"] in ("작은 액세서리", "희귀") and groups["shoe"] == "그 외" and groups["coat"] in ("비슷한 상의", "희귀")
    # 5. image-level tables cover every image exactly once
    assert sum(x[1] for x in r["labels"]) == N
    assert sum(x[1] for x in r["orient"][:3]) == N                    # portrait + landscape + square
    # 6. files written, markdown mentions the planted pair
    for f in ("diagnosis.md", "per_class_diagnosis.csv", "error_pairs.csv", "by_label_count.csv", "by_orientation.csv"):
        assert os.path.exists(os.path.join(out, f)), f
    md = open(os.path.join(out, "diagnosis.md"), encoding="utf-8").read()
    assert "cardigan" in md and "## 3." in md and "## 4." in md
    with open(os.path.join(out, "per_class_diagnosis.csv"), encoding="utf-8-sig") as fh:
        assert len(list(csv.reader(fh))) == 28

    # 7. per-class thresholds file gives the same result when every threshold is 0.5
    tj = os.path.join(d, "thresholds.json")
    json.dump({"global": 0.5, "per_class": {c: 0.5 for c in cats}}, open(tj, "w", encoding="utf-8"))
    r2 = diagnose.main(["--val", npz, "--out", os.path.join(d, "out2"), "--thresholds_json", tj])
    assert abs(r2["metrics"]["macro_f1"] - r["metrics"]["macro_f1"]) < 1e-12
    assert r2["orient"] == []                                         # no --images -> no orientation table
    # 7b. both precision and recall near zero is reported as such, not as "balanced"
    cd = next(x for x in r["rows"] if x[0] == "cardigan")
    assert cd[6] == 0.0 and cd[7] == 0.0 and cd[9] == "둘 다 낮음", cd
    assert "headband, head covering, hair accessory" in md and " · " in md      # group lists use a safe separator
    # 8. a stricter threshold removes the planted false jacket reports (0.9 < 0.95)
    r3 = diagnose.main(["--val", npz, "--out", os.path.join(d, "out3"), "--threshold", "0.95"])
    assert next(x for x in r3["rows"] if x[0] == "jacket")[4] == 0
print("all diagnose tests passed")
