"""Hand-computed checks. Run:  python common/tests/test_metrics.py"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))       # common/tests
COMMON = os.path.dirname(HERE)                          # common/
sys.path[:0] = [COMMON, os.path.join(COMMON, "tools")]
from labels import parse_categories, parse_labels_csv, to_names  # noqa: E402
from metrics import (binarize, compute_metrics, load_predictions,  # noqa: E402
                     save_predictions, sigmoid)


def close(a, b):
    assert abs(a - b) < 1e-9, (a, b)


# 1. exact prediction -> everything 1
t = np.array([[1, 0, 1], [0, 1, 0]])
m = compute_metrics(t, t)
close(m["micro_f1"], 1); close(m["macro_f1"], 1); close(m["image_f1"], 1)

# 2. partial errors (hand computed)
true = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
pred = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
m = compute_metrics(true, pred)
# class F1: 2/3, 2/3, 1 -> macro 7/9 ; TP=3 FP=0 FN=2 -> micro 6/8
close(m["macro_f1"], 7 / 9)
close(m["micro_f1"], 0.75)
# image F1: 2/3, 1, 0 -> 5/9
close(m["image_f1"], 5 / 9)
close(m["empty_pred_rate"], 1 / 3)
close(m["mean_pred_labels"], 1.0)
assert list(m["per_class"]["support"]) == [2, 2, 1]
assert list(m["per_class"]["fn"]) == [1, 1, 0]

# 3. both empty -> image F1 = 1 ; true non-empty but pred empty -> 0
m = compute_metrics(np.array([[0, 0, 0]]), np.array([[0, 0, 0]]))
close(m["image_f1"], 1)
m = compute_metrics(np.array([[1, 0, 0]]), np.array([[0, 0, 0]]))
close(m["image_f1"], 0)
m = compute_metrics(np.array([[0, 0, 0]]), np.array([[1, 0, 0]]))  # false positive on empty truth
close(m["image_f1"], 0)

# 4. support = 0 classes
true = np.array([[0, 0, 0], [1, 0, 0]])
m = compute_metrics(true, true)
close(m["image_f1"], 1)
close(m["macro_f1"], 1 / 3)                      # classes 1,2: 0 by default
close(compute_metrics(true, true, zero_division=1.0)["macro_f1"], 1.0)
m = compute_metrics(np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 0]]))
close(m["per_class"]["f1"][1], 0)                # support 0 with a false positive

# 5. sigmoid + thresholds (scalar and per class), no forced top-1
probs = np.array([[0.9, 0.4, 0.1], [0.2, 0.3, 0.4]])
assert binarize(probs, 0.5).tolist() == [[True, False, False], [False, False, False]]
assert binarize(probs, [0.5, 0.35, 0.05]).tolist() == [[True, True, True], [False, False, True]]
close(sigmoid(0.0), 0.5)

# 6. labels: commas inside names, JSON labels vs binary columns
cats = parse_categories("\n".join(f"c{i}" for i in range(25)) + "\nshirt, blouse\nbag, wallet\n")
assert cats[25] == "shirt, blouse"
header = "file_name,labels," + ",".join(f'"{c}"' if "," in c else c for c in cats)
bin_cols = ["0"] * 27
bin_cols[25] = "1"
good = header + '\na.jpg,"[""shirt, blouse""]",' + ",".join(bin_cols)
names, targets = parse_labels_csv(good, cats)
assert names == ["a.jpg"] and to_names(targets[0], cats) == ["shirt, blouse"]
bad = header + '\na.jpg,"[""bag, wallet""]",' + ",".join(bin_cols)
try:
    parse_labels_csv(bad, cats)
    raise SystemExit("mismatch not detected")
except ValueError:
    pass

# 7. prediction file round trip keeps exact names and order
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "p.npz")
    save_predictions(p, ["a.jpg", "b.jpg"], [[1, 0], [0, 1]], [[2.0, -1.0], [-3.0, 0.5]], ["x, y", "z"])
    f, y, lg, c = load_predictions(p)
    assert list(f) == ["a.jpg", "b.jpg"] and c == ["x, y", "z"] and y.tolist() == [[1, 0], [0, 1]]

print("all tests passed")
