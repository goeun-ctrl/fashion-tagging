"""results.csv header upgrade. Run:  python tests/test_results_csv.py  (needs torch + timm, imports scripts/train.py)"""
import csv
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]
import train as T  # noqa: E402

old = ["exp_id", "a", "c"]
new = ["exp_id", "a", "b", "c"]
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "results.csv")
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerows([old, ["r1", "1", "3"], ["r2", "1", "2", "3"]])   # r2 was appended under the new header
    assert T.upgrade_csv(p, new)
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    assert list(rows[0]) == new
    assert rows[0] == {"exp_id": "r1", "a": "1", "b": "", "c": "3"}
    assert rows[1] == {"exp_id": "r2", "a": "1", "b": "2", "c": "3"}
    assert not T.upgrade_csv(p, new), "second call is a no-op"
print("results csv tests passed")
