"""Tests for aggregate_results.py with synthetic experiments. Run:  python common/test_aggregate.py

Predictions here are fabricated from real val labels plus noise (NOT model output); they only
exercise the merge / verification logic.
"""
import csv
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aggregate_results import main as aggregate  # noqa: E402
from labels import load_split_from_zip  # noqa: E402
from metrics import binarize, compute_metrics, save_predictions, sigmoid  # noqa: E402

cats, names, T = load_split_from_zip(os.path.join(os.path.dirname(HERE), "MAT4091_fashion_val.zip"), "val")
HEADER = open(os.path.join(HERE, "configs", "results_template.csv"), encoding="utf-8").readline().strip().split(",")
rng = np.random.default_rng(1)


def make_run(runs, exp_id, noise):
    scores = {}
    for which, sl in (("val", slice(0, 400)), ("holdout", slice(400, 800))):
        y = T[sl]
        logits = (y * 4.0 - 2.0) + rng.normal(0, noise, size=y.shape)
        d = os.path.join(runs, exp_id)
        os.makedirs(d, exist_ok=True)
        save_predictions(os.path.join(d, f"preds_{which}.npz"), names[sl], y, logits, cats)
        m = compute_metrics(y, binarize(sigmoid(logits), 0.5))
        scores.update({f"{which}_{k}": f"{m[k]:.4f}" for k in ("micro_f1", "macro_f1", "image_f1")})
    return scores


def row(exp_id, scores, **kw):
    r = {c: "" for c in HEADER}
    r.update(exp_id=exp_id, owner="t", model="convnext_tiny", loss="bce", optimizer="adamw", seed="42", threshold="0.5",
            input_size="384", preprocess_mode="pad", params_total="27840891", **scores)
    r.update(kw)
    return r


def write(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)


with tempfile.TemporaryDirectory() as d:
    runs = os.path.join(d, "runs")
    sA, sB, sC, sD, sE = (make_run(runs, e, n) for e, n in
                          (("A_base", 2.5), ("B_better", 1.0), ("C_typo", 1.8), ("D_smoke", 2.0), ("E_big", 1.5)))
    tampered = dict(sC)
    tampered["val_macro_f1"] = "0.9999"                                   # someone mistyped the score
    write(os.path.join(d, "p1.csv"), [
        row("A_base", sA, owner="alice"),
        row("B_better", sB, owner="bob", preprocess_mode="resize"),
        row("C_typo", tampered, owner="carol"),
        row("D_smoke", sD, notes="x [no pretrained, smoke test]"),
        row("E_big", sE, params_total="61000000"),
        row("F_nopreds", sA, owner="dave", seed="7"),
    ])
    write(os.path.join(d, "p2.csv"), [row("A_base", sA, owner="alice2")])    # duplicate exp_id in another file

    out = os.path.join(d, "report")
    rows, warns = aggregate(["--csv", os.path.join(d, "p*.csv"), "--runs_dirs", runs, "--baseline", "A_base", "--out", out])
    text = "\n".join(warns)
    ids = [r["exp_id"] for r in rows]

    assert "D_smoke" not in ids and "smoke" in text                       # smoke rows excluded and reported
    assert ids.count("A_base") == 1 and "중복 exp_id 'A_base'" in text      # duplicate reported, one kept
    assert "E_big" in text and "50M 제한" in text                           # parameter limit
    assert next(r for r in rows if r["exp_id"] == "E_big")["_over"] is True
    assert not any(r["_over"] for r in rows if r["exp_id"] != "E_big")
    assert "C_typo" in text and "다시 계산" in text                          # CSV score disagrees with predictions
    c = next(r for r in rows if r["exp_id"] == "C_typo")
    assert c["val_macro_f1"] == sC["val_macro_f1"]                         # recomputed value replaces the typo
    f = next(r for r in rows if r["exp_id"] == "F_nopreds")
    assert f["_verified"] == "no" and "F_nopreds: preds_val.npz, preds_holdout.npz" in text   # one line per experiment
    assert "()" not in text                                                 # no empty date brackets
    assert "seed 값이 섞여" in text.replace("실험 간 ", "") or "seed" in text  # mixed seeds
    order = [r["exp_id"] for r in rows if r["_verified"] == "yes"]
    assert order[0] == "B_better"                                          # best holdout macro F1 first
    md = open(os.path.join(out, "results_summary.md"), encoding="utf-8").read()
    assert "E_big **(50M 초과)**" in md                                      # over-limit model is marked in the tables
    for token in ("## 확인 필요","## 1. 성능", "Δ val Macro", "## 2. 설정", "preprocess_mode", "## 3. 비용", "## 4. 클래스별 val F1", "**"):
        assert token in md, token
    for fn in ("results_summary.csv", "per_class_f1_val.csv", "per_class_f1_holdout.csv"):
        assert os.path.exists(os.path.join(out, fn)), fn
    with open(os.path.join(out, "per_class_f1_val.csv"), encoding="utf-8-sig") as fh:
        pc = list(csv.reader(fh))
    assert len(pc) == 28 and pc[0][:2] == ["class", "support"]             # 27 classes + header
    with open(os.path.join(tempfile.gettempdir(), "agg_check.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
print("all aggregate tests passed")
