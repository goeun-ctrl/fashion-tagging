"""Choose decision thresholds on the internal holdout and report what they do (no retraining).

    python scripts/tune_threshold.py --run runs/S1_cnxt_t_pad --out report/thr_S1
    python scripts/tune_threshold.py --holdout preds_holdout.npz --val preds_val.npz --out report/thr_S2

Steps (holdout only for every choice; the official val is only reported):
  1. global   one threshold for all classes, grid 0.05..0.95, maximising --objective on holdout
  2. per-class for classes with holdout support >= --min_support: the threshold maximising that class's F1,
              pulled toward the global value by --shrink (0 = keep global, 1 = fully per class);
              rarer classes keep the global threshold (their holdout F1 moves by a single image)
  3. check    2-fold cross-fit on holdout (choose on one half, score the other half, swap, average) to show
              whether the gain survives on images that were not used to choose it
Outputs: thresholds.json (global + per class, ready for inference), report.md, per_class.csv.
--objective: macro (default; the metric thresholds move most), micro, image, or mean (mean of the three F1).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fashion_tagging.evaluation.metrics import compute_metrics, load_predictions, sigmoid  # noqa: E402

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
KEYS = (("micro_f1", "Micro"), ("macro_f1", "Macro"), ("image_f1", "Image-wise"))


def score(m, objective):
    return (m["micro_f1"] + m["macro_f1"] + m["image_f1"]) / 3 if objective == "mean" else m[f"{objective}_f1"]


def metrics_at(y, p, thr):
    return compute_metrics(y, p >= np.asarray(thr)[None, :] if np.ndim(thr) else p >= thr)


def best_global(y, p, objective):
    scores = [score(metrics_at(y, p, t), objective) for t in GRID]
    return float(GRID[int(np.argmax(scores))]), scores


def class_f1(yc, pc, t):
    pred = pc >= t
    tp, fp, fn = (yc & pred).sum(), (~yc & pred).sum(), (yc & ~pred).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def per_class(y, p, g, min_support, shrink):
    thr = np.full(y.shape[1], g)
    tuned = np.zeros(y.shape[1], dtype=bool)
    for c in range(y.shape[1]):
        if y[:, c].sum() >= min_support:
            f = [class_f1(y[:, c], p[:, c], t) for t in GRID]
            thr[c] = g + shrink * (float(GRID[int(np.argmax(f))]) - g)
            tuned[c] = True
    return thr, tuned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run folder with preds_holdout.npz and preds_val.npz")
    ap.add_argument("--holdout")
    ap.add_argument("--val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", default="macro", choices=["macro", "micro", "image", "mean"])
    ap.add_argument("--min_support", type=int, default=50)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    hp = a.holdout or os.path.join(a.run, "preds_holdout.npz")
    vp = a.val or os.path.join(a.run, "preds_val.npz")
    _, yh, lh, cats = load_predictions(hp)
    _, yv, lv, cats_v = load_predictions(vp)
    assert cats == cats_v, "holdout and val category order differ"
    yh, yv = yh.astype(bool), yv.astype(bool)
    ph, pv = sigmoid(lh), sigmoid(lv)

    g, curve = best_global(yh, ph, a.objective)
    thr_c, tuned = per_class(yh, ph, g, a.min_support, a.shrink)
    settings = {"0.5": 0.5, f"global {g:.2f}": g, "per-class": thr_c}

    # 2-fold cross-fit: does the chosen threshold help on holdout images it was not chosen on?
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(yh))
    halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    cross = {k: [] for k in settings}
    for i in (0, 1):
        fit, ev = halves[i], halves[1 - i]
        gi, _ = best_global(yh[fit], ph[fit], a.objective)
        ti, _ = per_class(yh[fit], ph[fit], gi, a.min_support, a.shrink)
        for k, t in (("0.5", 0.5), (f"global {g:.2f}", gi), ("per-class", ti)):
            cross[k].append(metrics_at(yh[ev], ph[ev], t))

    os.makedirs(a.out, exist_ok=True)
    rows = {}
    L = [f"# 임계값 조정 ({os.path.basename(os.path.dirname(os.path.abspath(hp)))})", "",
         f"- 선택 기준: holdout {a.objective} F1, 후보 0.05~0.95 (0.01 간격)",
         f"- 공통 최적값: **{g:.2f}**", f"- 클래스별 조정: holdout 양성 {a.min_support}장 이상인 {int(tuned.sum())}개 클래스, "
         f"공통값 쪽으로 {a.shrink:g}만큼만 이동 (나머지 {int((~tuned).sum())}개는 공통값)", "",
         "| 설정 | holdout Micro | Macro | Image | holdout 반반 교차 Macro (선택에 안 쓴 절반) | **공식 val** Micro | Macro | Image |",
         "|---|---|---|---|---|---|---|---|"]
    for k, t in settings.items():
        mh, mv = metrics_at(yh, ph, t), metrics_at(yv, pv, t)
        cm = np.mean([m["macro_f1"] for m in cross[k]])
        rows[k] = (mh, mv, cm)
        L.append(f"| {k} | {mh['micro_f1']:.4f} | {mh['macro_f1']:.4f} | {mh['image_f1']:.4f} | {cm:.4f} | "
                 f"{mv['micro_f1']:.4f} | {mv['macro_f1']:.4f} | {mv['image_f1']:.4f} |")
    base_v = rows["0.5"][1]
    L += ["", "공식 val 변화 (0.5 대비): " + ", ".join(
        f"{k}: Micro {rows[k][1]['micro_f1'] - base_v['micro_f1']:+.4f} / Macro {rows[k][1]['macro_f1'] - base_v['macro_f1']:+.4f}"
        f" / Image {rows[k][1]['image_f1'] - base_v['image_f1']:+.4f}" for k in list(settings)[1:]), "",
        "## 클래스별 (공식 val, 0.5 vs 공통 최적값)", "",
        "| 클래스 | val 양성 | 임계값(클래스별) | F1 @0.5 | F1 @공통 | Δ | recall @0.5 → @공통 |", "|---|---|---|---|---|---|---|"]
    pc5, pcg = rows["0.5"][1]["per_class"], rows[f"global {g:.2f}"][1]["per_class"]
    out_rows = []
    for c, name in enumerate(cats):
        L.append(f"| {name} | {int(pc5['support'][c])} | {thr_c[c]:.2f}{'' if tuned[c] else ' (공통)'} | "
                 f"{pc5['f1'][c]:.3f} | {pcg['f1'][c]:.3f} | {pcg['f1'][c] - pc5['f1'][c]:+.3f} | "
                 f"{pc5['recall'][c]:.2f} → {pcg['recall'][c]:.2f} |")
        out_rows.append([name, int(yh[:, c].sum()), int(pc5['support'][c]), f"{thr_c[c]:.2f}", int(tuned[c]),
                         f"{pc5['f1'][c]:.4f}", f"{pcg['f1'][c]:.4f}"])
    L += ["", "읽는 법: '반반 교차' 열이 0.5보다 높으면 그 이득은 선택에 쓰지 않은 이미지에서도 유지된다는 뜻이다. "
          "공식 val은 보고용이며 임계값 선택에 쓰지 않았다."]
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"objective": a.objective, "global": g, "min_support": a.min_support, "shrink": a.shrink,
               "per_class": {c: round(float(t), 3) for c, t in zip(cats, thr_c)}},
              open(os.path.join(a.out, "thresholds.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "per_class.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows([["class", "holdout_support", "val_support", "threshold", "tuned", "val_f1_0.5",
                                   "val_f1_global"], *out_rows])
    print("\n".join(L[:11]))


if __name__ == "__main__":
    main()
