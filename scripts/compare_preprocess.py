"""Compare "keep aspect ratio + black padding" with "force resize to a square".

Runs scripts/train.py twice with everything identical (model, seed, data split, epochs, lr,
batch size) except --preprocess_mode, then compares the saved predictions:
  * Micro / Macro / Image-wise F1 on the internal holdout and on the official val
  * paired bootstrap over images: is the difference larger than resampling noise?
  * per-class F1 for both modes

    python scripts/compare_preprocess.py --prefix S4_cnxt_t --train_dir data/train --val_dir data/val \
        --out_dir runs --results_csv results/results.csv
Extra arguments (--model, --epochs, --size, --batch_size, ...) are passed through to scripts/train.py.
Use --skip_train to only redo the comparison of runs that already exist.
Result: <out_dir>/<prefix>_compare/comparison.md and comparison_per_class.csv
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # repo/src
sys.stdout.reconfigure(errors="replace")  # Windows consoles (cp949) must not crash on a special character
from fashion_tagging.evaluation.bootstrap import KEYS, bootstrap_diff  # noqa: E402
from fashion_tagging.evaluation.metrics import binarize, compute_metrics, load_predictions, sigmoid  # noqa: E402
from fashion_tagging.paths import TRAIN_SCRIPT  # noqa: E402

MODES = ("pad", "resize")


def run_pair(prefix, args, rest):
    for mode in MODES:
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--exp_id", f"{prefix}_{mode}",
               "--preprocess_mode", mode, "--seed", str(args.seed), "--threshold", str(args.threshold),
               "--out_dir", args.out_dir, "--results_csv", args.results_csv] + rest
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def load_probs(out_dir, prefix, mode, which):
    names, y, logits, cats = load_predictions(os.path.join(out_dir, f"{prefix}_{mode}", f"preds_{which}.npz"))
    return [str(n) for n in names], y, sigmoid(logits), cats


def verdict(delta, ci):
    lo, hi = ci
    if lo <= 0 <= hi:
        return "구분 어려움"
    return "pad 우세" if delta > 0 else "resize 우세"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--results_csv", default=os.path.join("results", "results.csv"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--skip_train", action="store_true")
    args, rest = ap.parse_known_args()
    if not args.skip_train:
        run_pair(args.prefix, args, rest)

    lines = [f"# 전처리 비교: pad(종횡비 유지 + 검정 padding) vs resize(강제 정사각형)  ({args.prefix})", "",
             f"- 두 실험은 모델·seed({args.seed})·split·학습 설정이 같고 preprocess_mode만 다르다.",
             f"- 임계값 {args.threshold}, 부트스트랩 {args.boot}회(이미지 단위, 같은 표본으로 두 모드 비교).",
             "- **학습 1회(seed 1개) 결과이므로 seed 변동은 반영되지 않았다.** 구간이 0을 포함하면 차이를 믿기 어렵다.", ""]
    per_class = {}
    for which, title in (("holdout", "내부 holdout (체크포인트 선택에 사용)"), ("val", "공식 val (최종 보고용)")):
        na, ya, pa, cats = load_probs(args.out_dir, args.prefix, "pad", which)
        nb, yb, pb, _ = load_probs(args.out_dir, args.prefix, "resize", which)
        assert na == nb and np.array_equal(ya, yb), "the two runs were evaluated on different images"
        bp_a, bp_b = binarize(pa, args.threshold), binarize(pb, args.threshold)
        ma, mb = compute_metrics(ya, bp_a), compute_metrics(yb, bp_b)
        ci = bootstrap_diff(ya, bp_a, bp_b, args.boot, args.seed)
        lines += [f"## {title}, {len(na):,}장", "", "| 지표 | pad | resize | pad-resize | 95% 구간 | 판단 |", "|---|---|---|---|---|---|"]
        for k, label in KEYS:
            d = ma[k] - mb[k]
            lines.append(f"| {label} | {ma[k]:.4f} | {mb[k]:.4f} | {d:+.4f} | [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}] | {verdict(d, ci[k])} |")
        lines += ["", f"예측 라벨 수 평균: pad {ma['mean_pred_labels']:.2f}, resize {mb['mean_pred_labels']:.2f} · "
                  f"빈 예측 비율: pad {ma['empty_pred_rate']:.3f}, resize {mb['empty_pred_rate']:.3f}", ""]
        per_class[which] = (cats, ma["per_class"], mb["per_class"])

    cats, pa_h, pb_h = per_class["holdout"]
    _, pa_v, pb_v = per_class["val"]
    lines += ["## 클래스별 F1", "", "| 클래스 | holdout support | holdout pad | holdout resize | Δ | val support | val pad | val resize | Δ |",
              "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for i, c in enumerate(cats):
        r = [c, int(pa_h["support"][i]), pa_h["f1"][i], pb_h["f1"][i], pa_h["f1"][i] - pb_h["f1"][i],
             int(pa_v["support"][i]), pa_v["f1"][i], pb_v["f1"][i], pa_v["f1"][i] - pb_v["f1"][i]]
        rows.append(r)
        lines.append(f"| {c} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:+.3f} | {r[5]} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:+.3f} |")
    lines += ["", "해석 시 주의: val 양성이 30장 미만인 클래스는 한두 장으로 F1이 크게 변한다. 이 비교는 seed 1개 기준이며, "
              "차이가 작으면 여러 seed로 반복해 확인해야 한다."]

    out = os.path.join(args.out_dir, f"{args.prefix}_compare")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "comparison_per_class.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "holdout_support", "holdout_f1_pad", "holdout_f1_resize", "holdout_delta",
                    "val_support", "val_f1_pad", "val_f1_resize", "val_delta"])
        w.writerows([[r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:+.4f}", r[5], f"{r[6]:.4f}", f"{r[7]:.4f}", f"{r[8]:+.4f}"] for r in rows])
    print("\n".join(lines[:22]))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
