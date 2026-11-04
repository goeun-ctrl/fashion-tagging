"""Compare two experiment groups (A = reference, B = change), e.g. AdamW vs Muon or linear vs query head.

Each group is one or more exp_ids (different seeds of the same setting). Uses only files that
train_eval.py already writes: runs/<exp_id>/preds_holdout.npz, preds_val.npz, log.json, and optionally
the results CSV for cost columns.

    python common/tools/compare_runs.py --a S1_cnxt_adamw_s42 S1_cnxt_adamw_s43 --b S5_cnxt_muon_s42 S5_cnxt_muon_s43 \
        --labels AdamW Muon --runs_dirs runs --csv results/results.csv --out report/adamw_vs_muon

Report (<out>/comparison.md, comparison_per_class.csv, curves.png if matplotlib is installed)
  * Micro / Macro / Image-wise F1 per run and mean +- std per group (holdout and official val)
  * paired bootstrap over images for each seed pair (a[i] vs b[i]): is the gap larger than resampling noise?
  * seed spread vs. gap: the plan only adopts the more complex option when the gap exceeds seed variation
  * convergence from log.json: best epoch, training time, and time/epochs until holdout macro F1 first
    reaches the target (default: 98% of group A's mean best holdout macro F1)
  * cost from the CSV: params, train minutes, inference ms/img, max VRAM
  * per-class F1 (group means) with support
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))       # common/tools
COMMON = os.path.dirname(HERE)                          # common/
sys.path[:0] = [COMMON, HERE]
sys.stdout.reconfigure(errors="replace")
from compare_preprocess import KEYS, bootstrap_diff  # noqa: E402
from metrics import binarize, compute_metrics, load_predictions, sigmoid  # noqa: E402

COST = (("params_total", "파라미터"), ("train_minutes", "학습 분"), ("infer_ms_per_img", "추론 ms/장"),
        ("max_vram_gb", "최대 VRAM GB"))


def find_run(exp_id, runs_dirs):
    for d in runs_dirs:
        p = os.path.join(d, exp_id)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"{exp_id}: not found in {runs_dirs}")


def load_run(exp_id, runs_dirs, threshold):
    d = find_run(exp_id, runs_dirs)
    run = {"id": exp_id, "dir": d, "eval": {}}
    for which in ("holdout", "val"):
        names, y, logits, cats = load_predictions(os.path.join(d, f"preds_{which}.npz"))
        pred = binarize(sigmoid(logits), threshold)
        run["eval"][which] = {"names": [str(n) for n in names], "y": y, "pred": pred, "m": compute_metrics(y, pred)}
        run["cats"] = cats
    log_path = os.path.join(d, "log.json")
    run["log"] = json.load(open(log_path, encoding="utf-8")) if os.path.exists(log_path) else None
    return run


def read_csv_rows(paths):
    rows = {}
    for p in paths or []:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows[(r.get("exp_id") or "").strip()] = r
    return rows


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else float("nan")


def fmt_ms(m, s):
    return f"{m:.4f}" if np.isnan(s) else f"{m:.4f} ± {s:.4f}"


def convergence(run, target):
    """(best epoch, epochs run, train minutes, epoch reaching target, minutes until then) from log.json."""
    log = run["log"]
    if not log:
        return None
    ep = log["log"]
    cum = np.cumsum([e["sec"] for e in ep]) / 60
    hit = next((i for i, e in enumerate(ep) if e["holdout_macro"] >= target), None)
    return (log["best_epoch"], len(ep), float(cum[-1]),
            None if hit is None else ep[hit]["epoch"], None if hit is None else float(cum[hit]))


def plot_curves(groups, labels, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g, lab, color in zip(groups, labels, ("#1f77b4", "#d62728")):
        for i, r in enumerate(g):
            if not r["log"]:
                continue
            ep = r["log"]["log"]
            x_ep = [e["epoch"] for e in ep]
            x_min = np.cumsum([e["sec"] for e in ep]) / 60
            y = [e["holdout_macro"] for e in ep]
            kw = dict(color=color, alpha=0.9 if i == 0 else 0.45, label=lab if i == 0 else None)
            axes[0].plot(x_ep, y, marker="o", ms=3, **kw)
            axes[1].plot(x_min, y, marker="o", ms=3, **kw)
    axes[0].set_xlabel("epoch")
    axes[1].set_xlabel("training minutes")
    for ax in axes:
        ax.set_ylabel("holdout macro F1")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=120)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True, help="reference exp_ids (one per seed)")
    ap.add_argument("--b", nargs="+", required=True, help="changed exp_ids, same seed order as --a")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--runs_dirs", nargs="+", default=["runs"])
    ap.add_argument("--csv", nargs="*", default=[], help="results CSVs for cost columns (optional)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--target", type=float, default=0.0, help="holdout macro F1 target for convergence "
                    "(default: 0.98 x group A mean best)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    la, lb = a.labels
    A = [load_run(e, a.runs_dirs, a.threshold) for e in a.a]
    B = [load_run(e, a.runs_dirs, a.threshold) for e in a.b]
    for r in A + B:
        for which in ("holdout", "val"):
            ref = A[0]["eval"][which]
            cur = r["eval"][which]
            assert cur["names"] == ref["names"] and np.array_equal(cur["y"], ref["y"]), \
                f"{r['id']}: evaluated on different {which} images than {A[0]['id']}"
    csv_rows = read_csv_rows(a.csv)

    L = [f"# 실험 비교: {la} vs {lb}", "",
         f"- {la}: {', '.join(a.a)}", f"- {lb}: {', '.join(a.b)}",
         f"- 임계값 {a.threshold}, 부트스트랩 {a.boot}회(이미지 단위, 같은 seed끼리 짝지어 비교).",
         "- 체크포인트는 holdout으로 골랐다. 공식 val은 보고용이다.", ""]
    if len(A) == 1 or len(B) == 1:
        L += ["**주의: seed가 1개뿐인 그룹이 있어 seed 변동을 알 수 없다. 차이가 작으면 seed를 늘려 다시 비교한다.**", ""]

    # ---- scores --------------------------------------------------------------------------------
    for which, title in (("holdout", "내부 holdout"), ("val", "공식 val")):
        n_img = len(A[0]["eval"][which]["names"])
        L += [f"## {title} ({n_img:,}장)", "", "| 실험 | 그룹 | " + " | ".join(lab for _, lab in KEYS) + " |",
              "|---|---|" + "---|" * len(KEYS)]
        for g, lab in ((A, la), (B, lb)):
            for r in g:
                m = r["eval"][which]["m"]
                L.append(f"| {r['id']} | {lab} | " + " | ".join(f"{m[k]:.4f}" for k, _ in KEYS) + " |")
        L += ["", f"| 지표 | {la} 평균 | {lb} 평균 | {lb}-{la} | seed 표준편차(큰 쪽) | 판단 |", "|---|---|---|---|---|---|"]
        for k, lab in KEYS:
            ma, sa = mean_std([r["eval"][which]["m"][k] for r in A])
            mb, sb = mean_std([r["eval"][which]["m"][k] for r in B])
            spread = np.nanmax([sa, sb]) if not (np.isnan(sa) and np.isnan(sb)) else float("nan")
            if np.isnan(spread):
                judge = "seed 1개: 판단 보류"
            elif abs(mb - ma) <= spread:
                judge = "seed 변동 이내"
            else:
                judge = f"{lb} 우세" if mb > ma else f"{la} 우세"
            L.append(f"| {lab} | {fmt_ms(ma, sa)} | {fmt_ms(mb, sb)} | {mb - ma:+.4f} | "
                     f"{'-' if np.isnan(spread) else f'{spread:.4f}'} | {judge} |")
        L += ["", f"짝별 부트스트랩 95% 구간 ({lb}-{la}):", "", "| 짝 | " + " | ".join(lab for _, lab in KEYS) + " |",
              "|---|" + "---|" * len(KEYS)]
        for ra, rb in zip(A, B):
            ea, eb = ra["eval"][which], rb["eval"][which]
            ci = bootstrap_diff(ea["y"], eb["pred"], ea["pred"], a.boot, a.seed)   # b - a
            cells = []
            for k, _ in KEYS:
                d = eb["m"][k] - ea["m"][k]
                lo, hi = ci[k]
                cells.append(f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("" if lo <= 0 <= hi else " *"))
            L.append(f"| {ra['id']} vs {rb['id']} | " + " | ".join(cells) + " |")
        L += ["", "`*` = 구간이 0을 포함하지 않음. 그래도 seed 1쌍의 결과이므로 seed 변동과 함께 읽는다.", ""]

    # ---- convergence ---------------------------------------------------------------------------
    best_a = [max(e["holdout_macro"] for e in r["log"]["log"]) for r in A if r["log"]]
    target = a.target or (0.98 * float(np.mean(best_a)) if best_a else 0.0)
    L += ["## 수렴 (log.json)", "", f"목표: holdout macro F1 ≥ {target:.4f}"
          + ("" if a.target else f" ({la} 평균 최고값의 98%)"), "",
          "| 실험 | 그룹 | best epoch | 돈 epoch | 학습 분(합) | 목표 도달 epoch | 목표 도달 분 |", "|---|---|---|---|---|---|---|"]
    for g, lab in ((A, la), (B, lb)):
        for r in g:
            c = convergence(r, target)
            if c is None:
                L.append(f"| {r['id']} | {lab} | log.json 없음 | | | | |")
                continue
            be, n, mins, hit_ep, hit_min = c
            L.append(f"| {r['id']} | {lab} | {be} | {n} | {mins:.1f} | {'미도달' if hit_ep is None else hit_ep} | "
                     f"{'-' if hit_min is None else f'{hit_min:.1f}'} |")
    L += ["", "학습 분은 epoch별 학습+holdout 평가 시간의 합이다(최종 val 예측 제외). 같은 GPU 종류에서 잰 값끼리만 비교한다.", ""]

    # ---- cost ----------------------------------------------------------------------------------
    if csv_rows:
        L += ["## 비용 (results CSV)", "", "| 실험 | 그룹 | " + " | ".join(lab for _, lab in COST) + " | optimizer_args |",
              "|---|---|" + "---|" * (len(COST) + 1)]
        for g, lab in ((A, la), (B, lb)):
            for r in g:
                row = csv_rows.get(r["id"], {})
                L.append(f"| {r['id']} | {lab} | " + " | ".join(row.get(k, "") or "-" for k, _ in COST)
                         + f" | {row.get('optimizer_args', '') or '-'} |")
        L.append("")

    # ---- per class -----------------------------------------------------------------------------
    cats = A[0]["cats"]
    pc = {}
    for which in ("holdout", "val"):
        fa = np.mean([r["eval"][which]["m"]["per_class"]["f1"] for r in A], axis=0)
        fb = np.mean([r["eval"][which]["m"]["per_class"]["f1"] for r in B], axis=0)
        pc[which] = (A[0]["eval"][which]["m"]["per_class"]["support"], fa, fb)
    L += ["## 클래스별 F1 (그룹 평균)", "",
          f"| 클래스 | holdout support | {la} | {lb} | Δ | val support | {la} | {lb} | Δ |", "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for i, c in enumerate(cats):
        sh, fah, fbh = (x[i] for x in pc["holdout"])
        sv, fav, fbv = (x[i] for x in pc["val"])
        rows.append([c, int(sh), fah, fbh, fbh - fah, int(sv), fav, fbv, fbv - fav])
        L.append(f"| {c} | {int(sh)} | {fah:.3f} | {fbh:.3f} | {fbh - fah:+.3f} | {int(sv)} | {fav:.3f} | {fbv:.3f} | {fbv - fav:+.3f} |")
    L += ["", "val 양성이 30장 미만인 클래스는 한두 장으로 F1이 크게 바뀐다."]

    os.makedirs(a.out, exist_ok=True)
    if plot_curves((A, B), (la, lb), a.out):
        L += ["", "![holdout macro F1 curves](curves.png)"]
    with open(os.path.join(a.out, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.out, "comparison_per_class.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "holdout_support", f"holdout_f1_{la}", f"holdout_f1_{lb}", "holdout_delta",
                    "val_support", f"val_f1_{la}", f"val_f1_{lb}", "val_delta"])
        w.writerows([[r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:+.4f}", r[5], f"{r[6]:.4f}", f"{r[7]:.4f}",
                      f"{r[8]:+.4f}"] for r in rows])
    print(f"saved to {a.out}")


if __name__ == "__main__":
    main()
