"""Where does one model fail?  Performance diagnosis from saved predictions (no retraining).

    python scripts/diagnose.py --run runs/R5_cnxt_hr_s42 --out report/diag_hr
    python scripts/diagnose.py --val preds_val.npz --images data/val --out report/diag_hr \
        --thresholds_json report/thr_hr/thresholds.json            # per-class thresholds from tune_threshold.py
    python scripts/diagnose.py --val preds_val.npz --images MAT4091_fashion_val.zip --zip_prefix val --out report/d

Report (<out>/diagnosis.md and CSVs):
  1. per-class precision / recall / F1 / support, and which side limits each class
     (recall-limited = it misses the class; precision-limited = it reports the class when absent)
  2. the same by group: small accessories / similar upper-body garments / rare classes / the rest
  3. error pairs: images where class A was missed AND class B was reported without being present
  4. F1 by number of true labels, over- vs under-prediction, and (with --images) by photo orientation / size

Groups are our working hypotheses, not facts about the data. Error pairs show errors that happen together
in the same photo; they do not prove the model confused A with B (a photo can contain several people).
"""
import argparse
import csv
import io
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.stdout.reconfigure(errors="replace")
from fashion_tagging.evaluation.metrics import binarize, compute_metrics, load_predictions, sigmoid  # noqa: E402

SMALL = ["glasses", "watch", "glove", "sock", "belt", "tie", "scarf", "headband, head covering, hair accessory"]
TOPS = ["shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "coat"]
LABEL_BUCKETS = [("1~2개", 1, 2), ("3~4개", 3, 4), ("5개 이상", 5, 99)]


def side(p, r):
    """Which side limits the class: the lower of precision and recall (gap under 0.05 = balanced)."""
    if max(p, r) < 0.05:
        return "둘 다 낮음"
    if abs(p - r) < 0.05:
        return "균형"
    return "놓침(recall)" if r < p else "과다(precision)"


def group_of(name, rare):
    if name in rare:
        return "희귀"
    if name in SMALL:
        return "작은 액세서리"
    if name in TOPS:
        return "비슷한 상의"
    return "그 외"


def image_sizes(root, names, zip_prefix):
    sizes = {}
    from PIL import Image
    zf = zipfile.ZipFile(root) if str(root).lower().endswith(".zip") else None
    for n in names:
        src = io.BytesIO(zf.read(f"{zip_prefix}/images/{n}")) if zf else os.path.join(root, "images", n)
        with Image.open(src) as im:
            sizes[n] = im.size  # (width, height), no pixel decoding
    return sizes


def image_f1(y, pred):
    tp = (y & pred).sum(1).astype(float)
    den = 2 * tp + (~y & pred).sum(1) + (y & ~pred).sum(1)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run folder containing preds_val.npz")
    ap.add_argument("--val", help="preds_val.npz (overrides --run)")
    ap.add_argument("--which", default="val", choices=["val", "holdout"], help="which preds file inside --run")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--thresholds_json", help="thresholds.json from tune_threshold.py (global + per class)")
    ap.add_argument("--images", help="folder with images/ or the ZIP, only for orientation / size analysis")
    ap.add_argument("--zip_prefix", default=None)
    ap.add_argument("--rare_max_support", type=int, default=30)
    ap.add_argument("--top_pairs", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    path = a.val or os.path.join(a.run, f"preds_{a.which}.npz")
    names, y, logits, cats = load_predictions(path)
    names, y = [str(n) for n in names], y.astype(bool)
    probs = sigmoid(logits)
    if a.thresholds_json:
        tj = json.load(open(a.thresholds_json, encoding="utf-8"))
        thr = np.array([tj["per_class"][c] for c in cats])
        thr_text = f"클래스별 임계값({a.thresholds_json}, 공통 {tj['global']})"
    else:
        thr, thr_text = a.threshold, f"공통 임계값 {a.threshold}"
    pred = binarize(probs, thr)
    m = compute_metrics(y, pred)
    pc = m["per_class"]
    rare = {c for c, s in zip(cats, pc["support"]) if s < a.rare_max_support}
    n = len(y)

    L = [f"# 성능 진단 ({os.path.basename(os.path.dirname(os.path.abspath(path))) or path})", "",
         f"- 파일 `{os.path.basename(path)}`, 이미지 {n:,}장, {thr_text}",
         f"- Micro {m['micro_f1']:.4f} / Macro {m['macro_f1']:.4f} / Image-wise {m['image_f1']:.4f}, "
         f"이미지당 예측 라벨 {m['mean_pred_labels']:.2f}개(정답 평균 {y.sum(1).mean():.2f}개), 빈 예측 {m['empty_pred_rate']:.3f}", ""]

    # 1. per-class
    rows = []
    for c, name in enumerate(cats):
        p, r, f = pc["precision"][c], pc["recall"][c], pc["f1"][c]
        rows.append([name, group_of(name, rare), int(pc["support"][c]), int(pc["tp"][c]), int(pc["fp"][c]), int(pc["fn"][c]),
                     round(float(p), 4), round(float(r), 4), round(float(f), 4), side(p, r)])
    L += ["## 1. 클래스별 진단 (F1 낮은 순)", "",
          "| 클래스 | 그룹 | support | TP | FP | FN | Precision | Recall | F1 | 한계 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r[8]):
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.3f} | {r[9]} |")
    L += ["", f"support {a.rare_max_support}장 미만(희귀)인 클래스는 한두 장으로 점수가 크게 바뀌므로 순위를 그대로 믿지 마세요.", ""]

    # 2. groups
    L += ["## 2. 그룹별 평균 (Macro 기준)", "", "| 그룹 | 클래스 수 | 평균 F1 | 평균 Precision | 평균 Recall | FN 합 | FP 합 |", "|---|---|---|---|---|---|---|"]
    for g in ("작은 액세서리", "비슷한 상의", "희귀", "그 외"):
        sel = [r for r in rows if r[1] == g]
        if sel:
            L.append(f"| {g} | {len(sel)} | {np.mean([r[8] for r in sel]):.3f} | {np.mean([r[6] for r in sel]):.3f} | "
                     f"{np.mean([r[7] for r in sel]):.3f} | {sum(r[5] for r in sel)} | {sum(r[4] for r in sel)} |")
    L += ["", "그룹 정의(우리의 가설, 클래스 이름 안에 쉼표가 있어 ` · `로 구분):", "",
          "- 작은 액세서리: " + " · ".join(SMALL), "- 비슷한 상의: " + " · ".join(TOPS),
          f"- 희귀: val support {a.rare_max_support}장 미만. 여러 그룹에 걸리면 희귀 > 작은 액세서리 > 상의 순으로 한 곳에만 센다.", ""]

    # 3. error pairs
    fn = y & ~pred
    fp = ~y & pred
    pairs = []
    for ai in range(len(cats)):
        nfn = int(fn[:, ai].sum())
        if nfn == 0:
            continue
        for bi in range(len(cats)):
            if bi != ai:
                k = int((fn[:, ai] & fp[:, bi]).sum())
                if k:
                    pairs.append((k, ai, bi, nfn))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    L += [f"## 3. 같이 나는 오류 짝 (A를 놓친 사진에서 B를 정답이 아닌데 낸 횟수, 상위 {a.top_pairs}개)", "",
          "| 놓친 클래스 A | 잘못 낸 클래스 B | 사진 수 | A의 FN 중 비율 |", "|---|---|---|---|"]
    for k, ai, bi, nfn in pairs[:a.top_pairs]:
        L.append(f"| {cats[ai]} | {cats[bi]} | {k} | {100 * k / nfn:.0f}% ({k}/{nfn}) |")
    tops_idx = [cats.index(t) for t in TOPS if t in cats]
    swap = [(cats[ai], int((fn[:, ai] & fp[:, tops_idx].any(1)).sum()), int(fn[:, ai].sum())) for ai in tops_idx if fn[:, ai].sum()]
    if swap:
        L += ["", "상의 클래스를 놓친 사진 중, 같은 사진에서 **다른 상의 클래스를 잘못 낸** 비율:", ""] + \
             [f"- {nm}: {k}/{t} ({100 * k / t:.0f}%)" for nm, k, t in swap]
    L += ["", "주의: 한 사진에 사람이 여러 명일 수 있어, 위는 '같은 사진에서 함께 난 오류'일 뿐 A와 B를 헷갈렸다는 증거가 아닙니다.", ""]

    # 4. image level
    ifs = image_f1(y, pred)
    gt_n, pr_n = y.sum(1), pred.sum(1)
    L += ["## 4. 사진 단위 진단", "", "### 정답 라벨 수별", "", "| 정답 라벨 수 | 사진 수 | Image-wise F1 | 정답 평균 | 예측 평균 | 예측-정답 |", "|---|---|---|---|---|---|"]
    lab_rows = []
    for nm, lo, hi in LABEL_BUCKETS:
        sel = (gt_n >= lo) & (gt_n <= hi)
        if sel.any():
            lab_rows.append([nm, int(sel.sum()), round(float(ifs[sel].mean()), 4), round(float(gt_n[sel].mean()), 2),
                             round(float(pr_n[sel].mean()), 2), round(float((pr_n[sel] - gt_n[sel]).mean()), 2)])
            L.append(f"| {nm} | {sel.sum()} | {ifs[sel].mean():.3f} | {gt_n[sel].mean():.2f} | {pr_n[sel].mean():.2f} | {(pr_n[sel] - gt_n[sel]).mean():+.2f} |")
    L += ["", "예측-정답이 음수면 라벨을 적게 내는(놓치는) 쪽, 양수면 많이 내는 쪽이다.", ""]

    orient_rows = []
    if a.images:
        sizes = image_sizes(a.images, names, a.zip_prefix)
        w = np.array([sizes[nm][0] for nm in names])
        h = np.array([sizes[nm][1] for nm in names])
        short = np.minimum(w, h)
        groups = [("세로(높이>너비)", h > w), ("가로(너비>높이)", w > h), ("정사각형", w == h),
                  ("짧은 변 < 600px", short < 600), ("짧은 변 600~720px", (short >= 600) & (short <= 720)), ("짧은 변 > 720px", short > 720)]
        L += ["### 사진 방향·크기별", "", "| 구간 | 사진 수 | Image-wise F1 | 놓친 라벨 비율 | 잘못 낸 라벨 비율 |", "|---|---|---|---|---|"]
        for nm, sel in groups:
            if sel.any():
                tot_gt, tot_pr = max(1, y[sel].sum()), max(1, pred[sel].sum())
                orient_rows.append([nm, int(sel.sum()), round(float(ifs[sel].mean()), 4),
                                    round(float(fn[sel].sum() / tot_gt), 4), round(float(fp[sel].sum() / tot_pr), 4)])
                L.append(f"| {nm} | {sel.sum()} | {ifs[sel].mean():.3f} | {100 * fn[sel].sum() / tot_gt:.1f}% | {100 * fp[sel].sum() / tot_pr:.1f}% |")
        L += ["", "놓친 라벨 비율 = FN/정답 라벨 수, 잘못 낸 라벨 비율 = FP/예측 라벨 수. 방향·크기는 서로 겹치는 구간이라 합이 사진 수와 다르다.", ""]

    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "diagnosis.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    def dump(fn_, header, data):
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows([header, *data])
    dump("per_class_diagnosis.csv", ["class", "group", "support", "tp", "fp", "fn", "precision", "recall", "f1", "limit"], rows)
    dump("error_pairs.csv", ["missed_A", "false_B", "images", "A_fn_total"], [[cats[ai], cats[bi], k, nfn] for k, ai, bi, nfn in pairs])
    dump("by_label_count.csv", ["bucket", "images", "image_f1", "gt_mean", "pred_mean", "pred_minus_gt"], lab_rows)
    if orient_rows:
        dump("by_orientation.csv", ["bucket", "images", "image_f1", "missed_label_rate", "false_label_rate"], orient_rows)
    print("\n".join(L[:5]))
    print(f"saved to {a.out}")
    return {"rows": rows, "pairs": pairs, "labels": lab_rows, "orient": orient_rows, "metrics": m}


if __name__ == "__main__":
    main()
