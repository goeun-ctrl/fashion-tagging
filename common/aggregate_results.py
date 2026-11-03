"""Merge the team's results CSVs into one comparison report.

    python common/aggregate_results.py --csv results/*.csv --runs_dirs runs server2/runs \
        --baseline S1_cnxt_t_pad --out report/

Inputs
  --csv        one or more results CSVs (rows written by train_eval.py or by hand, columns as in
               configs/results_template.csv)
  --runs_dirs  folders holding <exp_id>/preds_val.npz and preds_holdout.npz. When the predictions are
               found the three F1 scores are RECOMPUTED and compared with the CSV; a mismatch is reported.

Checks reported at the top of the report (nothing is silently dropped)
  duplicate exp_id (last row kept), smoke-test rows (excluded unless --include_smoke), parameters above
  50M, CSV score different from the score recomputed from predictions, missing predictions, different
  thresholds / image counts between experiments, mixed seeds.

Ranking is by the internal HOLDOUT (the set used for selection); the official val is shown for reporting.
Writes results_summary.md, results_summary.csv, per_class_f1_val.csv, per_class_f1_holdout.csv.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(errors="replace")
from metrics import binarize, compute_metrics, load_predictions, sigmoid  # noqa: E402

PARAM_LIMIT = 50_000_000
TOL = 6e-4  # CSV scores are rounded to 4 decimals
M3 = ("micro_f1", "macro_f1", "image_f1")
SETTING_COLS = ["model", "loss", "optimizer", "pooling", "head", "scheduler", "optimizer_args", "layer_decay",
                "drop_path", "lr_backbone", "lr_head", "weight_decay", "epochs_run",
                "effective_batch", "seed", "input_size", "preprocess_mode", "threshold"]
# CSVs written before these columns existed came from the fixed baseline code
LEGACY_DEFAULTS = {"pooling": "gap", "head": "linear", "scheduler": "cosine", "optimizer_args": "",
                   "layer_decay": "0.0", "drop_path": "0.0"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("exp_id") or "").strip():
                    r = {k: (v or "").strip() for k, v in r.items() if k}
                    for k, v in LEGACY_DEFAULTS.items():
                        r.setdefault(k, v)
                    r["_source"] = os.path.basename(p)
                    rows.append(r)
    return rows


def find_preds(exp_id, runs_dirs, which):
    for d in runs_dirs:
        p = os.path.join(d, exp_id, f"preds_{which}.npz")
        if os.path.exists(p):
            return p
    return None


def evaluate(path, thr):
    names, y, logits, cats = load_predictions(path)
    m = compute_metrics(y, binarize(sigmoid(logits), thr))
    return m, cats, len(names)


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="results CSV files (globs allowed)")
    ap.add_argument("--runs_dirs", nargs="*", default=["runs"])
    ap.add_argument("--baseline", default=None, help="exp_id used for the delta columns")
    ap.add_argument("--sort_by", default="holdout_macro_f1")
    ap.add_argument("--rare_max_support", type=int, default=30, help="val classes with fewer positives count as rare")
    ap.add_argument("--include_smoke", action="store_true")
    ap.add_argument("--out", default="report")
    a = ap.parse_args(argv)

    paths = [p for pat in a.csv for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit("no CSV matched --csv")
    rows = read_rows(paths)
    warns = []

    # ---- smoke rows, duplicates -------------------------------------------------------
    kept = []
    for r in rows:
        if "smoke" in r.get("notes", "").lower() and not a.include_smoke:
            warns.append(f"제외(smoke test): {r['exp_id']} ({r['_source']})")
        else:
            kept.append(r)
    by_id = {}
    for r in kept:
        if r["exp_id"] in by_id:
            old = by_id[r["exp_id"]]
            def where(x):
                return x["_source"] + (f" ({x['date']})" if x.get("date") else "")
            warns.append(f"중복 exp_id '{r['exp_id']}': {where(old)} 와 {where(r)} "
                         f"-> 마지막 행만 사용. 정말 다른 실험이면 exp_id를 바꾸세요.")
        by_id[r["exp_id"]] = r
    rows = list(by_id.values())
    if not rows:
        raise SystemExit("no experiments left after filtering")

    # ---- per experiment: predictions, recomputed scores ------------------------------
    per_class = {"val": {}, "holdout": {}}
    support_ref, cats_ref, n_images = {}, None, {}
    for r in rows:
        thr = num(r.get("threshold")) or 0.5
        r["_verified"] = "no"
        n_params = num(r.get("params_total"))
        r["_over"] = n_params is not None and n_params > PARAM_LIMIT
        if r["_over"]:
            warns.append(f"{r['exp_id']}: 파라미터 {int(n_params):,}개가 50M 제한을 넘음 -> 제출 후보가 될 수 없음")
        if n_params is None:
            warns.append(f"{r['exp_id']}: params_total 비어 있음")
        ok, missing = True, []
        for which in ("val", "holdout"):
            p = find_preds(r["exp_id"], a.runs_dirs, which)
            if p is None:
                missing.append(f"preds_{which}.npz")
                ok = False
                continue
            m, cats, n = evaluate(p, thr)
            if cats_ref is None:
                cats_ref = cats
            elif cats != cats_ref:
                warns.append(f"{r['exp_id']}: 클래스 목록/순서가 다른 예측 파일 ({which})")
            n_images.setdefault(which, {})[r["exp_id"]] = n
            if which == "val" and "val" not in support_ref:
                support_ref["val"] = m["per_class"]["support"]
            for k in M3:
                csv_v = num(r.get(f"{which}_{k}"))
                if csv_v is not None and abs(csv_v - m[k]) > TOL:
                    warns.append(f"{r['exp_id']}: CSV {which}_{k}={csv_v:.4f} 인데 예측 파일로 다시 계산하면 {m[k]:.4f} "
                                 f"(재계산 값을 사용)")
                r[f"{which}_{k}"] = f"{m[k]:.4f}"
            r[f"_{which}_pred_labels"] = m["mean_pred_labels"]
            r[f"_{which}_empty"] = m["empty_pred_rate"]
            per_class[which][r["exp_id"]] = m["per_class"]
        if missing:
            warns.append(f"{r['exp_id']}: {', '.join(missing)} 를 --runs_dirs 에서 못 찾음 -> CSV 값을 검증 없이 사용")
        if ok:
            r["_verified"] = "yes"
    for which, d in n_images.items():
        if len(set(d.values())) > 1:
            warns.append(f"{which} 이미지 수가 실험마다 다름: {d} -> 같은 데이터로 평가한 결과가 아니면 비교할 수 없음")
    for key in ("threshold", "seed"):
        vals = {r.get(key) for r in rows if r.get(key)}
        if len(vals) > 1:
            warns.append(f"실험 간 {key} 값이 섞여 있음: {sorted(vals)}")

    # rare-class macro F1 (val support below the limit)
    rare_idx = None
    if "val" in support_ref:
        rare_idx = np.flatnonzero(support_ref["val"] < a.rare_max_support)
    for r in rows:
        for which in ("val", "holdout"):
            pc = per_class[which].get(r["exp_id"])
            r[f"_rare_{which}"] = float(pc["f1"][rare_idx].mean()) if (pc is not None and rare_idx is not None and len(rare_idx)) else None

    # ---- ranking and baseline deltas ---------------------------------------------------
    key = a.sort_by
    rows.sort(key=lambda r: (num(r.get(key)) is None, -(num(r.get(key)) or 0)))
    base = next((r for r in rows if r["exp_id"] == a.baseline), None)
    if a.baseline and base is None:
        warns.append(f"--baseline '{a.baseline}' 를 찾지 못함 -> delta 열 생략")

    def lab(r):
        return r["exp_id"] + (" **(50M 초과)**" if r.get("_over") else "")

    # ---- markdown --------------------------------------------------------------------
    L = ["# 통합 결과표", "",
         f"- 입력 CSV {len(paths)}개, 실험 {len(rows)}개. 정렬 기준 `{key}` (내부 holdout = 모델 선택에 쓴 데이터).",
         "- 공식 val은 최종 보고용 열이다. 같은 val로 여러 모델을 고르면 점수가 낙관적으로 보일 수 있다.",
         "- `검증` 열이 yes이면 예측 파일로 세 F1을 다시 계산해 CSV와 맞춘 값이다. no이면 CSV 값을 그대로 옮긴 것이다.",
         "- 대부분 seed 1개의 단일 학습 결과이므로 작은 차이는 우연일 수 있다.", ""]
    if warns:
        L += ["## 확인 필요", ""] + [f"- {w}" for w in warns] + [""]
    head = ["실험", "담당", "모델", "holdout Micro", "holdout Macro", "holdout Image", "val Micro", "val Macro", "val Image"]
    if base:
        head += ["Δ val Macro", "Δ holdout Macro"]
    head += [f"희귀클래스 val Macro(<{a.rare_max_support}장)", "검증"]
    L += ["## 1. 성능", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [lab(r), r.get("owner", ""), r.get("model", ""),
                 f4(num(r.get("holdout_micro_f1"))), f4(num(r.get("holdout_macro_f1"))), f4(num(r.get("holdout_image_f1"))),
                 f4(num(r.get("val_micro_f1"))), f4(num(r.get("val_macro_f1"))), f4(num(r.get("val_image_f1")))]
        if base:
            for k in ("val_macro_f1", "holdout_macro_f1"):
                v, b = num(r.get(k)), num(base.get(k))
                cells.append("-" if v is None or b is None else f"{v - b:+.4f}")
        cells += [f4(r["_rare_val"]), r["_verified"]]
        L.append("| " + " | ".join(cells) + " |")
    if base:
        L += ["", f"Δ 는 baseline `{a.baseline}` 대비 차이."]

    L += ["", "## 2. 설정 (실험마다 다른 항목을 확인하세요)", "", "| 실험 | " + " | ".join(SETTING_COLS) + " |",
          "|" + "---|" * (len(SETTING_COLS) + 1)]
    for r in rows:
        L.append("| " + " | ".join([lab(r)] + [r.get(c, "") for c in SETTING_COLS]) + " |")
    varying = [c for c in SETTING_COLS if len({r.get(c, "") for r in rows}) > 1]
    L += ["", "실험 간에 값이 다른 설정: " + (", ".join(varying) if varying else "없음") +
          ". 한 요인만 달라야 그 요인의 효과로 해석할 수 있다."]

    L += ["", "## 3. 비용", "", "| 실험 | 파라미터(M) | 50M 대비 | 학습(분) | 추론(ms/장) | VRAM(GB) | 실행 epoch | val 평균 예측 라벨 | val 빈 예측 비율 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = num(r.get("params_total"))
        L.append("| " + " | ".join([lab(r), "-" if p is None else f"{p / 1e6:.2f}", "-" if p is None else f"{100 * p / PARAM_LIMIT:.1f}%",
                                    r.get("train_minutes", "") or "-", r.get("infer_ms_per_img", "") or "-", r.get("max_vram_gb", "") or "-",
                                    r.get("epochs_run", "") or "-",
                                    "-" if "_val_pred_labels" not in r else f"{r['_val_pred_labels']:.2f}",
                                    "-" if "_val_empty" not in r else f"{r['_val_empty']:.3f}"]) + " |")

    ids_v = [r["exp_id"] for r in rows if r["exp_id"] in per_class["val"]]
    if ids_v and cats_ref:
        L += ["", "## 4. 클래스별 val F1 (굵게 = 해당 클래스 최고)", "",
              "| 클래스 | val support | " + " | ".join(ids_v) + " |", "|---|---|" + "---|" * len(ids_v)]
        for c, name in enumerate(cats_ref):
            vals = [per_class["val"][i]["f1"][c] for i in ids_v]
            best = max(vals)
            sup = int(per_class["val"][ids_v[0]]["support"][c])
            cells = [("**%.3f**" % v) if (v == best and sup > 0 and len({round(x, 6) for x in vals}) > 1) else "%.3f" % v for v in vals]
            L.append(f"| {name} | {sup} | " + " | ".join(cells) + " |")
        L += ["", "val 양성이 30장 미만인 클래스는 한두 장으로 F1이 크게 바뀌므로 최고 표시를 순위로 읽지 마세요."]

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    cols = ["exp_id", "owner"] + SETTING_COLS + ["params_total"] + [f"{w}_{k}" for w in ("holdout", "val") for k in M3] + \
           ["train_minutes", "infer_ms_per_img", "max_vram_gb", "notes", "_source", "_verified"]
    with open(os.path.join(a.out, "results_summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for which in ("val", "holdout"):
        ids = [r["exp_id"] for r in rows if r["exp_id"] in per_class[which]]
        if not ids:
            continue
        with open(os.path.join(a.out, f"per_class_f1_{which}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "support"] + ids)
            for c, name in enumerate(cats_ref):
                w.writerow([name, int(per_class[which][ids[0]]["support"][c])] + [f"{per_class[which][i]['f1'][c]:.4f}" for i in ids])
    print(f"{len(rows)} experiments, {len(warns)} warnings -> {a.out}")
    for w_ in warns:
        print(" !", w_)
    return rows, warns


if __name__ == "__main__":
    main()
