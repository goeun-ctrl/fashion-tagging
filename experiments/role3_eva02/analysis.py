"""Export only aggregates to the report folder. Individual predictions are private."""
import csv
import json
import os
from pathlib import Path

import numpy as np

from data_check import HERE, require, write_json
from fashion_tagging.evaluation.metrics import binarize, evaluate_probs, sigmoid

METRICS = ("micro_f1", "macro_f1", "image_f1")
REF = json.loads((HERE / "convnext_reference.json").read_text(encoding="utf-8"))


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def export_split(split, ds, cats, logits, probs, measured, bce, timing):
    dest = HERE / f"private/baseline/preds_{split}.npz"
    np.savez_compressed(dest, filenames=ds.names, categories=cats, targets=ds.targets.astype(np.uint8),
                        logits=logits, probabilities=probs)
    with np.load(dest, allow_pickle=False) as z:
        require(list(z["filenames"]) == ds.names and list(z["categories"]) == cats, "prediction order changed")
        require(np.array_equal(sigmoid(z["logits"]), z["probabilities"]) and
                np.array_equal(z["targets"], ds.targets), "saved probability/target mismatch")
        pred = binarize(z["probabilities"], 0.5)
        score = evaluate_probs(z["targets"], z["probabilities"], 0.5)
        tp_by_image = (pred & z["targets"].astype(bool)).sum(1)
        true_counts = z["targets"].sum(1).astype(int)
        pred_counts = pred.sum(1)
        # Class-pair errors describe label co-occurrence, not localized object confusion.
        misses = z["targets"].astype(bool) & ~pred
        extra = ~z["targets"].astype(bool) & pred
        pairs = misses.astype(int).T @ extra.astype(int)
    require(all(measured[k] == score[k] for k in METRICS), "recomputed F1 differs")
    require(all(np.array_equal(measured["per_class"][k], v) for k, v in score["per_class"].items()),
            "per-class roundtrip differs")
    cls = score.pop("per_class")
    table(HERE / f"class_metrics_{split}.csv",
          [{"class": cat, **{k: v[i].item() for k, v in cls.items()}} for i, cat in enumerate(cats)])
    multi = true_counts >= 2
    i, j = np.unravel_index(pairs.argmax(), pairs.shape)
    write_json(HERE / f"metrics_{split}.json", {**score, **timing, "bce": bce, "threshold": 0.5,
               "saved_predictions_verified": True, "predicted_count_histogram_0_to_27": np.bincount(pred_counts, minlength=28).tolist(),
               "multilabel_images": int(multi.sum()), "multilabel_underpredicted_count": int((multi & (pred_counts < true_counts)).sum()),
               "multilabel_with_missed_label": int((multi & (tp_by_image < true_counts)).sum()),
               "most_common_simultaneous_fn_fp": {"fn": cats[i], "fp": cats[j], "images": int(pairs[i, j])}})


def reference_classes():
    full = HERE / "private/convnext_val.csv"
    if full.exists():
        with full.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        require(len(rows) == 27 and len({r["class"] for r in rows}) == 27, "baseline CSV needs 27 unique rows")
        counts = {r["class"]: tuple(int(r[k]) for k in ("tp", "fp", "fn")) for r in rows}
    else:
        counts = REF["val_counts_tp_fp_fn"]
    result = {}
    for cat, (tp, fp, fn) in counts.items():
        result[cat] = {"support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1),
                       "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
    for cat, rounded in REF["val_f1_rounded_only"].items():
        if cat not in result:
            result[cat] = {"f1": rounded}
    return result


def compare(history):
    summaries = []
    results = {}
    for split in ("holdout", "val"):
        results[split] = json.loads((HERE / f"metrics_{split}.json").read_text(encoding="utf-8"))
        for key in METRICS:
            a, b = results[split][key], REF[split][key]
            summaries.append({"split": split, "metric": key, "ConvNeXt_report": b, "EVA02": a, "delta": a-b})
    table(HERE / "comparison_summary.csv", summaries)
    with (HERE / "class_metrics_val.csv").open(encoding="utf-8-sig", newline="") as f:
        by_class = list(csv.DictReader(f))
    ref = reference_classes()
    out = []
    for r in by_class:
        cat = r["class"]
        b = ref.get(cat, {})
        require("support" not in b or int(r["support"]) == b["support"], "baseline/EVA support mismatch: " + cat)
        d = {"class": cat, "support": int(r["support"])}
        for k in ("precision", "recall", "f1", "tp", "fp", "fn"):
            a = float(r[k]) if k in ("precision", "recall", "f1") else int(r[k])
            d["EVA_" + k], d["ConvNeXt_" + k] = a, b.get(k)
            d["delta_" + k] = a - b[k] if k in b else None
        d["macro_contribution"] = d["delta_f1"] / 27 if "f1" in b else None
        d["baseline_precision_source"] = "counts" if "tp" in b else ("rounded F1 only" if b else "unavailable")
        out.append(d)
    if all(r["baseline_precision_source"] == "counts" for r in out):
        require(abs(sum(r["macro_contribution"] for r in out) - summaries[4]["delta"]) <= 0.00000051,
                "mean class delta does not match reported Macro delta")
    table(HERE / "class_f1_comparison.csv", out)
    curves(history, out)
    from reports import render_reports
    render_reports(history, results, summaries, out)


def curves(history, comparisons):
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / "private/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for metric, label in (("train_bce", "Train"), ("holdout_bce", "Holdout")):
        axes[0].plot(epochs, [r[metric] for r in history], marker=".", label=label)
    axes[0].legend()
    axes[0].set(xlabel="Epoch", ylabel="BCE")
    axes[1].plot(epochs, [r["holdout_macro"] for r in history], marker=".", color="#00685d")
    axes[1].set(xlabel="Epoch", ylabel="Holdout Macro F1")
    fig.savefig(HERE / "learning_curves.png", dpi=155)
    plt.close(fig)
    for metrics, filename in ((["f1"], "class_f1_comparison.png"),
                              (["precision", "recall"], "class_precision_recall_comparison.png")):
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 + 3 * (len(metrics) - 1), 9), squeeze=False,
                                 sharey=True, layout="constrained")
        y = np.arange(len(comparisons))
        for ax, metric in zip(axes[0], metrics):
            ax.scatter([r["EVA_" + metric] for r in comparisons], y + 0.12, s=21, color="#00796b", label="EVA02")
            values = [np.nan if r["ConvNeXt_" + metric] is None else r["ConvNeXt_" + metric] for r in comparisons]
            ax.scatter(values, y - 0.12, s=21, marker="x", color="#ac480e", label="ConvNeXt (reported)")
            ax.set(xlim=(-.03, 1.03), xlabel=metric, yticks=y, yticklabels=[r["class"] for r in comparisons])
            ax.grid(axis="x", alpha=.2)
            ax.legend(loc="lower left", fontsize=8)
        axes[0, 0].invert_yaxis()
        fig.savefig(HERE / filename, dpi=150)
        plt.close(fig)

