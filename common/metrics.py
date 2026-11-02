"""Shared evaluation code (numpy only).

Definitions
-----------
All F1 values use F1 = 2*TP / (2*TP + FP + FN).

* micro F1  : TP/FP/FN summed over every image and class, then F1.
* macro F1  : mean of the per-class F1 over ALL classes in the label matrix
              (27 for this task), each class weighted equally.
* image F1  : F1 of each image's predicted set vs. true set, averaged over
              images. If both sets are empty the image scores 1 (assignment rule).
              If only one of them is empty the image scores 0.

Zero-division rules
-------------------
* Class with 2TP+FP+FN == 0 (no positives and never predicted): per-class F1
  is `zero_division` (default 0.0, so a class absent from the data lowers
  macro F1). It is reported with support 0. The provided train/val sets have
  support > 0 for every class, so this only matters for other data.
* Class with support 0 and at least one false positive: F1 is 0 by the formula.
* Micro F1 with no positives and no predictions at all: `zero_division`.
* precision with no predictions / recall with no positives: `zero_division`.

Scores are turned into labels per class with sigmoid + threshold. There is no
softmax/argmax and no forced top-1 prediction, so an empty prediction is allowed.
"""
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def binarize(probs, threshold=0.5):
    """probs [N, C] -> bool [N, C]. threshold: scalar or per-class array [C]."""
    return np.asarray(probs) >= np.asarray(threshold)


def _f1(tp, fp, fn, zero_division):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), zero_division)


def compute_metrics(y_true, y_pred, zero_division=0.0):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    tp = (y_true & y_pred).sum(0).astype(np.float64)
    fp = (~y_true & y_pred).sum(0).astype(np.float64)
    fn = (y_true & ~y_pred).sum(0).astype(np.float64)
    support = y_true.sum(0)

    pred_cnt = tp + fp
    precision = np.where(pred_cnt > 0, tp / np.maximum(pred_cnt, 1), zero_division)
    recall = np.where(support > 0, tp / np.maximum(support, 1), zero_division)
    class_f1 = _f1(tp, fp, fn, zero_division)

    micro = float(_f1(tp.sum(), fp.sum(), fn.sum(), zero_division))
    macro = float(class_f1.mean())

    itp = (y_true & y_pred).sum(1).astype(np.float64)
    ifp = (~y_true & y_pred).sum(1).astype(np.float64)
    ifn = (y_true & ~y_pred).sum(1).astype(np.float64)
    idenom = 2 * itp + ifp + ifn
    image_f1 = np.where(idenom > 0, 2 * itp / np.maximum(idenom, 1), 1.0)  # both empty -> 1

    return {
        "micro_f1": micro,
        "macro_f1": macro,
        "image_f1": float(image_f1.mean()),
        "mean_pred_labels": float(y_pred.sum(1).mean()),
        "empty_pred_rate": float((y_pred.sum(1) == 0).mean()),
        "n_images": int(y_true.shape[0]),
        "per_class": {
            "support": support.astype(int),
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
        },
    }


def evaluate_probs(y_true, probs, threshold=0.5, zero_division=0.0):
    return compute_metrics(y_true, binarize(probs, threshold), zero_division)


def format_report(metrics, categories=None):
    pc = metrics["per_class"]
    lines = [
        f"images={metrics['n_images']}  micro={metrics['micro_f1']:.4f}  "
        f"macro={metrics['macro_f1']:.4f}  image={metrics['image_f1']:.4f}  "
        f"mean_pred={metrics['mean_pred_labels']:.2f}  empty_pred={metrics['empty_pred_rate']:.3f}",
        f"{'class':42s}{'support':>8s}{'TP':>7s}{'FP':>7s}{'FN':>7s}{'prec':>8s}{'rec':>8s}{'f1':>8s}",
    ]
    for i in range(len(pc["support"])):
        name = categories[i] if categories else str(i)
        lines.append(
            f"{name:42s}{pc['support'][i]:8d}{pc['tp'][i]:7d}{pc['fp'][i]:7d}{pc['fn'][i]:7d}"
            f"{pc['precision'][i]:8.3f}{pc['recall'][i]:8.3f}{pc['f1'][i]:8.3f}"
        )
    return "\n".join(lines)


def save_predictions(path, filenames, y_true, logits, categories):
    """Everything needed to re-evaluate later without retraining."""
    np.savez_compressed(
        path,
        filenames=np.asarray(filenames),
        targets=np.asarray(y_true, dtype=np.uint8),
        logits=np.asarray(logits, dtype=np.float32),
        categories=np.asarray(categories),
    )


def load_predictions(path):
    d = np.load(path, allow_pickle=False)
    return d["filenames"], d["targets"], d["logits"], list(d["categories"])
