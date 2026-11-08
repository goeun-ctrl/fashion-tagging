"""Paired bootstrap over images for comparing two sets of predictions on the same images."""
import numpy as np

from .metrics import compute_metrics

KEYS = (("micro_f1", "Micro F1"), ("macro_f1", "Macro F1"), ("image_f1", "Image-wise F1"))


def bootstrap_diff(y, pred_a, pred_b, n_boot, seed):
    """95% interval of metric(a) - metric(b), resampling images with replacement (same sample for a and b)."""
    rng = np.random.default_rng(seed)
    diffs = {k: [] for k, _ in KEYS}
    n = len(y)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        ma, mb = compute_metrics(y[ix], pred_a[ix]), compute_metrics(y[ix], pred_b[ix])
        for k, _ in KEYS:
            diffs[k].append(ma[k] - mb[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in diffs.items()}
