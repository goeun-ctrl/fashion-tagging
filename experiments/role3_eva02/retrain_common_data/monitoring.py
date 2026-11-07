"""Deterministic diagnostics only; never alter training draws or selection rules."""
import hashlib
import math

import numpy as np

from . import require

DIAGNOSTIC_SIZE = 2048
DIAGNOSTIC_SEED = 42
WARNING_TRANSITIONS = 3


def diagnostic_subset(groups, size=DIAGNOSTIC_SIZE, seed=DIAGNOSTIC_SEED):
    root, names, targets = groups["train_inner"]
    require(size > 0 and len(names) > 0 and len(names) == len(targets), "invalid diagnostic source")
    require(len(names) == len(set(names)), "duplicate diagnostic source IDs")
    held_out = set(groups["holdout"][1]) | set(groups["val"][1])
    require(not set(names) & held_out, "diagnostic source overlaps evaluation data")
    # A local generator leaves global NumPy state and published plans untouched.
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = np.sort(rng.choice(len(names), size=min(size, len(names)), replace=False))
    selected = [names[int(i)] for i in indices]
    subset = root, selected, targets[indices]
    metadata = {"source": "train_inner", "seed": seed, "requested_size": size, "size": len(selected),
                "image_ids": selected, "ids_sha256_lf": hashlib.sha256(
                    ("\n".join(selected) + "\n").encode("utf-8")).hexdigest(),
                "augmentation": False, "selection_use": False,
                "positive_counts": (targets[indices] == 1).sum(0).tolist()}
    return subset, metadata


def complete(epoch, best_epoch, cfg):
    return epoch >= cfg["max_epochs"] or epoch - best_epoch >= cfg["patience"]


def overfitting_warnings(history, transitions=WARNING_TRANSITIONS):
    """Warn on consecutive trends, without any absolute gap cutoff or stopping."""
    if len(history) < transitions + 1:
        return []
    recent = history[-transitions - 1:]
    pairs = list(zip(recent, recent[1:]))
    warnings = []
    if all(b["diagnostic_train_macro"] > a["diagnostic_train_macro"] and
           b["holdout_macro"] < a["holdout_macro"] for a, b in pairs):
        warnings.append("train_macro_up_holdout_macro_down")
    if all(b["train_holdout_macro_gap"] > a["train_holdout_macro_gap"] for a, b in pairs):
        warnings.append("train_holdout_macro_gap_widening")
    if all(b["holdout_bce"] > a["holdout_bce"] for a, b in pairs):
        warnings.append("holdout_bce_worsening")
    return warnings


def monitor_epoch(history, row, best, best_epoch, cfg):
    row = dict(row)
    epoch, macro = row["epoch"], row["holdout_macro"]
    require(epoch == len(history) + 1, "monitoring epoch/history mismatch")
    require(all(math.isfinite(row[key]) for key in
                ("holdout_macro", "holdout_bce", "diagnostic_train_macro", "diagnostic_train_bce")),
            "non-finite diagnostic or holdout metric")
    improved = macro > best  # A tie keeps the earlier checkpoint.
    if improved:
        best, best_epoch = macro, epoch
    early_stop = epoch - best_epoch >= cfg["patience"]
    row.update(best_epoch=best_epoch, best_holdout_macro=best, is_best=improved,
               epochs_without_improvement=epoch - best_epoch, early_stopping_triggered=early_stop,
               training_complete=complete(epoch, best_epoch, cfg),
               stop_reason="patience" if early_stop else "max_epochs" if epoch >= cfg["max_epochs"] else "",
               train_holdout_macro_gap=row["diagnostic_train_macro"] - macro)
    row["overfitting_warnings"] = "|".join(overfitting_warnings([*history, row]))
    return row, best, best_epoch
