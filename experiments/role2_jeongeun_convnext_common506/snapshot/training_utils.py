"""Small dependency-free guards shared by the trainer and its tests."""
import math


def accumulation_window_samples(n_samples, batch_size, accumulation, batch_index):
    """Actual sample count of this optimizer window, including an uneven final one."""
    if min(n_samples, batch_size, accumulation) < 1 or batch_index < 0:
        raise ValueError("positive sample/batch/accumulation counts are required")
    start_sample = (batch_index // accumulation) * accumulation * batch_size
    if start_sample >= n_samples:
        raise ValueError("batch index is outside the dataset")
    return min(batch_size * accumulation, n_samples - start_sample)


def lr_multiplier(update_index, total_updates, warmup_updates):
    """Multiplier for the next optimizer update (zero-based).

    The first `warmup_updates` updates rise linearly to the base LR. Remaining
    updates follow cosine decay to zero. Checkpointing stores LambdaLR's index.
    """
    if total_updates < 1 or not 0 <= warmup_updates <= total_updates:
        raise ValueError("invalid update counts")
    if update_index < 0:
        raise ValueError("update index must be nonnegative")
    if update_index >= total_updates:
        return 0.0
    if warmup_updates and update_index < warmup_updates:
        return (update_index + 1) / warmup_updates
    decay = total_updates - warmup_updates
    if decay == 0:
        return 1.0
    progress = (update_index - warmup_updates + 1) / decay
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def validate_partition(all_names, inner_names, holdout_names):
    """Reject ambiguous labels and split leakage before indexing any targets."""
    for label, names in (("labels", all_names), ("train_inner", inner_names), ("holdout", holdout_names)):
        if not names or len(names) != len(set(names)):
            raise ValueError(label + " is empty or contains duplicate filenames")
    available, inner, hold = set(all_names), set(inner_names), set(holdout_names)
    if inner & hold:
        raise ValueError("train_inner and holdout overlap")
    unknown = (inner | hold) - available
    missing = available - (inner | hold)
    if unknown or missing:
        raise ValueError(f"split/labels mismatch: unknown={len(unknown)}, unassigned={len(missing)}")
    return {"all": len(available), "train_inner": len(inner), "holdout": len(hold)}


def improved_macro(current, best):
    if not math.isfinite(current):
        raise ValueError("holdout Macro F1 is not finite")
    return current > best
