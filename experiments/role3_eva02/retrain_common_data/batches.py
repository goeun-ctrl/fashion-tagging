"""Consume one published draw at a time, with one denominator per optimizer window."""
from itertools import islice

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from . import require
from .preflight import CFG, MEAN, STD
from fashion_tagging.data.dataset import FashionDataset
from fashion_tagging.data.preprocess import preprocess, to_rgb

ENHANCERS = (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color)
BCE_SUM = torch.nn.BCEWithLogitsLoss(reduction="sum")


class PlannedDataset(Dataset):
    def __init__(self, paths, targets, plan):
        self.paths = paths
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        self.indices, self.flip, self.factors, self.order = plan

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, draw):
        idx = int(self.indices[draw])
        with Image.open(self.paths[idx]) as source:
            image = to_rgb(source)
        if self.flip[draw]:
            image = ImageOps.mirror(image)
        for op in self.order[draw]:
            factor = float(self.factors[draw, op])
            if factor != 1.0:
                image = ENHANCERS[op](image).enhance(factor)
        x = preprocess(image, CFG["input_size"], "pad", MEAN, STD)
        return torch.from_numpy(x), self.targets[idx]


def eval_dataset(group):
    root, names, targets = group
    return FashionDataset(root, names, targets, CFG["input_size"], "pad", False, MEAN, STD)


def loader(ds, workers):
    return DataLoader(ds, shuffle=False, batch_size=CFG["microbatch"], num_workers=workers, pin_memory=True)


def loss_sum(logits, targets, mode):
    require(logits.shape == targets.shape, "logits/targets shape mismatch")
    if mode == "partial":
        known = targets != -1
        logits, targets = logits[known], targets[known]
    elif mode != "complete":
        raise ValueError("invalid label mode")
    return BCE_SUM(logits.float(), targets.float())


def window_size(batches, mode):
    # Targets stay on CPU until the denominator is known; at most two batches are held.
    counts = 0
    for _, y in batches:
        known = (y == 0) | (y == 1)
        if mode == "complete":
            require(bool(known.all()), "incomplete labels in complete mode")
        else:
            require(mode == "partial" and bool((known | (y == -1)).all()), "invalid partial labels")
        counts += int(known.sum())
    return counts


def train_epoch(model, batches, opt, sched, scaler, mode, accum):
    from ..run import update

    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss, total_known, steps, skips, empty = 0., 0, 0, 0, 0
    positives = np.zeros(27, dtype=np.int64)
    known_by_class = np.zeros_like(positives)
    iterator = iter(batches)
    while window := list(islice(iterator, accum)):
        n = window_size(window, mode)
        if n == 0:
            empty += 1
            continue
        for x, y in window:
            positives += (y == 1).sum(0).numpy()
            known_by_class += (y != -1).sum(0).numpy()
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            numerator = loss_sum(logits, y, mode)
            value = numerator.item()
            require(np.isfinite(value), "non-finite train loss")
            scaler.scale(numerator / n).backward()
            total_loss += value
        total_known += n
        stepped = update(scaler, opt, sched)
        steps += int(stepped)
        skips += int(not stepped)
    return ({"train_bce": total_loss / total_known if total_known else None, "known_labels": total_known,
             "optimizer_steps": steps, "amp_skips": skips, "empty_windows": empty},
            positives.tolist(), known_by_class.tolist())
