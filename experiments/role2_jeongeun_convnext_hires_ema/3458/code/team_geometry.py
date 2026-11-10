"""Geometry and masked GAP from the team train_hires.py."""
import math

STRIDE = 32

import numpy as np
import torch
from torch import nn
import timm

def target_size(w, h, short, max_long):
    """(new_w, new_h) keeping the aspect ratio: short side -> short, long side capped at max_long
    (mmdet Resize keep_ratio with img_scale=(max_long, short))."""
    scale = min(short / min(w, h), max_long / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def ceil32(v):
    return math.ceil(v / STRIDE) * STRIDE


def long_cap(level, max_long):
    """Long-side cap for a short-side level: Fashionformer's 1333 at full scale, 2x the short side below it
    (aspect ratios above 2 are rare), so every level has one fixed canvas."""
    return min(max_long, 2 * level)


def canvas(level, portrait, max_long):
    long_, short_ = ceil32(long_cap(level, max_long)), ceil32(level)
    return (long_, short_) if portrait else (short_, long_)            # (H, W)


def short_levels(a):
    """Fixed short-side levels. Every new input shape costs cuDNN a slow first call (measured ~1.5 s/img on
    an A5000 vs 0.03 s/img once seen), so the per-image random short side of Fashionformer is quantised to a
    few levels and each batch is padded to that level's canvas: 2 orientations x len(levels) shapes in all."""
    if a.short_levels:
        return sorted(a.short_levels)
    if a.short_min == a.short_max:
        return [a.short_max]
    return sorted({min(a.short_max, ceil32(v)) for v in np.linspace(a.short_min, a.short_max, 4)})


class BucketBatchSampler:
    """Batches of (index, level) whose images share one canvas (orientation x level).
    train: each epoch draws a level per image (seed, epoch), shuffles inside each bucket and then the batch
    order. eval: one level, original order inside each orientation. Call set_epoch before iterating."""

    def __init__(self, portrait, levels, batch_size, seed, train, n_core=None, extra_per_epoch=0, pools=None):
        """Train only. pools = [(start, stop, k), ...]: indices in [start, stop) are an extra pool of which k are
        drawn per epoch (negatives, person positives), so their share stays fixed whatever their count; indices
        outside every pool are used every epoch. n_core / extra_per_epoch = one pool [n_core, n)."""
        self.portrait, self.levels, self.bs, self.seed, self.train = portrait, list(levels), batch_size, seed, train
        n = len(portrait)
        if pools is None and n_core is not None and n_core < n:
            pools = [(n_core, n, extra_per_epoch)]
        self.pools = pools or []
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def plan(self, epoch):
        rng = np.random.default_rng([self.seed, epoch])
        n = len(self.portrait)
        lv = rng.choice(self.levels, n) if self.train else np.full(n, self.levels[-1])
        use = np.ones(n, dtype=bool)
        if self.train:
            for start, stop, k in self.pools:
                use[start:stop] = False
                use[start + rng.choice(stop - start, min(k, stop - start), replace=False)] = True
        batches = []
        for o in (True, False):
            for level in self.levels:
                idx = np.flatnonzero(use & (self.portrait == o) & (lv == level))
                if self.train:
                    idx = rng.permutation(idx)
                batches += [[(int(i), int(level)) for i in idx[k:k + self.bs]] for k in range(0, len(idx), self.bs)]
        if self.train:
            batches = [batches[k] for k in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        return iter(self.plan(self.epoch))

    def __len__(self):
        return len(self.plan(self.epoch))

def valid_mask(sizes, h, w, H, W):
    """bool [B, h, w]: feature cells covering the real image area of each padded input (H, W) -> (h, w).
    A cell counts if any part of it is real, i.e. ceil(real * h / H) rows and columns from the top-left."""
    fh = torch.clamp(torch.ceil(sizes[:, 0].float() * h / H), min=1).long()
    fw = torch.clamp(torch.ceil(sizes[:, 1].float() * w / W), min=1).long()
    dev = sizes.device
    rows = torch.arange(h, device=dev)[None, :, None] < fh[:, None, None]
    cols = torch.arange(w, device=dev)[None, None, :] < fw[:, None, None]
    return rows & cols


def masked_pool(f, valid, mode="gap"):
    """f [B, C, h, w], valid bool [B, h, w] -> [B, C]: average / max / 0.5*(avg+max) over the valid cells only.
    Padded cells never enter: they are zeroed for the average and set to the dtype minimum for the max."""
    m = valid.unsqueeze(1)
    avg = (f * m.to(f.dtype)).sum((2, 3)) / m.sum((2, 3)).to(f.dtype)
    if mode == "gap":
        return avg
    mx = f.masked_fill(~m, torch.finfo(f.dtype).min).amax((2, 3))
    return mx if mode == "gmp" else 0.5 * (avg + mx)


class MaskedPoolConvNeXt(nn.Module):
    """timm ConvNeXt with its own head, but pooling only over the real image area (gap = the C1 runs)."""

    def __init__(self, name, num_classes, pretrained, drop_path, pool="gap"):
        super().__init__()
        self.pool = pool
        extra = {"drop_path_rate": drop_path} if drop_path else {}
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **extra)
        head = self.net.head
        assert all(hasattr(head, k) for k in ("norm", "flatten", "pre_logits", "drop", "fc")), \
            f"{name}: expected a timm ConvNeXt NormMlpClassifierHead"

    def forward(self, x, sizes):
        f = self.net.forward_features(x)                                   # [B, C, h, w], stride 32
        valid = valid_mask(sizes.to(f.device), *f.shape[-2:], *x.shape[-2:])
        pooled = masked_pool(f, valid, self.pool)[:, :, None, None]
        hd = self.net.head
        return hd.fc(hd.drop(hd.pre_logits(hd.flatten(hd.norm(pooled)))))

    def head_parameters(self):
        return list(self.net.head.fc.parameters())

