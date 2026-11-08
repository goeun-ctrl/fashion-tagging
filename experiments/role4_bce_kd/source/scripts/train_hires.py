"""High-resolution ConvNeXt classifier with the Fashionformer R50 input / optimisation recipe, adapted to
image-level labels. Design and reasons: docs/plan/hires_convnext_recipe.md.

What differs from scripts/train.py (team baseline: 384 square pad, flip, BCE, cosine):
  input      keep aspect ratio, no square padding. Train: short side random in [--short_min, --short_max],
             long side <= --max_long. Test: short side --test_short (Fashionformer test scale 1333x800).
             Bilinear resize, flip 0.5. No random crop (a crop would drop objects whose image label stays 1).
  batching   the random short side is quantised to 4 levels and each batch holds one orientation x level,
             padded to that level's fixed canvas (8 shapes in all; a new shape costs cuDNN ~1.5 s/img once).
             GAP is masked to the real image area so padding never enters the features. channels_last.
  loss       ASL (gamma_neg 4, gamma_pos 0, clip 0.05) by default, --loss bce for the baseline loss.
  optimiser  AdamW (wd 0.05), separate backbone / head lr, gradient clipping (max_norm 1.0).
  schedule   per optimizer update: linear warmup from --warmup_ratio over --warmup_iters, then step decay
             x0.1 at --milestones (fractions of all updates; Fashionformer: epochs 8 and 11 of 12).
  EMA        ModelEmaV3 (--ema 0.9995); holdout selection and final reports use the EMA weights.

    python scripts/train_hires.py --exp_id R5_cnxt_hr_s42 --train_dir data/train --val_dir data/val
    python scripts/train_hires.py ... --lr_backbone 2.5e-5 --lr_head 1e-4      # Fashionformer lr
    python scripts/train_hires.py ... --short_min 384 --short_max 384 --test_short 384   # fixed-scale control
    python scripts/train_hires.py ... --limit_train 64 --limit_eval 32 --epochs 1 --device cpu   # smoke

Outputs in <out_dir>/<exp_id>/: best.pt (EMA weights at the best holdout epoch), last.pt, history.csv,
preds_holdout.npz, preds_val.npz, summary.json. Best epoch = highest holdout macro F1 at threshold 0.5.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import io

from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fashion_tagging.data.dataset import read_names  # noqa: E402
from fashion_tagging.data.labels import load_split  # noqa: E402
from fashion_tagging.data.preprocess import timm_mean_std, to_rgb  # noqa: E402
from fashion_tagging.evaluation.metrics import binarize, compute_metrics, save_predictions, sigmoid  # noqa: E402
from fashion_tagging.paths import SPLITS_DIR  # noqa: E402
from fashion_tagging.training.losses import LOSSES, build_loss  # noqa: E402

PARAM_LIMIT = 50_000_000
STRIDE = 32


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--train_dir", default=os.path.join(ROOT, "data", "train"))
    ap.add_argument("--val_dir", default=os.path.join(ROOT, "data", "val"))
    ap.add_argument("--splits_dir", default=str(SPLITS_DIR))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k_384")
    ap.add_argument("--no_pretrained", action="store_true", help="smoke tests only")
    ap.add_argument("--head", default="linear", choices=["linear", "linear_ms", "query_ms"],
                    help="linear = masked pooling + timm head (C1). linear_ms = masked pooling of the stride-16 AND "
                         "stride-32 maps, concatenated -> LayerNorm -> Linear (~+33k params). query_ms = 27 class "
                         "queries cross-attending to both maps (C3; plan suggests --lr_head 1e-4)")
    ap.add_argument("--pool", default="gap", choices=["gap", "gmp", "gapgmp", "csra"],
                    help="linear / linear_ms pooling over the real image area: average, max, or 0.5*(avg+max) "
                         "(team names, src/fashion_tagging/models/heads.py). Max keeps small-object peaks. "
                         "csra (linear_ms only): class-specific residual attention, Zhu & Wu ICCV 2021 - every class "
                         "scores every cell and adds an attention-pooled score of its own region to the average")
    ap.add_argument("--csra_lambda", type=float, default=0.3)
    ap.add_argument("--csra_t", type=float, default=1.0, help="softmax temperature over cells (large = max)")
    ap.add_argument("--query_dim", type=int, default=384)
    ap.add_argument("--query_heads", type=int, default=8)
    ap.add_argument("--head_dropout", type=float, default=0.1)
    # empty-label negatives (scenes without people, e.g. COCO no-person + Places365) and a "garment present" gate
    ap.add_argument("--neg_list", default=None, help="text file, one absolute image path per line (train negatives)")
    ap.add_argument("--neg_holdout_list", default=None, help="held-out negatives for the empty-rate check")
    ap.add_argument("--neg_ratio", type=float, default=0.1, help="share of negatives in each training epoch")
    ap.add_argument("--pos_list", default=None,
                    help="person images from outside Fashionpedia (COCO / Places): gate 1, class labels unknown (-1)")
    ap.add_argument("--pos_holdout_list", default=None, help="held-out person images (wrongly-emptied rate)")
    ap.add_argument("--pos_ratio", type=float, default=0.1, help="share of --pos_list images in each epoch")
    ap.add_argument("--ext_list", default=None,
                    help="extra labelled images outside Fashionpedia (absolute paths), e.g. pseudo-labelled COCO crowds")
    ap.add_argument("--ext_labels", default=None,
                    help="npz with filenames + targets [N, 27] in {1, 0, -1 = unknown, masked} for --ext_list "
                         "(scripts/pseudo_label.py); with --gate they count as gate 1")
    ap.add_argument("--ext_ratio", type=float, default=0.2, help="share of --ext_list images in each epoch")
    ap.add_argument("--gray_p", type=float, default=0.0, help="random grayscale probability on training images")
    # OOD augmentations (test_public: crowds, distant people, old B/W photos). All keep image-level labels exact:
    # nothing is cropped away, mosaic labels are the union of its tiles.
    ap.add_argument("--zoom_p", type=float, default=0.0,
                    help="zoom-out: shrink the image to [--zoom_min, 0.85] of its size on a mean-colour canvas")
    ap.add_argument("--zoom_min", type=float, default=0.35)
    ap.add_argument("--mosaic_p", type=float, default=0.0,
                    help="mosaic of several Fashionpedia images (labels = union): many people / many labels")
    ap.add_argument("--mosaic_sizes", type=int, nargs="+", default=[2, 4], choices=[2, 4, 6, 9],
                    help="tile counts drawn uniformly per mosaic (9 = 3x3, about the test set's 9 labels per image)")
    ap.add_argument("--mask_dir", default=None,
                    help="per-class garment masks (scripts/build_masks.py, train_inner only): supervise the CSRA "
                         "score maps cell by cell so every class learns WHERE its garment is (needs --pool csra)")
    ap.add_argument("--mask_weight", type=float, default=0.5)
    ap.add_argument("--old_p", type=float, default=0.0,
                    help="old-photo look: sepia or gray + film grain + low resolution + low contrast + JPEG")
    ap.add_argument("--gate", action="store_true",
                    help="extra 28th output: P(any garment present). Positives 1, negatives 0. At test time an "
                         "image whose gate prob is below --gate_threshold gets the empty label set")
    ap.add_argument("--gate_weight", type=float, default=1.0)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--init_ckpt", default=None,
                    help="start from a train_hires checkpoint (e.g. the r5-cnxt-hr-s42 best.pt) instead of ImageNet")
    # input (Fashionformer: test 1333x800 keep ratio; train jitter; we drop the crop)
    ap.add_argument("--short_min", type=int, default=400)
    ap.add_argument("--short_max", type=int, default=800)
    ap.add_argument("--test_short", type=int, default=800)
    ap.add_argument("--max_long", type=int, default=1333)
    ap.add_argument("--short_levels", type=int, nargs="*", default=None,
                    help="train short-side levels (default: 4 levels from short_min to short_max, multiples of 32)")
    ap.add_argument("--flip", type=float, default=0.5)
    # optimisation
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0, help="gradient clipping max_norm (0 = off)")
    ap.add_argument("--warmup_iters", type=int, default=1000, help="optimizer updates")
    ap.add_argument("--warmup_ratio", type=float, default=0.001, help="lr factor at the first update")
    ap.add_argument("--milestones", type=float, nargs="*", default=[8 / 12, 11 / 12],
                    help="fractions of all updates where lr is multiplied by --gamma")
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--loss", default="asl", choices=LOSSES)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.9995, help="0 = off")
    # distillation from Fashionformer image scores (experiments/fashionformer_eval/ff_infer.py on data/train)
    ap.add_argument("--distill_npz", default=None, help="teacher image-level probabilities for train images")
    ap.add_argument("--distill_alpha", type=float, default=0.5, help="loss = (1-a) * hard loss + a * soft BCE")
    ap.add_argument("--distill_shift", type=float, default=0.3,
                    help="teacher operating threshold; scores are shifted in logit space so it maps to 0.5")
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=0, help="early stopping on holdout macro F1 (0 = off)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--limit_train", type=int, default=0, help="smoke: first N train_inner images")
    ap.add_argument("--limit_eval", type=int, default=0, help="smoke: first N holdout/val images")
    a = ap.parse_args(argv)
    assert a.short_min <= a.short_max, "short_min must be <= short_max"
    return a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32     # distinct per worker and per epoch, derived from the loader generator
    random.seed(s)
    np.random.seed(s)


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


def oriented_portrait(root, names):
    """True where the image (after EXIF transpose) is at least as tall as wide. Header reads only."""
    from concurrent.futures import ThreadPoolExecutor

    def one(n):
        with Image.open(os.path.join(str(root), "images", n)) as im:
            w, h = im.size
            if im.getexif().get(274, 1) in (5, 6, 7, 8):              # 90/270 degree EXIF orientations
                w, h = h, w
        return h >= w
    with ThreadPoolExecutor(16) as ex:
        return np.array(list(ex.map(one, names, chunksize=256)), dtype=bool)


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


class HiResDataset(Dataset):
    def __init__(self, root, names, targets, mean, std, train, max_long, flip, mirror=False, soft=None, gray_p=0.0,
                 zoom_p=0.0, zoom_min=0.35, mosaic_p=0.0, mosaic_n=0, old_p=0.0, mask_dir=None, mask_kind=None,
                 mosaic_sizes=(2, 4)):
        """mosaic_n: indices < mosaic_n (the labelled Fashionpedia images) can form or join a mosaic.
        mask_dir / mask_kind (train): per index 1 = mask file <mask_dir>/<name>.npz, 0 = no garment anywhere
        (negatives: an all-zero mask is exact), -1 = unknown (person positives: no mask loss). The masks follow
        every geometric augmentation (mosaic, zoom-out, mirror) and come out on the stride-16 grid."""
        """mirror: always flip (eval-time flip TTA; flipped before padding so the GAP mask stays valid).
        soft: optional [N, C] teacher probabilities (NaN rows = no teacher score) returned as a 5th item."""
        self.root, self.names = os.path.join(str(root), "images"), list(names)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.train, self.max_long, self.flip, self.mirror = train, max_long, flip, mirror
        self.soft = None if soft is None else np.asarray(soft, dtype=np.float32)
        self.gray_p, self.zoom_p, self.zoom_min, self.old_p = gray_p, zoom_p, zoom_min, old_p
        self.mosaic_p, self.mosaic_n, self.mosaic_sizes = mosaic_p, mosaic_n, tuple(mosaic_sizes)
        self.mask_dir, self.mask_kind = mask_dir, mask_kind
        self.fill = tuple(int(round(255 * m)) for m in np.asarray(mean).reshape(-1)[:3])

    def __len__(self):
        return len(self.names)

    def load_mask(self, i, size):
        """bool [27, gh, gw] aligned with the loaded image of `size` (w, h); None when unknown."""
        kind = self.mask_kind[i]
        w, h = size
        if kind < 0:
            return None
        if kind == 0:
            sc = 128 / max(w, h)
            return np.zeros((27, max(1, round(h * sc)), max(1, round(w * sc))), bool)
        z = np.load(os.path.join(self.mask_dir, os.path.basename(self.names[i]) + ".npz"))
        gh, gw = (int(v) for v in z["shape"])
        cls = z["cls"]
        bits = np.unpackbits(z["bits"])[:len(cls) * gh * gw].reshape(len(cls), gh, gw).astype(bool)
        m = np.zeros((27, gh, gw), bool)
        m[cls] = bits
        return m

    @staticmethod
    def resize_mask(m, hw):
        """nearest-neighbour resize of bool [K, h, w] to [K, H, W]."""
        H, W = max(1, hw[0]), max(1, hw[1])
        r = np.minimum((np.arange(H) + 0.5) * m.shape[1] / H, m.shape[1] - 1).astype(int)
        c = np.minimum((np.arange(W) + 0.5) * m.shape[2] / W, m.shape[2] - 1).astype(int)
        return m[:, r][:, :, c]

    @classmethod
    def paste_mask(cls, canvas_m, size, m, box):
        """paste tile mask m into canvas_m (grid of an image of `size` (w, h)) at pixel box (x, y, w, h)."""
        w, h = size
        gh, gw = canvas_m.shape[1:]
        x0, y0, bw, bh = box
        r0, c0 = int(y0 * gh / h), int(x0 * gw / w)
        r1, c1 = max(r0 + 1, int(round((y0 + bh) * gh / h))), max(c0 + 1, int(round((x0 + bw) * gw / w)))
        canvas_m[:, r0:r1, c0:c1] = cls.resize_mask(m, (r1 - r0, c1 - c0))[:, :min(r1, gh) - r0, :min(c1, gw) - c0]
        return canvas_m

    def blank_mask(self, size):
        w, h = size
        sc = 128 / max(w, h)
        return np.zeros((27, max(1, round(h * sc)), max(1, round(w * sc))), bool)

    def load(self, i):
        with Image.open(os.path.join(self.root, self.names[i])) as img:
            img.load()
            return to_rgb(img)                                   # EXIF transpose + RGB (as the team baseline)

    def mosaic(self, img, i, mask=None):
        """k tiles on the base image's own size (so the batch canvas / orientation stays the same): 2 side by side
        along the long side, 4 = 2x2, 6 = 3x2 along the long side, 9 = 3x3 (k from self.mosaic_sizes). Each tile
        keeps its aspect ratio, letterboxed with the mean colour."""
        w, h = img.size
        k = random.choice(self.mosaic_sizes)
        others = [j + (j >= i) for j in (random.randrange(self.mosaic_n - 1) for _ in range(k - 1))]   # never i
        rows, cols = {2: (2, 1), 4: (2, 2), 6: (3, 2), 9: (3, 3)}[k]
        if w > h:
            rows, cols = cols, rows
        xs, ys = [w * c // cols for c in range(cols + 1)], [h * r // rows for r in range(rows + 1)]
        cells = [(xs[c], ys[r], xs[c + 1] - xs[c], ys[r + 1] - ys[r]) for r in range(rows) for c in range(cols)]
        out = Image.new("RGB", (w, h), self.fill)
        om = self.blank_mask((w, h)) if mask is not None else None
        for (x0, y0, cw, ch), j in zip(cells, [i] + others):
            tile = img if j == i else self.load(j)
            tm = None if om is None else (mask if j == i else self.load_mask(j, tile.size))
            s = min(cw / tile.size[0], ch / tile.size[1])
            tile = tile.resize((max(1, round(tile.size[0] * s)), max(1, round(tile.size[1] * s))), Image.BILINEAR)
            px, py = x0 + (cw - tile.size[0]) // 2, y0 + (ch - tile.size[1]) // 2
            out.paste(tile, (px, py))
            if om is not None:
                om = self.paste_mask(om, (w, h), tm, (px, py, tile.size[0], tile.size[1]))
        return (out, [i] + others) if mask is None else (out, [i] + others, om)

    def zoom_out(self, img, mask=None):
        w, h = img.size
        s = random.uniform(self.zoom_min, 0.85)
        small = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        out = Image.new("RGB", (w, h), self.fill)
        px, py = random.randint(0, w - small.size[0]), random.randint(0, h - small.size[1])
        out.paste(small, (px, py))
        if mask is None:
            return out
        return out, self.paste_mask(self.blank_mask((w, h)), (w, h), mask, (px, py, small.size[0], small.size[1]))

    @staticmethod
    def old_photo(img):
        g = img.convert("L")
        img = ImageOps.colorize(g, (40, 26, 13), (240, 224, 192)) if random.random() < 0.5 else g.convert("RGB")
        if random.random() < 0.7:                                              # low resolution / soft edges
            f = random.uniform(0.25, 0.6)
            w, h = img.size
            img = img.resize((max(8, round(w * f)), max(8, round(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        if random.random() < 0.7:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 0.9))
        if random.random() < 0.7:                                              # film grain
            a = np.asarray(img, np.float32)
            a = a + np.random.normal(0, random.uniform(4, 18), a.shape[:2])[..., None]
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        if random.random() < 0.5:
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=random.randint(20, 60)); buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img

    def __getitem__(self, key):
        i, level = key
        img = self.load(i)
        portrait = img.size[1] >= img.size[0]
        y = self.targets[i]
        soft = None if self.soft is None else self.soft[i]
        use_mask = self.train and self.mask_dir is not None
        mask = self.load_mask(i, img.size) if use_mask else None
        if self.train and self.mosaic_p and i < self.mosaic_n and random.random() < self.mosaic_p:
            if mask is not None:
                img, idx, mask = self.mosaic(img, i, mask)
            else:
                img, idx = self.mosaic(img, i)
            y = self.targets[idx].max(0)                                    # union of the tiles' labels
            if soft is not None:
                with np.errstate(all="ignore"):
                    soft = np.nanmax(self.soft[idx], 0) if not np.isnan(self.soft[idx][:, 0]).all() else soft
        if self.train and self.zoom_p and random.random() < self.zoom_p:
            if mask is not None:
                img, mask = self.zoom_out(img, mask)
            else:
                img = self.zoom_out(img)
        img = img.resize(target_size(*img.size, level, long_cap(level, self.max_long)), Image.BILINEAR)
        if self.mirror or (self.train and random.random() < self.flip):
            img = ImageOps.mirror(img)
            if mask is not None:
                mask = mask[:, :, ::-1]
        if mask is not None:                                                   # stride-16 grid of the real image
            mask = self.resize_mask(mask, (math.ceil(img.size[1] / 16), math.ceil(img.size[0] / 16)))
        if self.train and self.old_p and random.random() < self.old_p:
            img = self.old_photo(img)
        elif self.train and self.gray_p and random.random() < self.gray_p:
            img = img.convert("L").convert("RGB")                           # old B/W photos in the test set
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x = (x - self.mean) / self.std
        item = (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), i,
                canvas(level, portrait, self.max_long))
        if self.soft is not None or use_mask:
            item += (None if self.soft is None else torch.from_numpy(np.ascontiguousarray(soft, dtype=np.float32)),
                     None if not use_mask or mask is None else torch.from_numpy(np.ascontiguousarray(mask)))
            item += (use_mask and mask is not None,)
        return item


def pad_collate(batch):
    """Pad every image to the batch's canvas (fixed per orientation x level). Returns images, targets, the
    real (h, w) of each image for masked pooling and the dataset indices."""
    H = max(max(b[3][0] for b in batch), *(b[0].shape[1] for b in batch))
    W = max(max(b[3][1] for b in batch), *(b[0].shape[2] for b in batch))
    out = torch.zeros(len(batch), 3, ceil32(H), ceil32(W))
    for k, b in enumerate(batch):
        out[k, :, :b[0].shape[1], :b[0].shape[2]] = b[0]
    res = (out, torch.stack([b[1] for b in batch]), torch.tensor([b[0].shape[1:] for b in batch]),
           torch.tensor([b[2] for b in batch]))
    if len(batch[0]) > 4:                                    # (teacher soft targets, stride-16 masks, mask valid)
        soft = torch.stack([b[4] for b in batch]) if batch[0][4] is not None else None
        masks = None
        if any(b[6] for b in batch):
            masks = torch.zeros(len(batch), 27, ceil32(H) // 16, ceil32(W) // 16, dtype=torch.bool)
            for k, b in enumerate(batch):
                if b[6]:
                    masks[k, :, :b[5].shape[1], :b[5].shape[2]] = b[5]
        res += (soft, masks, torch.tensor([bool(b[6]) for b in batch]))
    return res


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


class MaskedPoolMSConvNeXt(nn.Module):
    """Stride-16 (stage 3) and stride-32 (stage 4) maps, each pooled over the real image area (gap / gmp /
    gapgmp), concatenated -> LayerNorm -> Linear. The finer map and the max keep small accessories that a
    stride-32 average dilutes; ~33k new parameters instead of the 2.2M of the query head (C3)."""

    def __init__(self, name, num_classes, pretrained, drop_path, pool="gapgmp", csra_lambda=0.3, csra_t=1.0):
        super().__init__()
        extra = {"drop_path_rate": drop_path} if drop_path else {}
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="", **extra)
        info = self.net.feature_info[-2:]
        assert info[0]["reduction"] < info[1]["reduction"], f"{name}: needs two stages with different strides"
        chs = [f["num_chs"] for f in info]
        self.pool, self.lam, self.t = pool, csra_lambda, csra_t
        if pool == "csra":
            # per level: LayerNorm over channels of every cell, then a 1x1 class scorer -> [B, K, h, w] score maps
            self.norms = nn.ModuleList(nn.LayerNorm(c) for c in chs)
            self.scorers = nn.ModuleList(nn.Linear(c, num_classes, bias=False) for c in chs)
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.norm = nn.LayerNorm(sum(chs))
            self.fc = nn.Linear(sum(chs), num_classes)

    def forward(self, x, sizes, return_maps=False):
        maps = self.net.forward_intermediates(x, indices=[-2, -1], intermediates_only=True)  # stride 16, 32
        H, W = x.shape[-2:]
        valids = [valid_mask(sizes.to(m.device), *m.shape[-2:], H, W) for m in maps]
        if self.pool != "csra":
            assert not return_maps, "score maps exist only for --pool csra"
            pooled = [masked_pool(m, v, self.pool) for m, v in zip(maps, valids)]
            return self.fc(self.norm(torch.cat(pooled, 1)))
        out, smaps = self.bias, []
        for m, v, norm, scorer in zip(maps, valids, self.norms, self.scorers):
            s = scorer(norm(m.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).float()      # [B, K, h, w]
            out = out + csra_pool(s, v, self.lam, self.t)
            smaps.append((s, v))
        return (out, smaps) if return_maps else out

    def head_parameters(self):
        if self.pool == "csra":
            return list(self.norms.parameters()) + list(self.scorers.parameters()) + [self.bias]
        return list(self.norm.parameters()) + list(self.fc.parameters())


def mask_loss(smaps, masks, mvalid, loss_fn, n_cls):
    """Cell-level loss on the CSRA score maps: stride 16 against the masks, stride 32 against their 2x2 max.
    Padded cells and images without a mask (person positives) are excluded (target -1 = masked in ASL/BCE)."""
    t16 = masks.float()
    t32 = F.max_pool2d(t16, 2, ceil_mode=True)
    tot = 0.0
    for (s, v), t in zip(smaps, (t16, t32)):
        t = t[:, :, :s.shape[2], :s.shape[3]]
        keep = (v & mvalid[:, None, None]).unsqueeze(1).expand_as(t)
        tgt = torch.where(keep, t, torch.full_like(t, -1.0))
        tot = tot + loss_fn(s[:, :n_cls].permute(0, 2, 3, 1).reshape(-1, n_cls),
                            tgt.permute(0, 2, 3, 1).reshape(-1, n_cls))
    return tot / 2


def csra_pool(s, valid, lam, t):
    """Class-specific residual attention over real cells: mean score + lam * sum_i softmax(t * s_i) * s_i.
    s [B, K, h, w] float32, valid [B, h, w]. Each class attends to its own cells (the region of that garment),
    which a shared average cannot do; padding gets -inf before the softmax and 0 weight in the mean."""
    m = valid.unsqueeze(1).expand_as(s).flatten(2)                               # [B, K, L]
    s = s.flatten(2)
    mean = (s * m).sum(-1) / m.sum(-1)
    w = torch.softmax((t * s).masked_fill(~m, float("-inf")), dim=-1)
    return mean + lam * (w * s).sum(-1)


class MaskedQueryConvNeXt(nn.Module):
    """C3 of the plan on the high-res recipe: timm ConvNeXt trunk (no classifier) + MultiScaleQueryHead from
    src/fashion_tagging/models/heads.py. Each of the 27 class queries cross-attends to the stride-16 and
    stride-32 token maps; padded cells are excluded with a key padding mask (same rule as the masked GAP), so a
    padded image gives the same logits as the image alone. Aimed at similar-garment confusions
    (vest / sweater / cardigan / jumpsuit): a class query can look at its own region instead of one global mean."""

    def __init__(self, name, num_classes, pretrained, drop_path, dim=384, heads=8, dropout=0.1):
        super().__init__()
        from fashion_tagging.models.heads import MultiScaleQueryHead
        extra = {"drop_path_rate": drop_path} if drop_path else {}
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="", **extra)
        info = self.net.feature_info[-2:]
        assert info[0]["reduction"] < info[1]["reduction"], f"{name}: needs two stages with different strides"
        self.head = MultiScaleQueryHead([f["num_chs"] for f in info], num_classes, dim, heads, dropout=dropout)

    def forward(self, x, sizes):
        maps = self.net.forward_intermediates(x, indices=[-2, -1], intermediates_only=True)  # stride 16, 32
        H, W = x.shape[-2:]
        tokens, pads = [], []
        for m in maps:
            tokens.append(m.flatten(2).transpose(1, 2))                       # [B, h*w, C], row-major
            pads.append(~valid_mask(sizes.to(m.device), *m.shape[-2:], H, W).flatten(1))   # True = padding
        return self.head(tokens, pads)

    def head_parameters(self):
        return list(self.head.parameters())


def build_hires_model(cfg, num_classes, pretrained):
    """cfg: argparse namespace or a saved config dict (older checkpoints have no 'head' key = linear)."""
    g = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    if g("head", "linear") == "query_ms":
        return MaskedQueryConvNeXt(g("model"), num_classes, pretrained, g("drop_path"), g("query_dim", 384),
                                   g("query_heads", 8), g("head_dropout", 0.1))
    if g("head", "linear") == "linear_ms":
        return MaskedPoolMSConvNeXt(g("model"), num_classes, pretrained, g("drop_path"), g("pool", "gapgmp"),
                                    g("csra_lambda", 0.3), g("csra_t", 1.0))
    return MaskedPoolConvNeXt(g("model"), num_classes, pretrained, g("drop_path"), g("pool", "gap"))


def load_grown(model, sd):
    """load_state_dict that also accepts a checkpoint with fewer outputs (27 classes -> 27 + gate): tensors
    whose first dim grew are copied into their leading rows, the new rows keep their fresh init."""
    own, grown = model.state_dict(), []
    for k, v in sd.items():
        t = own[k]
        if v.shape != t.shape:
            assert v.shape[1:] == t.shape[1:] and v.shape[0] < t.shape[0], (k, v.shape, t.shape)
            t = t.clone(); t[:v.shape[0]] = v; v = t; grown.append(k)
        own[k] = v
    model.load_state_dict(own)
    return grown


def gate_split(logits, n_cls):
    """[N, n_cls(+1)] -> (class logits [N, n_cls], gate logit [N] or None)."""
    return (logits[:, :n_cls], logits[:, n_cls]) if logits.shape[1] > n_cls else (logits, None)


def gated(pred, gate_logit, thr):
    """Empty the predictions of images whose gate probability is below thr."""
    if gate_logit is None:
        return pred
    return pred & (sigmoid(gate_logit) >= thr)[:, None]


def lr_factor(step, total, warmup_iters, warmup_ratio, milestones, gamma):
    """Multiplier for update `step` (0-based): linear warmup from warmup_ratio to 1, then step decay."""
    f = gamma ** sum(step >= int(m * total) for m in milestones)
    if step < warmup_iters:
        f *= warmup_ratio + (1 - warmup_ratio) * step / warmup_iters
    return f


@torch.no_grad()
def predict(model, loader, device, use_amp):
    """Logits in dataset order (the bucket sampler groups images by orientation)."""
    model.eval()
    outs, order, fwd = [], [], 0.0
    for x, _, sizes, idx in loader:
        order.append(idx)
        x, sizes = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last), sizes.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            o = model(x, sizes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd += time.perf_counter() - t0
        outs.append(o.float().cpu())
    logits, order = torch.cat(outs).numpy(), torch.cat(order).numpy()
    out = np.empty_like(logits)
    out[order] = logits
    assert len(order) == len(set(order.tolist())) == len(out)
    return out, fwd


def bce(logits, y):
    return float(F.binary_cross_entropy_with_logits(torch.as_tensor(logits, dtype=torch.float32),
                                                    torch.as_tensor(y, dtype=torch.float32)).item())


def soft_bce(logits, soft):
    """BCE against teacher probabilities; rows without a teacher score (NaN) are left out."""
    have = ~torch.isnan(soft[:, 0])
    if not bool(have.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[have].float(), soft[have])


def load_teacher(a, names):
    """[len(names), C] teacher probabilities in `names` order, NaN rows where the teacher has no score.
    Scores are shifted in logit space so the teacher's operating threshold (--distill_shift) maps to 0.5."""
    z = np.load(a.distill_npz, allow_pickle=False)
    idx = {n: i for i, n in enumerate(z["filenames"].tolist())}
    s = np.clip(z["scores"].astype(np.float64), 1e-4, 1 - 1e-4)
    if a.distill_shift:
        lg = np.log(s / (1 - s)) - math.log(a.distill_shift / (1 - a.distill_shift))
        s = 1 / (1 + np.exp(-lg))
    out = np.full((len(names), s.shape[1]), np.nan, np.float32)
    hit = [k for k, n in enumerate(names) if n in idx]
    out[hit] = s[[idx[names[k]] for k in hit]]
    print(f"teacher scores for {len(hit)}/{len(names)} train images (alpha {a.distill_alpha}, "
          f"shift {a.distill_shift})", flush=True)
    assert hit, "no teacher scores match the train names"
    return out


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not a.no_amp
    out = os.path.join(a.out_dir, a.exp_id)
    assert not os.path.exists(os.path.join(out, "summary.json")), f"{out} already finished; use a new --exp_id"
    os.makedirs(out, exist_ok=True)

    cats, names, targets = load_split(a.train_dir, "train")
    vcats, vnames, vtargets = load_split(a.val_dir, "val")
    assert cats == vcats and len(cats) == 27
    index = {n: i for i, n in enumerate(names)}
    inner, hold = read_names(os.path.join(a.splits_dir, "train_inner.txt")), read_names(os.path.join(a.splits_dir, "holdout.txt"))
    assert not set(inner) & set(hold) and set(inner) | set(hold) == set(names), "fixed split mismatch"
    if a.limit_train:
        inner = inner[:a.limit_train]
    if a.limit_eval:
        hold, vnames, vtargets = hold[:a.limit_eval], vnames[:a.limit_eval], vtargets[:a.limit_eval]
    t_in, t_ho = targets[[index[n] for n in inner]], targets[[index[n] for n in hold]]
    n_cls, n_pos = len(cats), len(inner)
    negs = read_names(a.neg_list) if a.neg_list else []
    neg_ho = read_names(a.neg_holdout_list) if a.neg_holdout_list else []
    poss = read_names(a.pos_list) if a.pos_list else []
    pos_ho = read_names(a.pos_holdout_list) if a.pos_holdout_list else []
    assert not poss or a.gate, "--pos_list only trains the gate; add --gate"
    exts = read_names(a.ext_list) if a.ext_list else []
    if exts:
        z = np.load(a.ext_labels, allow_pickle=False)
        row = {str(f): k for k, f in enumerate(z["filenames"])}
        miss = [e for e in exts if e not in row]
        assert not miss, f"{len(miss)} --ext_list images have no row in --ext_labels, e.g. {miss[:3]}"
        t_ext = z["targets"][[row[e] for e in exts]].astype(np.float32)
        assert t_ext.shape[1] == len(cats) and set(np.unique(t_ext)) <= {-1.0, 0.0, 1.0}, "targets must be [N, 27] in {1, 0, -1}"
    if a.limit_train:
        negs, poss = negs[:max(1, a.limit_train // 4)] if negs else negs, poss[:max(1, a.limit_train // 4)] if poss else poss
        exts = exts[:max(1, a.limit_train // 4)] if exts else exts
        t_ext = t_ext[:len(exts)] if exts else None
    if a.limit_eval:
        neg_ho, pos_ho = neg_ho[:a.limit_eval], pos_ho[:a.limit_eval]
    assert all(os.path.isabs(n) for n in negs + neg_ho + poss + pos_ho + exts), "extra image lists must hold absolute paths"
    # order: Fashionpedia inner | negatives | person positives | labelled extras (absolute paths: join(root, abs) == abs)
    train_names = list(inner) + list(negs) + list(poss) + list(exts)
    t_train = np.concatenate([t_in.astype(np.float32), np.zeros((len(negs), n_cls), np.float32),
                              np.full((len(poss), n_cls), -1.0, np.float32)]            # -1 = unknown, masked
                             + ([t_ext] if exts else []))
    assert t_train.shape == (len(train_names), n_cls)
    n_out = n_cls + (1 if a.gate else 0)
    if a.gate:                                                # gate target as an extra column
        gate_t = np.r_[np.ones(n_pos), np.zeros(len(negs)), np.ones(len(poss)), np.ones(len(exts))].astype(np.float32)
        t_train = np.concatenate([t_train, gate_t[:, None]], 1)
    share = 1 - (a.neg_ratio if negs else 0) - (a.pos_ratio if poss else 0) - (a.ext_ratio if exts else 0)
    neg_per_epoch = round(a.neg_ratio / share * n_pos) if negs else 0
    pos_per_epoch = round(a.pos_ratio / share * n_pos) if poss else 0
    ext_per_epoch = round(a.ext_ratio / share * n_pos) if exts else 0
    mask_kind = np.r_[np.ones(n_pos), np.zeros(len(negs)), -np.ones(len(poss)), -np.ones(len(exts))].astype(np.int8)
    if a.mask_dir:
        assert a.head == "linear_ms" and a.pool == "csra", "--mask_dir supervises the CSRA score maps"
        assert not set(hold) & set(inner), "holdout must never get masks"
        miss = [n for n in inner if not os.path.exists(os.path.join(a.mask_dir, n + ".npz"))]
        assert not miss, f"{len(miss)} train_inner images have no mask file, e.g. {miss[:3]}"
    pools = [(n_pos, n_pos + len(negs), neg_per_epoch)] if negs else []
    if poss:
        pools.append((n_pos + len(negs), n_pos + len(negs) + len(poss), pos_per_epoch))
    if exts:
        pools.append((len(train_names) - len(exts), len(train_names), ext_per_epoch))
    soft = load_teacher(a, train_names) if a.distill_npz else None
    mean, std = timm_mean_std(a.model)

    levels = short_levels(a)
    t0 = time.perf_counter()
    por_in, por_ho, por_va = (oriented_portrait(r, n) for r, n in
                              ((a.train_dir, train_names), (a.train_dir, hold), (a.val_dir, vnames)))
    por_ng = oriented_portrait("/", neg_ho) if neg_ho else None
    por_ps = oriented_portrait("/", pos_ho) if pos_ho else None
    if negs or poss or exts:
        print(f"negatives: {len(negs)} train ({neg_per_epoch}/epoch, ratio {a.neg_ratio}), {len(neg_ho)} held out; "
              f"person positives: {len(poss)} train ({pos_per_epoch}/epoch), {len(pos_ho)} held out; "
              f"labelled extras: {len(exts)} ({ext_per_epoch}/epoch); gate={a.gate} gray_p={a.gray_p}", flush=True)
    print(f"levels {levels}, test {a.test_short}; portrait share {por_in.mean():.2f} "
          f"(header scan {time.perf_counter() - t0:.0f}s)", flush=True)

    def loader(root, n, t, por, train, bs):
        sampler = BucketBatchSampler(por, levels if train else [a.test_short], bs, a.seed, train,
                                     pools=pools if train else None)
        ds = HiResDataset(root, n, t, mean, std, train, a.max_long, a.flip, soft=soft if train else None,
                          gray_p=a.gray_p if train else 0.0, zoom_p=a.zoom_p, zoom_min=a.zoom_min,
                          mosaic_p=a.mosaic_p, mosaic_n=n_pos if train else 0, old_p=a.old_p,
                          mosaic_sizes=a.mosaic_sizes,
                          mask_dir=a.mask_dir if train else None, mask_kind=mask_kind if train else None)
        return DataLoader(ds, batch_sampler=sampler,
                          num_workers=a.workers, pin_memory=device.type == "cuda", collate_fn=pad_collate,
                          worker_init_fn=seed_worker, persistent_workers=train and a.workers > 0)

    train_loader = loader(a.train_dir, train_names, t_train, por_in, True, a.batch_size)
    neg_loader = loader("/", neg_ho, np.zeros((len(neg_ho), n_cls), np.float32), por_ng, False,
                        a.eval_batch_size) if neg_ho else None
    pos_loader = loader("/", pos_ho, np.zeros((len(pos_ho), n_cls), np.float32), por_ps, False,
                        a.eval_batch_size) if pos_ho else None
    hold_loader = loader(a.train_dir, hold, t_ho, por_ho, False, a.eval_batch_size)
    val_loader = loader(a.val_dir, vnames, vtargets, por_va, False, a.eval_batch_size)

    # channels_last: ConvNeXt's 7x7 depthwise conv backward is ~20x slower in NCHW at these sizes (A5000, AMP:
    # 2.18 s vs 0.093 s per 4-image step at ~800x1216). Same math, memory layout only.
    model = build_hires_model(a, n_out, not a.no_pretrained).to(device)
    if a.init_ckpt:
        st = torch.load(a.init_ckpt, map_location="cpu")
        assert list(st["categories"]) == list(cats), "init_ckpt categories differ"
        grown = load_grown(model, st["model"])
        print(f"init from {a.init_ckpt} (epoch {st.get('epoch')})", flush=True)
    if a.gate and (not a.init_ckpt or grown):    # a checkpoint that already has the gate keeps its gate
        # prior init: gate bias = logit of the positive share, so training starts from "garments present" (the
        # behaviour of the 27-class model) instead of p=0.5, which would empty half of the holdout at first.
        b = {"query_ms": lambda: model.head.fc_b,
             "linear_ms": lambda: model.bias if a.pool == "csra" else model.fc.bias}.get(
            a.head, lambda: model.net.head.fc.bias)()
        with torch.no_grad():
            b[n_cls] = math.log((1 - a.neg_ratio) / a.neg_ratio) if negs else 4.0   # P(gate=1) = 1 - neg_ratio
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= PARAM_LIMIT, f"{n_params:,} parameters exceed the 50M limit"
    head_ids = {id(p) for p in model.head_parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": a.lr_backbone, "name": "backbone"},
        {"params": model.head_parameters(), "lr": a.lr_head, "name": "head"},
    ], weight_decay=a.weight_decay)
    sampler = train_loader.batch_sampler
    per_epoch = [math.ceil(len(sampler.plan(e)) / a.accum) for e in range(1, a.epochs + 1)]
    steps_per_epoch, total = round(sum(per_epoch) / len(per_epoch)), sum(per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, total, a.warmup_iters, a.warmup_ratio, a.milestones, a.gamma))
    loss_fn = build_loss(a.loss, a.asl_gamma_neg, a.asl_gamma_pos, a.asl_clip)
    ema = None
    if a.ema:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=a.ema, use_warmup=True)
    eval_model = ema.module if ema else model
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cfg = {k: v for k, v in vars(a).items()}
    cfg.update(params_total=n_params, steps_per_epoch=steps_per_epoch, total_updates=total,
               torch=str(torch.__version__), timm=str(timm.__version__), device=str(device),
               train=len(inner), holdout=len(hold), val=len(vnames), negatives=len(negs),
               neg_per_epoch=neg_per_epoch, neg_holdout=len(neg_ho), person_pos=len(poss), ext=len(exts), ext_per_epoch=ext_per_epoch,
               pos_per_epoch=pos_per_epoch, pos_holdout=len(pos_ho))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"[{a.exp_id}] params={n_params:,} train={len(inner)} holdout={len(hold)} updates={total} "
          f"short=[{a.short_min},{a.short_max}] test_short={a.test_short} loss={a.loss} ema={a.ema} head={a.head}", flush=True)

    history, best, best_epoch, step, skipped = [], -1.0, 0, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss, seen, t0 = 0.0, 0, time.perf_counter()
        sampler.set_epoch(epoch)
        n_batches = len(sampler)
        for bi, batch in enumerate(train_loader):
            x, y, sizes = batch[0], batch[1], batch[2]
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y, sizes = y.to(device, non_blocking=True), sizes.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if a.mask_dir:
                    logits, smaps = model(x, sizes, return_maps=True)
                else:
                    logits = model(x, sizes)
            loss = loss_fn(logits[:, :n_cls].float(), y[:, :n_cls])
            if a.mask_dir and batch[5] is not None:
                loss = loss + a.mask_weight * mask_loss(smaps, batch[5].to(device), batch[6].to(device), loss_fn, n_cls)
            if a.gate:
                loss = loss + a.gate_weight * F.binary_cross_entropy_with_logits(logits[:, n_cls].float(), y[:, n_cls])
            if soft is not None:
                loss = (1 - a.distill_alpha) * loss + a.distill_alpha * soft_bce(logits[:, :n_cls], batch[4].to(device))
            assert torch.isfinite(loss), "non-finite training loss"
            scaler.scale(loss / a.accum).backward()
            run_loss += loss.item() * len(x)
            seen += len(x)
            if (bi + 1) % a.accum == 0 or bi + 1 == n_batches:
                scaler.unscale_(opt)
                if a.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if use_amp and scaler.get_scale() < before:
                    skipped += 1                                   # overflow: no update, schedule waits
                    continue
                sched.step()
                step += 1
                if ema:
                    ema.update(model, step=step)
        train_sec = time.perf_counter() - t0
        lg_ho, g_ho = gate_split(predict(eval_model, hold_loader, device, use_amp)[0], n_cls)
        m = compute_metrics(t_ho, gated(binarize(sigmoid(lg_ho), a.threshold), g_ho, a.gate_threshold))
        row = dict(epoch=epoch, train_loss=run_loss / max(1, seen), holdout_bce=bce(lg_ho, t_ho),
                   holdout_micro=m["micro_f1"], holdout_macro=m["macro_f1"], holdout_image=m["image_f1"],
                   lr_backbone=opt.param_groups[0]["lr"], lr_head=opt.param_groups[1]["lr"],
                   updates=step, amp_skipped=skipped, train_sec=round(train_sec, 1))
        if g_ho is not None:
            row["holdout_gate_drop"] = float((sigmoid(g_ho) < a.gate_threshold).mean())   # positives wrongly emptied
        if neg_loader is not None:
            lg_ng, g_ng = gate_split(predict(eval_model, neg_loader, device, use_amp)[0], n_cls)
            pn = binarize(sigmoid(lg_ng), a.threshold)
            row["neg_empty_cls"] = float((~pn.any(1)).mean())                        # empty from classes alone
            row["neg_empty"] = float((~gated(pn, g_ng, a.gate_threshold).any(1)).mean())
        if pos_loader is not None:                                                  # must stay near 0
            lg_ps, g_ps = gate_split(predict(eval_model, pos_loader, device, use_amp)[0], n_cls)
            pp = binarize(sigmoid(lg_ps), a.threshold)
            row["pos_empty_cls"] = float((~pp.any(1)).mean())
            row["pos_empty"] = float((~gated(pp, g_ps, a.gate_threshold).any(1)).mean())
        history.append(row)
        state = {"model": eval_model.state_dict(), "epoch": epoch, "config": cfg, "categories": cats}
        torch.save(state, os.path.join(out, "last.pt"))
        if m["macro_f1"] > best:
            best, best_epoch = m["macro_f1"], epoch
            torch.save(state, os.path.join(out, "best.pt"))
        with open(os.path.join(out, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(history)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} holdout macro {m['macro_f1']:.4f} micro {m['micro_f1']:.4f} "
              f"bce {row['holdout_bce']:.4f} best {best_epoch} ({train_sec / 60:.1f} min)", flush=True)
        if a.patience and epoch - best_epoch >= a.patience:
            print("early stop", flush=True)
            break

    eval_model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device)["model"])
    summary = {"exp_id": a.exp_id, "best_epoch": best_epoch, "epochs_run": len(history), "params_total": n_params,
               "train_minutes": round((time.perf_counter() - t_start) / 60, 1), "amp_skipped": skipped}
    finals = [("holdout", hold_loader, t_ho, hold), ("val", val_loader, vtargets, vnames)]
    if neg_loader is not None:
        finals.append(("neg_holdout", neg_loader, np.zeros((len(neg_ho), n_cls), np.float32), neg_ho))
    if pos_loader is not None:
        finals.append(("pos_holdout", pos_loader, np.zeros((len(pos_ho), n_cls), np.float32), pos_ho))
    for split, loader, y, n in finals:
        lg, fwd = predict(eval_model, loader, device, use_amp)
        lg, g = gate_split(lg, n_cls)
        save_predictions(os.path.join(out, f"preds_{split}.npz"), n, y, lg, cats)
        if g is not None:
            np.savez(os.path.join(out, f"gate_{split}.npz"), filenames=np.array(n), gate_logits=g)
        pr = gated(binarize(sigmoid(lg), a.threshold), g, a.gate_threshold)
        if split in ("neg_holdout", "pos_holdout"):
            summary[split] = {"empty": float((~pr.any(1)).mean()), "pred_per_image": float(pr.sum(1).mean())}
            continue
        mm = compute_metrics(y, pr)
        summary[split] = {k: mm[k] for k in ("micro_f1", "macro_f1", "image_f1")}
        summary[split]["bce"] = bce(lg, y)
        summary[f"{split}_forward_ms_per_image"] = 1000 * fwd / len(n)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
