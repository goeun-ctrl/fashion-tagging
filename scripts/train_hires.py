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
from PIL import Image, ImageOps
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

    def __init__(self, portrait, levels, batch_size, seed, train):
        self.portrait, self.levels, self.bs, self.seed, self.train = portrait, list(levels), batch_size, seed, train
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def plan(self, epoch):
        rng = np.random.default_rng([self.seed, epoch])
        n = len(self.portrait)
        lv = rng.choice(self.levels, n) if self.train else np.full(n, self.levels[-1])
        batches = []
        for o in (True, False):
            for level in self.levels:
                idx = np.flatnonzero((self.portrait == o) & (lv == level))
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
    def __init__(self, root, names, targets, mean, std, train, max_long, flip):
        self.root, self.names = os.path.join(str(root), "images"), list(names)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.train, self.max_long, self.flip = train, max_long, flip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key):
        i, level = key
        with Image.open(os.path.join(self.root, self.names[i])) as img:
            img.load()
            img = to_rgb(img)                                    # EXIF transpose + RGB (as the team baseline)
        portrait = img.size[1] >= img.size[0]
        img = img.resize(target_size(*img.size, level, long_cap(level, self.max_long)), Image.BILINEAR)
        if self.train and random.random() < self.flip:
            img = ImageOps.mirror(img)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x = (x - self.mean) / self.std
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.targets[i]), i,
                canvas(level, portrait, self.max_long))


def pad_collate(batch):
    """Pad every image to the batch's canvas (fixed per orientation x level). Returns images, targets, the
    real (h, w) of each image for masked pooling and the dataset indices."""
    H = max(max(c[0] for *_, c in batch), *(x.shape[1] for x, *_ in batch))
    W = max(max(c[1] for *_, c in batch), *(x.shape[2] for x, *_ in batch))
    out = torch.zeros(len(batch), 3, ceil32(H), ceil32(W))
    for k, (x, *_) in enumerate(batch):
        out[k, :, :x.shape[1], :x.shape[2]] = x
    return (out, torch.stack([t for _, t, *_ in batch]), torch.tensor([x.shape[1:] for x, *_ in batch]),
            torch.tensor([i for _, _, i, _ in batch]))


class MaskedPoolConvNeXt(nn.Module):
    """timm ConvNeXt with its own head, but global average pooling only over the real image area."""

    def __init__(self, name, num_classes, pretrained, drop_path):
        super().__init__()
        extra = {"drop_path_rate": drop_path} if drop_path else {}
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **extra)
        head = self.net.head
        assert all(hasattr(head, k) for k in ("norm", "flatten", "pre_logits", "drop", "fc")), \
            f"{name}: expected a timm ConvNeXt NormMlpClassifierHead"

    def forward(self, x, sizes):
        f = self.net.forward_features(x)                                   # [B, C, h, w], stride 32
        h, w = f.shape[-2:]
        H, W = x.shape[-2:]
        fh = torch.clamp(torch.ceil(sizes[:, 0].float() * h / H), min=1).long()
        fw = torch.clamp(torch.ceil(sizes[:, 1].float() * w / W), min=1).long()
        rows = torch.arange(h, device=f.device)[None, :, None] < fh.to(f.device)[:, None, None]
        cols = torch.arange(w, device=f.device)[None, None, :] < fw.to(f.device)[:, None, None]
        m = (rows & cols).unsqueeze(1).to(f.dtype)                          # [B, 1, h, w]
        pooled = (f * m).sum((2, 3), keepdim=True) / m.sum((2, 3), keepdim=True)
        hd = self.net.head
        return hd.fc(hd.drop(hd.pre_logits(hd.flatten(hd.norm(pooled)))))

    def head_parameters(self):
        return list(self.net.head.fc.parameters())


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
    mean, std = timm_mean_std(a.model)

    levels = short_levels(a)
    t0 = time.perf_counter()
    por_in, por_ho, por_va = (oriented_portrait(r, n) for r, n in
                              ((a.train_dir, inner), (a.train_dir, hold), (a.val_dir, vnames)))
    print(f"levels {levels}, test {a.test_short}; portrait share {por_in.mean():.2f} "
          f"(header scan {time.perf_counter() - t0:.0f}s)", flush=True)

    def loader(root, n, t, por, train, bs):
        sampler = BucketBatchSampler(por, levels if train else [a.test_short], bs, a.seed, train)
        return DataLoader(HiResDataset(root, n, t, mean, std, train, a.max_long, a.flip), batch_sampler=sampler,
                          num_workers=a.workers, pin_memory=device.type == "cuda", collate_fn=pad_collate,
                          worker_init_fn=seed_worker, persistent_workers=train and a.workers > 0)

    train_loader = loader(a.train_dir, inner, t_in, por_in, True, a.batch_size)
    hold_loader = loader(a.train_dir, hold, t_ho, por_ho, False, a.eval_batch_size)
    val_loader = loader(a.val_dir, vnames, vtargets, por_va, False, a.eval_batch_size)

    # channels_last: ConvNeXt's 7x7 depthwise conv backward is ~20x slower in NCHW at these sizes (A5000, AMP:
    # 2.18 s vs 0.093 s per 4-image step at ~800x1216). Same math, memory layout only.
    model = MaskedPoolConvNeXt(a.model, len(cats), not a.no_pretrained, a.drop_path).to(device)
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
               train=len(inner), holdout=len(hold), val=len(vnames))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"[{a.exp_id}] params={n_params:,} train={len(inner)} holdout={len(hold)} updates={total} "
          f"short=[{a.short_min},{a.short_max}] test_short={a.test_short} loss={a.loss} ema={a.ema}", flush=True)

    history, best, best_epoch, step, skipped = [], -1.0, 0, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss, seen, t0 = 0.0, 0, time.perf_counter()
        sampler.set_epoch(epoch)
        n_batches = len(sampler)
        for bi, (x, y, sizes, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y, sizes = y.to(device, non_blocking=True), sizes.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x, sizes)
            loss = loss_fn(logits.float(), y)
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
        lg_ho, _ = predict(eval_model, hold_loader, device, use_amp)
        m = compute_metrics(t_ho, binarize(sigmoid(lg_ho), a.threshold))
        row = dict(epoch=epoch, train_loss=run_loss / max(1, seen), holdout_bce=bce(lg_ho, t_ho),
                   holdout_micro=m["micro_f1"], holdout_macro=m["macro_f1"], holdout_image=m["image_f1"],
                   lr_backbone=opt.param_groups[0]["lr"], lr_head=opt.param_groups[1]["lr"],
                   updates=step, amp_skipped=skipped, train_sec=round(train_sec, 1))
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
    for split, loader, y, n in (("holdout", hold_loader, t_ho, hold), ("val", val_loader, vtargets, vnames)):
        lg, fwd = predict(eval_model, loader, device, use_amp)
        save_predictions(os.path.join(out, f"preds_{split}.npz"), n, y, lg, cats)
        mm = compute_metrics(y, binarize(sigmoid(lg), a.threshold))
        summary[split] = {k: mm[k] for k in ("micro_f1", "macro_f1", "image_f1")}
        summary[split]["bce"] = bce(lg, y)
        summary[f"{split}_forward_ms_per_image"] = 1000 * fwd / len(n)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
