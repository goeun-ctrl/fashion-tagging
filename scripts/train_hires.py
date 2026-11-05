"""High-resolution ConvNeXt classifier with the Fashionformer R50 input / optimisation recipe, adapted to
image-level labels. Design and reasons: docs/plan/hires_convnext_recipe.md.

What differs from scripts/train.py (team baseline: 384 square pad, flip, BCE, cosine):
  input      keep aspect ratio, no square padding. Train: short side random in [--short_min, --short_max],
             long side <= --max_long. Test: short side --test_short (Fashionformer test scale 1333x800).
             Bilinear resize, flip 0.5. No random crop (a crop would drop objects whose image label stays 1).
  batching   images padded to the largest H, W in the batch (multiple of 32); GAP is masked to the real
             image area so padding never enters the features.
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


class HiResDataset(Dataset):
    def __init__(self, root, names, targets, mean, std, train, short_range, max_long, flip):
        self.root, self.names = os.path.join(str(root), "images"), list(names)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.train, self.short_range, self.max_long, self.flip = train, short_range, max_long, flip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with Image.open(os.path.join(self.root, self.names[i])) as img:
            img.load()
            img = to_rgb(img)                                    # EXIF transpose + RGB (as the team baseline)
        lo, hi = self.short_range
        short = random.randint(lo, hi) if self.train else hi
        img = img.resize(target_size(*img.size, short, self.max_long), Image.BILINEAR)
        if self.train and random.random() < self.flip:
            img = ImageOps.mirror(img)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.targets[i])


def pad_collate(batch):
    """Pad to the batch's max H, W (rounded up to a multiple of 32). Returns images, targets and the real
    (h, w) of each image for masked pooling."""
    hs = [x.shape[1] for x, _ in batch]
    ws = [x.shape[2] for x, _ in batch]
    H, W = math.ceil(max(hs) / STRIDE) * STRIDE, math.ceil(max(ws) / STRIDE) * STRIDE
    out = torch.zeros(len(batch), 3, H, W)
    for k, (x, _) in enumerate(batch):
        out[k, :, :x.shape[1], :x.shape[2]] = x
    return out, torch.stack([t for _, t in batch]), torch.tensor(list(zip(hs, ws)))


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
    model.eval()
    outs, fwd = [], 0.0
    for x, _, sizes in loader:
        x, sizes = x.to(device, non_blocking=True), sizes.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            o = model(x, sizes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd += time.perf_counter() - t0
        outs.append(o.float().cpu())
    return torch.cat(outs).numpy(), fwd


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

    def ds(root, n, t, train):
        rng = (a.short_min, a.short_max) if train else (a.test_short, a.test_short)
        return HiResDataset(root, n, t, mean, std, train, rng, a.max_long, a.flip)

    g = torch.Generator().manual_seed(a.seed)
    kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda", collate_fn=pad_collate,
              worker_init_fn=seed_worker)
    train_loader = DataLoader(ds(a.train_dir, inner, t_in, True), batch_size=a.batch_size, shuffle=True,
                              drop_last=False, generator=g, persistent_workers=a.workers > 0, **kw)
    hold_loader = DataLoader(ds(a.train_dir, hold, t_ho, False), batch_size=a.eval_batch_size, **kw)
    val_loader = DataLoader(ds(a.val_dir, vnames, vtargets, False), batch_size=a.eval_batch_size, **kw)

    model = MaskedPoolConvNeXt(a.model, len(cats), not a.no_pretrained, a.drop_path).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= PARAM_LIMIT, f"{n_params:,} parameters exceed the 50M limit"
    head_ids = {id(p) for p in model.head_parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": a.lr_backbone, "name": "backbone"},
        {"params": model.head_parameters(), "lr": a.lr_head, "name": "head"},
    ], weight_decay=a.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / a.accum)
    total = a.epochs * steps_per_epoch
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
        for bi, (x, y, sizes) in enumerate(train_loader):
            x, y, sizes = x.to(device, non_blocking=True), y.to(device, non_blocking=True), sizes.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x, sizes)
            loss = loss_fn(logits.float(), y)
            assert torch.isfinite(loss), "non-finite training loss"
            scaler.scale(loss / a.accum).backward()
            run_loss += loss.item() * len(x)
            seen += len(x)
            if (bi + 1) % a.accum == 0 or bi + 1 == len(train_loader):
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
