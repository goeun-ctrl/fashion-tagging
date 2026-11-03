"""Train one model on train_inner, pick the epoch on the internal holdout, report on both
holdout and the official val, and append one row to the shared results CSV.

The defaults are the team baseline (GAP + linear head, BCE, AdamW, warmup + cosine). Each part can
be swapped on its own (change one thing per experiment):
  --pooling gap|gmp|gapgmp   --head linear|mlp|query         (common/heads.py)
  --loss bce|asl                                             (common/losses.py)
  --optimizer adamw|muon|soap   --scheduler cosine|constant  (common/optim.py)

    python common/train_eval.py --exp_id S1_cnxt_t_pad --train_dir data/train --val_dir data/val
    python common/train_eval.py ... --preprocess_mode resize      # only the change under study
    python common/train_eval.py ... --head query --lr_head 1e-4  # C2 query head

Outputs in <out_dir>/<exp_id>/: best.pt, last.pt, preds_holdout.npz, preds_val.npz, log.json
and one appended row in --results_csv (columns follow configs/results_template.csv).
Resuming an interrupted run is not implemented: rerun from scratch.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import date

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import FashionDataset, read_names  # noqa: E402
from heads import HEADS, POOLS, build_model  # noqa: E402
from labels import load_split  # noqa: E402
from losses import LOSSES, build_loss  # noqa: E402
from optim import OPTIMIZERS, SCHEDULERS, build_optimizers, build_schedulers, describe  # noqa: E402
from metrics import binarize, compute_metrics, save_predictions, sigmoid  # noqa: E402
from preprocess import IMAGENET_MEAN, IMAGENET_STD, timm_mean_std  # noqa: E402

PARAM_LIMIT = 50_000_000


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--owner", default="")
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k_384")
    ap.add_argument("--no_pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--set_img_size", action="store_true", help="pass img_size to timm (ViT models such as EVA02)")
    ap.add_argument("--train_dir", required=True, help="folder with labels.csv+images/, or the train ZIP")
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--splits_dir", default=os.path.join(HERE, "splits"))
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--preprocess_mode", default="pad", choices=["pad", "resize"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--scheduler", default="cosine", choices=SCHEDULERS)
    # pooling / head
    ap.add_argument("--pooling", default="gap", choices=list(POOLS), help="ignored by --head query")
    ap.add_argument("--head", default="linear", choices=HEADS)
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--query_dim", type=int, default=384)
    ap.add_argument("--query_heads", type=int, default=8)
    ap.add_argument("--head_dropout", type=float, default=0.1, help="mlp and query heads")
    # loss
    ap.add_argument("--loss", default="bce", choices=LOSSES)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    # optimizer
    ap.add_argument("--optimizer", default="adamw", choices=OPTIMIZERS)
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--muon_lr_scale", type=float, default=1.0, help="multiplies lr of the Muon groups only")
    ap.add_argument("--muon_head", action="store_true", help="also put the head's inner matrices in Muon")
    ap.add_argument("--layer_decay", type=float, default=0.0, help="layer-wise lr decay, e.g. 0.75 (0 = off)")
    ap.add_argument("--drop_path", type=float, default=0.0, help="stochastic depth rate, e.g. 0.1 (0 = off)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train", type=int, default=0, help="use only the first N train images (smoke test)")
    ap.add_argument("--max_eval", type=int, default=0, help="use only the first N holdout/val images (smoke test)")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--results_csv", default=os.path.join("results", "results.csv"))
    ap.add_argument("--notes", default="")
    return ap.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_ds(path, split, names, targets, a, mean, std, train):
    zp = split if str(path).lower().endswith(".zip") else None
    return FashionDataset(path, names, targets, size=a.size, mode=a.preprocess_mode, train=train,
                          mean=mean, std=std, zip_prefix=zp)


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    outs, fwd = [], 0.0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            o = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd += time.perf_counter() - t0
        outs.append(o.float().cpu())
    return torch.cat(outs).numpy(), fwd


def append_row(csv_path, row):
    header = open(os.path.join(HERE, "configs", "results_template.csv"), encoding="utf-8").readline().strip().split(",")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    run_dir = os.path.join(a.out_dir, a.exp_id)
    os.makedirs(run_dir, exist_ok=True)

    # ---- data -------------------------------------------------------------------------
    cats, names, targets = load_split(a.train_dir, "train")
    vcats, vnames, vtargets = load_split(a.val_dir, "val")
    assert cats == vcats, "train and val category lists differ"
    index = {n: i for i, n in enumerate(names)}
    inner = read_names(os.path.join(a.splits_dir, "train_inner.txt"))
    hold = read_names(os.path.join(a.splits_dir, "holdout.txt"))
    if a.max_train:
        inner = inner[:a.max_train]
    if a.max_eval:
        hold, vnames, vtargets = hold[:a.max_eval], vnames[:a.max_eval], vtargets[:a.max_eval]
    t_in = targets[[index[n] for n in inner]]
    t_ho = targets[[index[n] for n in hold]]

    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if a.no_pretrained else timm_mean_std(a.model)
    ds_train = make_ds(a.train_dir, "train", inner, t_in, a, mean, std, True)
    ds_hold = make_ds(a.train_dir, "train", hold, t_ho, a, mean, std, False)
    ds_val = make_ds(a.val_dir, "val", vnames, vtargets, a, mean, std, False)
    g = torch.Generator().manual_seed(a.seed)
    kw = dict(num_workers=a.num_workers, pin_memory=device.type == "cuda", persistent_workers=a.num_workers > 0)
    train_loader = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True, generator=g, **kw)
    hold_loader = DataLoader(ds_hold, batch_size=a.batch_size, **kw)
    val_loader = DataLoader(ds_val, batch_size=a.batch_size, **kw)

    # ---- model ------------------------------------------------------------------------
    model = build_model(a.model, len(cats), pooling=a.pooling, head=a.head, pretrained=not a.no_pretrained,
                        img_size=a.size if a.set_img_size else None, mlp_hidden=a.mlp_hidden,
                        query_dim=a.query_dim, query_heads=a.query_heads, head_dropout=a.head_dropout,
                        drop_path=a.drop_path).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= PARAM_LIMIT, f"{n_params:,} parameters exceed the 50M limit"
    loss_fn = build_loss(a.loss, a.asl_gamma_neg, a.asl_gamma_pos, a.asl_clip)
    opts = build_optimizers(model, a.optimizer, a.lr_backbone, a.lr_head, a.weight_decay,
                            muon_momentum=a.muon_momentum, muon_head=a.muon_head, muon_lr_scale=a.muon_lr_scale,
                            layer_decay=a.layer_decay or None)
    opt_desc = describe(opts)   # before the schedulers, which scale lr by the first warmup factor
    optimizer_args = (f"momentum={a.muon_momentum};lr_scale={a.muon_lr_scale};muon_head={int(a.muon_head)}"
                      if a.optimizer == "muon" else "")
    updates = a.epochs * math.ceil(len(train_loader) / a.accum)
    warm = max(1, int(a.warmup_ratio * updates))
    scheds = build_schedulers(opts, a.scheduler, warm, updates)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    setting = f"pool={a.pooling if a.head != 'query' else '-'} head={a.head} sched={a.scheduler}"
    print(f"[{a.exp_id}] device={device} params={n_params:,} train={len(ds_train)} holdout={len(ds_hold)} "
          f"val={len(ds_val)} mode={a.preprocess_mode} size={a.size} {setting} loss={a.loss}\n"
          f"  optimizer: {opt_desc}", flush=True)

    # ---- train ------------------------------------------------------------------------
    best, best_ep, log, t_start = -1.0, -1, [], time.time()
    for ep in range(a.epochs):
        model.train()
        for o in opts:
            o.zero_grad(set_to_none=True)
        run_loss, seen, t_ep = 0.0, 0, time.time()
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss = loss_fn(out.float(), y)
            scaler.scale(loss / a.accum).backward()
            if (it + 1) % a.accum == 0 or it + 1 == len(train_loader):
                for o in opts:          # GradScaler unscales each optimizer's grads before its step
                    scaler.step(o)
                scaler.update()
                for o in opts:
                    o.zero_grad(set_to_none=True)
                for s in scheds:
                    s.step()
            run_loss += loss.item() * len(x)
            seen += len(x)
        logits, _ = predict(model, hold_loader, device, use_amp)
        m = compute_metrics(t_ho, binarize(sigmoid(logits), a.threshold))
        log.append({"epoch": ep + 1, "train_loss": run_loss / seen, "holdout_micro": m["micro_f1"],
                    "holdout_macro": m["macro_f1"], "holdout_image": m["image_f1"], "sec": time.time() - t_ep})
        print(f"  epoch {ep + 1:02d} loss {run_loss / seen:.4f} holdout micro {m['micro_f1']:.4f} "
              f"macro {m['macro_f1']:.4f} image {m['image_f1']:.4f}  ({time.time() - t_ep:.0f}s)", flush=True)
        if m["macro_f1"] > best:
            best, best_ep = m["macro_f1"], ep + 1
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
        if ep + 1 - best_ep >= a.patience:
            print(f"  early stop (no holdout macro-F1 gain for {a.patience} epochs)", flush=True)
            break
    minutes = (time.time() - t_start) / 60

    # ---- final evaluation with the best epoch ----------------------------------------
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    lg_ho, _ = predict(model, hold_loader, device, use_amp)
    lg_val, fwd = predict(model, val_loader, device, use_amp)
    save_predictions(os.path.join(run_dir, "preds_holdout.npz"), hold, t_ho, lg_ho, cats)
    save_predictions(os.path.join(run_dir, "preds_val.npz"), vnames, vtargets, lg_val, cats)
    m_ho = compute_metrics(t_ho, binarize(sigmoid(lg_ho), a.threshold))
    m_val = compute_metrics(vtargets, binarize(sigmoid(lg_val), a.threshold))
    print(f"  best epoch {best_ep}: holdout macro {m_ho['macro_f1']:.4f} | official val micro {m_val['micro_f1']:.4f} "
          f"macro {m_val['macro_f1']:.4f} image {m_val['image_f1']:.4f}", flush=True)

    json.dump({"args": vars(a), "best_epoch": best_ep, "params": n_params, "log": log},
              open(os.path.join(run_dir, "log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    append_row(a.results_csv, {
        "exp_id": a.exp_id, "owner": a.owner, "date": date.today().isoformat(), "model": a.model, "loss": a.loss,
        "optimizer": a.optimizer, "pooling": a.pooling if a.head != "query" else "-", "head": a.head,
        "scheduler": a.scheduler, "optimizer_args": optimizer_args, "layer_decay": a.layer_decay,
        "drop_path": a.drop_path, "lr_backbone": a.lr_backbone, "lr_head": a.lr_head, "weight_decay": a.weight_decay,
        "epochs_run": len(log), "batch_size": a.batch_size, "effective_batch": a.batch_size * a.accum, "seed": a.seed,
        "input_size": a.size, "preprocess_mode": a.preprocess_mode, "threshold": a.threshold, "params_total": n_params,
        "holdout_micro_f1": f"{m_ho['micro_f1']:.4f}", "holdout_macro_f1": f"{m_ho['macro_f1']:.4f}",
        "holdout_image_f1": f"{m_ho['image_f1']:.4f}", "val_micro_f1": f"{m_val['micro_f1']:.4f}",
        "val_macro_f1": f"{m_val['macro_f1']:.4f}", "val_image_f1": f"{m_val['image_f1']:.4f}",
        "val_mean_pred_labels": f"{m_val['mean_pred_labels']:.2f}", "val_empty_pred_rate": f"{m_val['empty_pred_rate']:.3f}",
        "train_minutes": f"{minutes:.1f}", "infer_ms_per_img": f"{1000 * fwd / len(ds_val):.1f}",
        "max_vram_gb": f"{torch.cuda.max_memory_allocated() / 1e9:.1f}" if device.type == "cuda" else "",
        "torch": torch.__version__, "timm": timm.__version__, "cuda": torch.version.cuda or "cpu",
        "checkpoint": os.path.join(run_dir, "best.pt"), "pred_file": os.path.join(run_dir, "preds_val.npz"),
        "notes": a.notes + (" [no pretrained, smoke test]" if a.no_pretrained else "")})
    return run_dir


if __name__ == "__main__":
    main()
