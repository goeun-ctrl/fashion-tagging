"""Redesign B training: segmentation-supervised tagging (plan: docs/plan/재설계B_구현계획.md).

    python scripts/train_segb.py --exp_id B_base_s42 --data D --prep D/segb_prep               # main (24 epochs)
    python scripts/train_segb.py --exp_id B_smoke --data D --prep D/segb_prep --max_updates 200 --eval_limit 64

Batch mix per epoch: every Fashionpedia train_inner image once (80%) + COCO person photos (10%) + external (5%)
+ empty photos (5%). Loss = tag BCE (+ teacher KD on Fashionpedia) + lam * (seg CE + seg Dice)
+ part_w * lam * (part BCE + part Dice). Tags come only from the segmentation map (ratio top-k).
Outputs in <out_dir>/<exp_id>/: best.pt (EMA, best holdout macro F1 @0.5), last.pt, history.csv,
preds_{holdout,val,neg_holdout,crowd}.npz (logits = logit of the pooled probability), summary.json.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_hires import BucketBatchSampler, canvas, lr_factor, oriented_portrait, seed_worker, set_seed  # noqa: E402
from fashion_tagging.data.labels import load_split  # noqa: E402
from fashion_tagging.data.preprocess import color_stats, timm_mean_std  # noqa: E402
from fashion_tagging.evaluation.metrics import compute_metrics, save_predictions  # noqa: E402
from fashion_tagging.segb.data import KIND_ID, SegBDataset, collate  # noqa: E402
from fashion_tagging.segb.losses import group_weights, part_loss, region_ce, seg_ce, seg_dice, tag_loss  # noqa: E402
from fashion_tagging.segb.model import SegB  # noqa: E402

PARAM_LIMIT = 50_000_000


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--model", default="convnext_tiny.dinov3_lvd1689m")
    # loss (user decisions)
    ap.add_argument("--lam", type=float, default=1.0, help="seg loss weight vs tag loss")
    ap.add_argument("--coco_seg_w", type=float, default=1.0, help="extra seg weight on COCO person photos")
    ap.add_argument("--part_w", type=float, default=0.5, help="part loss = part_w x lam x (BCE + Dice)")
    ap.add_argument("--cw_cap", type=float, default=10.0, help="class weight = sqrt(n_max / n_c), capped")
    ap.add_argument("--none_w", type=float, default=1.0)
    ap.add_argument("--distill_alpha", type=float, default=0.5)
    ap.add_argument("--distill_shift", type=float, default=0.3)
    ap.add_argument("--ratio_mult", type=float, default=0.5, help="top-k ratio = mult x 10th-percentile class area")
    ap.add_argument("--min_k", type=int, default=4)
    # region consistency (option B, user 2026-11-23): mean group logits inside each GT class region -> CE
    ap.add_argument("--rc_w", type=float, default=0.0, help="region-consistency weight vs seg loss (0 = off)")
    ap.add_argument("--rc_groups", type=int, nargs="+", default=[0, 1], help="spec.GROUPS indices (0 upper, 1 lower)")
    ap.add_argument("--rc_thr", type=float, default=0.5, help="region = soft target >= this (no boundary cells)")
    ap.add_argument("--rc_min", type=int, default=16, help="regions smaller than this many cells are skipped")
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--kd_point", type=float, default=0.3,
                    help="inside-person KD: teacher prob x score divided by this (clipped at 1); 0 = raw")
    ap.add_argument("--kd_min", type=float, default=0.3,
                    help="person cells whose raw teacher sum over 27 classes is below this are ignored; 0 = off")
    # data mix
    ap.add_argument("--coco_ratio", type=float, default=0.10)
    ap.add_argument("--ext_ratio", type=float, default=0.05)
    ap.add_argument("--empty_ratio", type=float, default=0.05)
    # augmentation (0 = off; short experiments turn one on)
    ap.add_argument("--gray_p", type=float, default=0.0)
    ap.add_argument("--old_p", type=float, default=0.0)
    ap.add_argument("--mosaic_p", type=float, default=0.0)
    ap.add_argument("--mosaic_sizes", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--zoom_p", type=float, default=0.0)
    ap.add_argument("--zoom_min", type=float, default=0.35)
    ap.add_argument("--flip", type=float, default=0.5)
    ap.add_argument("--ext_color_jitter", type=float, default=0.2)
    # optimisation
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--lr_backbone", type=float, default=3e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9995, help="0 = off")
    ap.add_argument("--warmup_iters", type=int, default=2000)
    ap.add_argument("--warmup_ratio", type=float, default=0.001)
    ap.add_argument("--milestones", type=float, nargs="*", default=[2 / 3, 11 / 12])
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--short_levels", type=int, nargs="+", default=[416, 544, 672, 800])
    ap.add_argument("--test_short", type=int, default=800)
    ap.add_argument("--max_long", type=int, default=1333)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")
    # smoke / debugging
    ap.add_argument("--max_updates", type=int, default=0, help="stop after this many updates (smoke)")
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    return ap.parse_args(argv)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def average_precision(y, s):
    o = np.argsort(-s)
    y = y[o]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def micro_ap(Y, S):
    return average_precision(Y.reshape(-1).astype(bool), S.reshape(-1))


def class_stats(data):
    """Image count and 10th-percentile area fraction per class over train_inner (Fashionpedia GT)."""
    d = json.load(open(os.path.join(data, "train_inner", "fp_train_inner_ann.json")))
    ims = {im["id"]: im["width"] * im["height"] for im in d["images"]}
    area = {}
    for a in d["annotations"]:
        c = a["category_id"]
        if c < 27:
            area.setdefault(c, {}).setdefault(a["image_id"], 0.0)
            area[c][a["image_id"]] += a["area"] / ims[a["image_id"]]
    n = np.array([len(area.get(c, {})) for c in range(27)], np.float64)
    p10 = np.array([np.percentile(list(area[c].values()), 10) for c in range(27)])
    return n, p10


def load_teacher(npz, names, shift):
    z = np.load(npz, allow_pickle=False)
    idx = {n: i for i, n in enumerate(z["filenames"].tolist())}
    s = np.clip(z["scores"].astype(np.float64), 1e-4, 1 - 1e-4)
    if shift:
        lg = np.log(s / (1 - s)) - math.log(shift / (1 - shift))
        s = 1 / (1 + np.exp(-lg))
    out = np.full((len(names), 27), np.nan, np.float32)
    hit = [k for k, n in enumerate(names) if n in idx]
    out[hit] = s[[idx[names[k]] for k in hit], :27]
    return out, len(hit)


@torch.no_grad()
def predict(model, loader, device, use_amp, n):
    model.eval()
    out = np.zeros((n, 27), np.float32)
    seen = np.zeros(n, bool)
    for b in loader:
        x = b["x"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            o = model(x, b["sizes"].to(device))
        out[b["idx"].numpy()] = o["tags"].float().cpu().numpy()
        seen[b["idx"].numpy()] = True
    assert seen.all()
    return out


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed)
    D = a.data
    out = os.path.join(a.out_dir or os.path.join(os.path.dirname(D), "runs_segb"), a.exp_id)
    assert not os.path.exists(os.path.join(out, "summary.json")), f"{out} finished; use a new --exp_id"
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda")
    use_amp = not a.no_amp

    # ---------------- training lists (built and leak-checked by segb_prep.py lists)
    rl = lambda f: [l.strip() for l in open(os.path.join(a.prep, f)) if l.strip()]
    fp_names, coco, empty, ext = rl("list_fp.txt"), rl("list_coco.txt"), rl("list_empty.txt"), rl("list_ext.txt")
    cats, all_names, all_t = load_split(os.path.join(D, "meta", "train"), "train")
    row = {n: i for i, n in enumerate(all_names)}
    t_fp = all_t[[row[n] for n in fp_names]].astype(np.float32)
    soft_fp, hit = load_teacher(os.path.join(D, "teacher", "ff_r50_train.npz"), fp_names, a.distill_shift)
    print(f"teacher scores for {hit}/{len(fp_names)} train_inner images", flush=True)
    t_coco = np.full((len(coco), 27), -1.0, np.float32)
    for k, p in enumerate(coco):                       # tie / bag / umbrella: COCO yes -> 1, FashionFormer only -> -1, none -> 0
        z = np.load(os.path.join(a.prep, "coco", os.path.basename(p)[:-4] + ".npz"))
        has, ff = z["coco_has"], z["ff_found"]
        for c in (16, 24, 26):
            t_coco[k, c] = 1.0 if has[c] else (-1.0 if ff[c] else 0.0)
    E = os.path.join(D, "external", "external_merged")
    ext_rows = {r["file_name"]: r for r in csv.DictReader(open(os.path.join(E, "labels.csv"), encoding="utf-8-sig"))}
    t_ext = np.array([[float(ext_rows[f][c]) for c in cats] for f in ext], np.float32)
    ext_paths = [os.path.join(E, "images", f) for f in ext]
    fp_paths = [os.path.join(D, "train_inner", "images", n) for n in fp_names]
    paths = fp_paths + coco + ext_paths + empty
    kinds = ["fp"] * len(fp_paths) + ["coco"] * len(coco) + ["ext"] * len(ext) + ["empty"] * len(empty)
    tags = np.concatenate([t_fp, t_coco, t_ext, np.zeros((len(empty), 27), np.float32)])
    soft = np.concatenate([soft_fp, np.full((len(paths) - len(fp_paths), 27), np.nan, np.float32)])
    n_fp = len(fp_paths)
    share = 1 - a.coco_ratio - a.ext_ratio - a.empty_ratio
    per = {k: round(r / share * n_fp) for k, r in (("coco", a.coco_ratio), ("ext", a.ext_ratio), ("empty", a.empty_ratio))}
    s0, s1, s2 = n_fp, n_fp + len(coco), n_fp + len(coco) + len(ext)
    pools = [(s0, s1, per["coco"]), (s1, s2, per["ext"]), (s2, len(paths), per["empty"])]

    # ---------------- external colour matching (per source group, as scripts/train.py --extra_color match)
    from PIL import Image
    src = {r["file_name"]: r["group"] for r in csv.DictReader(open(os.path.join(E, "sources.csv")))}
    rng = random.Random(0)
    ref_s, ref_v = color_stats(Image.open(p) for p in rng.sample(fp_paths, 300))
    color = {}
    for g in sorted({src[f] for f in ext}):
        idx = [k for k, f in enumerate(ext) if src[f] == g]
        es, ev = color_stats(Image.open(ext_paths[k]) for k in rng.sample(idx, min(300, len(idx))))
        for k in idx:
            color[s1 + k] = (ref_s / max(es, 1e-6), ref_v / max(ev, 1e-6), a.ext_color_jitter)
        print(f"colour match {g}: sat x{ref_s / es:.2f} val x{ref_v / ev:.2f} ({len(idx)} images)", flush=True)

    # ---------------- class weights and top-k ratios from train_inner
    n_img, p10 = class_stats(D)
    class_w = np.minimum(np.sqrt(n_img.max() / n_img), a.cw_cap)
    ratio = a.ratio_mult * p10
    print("class weights", np.round(class_w, 2).tolist(), flush=True)
    print("top-k ratio (%)", np.round(ratio * 100, 4).tolist(), flush=True)

    mean, std = timm_mean_std(a.model)
    levels = sorted(a.short_levels)
    por = oriented_portrait("/", paths)

    def make_loader(ps, ks, ts, sf, por_, train, bs):
        ds = SegBDataset(ps, ks, ts, sf, a.prep, mean, std, train, a.max_long, a.flip if train else 0.0,
                         gray_p=a.gray_p if train else 0.0, old_p=a.old_p if train else 0.0,
                         mosaic_p=a.mosaic_p if train else 0.0, mosaic_sizes=a.mosaic_sizes,
                         zoom_p=a.zoom_p if train else 0.0, zoom_min=a.zoom_min, band=a.band,
                         color=color if train else None, kd_point=a.kd_point, kd_min=a.kd_min)
        sampler = BucketBatchSampler(por_, levels if train else [a.test_short], bs, a.seed, train,
                                     pools=pools if train else None)

        def coll(batch):
            b0 = batch[0]
            return collate(batch, canvas(b0["level"], b0["portrait"], a.max_long))
        return DataLoader(ds, batch_sampler=sampler, num_workers=a.workers, pin_memory=True, collate_fn=coll,
                          worker_init_fn=seed_worker, persistent_workers=train and a.workers > 0)

    train_loader = make_loader(paths, kinds, tags, soft, por, True, a.batch_size)

    # ---------------- evaluation sets (never trained on)
    hold = [l.strip() for l in open(os.path.join(D, "meta", "splits", "holdout.txt")) if l.strip()]
    vcats, vnames, vt = load_split(os.path.join(D, "val"), "val")
    assert vcats == cats
    nd = os.path.join(D, "neg_data")
    neg_ho = [l.strip().replace("/data/eunjin/DL/neg_data", nd) for l in open(os.path.join(nd, "neg_holdout.txt")) if l.strip()]
    crow = list(csv.DictReader(open(os.path.join(D, "crowd_eval", "labels.csv"), encoding="utf-8-sig")))
    evals = {
        "holdout": ([os.path.join(D, "holdout", "images", n) for n in hold], all_t[[row[n] for n in hold]], hold),
        "val": ([os.path.join(D, "val", "images", n) for n in vnames], vt, vnames),
        "neg_holdout": (neg_ho, np.zeros((len(neg_ho), 27), np.uint8), [os.path.basename(p) for p in neg_ho]),
        "crowd": ([os.path.join(D, "crowd_eval", "images", r["file"]) for r in crow],
                  np.array([[int(r[c]) for c in cats] for r in crow], np.uint8), [r["file"] for r in crow]),
    }
    if a.eval_limit:
        evals = {k: (p[:a.eval_limit], y[:a.eval_limit], n[:a.eval_limit]) for k, (p, y, n) in evals.items()}
    eval_loaders = {k: make_loader(p, ["eval"] * len(p), np.zeros((len(p), 27), np.float32), None,
                                   oriented_portrait("/", p), False, a.eval_batch_size)
                    for k, (p, y, n) in evals.items()}

    # ---------------- model / optimiser
    model = SegB(a.model, True, a.drop_path, ratio=ratio, min_k=a.min_k).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= PARAM_LIMIT, f"{n_params:,} parameters exceed 50M"
    head_ids = {id(p) for p in model.head_parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": a.lr_backbone, "name": "backbone"},
        {"params": model.head_parameters(), "lr": a.lr_head, "name": "decoder"}], weight_decay=a.weight_decay)
    sampler = train_loader.batch_sampler
    per_epoch = [math.ceil(len(sampler.plan(e)) / a.accum) for e in range(1, a.epochs + 1)]
    total = sum(per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, a.warmup_iters, a.warmup_ratio,
                                                                          a.milestones, a.gamma))
    ema = None
    if a.ema:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=a.ema, use_warmup=True)
    eval_model = ema.module if ema else model
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cw = torch.as_tensor(class_w, dtype=torch.float32, device=device)
    gw = group_weights(cw, a.none_w)
    cfg = dict(vars(a), params_total=n_params, total_updates=total, per_epoch=per, n_fp=n_fp, n_coco=len(coco),
               n_ext=len(ext), n_empty=len(empty), class_w=class_w.tolist(), ratio=ratio.tolist(),
               torch=str(torch.__version__))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"[{a.exp_id}] params {n_params:,} | per epoch: fp {n_fp} + {per} | updates {total}", flush=True)

    history, best, best_epoch, step, skipped = [], -1.0, 0, 0, 0
    t_start = time.perf_counter()
    stop = False
    for epoch in range(1, a.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        n_batches = len(sampler)
        opt.zero_grad(set_to_none=True)
        acc = {k: 0.0 for k in ("loss", "tag", "ce", "dice", "part", "rc")}
        seen, t0, tl = 0, time.perf_counter(), time.perf_counter()
        for bi, b in enumerate(train_loader):
            x = b["x"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            g = {k: b[k].to(device, non_blocking=True) for k in ("y", "sizes", "kind", "soft", "t", "v", "hard", "pt", "pv")}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                o = model(x, g["sizes"])
            img_w = torch.where(g["kind"] == KIND_ID["coco"], torch.full_like(g["y"][:, 0], a.coco_seg_w),
                                torch.ones_like(g["y"][:, 0]))
            l_tag = tag_loss(o["tags"], g["y"], g["soft"], cw, a.distill_alpha)
            l_ce = seg_ce(o["logp"], g["t"], g["v"], gw, img_w)
            l_dice = seg_dice(o["logp"], g["t"], g["hard"], img_w)
            l_part = part_loss(o["part"], g["pt"], g["pv"])
            l_rc = region_ce(o["seg"], g["t"], g["kind"] == KIND_ID["fp"], a.rc_groups, cw, a.rc_thr, a.rc_min) \
                if a.rc_w else o["seg"].sum() * 0.0
            loss = l_tag + a.lam * (l_ce + l_dice + a.rc_w * l_rc) + a.part_w * a.lam * l_part
            assert torch.isfinite(loss), f"non-finite loss: tag {l_tag.item()} ce {l_ce.item()} dice {l_dice.item()} part {l_part.item()} rc {l_rc.item()}"
            scaler.scale(loss / a.accum).backward()
            for k, v in (("loss", loss), ("tag", l_tag), ("ce", l_ce), ("dice", l_dice), ("part", l_part), ("rc", l_rc)):
                acc[k] += float(v.item()) * len(x)
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
                    skipped += 1
                    continue
                sched.step()
                step += 1
                if ema:
                    ema.update(model, step=step)
                if a.log_every and step % a.log_every == 0:
                    dt = time.perf_counter() - tl
                    tl = time.perf_counter()
                    print(f"  upd {step}: " + " ".join(f"{k} {v / seen:.4f}" for k, v in acc.items())
                          + f" | {dt / a.log_every:.2f} s/upd, mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
                if a.max_updates and step >= a.max_updates:
                    stop = True
                    break
        train_sec = time.perf_counter() - t0
        ph = predict(eval_model, eval_loaders["holdout"], device, use_amp, len(evals["holdout"][0]))
        yh = evals["holdout"][1]
        m = compute_metrics(yh, ph >= 0.5)
        rowh = dict(epoch=epoch, **{f"train_{k}": v / max(1, seen) for k, v in acc.items()},
                    holdout_micro=m["micro_f1"], holdout_macro=m["macro_f1"], holdout_image=m["image_f1"],
                    holdout_microAP=micro_ap(yh, ph), lr_backbone=opt.param_groups[0]["lr"],
                    lr_decoder=opt.param_groups[1]["lr"], updates=step, amp_skipped=skipped, train_sec=round(train_sec, 1))
        history.append(rowh)
        state = {"model": eval_model.state_dict(), "epoch": epoch, "config": cfg, "categories": cats}
        torch.save(state, os.path.join(out, "last.pt"))
        if m["macro_f1"] > best:
            best, best_epoch = m["macro_f1"], epoch
            torch.save(state, os.path.join(out, "best.pt"))
        with open(os.path.join(out, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rowh)); w.writeheader(); w.writerows(history)
        print(f"epoch {epoch}: loss {rowh['train_loss']:.4f} holdout macro {m['macro_f1']:.4f} micro {m['micro_f1']:.4f} "
              f"microAP {rowh['holdout_microAP']:.4f} best {best_epoch} ({train_sec / 60:.1f} min)", flush=True)
        if stop:
            break

    eval_model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device)["model"])
    summary = {"exp_id": a.exp_id, "best_epoch": best_epoch, "epochs_run": len(history), "params_total": n_params,
               "train_minutes": round((time.perf_counter() - t_start) / 60, 1), "amp_skipped": skipped}
    for split, (ps, y, names) in evals.items():
        t0 = time.perf_counter()
        p = predict(eval_model, eval_loaders[split], device, use_amp, len(ps))
        save_predictions(os.path.join(out, f"preds_{split}.npz"), names, y, logit(p), cats)
        pr = p >= 0.5
        if split == "neg_holdout":
            summary[split] = {"empty": float((~pr.any(1)).mean()), "pred_per_image": float(pr.sum(1).mean())}
            continue
        mm = compute_metrics(y, pr)
        summary[split] = {k: mm[k] for k in ("micro_f1", "macro_f1", "image_f1")}
        summary[split]["microAP"] = micro_ap(y, p)
        summary[f"{split}_ms_per_image"] = 1000 * (time.perf_counter() - t0) / len(ps)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
