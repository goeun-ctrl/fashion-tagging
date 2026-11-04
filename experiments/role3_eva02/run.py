"""Run the fixed Role 3 baseline. Images, state and per-image outputs stay local."""
import argparse
import csv
import gc
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import timm
from torch.utils.data import DataLoader

from data_check import HERE, FashionDataset, require, verified_data, write_json
from eva import CFG, architecture, describe, from_pretrained
from labels import to_names
from metrics import binarize, evaluate_probs, sigmoid
from optim import build_optimizers, build_schedulers
from preprocess import preprocess, timm_mean_std
from train_eval import predict, set_seed
from PIL import Image

RUN = HERE / "private" / "baseline"
OUT = HERE
LOSS = torch.nn.BCEWithLogitsLoss()


def save(path, obj):
    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def optimizer(model):
    opt, = build_optimizers(model, "adamw", CFG["lr_backbone"], CFG["lr_head"], CFG["weight_decay"])
    head = {id(p) for p in model.head.parameters()}
    groups = {g["lr"]: {id(p) for p in g["params"]} for g in opt.param_groups}
    require(groups[CFG["lr_head"]] == head and groups[CFG["lr_backbone"]] ==
            {id(p) for p in model.parameters() if p.requires_grad} - head, "incorrect AdamW groups")
    return opt


def update(scaler, opt, scheduler):
    before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    stepped = scaler.get_scale() >= before
    if stepped:
        scheduler.step()
    opt.zero_grad(set_to_none=True)
    return stepped


def smoke(batch, size):
    device = "cuda"
    model = from_pretrained(size).to(device)
    info = describe(model)
    opt = optimizer(model)
    scheduler, = build_schedulers([opt], "cosine", 2, 10)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.randint(2, (batch, 27), device=device).float()
    old = [p.detach().clone() for p in (model.head.weight, model.patch_embed.proj.weight)]
    skipped = 0
    for attempt in range(20):
        with torch.autocast(device, dtype=torch.float16):
            logits = model(x)
        require(logits.shape == y.shape and torch.isfinite(logits).all(), "invalid logits")
        loss = LOSS(logits.float(), y)
        scaler.scale(loss).backward()
        grads = [p.grad for p in (model.head.weight, model.patch_embed.proj.weight)]
        grad_norms = [g.float().norm().item() / scaler.get_scale() for g in grads]
        previous_epoch = scheduler.last_epoch
        stepped = update(scaler, opt, scheduler)
        require(scheduler.last_epoch == previous_epoch + int(stepped), "scheduler/optimizer desynchronized")
        if stepped:
            require(all(math.isfinite(g) and g > 0 for g in grad_norms), "missing finite gradients")
            break
        skipped += 1
    else:
        raise RuntimeError("AMP could not take a finite optimizer step")
    require(all(not torch.equal(old_p, p) for old_p, p in zip(old, (model.head.weight, model.patch_embed.proj.weight))),
            "classifier or backbone did not update")
    model.eval()
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        expected = model(x[:1])
    # Temporary smoke checkpoint in its own local directory, never a best/last artifact.
    path = HERE / "private/smoke_roundtrip.tmp.pt"
    save(path, model.state_dict())
    reload_model = architecture(size)
    reload_model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
    reload_model.to(device).eval()
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        torch.testing.assert_close(reload_model(x[:1]), expected, rtol=0, atol=0)
    path.unlink()
    info.update({"synthetic_batch": batch, "output_shape": list(logits.shape), "synthetic_bce": loss.item(),
                 "head_backbone_gradient_norms": grad_norms, "amp_skipped_before_step": skipped,
                 "strict_roundtrip": True, "optimizer_and_scheduler_step": True})
    return info


def environment():
    write_json(OUT / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "timm": timm.__version__, "numpy": np.__version__,
        "pillow": Image.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None})


def preflight():
    environment()
    require(torch.cuda.is_available(), "CUDA PyTorch + visible GPU required")
    cats, groups = verified_data()
    size = CFG["input_size"]
    mean, std = timm_mean_std(CFG["model"])
    changes = []
    set_seed(CFG["seed"])
    native = describe(from_pretrained(336))
    for batch in (CFG["microbatch"], 8, 4):
        try:
            actual = smoke(batch, size)
            break
        except torch.cuda.OutOfMemoryError:
            changes.append(f"CUDA OOM at microbatch {batch}")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("microbatch 4 does not fit in GPU memory")
    for split, (root, names, y) in groups.items():
        ds = FashionDataset(root, names, y, size, "pad", split == "train_inner", mean, std)
        x, sample_y = ds[0]
        require(tuple(x.shape) == (3, size, size) and np.isfinite(x.numpy()).all() and
                np.array_equal(sample_y.numpy(), y[0]), "loader/target smoke failed")
    metrics = evaluate_probs([[0, 0], [1, 0]], [[0.1, 0.2], [0.5, 0.1]], 0.5)
    require(metrics["image_f1"] == metrics["micro_f1"] == 1 and metrics["macro_f1"] == 0.5,
            "shared metric convention differs")
    write_json(OUT / "model_parameters.json", {"native_336": native, "selected": actual})
    settings = {**CFG, "microbatch": batch, "accumulation": CFG["effective_batch"] // batch,
                "mean": list(mean), "std": list(std), "amp": "float16", "adjustments": changes,
                "weight_source": "timm/" + CFG["model"], "weight_license": "MIT", "loss": "BCEWithLogitsLoss",
                "optimizer": "AdamW", "decay_scope": "all parameters including norm and bias",
                "scheduler": "cosine", "selection": "strictly higher holdout macro F1; earliest tie",
                "preprocess": "common padded bicubic RGB, EXIF, train-only horizontal flip p=0.5",
                "planned_steps": math.ceil(len(groups["train_inner"][1]) / CFG["effective_batch"]) * CFG["max_epochs"]}
    settings["warmup_steps"] = max(1, int(settings["planned_steps"] * CFG["warmup_ratio"]))
    write_json(OUT / "resolved_config.json", settings)
    print(f"Preflight passed: {len(cats)} classes, input {size}, batch {batch}, {actual['total']:,} parameters", flush=True)
    return settings, cats, groups


def loaders(groups, cfg, generator):
    kw = {"num_workers": cfg["workers"], "pin_memory": True, "batch_size": cfg["microbatch"]}
    loaders_by_split = {}
    for split, (root, names, y) in groups.items():
        train = split == "train_inner"
        ds = FashionDataset(root, names, y, cfg["input_size"], "pad", train, cfg["mean"], cfg["std"])
        loaders_by_split[split] = DataLoader(ds, shuffle=train, generator=generator if train else None, **kw)
    return loaders_by_split


def train_epoch(model, loader, opt, sched, scaler, cfg):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tic = time.perf_counter()
    model.train()
    opt.zero_grad(set_to_none=True)
    total, seen, skipped, steps = 0.0, 0, 0, 0
    for i, (x, y) in enumerate(loader):
        if i % cfg["accumulation"] == 0:
            window = min(cfg["effective_batch"], len(loader.dataset) - seen)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        loss = LOSS(logits.float(), y)
        value = loss.item()
        require(math.isfinite(value), "non-finite train BCE")
        scaler.scale(loss * (len(x) / window)).backward()
        total += value * len(x)
        seen += len(x)
        if (i + 1) % cfg["accumulation"] == 0 or i + 1 == len(loader):
            took_step = update(scaler, opt, sched)
            steps += int(took_step)
            skipped += int(not took_step)
        if (i + 1) % 500 == 0:
            print(f"  {seen}/{len(loader.dataset)} images, BCE {total / seen:.5f}", flush=True)
    torch.cuda.synchronize()
    return {"train_bce": total / seen, "train_seconds": time.perf_counter() - tic,
            "allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_steps": steps, "amp_skips": skipped}


def measure(model, loader, cfg):
    tic = time.perf_counter()
    logits, forward = predict(model, loader, torch.device("cuda"), True)
    wall = time.perf_counter() - tic
    y = loader.dataset.targets
    require(np.isfinite(logits).all(), "non-finite evaluation logits")
    probs = sigmoid(logits)
    m = evaluate_probs(y, probs, cfg["threshold"])
    loss = LOSS(torch.as_tensor(logits).float(), torch.from_numpy(y)).item()
    return logits, probs, m, loss, {"forward_ms_per_image": 1000 * forward / len(y),
                                    "end_to_end_ms_per_image": 1000 * wall / len(y),
                                    "inference_seconds": wall, "forward_seconds": forward}


def history_csv(rows):
    path = OUT / "training_history.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def train(cfg, cats, groups, resume):
    RUN.mkdir(exist_ok=True, parents=True)
    checkpoints = [p for p in (RUN / "last.pt", RUN / "best.pt") if p.exists()]
    require(resume == bool(checkpoints), "existing checkpoint: use --resume; fresh training requires empty run")
    set_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    loaders_by_split = loaders(groups, cfg, generator)
    model = from_pretrained(cfg["input_size"]).cuda()
    opt = optimizer(model)
    sched, = build_schedulers([opt], "cosine", cfg["warmup_steps"], cfg["planned_steps"])
    scaler = torch.amp.GradScaler("cuda")
    rows, best, best_ep = [], -1.0, 0
    if resume:
        state = torch.load(max(checkpoints, key=lambda p: p.stat().st_mtime_ns), weights_only=False, map_location="cpu")
        require(state["config"] == cfg and state["categories"] == cats, "resume settings mismatch")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        rows, best, best_ep = state["history"], state["best_score"], state["best_epoch"]
        del state
        history_csv(rows)
    for ep in range(len(rows) + 1, cfg["max_epochs"] + 1):
        if rows and len(rows) - best_ep >= cfg["patience"]:
            break
        lrs = [g["lr"] for g in opt.param_groups]
        row = train_epoch(model, loaders_by_split["train_inner"], opt, sched, scaler, cfg)
        _, _, m, bce, _ = measure(model, loaders_by_split["holdout"], cfg)
        row = {"epoch": ep, **row, "holdout_bce": bce, "holdout_micro": m["micro_f1"],
               "holdout_macro": m["macro_f1"], "holdout_image": m["image_f1"],
               "backbone_lr_start": lrs[0], "head_lr_start": lrs[1],
               "backbone_lr_end": opt.param_groups[0]["lr"], "head_lr_end": opt.param_groups[1]["lr"]}
        rows.append(row)
        improved = m["macro_f1"] > best
        if improved:
            best, best_ep = m["macro_f1"], ep
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                 "scaler": scaler.state_dict(), "history": rows, "config": cfg, "categories": cats,
                 "best_score": best, "best_epoch": best_ep, "epoch": ep, "generator": generator.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
        if improved:
            save(RUN / "best.pt", state)
        save(RUN / "last.pt", state)
        history_csv(rows)
        print(f"epoch {ep} train {row['train_bce']:.5f} hold {bce:.5f} macro {m['macro_f1']:.6f} "
              f"(best {best:.6f}, epoch {best_ep})", flush=True)
    return rows


def evaluate():
    from analysis import export_split, compare
    require((RUN / "last.pt").is_file(), "final epoch checkpoint unavailable")
    final = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
    cfg, rows, best_ep = final["config"], final["history"], final["best_epoch"]
    require(len(rows) == cfg["max_epochs"] or len(rows) - best_ep >= cfg["patience"], "training incomplete")
    del final
    best = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    require(best["epoch"] == best_ep, "best checkpoint epoch mismatch")
    model = architecture(cfg["input_size"])
    model.load_state_dict(best["model"], strict=True)
    model.cuda()
    cats, groups = verified_data()
    require(cats == best["categories"], "checkpoint category order mismatch")
    del best
    loaders_by_split = loaders(groups, cfg, None)
    history_csv(rows)
    for split in ("holdout", "val"):
        loader = loaders_by_split[split]
        logits, probs, m, loss, timing = measure(model, loader, cfg)
        if split == "holdout":
            require(abs(m["macro_f1"] - rows[best_ep - 1]["holdout_macro"]) < 1e-9, "best holdout not reproduced")
        export_split(split, loader.dataset, cats, logits, probs, m, loss, timing)
    compare(rows)


def infer(path):
    ck = torch.load(RUN / "best.pt", weights_only=False, map_location="cpu")
    cfg = ck["config"]
    model = architecture(cfg["input_size"])
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    with Image.open(path) as image, torch.inference_mode():
        x = preprocess(image, cfg["input_size"], "pad", cfg["mean"], cfg["std"])
        logits = model(torch.from_numpy(x).unsqueeze(0)).numpy()[0]
    print(json.dumps(to_names(binarize(sigmoid(logits), cfg["threshold"]), ck["categories"]), ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preflight", "train", "evaluate", "infer"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--image", type=Path)
    args = ap.parse_args()
    if args.mode == "train":
        if not args.resume and any((RUN / name).exists() for name in ("best.pt", "last.pt")):
            ap.error("Existing baseline checkpoints. Use evaluate for the final result or train --resume after an interruption.")
        config, classes, datasets = preflight()
        train(config, classes, datasets, args.resume)
        evaluate()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "evaluate":
        evaluate()
    else:
        require(args.image is not None, "supply --image PATH")
        infer(args.image)

