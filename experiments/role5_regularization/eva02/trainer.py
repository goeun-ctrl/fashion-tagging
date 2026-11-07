"""Role 3 EVA02 common-data trainer (experiments/role3_eva02/retrain_common_data/trainer.py) with the Role 5
regularization options of reg.py. Changes against the Role 3 file are marked `# role5:`.
Data plan, masked BCE, preprocessing, AMP step rule, schedule, early stopping and metrics are the Role 3 ones."""
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision
from PIL import Image

from experiments.role3_eva02.retrain_common_data import ROLE, require
from experiments.role3_eva02.retrain_common_data.batches import PlannedDataset, eval_dataset, loader, train_epoch
from experiments.role3_eva02.retrain_common_data.package import load_plan
from experiments.role3_eva02.retrain_common_data.preflight import CFG, MEAN, STD
from experiments.role3_eva02.retrain_common_data.trainer import (complete, copy_best, exclusive_run,
                                                                  publish_progress, write_table)
from experiments.role3_eva02 import run as baseline
from data_check import write_json
from fashion_tagging.evaluation.metrics import save_predictions
from fashion_tagging.training.optim import build_schedulers
from train import set_seed  # scripts/train.py

from . import reg

RUN_ROOT = Path(__file__).resolve().parent / "private"   # role5: own output root (git-ignored)


def resolved(root, shared, options, lr_backbone, smoke):
    steps = math.ceil(shared["draws"] / CFG["effective_batch"]) * CFG["max_epochs"]
    cfg = {**CFG, "experiment": "role5_eva02_regularization", "run_id": root.name,
           "shared": shared, "label_mode": shared["label_mode"],
           "loss": "BCEWithLogitsLoss" if shared["label_mode"] == "complete" else "masked BCEWithLogitsLoss",
           "accumulation": CFG["effective_batch"] // CFG["microbatch"], "mean": list(MEAN), "std": list(STD),
           "optimizer": "AdamW", "amp": "cuda-fp16", "planned_steps": steps,
           "scheduler": "cosine", "warmup_steps": max(1, int(steps * CFG["warmup_ratio"])),
           "selection": "strictly higher holdout macro F1",
           "initialization": "official backbone + fresh head",
           # role5:
           "lr_backbone": lr_backbone, "reg": options, "role3_equivalent": reg.is_off(options, lr_backbone),
           "evaluated_weights": "EMA" if options["ema"] else "trained model",
           "smoke": smoke}
    if smoke:
        cfg["max_epochs"] = 1
    return cfg


def read_checkpoint(root, name, cfg):
    state = torch.load(root / name, map_location="cpu", weights_only=False)
    require(state.get("config") == cfg, "checkpoint is from a different experiment, run or package")
    return state


def restore_progress(root, cfg, *, require_complete=False):
    last = read_checkpoint(root, "last.pt", cfg)
    require(last["epoch"] == len(last["history"]) and 1 <= last["best_epoch"] <= last["epoch"],
            "checkpoint epoch/history mismatch")
    if require_complete:
        require(complete(last["epoch"], last["best_epoch"], cfg), "training must finish before final evaluation")
    if last["epoch"] == last["best_epoch"]:
        copy_best(root)
    else:
        require((root / "best.pt").is_file(), "earlier best checkpoint is missing")
    publish_progress(root, last["history"], last["exposure"])
    return last


def smoke_plan(plan, smoke):
    """role5: --smoke N keeps the first N published draws plus the first 4 external draws."""
    if not smoke:
        return plan
    keep = np.array(sorted(set(np.flatnonzero(plan[0] >= 43197)[:4].tolist()) | set(range(smoke))))
    return tuple(a[keep] for a in plan)


def smoke_holdout(group, smoke):
    root, names, targets = group
    return (root, names[:64], targets[:64]) if smoke else group


def train(root, inputs, data, cfg, resume):
    set_seed(cfg["seed"])
    opts = cfg["reg"]
    model = (reg.architecture(cfg["input_size"], opts["drop_path"]) if resume else             # role5: drop path
             reg.from_pretrained(cfg["input_size"], inputs.weights, opts["drop_path"])).cuda()
    opt = reg.optimizer(model, opts, cfg["lr_backbone"], cfg["lr_head"], cfg["weight_decay"])  # role5
    sched, = build_schedulers([opt], "cosine", cfg["warmup_steps"], cfg["planned_steps"])
    if opts["ema"]:
        sched = reg.EmaScheduler(sched, model, opts["ema"])                                   # role5: EMA
    scaler = torch.amp.GradScaler("cuda")
    history, exposure, best, best_epoch = [], [], -1., 0
    if resume:
        last = restore_progress(root, cfg)
        model.load_state_dict(last["model"], strict=True)
        opt.load_state_dict(last["optimizer"])
        sched.load_state_dict(last["scheduler"])
        if opts["ema"]:
            sched.ema.module.load_state_dict(last["ema"], strict=True)
        scaler.load_state_dict(last["scaler"])
        torch.set_rng_state(last["torch_rng"])
        torch.cuda.set_rng_state(last["cuda_rng"])
        random.setstate(last["python_rng"])
        np.random.set_state(last["numpy_rng"])
        history, exposure, best, best_epoch = last["history"], last["exposure"], last["best_score"], last["best_epoch"]
        del last
    eval_model = sched.ema.module if opts["ema"] else model                                   # role5
    paths = data["image_hashes_duplicates_leakage"]
    _, targets = data["manifest_targets_label_mode"]
    probabilities = data["weights_counts_augmentation"][0]
    shared = cfg["shared"]
    holdout = loader(eval_dataset(smoke_holdout(data["original_splits_labels"]["holdout"], cfg["smoke"])),
                     cfg["workers"])
    for epoch in range(len(history) + 1, cfg["max_epochs"] + 1):
        if complete(epoch - 1, best_epoch, cfg):
            break
        plan = smoke_plan(load_plan(inputs.package, shared, epoch, probabilities), cfg["smoke"])
        draws = PlannedDataset(paths, targets, plan)
        row, positives, known = train_epoch(model, loader(draws, cfg["workers"]), opt, sched, scaler,
                                           cfg["label_mode"], cfg["accumulation"])
        require(row["optimizer_steps"] > 0, f"epoch {epoch}: zero optimizer steps; check known labels and AMP skips")
        _, _, scores, bce, _ = baseline.measure(eval_model, holdout, cfg)
        macro = scores["macro_f1"]
        body_lr, head_lr = reg.lr_report(opt)
        history.append({"epoch": epoch, **row, "holdout_bce": bce, "holdout_micro": scores["micro_f1"],
                        "holdout_macro": macro, "holdout_image": scores["image_f1"],
                        "backbone_lr_end": body_lr, "head_lr_end": head_lr})
        exposure.extend({"epoch": epoch, "class": c, "draws": len(draws), "positive": p,
                         "known": k, "unknown": len(draws) - k}
                        for c, p, k in zip(shared["categories"], positives, known, strict=True))
        if macro > best:
            best, best_epoch = macro, epoch
        state = {"epoch": epoch, "best_epoch": best_epoch, "best_score": best, "config": cfg,
                 "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                 "ema": sched.ema.module.state_dict() if opts["ema"] else None,                # role5
                 "scaler": scaler.state_dict(), "history": history, "exposure": exposure,
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
        baseline.save(root / "last.pt", state)
        if epoch == best_epoch:
            copy_best(root)
        publish_progress(root, history, exposure)
        print(f"epoch {epoch}: holdout macro {macro:.6f} bce {bce:.4f}, best epoch {best_epoch}", flush=True)


def evaluate(root, data, cfg, splits):
    last = restore_progress(root, cfg, require_complete=True)
    epoch = last["best_epoch"]
    best_score = last["best_score"]
    del last
    best = read_checkpoint(root, "best.pt", cfg)
    require(best["epoch"] == epoch and best["best_score"] == best_score, "incorrect best checkpoint")
    model = reg.architecture(cfg["input_size"], cfg["reg"]["drop_path"])
    model.load_state_dict(best["ema"] if cfg["reg"]["ema"] else best["model"], strict=True)   # role5
    del best
    model.cuda()
    cats = cfg["shared"]["categories"]
    for split in splits:
        group = data["original_splits_labels"][split]
        if cfg["smoke"]:
            if split != "holdout":
                continue                                    # role5: smoke never touches official val
            group = smoke_holdout(group, cfg["smoke"])
        ds = eval_dataset(group)
        logits, _, scores, loss, timing = baseline.measure(model, loader(ds, cfg["workers"]), cfg)
        if split == "holdout":
            require(abs(scores["macro_f1"] - best_score) < 1e-9, "best holdout score not reproduced")
        tmp = root / "preds.tmp.npz"
        save_predictions(tmp, ds.names, ds.targets, logits, cats)
        tmp.replace(root / f"preds_{split}.npz")
        per_class = scores.pop("per_class")
        write_table(root / f"per_class_{split}.csv", [{"class": c, **{k: v[i].item() for k, v in per_class.items()}}
                                                      for i, c in enumerate(cats)])
        old = json.loads((ROLE / f"metrics_{split}.json").read_text(encoding="utf-8"))
        delta = {k: scores[k] - old[k] for k in ("micro_f1", "macro_f1", "image_f1")}
        write_json(root / f"metrics_{split}.json", {**scores, "bce": loss, **timing, "best_epoch": epoch,
                   "threshold": cfg["threshold"], "label_mode": cfg["label_mode"], "reg": cfg["reg"],
                   "lr_backbone": cfg["lr_backbone"], "delta_vs_eva_baseline": delta})


def execute(args, inputs, audit, data, options):
    require(torch.cuda.is_available(), "CUDA is required for server training/evaluation")
    root = RUN_ROOT / args.run_id
    require(not args.smoke or "smoke" in args.run_id, "smoke run IDs must contain 'smoke'")
    require(args.smoke or "smoke" not in args.run_id, "full run IDs must not contain 'smoke'")
    cfg = resolved(root, audit["shared"], options, args.lr_backbone, args.smoke)
    fresh = args.command == "train" and not args.resume
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root, fresh):
        if fresh:
            write_json(root / "resolved_config.json", cfg)
            write_json(root / "data_audit.json", audit)
            write_json(root / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
                       "torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__,
                       "numpy": np.__version__, "pillow": Image.__version__, "cuda_runtime": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(), "cudnn": torch.backends.cudnn.version()})
        else:
            require(json.loads((root / "resolved_config.json").read_text(encoding="utf-8")) == cfg,
                    "run config differs from current preflight")
        if args.command == "train":
            train(root, inputs, data, cfg, args.resume)
        splits = ("holdout", "val") if args.command == "train" or args.split == "both" else (args.split,)
        evaluate(root, data, cfg, splits)
