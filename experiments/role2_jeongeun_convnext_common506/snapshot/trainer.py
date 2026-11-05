"""Jeongeun's separate ConvNeXt experiment, using the team's unchanged data/metrics.

Run --help for explicit configuration paths. This script really loads config.json;
the team's reference YAML is never silently assumed to affect a training command.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
import traceback

from training_utils import accumulation_window_samples, improved_macro, lr_multiplier, validate_partition
from augmented_data import make_aug_dataset, known_window_count, save_exposure, validate_prepared

HERE = Path(__file__).resolve().parent


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(type(obj).__name__)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def atomic_torch(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=HERE.parent / ".reference" / "fashion-tagging-c5827dfa")
    parser.add_argument("--train-dir", type=Path, required=True, help="extracted train/ or original train ZIP")
    parser.add_argument("--val-dir", type=Path, required=True, help="extracted val/ or original official val ZIP")
    parser.add_argument("--audit-report", type=Path, help="optional successful tools/audit_data.py JSON report")
    parser.add_argument("--output-dir", type=Path, help="exact run folder; smoke runs require 'smoke' in this path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accum", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", nargs="?", const="auto", help="auto or an own trusted checkpoint path")
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate existing best.pt without training")
    parser.add_argument("--smoke", action="store_true", help="separate small run; never evaluates official val")
    parser.add_argument("--smoke-train", type=int, default=64)
    parser.add_argument("--smoke-holdout", type=int, default=32)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true", help="allowed only for a smoke infrastructure test")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


def load_config(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    baseline = {
        "model": "convnext_tiny.fb_in22k_ft_in1k_384", "input_size": 384,
        "num_classes": 27, "head": "gap_linear_27", "input_color": "RGB",
        "preprocess_mode": "pad", "resize_interp": "bicubic", "pad_color": [0, 0, 0],
        "mean_std": "from_timm", "augmentation": {'hflip_p': 0.5, 'color_jitter': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.05, 'hue': 0, 'high_p': 0.35, 'general_p': 0.2}},
        "loss": "bce", "optimizer": "adamw", "lr_backbone": 1e-4,
        "lr_head": 1e-3, "weight_decay": 0.05, "scheduler": "cosine",
        "warmup_ratio": 0.05, "seed": 42, "threshold": 0.5,
        "early_stopping": {"patience": 5, "metric": "macro_f1", "on": "holdout"},
    }
    for key, expected in baseline.items():
        if cfg.get(key) != expected:
            raise ValueError(f"baseline contract mismatch: {key}={cfg.get(key)!r}; expected {expected!r}")
    if not args.smoke and (cfg.get("epochs") != 30 or cfg.get("pretrained") is not True or args.no_pretrained):
        raise ValueError("full baseline requires pretrained=True and maximum 30 epochs")
    for arg, key in ((args.batch_size, "batch_size"), (args.accum, "accum"), (args.workers, "num_workers")):
        if arg is not None:
            cfg[key] = arg
    if cfg["batch_size"] < 1 or cfg["accum"] < 1 or cfg["num_workers"] < 0:
        raise ValueError("invalid batch/accumulation/worker count")
    if cfg["batch_size"] * cfg["accum"] != 32:
        raise ValueError("effective batch must remain 32 (e.g. batch16, accum2)")
    cfg["run_kind"] = "smoke" if args.smoke else "external_augmented_v1"
    cfg["base_exp_id"] = cfg["exp_id"]
    if args.smoke:
        if min(args.smoke_train, args.smoke_holdout, args.smoke_epochs) < 1:
            raise ValueError("smoke counts must be positive")
        cfg["exp_id"] += "__smoke"
        cfg["epochs"] = args.smoke_epochs
        cfg["pretrained"] = not args.no_pretrained
        cfg["notes"] += " [smoke only; excluded from baseline performance; official val not evaluated]"
    cfg["effective_batch"] = cfg["batch_size"] * cfg["accum"]
    return cfg


def initialize_libraries(reference):
    global np, torch, F, timm, DataLoader, FashionDataset, read_names, load_split
    global compute_metrics, binarize, save_predictions, sigmoid, timm_mean_std
    sys.path.insert(0, str(reference / "common"))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    from torch.utils.data import DataLoader
    from dataset import FashionDataset, read_names
    from labels import load_split
    from metrics import compute_metrics, binarize, save_predictions, sigmoid
    from preprocess import timm_mean_std


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_dataset(path, split, names, targets, cfg, mean, std, training):
    return FashionDataset(path, names, targets, size=cfg["input_size"], mode="pad", train=training,
                          mean=mean, std=std, zip_prefix=split if path.suffix.lower() == ".zip" else None)


def make_loader(dataset, cfg, device, generator, training):
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=training, drop_last=False,
                      num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                      persistent_workers=False, generator=generator, worker_init_fn=seed_worker)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict(model, loader, device, use_amp):
    model.eval()
    outputs, loss_sum, samples, forward_seconds = [], 0.0, 0, 0.0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for x, targets in loader:
            x = x.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sync(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            sync(device)
            forward_seconds += time.perf_counter() - started
            loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
            loss_sum += float(loss) * len(x)
            samples += len(x)
            outputs.append(logits.float().cpu().numpy())
    if not samples:
        raise ValueError("cannot evaluate an empty split")
    return np.concatenate(outputs), loss_sum / samples, forward_seconds, time.perf_counter() - wall_start


def record_rng(generators, device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "generators": {key: value.get_state() for key, value in generators.items()}}


def restore_rng(state, generators, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    for key, value in state["generators"].items():
        generators[key].set_state(value.cpu())


def load_checkpoint(path, device):
    # These files are made by this experiment, never third-party pickle downloads.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def export_reports(run_dir, split, dataset, names, targets, logits, categories, threshold, stats):
    predictions = binarize(sigmoid(logits), threshold)
    metrics = compute_metrics(targets, predictions)
    save_predictions(run_dir / f"preds_{split}.npz", names, targets, logits, categories)
    atomic_json(run_dir / f"metrics_{split}.json", {**metrics, **stats})
    per_class = metrics["per_class"]
    rows = [{"class_index": i, "class_name": category,
             **{key: per_class[key][i] for key in ("support", "precision", "recall", "f1", "tp", "fp", "fn")}}
            for i, category in enumerate(categories)]
    atomic_csv(run_dir / f"per_class_{split}.csv", rows,
               ["class_index", "class_name", "support", "precision", "recall", "f1", "tp", "fp", "fn"])
    targets_bool = targets.astype(bool)
    tp = (targets_bool & predictions).sum(1)
    fp = (~targets_bool & predictions).sum(1)
    fn = (targets_bool & ~predictions).sum(1)
    denom = 2 * tp + fp + fn
    image_f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 1.0)
    errors = []
    for index, name in enumerate(names):
        if fp[index] + fn[index] == 0:
            continue
        as_names = lambda mask: [categories[j] for j in np.flatnonzero(mask)]
        errors.append({"index": index, "file_name": name, "image_f1": float(image_f1[index]),
                       "tp": int(tp[index]), "fp": int(fp[index]), "fn": int(fn[index]),
                       "true_labels": json.dumps(as_names(targets_bool[index]), ensure_ascii=False),
                       "predicted_labels": json.dumps(as_names(predictions[index]), ensure_ascii=False),
                       "false_positive_labels": json.dumps(as_names(~targets_bool[index] & predictions[index]), ensure_ascii=False),
                       "false_negative_labels": json.dumps(as_names(targets_bool[index] & ~predictions[index]), ensure_ascii=False)})
    errors.sort(key=lambda row: (row["image_f1"], -row["fp"] - row["fn"], row["file_name"]))
    atomic_csv(run_dir / f"errors_{split}.csv", errors,
               ["index", "file_name", "image_f1", "tp", "fp", "fn", "true_labels", "predicted_labels",
                "false_positive_labels", "false_negative_labels"])
    example_dir = run_dir / "error_examples" / split
    example_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw, ImageOps
    examples = errors[:12]
    if examples:
        width, height, columns = 400, 340, 3
        canvas = Image.new("RGB", (width * columns, height * math.ceil(len(examples) / columns)), "white")
        for rank, error in enumerate(examples):
            card = Image.new("RGB", (width, height), "white")
            image = dataset.load_image(error["file_name"])
            image = ImageOps.contain(image, (width, 240))
            card.paste(image, ((width - image.width) // 2, 0))
            draw = ImageDraw.Draw(card)
            text = [f"{rank + 1}. {error['file_name']}  F1={error['image_f1']:.3f}",
                    "FP: " + error["false_positive_labels"][:72],
                    "FN: " + error["false_negative_labels"][:72]]
            draw.multiline_text((8, 248), "\n".join(text), fill="black", spacing=5)
            card.save(example_dir / f"example_{rank + 1:02d}.jpg", quality=90)
            canvas.paste(card, ((rank % columns) * width, (rank // columns) * height))
        canvas.save(example_dir / "contact_sheet.jpg", quality=90)
    atomic_json(example_dir / "examples.json", {"selection": "lowest image-wise F1, then most FP+FN; reporting only", "examples": examples})
    return metrics


def save_curves(run_dir, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train BCE")
    axes[0].plot(epochs, [row["holdout_loss"] for row in history], label="holdout BCE")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    for key, label in (("holdout_macro_f1", "Macro F1"), ("holdout_micro_f1", "Micro F1"), ("holdout_image_f1", "Image-wise F1")):
        axes[1].plot(epochs, [row[key] for row in history], label=label)
    axes[1].set(xlabel="Epoch", ylabel="Holdout F1", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(run_dir / "learning_curves.png", dpi=180)
    figure.savefig(run_dir / "learning_curves.svg")
    plt.close(figure)


def save_team_csv(run_dir, reference, cfg, summary):
    with (reference / "common" / "configs" / "results_template.csv").open(encoding="utf-8-sig", newline="") as file:
        fields = next(csv.reader(file))
    hold, val = summary["holdout"], summary.get("val")
    try:
        from zoneinfo import ZoneInfo
        run_date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    except Exception:
        run_date = datetime.now(timezone.utc).date().isoformat()
    row = {**cfg, "date": run_date, "epochs_run": summary["epochs_run"],
           "params_total": summary["params_total"], "train_minutes": summary["train_wall_seconds"] / 60,
           "max_vram_gb": summary["train_peak_allocated_bytes"] / 1e9 if summary["device"].startswith("cuda") else "",
           "infer_ms_per_img": summary.get("val_forward_ms_per_image", ""), "torch": str(torch.__version__),
           "timm": timm.__version__, "cuda": torch.version.cuda or "cpu", "checkpoint": str(run_dir / "best.pt"),
           "pred_file": str(run_dir / "preds_val.npz") if val else str(run_dir / "preds_holdout.npz")}
    for split, values in (("holdout", hold), ("val", val)):
        if values:
            for metric in ("micro_f1", "macro_f1", "image_f1"):
                row[f"{split}_{metric}"] = values[metric]
    if val:
        row["val_mean_pred_labels"] = val["mean_pred_labels"]
        row["val_empty_pred_rate"] = val["empty_pred_rate"]
    atomic_csv(run_dir / "results.csv", [row], fields)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args)
    reference = args.reference_dir.expanduser().resolve()
    if not (reference / "common" / "train_eval.py").is_file():
        raise FileNotFoundError("reference common code missing: " + str(reference))
    initialize_libraries(reference)
    run_dir = (args.output_dir or HERE / "runs" / cfg["exp_id"]).expanduser().resolve()
    if args.smoke and "smoke" not in str(run_dir).lower():
        raise ValueError("smoke output directory must include 'smoke' to avoid mixing baseline results")
    if not args.smoke and "smoke" in str(run_dir).lower():
        raise ValueError("baseline output directory cannot be a smoke directory")
    if (run_dir / "resolved_config.json").exists() and not (args.resume or args.evaluate_only):
        raise FileExistsError("run already exists; use --resume auto or a new output directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    start_all = time.perf_counter()
    atomic_json(status_path, {"state": "preparing", "run_kind": cfg["run_kind"], "pid": os.getpid(), "hostname": socket.gethostname()})
    try:
        if args.audit_report:
            audit = json.loads(args.audit_report.read_text(encoding="utf-8-sig"))
            if audit.get("passed") is not True:
                raise ValueError("data audit report does not state passed=true")
        seed_everything(cfg["seed"], cfg["deterministic"])
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; do not silently train the baseline on CPU")
            torch.cuda.set_device(device)
        use_amp = cfg["amp"] and device.type == "cuda"
        train_path, val_path = args.train_dir.expanduser().resolve(), args.val_dir.expanduser().resolve()
        categories, all_names, all_targets = load_split(train_path, "train")
        inner_all = read_names(reference / "common" / "splits" / "train_inner.txt")
        hold_all = read_names(reference / "common" / "splits" / "holdout.txt")
        counts = validate_partition(all_names, inner_all, hold_all)
        if len(categories) != 27:
            raise ValueError("expected exactly 27 classes")
        if not args.smoke and counts != {"all": 45622, "train_inner": 43197, "holdout": 2425}:
            raise ValueError("unexpected fixed split counts: " + repr(counts))
        index = {name: i for i, name in enumerate(all_names)}
        inner = inner_all[:args.smoke_train] if args.smoke else inner_all
        hold = hold_all[:args.smoke_holdout] if args.smoke else hold_all
        train_targets = all_targets[[index[name] for name in inner]]
        hold_targets = all_targets[[index[name] for name in hold]]
        mean, std = timm_mean_std(cfg["model"])
        prepared = args.prepared_dir.expanduser().resolve()
        prepared_info = validate_prepared(prepared, categories, inner_all, all_names, all_targets)
        for split, root in (("train", train_path), ("val", val_path)):
            if sha256(root / "labels.npz") != prepared_info["shared"]["sha256"][split + "/labels.npz"]:
                raise ValueError("official original label hash mismatch: " + split)
        train_dataset = make_aug_dataset(prepared, train_path, 1, cfg, mean, std, args.smoke_train if args.smoke else None)
        hold_dataset = make_dataset(train_path, "train", hold, hold_targets, cfg, mean, std, False)
        generators = {key: torch.Generator().manual_seed(cfg["seed"] + offset) for offset, key in enumerate(("train", "holdout", "val"))}
        train_loader = make_loader(train_dataset, cfg, device, generators["train"], False)
        hold_loader = make_loader(hold_dataset, cfg, device, generators["holdout"], False)
        # No official val labels, loader, or prediction is used to select a checkpoint.
        model = timm.create_model(cfg["model"], pretrained=cfg["pretrained"], num_classes=27, global_pool="avg").to(device)
        classifier = model.get_classifier()
        if not isinstance(classifier, torch.nn.Linear) or classifier.out_features != 27:
            raise ValueError("classifier must be Linear(...,27), producing logits")
        if model.head.global_pool.pool_type != "avg" or not isinstance(model.head.pre_logits, torch.nn.Identity):
            raise ValueError("expected standard GAP head without hidden MLP")
        if getattr(model.head.drop, "p", 0) != 0:
            raise ValueError("baseline head dropout must be zero")
        n_params = sum(param.numel() for param in model.parameters())
        if n_params > cfg["parameter_limit"]:
            raise ValueError(f"parameter limit exceeded: {n_params:,}")
        head_ids = {id(param) for param in classifier.parameters()}
        optimizer = torch.optim.AdamW([
            {"params": [param for param in model.parameters() if id(param) not in head_ids], "lr": cfg["lr_backbone"], "name": "backbone"},
            {"params": list(classifier.parameters()), "lr": cfg["lr_head"], "name": "linear_head"},
        ], weight_decay=cfg["weight_decay"])
        planned_updates = cfg["epochs"] * math.ceil(len(train_loader) / cfg["accum"])
        warmup_updates = max(1, int(cfg["warmup_ratio"] * planned_updates))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, planned_updates, warmup_updates))
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        source_files = ["labels.py", "dataset.py", "preprocess.py", "metrics.py", "train_eval.py", "splits/train_inner.txt", "splits/holdout.txt"]
        versions = {package: importlib.metadata.version(package) for package in ("torch", "torchvision", "timm", "numpy", "Pillow", "matplotlib")}
        resolved = {**cfg, "mean": mean, "std": std, "params_total": n_params,
                    "head_structure": repr(model.head), "head_explanation": "standard pretrained GAP + LayerNorm + Linear(27), no hidden MLP or output sigmoid",
                    "head_parameter_names": [name for name, param in model.named_parameters() if id(param) in head_ids],
                    "reference_dir": str(reference), "train_dir": str(train_path), "val_dir": str(val_path),
                    "split_counts": counts, "used_train_count": len(train_dataset), "used_holdout_count": len(hold),
                    "combined_unique_training_images": prepared_info["combined_unique_images"], "external_images": prepared_info["adopted_external_images"],
                    "prepared_dir": str(prepared), "prepared_fingerprint": prepared_info["fingerprint"],
                    "augmentation_policy": prepared_info["policy"], "label_mode": "partial", "shared": prepared_info["shared"],
                    "initialization": "fresh original ImageNet pretrained initialization; new optimizer and scheduler; baseline checkpoint not resumed",
                    "source_file_sha256": {file: sha256(reference / "common" / file) for file in source_files},
                    "own_code_sha256": {file: sha256(HERE / file) for file in ("trainer.py", "training_utils.py", "config.json", "augmentation.py", "augmented_data.py")},
                    "planned_optimizer_updates": planned_updates, "warmup_updates": warmup_updates,
                    "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "hostname": socket.gethostname(),
                    "python": sys.version, "platform": platform.platform(), "library_versions": versions,
                    "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "argv": sys.argv,
                    "audit_report": str(args.audit_report) if args.audit_report else None,
                    "official_val_policy": "loaded only after holdout-selected best checkpoint; never used in smoke, early stopping or threshold tuning",
                    "accumulation_policy": "sum of BCE over known labels / known label count across the full actual accumulation window; -1 never negative",
                    "scheduler_policy": "5% of planned optimizer updates (floor, minimum1) linear warmup, then per-successful-update cosine; AMP-overflow skips do not advance scheduler"}
        history, best_macro, best_epoch, start_epoch, global_step, skipped_steps = [], -1.0, 0, 0, 0, 0
        train_wall_seconds, holdout_selection_seconds, peak_allocated, peak_reserved = 0.0, 0.0, 0, 0
        resume_path = run_dir / "last.pt" if args.resume == "auto" else Path(args.resume).expanduser().resolve() if args.resume else None
        if resume_path:
            checkpoint = load_checkpoint(resume_path, device)
            previous = checkpoint["config"]
            resume_keys = ("exp_id", "run_kind", "model", "pretrained", "input_size", "seed", "threshold", "batch_size", "accum", "epochs", "source_commit", "used_train_count", "used_holdout_count", "source_file_sha256", "prepared_fingerprint")
            for key in resume_keys:
                if previous.get(key) != resolved.get(key):
                    raise ValueError(f"resume configuration mismatch: {key}")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            history, best_macro, best_epoch = checkpoint["history"], checkpoint["best_macro"], checkpoint["best_epoch"]
            start_epoch, global_step = checkpoint["completed_epoch"], checkpoint["global_step"]
            skipped_steps = checkpoint.get("skipped_steps", 0)
            train_wall_seconds = checkpoint["train_wall_seconds"]
            holdout_selection_seconds = checkpoint["holdout_selection_seconds"]
            peak_allocated, peak_reserved = checkpoint["peak_allocated"], checkpoint["peak_reserved"]
            restore_rng(checkpoint["rng"], generators, device)
            resolved["resumed_from"] = str(resume_path)
        atomic_json(run_dir / "resolved_config.json", resolved)
        atomic_json(run_dir / "library_versions.json", versions)
        (run_dir / "model_structure.txt").write_text(repr(model) + "\n", encoding="utf-8")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print(f"[{cfg['exp_id']}] {cfg['run_kind']} device={device} params={n_params:,} train={len(inner)} holdout={len(hold)} batch={cfg['batch_size']} accum={cfg['accum']} effective=32", flush=True)
        atomic_json(status_path, {"state": "training" if not args.evaluate_only else "final_evaluation", "run_kind": cfg["run_kind"], "pid": os.getpid(), "start_epoch": start_epoch})
        stopped_early = bool(history and start_epoch - best_epoch >= cfg["early_stopping"]["patience"])
        if not args.evaluate_only and not stopped_early:
            for epoch in range(start_epoch, cfg["epochs"]):
                train_dataset = make_aug_dataset(prepared, train_path, epoch + 1, cfg, mean, std, args.smoke_train if args.smoke else None)
                train_loader = make_loader(train_dataset, cfg, device, generators["train"], False)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running_loss, seen = 0.0, 0
                sync(device)
                epoch_start = time.perf_counter()
                epoch_start_step = global_step
                for batch_index, (x, targets) in enumerate(train_loader):
                    x, targets = x.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        logits = model(x)
                    known = targets >= 0
                    per_entry = F.binary_cross_entropy_with_logits(logits.float(), targets.clamp(min=0), reduction="none")
                    loss_sum = (per_entry * known).sum()
                    loss = loss_sum / known.sum().clamp(min=1)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("non-finite train BCE")
                    window_known = known_window_count(train_dataset, cfg["batch_size"], cfg["accum"], batch_index)
                    scaler.scale(loss_sum / window_known).backward()
                    running_loss += float(loss_sum.detach())
                    seen += int(known.sum())
                    if (batch_index + 1) % cfg["accum"] == 0 or batch_index + 1 == len(train_loader):
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if not use_amp or scaler.get_scale() >= old_scale:
                            scheduler.step()
                            global_step += 1
                        else:
                            skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                    if args.log_every and (batch_index + 1) % args.log_every == 0:
                        print(f"epoch={epoch+1} batch={batch_index+1}/{len(train_loader)} train_loss={running_loss/seen:.6f} optimizer_steps={global_step}", flush=True)
                sync(device)
                if global_step == epoch_start_step:
                    raise RuntimeError("zero optimizer updates in epoch; inspect AMP and labels")
                train_seconds = time.perf_counter() - epoch_start
                train_wall_seconds += train_seconds
                if device.type == "cuda":
                    peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
                hold_logits, hold_loss, hold_forward_seconds, hold_wall = predict(model, hold_loader, device, use_amp)
                holdout_selection_seconds += hold_wall
                metric = compute_metrics(hold_targets, binarize(sigmoid(hold_logits), cfg["threshold"]))
                is_best = improved_macro(metric["macro_f1"], best_macro)
                if is_best:
                    best_macro, best_epoch = metric["macro_f1"], epoch + 1
                row = {"epoch": epoch + 1, "train_loss": running_loss / seen, "holdout_loss": hold_loss,
                       "holdout_micro_f1": metric["micro_f1"], "holdout_macro_f1": metric["macro_f1"], "holdout_image_f1": metric["image_f1"],
                       "train_seconds": train_seconds, "holdout_seconds": hold_wall, "holdout_forward_seconds": hold_forward_seconds,
                       "optimizer_steps": global_step, "amp_skipped_steps": skipped_steps,
                       "next_lr_backbone": optimizer.param_groups[0]["lr"], "next_lr_head": optimizer.param_groups[1]["lr"],
                       "best_epoch": best_epoch, "train_peak_allocated_bytes": peak_allocated}
                exposure = save_exposure(run_dir, train_dataset, epoch + 1)
                row.update(exposure)
                history.append(row)
                checkpoint = {"format": "jeongeun_convnext_resume_v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": resolved,
                              "completed_epoch": epoch + 1, "history": history, "best_macro": best_macro, "best_epoch": best_epoch,
                              "global_step": global_step, "skipped_steps": skipped_steps, "rng": record_rng(generators, device),
                              "train_wall_seconds": train_wall_seconds, "holdout_selection_seconds": holdout_selection_seconds,
                              "peak_allocated": peak_allocated, "peak_reserved": peak_reserved}
                atomic_torch(run_dir / "last.pt", checkpoint)
                if is_best:
                    atomic_torch(run_dir / "best.pt", checkpoint)
                atomic_json(run_dir / "history.json", history)
                atomic_csv(run_dir / "history.csv", history, list(row))
                save_curves(run_dir, history)
                atomic_json(status_path, {"state": "training", "run_kind": cfg["run_kind"], "pid": os.getpid(), "completed_epoch": epoch + 1, "best_epoch": best_epoch, "holdout_macro_f1": metric["macro_f1"], "optimizer_steps": global_step})
                print(f"epoch={epoch+1:02d} train_loss={running_loss/seen:.6f} holdout_loss={hold_loss:.6f} micro={metric['micro_f1']:.6f} macro={metric['macro_f1']:.6f} image={metric['image_f1']:.6f} train_sec={train_seconds:.1f} best_epoch={best_epoch}", flush=True)
                if epoch + 1 - best_epoch >= cfg["early_stopping"]["patience"]:
                    stopped_early = True
                    print("Early stopping: holdout Macro F1 patience reached.", flush=True)
                    break
        best_path = run_dir / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError("best.pt is required for final evaluation")
        best_checkpoint = load_checkpoint(best_path, device)
        model.load_state_dict(best_checkpoint["model"])
        if args.evaluate_only:
            history = best_checkpoint["history"] if not history else history
            best_epoch, best_macro = best_checkpoint["best_epoch"], best_checkpoint["best_macro"]
            train_wall_seconds = max(train_wall_seconds, best_checkpoint["train_wall_seconds"])
            peak_allocated = max(peak_allocated, best_checkpoint["peak_allocated"])
            peak_reserved = max(peak_reserved, best_checkpoint["peak_reserved"])
        atomic_json(status_path, {"state": "final_evaluation", "run_kind": cfg["run_kind"], "pid": os.getpid(), "best_epoch": best_epoch})
        hold_logits, hold_loss, hold_forward, hold_wall = predict(model, hold_loader, device, use_amp)
        hold_metrics = export_reports(run_dir, "holdout", hold_dataset, hold, hold_targets, hold_logits, categories, cfg["threshold"],
                                      {"bce_loss": hold_loss, "forward_seconds": hold_forward, "wall_seconds": hold_wall})
        summary = {"state": "completed", "run_kind": cfg["run_kind"], "exp_id": cfg["exp_id"], "owner": cfg["owner"],
                   "source_commit": cfg["source_commit"], "best_epoch": best_epoch, "best_selection": "holdout Macro F1 only",
                   "epochs_run": len(history), "stopped_early": stopped_early, "params_total": n_params,
                   "device": str(device), "device_name": resolved["device_name"], "batch_size": cfg["batch_size"], "accum": cfg["accum"],
                   "effective_batch": 32, "optimizer_steps": global_step, "amp_skipped_steps": skipped_steps,
                   "train_wall_seconds": train_wall_seconds, "holdout_selection_seconds": holdout_selection_seconds,
                   "train_peak_allocated_bytes": peak_allocated, "train_peak_reserved_bytes": peak_reserved,
                   "holdout_forward_ms_per_image": hold_forward * 1000 / len(hold), "holdout": hold_metrics,
                   "official_val_used_for_selection": False, "checkpoint_format": "full resumable dictionary; model weights at checkpoint['model']"}
        if not args.smoke:
            # First official-val access in this trainer happens here, after best.pt selection.
            val_categories, val_names, val_targets = load_split(val_path, "val")
            if val_categories != categories:
                raise ValueError("official val class order differs from training")
            if len(val_names) != len(set(val_names)) or set(val_names) & set(all_names):
                raise ValueError("duplicate val filenames or train/val filename overlap")
            val_dataset = make_dataset(val_path, "val", val_names, val_targets, cfg, mean, std, False)
            val_loader = make_loader(val_dataset, cfg, device, generators["val"], False)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            val_logits, val_loss, val_forward, val_wall = predict(model, val_loader, device, use_amp)
            val_metrics = export_reports(run_dir, "val", val_dataset, val_names, val_targets, val_logits, categories, cfg["threshold"],
                                         {"bce_loss": val_loss, "forward_seconds": val_forward, "wall_seconds": val_wall})
            summary.update(val=val_metrics, val_forward_ms_per_image=val_forward * 1000 / len(val_names),
                           val_end_to_end_ms_per_image=val_wall * 1000 / len(val_names),
                           val_inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
        else:
            summary["val_not_run_reason"] = "smoke tests never evaluate official val and are not final baseline scores"
        summary.update(external_images=prepared_info["adopted_external_images"], combined_unique_training_images=prepared_info["combined_unique_images"], prepared_fingerprint=prepared_info["fingerprint"], training_loss="masked BCE known-label mean", initialization=resolved["initialization"])
        summary["this_process_wall_seconds"] = time.perf_counter() - start_all
        atomic_json(run_dir / "summary.json", summary)
        save_team_csv(run_dir, reference, cfg, summary)
        atomic_json(status_path, {"state": "completed", "run_kind": cfg["run_kind"], "best_epoch": best_epoch, "epochs_run": len(history), "official_val_evaluated": not args.smoke})
        print(json.dumps({key: value for key, value in summary.items() if key in ("exp_id", "run_kind", "best_epoch", "epochs_run", "params_total", "train_wall_seconds")}, ensure_ascii=False), flush=True)
        if summary.get("val"):
            print("FINAL official val: " + json.dumps({key: summary["val"][key] for key in ("micro_f1", "macro_f1", "image_f1")}), flush=True)
        return run_dir
    except BaseException as error:
        atomic_json(status_path, {"state": "failed", "run_kind": cfg["run_kind"], "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "pid": os.getpid()})
        raise


if __name__ == "__main__":
    main()
