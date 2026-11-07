"""Role 5 EVA02 regularization runs. Same inputs and preflight as the Role 3 CLI, plus the options of reg.py.

  python -m experiments.role5_regularization.eva02.cli train --package ... --label-mode partial \\
      --run-id R5_eva_wdx_s42 --wd-exclude
All options off reproduces the Role 3 trainer (role3_equivalent=true in resolved_config.json).
"""
import argparse
import json
from pathlib import Path

import experiments.role3_eva02.retrain_common_data  # noqa: F401
from experiments.role3_eva02.retrain_common_data import ROLE
from experiments.role3_eva02.retrain_common_data.preflight import CFG, Inputs, preflight


def options(args):
    return {"drop_path": args.drop_path, "wd_exclude": args.wd_exclude,
            "layer_decay": args.layer_decay, "ema": args.ema}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--package", required=True, type=Path)
        p.add_argument("--shared-lock", required=True, type=Path)
        p.add_argument("--train-root", required=True, type=Path)
        p.add_argument("--val-root", required=True, type=Path)
        p.add_argument("--external-root", required=True, type=Path)
        p.add_argument("--weights", type=Path, default=ROLE / "private/model.safetensors")
        p.add_argument("--label-mode", choices=("complete", "partial"), default="complete")
        p.add_argument("--run-id", required=True)
        p.add_argument("--drop-path", type=float, default=0.0)
        p.add_argument("--wd-exclude", action="store_true")
        p.add_argument("--layer-decay", type=float, default=None)
        p.add_argument("--ema", type=float, default=0.0)
        p.add_argument("--lr-backbone", type=float, default=CFG["lr_backbone"])
        p.add_argument("--smoke", type=int, default=0, help="N draws, 1 epoch, 64 holdout images, no val")
        if name == "train":
            p.add_argument("--resume", action="store_true")
        if name == "evaluate":
            p.add_argument("--split", choices=("holdout", "val", "both"), default="both")
    args = parser.parse_args(argv)
    inputs = Inputs(args.package, args.shared_lock, args.train_root, args.val_root, args.external_root, args.weights)
    result, data = preflight(inputs, args.label_mode, args.run_id)
    print(json.dumps({k: v for k, v in result.items() if k != "shared"}, indent=2, ensure_ascii=False), flush=True)
    if result["status"] != "PASS":
        return 2 if result["status"] == "BLOCKED" else 1
    if args.command in {"train", "evaluate"}:
        from .trainer import execute
        execute(args, inputs, result, data, options(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
