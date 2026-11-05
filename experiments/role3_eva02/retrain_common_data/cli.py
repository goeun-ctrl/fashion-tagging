"""Server commands. Preflight prints JSON without creating a run or loading an EVA model."""
import argparse
import json
from pathlib import Path

from . import ROLE
from .package import check_lock
from .preflight import Inputs, preflight


def compare(left, right):
    def signature(path):
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        shared = config["shared"]
        check_lock(shared, config["label_mode"])
        return shared

    eva, peer = signature(left), signature(right)
    differences = sorted(k for k in set(eva) | set(peer) if eva.get(k) != peer.get(k))
    files = sorted(k for k in eva["sha256"] if eva["sha256"][k] != peer["sha256"][k])
    return {"status": "FAIL" if differences else "PASS", "different_fields": differences,
            "different_file_hashes": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
        if name == "train":
            p.add_argument("--resume", action="store_true")
        if name == "evaluate":
            p.add_argument("--split", choices=("holdout", "val", "both"), default="both")
    p = commands.add_parser("compare")
    p.add_argument("--eva-config", required=True, type=Path)
    p.add_argument("--peer-config", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "compare":
        try:
            result = compare(args.eva_config, args.peer_config)
        except FileNotFoundError as exc:
            result = {"status": "BLOCKED", "reason": str(exc)}
        except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
            result = {"status": "FAIL", "reason": str(exc)}
    else:
        inputs = Inputs(args.package, args.shared_lock, args.train_root, args.val_root, args.external_root, args.weights)
        result, data = preflight(inputs, args.label_mode, args.run_id)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result["status"] != "PASS":
        return 2 if result["status"] == "BLOCKED" else 1
    if args.command in {"train", "evaluate"}:
        from .trainer import execute
        execute(args, inputs, result, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
