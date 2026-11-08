"""Build the submission package from a scripts/train_hires.py run.

    python scripts/make_submission.py --run_dir /data/eunjin/DL/runs_hr/R5_final_msmax_s42 --out submission_build \
        --threshold 0.6 --class_threshold "sweater=0.5" "leg warmer=0.5" "umbrella=0.5" --zip team_name.zip
    python scripts/make_submission.py --run_dir RUN --out submission_build --thresholds_json report/thr/thresholds.json

Writes into --out: predict.py (copy of submission/predict.py), model.pt, requirements.txt, README.md. With --zip the
four files are zipped at the top level of team_name.zip, as the assignment asks.

model.pt holds only tensors and plain Python values (loads with torch.load(weights_only=True)): the EMA weights of
best.pt, the settings needed to rebuild the model, the preprocessing numbers (mean / std are written out, so no timm
lookup is needed offline), the thresholds and the gate threshold. The model is rebuilt with submission/predict.py and the
weights are loaded strictly; the parameter count must stay at or below 50,000,000.
"""
import argparse
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import sys
import zipfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fashion_tagging.data.preprocess import timm_mean_std  # noqa: E402

SUB = os.path.join(ROOT, "submission")
CONFIG_KEYS = ("model", "head", "pool", "csra_lambda", "csra_t", "query_dim", "query_heads", "head_dropout")


def load_predict_module():
    spec = importlib.util.spec_from_file_location("submission_predict", os.path.join(SUB, "predict.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_class_thresholds(specs, cats):
    out = {}
    for spec in specs:
        name, v = spec.rsplit("=", 1)
        hit = [c for c in cats if c == name or c.split(",")[0].strip() == name]
        if len(hit) != 1:
            raise SystemExit(f"class {name!r} not found (or ambiguous) in the 27 categories")
        out[hit[0]] = float(v)
    return out


def convert(run_dir, out_dir, ckpt="best.pt", threshold=0.5, class_threshold=(), thresholds_json=None,
            gate_threshold=None, tta="flip", short=None, max_long=None, note=""):
    pred = load_predict_module()
    st = torch.load(os.path.join(run_dir, ckpt), map_location="cpu")
    cfg, cats = dict(st["config"]), list(st["categories"])
    if cats != pred.CATEGORIES:
        raise SystemExit("categories of the checkpoint differ from the official 27 names / order")
    config = {k: cfg[k] for k in CONFIG_KEYS if cfg.get(k) is not None}
    config.setdefault("head", "linear")
    config["gate"] = bool(cfg.get("gate"))
    mean, std = timm_mean_std(cfg["model"])
    pp = {"short": int(short or cfg.get("test_short", 800)), "max_long": int(max_long or cfg.get("max_long", 1333)),
          "mean": [float(v) for v in mean], "std": [float(v) for v in std]}
    if thresholds_json:
        tj = json.load(open(thresholds_json, encoding="utf-8"))
        thresholds = {"global": float(tj["global"]), "per_class": {c: float(v) for c, v in tj.get("per_class", {}).items()}}
    else:
        thresholds = {"global": float(threshold), "per_class": parse_class_thresholds(class_threshold, cats)}
    gate_thr = float(gate_threshold if gate_threshold is not None else cfg.get("gate_threshold", 0.5))

    model = pred.build_model(config, len(cats) + (1 if config["gate"] else 0))
    model.load_state_dict(st["model"])                                        # strict: every key must match
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > pred.PARAM_LIMIT:
        raise SystemExit(f"{n_params:,} parameters exceed the {pred.PARAM_LIMIT:,} limit")
    obj = {"format": pred.FORMAT, "config": config, "state_dict": {k: v.detach().cpu().contiguous() for k, v in st["model"].items()},
           "categories": cats, "preprocess": pp, "tta": tta, "thresholds": thresholds, "gate_threshold": gate_thr,
           "notes": {"source_run": os.path.basename(os.path.normpath(run_dir)), "checkpoint": ckpt, "epoch": int(st.get("epoch", -1)),
                     "params": int(n_params), "note": note}}
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(obj, model_path)
    torch.load(model_path, map_location="cpu", weights_only=True)             # must load with the safe loader
    shutil.copy(os.path.join(SUB, "predict.py"), os.path.join(out_dir, "predict.py"))
    versions = {p: md.version(p) for p in ("torch", "timm", "numpy", "Pillow")}
    write_requirements(os.path.join(out_dir, "requirements.txt"), versions)
    write_readme(os.path.join(out_dir, "README.md"), obj, versions)
    return {"params": n_params, "dir": out_dir, "model_pt_mb": os.path.getsize(model_path) / 1e6}


def write_requirements(path, versions):
    lines = ["# tested with: " + ", ".join(f"{k} {v}" for k, v in versions.items())]
    for k, v in versions.items():
        lines.append(f"{k}=={v.split('+')[0]}")                             # drop local tags such as +cu126
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def write_readme(path, obj, versions):
    tpl = open(os.path.join(SUB, "README.template.md"), encoding="utf-8").read()
    th = obj["thresholds"]
    thr_text = f"global {th['global']:g}" + (", per class: " + ", ".join(f"{c} {v:g}" for c, v in th["per_class"].items())
                                              if th["per_class"] else "")
    fill = {"@@PARAMS@@": f"{obj['notes']['params']:,}", "@@MODEL@@": obj["config"]["model"], "@@HEAD@@": obj["config"]["head"] +
            (f" / {obj['config']['pool']}" if obj["config"].get("pool") else ""), "@@GATE@@": "yes" if obj["config"]["gate"] else "no",
            "@@SHORT@@": str(obj["preprocess"]["short"]), "@@MAXLONG@@": str(obj["preprocess"]["max_long"]),
            "@@TTA@@": "horizontal flip (mean of logits)" if obj["tta"] == "flip" else "none", "@@THRESHOLDS@@": thr_text,
            "@@RUN@@": obj["notes"]["source_run"], "@@EPOCH@@": str(obj["notes"]["epoch"]),
            "@@VERSIONS@@": ", ".join(f"{k} {v}" for k, v in versions.items())}
    for k, v in fill.items():
        tpl = tpl.replace(k, v)
    open(path, "w", encoding="utf-8", newline="\n").write(tpl)


def make_zip(out_dir, zip_path):
    names = ["predict.py", "model.pt", "requirements.txt", "README.md"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out_dir, n), arcname=n)                     # top level, no folder
    return zip_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--class_threshold", nargs="*", default=[], help='e.g. "sweater=0.5" "leg warmer=0.5"')
    ap.add_argument("--thresholds_json", default=None, help="thresholds.json from scripts/tune_threshold.py")
    ap.add_argument("--gate_threshold", type=float, default=None)
    ap.add_argument("--tta", default="flip", choices=["flip", "none"])
    ap.add_argument("--short", type=int, default=None)
    ap.add_argument("--max_long", type=int, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--zip", default=None, help="also write team_name.zip")
    a = ap.parse_args()
    info = convert(a.run_dir, a.out, a.ckpt, a.threshold, a.class_threshold, a.thresholds_json, a.gate_threshold, a.tta,
                   a.short, a.max_long, a.note)
    print(f"model.pt: {info['model_pt_mb']:.0f} MB, parameters {info['params']:,} -> {info['dir']}")
    if a.zip:
        print("zip:", make_zip(a.out, a.zip))
    print("next: python scripts/check_submission.py --zip", a.zip or "<team_name.zip>", "--input_dir <some images>")


if __name__ == "__main__":
    main()
