"""Model soup: average the weights of runs that share one architecture/setting (e.g. different seeds) and
evaluate the averaged model like a normal run (Wortsman et al., ICML 2022). Inference cost does not change.

    python scripts/model_soup.py --exp_id R5_cnxt_adamw_soup --runs R5_cnxt_adamw_s42 R5_cnxt_adamw_s43 \
        R5_cnxt_adamw_s44 --runs_dir runs --train_dir data/train --val_dir data/val [--greedy]

uniform (default): average all runs' best.pt.
--greedy: sort runs by their own holdout macro F1, start from the best, and keep each next run only if the
          soup's holdout macro F1 does not drop (selection uses holdout only; the official val is just reported).
Writes <runs_dir>/<exp_id>/ with best.pt, preds_holdout.npz, preds_val.npz, log.json and appends a results row,
so compare_runs.py / aggregate_results.py treat it as a run.
"""
import argparse
import json
import os
import sys
from datetime import date

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "src"), HERE]
import train as T  # noqa: E402  (scripts/train.py: shared helpers)
from fashion_tagging.data.dataset import read_names  # noqa: E402
from fashion_tagging.data.labels import load_split  # noqa: E402
from fashion_tagging.data.preprocess import IMAGENET_MEAN, IMAGENET_STD, timm_mean_std  # noqa: E402
from fashion_tagging.evaluation.metrics import binarize, compute_metrics, save_predictions, sigmoid  # noqa: E402
from fashion_tagging.models.heads import TOKEN_HEADS, build_model  # noqa: E402

SAME = ("model", "head", "pooling", "size", "set_img_size", "preprocess_mode", "mlp_hidden", "query_dim",
        "query_heads", "head_dropout", "threshold")


def average(states):
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = sum(s[k].float() for s in states) / len(states)
            out[k] = out[k].to(v.dtype)
        else:
            out[k] = v.clone()                         # integer buffers (e.g. counters): take the first
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="exp_ids of the runs to average")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--results_csv", default=os.path.join("results", "results.csv"))
    ap.add_argument("--owner", default="")
    a = ap.parse_args()

    logs = {r: json.load(open(os.path.join(a.runs_dir, r, "log.json"), encoding="utf-8")) for r in a.runs}
    ref = logs[a.runs[0]]["args"]
    for r, lg in logs.items():
        diff = [k for k in SAME if lg["args"].get(k) != ref.get(k)]
        if diff:
            raise SystemExit(f"{r} differs from {a.runs[0]} in {diff}; a soup needs the same architecture/setting")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    args = argparse.Namespace(**ref)

    cats, names, targets = load_split(a.train_dir, "train")
    _, vnames, vtargets = load_split(a.val_dir, "val")
    index = {n: i for i, n in enumerate(names)}
    hold = read_names(os.path.join(ref.get("splits_dir") or str(T.SPLITS_DIR), "holdout.txt"))
    if ref.get("max_eval"):
        hold, vnames, vtargets = hold[:ref["max_eval"]], vnames[:ref["max_eval"]], vtargets[:ref["max_eval"]]
    t_ho = targets[[index[n] for n in hold]]
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if ref.get("no_pretrained") else timm_mean_std(ref["model"])
    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=device.type == "cuda")
    hold_loader = DataLoader(T.make_ds(a.train_dir, "train", hold, t_ho, args, mean, std, False), **kw)
    val_loader = DataLoader(T.make_ds(a.val_dir, "val", vnames, vtargets, args, mean, std, False), **kw)

    model = build_model(ref["model"], len(cats), pooling=ref["pooling"], head=ref["head"], pretrained=False,
                        img_size=ref["size"] if ref.get("set_img_size") else None, mlp_hidden=ref["mlp_hidden"],
                        query_dim=ref["query_dim"], query_heads=ref["query_heads"],
                        head_dropout=ref["head_dropout"]).to(device)
    states = {r: torch.load(os.path.join(a.runs_dir, r, "best.pt"), map_location="cpu") for r in a.runs}

    def holdout_macro(sd):
        model.load_state_dict(sd)
        lg, _ = T.predict(model, hold_loader, device, use_amp)
        return compute_metrics(t_ho, binarize(sigmoid(lg), ref["threshold"]))["macro_f1"]

    if a.greedy:
        own = {r: max(e["holdout_macro"] for e in logs[r]["log"]) for r in a.runs}
        order = sorted(a.runs, key=lambda r: -own[r])
        chosen = [order[0]]
        best = holdout_macro(states[order[0]])
        print(f"  start {order[0]}: holdout macro {best:.4f}", flush=True)
        for r in order[1:]:
            score = holdout_macro(average([states[c] for c in chosen + [r]]))
            keep = score >= best
            print(f"  + {r}: {score:.4f} {'keep' if keep else 'skip'}", flush=True)
            if keep:
                chosen, best = chosen + [r], score
    else:
        chosen = list(a.runs)
    soup = average([states[c] for c in chosen])

    run_dir = os.path.join(a.runs_dir, a.exp_id)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(soup, os.path.join(run_dir, "best.pt"))
    model.load_state_dict(soup)
    lg_ho, _ = T.predict(model, hold_loader, device, use_amp)
    lg_val, fwd = T.predict(model, val_loader, device, use_amp)
    save_predictions(os.path.join(run_dir, "preds_holdout.npz"), hold, t_ho, lg_ho, cats)
    save_predictions(os.path.join(run_dir, "preds_val.npz"), vnames, vtargets, lg_val, cats)
    m_ho = compute_metrics(t_ho, binarize(sigmoid(lg_ho), ref["threshold"]))
    m_val = compute_metrics(vtargets, binarize(sigmoid(lg_val), ref["threshold"]))
    kind = "greedy" if a.greedy else "uniform"
    print(f"[{a.exp_id}] {kind} soup of {chosen}: holdout macro {m_ho['macro_f1']:.4f} | "
          f"val micro {m_val['micro_f1']:.4f} macro {m_val['macro_f1']:.4f} image {m_val['image_f1']:.4f}")
    json.dump({"args": {**ref, "exp_id": a.exp_id}, "soup": {"kind": kind, "of": chosen, "candidates": a.runs},
               "best_epoch": None, "params": sum(p.numel() for p in model.parameters()), "log": []},
              open(os.path.join(run_dir, "log.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    import timm
    T.append_row(a.results_csv, {
        "exp_id": a.exp_id, "owner": a.owner, "date": date.today().isoformat(), "model": ref["model"],
        "loss": ref.get("loss", "bce"), "optimizer": ref.get("optimizer", "adamw"),
        "pooling": ref["pooling"] if ref["head"] not in TOKEN_HEADS else "-", "head": ref["head"],
        "scheduler": ref.get("scheduler", "cosine"), "layer_decay": ref.get("layer_decay", 0),
        "drop_path": ref.get("drop_path", 0), "ema": ref.get("ema", 0), "lp_epochs": ref.get("lp_epochs", 0),
        "init_seed": ref.get("init_seed", -1),
        "lr_backbone": ref["lr_backbone"], "lr_head": ref["lr_head"], "weight_decay": ref["weight_decay"],
        "batch_size": ref["batch_size"], "effective_batch": ref["batch_size"] * ref.get("accum", 1),
        "seed": "+".join(str(logs[c]["args"]["seed"]) for c in chosen), "input_size": ref["size"],
        "preprocess_mode": ref["preprocess_mode"], "threshold": ref["threshold"],
        "params_total": sum(p.numel() for p in model.parameters()),
        "holdout_micro_f1": f"{m_ho['micro_f1']:.4f}", "holdout_macro_f1": f"{m_ho['macro_f1']:.4f}",
        "holdout_image_f1": f"{m_ho['image_f1']:.4f}", "val_micro_f1": f"{m_val['micro_f1']:.4f}",
        "val_macro_f1": f"{m_val['macro_f1']:.4f}", "val_image_f1": f"{m_val['image_f1']:.4f}",
        "val_mean_pred_labels": f"{m_val['mean_pred_labels']:.2f}", "val_empty_pred_rate": f"{m_val['empty_pred_rate']:.3f}",
        "infer_ms_per_img": f"{1000 * fwd / len(vnames):.1f}", "torch": torch.__version__, "timm": timm.__version__,
        "cuda": torch.version.cuda or "cpu", "checkpoint": os.path.join(run_dir, "best.pt"),
        "pred_file": os.path.join(run_dir, "preds_val.npz"),
        "notes": f"{kind} soup of {'+'.join(chosen)}" + (" [no pretrained, smoke test]" if ref.get("no_pretrained") else "")})


if __name__ == "__main__":
    main()
