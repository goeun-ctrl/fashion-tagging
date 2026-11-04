"""Optimizers and LR schedules. Returns a LIST of optimizers (Muon hybrid needs two).

  adamw  AdamW, two groups: backbone (lr_backbone) and head (lr_head). Team baseline, unchanged:
         weight decay on every parameter.
  muon   Muon + AdamW hybrid (torch.optim.Muon, PyTorch >= 2.9).
           Muon : 2D weight matrices of the backbone (ViT qkv/proj/mlp, ConvNeXt pwconv Linear).
                  With --muon_head also the head's inner matrices (proj/attn/ffn/fc1).
           AdamW: everything else - 4D conv kernels (ConvNeXt stem/dwconv, patch embedding), biases,
                  norms, position embeddings / tokens, class queries and the final classifier.
         Muon uses adjust_lr_fn="match_rms_adamw", so its lr is on the AdamW scale and lr_backbone /
         lr_head keep their meaning. muon_lr_scale multiplies the Muon groups only.
  soap   SOAP from the pytorch-optimizer package (pip install pytorch-optimizer), same two groups as
         adamw. Optional: only for when time and GPU allow.

layer_decay (e.g. 0.65-0.75 for EVA02 in the plan): lr of backbone layer i is multiplied by
  layer_decay ** (depth - i) using timm's layer map, and biases / norms get no weight decay. Applies to
  every optimizer above, so AdamW and Muon runs can share it. Off (None) = team baseline.

Changing the optimizer usually needs its own lr / weight decay; record them in the results CSV.
"""
import math

import torch

from ..models.heads import FashionTagger, head_named_parameters

OPTIMIZERS = ("adamw", "muon", "soap")
SCHEDULERS = ("cosine", "constant")
_BODY_NOT_MUON = ("embed", "token")                                   # embedding-like matrices
_HEAD_NOT_MUON = ("head.query", "head.fc_w", "head.fc2.weight", "classifier.")   # queries and output layers


def _muon_ok(name, p, in_head=False):
    if p.ndim != 2:
        return False
    if in_head:
        return not any(name.startswith(k) for k in _HEAD_NOT_MUON)
    return not any(k in name for k in _BODY_NOT_MUON)


def _settings(model, lr_backbone, lr_head, weight_decay, layer_decay):
    """(name, param, lr, weight_decay, in_head) for every trainable parameter. Head params are named as in
    head_named_parameters ("head.*" / "classifier.*") so the Muon rules above apply to them."""
    head_names = {id(p): n for n, p in head_named_parameters(model)}
    scale, wd_of = {}, {}
    if layer_decay:
        # timm's ViT/ConvNeXt layer map: lr_scale = layer_decay ** (depth - layer); 1D params get no decay
        from timm.optim._param_groups import param_groups_layer_decay
        target = model.backbone if isinstance(model, FashionTagger) else model
        for g in param_groups_layer_decay(target, weight_decay=weight_decay, layer_decay=layer_decay):
            for p in g["params"]:
                scale[id(p)], wd_of[id(p)] = g["lr_scale"], g["weight_decay"]
    out = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            h = id(p) in head_names
            lr = (lr_head if h else lr_backbone) * scale.get(id(p), 1.0)   # wrapper heads: scale 1 (top)
            out.append((head_names.get(id(p), n), p, lr, wd_of.get(id(p), weight_decay), h))
    return out


def _groups(items, lr_mult=1.0):
    """Merge parameters with the same (lr, weight decay) into one param group, in model order."""
    by = {}
    for _, p, lr, wd, _ in items:
        by.setdefault((lr * lr_mult, wd), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in by.items()]


def build_optimizers(model, name="adamw", lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05,
                     muon_momentum=0.95, muon_head=False, muon_lr_scale=1.0, layer_decay=None):
    items = _settings(model, lr_backbone, lr_head, weight_decay, layer_decay)

    if name == "adamw":
        return [torch.optim.AdamW(_groups(items))]

    if name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(f"torch.optim.Muon needs PyTorch >= 2.9 (found {torch.__version__})")
        to_muon = [it for it in items if _muon_ok(it[0], it[1], in_head=it[4]) and (muon_head or not it[4])]
        m_ids = {id(it[1]) for it in to_muon}
        opts = [torch.optim.AdamW(_groups([it for it in items if id(it[1]) not in m_ids]))]
        if to_muon:
            opts.insert(0, torch.optim.Muon(_groups(to_muon, muon_lr_scale), momentum=muon_momentum,
                                            adjust_lr_fn="match_rms_adamw"))
        return opts

    if name == "soap":
        try:
            from pytorch_optimizer import SOAP
        except ImportError as e:
            raise ImportError("SOAP needs the pytorch-optimizer package: pip install pytorch-optimizer") from e
        return [SOAP(_groups(items))]

    raise ValueError(f"unknown optimizer {name!r}; choose from {list(OPTIMIZERS)}")


def lr_lambda(kind, warmup_steps, total_steps):
    """Linear warmup, then cosine decay to 0 ('cosine', team baseline) or a flat lr ('constant')."""
    if kind not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {kind!r}; choose from {list(SCHEDULERS)}")

    def f(s):
        if s < warmup_steps:
            return (s + 1) / warmup_steps
        if kind == "constant":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps)))
    return f


def build_schedulers(optimizers, kind, warmup_steps, total_steps):
    return [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda(kind, warmup_steps, total_steps)) for o in optimizers]


def describe(optimizers):
    """Short text for logs: optimizer type and parameter count per group (summarised when there are many)."""
    out = []
    for o in optimizers:
        gs = o.param_groups
        if len(gs) <= 4:
            body = ", ".join(f"lr={g['lr']:g}:{sum(p.numel() for p in g['params']):,}" for g in gs)
        else:
            lrs = [g["lr"] for g in gs]
            body = (f"{len(gs)} groups, lr {min(lrs):.3g}..{max(lrs):.3g}, "
                    f"{sum(p.numel() for g in gs for p in g['params']):,} params")
        out.append(f"{type(o).__name__}[{body}]")
    return " + ".join(out)
