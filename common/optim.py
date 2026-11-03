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

Changing the optimizer usually needs its own lr / weight decay; record them in the results CSV.
"""
import math

import torch

from heads import head_named_parameters

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


def _groups(pairs_by_lr, weight_decay):
    return [{"params": [p for _, p in ps], "lr": lr, "weight_decay": weight_decay}
            for lr, ps in pairs_by_lr if ps]


def build_optimizers(model, name="adamw", lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05,
                     muon_momentum=0.95, muon_head=False, muon_lr_scale=1.0):
    head = [(n, p) for n, p in head_named_parameters(model) if p.requires_grad]
    head_ids = {id(p) for _, p in head}
    body = [(n, p) for n, p in model.named_parameters() if p.requires_grad and id(p) not in head_ids]

    if name == "adamw":
        return [torch.optim.AdamW(_groups([(lr_backbone, body), (lr_head, head)], weight_decay))]

    if name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(f"torch.optim.Muon needs PyTorch >= 2.9 (found {torch.__version__})")
        body_m = [(n, p) for n, p in body if _muon_ok(n, p)]
        head_m = [(n, p) for n, p in head if muon_head and _muon_ok(n, p, in_head=True)]
        m_ids = {id(p) for _, p in body_m + head_m}
        body_a = [(n, p) for n, p in body if id(p) not in m_ids]
        head_a = [(n, p) for n, p in head if id(p) not in m_ids]
        opts = [torch.optim.AdamW(_groups([(lr_backbone, body_a), (lr_head, head_a)], weight_decay))]
        m_groups = _groups([(lr_backbone * muon_lr_scale, body_m), (lr_head * muon_lr_scale, head_m)],
                           weight_decay)
        if m_groups:
            opts.insert(0, torch.optim.Muon(m_groups, momentum=muon_momentum, adjust_lr_fn="match_rms_adamw"))
        return opts

    if name == "soap":
        try:
            from pytorch_optimizer import SOAP
        except ImportError as e:
            raise ImportError("SOAP needs the pytorch-optimizer package: pip install pytorch-optimizer") from e
        return [SOAP(_groups([(lr_backbone, body), (lr_head, head)], weight_decay))]

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
    """Short text for logs: optimizer type and parameter count per group."""
    out = []
    for o in optimizers:
        sizes = [f"lr={g['lr']:g}:{sum(p.numel() for p in g['params']):,}" for g in o.param_groups]
        out.append(f"{type(o).__name__}[{', '.join(sizes)}]")
    return " + ".join(out)
