"""Role 5 regularization options for the Role 3 EVA02 common-data retrain.

With every option off, each function returns exactly what the Role 3 code builds:
  model      eva.from_pretrained / eva.architecture       (drop path 0)
  optimizer  run.optimizer: AdamW, 2 groups, wd 0.05 on every parameter
  scheduler  build_schedulers cosine, no EMA

Options (see experiments/role5_regularization/README.md):
  drop_path    stochastic depth rate passed to timm (`drop_path_rate`)
  wd_exclude   no weight decay for 1-D params (norm weights, biases) and model.no_weight_decay()
               (EVA: pos_embed, cls_token); every other tensor keeps weight_decay
  layer_decay  backbone lr * layer_decay ** (depth - layer) via timm's EVA layer map (implies wd_exclude)
  ema          timm ModelEmaV3 decay, updated after every successful optimizer step (AMP skips excluded)
"""
import timm
import torch

import experiments.role3_eva02.retrain_common_data  # noqa: F401  (puts role3 modules on sys.path)
import eva
from data_check import require
from experiments.role3_eva02 import run as baseline

OFF = {"drop_path": 0.0, "wd_exclude": False, "layer_decay": None, "ema": 0.0}


def is_off(reg, lr_backbone):
    return reg == OFF and lr_backbone == eva.CFG["lr_backbone"]


def architecture(size, drop_path=0.0):
    if not drop_path:
        return eva.architecture(size)
    return timm.create_model(eva.CFG["model"], pretrained=False, num_classes=27, img_size=size,
                             drop_path_rate=drop_path)


def from_pretrained(size, weights, drop_path=0.0):
    """Role 3 loader (hash pin, strict backbone match, pos-embed interpolation check) on a drop-path model."""
    if not drop_path:
        return eva.from_pretrained(size, weights)
    original = eva.architecture
    eva.architecture = lambda s: architecture(s, drop_path)
    try:
        return eva.from_pretrained(size, weights)
    finally:
        eva.architecture = original


def optimizer(model, reg, lr_backbone, lr_head, weight_decay):
    """AdamW. Groups are ordered backbone first, head last, and carry `is_head`."""
    if is_off(reg, lr_backbone):
        opt = baseline.optimizer(model)
        for g in opt.param_groups:
            g["is_head"] = g["lr"] == lr_head
        return opt
    from timm.optim._param_groups import param_groups_layer_decay
    nwd = set(model.no_weight_decay()) if (reg["wd_exclude"] or reg["layer_decay"]) else set()
    scale, wd_of = {}, {}
    if reg["layer_decay"]:
        for g in param_groups_layer_decay(model, weight_decay=weight_decay, no_weight_decay_list=nwd,
                                          layer_decay=reg["layer_decay"]):
            for p in g["params"]:
                scale[id(p)], wd_of[id(p)] = g["lr_scale"], g["weight_decay"]
    head = {id(p) for p in model.head.parameters()}
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        h = id(p) in head
        if id(p) in wd_of:
            wd = wd_of[id(p)]
        elif reg["wd_exclude"] and (p.ndim <= 1 or name in nwd):
            wd = 0.0
        else:
            wd = weight_decay
        lr = lr_head if h else lr_backbone * scale.get(id(p), 1.0)
        groups.setdefault((h, lr, wd), []).append(p)
    ordered = sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1], -kv[0][2]))
    opt = torch.optim.AdamW([{"params": ps, "lr": lr, "weight_decay": wd, "is_head": h}
                             for (h, lr, wd), ps in ordered], weight_decay=weight_decay)
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    require(len(seen) == len(set(seen)) == sum(p.requires_grad for p in model.parameters()),
            "every trainable parameter must be in exactly one group")
    return opt


class EmaScheduler:
    """Scheduler proxy. run.update() calls step() only after a successful optimizer step, so the EMA sees
    exactly the applied updates. state_dict/load_state_dict are the wrapped scheduler's."""

    def __init__(self, scheduler, model, decay):
        from timm.utils import ModelEmaV3
        self.scheduler, self.model = scheduler, model
        self.ema = ModelEmaV3(model, decay=decay, use_warmup=True)  # same as scripts/train.py --ema

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch

    def step(self):
        self.scheduler.step()
        self.ema.update(self.model, step=self.scheduler.last_epoch)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state)


def lr_report(opt):
    """(backbone max lr, head lr) for history rows; robust to many layer-decay groups."""
    body = [g["lr"] for g in opt.param_groups if not g.get("is_head")]
    head = [g["lr"] for g in opt.param_groups if g.get("is_head")]
    return max(body), head[0]
