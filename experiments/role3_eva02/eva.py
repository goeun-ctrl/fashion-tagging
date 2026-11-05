"""Native timm classifier, initialized from a verified local public checkpoint."""
import hashlib
import json
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from timm.layers import resample_abs_pos_embed
from timm.models.eva import checkpoint_filter_fn

from data_check import HERE, require

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
WEIGHTS = HERE / "private/model.safetensors"


def architecture(size):
    return timm.create_model(CFG["model"], pretrained=False, num_classes=27, img_size=size)


def from_pretrained(size, weights=None):
    # Use timm's actual filter (including any key renames) then require every
    # backbone tensor. The official file is pinned independently of its filename.
    weights = WEIGHTS if weights is None else Path(weights)
    with weights.open("rb") as f:
        require(hashlib.file_digest(f, "sha256").hexdigest() == CFG["weight_sha256"], "wrong pretrained artifact")
    raw = load_file(str(weights))
    model = architecture(size)
    adapted = checkpoint_filter_fn(dict(raw), model)
    head_keys = {"head.weight", "head.bias"}
    require(head_keys <= adapted.keys(), "official classifier missing")
    for k in head_keys:
        del adapted[k]
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    require(set(missing) == head_keys and not unexpected, f"incomplete backbone {missing}, {unexpected}")
    require(all(torch.equal(p, adapted[n]) for n, p in model.named_parameters() if n not in head_keys),
            "backbone tensor does not match the official checkpoint")
    expected_pos = resample_abs_pos_embed(raw["pos_embed"], model.patch_embed.grid_size, num_prefix_tokens=1)
    require(torch.equal(model.pos_embed, expected_pos), "incorrect positional interpolation")
    require(model.global_pool == "avg" and isinstance(model.fc_norm, torch.nn.LayerNorm), "unexpected pooling")
    require(model.head.in_features == 384 and model.head.out_features == 27, "unexpected classifier")
    return model


def describe(model):
    total = sum(p.numel() for p in model.parameters())
    require(total <= 50_000_000, "inference parameter budget exceeded")
    return {"total": total, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "grid": list(model.patch_embed.grid_size), "pos_embed": list(model.pos_embed.shape),
            "pool": model.global_pool, "prepool_norm": type(model.norm).__name__,
            "postpool_norm": type(model.fc_norm).__name__, "classifier": str(model.head),
            "num_prefix_tokens": model.num_prefix_tokens, "static_size": model.patch_embed.strict_img_size}

