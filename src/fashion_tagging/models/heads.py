"""Pooling + classification head on top of a timm backbone.

    model = build_model("convnext_tiny.fb_in22k_ft_in1k_384", 27, pooling="gap", head="linear")
    logits = model(x)                       # [B, 27], no sigmoid
    head = head_named_parameters(model)     # the newly initialised part (gets lr_head)

Pooling (spatial features -> one vector; used by the linear and mlp heads)
  gap     global average pooling                (timm global_pool="avg", team baseline)
  gmp     global max pooling                    (timm global_pool="max")
  gapgmp  0.5 * (average + max), same width     (timm global_pool="avgmax")

Heads (vector or tokens -> 27 logits)
  linear  the timm classifier itself. With pooling=gap this IS the team baseline (C1): the returned
          object is the plain timm model, so checkpoints keep timm's key names.
  mlp     Linear -> GELU -> Dropout -> Linear on the pooled vector.
  query   Query2Label-style (C2 in the plan): 27 learned class queries cross-attend to the
          backbone's token map (no self-attention), FFN, then one logit per query. Pooling is not
          used. Tokens get no positional encoding.
  query_ms  C3 in the plan: the same decoder, but keys/values are the stride-16 AND stride-32 feature
          maps (each with its own LayerNorm + projection and a learned level embedding), so small
          accessories keep a finer map. Needs a backbone whose last two stages have different strides
          (ConvNeXt); a plain ViT such as EVA02 has one stride and is rejected.

Pooling goes through timm's own pooling, so the pretrained post-pool LayerNorm of ConvNeXt is kept
for every pooling type. ViT-family models (EVA02) may not accept "max"/"avgmax" in every timm version;
timm then raises at create time.
"""
import torch
import torch.nn as nn
import timm

POOLS = {"gap": "avg", "gmp": "max", "gapgmp": "avgmax"}
HEADS = ("linear", "mlp", "query", "query_ms")
TOKEN_HEADS = ("query", "query_ms")   # heads that read the token map instead of a pooled vector


class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class QueryHead(nn.Module):
    """Class queries [C, D] attend to tokens [B, L, in_dim]; logit_c = <w_c, out_c> + b_c."""

    def __init__(self, in_dim, num_classes, dim=384, num_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, dim)
        self.query = nn.Parameter(torch.empty(num_classes, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * ffn_mult, dim))
        self.norm2 = nn.LayerNorm(dim)
        # one weight vector per class (a "group" linear layer): query c only produces logit c
        self.fc_w = nn.Parameter(torch.empty(num_classes, dim))
        self.fc_b = nn.Parameter(torch.zeros(num_classes))
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.fc_w, std=0.02)

    def decode(self, kv, pad_mask=None):
        """kv: projected tokens [B, L, dim] -> logits [B, C]. pad_mask: optional bool [B, L], True = padding
        (ignored by the attention; used by scripts/train_hires.py, whose batches are padded canvases)."""
        q = self.query.unsqueeze(0).expand(kv.size(0), -1, -1)
        a, _ = self.attn(q, kv, kv, key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(q + a)
        x = self.norm2(x + self.ffn(x))
        return (x * self.fc_w).sum(-1) + self.fc_b

    def forward(self, tokens, pad_mask=None):
        return self.decode(self.proj(self.in_norm(tokens)), pad_mask)


class MultiScaleQueryHead(QueryHead):
    """C3: QueryHead whose keys/values are several feature maps (finest first, coarsest last).
    The coarsest level reuses QueryHead's in_norm/proj, so the extra cost over C2 is one LayerNorm +
    Linear per finer level and a [levels, dim] level embedding."""

    def __init__(self, in_dims, num_classes, dim=384, num_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__(in_dims[-1], num_classes, dim, num_heads, ffn_mult, dropout)
        self.extra_norms = nn.ModuleList(nn.LayerNorm(d) for d in in_dims[:-1])
        self.extra_projs = nn.ModuleList(nn.Linear(d, dim) for d in in_dims[:-1])
        self.level_embed = nn.Parameter(torch.empty(len(in_dims), dim))
        nn.init.trunc_normal_(self.level_embed, std=0.02)

    def forward(self, levels, pad_masks=None):
        """levels: token lists [B, L_i, C_i], finest first. pad_masks: optional matching bool [B, L_i]."""
        norms = list(self.extra_norms) + [self.in_norm]
        projs = list(self.extra_projs) + [self.proj]
        kv = [p(n(t)) + self.level_embed[i] for i, (n, p, t) in enumerate(zip(norms, projs, levels))]
        mask = None if pad_masks is None else torch.cat(pad_masks, dim=1)
        return self.decode(torch.cat(kv, dim=1), mask)


class FashionTagger(nn.Module):
    """timm backbone without its classifier + one of the heads above (mlp / query)."""

    def __init__(self, backbone, head, head_type):
        super().__init__()
        self.backbone, self.head, self.head_type = backbone, head, head_type

    def tokens(self, x):
        f = self.backbone.forward_features(x)
        if f.ndim == 4:                                   # CNN: [B, C, H, W] -> [B, HW, C]
            return f.flatten(2).transpose(1, 2)
        return f[:, getattr(self.backbone, "num_prefix_tokens", 0):]   # ViT: drop cls/register tokens

    def levels(self, x):
        """Last two stage outputs (stride 16, stride 32) as token lists [B, HW, C], finest first."""
        maps = self.backbone.forward_intermediates(x, indices=[-2, -1], intermediates_only=True)
        return [m.flatten(2).transpose(1, 2) for m in maps]

    def forward(self, x):
        if self.head_type == "query":
            return self.head(self.tokens(x))
        if self.head_type == "query_ms":
            return self.head(self.levels(x))
        return self.head(self.backbone(x))               # backbone(x) = pooled (+normed) features


def build_model(model_name, num_classes, pooling="gap", head="linear", pretrained=True, img_size=None,
                mlp_hidden=1024, query_dim=384, query_heads=8, head_dropout=0.1, drop_path=0.0):
    if pooling not in POOLS:
        raise ValueError(f"unknown pooling {pooling!r}; choose from {list(POOLS)}")
    if head not in HEADS:
        raise ValueError(f"unknown head {head!r}; choose from {list(HEADS)}")
    extra = {"img_size": img_size} if img_size else {}
    if drop_path:
        extra["drop_path_rate"] = drop_path   # stochastic depth (plan: 0.1 for ViT); 0 = team baseline
    if head == "linear":
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes,
                                 global_pool=POOLS[pooling], **extra)
    if head == "mlp":
        bb = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                               global_pool=POOLS[pooling], **extra)
        return FashionTagger(bb, MLPHead(bb.num_features, num_classes, mlp_hidden, head_dropout), head)
    bb = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="", **extra)
    if head == "query":
        return FashionTagger(bb, QueryHead(bb.num_features, num_classes, query_dim, query_heads,
                                           dropout=head_dropout), head)
    info = bb.feature_info[-2:]
    if info[0]["reduction"] == info[1]["reduction"]:
        raise ValueError(f"{model_name}: last two stages share stride {info[0]['reduction']}; "
                         "query_ms needs a multi-scale (CNN) backbone. Use --head query instead.")
    return FashionTagger(bb, MultiScaleQueryHead([f["num_chs"] for f in info], num_classes, query_dim,
                                                 query_heads, dropout=head_dropout), head)


def head_named_parameters(model):
    """(name, param) of the newly initialised classification part, for lr_head and the optimizer split."""
    if isinstance(model, FashionTagger):
        return list(model.head.named_parameters(prefix="head"))
    return list(model.get_classifier().named_parameters(prefix="classifier"))
