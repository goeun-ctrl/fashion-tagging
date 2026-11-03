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

Pooling goes through timm's own pooling, so the pretrained post-pool LayerNorm of ConvNeXt is kept
for every pooling type. ViT-family models (EVA02) may not accept "max"/"avgmax" in every timm version;
timm then raises at create time.
"""
import torch
import torch.nn as nn
import timm

POOLS = {"gap": "avg", "gmp": "max", "gapgmp": "avgmax"}
HEADS = ("linear", "mlp", "query")


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

    def forward(self, tokens):
        kv = self.proj(self.in_norm(tokens))
        q = self.query.unsqueeze(0).expand(tokens.size(0), -1, -1)
        a, _ = self.attn(q, kv, kv, need_weights=False)
        x = self.norm1(q + a)
        x = self.norm2(x + self.ffn(x))
        return (x * self.fc_w).sum(-1) + self.fc_b


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

    def forward(self, x):
        if self.head_type == "query":
            return self.head(self.tokens(x))
        return self.head(self.backbone(x))               # backbone(x) = pooled (+normed) features


def build_model(model_name, num_classes, pooling="gap", head="linear", pretrained=True, img_size=None,
                mlp_hidden=1024, query_dim=384, query_heads=8, head_dropout=0.1):
    if pooling not in POOLS:
        raise ValueError(f"unknown pooling {pooling!r}; choose from {list(POOLS)}")
    if head not in HEADS:
        raise ValueError(f"unknown head {head!r}; choose from {list(HEADS)}")
    extra = {"img_size": img_size} if img_size else {}
    if head == "linear":
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes,
                                 global_pool=POOLS[pooling], **extra)
    if head == "mlp":
        bb = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                               global_pool=POOLS[pooling], **extra)
        return FashionTagger(bb, MLPHead(bb.num_features, num_classes, mlp_hidden, head_dropout), head)
    bb = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="", **extra)
    return FashionTagger(bb, QueryHead(bb.num_features, num_classes, query_dim, query_heads,
                                       dropout=head_dropout), head)


def head_named_parameters(model):
    """(name, param) of the newly initialised classification part, for lr_head and the optimizer split."""
    if isinstance(model, FashionTagger):
        return list(model.head.named_parameters(prefix="head"))
    return list(model.get_classifier().named_parameters(prefix="classifier"))
