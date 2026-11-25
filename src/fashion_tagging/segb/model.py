"""Redesign B model: ConvNeXt-T (DINOv3) stages 1-4 -> UPerNet (width 256, stride 4, fp32) ->
group-wise softmax garment map (31 ch = 4 groups x (none + classes)) + 7 part logits (sigmoid).
Tags come ONLY from the garment map: per class, mean of the top-k cell probabilities inside the real image,
k = max(min_k, round(ratio_c x valid cells)). No global head."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .spec import GROUPS, GROUP_SIZES, N_SEG, PARTS, STRIDE


def cbr(i, o, k):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.GroupNorm(32, o), nn.ReLU(inplace=True))


class UPerNet(nn.Module):
    def __init__(self, chs, w=256, bins=(1, 2, 3, 6)):
        super().__init__()
        self.bins = bins
        self.ppm = nn.ModuleList(cbr(chs[-1], w, 1) for _ in bins)
        self.bottleneck = cbr(chs[-1] + len(bins) * w, w, 3)
        self.lat = nn.ModuleList(cbr(c, w, 1) for c in chs[:-1])
        self.fpn = nn.ModuleList(cbr(w, w, 3) for _ in chs[:-1])
        self.fuse = cbr(len(chs) * w, w, 3)

    def forward(self, maps):
        top = maps[-1]
        hw = top.shape[-2:]
        pool = (lambda t, b: F.adaptive_avg_pool2d(t.cpu(), b).to(t.device)) if top.device.type == "mps" \
            else F.adaptive_avg_pool2d                    # MPS has no adaptive pool for non-divisible sizes
        p = [top] + [F.interpolate(m(pool(top, b)), hw, mode="bilinear", align_corners=False)
                     for b, m in zip(self.bins, self.ppm)]
        f = [l(x) for l, x in zip(self.lat, maps[:-1])] + [self.bottleneck(torch.cat(p, 1))]
        for i in range(len(f) - 2, -1, -1):
            f[i] = f[i] + F.interpolate(f[i + 1], f[i].shape[-2:], mode="bilinear", align_corners=False)
        f = [c(x) for c, x in zip(self.fpn, f[:-1])] + [f[-1]]
        hw0 = f[0].shape[-2:]
        f = [f[0]] + [F.interpolate(x, hw0, mode="bilinear", align_corners=False) for x in f[1:]]
        return self.fuse(torch.cat(f, 1))


def class_probs(seg_logits):
    """[B, 31, h, w] group logits -> ([B, 27, h, w] class probabilities, list of per-group log-softmax)."""
    out = seg_logits.new_zeros(seg_logits.shape[0], 27, *seg_logits.shape[2:])
    logp, s = [], 0
    for g, n in zip(GROUPS, GROUP_SIZES):
        lp = F.log_softmax(seg_logits[:, s:s + n], 1)
        logp.append(lp)
        out[:, g] = lp[:, 1:].exp()
        s += n
    return out, logp


def valid_cells(sizes, h, w):
    """bool [B, h, w]: output cells that cover the real (unpadded) image area; sizes = real (H, W) in pixels."""
    fh = torch.clamp(torch.ceil(sizes[:, 0].float() / STRIDE), 1, h).long()
    fw = torch.clamp(torch.ceil(sizes[:, 1].float() / STRIDE), 1, w).long()
    rows = torch.arange(h, device=sizes.device)[None, :, None] < fh[:, None, None]
    cols = torch.arange(w, device=sizes.device)[None, None, :] < fw[:, None, None]
    return rows & cols


def ratio_topk(probs, valid, ratio, min_k):
    """probs [B, C, h, w], valid [B, h, w], ratio [C] -> [B, C] mean of the top-k valid cells per class,
    k = clamp(round(ratio_c * n_valid_b), min_k, n_valid_b)."""
    B, C = probs.shape[:2]
    p = probs.flatten(2).masked_fill(~valid.flatten(1)[:, None], -1.0)
    srt = p.sort(2, descending=True).values.clamp(min=0)
    cs = srt.cumsum(2)
    n = valid.flatten(1).sum(1).float()                                    # [B]
    k = torch.round(ratio[None, :] * n[:, None]).clamp(min=min_k)
    k = torch.minimum(k, n[:, None]).clamp(min=1).long()                   # [B, C]
    return cs.gather(2, (k - 1)[..., None])[..., 0] / k.float()


class SegB(nn.Module):
    def __init__(self, name="convnext_tiny.dinov3_lvd1689m", pretrained=True, drop_path=0.1, width=256,
                 ratio=None, min_k=4):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        chs = [f["num_chs"] for f in self.net.feature_info]
        self.dec = UPerNet(chs, width)
        self.seg = nn.Conv2d(width, N_SEG, 1)
        self.part = nn.Conv2d(width, len(PARTS), 1)
        self.register_buffer("ratio", torch.as_tensor(ratio if ratio is not None else [0.001] * 27, dtype=torch.float32))
        self.min_k = min_k
        with torch.no_grad():                    # start every cell at "none" ~ 0.9 so early tag scores are low
            s = 0
            for n in GROUP_SIZES:
                self.seg.bias[s] = math.log(9 * (n - 1))
                s += n
            self.part.bias.fill_(-3.0)

    def forward(self, x, sizes):
        maps = self.net.forward_intermediates(x, indices=[0, 1, 2, 3], intermediates_only=True)
        with torch.autocast(device_type=x.device.type, enabled=False):      # decoder in fp32 (user decision)
            f = self.dec([m.float() for m in maps])
            seg, part = self.seg(f), self.part(f)
            probs, logp = class_probs(seg)
            valid = valid_cells(sizes, *seg.shape[-2:])
            tags = ratio_topk(probs, valid, self.ratio, self.min_k)
        return {"tags": tags, "seg": seg, "logp": logp, "part": part, "valid": valid, "probs": probs}

    def head_parameters(self):
        return list(self.dec.parameters()) + list(self.seg.parameters()) + list(self.part.parameters())
