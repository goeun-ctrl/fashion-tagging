"""Redesign B losses. All per-cell terms are averaged over the valid cells of each image first (user decision:
per-image mean), then over the images that have any valid cell."""
import torch
import torch.nn.functional as F

from .spec import GROUPS, GROUP_SIZES

EPS = 1e-6


def group_weights(class_w, none_w=1.0):
    """[27] class weights -> list of per-group [n] weights (index 0 = none)."""
    return [torch.cat([class_w.new_tensor([none_w]), class_w[g]]) for g in GROUPS]


def seg_ce(logp, t, v, gw, img_w):
    """Soft cross-entropy per group. logp: list of [B, n, h, w]; t [B, 31, h, w]; v [B, 4, h, w];
    gw: list of [n] weights; img_w [B] per-image weight (COCO seg weight). Cell weight = sum_c t_c w_c."""
    num = t.new_zeros(t.shape[0])
    den = t.new_zeros(t.shape[0])
    s = 0
    for gi, (lp, n) in enumerate(zip(logp, GROUP_SIZES)):
        tg = t[:, s:s + n]
        cw = (tg * gw[gi][None, :, None, None]).sum(1)
        ce = -(tg * lp).sum(1)
        m = v[:, gi].float()
        num = num + (ce * cw * m).flatten(1).sum(1)
        den = den + m.flatten(1).sum(1)
        s += n
    has = den > 0
    if not bool(has.any()):
        return t.sum() * 0.0
    per = num[has] / den[has] * img_w[has]
    return per.mean()


def seg_dice(logp, t, hard, img_w):
    """Soft Dice per class over the GT-supervised cells (hard), only for classes present in the target."""
    s, terms = 0, []
    for gi, (lp, n) in enumerate(zip(logp, GROUP_SIZES)):
        p = lp[:, 1:].exp()
        tg = t[:, s + 1:s + n]
        m = hard[:, gi:gi + 1].float()
        inter = (p * tg * m).flatten(2).sum(2)
        den = (p * m).flatten(2).sum(2) + (tg * m).flatten(2).sum(2)
        present = (tg * m).flatten(2).sum(2) > 0.5                       # at least half a cell of the class
        d = 1 - (2 * inter + 1) / (den + 1)
        terms.append((d * present, present))
        s += n
    d = torch.cat([x for x, _ in terms], 1)
    pr = torch.cat([y for _, y in terms], 1).float()
    cnt = pr.sum(1)
    has = cnt > 0
    if not bool(has.any()):
        return t.sum() * 0.0
    return (d.sum(1)[has] / cnt[has] * img_w[has]).mean()


def region_ce(seg, t, use, groups, class_w, thr=0.5, min_cells=16):
    """Region consistency (user decision 2026-11-23, option B): for every GT class region of a softmax group,
    region = cells whose soft target >= thr (boundary cells left out), kept when >= min_cells cells; the region's
    mean LOGITS over the group's garment classes ("none" excluded) -> softmax -> CE against that class, weighted
    by the class weight. seg [B, 31, h, w] raw group logits; t [B, 31, h, w] soft targets; use [B] bool
    (images with real GT masks, i.e. Fashionpedia); groups: indices into spec.GROUPS. Per-image mean over its
    regions (weights multiply, divide by the region count, like seg_ce), then mean over images with a region."""
    num = seg.new_zeros(seg.shape[0])
    den = seg.new_zeros(seg.shape[0])
    starts = [sum(GROUP_SIZES[:i]) for i in range(len(GROUP_SIZES))]
    for gi in groups:
        s, n = starts[gi], GROUP_SIZES[gi]
        lg = seg[:, s + 1:s + n].float()                                   # [B, n-1, h, w] garment logits
        m = (t[:, s + 1:s + n] >= thr) & use[:, None, None, None]           # [B, n-1, h, w] class regions
        cnt = m.flatten(2).sum(2)                                          # [B, n-1]
        ok = cnt >= min_cells
        if not bool(ok.any()):
            continue
        mean = torch.einsum("bjhw,bkhw->bjk", m.float(), lg) / cnt.clamp(min=1)[..., None].float()
        ce = -torch.diagonal(F.log_softmax(mean, 2), dim1=1, dim2=2)        # [B, n-1]: region j vs class j
        w = class_w[GROUPS[gi]][None].expand_as(ce)
        num = num + (ce * w * ok).sum(1)
        den = den + ok.sum(1)
    has = den > 0
    if not bool(has.any()):
        return seg.sum() * 0.0
    return (num[has] / den[has]).mean()


def part_loss(logits, pt, pv):
    """Parts: per-cell BCE over valid cells + soft Dice for present parts, 1:1."""
    m = pv[:, None].float()
    n = m.flatten(1).sum(1) * logits.shape[1]
    has = n > 0
    if not bool(has.any()):
        return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits, pt, reduction="none")
    bce = (bce * m).flatten(1).sum(1)[has] / n[has]
    p = torch.sigmoid(logits)
    inter = (p * pt * m).flatten(2).sum(2)
    den = (p * m).flatten(2).sum(2) + (pt * m).flatten(2).sum(2)
    present = (pt * m).flatten(2).sum(2) > 0.5
    d = (1 - (2 * inter + 1) / (den + 1)) * present
    cnt = present.float().sum(1)
    dice = torch.where(cnt > 0, d.sum(1) / cnt.clamp(min=1), torch.zeros_like(cnt))
    return bce.mean() + dice[has].mean()


def tag_loss(p, y, soft, class_w, alpha):
    """BCE on the pooled tag probability p [B, 27] (labels 1 / 0 / -1 = unknown), class-weighted;
    rows with a teacher score: (1 - alpha) * hard + alpha * soft BCE (the teacher covers all 27 classes)."""
    p = p.clamp(EPS, 1 - EPS)
    known = (y >= 0).float()
    hard = F.binary_cross_entropy(p, y.clamp(min=0), reduction="none") * class_w[None] * known
    hard_row = hard.sum(1) / known.sum(1).clamp(min=1)
    has_t = ~torch.isnan(soft[:, 0])
    row = hard_row
    if bool(has_t.any()) and alpha > 0:
        st = torch.nan_to_num(soft, nan=0.0)
        sb = (F.binary_cross_entropy(p, st, reduction="none") * class_w[None]).mean(1)
        row = torch.where(has_t, (1 - alpha) * hard_row + alpha * sb, hard_row)
    keep = (known.sum(1) > 0) | has_t
    return row[keep].mean() if bool(keep.any()) else p.sum() * 0.0
