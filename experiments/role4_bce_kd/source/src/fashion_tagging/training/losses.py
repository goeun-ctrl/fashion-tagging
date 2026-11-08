"""Multi-label losses on raw logits (never apply sigmoid before these).

  bce  BCEWithLogitsLoss, mean over images x classes (team baseline)
  asl  Asymmetric Loss (Ridnik et al., ICCV 2021). Plan defaults: gamma_neg=4, gamma_pos=0, clip=0.05.
       Reduced as a MEAN over images x classes like bce, so the same lr is a fair starting point.
       (The reference implementation returns the SUM, i.e. B*27 times larger.)

Unknown labels: a target of -1 means "not annotated" (external data whose labels do not cover every class).
Those entries are left out and the mean is taken over the known entries only. With no -1 in the batch
(the original data) both losses are exactly the plain mean, so the baseline is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ("bce", "asl")
UNKNOWN = -1


def _masked_mean(per_entry, targets):
    known = targets >= 0
    if bool(known.all()):
        return per_entry.mean()
    return (per_entry * known).sum() / known.sum().clamp(min=1)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        t = targets.clamp(min=0)                     # -1 -> 0 here; masked out below
        p = torch.sigmoid(logits.float())
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p   # probability shifting for negatives
        pos = t * (1 - p).pow(self.gamma_pos) * torch.log(p.clamp(min=self.eps))
        neg = (1 - t) * p_neg.pow(self.gamma_neg) * torch.log((1 - p_neg).clamp(min=self.eps))
        return _masked_mean(-(pos + neg), targets)


class BCELoss(nn.Module):
    def forward(self, logits, targets):
        per = F.binary_cross_entropy_with_logits(logits.float(), targets.clamp(min=0), reduction="none")
        return _masked_mean(per, targets)


def build_loss(name, gamma_neg=4.0, gamma_pos=0.0, clip=0.05):
    if name == "bce":
        return BCELoss()
    if name == "asl":
        return AsymmetricLoss(gamma_neg, gamma_pos, clip)
    raise ValueError(f"unknown loss {name!r}; choose from {list(LOSSES)}")
