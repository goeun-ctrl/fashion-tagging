"""Checks for unknown-label (-1) masking in the losses. Run:  python tests/test_losses.py  (needs torch)"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(ROOT, "src")]
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from fashion_tagging.training.losses import AsymmetricLoss, BCELoss  # noqa: E402

torch.manual_seed(0)
logits = torch.randn(6, 27)
full = (torch.rand(6, 27) > 0.7).float()

# 1) no unknowns -> exactly the plain mean (team baseline unchanged)
assert torch.allclose(BCELoss()(logits, full), F.binary_cross_entropy_with_logits(logits, full))
asl = AsymmetricLoss()
ref = AsymmetricLoss()(logits, full)
assert torch.isfinite(ref)

# 2) with unknowns -> mean over the known entries only
part = full.clone()
part[:, ::3] = -1
known = part >= 0
bce_known = F.binary_cross_entropy_with_logits(logits[known], full[known])
assert torch.allclose(BCELoss()(logits, part), bce_known), "BCE must average known entries only"
per = []
for i, j in zip(*known.nonzero(as_tuple=True)):
    per.append(asl(logits[i:i + 1, j:j + 1], full[i:i + 1, j:j + 1]))
assert torch.allclose(asl(logits, part), torch.stack(per).mean(), atol=1e-6), "ASL must average known entries only"

# 3) unknown entries get no gradient; changing their logits does not change the loss
for loss in (BCELoss(), asl):
    x = logits.clone().requires_grad_(True)
    loss(x, part).backward()
    assert torch.all(x.grad[~known] == 0), "unknown entries must not receive gradient"
    y = logits.clone()
    y[~known] += 5.0
    assert torch.allclose(loss(logits, part), loss(y, part))

# 4) a batch whose labels are all unknown gives 0, not NaN
assert loss(logits, torch.full_like(full, -1)).item() == 0.0
print("all loss tests passed")
