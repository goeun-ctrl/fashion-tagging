"""Checks for LP-FT scheduling and EMA. Run:  python tests/test_optim_schedule.py  (needs torch + timm)"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path[:0] = [os.path.join(ROOT, "src")]
import torch  # noqa: E402
from fashion_tagging.models.heads import build_model, head_named_parameters  # noqa: E402
from fashion_tagging.training.optim import build_optimizers, build_schedulers  # noqa: E402

model = build_model("convnext_tiny.fb_in22k_ft_in1k_384", 27, pretrained=False)

# 1) baseline groups are unchanged: AdamW, backbone 1e-4 / head 1e-3, every group flagged head or not
(opt,) = build_optimizers(model, "adamw", 1e-4, 1e-3, 0.05)
assert [(g["lr"], g["is_head"]) for g in opt.param_groups] == [(1e-4, False), (1e-3, True)]

# 2) no LP: one cosine schedule for all groups (same factors as before the change)
(opt,) = build_optimizers(model, "adamw", 1e-4, 1e-3, 0.05)
(s,) = build_schedulers([opt], "cosine", warmup_steps=10, total_steps=100)
first = [g["lr"] for g in opt.param_groups]
assert abs(first[0] - 1e-4 * 0.1) < 1e-12 and abs(first[1] - 1e-3 * 0.1) < 1e-12

# 3) LP-FT: backbone lr 0 and head lr full for lp_steps, then both warm up from the start
(opt,) = build_optimizers(model, "adamw", 1e-4, 1e-3, 0.05)
(s,) = build_schedulers([opt], "cosine", warmup_steps=10, total_steps=100, lp_steps=20)
seen = []
for step in range(30):
    seen.append([g["lr"] for g in opt.param_groups])
    opt.step()
    s.step()
assert all(b == 0.0 and h == 1e-3 for b, h in seen[:20]), "LP phase: backbone 0, head full"
assert abs(seen[20][0] - 1e-4 * 0.1) < 1e-12 and abs(seen[20][1] - 1e-3 * 0.1) < 1e-12, "FT warmup restarts"

# 4) Muon groups get the same treatment (head/backbone never merged)
opts = build_optimizers(model, "muon", 1e-4, 1e-4, 0.05)            # equal lrs: still separate groups
assert all("is_head" in g for o in opts for g in o.param_groups)
assert any(g["is_head"] for g in opts[-1].param_groups) and any(not g["is_head"] for g in opts[-1].param_groups)

# 5) frozen backbone with AdamW: no grad -> no update and no weight decay
head = {id(p) for _, p in head_named_parameters(model)}
body = [p for p in model.parameters() if id(p) not in head]
for p in body:
    p.requires_grad_(False)
(opt,) = build_optimizers(model, "adamw", 1e-4, 1e-3, 0.05)
before = body[0].detach().clone()
model(torch.randn(2, 3, 64, 64)).sum().backward()
opt.step()
assert torch.equal(before, body[0]), "frozen backbone must not move"
for p in body:
    p.requires_grad_(True)

# 6) EMA weights move slower than the model
from timm.utils import ModelEmaV3  # noqa: E402
ema = ModelEmaV3(model, decay=0.99, use_warmup=True)
(opt,) = build_optimizers(model, "adamw", 1e-2, 1e-2, 0.0)
w0 = next(model.parameters()).detach().clone()
for k in range(5):
    opt.zero_grad()
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    opt.step()
    ema.update(model, step=k + 1)
w_model, w_ema = next(model.parameters()).detach(), next(ema.module.parameters()).detach()
assert (w_ema - w0).abs().sum() < (w_model - w0).abs().sum(), "EMA must lag the raw weights"
print("all optim/schedule tests passed")
