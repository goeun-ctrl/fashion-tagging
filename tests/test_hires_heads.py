"""Checks for the high-res heads in scripts/train_hires.py. Run:  python tests/test_hires_heads.py [--pretrained]
(needs torch + timm; --pretrained downloads convnext_tiny weights and gives the realistic padding numbers).

1) valid_mask reproduces the masked-GAP rule used for the R5_cnxt_hr_* runs.
2) C3 (query_ms) parameter count and shapes.
3) Padding. (a) Exact: junk written into the padded token positions does not change the query head's
   output when the mask is on (and does when it is off). (b) Informational: the same image alone vs. inside a
   larger canvas. Padded cells are excluded from the GAP / attention, but real cells near the padding still
   see it through the convolutions (pretrained biases make padded activations non-zero), so the two differ.
   This holds for the linear head of the R5_cnxt_hr_* runs too; train and test use the same fixed canvases.
4) Old checkpoints (config without 'head') still build the linear model.
5) masked_pool: padded cells never change gap / gmp / gapgmp; linear_ms (stride 16 + 32) builds, ~+33k params.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src")]
import torch  # noqa: E402
import train_hires as th  # noqa: E402

PRE = "--pretrained" in sys.argv
NAME = "convnext_tiny.fb_in22k_ft_in1k_384"
torch.manual_seed(0)

# 1) mask rule
sizes = torch.tensor([[800, 544], [416, 800], [33, 31]])
H, W, h, w = 1344, 800, 42, 25
m = th.valid_mask(sizes, h, w, H, W)
for k, (sh, sw) in enumerate(sizes.tolist()):
    fh, fw = max(1, -(-sh * h // H)), max(1, -(-sw * w // W))
    assert m[k].sum() == fh * fw and m[k, :fh, :fw].all(), (k, fh, fw)
print("1) valid_mask ok")

# 2) build + params
lin = th.build_hires_model({"model": NAME, "drop_path": 0.0}, 27, PRE).eval()
c3 = th.build_hires_model({"model": NAME, "drop_path": 0.0, "head": "query_ms"}, 27, PRE).eval()
n_lin, n_c3 = (sum(p.numel() for p in x.parameters()) for x in (lin, c3))
n_head = sum(p.numel() for p in c3.head_parameters())
print(f"2) params linear {n_lin:,}  query_ms {n_c3:,} (+{n_c3 - n_lin:,}; head {n_head:,})")
assert n_c3 <= th.PARAM_LIMIT and 1.9e6 < n_c3 - n_lin < 2.6e6
assert isinstance(th.build_hires_model({"model": NAME, "drop_path": 0.1}, 27, False), th.MaskedPoolConvNeXt)
print("4) old config -> linear ok")

# 3) padding
x = torch.randn(1, 3, 416, 288)
big = torch.zeros(1, 3, 544, 416)
big[..., :416, :288] = x
noisy = big.clone()
noisy[..., 416:, :] = torch.randn_like(noisy[..., 416:, :]) * 3
noisy[..., :, 288:] = torch.randn_like(noisy[..., :, 288:]) * 3
sz = torch.tensor([[416, 288]])
with torch.no_grad():
    for name, mdl in (("linear", lin), ("query_ms", c3)):
        a, b, c = mdl(x, sz), mdl(big, sz), mdl(noisy, sz)
        print(f"3) {name:8s} max|alone - zero pad| {float((a - b).abs().max()):.2e}  "
              f"max|alone - noise pad| {float((a - c).abs().max()):.2e}")
    # The mask itself: garbage in the padded token positions must not change the head output at all,
    # while the same garbage with the mask off must.
    maps = c3.net.forward_intermediates(big, indices=[-2, -1], intermediates_only=True)
    toks = [t.flatten(2).transpose(1, 2) for t in maps]
    pads = [~th.valid_mask(sz, *t.shape[-2:], *big.shape[-2:]).flatten(1) for t in maps]
    junk = [torch.where(p[..., None], torch.randn_like(t) * 10, t) for t, p in zip(toks, pads)]
    d_on = float((c3.head(toks, pads) - c3.head(junk, pads)).abs().max())
    none = [torch.zeros_like(p) for p in pads]
    d_off = float((c3.head(toks, none) - c3.head(junk, none)).abs().max())
    print(f"3) query_ms head: junk in padded tokens, mask on {d_on:.2e} / mask off {d_off:.2e}")
    assert d_on < 1e-5 and d_off > 1e-3, "padded tokens must be fully masked out of the attention"

# 5) masked pooling (linear / linear_ms heads): junk in padded cells never changes gap / gmp / gapgmp
f = torch.randn(2, 8, 6, 5)
valid = th.valid_mask(torch.tensor([[96, 64], [160, 160]]), 6, 5, 192, 160)
junk = torch.where(valid.unsqueeze(1), f, torch.randn_like(f) * 50 + 30)
for mode in ("gap", "gmp", "gapgmp"):
    assert torch.allclose(th.masked_pool(f, valid, mode), th.masked_pool(junk, valid, mode)), mode
k = 0
ref_avg = f[k, :, :3, :2].mean((1, 2))
assert torch.allclose(th.masked_pool(f, valid, "gap")[k], ref_avg, atol=1e-6)
assert torch.allclose(th.masked_pool(f, valid, "gmp")[k], f[k, :, :3, :2].amax((1, 2)))
ms = th.build_hires_model({"model": NAME, "drop_path": 0.0, "head": "linear_ms", "pool": "gapgmp"}, 27, PRE).eval()
n_ms = sum(p.numel() for p in ms.parameters())
print(f"5) masked_pool ok; linear_ms params {n_ms:,} (+{n_ms - n_lin:,} over linear)")
assert 0 < n_ms - n_lin < 60_000
with torch.no_grad():
    assert ms(x, sz).shape == (1, 27) and ms(big, sz).shape == (1, 27)

# 6) CSRA (linear_ms + pool csra): padded cells never change the class scores; reduces to mean when lam = 0
sc = torch.randn(2, 27, 6, 5)
junk_s = torch.where(valid.unsqueeze(1), sc, torch.randn_like(sc) * 50 + 30)
assert torch.allclose(th.csra_pool(sc, valid, 0.3, 1.0), th.csra_pool(junk_s, valid, 0.3, 1.0), atol=1e-5)
assert torch.allclose(th.csra_pool(sc, valid, 0.0, 1.0)[0], sc[0, :, :3, :2].mean((1, 2)), atol=1e-5)
big_t = th.csra_pool(sc, valid, 1.0, 1e4)[0] - sc[0, :, :3, :2].mean((1, 2))      # t -> inf: max over real cells
assert torch.allclose(big_t, sc[0, :, :3, :2].amax((1, 2)), atol=1e-3)
cs = th.build_hires_model({"model": NAME, "drop_path": 0.0, "head": "linear_ms", "pool": "csra"}, 27, PRE).eval()
n_cs = sum(p.numel() for p in cs.parameters())
with torch.no_grad():
    assert cs(x, sz).shape == (1, 27)
print(f"6) csra ok; params {n_cs:,} (+{n_cs - n_lin:,} over linear)")
print("all ok")
