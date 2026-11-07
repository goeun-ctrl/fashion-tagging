"""Checks for the negatives + gate path of scripts/train_hires.py. Run:  python tests/test_hires_negatives.py
(needs torch + timm, no download: models are built without pretrained weights).

1) BucketBatchSampler with n_core / extra_per_epoch or several pools: every core index once per epoch, exactly
   k images of each pool per epoch, different picks across epochs, no pools in eval mode.
2) load_grown: a 27-output checkpoint loads into a 28-output (gate) model; the first 27 rows are copied, the new
   row keeps its init, and the 27 class logits are unchanged.
3) gated / gate_split: images whose gate prob is below the threshold get no labels; others are untouched.
4) zoom-out / mosaic / old-photo: size and orientation kept, mosaic labels are the union, eval mode untouched.
5) garment masks: stride-16 grid, follow mirror / mosaic / zoom-out, negatives get exact all-zero masks, and the
   CSRA cell-level mask loss runs, ignores images without a mask and back-propagates into the class scorers.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_hires as th  # noqa: E402

# 1) sampler
n_core, n_neg, extra = 100, 40, 12
por = np.random.default_rng(0).random(n_core + n_neg) < 0.7
s = th.BucketBatchSampler(por, [416, 544], 4, 0, True, n_core=n_core, extra_per_epoch=extra)
seen = []
for ep in (1, 2):
    idx = [i for b in s.plan(ep) for i, _ in b]
    assert sorted(i for i in idx if i < n_core) == list(range(n_core)), "every positive once per epoch"
    neg = sorted(i for i in idx if i >= n_core)
    assert len(neg) == extra and len(set(neg)) == extra
    seen.append(set(neg))
    for b in s.plan(ep):                                     # one canvas (orientation x level) per batch
        assert len({(bool(por[i]), lv) for i, lv in b}) == 1
assert seen[0] != seen[1], "negatives are redrawn every epoch"
ev = th.BucketBatchSampler(por[:n_core], [800], 4, 0, False)
assert sorted(i for b in ev.plan(1) for i, _ in b) == list(range(n_core))
# two pools (negatives, person positives) with their own per-epoch counts
n_a, n_b = 30, 25
por2 = np.random.default_rng(1).random(n_core + n_a + n_b) < 0.6
s2 = th.BucketBatchSampler(por2, [416], 4, 0, True, pools=[(n_core, n_core + n_a, 7), (n_core + n_a, n_core + n_a + n_b, 5)])
idx = [i for b in s2.plan(3) for i, _ in b]
assert sorted(i for i in idx if i < n_core) == list(range(n_core))
assert sum(n_core <= i < n_core + n_a for i in idx) == 7 and sum(i >= n_core + n_a for i in idx) == 5
print("1) sampler ok")

# 2) load_grown
torch.manual_seed(0)
NAME = "convnext_atto"
old = th.build_hires_model({"model": NAME, "drop_path": 0.0}, 27, False).eval()
new = th.build_hires_model({"model": NAME, "drop_path": 0.0, "gate": True}, 28, False).eval()
fresh = new.state_dict()["net.head.fc.weight"][27].clone()
th.load_grown(new, old.state_dict())
assert torch.equal(new.state_dict()["net.head.fc.weight"][:27], old.state_dict()["net.head.fc.weight"])
assert torch.equal(new.state_dict()["net.head.fc.weight"][27], fresh)
x, sz = torch.randn(2, 3, 128, 96), torch.tensor([[128, 96], [96, 64]])
with torch.no_grad():
    assert torch.allclose(new(x, sz)[:, :27], old(x, sz), atol=1e-5)
q_old = th.build_hires_model({"model": NAME, "drop_path": 0.0, "head": "query_ms"}, 27, False)
q_new = th.build_hires_model({"model": NAME, "drop_path": 0.0, "head": "query_ms"}, 28, False)
th.load_grown(q_new, q_old.state_dict())                      # query / fc_w / fc_b grow 27 -> 28
assert torch.equal(q_new.head.query[:27], q_old.head.query)
print("2) load_grown ok")

# 2b) unknown (-1) class targets of the person positives are masked out of the class loss
from fashion_tagging.training.losses import AsymmetricLoss  # noqa: E402
lgt = torch.randn(3, 27)
tg = torch.tensor([[1.0] * 3 + [0.0] * 24, [0.0] * 27, [-1.0] * 27])
asl = AsymmetricLoss()
assert torch.allclose(asl(lgt, tg), asl(lgt[:2], tg[:2]), atol=1e-6), "all -1 rows must not change the loss"
print("2b) unknown rows masked ok")

# 3) gating
lg = np.array([[2.0, -1.0, 3.0], [2.0, 1.0, -3.0]])        # 2 classes + gate column
cls, g = th.gate_split(lg, 2)
pred = th.binarize(th.sigmoid(cls), 0.5)
out = th.gated(pred, g, 0.5)
assert out[0].tolist() == [True, False] and out[1].tolist() == [False, False]
assert th.gate_split(lg[:, :2], 2)[1] is None and th.gated(pred, None, 0.5) is pred
print("3) gating ok")

# 4) OOD augmentations: image size / orientation kept, mosaic labels = union, nothing applied in eval mode
import random as _r  # noqa: E402
import tempfile  # noqa: E402
from PIL import Image  # noqa: E402
tmp = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp, "images"))
sizes_wh = [(300, 450), (500, 320), (240, 360), (400, 400)]
for k, (w, h) in enumerate(sizes_wh):
    Image.new("RGB", (w, h), (40 * k, 100, 200)).save(os.path.join(tmp, "images", f"{k}.jpg"))
tg = np.zeros((4, 27), np.float32); tg[0, 0] = tg[1, 5] = tg[2, 9] = tg[3, 20] = 1
mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ds = th.HiResDataset(tmp, [f"{k}.jpg" for k in range(4)], tg, mean, std, True, 1333, 0.5, zoom_p=1.0,
                     mosaic_p=1.0, mosaic_n=4, old_p=1.0)
_r.seed(0); np.random.seed(0)
for _ in range(10):
    x, yy, i, cv = ds[(0, 416)]
    assert yy[0] == 1 and yy.sum() >= 2, "mosaic keeps the base label and adds the tiles' labels"
    assert x.shape[1] >= x.shape[2], "portrait base stays portrait"
    assert set(np.unique(yy.numpy())) <= {0.0, 1.0}
img = Image.open(os.path.join(tmp, "images", "1.jpg"))
assert ds.zoom_out(img).size == img.size and ds.old_photo(img).size == img.size and ds.old_photo(img).mode == "RGB"
m, idx = ds.mosaic(img.convert("RGB"), 1)
assert m.size == img.size and idx[0] == 1
ev = th.HiResDataset(tmp, [f"{k}.jpg" for k in range(4)], tg, mean, std, False, 1333, 0.0, zoom_p=1.0, mosaic_p=1.0,
                     mosaic_n=4, old_p=1.0)
assert torch.equal(ev[(2, 416)][1], torch.from_numpy(tg[2])), "eval mode: no mosaic"
# 6- and 9-tile grids: size kept, k tiles, base image first and never repeated as its own tile, union of labels
for k in (6, 9):
    big = th.HiResDataset(tmp, [f"{j}.jpg" for j in range(4)], tg, mean, std, True, 1333, 0.0, mosaic_p=1.0,
                          mosaic_n=4, mosaic_sizes=(k,))
    for base in (0, 1):                                      # portrait and landscape bases
        im0 = big.load(base)
        m, idx = big.mosaic(im0, base)
        assert m.size == im0.size and len(idx) == k and idx[0] == base and base not in idx[1:]
        x, yy, _, _ = big[(base, 416)]
        assert yy[base * 5] == 1 and yy.sum() >= 2 and x.shape[1:] == big[(base, 416)][0].shape[1:]
print("4) OOD augmentations ok (mosaic 2/4/6/9)")

# 5) garment masks follow the augmentations and reach the CSRA mask loss
mdir = os.path.join(tmp, "masks")
os.makedirs(mdir)
for k, (w, h) in enumerate(sizes_wh):
    sc = 128 / max(w, h); gh, gw = round(h * sc), round(w * sc)
    m = np.zeros((1, gh, gw), bool); m[0, :, : gw // 3] = True               # garment on the LEFT third
    np.savez_compressed(os.path.join(mdir, f"{k}.jpg.npz"), cls=np.array([[0, 5, 9, 20][k]], np.uint8),
                        bits=np.packbits(m.reshape(-1)), shape=np.array([gh, gw]))
kinds = np.array([1, 1, 1, 0], np.int8)                                       # index 3 = a negative
plain = th.HiResDataset(tmp, [f"{k}.jpg" for k in range(4)], tg, mean, std, True, 1333, 0.0, mask_dir=mdir, mask_kind=kinds)
x, yy, i, cv, sf, mk, ok = plain[(0, 416)]
assert ok and mk.shape == (27, -(-x.shape[1] // 16), -(-x.shape[2] // 16)) and mk[0].any() and not mk[1:].any()
assert mk[0, :, : mk.shape[2] // 4].all() and not mk[0, :, -2:].any(), "mask stays on the left"
flip = th.HiResDataset(tmp, [f"{k}.jpg" for k in range(4)], tg, mean, std, True, 1333, 1.0, mask_dir=mdir, mask_kind=kinds)
mk2 = flip[(0, 416)][5]
assert mk2[0, :, -2:].all() and not mk2[0, :, :2].any(), "mirror moves the mask to the right"
neg = plain[(3, 416)]
assert neg[6] and not neg[5].any(), "negatives: all-zero mask, still supervised"
aug = th.HiResDataset(tmp, [f"{k}.jpg" for k in range(4)], tg, mean, std, True, 1333, 0.5, mosaic_p=1.0, mosaic_n=3,
                      zoom_p=1.0, mask_dir=mdir, mask_kind=kinds)
_r.seed(3)
x, yy, i, cv, sf, mk, ok = aug[(0, 416)]
present = {j for j in range(27) if mk[j].any()}
assert present and present <= {int(j) for j in np.flatnonzero(yy.numpy())}, "mask classes are a subset of the labels"
b = th.pad_collate([plain[(0, 416)], plain[(2, 416)], plain[(3, 416)]])
assert b[5].shape[1] == 27 and b[5].shape[2] == b[0].shape[2] // 16 and b[6].tolist() == [True, True, True]
model = th.build_hires_model({"model": NAME, "drop_path": 0.0, "head": "linear_ms", "pool": "csra"}, 27, False)
out, smaps = model(b[0], b[2], return_maps=True)
from fashion_tagging.training.losses import AsymmetricLoss as _ASL  # noqa: E402
l = th.mask_loss(smaps, b[5], b[6], _ASL(), 27)
l_none = th.mask_loss(smaps, b[5], torch.tensor([False, False, True]), _ASL(), 27)
assert torch.isfinite(l) and torch.isfinite(l_none) and not torch.allclose(l, l_none)
l.backward()
assert model.scorers[0].weight.grad is not None
print("5) masks ok")

# 6) labelled extras (--ext_list / --ext_labels): pseudo-label rule of scripts/pseudo_label.py
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("pseudo_label", os.path.join(ROOT, "scripts", "pseudo_label.py"))
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
d = tempfile.mkdtemp()
probs = np.array([[0.9, 0.5, 0.01] + [0.0] * 24, [0.8, 0.04, 0.2] + [0.0] * 24], np.float32)
np.savez(os.path.join(d, "pl_probs_0.npz"), filenames=np.array(["/a.jpg", "/b.jpg"]), probs=probs,
         n_crops=np.array([2, 0]), categories=np.array([f"c{j}" for j in range(27)]))
pl.label(type("A", (), {"out": os.path.join(d, "pl"), "pos_thr": 0.8, "neg_thr": 0.05})())
t = np.load(os.path.join(d, "pl.npz"))["targets"]
assert t[0, :3].tolist() == [1, -1, 0] and t[1, :3].tolist() == [1, 0, -1], t[:, :3]
print("6) pseudo labels ok")
print("all ok")
