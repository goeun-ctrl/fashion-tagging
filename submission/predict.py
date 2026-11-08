#!/usr/bin/env python
"""Offline inference for the MAT4091 Fashion Category Tagging submission.

    python predict.py --input_dir TEST_IMAGES --output predictions.jsonl

One JSON object per image: {"test_id":"<file name without extension>","labels":["jacket","pants"]}.
`labels` holds exact category names (the 27 names of categories.txt) and is [] when nothing is predicted.

Self-contained on purpose: the model code below is copied from scripts/train_hires.py of the team repository
(tests/test_predict.py checks that both give identical outputs), the weights and every setting come from
model.pt next to this file, and nothing is downloaded (timm models are created with pretrained=False).

model.pt (written by scripts/make_submission.py) is a dict with:
  format, config (model / head / pool / csra / query settings, gate), state_dict, categories,
  preprocess (short, max_long, mean, std), tta ("flip" or "none"),
  thresholds (global + per_class by exact name), gate_threshold, notes.

Pipeline per image: EXIF transpose + RGB -> keep aspect ratio (short side `short`, long side <= min(max_long,
2*short), bilinear) -> normalise -> pad to the fixed canvas of its orientation -> network with pooling over the
real image area only -> (optional) mean of the logits of the image and its mirror -> sigmoid -> thresholds ->
(optional) gate: images the gate calls "no garment" get []. No softmax, no forced top-1.

Exit codes: 0 ok, 2 bad input (missing folder, no images, duplicate test_id), 3 model / parameter-limit problem,
4 unreadable images with --strict.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageOps

FORMAT = "fashion-submission-v1"
PARAM_LIMIT = 50_000_000
STRIDE = 32
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR

# exact names and order of categories.txt (names contain commas; one name = one class)
CATEGORIES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "pants", "shorts", "skirt",
    "coat", "dress", "jumpsuit", "cape", "glasses", "hat", "headband, head covering, hair accessory", "tie", "glove",
    "watch", "belt", "leg warmer", "tights, stockings", "sock", "shoe", "bag, wallet", "scarf", "umbrella",
]


# ------------------------------------------------------------------------------------------------ model code
# (copied from scripts/train_hires.py and src/fashion_tagging/models/heads.py; keep in sync via tests/test_predict.py)
def valid_mask(sizes, h, w, H, W):
    """bool [B, h, w]: feature cells covering the real image area of each padded input (H, W) -> (h, w)."""
    fh = torch.clamp(torch.ceil(sizes[:, 0].float() * h / H), min=1).long()
    fw = torch.clamp(torch.ceil(sizes[:, 1].float() * w / W), min=1).long()
    dev = sizes.device
    rows = torch.arange(h, device=dev)[None, :, None] < fh[:, None, None]
    cols = torch.arange(w, device=dev)[None, None, :] < fw[:, None, None]
    return rows & cols


def masked_pool(f, valid, mode="gap"):
    m = valid.unsqueeze(1)
    avg = (f * m.to(f.dtype)).sum((2, 3)) / m.sum((2, 3)).to(f.dtype)
    if mode == "gap":
        return avg
    mx = f.masked_fill(~m, torch.finfo(f.dtype).min).amax((2, 3))
    return mx if mode == "gmp" else 0.5 * (avg + mx)


def csra_pool(s, valid, lam, t):
    m = valid.unsqueeze(1).expand_as(s).flatten(2)
    s = s.flatten(2)
    mean = (s * m).sum(-1) / m.sum(-1)
    w = torch.softmax((t * s).masked_fill(~m, float("-inf")), dim=-1)
    return mean + lam * (w * s).sum(-1)


class MaskedPoolConvNeXt(nn.Module):
    """head = "linear": timm ConvNeXt with its own head, pooled over the real image area only."""

    def __init__(self, name, num_classes, pool="gap"):
        super().__init__()
        self.pool = pool
        self.net = timm.create_model(name, pretrained=False, num_classes=num_classes)

    def forward(self, x, sizes):
        f = self.net.forward_features(x)
        valid = valid_mask(sizes.to(f.device), *f.shape[-2:], *x.shape[-2:])
        pooled = masked_pool(f, valid, self.pool)[:, :, None, None]
        hd = self.net.head
        return hd.fc(hd.drop(hd.pre_logits(hd.flatten(hd.norm(pooled)))))


class MaskedPoolMSConvNeXt(nn.Module):
    """head = "linear_ms": stride-16 and stride-32 maps pooled (gap / gmp / gapgmp) or CSRA-scored."""

    def __init__(self, name, num_classes, pool="gapgmp", csra_lambda=0.3, csra_t=1.0):
        super().__init__()
        self.net = timm.create_model(name, pretrained=False, num_classes=0, global_pool="")
        info = self.net.feature_info[-2:]
        chs = [f["num_chs"] for f in info]
        self.pool, self.lam, self.t = pool, csra_lambda, csra_t
        if pool == "csra":
            self.norms = nn.ModuleList(nn.LayerNorm(c) for c in chs)
            self.scorers = nn.ModuleList(nn.Linear(c, num_classes, bias=False) for c in chs)
            self.bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.norm = nn.LayerNorm(sum(chs))
            self.fc = nn.Linear(sum(chs), num_classes)

    def forward(self, x, sizes):
        maps = self.net.forward_intermediates(x, indices=[-2, -1], intermediates_only=True)
        H, W = x.shape[-2:]
        valids = [valid_mask(sizes.to(m.device), *m.shape[-2:], H, W) for m in maps]
        if self.pool != "csra":
            pooled = [masked_pool(m, v, self.pool) for m, v in zip(maps, valids)]
            return self.fc(self.norm(torch.cat(pooled, 1)))
        out = self.bias
        for m, v, norm, scorer in zip(maps, valids, self.norms, self.scorers):
            s = scorer(norm(m.permute(0, 2, 3, 1))).permute(0, 3, 1, 2).float()
            out = out + csra_pool(s, v, self.lam, self.t)
        return out


class QueryHead(nn.Module):
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
        self.fc_w = nn.Parameter(torch.empty(num_classes, dim))
        self.fc_b = nn.Parameter(torch.zeros(num_classes))

    def decode(self, kv, pad_mask=None):
        q = self.query.unsqueeze(0).expand(kv.size(0), -1, -1)
        a, _ = self.attn(q, kv, kv, key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(q + a)
        x = self.norm2(x + self.ffn(x))
        return (x * self.fc_w).sum(-1) + self.fc_b


class MultiScaleQueryHead(QueryHead):
    def __init__(self, in_dims, num_classes, dim=384, num_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__(in_dims[-1], num_classes, dim, num_heads, ffn_mult, dropout)
        self.extra_norms = nn.ModuleList(nn.LayerNorm(d) for d in in_dims[:-1])
        self.extra_projs = nn.ModuleList(nn.Linear(d, dim) for d in in_dims[:-1])
        self.level_embed = nn.Parameter(torch.empty(len(in_dims), dim))

    def forward(self, levels, pad_masks=None):
        norms = list(self.extra_norms) + [self.in_norm]
        projs = list(self.extra_projs) + [self.proj]
        kv = [p(n(t)) + self.level_embed[i] for i, (n, p, t) in enumerate(zip(norms, projs, levels))]
        mask = None if pad_masks is None else torch.cat(pad_masks, dim=1)
        return self.decode(torch.cat(kv, dim=1), mask)


class MaskedQueryConvNeXt(nn.Module):
    """head = "query_ms"."""

    def __init__(self, name, num_classes, dim=384, heads=8, dropout=0.1):
        super().__init__()
        self.net = timm.create_model(name, pretrained=False, num_classes=0, global_pool="")
        info = self.net.feature_info[-2:]
        self.head = MultiScaleQueryHead([f["num_chs"] for f in info], num_classes, dim, heads, dropout=dropout)

    def forward(self, x, sizes):
        maps = self.net.forward_intermediates(x, indices=[-2, -1], intermediates_only=True)
        H, W = x.shape[-2:]
        tokens, pads = [], []
        for m in maps:
            tokens.append(m.flatten(2).transpose(1, 2))
            pads.append(~valid_mask(sizes.to(m.device), *m.shape[-2:], H, W).flatten(1))
        return self.head(tokens, pads)


def build_model(cfg, num_out):
    """cfg: the `config` dict of model.pt. num_out = number of classes (+1 when the model has a gate output)."""
    head = cfg.get("head", "linear")
    if head == "query_ms":
        return MaskedQueryConvNeXt(cfg["model"], num_out, cfg.get("query_dim", 384), cfg.get("query_heads", 8),
                                   cfg.get("head_dropout", 0.1))
    if head == "linear_ms":
        return MaskedPoolMSConvNeXt(cfg["model"], num_out, cfg.get("pool", "gapgmp"), cfg.get("csra_lambda", 0.3),
                                    cfg.get("csra_t", 1.0))
    if head == "linear":
        return MaskedPoolConvNeXt(cfg["model"], num_out, cfg.get("pool", "gap"))
    raise ValueError(f"unknown head {head!r}")


# ------------------------------------------------------------------------------------------------ preprocessing
def ceil32(v):
    return math.ceil(v / STRIDE) * STRIDE


def long_cap(level, max_long):
    return min(max_long, 2 * level)


def target_size(w, h, short, cap):
    scale = min(short / min(w, h), cap / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def canvas(level, portrait, max_long):
    long_, short_ = ceil32(long_cap(level, max_long)), ceil32(level)
    return (long_, short_) if portrait else (short_, long_)          # (H, W)


def to_rgb(img):
    return ImageOps.exif_transpose(img).convert("RGB")


def load_tensors(path, pp, mirror_too):
    """-> (x [3,h,w] float32 normalised, mirrored x or None, portrait). Same arithmetic as HiResDataset (eval)."""
    short, max_long = pp["short"], pp["max_long"]
    mean = np.asarray(pp["mean"], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(pp["std"], dtype=np.float32).reshape(3, 1, 1)
    with Image.open(path) as im:
        im.load()
        img = to_rgb(im)
    portrait = img.size[1] >= img.size[0]
    img = img.resize(target_size(*img.size, short, long_cap(short, max_long)), BILINEAR)

    def tensor(i):
        x = np.asarray(i, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(np.ascontiguousarray((x - mean) / std))
    return tensor(img), (tensor(ImageOps.mirror(img)) if mirror_too else None), portrait


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, paths, pp, tta):
        self.paths, self.pp, self.tta = paths, pp, tta

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        try:
            x, xm, portrait = load_tensors(self.paths[k], self.pp, self.tta)
            return k, x, xm, portrait, None
        except Exception as e:                                     # unreadable / truncated image
            return k, None, None, True, f"{type(e).__name__}: {e}"


class Collate:
    """Pads a batch of one orientation to its fixed canvas. A class (not a nested function) so that DataLoader
    workers can receive it by pickling on Windows / macOS (spawn)."""

    def __init__(self, pp):
        self.pp = pp

    def __call__(self, batch):
        pp = self.pp
        ok = [b for b in batch if b[4] is None]
        failed = [(b[0], b[4]) for b in batch if b[4] is not None]
        if not ok:
            return None, None, None, None, failed
        portrait = ok[0][3]
        ch, cw = canvas(pp["short"], portrait, pp["max_long"])
        H = ceil32(max([ch] + [b[1].shape[1] for b in ok]))
        W = ceil32(max([cw] + [b[1].shape[2] for b in ok]))

        def pad(i):
            out = torch.zeros(len(ok), 3, H, W)
            for r, b in enumerate(ok):
                t = b[i]
                out[r, :, :t.shape[1], :t.shape[2]] = t
            return out
        x = pad(1)
        xm = pad(2) if ok[0][2] is not None else None
        sizes = torch.tensor([list(b[1].shape[1:]) for b in ok])
        return [b[0] for b in ok], x, xm, sizes, failed


def read_orientation(path):
    """portrait flag from the header only (no pixel decoding); raises for unreadable files."""
    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(274, 1) in (5, 6, 7, 8):
            w, h = h, w
    return h >= w


# ------------------------------------------------------------------------------------------------ main
def list_images(input_dir):
    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(IMAGE_EXTS))
    ids = [os.path.splitext(f)[0] for f in files]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    return files, ids, dup


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path}: unexpected format {ck.get('format')!r} (expected {FORMAT!r})")
    cats = list(ck["categories"])
    if cats != CATEGORIES:
        raise ValueError("categories in model.pt differ from the official 27 names / order")
    cfg = ck["config"]
    model = build_model(cfg, len(cats) + (1 if cfg.get("gate") else 0))
    model.load_state_dict(ck["state_dict"])                          # strict
    n_params = sum(p.numel() for p in model.parameters())
    return ck, model.to(device).eval(), n_params


def threshold_vector(th):
    t = np.full(len(CATEGORIES), float(th["global"]), dtype=np.float64)
    for name, v in th.get("per_class", {}).items():
        if name not in CATEGORIES:
            raise ValueError(f"threshold for unknown class {name!r}")
        t[CATEGORIES.index(name)] = float(v)
    return t


@torch.no_grad()
def infer(model, ck, paths, device, batch_size, workers, use_amp):
    """-> logits [N, n_out] (float32, NaN for failed images), dict failed {index: message}."""
    pp, tta = ck["preprocess"], ck.get("tta", "flip") == "flip"
    n_out = len(CATEGORIES) + (1 if ck["config"].get("gate") else 0)
    logits = np.full((len(paths), n_out), np.nan, dtype=np.float32)
    failed = {}
    groups = {True: [], False: []}
    for k, p in enumerate(paths):
        try:
            groups[read_orientation(p)].append(k)
        except Exception as e:
            failed[k] = f"{type(e).__name__}: {e}"
    batches = [g[i:i + batch_size] for g in groups.values() for i in range(0, len(g), batch_size)]
    ds = ImageSet(paths, pp, tta)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=workers, collate_fn=Collate(pp))
    for idx, x, xm, sizes, bad in dl:
        failed.update(dict(bad))
        if idx is None:
            continue
        outs = []
        for inp in (x, xm) if xm is not None else (x,):
            inp = inp.to(device)
            if device.type == "cuda":
                inp = inp.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outs.append(model(inp, sizes.to(device)).float().cpu())
        logits[idx] = (sum(outs) / len(outs)).numpy()
    return logits, failed


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -500, 500)))


def decide(logits, ck):
    """logits [N, n_out] -> bool [N, 27] predictions (threshold per class, then the gate)."""
    n_cls = len(CATEGORIES)
    pred = sigmoid(logits[:, :n_cls]) >= threshold_vector(ck["thresholds"])[None, :]
    if logits.shape[1] > n_cls:
        pred &= (sigmoid(logits[:, n_cls]) >= float(ck.get("gate_threshold", 0.5)))[:, None]
    return pred


def write_jsonl(path, ids, pred):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for i, row in zip(ids, pred):
            labels = [CATEGORIES[j] for j in np.flatnonzero(row)]
            f.write(json.dumps({"test_id": i, "labels": labels}, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir")
    ap.add_argument("--output", default="predictions.jsonl")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt"))
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2, help="image-loading workers (0 = in the main process)")
    ap.add_argument("--no_amp", action="store_true", help="fp32 on GPU (default: fp16 autocast like training)")
    ap.add_argument("--strict", action="store_true", help="exit 4 when an image cannot be read (default: [] + warning)")
    ap.add_argument("--dump_logits", default=None, help="also save logits to this .npz (debugging)")
    ap.add_argument("--print_params", action="store_true", help="print the parameter count and exit")
    a = ap.parse_args(argv)

    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    try:
        ck, model, n_params = load_model(a.model, device)
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        return 3
    print(f"parameters: {n_params:,} (limit {PARAM_LIMIT:,})", file=sys.stderr)
    if n_params > PARAM_LIMIT:
        print("ERROR: parameter limit exceeded", file=sys.stderr)
        return 3
    if a.print_params:
        print(f"{n_params:,}")
        return 0
    if not a.input_dir or not os.path.isdir(a.input_dir):
        print(f"ERROR: --input_dir {a.input_dir!r} is not a folder", file=sys.stderr)
        return 2
    files, ids, dup = list_images(a.input_dir)
    if not files:
        print(f"ERROR: no images ({', '.join(IMAGE_EXTS)}) in {a.input_dir}", file=sys.stderr)
        return 2
    if dup:
        print(f"ERROR: several files share a test_id (name without extension): {dup[:5]}", file=sys.stderr)
        return 2

    paths = [os.path.join(a.input_dir, f) for f in files]
    t0 = time.perf_counter()
    logits, failed = infer(model, ck, paths, device, a.batch_size, a.workers, device.type == "cuda" and not a.no_amp)
    ok = np.ones(len(paths), dtype=bool)
    for k in failed:
        ok[k] = False
    pred = np.zeros((len(paths), len(CATEGORIES)), dtype=bool)
    pred[ok] = decide(logits[ok], ck)
    write_jsonl(a.output, ids, pred)
    if a.dump_logits:
        np.savez_compressed(a.dump_logits, filenames=np.array(ids), logits=logits, categories=np.array(CATEGORIES))
    for k, msg in sorted(failed.items()):
        print(f"WARNING: {files[k]} could not be read ({msg}); wrote labels [] for it", file=sys.stderr)
    sec = time.perf_counter() - t0
    n_lab = pred.sum(1)
    print(f"{len(files)} images -> {a.output} | {n_lab.mean():.2f} labels/image, {(n_lab == 0).sum()} empty, "
          f"{len(failed)} unreadable | {sec:.1f}s ({1000 * sec / len(files):.0f} ms/image) on {device}", file=sys.stderr)
    return 4 if (failed and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
