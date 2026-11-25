"""Redesign B dataset: image + per-cell targets on the stride-4 grid of the resized image.

Kinds (rules: docs/plan/재설계B_구현계획.md, table "데이터별 칸 정답"):
  fp     Fashionpedia train_inner: group maps + parts from GT masks (soft = covered fraction)
  coco   COCO person photos: inside persons = FashionFormer KD; bag/tie/umbrella masks = that class in the
         accessory group (other groups ignored); band of --band cells outside persons/objects = ignored;
         FashionFormer garments outside persons and detector persons missing from COCO = ignored;
         remaining background = "none" in every group
  ext    external_merged: LVIS images supervise only cells inside LVIS positive masks; others no seg loss
  empty  no-garment photos: every group "none", parts 0
  eval   image only
"""
import io
import math
import os
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from .spec import GROUPS, GROUP_SIZES, IGNORE, N_SEG, PARTS, STRIDE

KIND_ID = {"fp": 0, "coco": 1, "ext": 2, "empty": 3, "eval": 4}
GOFF = np.cumsum([0] + GROUP_SIZES)[:-1]          # first channel of each group in the 31-channel map


def to_rgb(img):
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def target_size(w, h, short, max_long):
    s = min(short / min(w, h), max_long / max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def resize_nn(a, size):
    """uint8/bool [H, W] or [K, H, W] -> size (w, h), nearest."""
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.uint8)).resize(size, Image.NEAREST))
    return np.stack([resize_nn(x, size) for x in a])


def block(a, fill):
    """[..., H, W] -> [..., hc, wc, 16] pixels of each 4x4 cell (padding with `fill`)."""
    H, W = a.shape[-2:]
    hc, wc = math.ceil(H / STRIDE), math.ceil(W / STRIDE)
    pad = [(0, 0)] * (a.ndim - 2) + [(0, hc * STRIDE - H), (0, wc * STRIDE - W)]
    a = np.pad(a, pad, constant_values=fill)
    a = a.reshape(*a.shape[:-2], hc, STRIDE, wc, STRIDE)
    a = np.moveaxis(a, -3, -2)
    return a.reshape(*a.shape[:-2], STRIDE * STRIDE)


def group_targets(g, min_valid=8):
    """g uint8 [4, H, W] (0 none, j class, 255 unknown) -> soft [31, hc, wc] float32, valid bool [4, hc, wc]."""
    b = block(g, IGNORE)                                    # [4, hc, wc, 16]
    known = (b != IGNORE).sum(-1)                           # [4, hc, wc]
    hc, wc = b.shape[1:3]
    t = np.zeros((N_SEG, hc, wc), np.float32)
    for gi, n in enumerate(GROUP_SIZES):
        for v in np.unique(g[gi]):
            if v == IGNORE:
                continue
            t[GOFF[gi] + v] = (b[gi] == v).sum(-1) / np.maximum(known[gi], 1)
    return t, known >= min_valid


def part_targets(parts):
    """parts uint8 bitmask [H, W] -> soft [7, hc, wc]."""
    b = block(parts, 0)
    return np.stack([((b >> k) & 1).mean(-1) for k in range(len(PARTS))]).astype(np.float32)


def frac(m):
    """bool [H, W] -> fraction of each cell (padding counts as False)."""
    return block(m.astype(np.uint8), 0).mean(-1).astype(np.float32)


def old_photo(img):
    g = img.convert("L")
    img = ImageOps.colorize(g, (40, 26, 13), (240, 224, 192)) if random.random() < 0.5 else g.convert("RGB")
    if random.random() < 0.7:
        f = random.uniform(0.25, 0.6)
        w, h = img.size
        img = img.resize((max(8, round(w * f)), max(8, round(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 0.9))
    if random.random() < 0.7:
        a = np.asarray(img, np.float32)
        a = a + np.random.normal(0, random.uniform(4, 18), a.shape[:2])[..., None]
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.5:
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=random.randint(20, 60)); buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


class SegBDataset(Dataset):
    def __init__(self, paths, kinds, tags, soft, prep, mean, std, train, max_long=1333, flip=0.5,
                 gray_p=0.0, old_p=0.0, mosaic_p=0.0, mosaic_sizes=(2, 4), zoom_p=0.0, zoom_min=0.35, band=3,
                 color=None, kd_point=0.3, kd_min=0.3):
        """paths: absolute image paths. kinds: list of kind names. tags [N, 27] in {1, 0, -1}. soft [N, 27] teacher
        probabilities or NaN. prep: segb_prep output dir. color: {index: (sat, val, jitter)} colour matching."""
        self.paths, self.kinds = list(paths), [KIND_ID[k] for k in kinds]
        self.tags = np.asarray(tags, np.float32)
        self.soft = None if soft is None else np.asarray(soft, np.float32)
        self.prep = prep
        self.mean = np.asarray(mean, np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, np.float32).reshape(3, 1, 1)
        self.train, self.max_long, self.flip = train, max_long, flip
        self.gray_p, self.old_p, self.zoom_p, self.zoom_min = gray_p, old_p, zoom_p, zoom_min
        self.mosaic_p, self.mosaic_sizes = mosaic_p, tuple(mosaic_sizes)
        self.band = band
        self.kd_point = kd_point
        self.kd_min = kd_min
        self.color = color or {}
        self.fp_idx = [i for i, k in enumerate(self.kinds) if k == KIND_ID["fp"]]
        self.fill = tuple(int(round(255 * m)) for m in np.asarray(mean).reshape(-1)[:3])

    def __len__(self):
        return len(self.paths)

    # ---------------------------------------------------------------- loading
    def load_img(self, i):
        with Image.open(self.paths[i]) as im:
            im.load()
            img = to_rgb(im)
        if i in self.color:
            s, v, j = self.color[i]
            img = ImageEnhance.Color(img).enhance(s * random.uniform(1 - j, 1 + j))
            img = ImageEnhance.Brightness(img).enhance(v * random.uniform(1 - j, 1 + j))
        return img

    def load_layers(self, i, size):
        """Label layers aligned with the loaded image of `size` (w, h)."""
        k, base = self.kinds[i], os.path.basename(self.paths[i])
        if k == KIND_ID["fp"]:
            z = np.load(os.path.join(self.prep, "fp", base + ".npz"))
            return {"g": resize_nn(z["g"], size), "parts": resize_nn(z["parts"], size)}
        if k == KIND_ID["coco"]:
            z = np.load(os.path.join(self.prep, "coco", base[:-4] + ".npz"))
            return {"P": resize_nn(z["P"], size).astype(bool), "O": resize_nn(z["O"], size),
                    "E": resize_nn(z["E"], size).astype(bool), "kd": z["kd"]}
        if k == KIND_ID["ext"]:
            f = os.path.join(self.prep, "lvis", base + ".npz")
            if os.path.exists(f):
                return {"g": resize_nn(np.load(f)["g"], size)}
            return {}
        if k == KIND_ID["empty"]:
            w, h = size
            return {"g": np.zeros((4, h, w), np.uint8), "parts": np.zeros((h, w), np.uint8)}
        return {}

    # ---------------------------------------------------------------- geometric ops on image + layers
    @staticmethod
    def geo(L, fn):
        out = {}
        for key, v in L.items():
            out[key] = v if key == "kd" else fn(v)
        return out

    def mosaic(self, img, L, i):
        w, h = img.size
        k = random.choice(self.mosaic_sizes)
        others = [random.choice(self.fp_idx) for _ in range(k - 1)]
        rows, cols = {2: (2, 1), 4: (2, 2), 6: (3, 2), 9: (3, 3)}[k]
        if w > h:
            rows, cols = cols, rows
        xs, ys = [w * c // cols for c in range(cols + 1)], [h * r // rows for r in range(rows + 1)]
        cells = [(xs[c], ys[r], xs[c + 1] - xs[c], ys[r + 1] - ys[r]) for r in range(rows) for c in range(cols)]
        out = Image.new("RGB", (w, h), self.fill)
        g, parts = np.zeros((4, h, w), np.uint8), np.zeros((h, w), np.uint8)   # letterbox = none
        for (x0, y0, cw, ch), j in zip(cells, [i] + others):
            tile = img if j == i else self.load_img(j)
            tl = L if j == i else self.load_layers(j, tile.size)
            s = min(cw / tile.size[0], ch / tile.size[1])
            ts = (max(1, round(tile.size[0] * s)), max(1, round(tile.size[1] * s)))
            tile = tile.resize(ts, Image.BILINEAR)
            px, py = x0 + (cw - ts[0]) // 2, y0 + (ch - ts[1]) // 2
            out.paste(tile, (px, py))
            g[:, py:py + ts[1], px:px + ts[0]] = resize_nn(tl["g"], ts)
            parts[py:py + ts[1], px:px + ts[0]] = resize_nn(tl["parts"], ts)
        return out, {"g": g, "parts": parts}, [i] + others

    def zoom_out(self, img, L):
        w, h = img.size
        s = random.uniform(self.zoom_min, 0.85)
        ts = (max(1, round(w * s)), max(1, round(h * s)))
        out = Image.new("RGB", (w, h), self.fill)
        px, py = random.randint(0, w - ts[0]), random.randint(0, h - ts[1])
        out.paste(img.resize(ts, Image.BILINEAR), (px, py))
        g, parts = np.zeros((4, h, w), np.uint8), np.zeros((h, w), np.uint8)
        g[:, py:py + ts[1], px:px + ts[0]] = resize_nn(L["g"], ts)
        parts[py:py + ts[1], px:px + ts[0]] = resize_nn(L["parts"], ts)
        return out, {"g": g, "parts": parts}

    # ---------------------------------------------------------------- targets on the cell grid
    def coco_targets(self, L, flipped, hc, wc):
        Pf, Ef = frac(L["P"]), frac(L["E"])
        O = L["O"]
        objs = [c for c in np.unique(O) if c]
        Of = {c: frac(O == c) for c in objs}
        Oany = sum(Of.values()) if objs else np.zeros_like(Pf)
        person, obj = Pf > 0, Oany > 0
        band = binary_dilation(person | obj, iterations=self.band) & ~(person | obj) if self.band else np.zeros_like(person)
        bg = ~(person | obj | band | (Ef > 0))
        t = np.zeros((N_SEG, hc, wc), np.float32)
        v = np.zeros((4, hc, wc), bool)
        hard = np.zeros((4, hc, wc), bool)
        for gi in range(4):                                      # background: none everywhere
            t[GOFF[gi]][bg] = 1.0
            v[gi] |= bg
            hard[gi] |= bg
        # inside persons: FashionFormer distribution per group (class prob x score, normalised when the sum exceeds 1)
        kd = L["kd"].astype(np.float32) / 255.0
        kd = np.stack([np.asarray(Image.fromarray(x).resize((wc, hc), Image.BILINEAR)) for x in kd])
        if flipped:
            kd = kd[:, :, ::-1]
        # user: person cells where the teacher found (almost) nothing are unknown, not "none"
        found = kd.sum(0) >= self.kd_min if self.kd_min else np.ones_like(person)
        if self.kd_point:                                        # teacher operating point -> 1 (user: 0.3 basis)
            kd = np.clip(kd / self.kd_point, 0, 1)
        for gi, cs in enumerate(GROUPS):
            q = kd[cs]
            s = q.sum(0)
            q = np.where(s > 1, q / np.maximum(s, 1e-6), q)
            m = person & found & ~(obj if gi == 3 else np.zeros_like(obj))
            t[GOFF[gi] + 1:GOFF[gi] + GROUP_SIZES[gi]][:, m] = q[:, m]
            t[GOFF[gi]][m] = np.clip(1 - q.sum(0), 0, 1)[m]
            v[gi] |= m
        # bag / tie / umbrella: accessory group only (other groups at those cells stay as set: ignored unless person)
        if objs:
            gi = 3
            for c in objs:
                j = GROUPS[3].index(int(c) - 1) + 1
                t[GOFF[gi] + j][obj] = Of[c][obj]
            rest = np.clip(1 - Oany, 0, 1)
            t[GOFF[gi]][obj] = rest[obj]
            for gg in range(3):                                  # non-person object cells: other groups unknown
                m = obj & ~person
                v[gg][m] = False
                t[GOFF[gg]:GOFF[gg] + GROUP_SIZES[gg]][:, m] = 0
            v[gi] |= obj
            hard[gi] |= obj
        return t, v, hard

    def __getitem__(self, key):
        i, level = key
        k = self.kinds[i]
        img = self.load_img(i)
        y = self.tags[i].copy()
        soft = None if self.soft is None else self.soft[i].copy()
        L = self.load_layers(i, img.size) if self.train else {}
        if self.train and k == KIND_ID["fp"]:
            if self.mosaic_p and random.random() < self.mosaic_p:
                img, L, idx = self.mosaic(img, L, i)
                y = self.tags[idx].max(0)
                if soft is not None:
                    with np.errstate(all="ignore"):
                        st = self.soft[idx]
                        soft = np.nanmax(st, 0) if not np.isnan(st[:, 0]).all() else soft
            if self.zoom_p and random.random() < self.zoom_p:
                img, L = self.zoom_out(img, L)
        size = target_size(*img.size, level, min(self.max_long, 2 * level))
        img = img.resize(size, Image.BILINEAR)
        L = self.geo(L, lambda a: resize_nn(a, size))
        flipped = self.train and random.random() < self.flip
        if flipped:
            img = ImageOps.mirror(img)
            L = self.geo(L, lambda a: np.ascontiguousarray(a[..., ::-1]))
        if self.train and self.old_p and random.random() < self.old_p:
            img = old_photo(img)
        elif self.train and self.gray_p and random.random() < self.gray_p:
            img = img.convert("L").convert("RGB")
        x = (np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0 - self.mean) / self.std
        w, h = size
        hc, wc = math.ceil(h / STRIDE), math.ceil(w / STRIDE)
        item = {"x": torch.from_numpy(np.ascontiguousarray(x)), "y": torch.from_numpy(y.astype(np.float32)),
                "idx": i, "kind": k, "level": level, "portrait": img.size[1] >= img.size[0]}
        if not self.train:
            return item
        item["soft"] = torch.from_numpy(soft) if soft is not None else torch.full((27,), float("nan"))
        t = np.zeros((N_SEG, hc, wc), np.float32)
        v = np.zeros((4, hc, wc), bool)
        hard = np.zeros((4, hc, wc), bool)
        pt = np.zeros((len(PARTS), hc, wc), np.float32)
        pv = np.zeros((hc, wc), bool)
        if "g" in L:
            t, v = group_targets(L["g"])
            hard = v.copy()
        if "parts" in L:
            pt, pv = part_targets(L["parts"]), np.ones((hc, wc), bool)
        if k == KIND_ID["coco"]:
            t, v, hard = self.coco_targets(L, flipped, hc, wc)
        item.update(t=torch.from_numpy(t), v=torch.from_numpy(v), hard=torch.from_numpy(hard),
                    pt=torch.from_numpy(pt), pv=torch.from_numpy(pv))
        return item


def collate(batch, canvas_hw):
    """Pad images to the canvas (H, W) and cell targets to (H/4, W/4)."""
    H, W = canvas_hw
    H = max(H, *(b["x"].shape[1] for b in batch))
    W = max(W, *(b["x"].shape[2] for b in batch))
    H, W = math.ceil(H / 32) * 32, math.ceil(W / 32) * 32
    B = len(batch)
    out = {"x": torch.zeros(B, 3, H, W), "y": torch.stack([b["y"] for b in batch]),
           "sizes": torch.tensor([b["x"].shape[1:] for b in batch]), "idx": torch.tensor([b["idx"] for b in batch]),
           "kind": torch.tensor([b["kind"] for b in batch])}
    for k, b in enumerate(batch):
        out["x"][k, :, :b["x"].shape[1], :b["x"].shape[2]] = b["x"]
    if "t" in batch[0]:
        hc, wc = H // STRIDE, W // STRIDE
        out["soft"] = torch.stack([b["soft"] for b in batch])
        for key, shape, dt in (("t", (N_SEG,), torch.float32), ("v", (4,), torch.bool), ("hard", (4,), torch.bool),
                               ("pt", (len(PARTS),), torch.float32), ("pv", (), torch.bool)):
            z = torch.zeros(B, *shape, hc, wc, dtype=dt)
            for k, b in enumerate(batch):
                a = b[key]
                z[k, ..., :a.shape[-2], :a.shape[-1]] = a
            out[key] = z
    return out
