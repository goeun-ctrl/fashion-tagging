"""Dataset for the fashion task. Reads images from an extracted folder or a ZIP.

    ds = FashionDataset(root="data/train", names=names, targets=targets,
                        size=384, mode="pad", train=True)
    x, y = ds[0]     # x: float32 [3,S,S] (torch tensor if torch is installed), y: float32 [27]

Training adds only a horizontal flip with p=0.5; validation is deterministic.
`root` is either the folder that contains `images/` or a .zip whose inner
folder is `train/` or `val/` (set `zip_prefix`).
"""
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocess import IMAGENET_MEAN, IMAGENET_STD, adjust_color, preprocess

try:
    import torch
    from torch.utils.data import Dataset as _Base
except ImportError:  # CPU-only dev machines without torch can still test preprocessing
    torch = None
    _Base = object


def read_names(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class FashionDataset(_Base):
    def __init__(self, root, names, targets, size=384, mode="pad", train=False,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, zip_prefix=None, seed=0, color=None):
        """color: optional (saturation, brightness, jitter) applied to every training image, each factor
        multiplied by U(1 - jitter, 1 + jitter) per load. Used to bring colourful external data closer to the
        original train colours; None (default) leaves images unchanged."""
        self.root, self.names = str(root), list(names)
        self.targets = np.asarray(targets, dtype=np.float32)
        assert len(self.names) == len(self.targets)
        self.size, self.mode, self.train = size, mode, train
        self.mean, self.std = mean, std
        self.zip_prefix = zip_prefix
        self.is_zip = self.root.lower().endswith(".zip")
        if self.is_zip and not zip_prefix:
            raise ValueError("zip_prefix ('train' or 'val') is required when root is a ZIP")
        self._zf = None  # opened lazily so each DataLoader worker has its own handle
        self._rng = random.Random(seed)
        self.color = color

    def __len__(self):
        return len(self.names)

    def _open(self, name):
        if self.is_zip:
            if self._zf is None:
                self._zf = zipfile.ZipFile(self.root)
            return Image.open(io.BytesIO(self._zf.read(f"{self.zip_prefix}/images/{name}")))
        return Image.open(Path(self.root) / "images" / name)

    def load_image(self, name):
        """PIL RGB image by file name (for visualisation, no resizing)."""
        from .preprocess import to_rgb
        with self._open(name) as img:
            img.load()
            return to_rgb(img)

    def __getitem__(self, i):
        with self._open(self.names[i]) as img:
            img.load()
            # global `random` is re-seeded per DataLoader worker by PyTorch, so workers do not
            # repeat the same flip sequence, and torch.manual_seed keeps runs reproducible
            flip = self.train and random.random() < 0.5
            if self.train and self.color:
                s, v, j = self.color
                img = adjust_color(img, s * random.uniform(1 - j, 1 + j), v * random.uniform(1 - j, 1 + j))
            x = preprocess(img, self.size, self.mode, self.mean, self.std, hflip=flip)
        y = self.targets[i]
        if torch is not None:
            return torch.from_numpy(x), torch.from_numpy(y)
        return x, y
