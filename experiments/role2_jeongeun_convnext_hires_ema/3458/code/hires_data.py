"""Replay V2 draws and augmentations on the team's high-resolution canvases."""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from augmented_data import PlannedDataset
from original_augmentation import transform_rgb
from team_geometry import BucketBatchSampler, canvas, ceil32, long_cap, short_levels, target_size


@lru_cache(maxsize=55000)
def portrait(path):
    with Image.open(path) as image:
        w, h = image.size
        if image.getexif().get(274, 1) in (5, 6, 7, 8):
            w, h = h, w
    return h >= w


def image_tensor(image, level, mean, std):
    shape = canvas(level, image.height >= image.width, 1333)
    image = image.resize(target_size(*image.size, level, long_cap(level, 1333)), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
    pixels = (pixels - np.asarray(mean, dtype=np.float32)[:, None, None]) / np.asarray(std, dtype=np.float32)[:, None, None]
    return torch.from_numpy(np.ascontiguousarray(pixels)), shape


class HiResPlan(PlannedDataset):
    def __init__(self, package, original_root, epoch, cfg, mean, std, smoke_samples):
        super().__init__(package, original_root, epoch, cfg, mean, std, smoke_samples)
        self.epoch = epoch
        self.paths = [self.roots[self.rows[i]["source"]] / self.rows[i]["relative_path"] for i in self.plan["image_index"]]

    def __getitem__(self, key):
        pos, level = key
        with Image.open(self.paths[pos]) as image:
            image = transform_rgb(image, bool(self.plan["flip"][pos]), self.plan["factors"][pos], self.plan["order"][pos])
            pixels, shape = image_tensor(image, level, self.mean, self.std)
        targets = self.all_targets[self.plan["image_index"][pos]].astype(np.float32)
        return pixels, torch.from_numpy(targets), pos, shape


class HiResEval:
    def __init__(self, dataset, root, names, targets, mean, std):
        self.dataset = dataset
        self.paths = [root / "images" / name for name in names]
        self.targets, self.mean, self.std = targets, mean, std

    def __len__(self):
        return len(self.paths)

    def load_image(self, name):
        return self.dataset.load_image(name)

    def __getitem__(self, key):
        pos, level = key
        with Image.open(self.paths[pos]) as image:
            pixels, shape = image_tensor(transform_rgb(image), level, self.mean, self.std)
        return pixels, torch.tensor(self.targets[pos], dtype=torch.float32), pos, shape


def collate(batch):
    h = ceil32(max(max(item[3][0], item[0].shape[1]) for item in batch))
    w = ceil32(max(max(item[3][1], item[0].shape[2]) for item in batch))
    out = torch.zeros(len(batch), 3, h, w)
    for i, (pixels, _, _, _) in enumerate(batch):
        out[i, :, :pixels.shape[1], :pixels.shape[2]] = pixels
    return out, torch.stack([item[1] for item in batch]), torch.tensor([item[0].shape[1:] for item in batch]), torch.tensor([item[2] for item in batch])


def make_hires_loader(dataset, cfg, device, generator, worker_init):
    training = isinstance(dataset, HiResPlan)
    levels = short_levels(SimpleNamespace(short_min=400, short_max=800, short_levels=None)) if training else [800]
    orientations = np.array([portrait(path) for path in dataset.paths])
    sampler = BucketBatchSampler(orientations, levels, cfg["batch_size"], cfg["seed"], training)
    batches = sampler.plan(dataset.epoch if training else 0)
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate, generator=generator,
                        worker_init_fn=worker_init, num_workers=cfg["num_workers"], pin_memory=device.type == "cuda")
    if training:
        counts = [sum(dataset.known_per_draw[index] for index, _ in batch) for batch in batches]
        accum = cfg["accum"]
        loader.window_known = [sum(counts[i:i + accum]) for i in range(0, len(counts), accum)]
    return loader
