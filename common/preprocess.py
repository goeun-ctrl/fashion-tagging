"""Deterministic image preprocessing shared by every experiment (PIL + numpy).

mode="pad"    : keep aspect ratio, resize the long side to `size`, pad to a
                square with black (0,0,0) BEFORE normalisation. (team baseline)
mode="resize" : force-resize to size x size (distorts the aspect ratio).

Images are always converted to RGB first (186 train images are grayscale).
Resize uses bicubic, as in the meeting notes.
"""
import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_rgb(img):
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def resize_pad(img, size):
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def resize_square(img, size):
    return img.resize((size, size), Image.BICUBIC)


def preprocess(img, size=384, mode="pad", mean=IMAGENET_MEAN, std=IMAGENET_STD, hflip=False):
    """PIL image -> float32 array [3, size, size], normalised."""
    img = to_rgb(img)
    if mode == "pad":
        img = resize_pad(img, size)
    elif mode == "resize":
        img = resize_square(img, size)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if hflip:
        img = ImageOps.mirror(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return arr.transpose(2, 0, 1).copy()


def timm_mean_std(model_name):
    """Read mean/std of a timm checkpoint config (needs timm; not tested on the dev PC)."""
    import timm
    cfg = timm.models.get_pretrained_cfg(model_name)
    return tuple(cfg.mean), tuple(cfg.std)
