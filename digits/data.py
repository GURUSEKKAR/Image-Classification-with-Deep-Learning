"""Image loading, stratified splitting and input preprocessing."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from torch.nn import functional as F


def dataset():
    raw = load_digits()
    pixels = np.ascontiguousarray(raw.images, dtype=np.uint8)
    labels = np.ascontiguousarray(raw.target, dtype=np.uint8)
    digest = hashlib.sha256(pixels.tobytes() + labels.tobytes()).hexdigest()
    images = pixels[:, None].astype(np.float32) / 16.0
    return images, labels.astype(np.int64), digest


def split_indices(labels, seed):
    """Reserve approximately 15% of each class for each held-out split."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "validation": [], "test": []}
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n = round(0.15 * len(indices))
        parts["test"].extend(indices[:n])
        parts["validation"].extend(indices[n : 2 * n])
        parts["train"].extend(indices[2 * n :])
    return {name: rng.permutation(indices).tolist() for name, indices in parts.items()}


def validate_splits(parts, size):
    expected = {"train", "validation", "test"}
    if not isinstance(parts, dict) or set(parts) != expected:
        raise ValueError("Checkpoint must contain train, validation and test partitions.")
    all_indices = []
    for name, indices in parts.items():
        if not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices):
            raise ValueError(f"Invalid indices in the {name} partition.")
        all_indices.extend(indices)
    if sorted(all_indices) != list(range(size)):
        raise ValueError("Saved partitions must cover every dataset image exactly once.")


def has_dark_ink(rgba):
    gray = np.asarray(rgba.convert("L"))
    alpha = np.asarray(rgba.getchannel("A"))
    border_alpha = np.concatenate((alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]))
    if np.median(border_alpha) < 128:
        visible = gray[alpha >= 128]
        return bool(visible.size and np.median(visible) < 128)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    return bool(np.median(border[border_alpha >= 128]) >= 128)


def image_tensor(path: Path, dark_ink: bool = False, *, auto_ink=False, trim=False):
    """Fit one cropped digit inside an 8×8 square, preserving its aspect ratio."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert("RGBA")
        if auto_ink:
            dark_ink = has_dark_ink(rgba)
        background = "white" if dark_ink else "black"
        gray = Image.alpha_composite(Image.new("RGBA", rgba.size, background), rgba).convert("L")
    if dark_ink:
        gray = ImageOps.invert(gray)
    if trim:
        low, high = gray.getextrema()
        threshold = low + 0.12 * (high - low)
        bounds = gray.point(lambda pixel: 255 if pixel > threshold else 0).getbbox()
        if bounds:
            gray = gray.crop(bounds)
    tile = ImageOps.pad(gray, (8, 8), method=Image.Resampling.LANCZOS, color=0)
    pixels = np.asarray(tile, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels)[None, None]


def augment(images):
    """Sample a small independent rotation and translation for each image."""
    batch_size = len(images)
    angle = (torch.rand(batch_size) - 0.5) * np.deg2rad(20.0)
    cosine, sine = angle.cos(), angle.sin()
    # affine_grid expresses translations in normalized image coordinates.
    dx, dy = (torch.rand(2, batch_size) - 0.5) * 0.24
    theta = torch.stack((cosine, -sine, dx, sine, cosine, dy), dim=1).reshape(-1, 2, 3)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, padding_mode="zeros", align_corners=False)
