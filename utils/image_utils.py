from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np
import torch


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """Convert [C,H,W] or [1,C,H,W] float tensor in [0,1] to HWC uint8."""
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return arr


def save_tensor_image(x: torch.Tensor, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8(x)).save(path)


def save_prior_map(prior: torch.Tensor, path: str | Path):
    """Save prior [1,1,H,W] or [1,H,W] as grayscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if prior.dim() == 4:
        prior = prior[0, 0]
    elif prior.dim() == 3:
        prior = prior[0]
    p = prior.detach().float().cpu()
    p = (p - p.min()) / (p.max() - p.min() + 1e-8)
    arr = (p.numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    """Pad tensor [B,C,H,W] to spatial multiple, return padded tensor and original size."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, (h, w)


def crop_to_size(x: torch.Tensor, size):
    h, w = size
    return x[..., :h, :w]
