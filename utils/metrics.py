from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse <= 1e-12:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def _gaussian_window(window_size: int, sigma: float, channels: int,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5, max_val: float = 1.0) -> float:
    """Gaussian-window SSIM (Wang et al., 2004), mean over the batch.

    This matches the standard 11x11 Gaussian (sigma=1.5) protocol and is close to
    skimage/MATLAB results. For the camera-ready tables we still recommend
    reporting with your fixed official protocol (e.g. skimage.metrics or the
    common MATLAB script) so numbers are directly comparable with prior work.
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    c = pred.shape[1]
    win = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(pred, win, padding=pad, groups=c)
    mu_y = F.conv2d(target, win, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, win, padding=pad, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, win, padding=pad, groups=c) - mu_xy

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return ssim_map.mean().item()


# Backward-compatible alias: previous code imported `simple_ssim`.
# It now points to the proper Gaussian-window implementation.
def simple_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ssim(pred, target)
