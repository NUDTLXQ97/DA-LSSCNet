from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt((x - y) ** 2 + self.eps ** 2).mean()


def gradient_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    def grad(t):
        dx = t[:, :, :, 1:] - t[:, :, :, :-1]
        dy = t[:, :, 1:, :] - t[:, :, :-1, :]
        return dx, dy
    xdx, xdy = grad(x)
    ydx, ydy = grad(y)
    return F.l1_loss(xdx, ydx) + F.l1_loss(xdy, ydy)


def local_log_amplitude_frequency_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 16) -> torch.Tensor:
    """Local log-amplitude spectrum loss used in revised Method.

    When window_size <= 0 (or >= min(H, W)), the loss falls back to a GLOBAL FFT
    over the whole image instead of window-wise FFT. This matches the spectral
    branch's convention (window_size <= 0 => global FFT) so the `global_fft`
    ablation is consistent between the model and its training loss.
    """
    b, c, h, w = x.shape

    # Global-FFT fallback (used by the global_fft ablation, window_size=0).
    if window_size is None or window_size <= 0 or window_size >= min(h, w):
        ax = torch.log(torch.abs(torch.fft.fft2(x, norm='ortho')) + 1e-6)
        ay = torch.log(torch.abs(torch.fft.fft2(y, norm='ortho')) + 1e-6)
        return F.l1_loss(ax, ay)

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        y = F.pad(y, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, hp, wp = x.shape
    nh, nw = hp // window_size, wp // window_size

    def part(t):
        t = t.view(b, c, nh, window_size, nw, window_size)
        t = t.permute(0, 2, 4, 1, 3, 5).contiguous()
        return t.view(b * nh * nw, c, window_size, window_size)

    xw, yw = part(x), part(y)
    ax = torch.log(torch.abs(torch.fft.fft2(xw, norm='ortho')) + 1e-6)
    ay = torch.log(torch.abs(torch.fft.fft2(yw, norm='ortho')) + 1e-6)
    return F.l1_loss(ax, ay)
