"""DA-LSSCNet.

Active design
-------------
* Geometry-sensitive local prior for spatially varying turbulence.
* 50% overlapping sqrt-Hann local FFT processing with overlap-add synthesis.
* Bounded image residual and learned local restoration gate.
* A separately trained image-space Siamese ordinal severity encoder.
* A small severity controller, initially neutral, that is connected only after
  the ordinal severity model has passed pairwise-ranking validation.

The restoration parameter names inherited from v2.1 are preserved so the
selected v2.1 calibration checkpoint can be loaded with ``strict=False``.
Only the new ``global_severity`` and ``severity_controller`` parameters are
expected to be missing.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .severity_siamese import ImageSpaceSiameseSeverityEncoder


# ---------------------------------------------------------------------------
# Stable fixed operators / observable cues
# ---------------------------------------------------------------------------
def _make_preprocess_kernels(
    device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = (
        torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    ky = (
        torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        / 8.0
    )
    return kx, ky


def image_gradient_mag(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel gradient magnitude, output [B,1,H,W]."""
    _, c, _, _ = x.shape
    kx, ky = _make_preprocess_kernels(x.device, x.dtype)
    gx = F.conv2d(x, kx.repeat(c, 1, 1, 1), padding=1, groups=c)
    gy = F.conv2d(x, ky.repeat(c, 1, 1, 1), padding=1, groups=c)
    # FP32 avoids half-precision epsilon underflow and sqrt(0) NaN gradients.
    g = torch.sqrt(gx.float().square() + gy.float().square() + 1e-6)
    return g.mean(dim=1, keepdim=True).to(dtype=x.dtype)


def image_laplacian_abs(x: torch.Tensor) -> torch.Tensor:
    _, c, _, _ = x.shape
    k = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    y = F.conv2d(x, k.repeat(c, 1, 1, 1), padding=1, groups=c).abs()
    return y.mean(dim=1, keepdim=True)


def local_std(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    pad = kernel_size // 2
    x32 = x.float()
    mean = F.avg_pool2d(x32, kernel_size, stride=1, padding=pad)
    mean2 = F.avg_pool2d(x32.square(), kernel_size, stride=1, padding=pad)
    var = (mean2 - mean.square()).clamp_min(0.0)
    std = torch.sqrt(var + 1e-6).mean(dim=1, keepdim=True)
    return std.to(dtype=x.dtype)


def _standardize_per_image(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x32 = x.float()
    mu = x32.mean(dim=(2, 3), keepdim=True)
    var = (x32 - mu).square().mean(dim=(2, 3), keepdim=True)
    y = (x32 - mu) * torch.rsqrt(var + float(eps))
    return y.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW, evaluated in FP32."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            mean = x32.mean(dim=1, keepdim=True)
            var = (x32 - mean).square().mean(dim=1, keepdim=True)
            y = (x32 - mean) * torch.rsqrt(var + self.eps)
            y = y * self.weight.float() + self.bias.float()
        return y.to(dtype=out_dtype)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x) * self.scale


class DifferentialConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.pw = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        ksum = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        return self.pw(y - x * ksum.view(1, -1, 1, 1))


class FixedEdgeFilter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = x.shape
        kx, ky = _make_preprocess_kernels(x.device, x.dtype)
        gx = F.conv2d(x, kx.repeat(c, 1, 1, 1), padding=1, groups=c)
        gy = F.conv2d(x, ky.repeat(c, 1, 1, 1), padding=1, groups=c)
        mag = torch.sqrt(gx.float().square() + gy.float().square() + 1e-6)
        return self.proj(mag.to(dtype=x.dtype))


class MSGeometrySensitiveEncoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.d1 = nn.Conv2d(
            channels, channels, 3, padding=1, dilation=1, groups=channels
        )
        self.d2 = nn.Conv2d(
            channels, channels, 3, padding=2, dilation=2, groups=channels
        )
        self.d4 = nn.Conv2d(
            channels, channels, 3, padding=4, dilation=4, groups=channels
        )
        self.diff = DifferentialConv(channels)
        self.edge = FixedEdgeFilter(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 5, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.d1(x), self.d2(x), self.d4(x), self.diff(x), self.edge(x)]
        return self.fuse(torch.cat(feats, dim=1))


class DegradationPriorEstimator(nn.Module):
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 3, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, fg: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        cues = torch.cat(
            [local_std(img), image_gradient_mag(img), image_laplacian_abs(img)],
            dim=1,
        )
        cues = _standardize_per_image(cues)
        return self.net(torch.cat([fg, cues], dim=1))


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------
def pad_to_window(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def unpad(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h:
        x = x[:, :, :-pad_h, :]
    if pad_w:
        x = x[:, :, :, :-pad_w]
    return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, int, int]:
    b, c, h, w = x.shape
    nh, nw = h // window_size, w // window_size
    x = x.view(b, c, nh, window_size, nw, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(b * nh * nw, c, window_size, window_size), nh, nw


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    nh: int,
    nw: int,
    batch: int,
) -> torch.Tensor:
    c = windows.shape[1]
    x = windows.view(batch, nh, nw, c, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(batch, c, nh * window_size, nw * window_size)


def radial_band_index(
    h: int, w: int, num_bands: int, device: torch.device
) -> torch.Tensor:
    fy = torch.arange(h, device=device).float()
    fy = torch.minimum(fy, h - fy) / max(h // 2, 1)
    fx = torch.arange(w, device=device).float()
    fx = torch.minimum(fx, w - fx) / max(w // 2, 1)
    ry = fy.view(h, 1).expand(h, w)
    rx = fx.view(1, w).expand(h, w)
    radius = torch.sqrt(ry.square() + rx.square()) / math.sqrt(2.0)
    return torch.clamp((radius * num_bands).long(), 0, num_bands - 1)


class LocalSpectralCompensation(nn.Module):
    """Prior-conditioned local spectral processing with overlap-add synthesis.

    Parameter names match v1, so old checkpoints remain loadable.  With
    ``overlap_ratio=0`` it reproduces the old hard non-overlapping path for
    ablation; the v2 main config uses 0.5.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 16,
        num_bands: int = 8,
        strength: float = 0.25,
        strength_max: float | None = None,
        overlap_ratio: float = 0.5,
    ):
        super().__init__()
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0,1).")
        self.window_size = int(window_size)
        self.num_bands = int(num_bands)
        self.strength = float(strength)
        self.strength_max = float(
            self.strength if strength_max is None else strength_max
        )
        if self.strength_max < self.strength:
            raise ValueError("strength_max must be >= strength.")
        self.overlap_ratio = float(overlap_ratio)
        self.base_profile = nn.Parameter(torch.zeros(channels, num_bands))
        hidden = max(channels // 2, 16)
        self.prior_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * num_bands),
        )
        nn.init.zeros_(self.prior_mlp[-1].weight)
        nn.init.zeros_(self.prior_mlp[-1].bias)
        self.post = nn.Conv2d(channels, channels, 1)
        self._band_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._window_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def _get_band(
        self, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        key = (h, w, str(device))
        if key not in self._band_cache:
            self._band_cache[key] = radial_band_index(
                h, w, self.num_bands, device
            )
        return self._band_cache[key]

    def _get_sqrt_hann(
        self, ws: int, device: torch.device
    ) -> torch.Tensor:
        key = (ws, str(device))
        if key not in self._window_cache:
            one_d = torch.hann_window(
                ws, periodic=True, dtype=torch.float32, device=device
            )
            two_d = torch.outer(one_d, one_d).clamp_min(0.0).sqrt()
            self._window_cache[key] = two_d.view(1, 1, ws, ws)
        return self._window_cache[key]

    def _modulate(
        self,
        xw: torch.Tensor,
        pw: torch.Tensor,
        severity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, c, h, w = xw.shape
        band = self._get_band(h, w, xw.device)
        prior_scalar = pw.mean(dim=(2, 3))
        prior_gain = self.prior_mlp(prior_scalar).view(
            n, c, self.num_bands
        )
        gain = self.base_profile.unsqueeze(0) + prior_gain
        gain_map = gain[:, :, band.reshape(-1)].reshape(n, c, h, w)
        if severity is None:
            effective_strength = xw.new_full((n, 1, 1, 1), self.strength)
        else:
            severity = severity.reshape(n, 1, 1, 1).to(dtype=xw.dtype)
            effective_strength = self.strength + (
                self.strength_max - self.strength
            ) * severity.clamp(0.0, 1.0)
        multiplier = 1.0 + effective_strength * torch.tanh(gain_map)
        with torch.autocast(device_type=xw.device.type, enabled=False):
            z = torch.fft.fft2(xw.float(), norm="ortho")
            y = torch.fft.ifft2(
                z * multiplier.float().to(z.dtype), norm="ortho"
            ).real
        return y.to(dtype=xw.dtype)

    def _forward_nonoverlap(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
        severity: torch.Tensor | None,
    ) -> torch.Tensor:
        b = x.shape[0]
        ws = self.window_size
        x_pad, pad_hw = pad_to_window(x, ws)
        p_pad, _ = pad_to_window(prior, ws)
        xw, nh, nw = window_partition(x_pad, ws)
        pw, _, _ = window_partition(p_pad, ws)
        severity_w = None
        if severity is not None:
            severity_w = severity.repeat_interleave(nh * nw, dim=0)
        yw = self._modulate(xw, pw, severity_w)
        return unpad(window_reverse(yw, ws, nh, nw, b), pad_hw)

    def _forward_overlap(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
        severity: torch.Tensor | None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        stride = max(1, int(round(ws * (1.0 - self.overlap_ratio))))
        stride = min(stride, ws)

        # A half-window reflect border guarantees non-zero overlap weights in
        # the final crop and prevents edge darkening.
        border = ws // 2
        x_pad = F.pad(x, (border, border, border, border), mode="reflect")
        p_pad = F.pad(
            prior, (border, border, border, border), mode="reflect"
        )
        hp, wp = x_pad.shape[-2:]
        extra_h = (stride - ((hp - ws) % stride)) % stride
        extra_w = (stride - ((wp - ws) % stride)) % stride
        if extra_h or extra_w:
            x_pad = F.pad(x_pad, (0, extra_w, 0, extra_h), mode="reflect")
            p_pad = F.pad(p_pad, (0, extra_w, 0, extra_h), mode="reflect")
        hp, wp = x_pad.shape[-2:]

        x_cols = F.unfold(x_pad, kernel_size=ws, stride=stride)
        p_cols = F.unfold(p_pad, kernel_size=ws, stride=stride)
        num_windows = x_cols.shape[-1]
        xw = x_cols.transpose(1, 2).reshape(
            b * num_windows, c, ws, ws
        )
        pw = p_cols.transpose(1, 2).reshape(
            b * num_windows, 1, ws, ws
        )

        win = self._get_sqrt_hann(ws, x.device).to(dtype=xw.dtype)
        severity_w = None
        if severity is not None:
            severity_w = severity.repeat_interleave(num_windows, dim=0)
        yw = self._modulate(xw * win, pw, severity_w) * win
        y_cols = yw.reshape(
            b, num_windows, c * ws * ws
        ).transpose(1, 2)
        y_sum = F.fold(
            y_cols,
            output_size=(hp, wp),
            kernel_size=ws,
            stride=stride,
        )

        w2 = win.float().square().reshape(1, ws * ws, 1)
        w2 = w2.expand(b, -1, num_windows)
        denominator = F.fold(
            w2,
            output_size=(hp, wp),
            kernel_size=ws,
            stride=stride,
        )
        y = (y_sum.float() / denominator.clamp_min(1e-6)).to(x.dtype)
        return y[..., border : border + h, border : border + w]

    def forward(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
        severity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        ws = self.window_size
        if ws is None or ws <= 0 or ws >= min(h, w):
            return self.post(self._modulate(x, prior, severity))
        if self.overlap_ratio > 0.0:
            y = self._forward_overlap(x, prior, severity)
        else:
            y = self._forward_nonoverlap(x, prior, severity)
        return self.post(y)


class SpatialStructuralRefinement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.prior_proj = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1), nn.Sigmoid()
        )
        self.blocks = nn.Sequential(
            ResidualBlock(channels), ResidualBlock(channels)
        )
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        modulation = self.prior_proj(prior)
        y = self.blocks(x * (1.0 + modulation))
        return self.fuse(y)


class LSSCBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 16,
        num_bands: int = 8,
        spectral_strength: float = 0.25,
        spectral_strength_max: float | None = None,
        spectral_overlap: float = 0.5,
        use_spectral: bool = True,
        use_spatial: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        if not (use_spectral or use_spatial):
            raise ValueError("At least one branch must be enabled.")
        self.use_spectral = use_spectral
        self.use_spatial = use_spatial
        self.use_gate = use_gate and use_spectral and use_spatial
        self.norm = LayerNorm2d(channels)
        if use_spectral:
            self.spectral = LocalSpectralCompensation(
                channels,
                window_size=window_size,
                num_bands=num_bands,
                strength=spectral_strength,
                strength_max=spectral_strength_max,
                overlap_ratio=spectral_overlap,
            )
        if use_spatial:
            self.spatial = SpatialStructuralRefinement(channels)
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2 + 1, channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid(),
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
        severity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xn = self.norm(x)
        if self.use_spectral and self.use_spatial:
            spectral = self.spectral(xn, prior, severity)
            spatial = self.spatial(xn, prior)
            if self.use_gate:
                gate = self.gate(torch.cat([spatial, spectral, prior], dim=1))
                mixed = gate * spatial + (1.0 - gate) * spectral
            else:
                mixed = 0.5 * (spatial + spectral)
        elif self.use_spectral:
            mixed = self.spectral(xn, prior, severity)
        else:
            mixed = self.spatial(xn, prior)
        return x + self.fuse(mixed) * self.scale


class RestorationGate(nn.Module):
    """Spatially adaptive output-restoration gate inherited from v2.1."""

    def __init__(self, channels: int, init_value: float = 0.3):
        super().__init__()
        if not 0.0 < init_value < 1.0:
            raise ValueError("gate init_value must be strictly between 0 and 1.")
        hidden = max(channels // 2, 16)
        self.local_body = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.local_out = nn.Conv2d(hidden, 1, 3, padding=1)
        self.global_body = nn.Sequential(
            nn.Linear(channels + 1, hidden),
            nn.GELU(),
        )
        self.global_out = nn.Linear(hidden, 1)
        init_logit = math.log(init_value / (1.0 - init_value))
        self.base_logit = nn.Parameter(torch.tensor(float(init_logit)))
        nn.init.zeros_(self.local_out.weight)
        nn.init.zeros_(self.local_out.bias)
        nn.init.zeros_(self.global_out.weight)
        nn.init.zeros_(self.global_out.bias)

    def forward(
        self,
        features: torch.Tensor,
        prior: torch.Tensor,
        global_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_logits = self.local_out(
            self.local_body(torch.cat([features, prior], dim=1))
        )
        pooled = torch.cat(
            [features.mean(dim=(2, 3)), prior.mean(dim=(2, 3))], dim=1
        )
        global_logits = self.global_out(self.global_body(pooled))
        if global_logit_bias is not None:
            global_logits = global_logits + global_logit_bias.to(
                dtype=global_logits.dtype
            )
        return torch.sigmoid(
            self.base_logit + local_logits + global_logits[:, :, None, None]
        )


class SeverityController(nn.Module):
    """Safely connect calibrated severity to restoration strength.

    Both connections are initially neutral.  The spectral drive starts near
    zero, so the effective spectral strength remains at its conservative base.
    The output-gate logit bias starts exactly at zero.  A short connection stage
    trains only this module before any LSSC block is unfrozen.
    """

    def __init__(self):
        super().__init__()
        self.connection = nn.Linear(1, 2)
        nn.init.zeros_(self.connection.weight)
        nn.init.zeros_(self.connection.bias)
        with torch.no_grad():
            self.connection.bias[0] = -6.0

    def forward(
        self, severity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = self.connection(severity.float())
        spectral_drive = torch.sigmoid(values[:, 0:1])
        gate_logit_bias = values[:, 1:2]
        return spectral_drive.to(severity.dtype), gate_logit_bias.to(
            severity.dtype
        )


class DALSSCNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 32,
        num_blocks: int = 4,
        window_size: int = 16,
        num_bands: int = 8,
        spectral_strength: float = 0.25,
        spectral_strength_max: float = 0.35,
        spectral_overlap: float = 0.5,
        residual_bound: float = 0.25,
        severity_input_size: int = 96,
        severity_base_channels: int = 24,
        severity_embedding_dim: int = 128,
        severity_dropout: float = 0.10,
        use_severity_controller: bool = True,
        use_output_gate: bool = True,
        output_gate_init: float = 0.3,
        use_prior: bool = True,
        use_spectral: bool = True,
        use_spatial: bool = True,
        use_gate: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_prior = bool(use_prior)
        self.use_checkpoint = bool(use_checkpoint)
        self.residual_bound = float(residual_bound)
        self.use_output_gate = bool(use_output_gate)
        self.use_severity_controller = bool(use_severity_controller)

        self.embed = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.geo_encoder = MSGeometrySensitiveEncoder(channels)
        if self.use_prior:
            self.prior_estimator = DegradationPriorEstimator(channels)
        self.global_severity = ImageSpaceSiameseSeverityEncoder(
            input_size=severity_input_size,
            base_channels=severity_base_channels,
            embedding_dim=severity_embedding_dim,
            dropout=severity_dropout,
        )
        if self.use_severity_controller:
            self.severity_controller = SeverityController()
        self.blocks = nn.ModuleList(
            [
                LSSCBlock(
                    channels,
                    window_size=window_size,
                    num_bands=num_bands,
                    spectral_strength=spectral_strength,
                    spectral_strength_max=spectral_strength_max,
                    spectral_overlap=spectral_overlap,
                    use_spectral=use_spectral,
                    use_spatial=use_spatial,
                    use_gate=use_gate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.reconstruction = nn.Conv2d(
            channels, out_channels, 3, padding=1
        )
        if self.use_output_gate:
            self.restoration_gate = RestorationGate(
                channels, init_value=output_gate_init
            )

    def forward(
        self,
        x: torch.Tensor,
        return_prior: bool = False,
        return_gate: bool = False,
        return_severity: bool = False,
        return_ordinal: bool = False,
    ):
        f0 = self.embed(x)
        fg = self.geo_encoder(f0)
        if self.use_prior:
            prior = self.prior_estimator(fg, x)
        else:
            prior = torch.full(
                (x.shape[0], 1, x.shape[2], x.shape[3]),
                0.5,
                device=x.device,
                dtype=x.dtype,
            )

        ordinal_logits, severity, ordinal_probs = self.global_severity(x)
        severity = severity.to(dtype=x.dtype)
        if self.use_severity_controller:
            spectral_drive, gate_logit_bias = self.severity_controller(
                severity
            )
        else:
            spectral_drive = x.new_zeros((x.shape[0], 1))
            gate_logit_bias = x.new_zeros((x.shape[0], 1))

        features = fg
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                features = cp.checkpoint(
                    block,
                    features,
                    prior,
                    spectral_drive,
                    use_reentrant=False,
                )
            else:
                features = block(features, prior, spectral_drive)

        residual = self.reconstruction(features)
        if self.residual_bound > 0.0:
            bound = self.residual_bound
            residual = bound * torch.tanh(residual / bound)

        if self.use_output_gate:
            output_gate = self.restoration_gate(
                features, prior, gate_logit_bias
            )
            residual = residual * output_gate
        else:
            output_gate = torch.ones_like(prior)

        restored = x + residual
        outputs = [restored]
        if return_prior:
            outputs.append(prior)
        if return_gate:
            outputs.append(output_gate)
        if return_severity:
            outputs.append(severity)
        if return_ordinal:
            outputs.extend([ordinal_logits, ordinal_probs, spectral_drive])
        return restored if len(outputs) == 1 else tuple(outputs)


def build_model_from_config(cfg: dict) -> DALSSCNet:
    mcfg = cfg.get("model", cfg)
    return DALSSCNet(
        in_channels=mcfg.get("in_channels", 3),
        out_channels=mcfg.get("out_channels", 3),
        channels=mcfg.get("channels", 32),
        num_blocks=mcfg.get("num_blocks", 4),
        window_size=mcfg.get("window_size", 16),
        num_bands=mcfg.get("num_bands", 8),
        spectral_strength=mcfg.get("spectral_strength", 0.25),
        spectral_strength_max=mcfg.get("spectral_strength_max", 0.35),
        spectral_overlap=mcfg.get("spectral_overlap", 0.5),
        residual_bound=mcfg.get("residual_bound", 0.25),
        severity_input_size=mcfg.get("severity_input_size", 96),
        severity_base_channels=mcfg.get("severity_base_channels", 24),
        severity_embedding_dim=mcfg.get("severity_embedding_dim", 128),
        severity_dropout=mcfg.get("severity_dropout", 0.10),
        use_severity_controller=mcfg.get(
            "use_severity_controller", True
        ),
        use_output_gate=mcfg.get("use_output_gate", True),
        output_gate_init=mcfg.get("output_gate_init", 0.3),
        use_prior=mcfg.get("use_prior", True),
        use_spectral=mcfg.get("use_spectral", True),
        use_spatial=mcfg.get("use_spatial", True),
        use_gate=mcfg.get("use_gate", True),
        use_checkpoint=mcfg.get("use_checkpoint", False),
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    net = DALSSCNet(channels=8, num_blocks=1, window_size=8, num_bands=4)
    inp = torch.rand(1, 3, 24, 24)
    outputs = net(
        inp,
        return_prior=True,
        return_gate=True,
        return_severity=True,
        return_ordinal=True,
    )
    out, prior, gate, severity, logits, probs, drive = outputs
    print(tuple(out.shape), float(gate.mean()), float(severity.mean()), float(drive.mean()))
