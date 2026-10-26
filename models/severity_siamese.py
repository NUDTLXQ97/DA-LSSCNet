"""Image-space Siamese ordinal severity encoder for DA-LSSCNet.

The encoder reads the degraded RGB image directly rather than relying on a
cached hand-crafted descriptor.  During severity training the same encoder is
applied to a same-content low/high degradation pair.  The model predicts a
single latent score with three ordered thresholds (CORAL-style ordinal
regression), plus a four-class auxiliary classifier.

The image representation combines a compact RGB stream with differentiable
high-pass cues.  This keeps the network small while making mild/medium blur
changes easier to distinguish than with global pooled statistics alone.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LEVEL_NAMES = ("identity", "mild", "medium", "strong")


def _group_count(channels: int) -> int:
    # Keep at least two channels per group so training also works for tiny
    # smoke-test inputs whose last spatial map can be 1x1.
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvGNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvGNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )
        self.use_residual = stride == 1 and in_channels == out_channels
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pointwise(self.depthwise(x))
        if self.use_residual:
            return x + self.residual_scale * y
        return y


def _fixed_sobel(
    gray: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = gray.dtype
    device = gray.device
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(2, 3).contiguous()
    lap_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(gx.float().square() + gy.float().square() + 1e-6)
    lap = F.conv2d(gray, lap_kernel, padding=1).abs().float()
    return gx.float(), grad, lap


def _build_input_channels(x: torch.Tensor) -> torch.Tensor:
    """Return RGB + content-normalized high-frequency cues (6 channels)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x32 = x.float().clamp(0.0, 1.0)
        gray = (
            0.2989 * x32[:, 0:1]
            + 0.5870 * x32[:, 1:2]
            + 0.1140 * x32[:, 2:3]
        )
        coarse = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        highpass = (gray - coarse).abs()
        _, grad, lap = _fixed_sobel(gray)

        # Compress large responses but preserve their absolute magnitude.
        highpass = torch.log1p(12.0 * highpass) / math.log(13.0)
        grad = torch.log1p(12.0 * grad) / math.log(13.0)
        lap = torch.log1p(6.0 * lap) / math.log(7.0)
        rgb = (x32 - 0.5) / 0.5
        return torch.cat([rgb, highpass, grad, lap], dim=1)


class ImageSpaceSiameseSeverityEncoder(nn.Module):
    """Compact image-space ordinal severity encoder.

    Parameters are intentionally modest so the encoder can be trained on a
    single 16-GB GPU with pair batches.  Inputs of arbitrary spatial size are
    resized internally to ``input_size``.
    """

    NUM_ORDINAL_LOGITS = 3

    def __init__(
        self,
        input_size: int = 96,
        base_channels: int = 24,
        embedding_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        c1 = int(base_channels)
        c2 = int(round(c1 * 4 / 3))
        c3 = int(round(c1 * 2))
        c4 = int(round(c1 * 8 / 3))
        c5 = int(round(c1 * 4))

        self.backbone = nn.Sequential(
            ConvGNAct(6, c1, kernel_size=5, stride=2),
            DepthwiseSeparableBlock(c1, c2, stride=2),
            DepthwiseSeparableBlock(c2, c3, stride=2),
            DepthwiseSeparableBlock(c3, c4, stride=2),
            DepthwiseSeparableBlock(c4, c5, stride=2),
            DepthwiseSeparableBlock(c5, c5, stride=1),
        )
        pooled_dim = c5 * 2
        self.embedding = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.score = nn.Linear(embedding_dim, 1)
        self.classifier = nn.Linear(embedding_dim, 4)

        # Ordered thresholds t0 < t1 < t2.
        self.threshold_base = nn.Parameter(torch.tensor(-0.75))
        self.threshold_gap_raw = nn.Parameter(torch.tensor([-0.1, -0.1]))

    @staticmethod
    def ordinal_target(level: torch.Tensor) -> torch.Tensor:
        thresholds = torch.arange(3, device=level.device)[None, :]
        return (level[:, None] > thresholds).float()

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.threshold_gap_raw) + 1e-4
        t0 = self.threshold_base.view(1)
        t1 = t0 + gaps[0:1]
        t2 = t1 + gaps[1:2]
        return torch.cat([t0, t1, t2], dim=0)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                f"Expected BCHW RGB input, got shape={tuple(image.shape)}"
            )
        resized = F.interpolate(
            image,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        features = self.backbone(_build_input_channels(resized))
        mean = features.mean(dim=(2, 3))
        var = (features - mean[:, :, None, None]).square().mean(dim=(2, 3))
        std = torch.sqrt(var + 1e-6)
        return self.embedding(torch.cat([mean, std], dim=1))

    def forward_with_aux(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encode(image)
        score = self.score(embedding)
        logits = score - self.thresholds()[None, :]
        probabilities = torch.sigmoid(logits)
        severity = probabilities.mean(dim=1, keepdim=True)
        class_logits = self.classifier(embedding)
        return logits, severity, probabilities, class_logits, embedding

    def forward(
        self, image: torch.Tensor, return_aux: bool = False
    ):
        outputs = self.forward_with_aux(image)
        if return_aux:
            return outputs
        logits, severity, probabilities, _, _ = outputs
        return logits, severity, probabilities


def build_severity_encoder_from_config(cfg: dict) -> ImageSpaceSiameseSeverityEncoder:
    model_cfg = cfg.get("model", cfg)
    return ImageSpaceSiameseSeverityEncoder(
        input_size=int(model_cfg.get("severity_input_size", 96)),
        base_channels=int(model_cfg.get("severity_base_channels", 24)),
        embedding_dim=int(model_cfg.get("severity_embedding_dim", 128)),
        dropout=float(model_cfg.get("severity_dropout", 0.10)),
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    net = ImageSpaceSiameseSeverityEncoder(input_size=64, base_channels=8, embedding_dim=32)
    image = torch.rand(2, 3, 80, 96)
    logits, severity, probabilities, class_logits, embedding = net.forward_with_aux(image)
    print(logits.shape, severity.shape, probabilities.shape, class_logits.shape, embedding.shape)
