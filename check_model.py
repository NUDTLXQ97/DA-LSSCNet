#!/usr/bin/env python3
"""Print and validate the DA-LSSCNet architecture with a random input."""
from __future__ import annotations

import argparse
import time
import torch

from models.da_lsscnet import build_model_from_config
from utils.common import load_yaml, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DA-LSSCNet structure, outputs, parameters, and latency.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--print-model", action="store_true")
    args = parser.parse_args()

    if args.size < 16:
        raise ValueError("--size must be at least 16.")

    device = resolve_device(args.device)
    model = build_model_from_config(load_yaml(args.config)).to(device).eval()
    if args.print_model:
        print(model)
        print("=" * 80)

    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    image = torch.rand(1, 3, args.size, args.size, device=device)

    with torch.inference_mode():
        outputs = model(
            image,
            return_prior=True,
            return_gate=True,
            return_severity=True,
            return_ordinal=True,
        )
        restored, prior, gate, severity, ordinal_logits, ordinal_probs, spectral_drive = outputs
        for _ in range(max(args.warmup, 0)):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(max(args.runs, 1)):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0 / max(args.runs, 1)

    expected = {
        "restored": tuple(image.shape),
        "prior": (1, 1, args.size, args.size),
        "gate": (1, 1, args.size, args.size),
        "severity": (1, 1),
        "ordinal_logits": (1, 3),
        "ordinal_probs": (1, 3),
        "spectral_drive": (1, 1),
    }
    actual = {
        "restored": tuple(restored.shape),
        "prior": tuple(prior.shape),
        "gate": tuple(gate.shape),
        "severity": tuple(severity.shape),
        "ordinal_logits": tuple(ordinal_logits.shape),
        "ordinal_probs": tuple(ordinal_probs.shape),
        "spectral_drive": tuple(spectral_drive.shape),
    }
    for key, shape in expected.items():
        if actual[key] != shape:
            raise RuntimeError(f"Unexpected {key} shape: {actual[key]} != {shape}")

    print("DA-LSSCNet structure check passed.")
    print(f"Parameters : {params:,} ({params / 1e6:.6f} M)")
    print(f"Trainable  : {trainable:,}")
    for key, shape in actual.items():
        print(f"{key:15s}: {shape}")
    print(f"Latency    : {latency_ms:.3f} ms | batch=1 | {args.size}x{args.size} | {device}")
    if device.type == "cuda":
        print(f"Peak memory: {torch.cuda.max_memory_allocated(device) / 2**30:.4f} GiB")


if __name__ == "__main__":
    main()
