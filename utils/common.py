from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML mapping: {path}")
    return data


def save_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build, or select --device cpu explicitly.")
    return torch.device(name)


def resolve_precision(device: torch.device, name: str):
    name = name.lower()
    if device.type != "cuda" or name == "fp32":
        return False, torch.float32
    if name in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            print("[warn] BF16 unsupported; using FP32.")
            return False, torch.float32
        return True, torch.bfloat16
    if name in {"fp16", "float16"}:
        return True, torch.float16
    raise ValueError(f"Unsupported precision: {name}")




def load_checkpoint(path: str | Path, map_location: object = "cpu") -> object:
    """Load tensor/basic-type checkpoints by default.

    For a trusted legacy checkpoint containing custom Python objects, explicitly
    set DA_LSSCNET_TRUST_CHECKPOINT=1 for that invocation.
    """
    trusted = os.environ.get("DA_LSSCNET_TRUST_CHECKPOINT", "0") == "1"
    try:
        return torch.load(path, map_location=map_location, weights_only=not trusted)
    except TypeError as exc:
        raise RuntimeError("Use a recent PyTorch with weights_only checkpoint loading.") from exc
    except Exception as exc:
        if not trusted:
            raise RuntimeError(
                "Restricted checkpoint loading failed. Verify the checkpoint path and source. "
                "For your own trusted legacy checkpoint only, set "
                "DA_LSSCNET_TRUST_CHECKPOINT=1 and retry."
            ) from exc
        raise


def extract_model_state(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
            return checkpoint["model"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            return checkpoint["state_dict"]
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint  # type: ignore[return-value]
    raise RuntimeError("Checkpoint does not contain a recognizable model state.")


def load_restoration_anchor(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    raw_state = extract_model_state(checkpoint)
    # Ignore any obsolete global-severity/connection parameters that may be
    # present in a v2.1/v2.2 checkpoint.  The restoration anchor is defined by
    # the shared restoration backbone, prior, LSSC blocks, reconstruction and
    # local output gate.
    ignored_prefixes = (
        "global_severity.",
        "severity_controller.",
        "restoration_gate.severity_to_logit.",
    )
    state = {
        key: value
        for key, value in raw_state.items()
        if not key.startswith(ignored_prefixes)
    }
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = ("global_severity.", "severity_controller.")
    bad_missing = [key for key in missing if not key.startswith(allowed_missing)]
    if bad_missing:
        raise RuntimeError(
            "Restoration anchor is incompatible. Unexpected missing keys:\n  "
            + "\n  ".join(bad_missing)
        )
    if unexpected:
        raise RuntimeError(
            "Restoration anchor has unexpected keys:\n  "
            + "\n  ".join(unexpected)
        )
    print(f"Loaded restoration anchor: {checkpoint_path}")
    print(f"Expected new keys missing: {len(missing)}")
    return missing, unexpected


def load_severity_head(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or "global_severity" not in checkpoint:
        raise RuntimeError(
            "Severity checkpoint must contain a 'global_severity' state dict."
        )
    model.global_severity.load_state_dict(checkpoint["global_severity"], strict=True)
    print(f"Loaded image-space Siamese severity encoder: {checkpoint_path}")


def trainable_parameter_summary(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_tensors": tensors}


def set_trainable_prefixes(
    model: torch.nn.Module, prefixes: Sequence[str]
) -> None:
    prefixes = tuple(prefixes)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        )


def cosine_lr(base_lr: float, min_lr: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return min_lr
    ratio = min(max(step / (total_steps - 1), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))
