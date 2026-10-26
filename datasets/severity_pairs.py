"""Same-content image-pair dataset for image-space severity learning."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from PIL.Image import Resampling
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF

TYPES: Tuple[str, ...] = ("identity", "mild", "medium", "strong")
LEVEL: Dict[str, int] = {name: idx for idx, name in enumerate(TYPES)}
PAIR_TYPES: Tuple[Tuple[str, str], ...] = tuple(
    (TYPES[i], TYPES[j])
    for i in range(len(TYPES))
    for j in range(i + 1, len(TYPES))
)
PAIR_NAMES: Tuple[str, ...] = tuple(f"{low}<{high}" for low, high in PAIR_TYPES)


def _split_names(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


@dataclass(frozen=True)
class PairRecord:
    group_id: str
    low_type: str
    high_type: str
    low_files: Tuple[str, ...]
    high_files: Tuple[str, ...]

    @property
    def low_level(self) -> int:
        return LEVEL[self.low_type]

    @property
    def high_level(self) -> int:
        return LEVEL[self.high_type]

    @property
    def pair_name(self) -> str:
        return f"{self.low_type}<{self.high_type}"

    @property
    def level_gap(self) -> int:
        return self.high_level - self.low_level


def read_pair_records(
    pair_csv: str | Path,
    available_names: Iterable[str] | None = None,
) -> List[PairRecord]:
    pair_csv = Path(pair_csv)
    if not pair_csv.is_file():
        raise FileNotFoundError(pair_csv)
    available = set(available_names) if available_names is not None else None
    table = pd.read_csv(pair_csv)
    records: List[PairRecord] = []
    for row in table.to_dict(orient="records"):
        by_type: Dict[str, Tuple[str, ...]] = {}
        for sample_type in TYPES:
            names = _split_names(row.get(f"{sample_type}_files"))
            if available is not None:
                names = [name for name in names if name in available]
            if names:
                by_type[sample_type] = tuple(names)
        for low, high in PAIR_TYPES:
            if low in by_type and high in by_type:
                records.append(
                    PairRecord(
                        group_id=str(row.get("group_id", "")),
                        low_type=low,
                        high_type=high,
                        low_files=by_type[low],
                        high_files=by_type[high],
                    )
                )
    if not records:
        raise RuntimeError(f"No usable pair records found in {pair_csv}")
    return records


def _read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _shared_augment(
    low: Image.Image,
    high: Image.Image,
    enabled: bool,
    shared_photometric_prob: float,
) -> Tuple[Image.Image, Image.Image]:
    if not enabled:
        return low, high
    if random.random() < 0.5:
        low, high = TF.hflip(low), TF.hflip(high)
    if random.random() < 0.5:
        low, high = TF.vflip(low), TF.vflip(high)
    rotation = random.choice((0, 90, 180, 270))
    if rotation:
        low, high = TF.rotate(low, rotation), TF.rotate(high, rotation)

    # The same mild photometric transform is applied to both images so the
    # content relationship is preserved while discouraging scene memorization.
    if random.random() < shared_photometric_prob:
        brightness = random.uniform(0.92, 1.08)
        contrast = random.uniform(0.92, 1.08)
        gamma = random.uniform(0.92, 1.08)
        low = ImageEnhance.Brightness(low).enhance(brightness)
        high = ImageEnhance.Brightness(high).enhance(brightness)
        low = ImageEnhance.Contrast(low).enhance(contrast)
        high = ImageEnhance.Contrast(high).enhance(contrast)
        low = TF.adjust_gamma(low, gamma)
        high = TF.adjust_gamma(high, gamma)
    return low, high


class SeverityPairImageDataset(Dataset):
    """Sample same-content low/high degradation image pairs."""

    def __init__(
        self,
        turb_dir: str | Path,
        pair_csv: str | Path,
        anchor_targets: Mapping[str, float],
        base_margin: float = 0.08,
        augment: bool = True,
        shared_photometric_prob: float = 0.5,
        pair_loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.turb_dir = Path(turb_dir)
        if not self.turb_dir.is_dir():
            raise FileNotFoundError(self.turb_dir)
        self.files = {
            path.name: path
            for path in self.turb_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        }
        if not self.files:
            raise RuntimeError(f"No images found in {self.turb_dir}")
        self.records = read_pair_records(pair_csv, self.files.keys())
        self.anchor_targets = {key: float(value) for key, value in anchor_targets.items()}
        self.base_margin = float(base_margin)
        self.augment = bool(augment)
        self.shared_photometric_prob = float(shared_photometric_prob)
        self.pair_loss_weights = {
            name: float((pair_loss_weights or {}).get(name, 1.0))
            for name in PAIR_NAMES
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        low_name = random.choice(record.low_files)
        high_name = random.choice(record.high_files)
        low = _read_rgb(self.files[low_name])
        high = _read_rgb(self.files[high_name])
        if low.size != high.size:
            high = high.resize(low.size, Resampling.BICUBIC)
        low, high = _shared_augment(
            low,
            high,
            enabled=self.augment,
            shared_photometric_prob=self.shared_photometric_prob,
        )
        return {
            "low_image": TF.to_tensor(low),
            "high_image": TF.to_tensor(high),
            "low_level": record.low_level,
            "high_level": record.high_level,
            "low_target": self.anchor_targets[record.low_type],
            "high_target": self.anchor_targets[record.high_type],
            "margin": self.base_margin * record.level_gap,
            "loss_weight": self.pair_loss_weights[record.pair_name],
            "pair_name": record.pair_name,
            "group_id": record.group_id,
            "low_name": low_name,
            "high_name": high_name,
        }

    def balanced_sampler(
        self,
        samples_per_epoch: int,
        sampling_multipliers: Mapping[str, float] | None = None,
    ) -> WeightedRandomSampler:
        counts = Counter(record.pair_name for record in self.records)
        multipliers = sampling_multipliers or {}
        weights = [
            float(multipliers.get(record.pair_name, 1.0))
            / max(counts[record.pair_name], 1)
            for record in self.records
        ]
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(samples_per_epoch),
            replacement=True,
        )


class SeverityImageDataset(Dataset):
    """Unique images referenced by pair metadata, used for validation."""

    def __init__(self, turb_dir: str | Path, pair_csv: str | Path) -> None:
        self.turb_dir = Path(turb_dir)
        if not self.turb_dir.is_dir():
            raise FileNotFoundError(self.turb_dir)
        available = {
            path.name: path
            for path in self.turb_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        }
        records = read_pair_records(pair_csv, available.keys())
        names = sorted(
            {
                name
                for record in records
                for name in (*record.low_files, *record.high_files)
                if name in available
            }
        )
        if not names:
            raise RuntimeError("No validation images referenced by pair metadata.")
        self.names = names
        self.files = available
        self.records = records

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Dict[str, object]:
        name = self.names[index]
        sample_type = name.split("_", 1)[0].lower()
        if sample_type not in LEVEL:
            raise RuntimeError(f"Cannot infer severity type from filename: {name}")
        return {
            "image": TF.to_tensor(_read_rgb(self.files[name])),
            "name": name,
            "level": LEVEL[sample_type],
            "type": sample_type,
        }


def describe_records(records: Sequence[PairRecord]) -> Dict[str, int]:
    counts = Counter(record.pair_name for record in records)
    return {name: int(counts.get(name, 0)) for name in PAIR_NAMES}
