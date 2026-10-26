from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from PIL.Image import Resampling
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEGRADATION_TYPES = {"identity", "mild", "medium", "strong"}


def infer_degradation_type(stem: str) -> str:
    prefix = stem.split("_", 1)[0].lower()
    return prefix if prefix in DEGRADATION_TYPES else "unknown"


def list_images(root: str | Path) -> List[Path]:
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.suffix.lower() in IMG_EXTS)
    return files


def read_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def unique_index(paths: List[Path], key_fn, label: str) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in paths:
        key = key_fn(path)
        if key in index:
            raise RuntimeError(
                f"Duplicate {label} key {key!r}:\n  {index[key]}\n  {path}"
            )
        index[key] = path
    return index


class PairedImageDataset(Dataset):
    """Strict paired turbulent/GT image dataset.

    Files are matched by relative path first and then by a unique filename stem.
    By default, every turbulent image must have exactly one target and image sizes
    must match.  These checks prevent silent sample loss and accidental resizing
    from contaminating training or paper metrics.
    """

    def __init__(
        self,
        turb_dir: str,
        gt_dir: str,
        patch_size: int = 256,
        augment: bool = True,
        *,
        strict_pairs: bool = True,
        allow_resize_mismatch: bool = False,
    ) -> None:
        self.turb_dir = Path(turb_dir)
        self.gt_dir = Path(gt_dir)
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.strict_pairs = bool(strict_pairs)
        self.allow_resize_mismatch = bool(allow_resize_mismatch)

        turb_files = list_images(self.turb_dir)
        gt_files = list_images(self.gt_dir)
        if not turb_files:
            raise FileNotFoundError(f"No images found in turb_dir={self.turb_dir}")
        if not gt_files:
            raise FileNotFoundError(f"No images found in gt_dir={self.gt_dir}")

        gt_relative = unique_index(
            gt_files,
            lambda path: path.relative_to(self.gt_dir).as_posix(),
            "GT relative-path",
        )
        gt_stem = unique_index(gt_files, lambda path: path.stem, "GT stem")

        pairs: List[Tuple[Path, Path]] = []
        unmatched: List[Path] = []
        for turb_path in turb_files:
            relative = turb_path.relative_to(self.turb_dir).as_posix()
            gt_path = gt_relative.get(relative) or gt_stem.get(turb_path.stem)
            if gt_path is None:
                unmatched.append(turb_path)
            else:
                pairs.append((turb_path, gt_path))

        if unmatched and self.strict_pairs:
            preview = "\n  ".join(str(path) for path in unmatched[:10])
            raise RuntimeError(
                f"{len(unmatched)} turbulent images have no GT match. Examples:\n  {preview}"
            )
        if not pairs:
            raise RuntimeError(
                "No paired turbulent/GT images matched. Check filenames and folders."
            )

        self.pairs = pairs
        self.degradation_types = [
            infer_degradation_type(turb_path.stem) for turb_path, _ in pairs
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop_pair(
        self, turb: Image.Image, gt: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        width, height = turb.size
        patch_size = self.patch_size
        if patch_size <= 0:
            return turb, gt
        if width < patch_size or height < patch_size:
            new_width, new_height = max(width, patch_size), max(height, patch_size)
            turb = TF.resize(
                turb, [new_height, new_width], interpolation=Resampling.BICUBIC
            )
            gt = TF.resize(gt, [new_height, new_width], interpolation=Resampling.BICUBIC)
            width, height = turb.size
        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        box = (left, top, left + patch_size, top + patch_size)
        return turb.crop(box), gt.crop(box)

    @staticmethod
    def _augment_pair(
        turb: Image.Image, gt: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            turb, gt = TF.hflip(turb), TF.hflip(gt)
        if random.random() < 0.5:
            turb, gt = TF.vflip(turb), TF.vflip(gt)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            turb, gt = TF.rotate(turb, angle), TF.rotate(gt, angle)
        return turb, gt

    def __getitem__(self, index: int):
        turb_path, gt_path = self.pairs[index]
        turb, gt = read_rgb(turb_path), read_rgb(gt_path)
        if turb.size != gt.size:
            if not self.allow_resize_mismatch:
                raise RuntimeError(
                    f"Paired image size mismatch for {turb_path.name}: "
                    f"turb={turb.size}, gt={gt.size}. Regenerate aligned pairs or "
                    "construct the dataset with allow_resize_mismatch=True for diagnostics."
                )
            gt = gt.resize(turb.size, Resampling.BICUBIC)
        if self.patch_size > 0:
            turb, gt = self._crop_pair(turb, gt)
        if self.augment:
            turb, gt = self._augment_pair(turb, gt)
        degradation_type = infer_degradation_type(turb_path.stem)
        turb_tensor = TF.to_tensor(turb)
        return {
            "turb": turb_tensor,
            "lq": turb_tensor,  # backward-compatible alias
            "gt": TF.to_tensor(gt),
            "name": turb_path.stem,
            "turb_path": str(turb_path),
            "lq_path": str(turb_path),
            "gt_path": str(gt_path),
            "degradation_type": degradation_type,
            "is_identity": degradation_type == "identity",
        }
