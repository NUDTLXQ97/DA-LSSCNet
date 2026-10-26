"""Evaluate restored images or the ordinal severity encoder."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_STRIP = [
    "_restored",
    "_output",
    "_out",
    "_pred",
    "_result",
    "_sr",
    "_dn",
    "_clean",
]


def list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMG_EXTS)
    if not files:
        raise FileNotFoundError(f"No supported images found in: {root}")
    return files


def normalize_stem(stem: str, strip_suffixes: Sequence[str]) -> str:
    value = stem
    for suffix in strip_suffixes:
        if suffix and value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def build_index(root: Path, strip_suffixes: Sequence[str], key_mode: str = "relative") -> Dict[str, Path]:
    """Build a collision-free normalized-stem index."""
    index: Dict[str, Path] = {}
    collisions: Dict[str, List[Path]] = {}
    for path in list_images(root):
        key = normalize_stem(path.stem, strip_suffixes)
        if key_mode == "relative":
            key = (path.relative_to(root).parent / key).as_posix()
        if key in index:
            collisions.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path
    if collisions:
        key, paths = next(iter(collisions.items()))
        joined = "\n  ".join(str(path) for path in paths)
        raise RuntimeError(
            f"Filename collision after suffix normalization for stem {key!r}:\n  {joined}\n"
            "Rename the files or use a more specific --strip_suffix rule."
        )
    return index


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def to_y_channel(image: np.ndarray) -> np.ndarray:
    image = image / 255.0
    return np.dot(image, [65.481, 128.553, 24.966]) + 16.0


def crop_image(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    if image.shape[0] <= 2 * border or image.shape[1] <= 2 * border:
        raise ValueError(
            f"crop_border={border} is too large for image shape {image.shape}."
        )
    return image[border:-border, border:-border, ...]


def calc_psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    crop_border: int,
    y_channel: bool,
) -> float:
    prediction = crop_image(prediction, crop_border).astype(np.float64)
    target = crop_image(target, crop_border).astype(np.float64)
    if y_channel:
        prediction, target = to_y_channel(prediction), to_y_channel(target)
    mse = np.mean((prediction - target) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size) - (size - 1) / 2.0
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return np.outer(kernel, kernel)


def ssim_single(prediction: np.ndarray, target: np.ndarray) -> float:
    import torch
    import torch.nn.functional as F

    if prediction.shape[0] < 11 or prediction.shape[1] < 11:
        raise ValueError(
            f"SSIM requires images at least 11x11, got {prediction.shape}."
        )
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    window = torch.from_numpy(gaussian_kernel_2d()).float().view(1, 1, 11, 11)
    pred_tensor = torch.from_numpy(prediction).float().view(1, 1, *prediction.shape)
    target_tensor = torch.from_numpy(target).float().view(1, 1, *target.shape)
    mu1 = F.conv2d(pred_tensor, window)
    mu2 = F.conv2d(target_tensor, window)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1 = F.conv2d(pred_tensor * pred_tensor, window) - mu1_sq
    sigma2 = F.conv2d(target_tensor * target_tensor, window) - mu2_sq
    sigma12 = F.conv2d(pred_tensor * target_tensor, window) - mu12
    score = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return float(score.mean().item())


def calc_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    crop_border: int,
    y_channel: bool,
) -> float:
    prediction = crop_image(prediction, crop_border).astype(np.float64)
    target = crop_image(target, crop_border).astype(np.float64)
    if y_channel:
        return ssim_single(to_y_channel(prediction), to_y_channel(target))
    return float(
        np.mean(
            [
                ssim_single(prediction[..., channel], target[..., channel])
                for channel in range(prediction.shape[2])
            ]
        )
    )


class LPIPSWrapper:
    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        import lpips  # type: ignore
        import torch

        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        torch = self.torch

        def prepare(image: np.ndarray):
            tensor = (
                torch.from_numpy(image / 255.0 * 2.0 - 1.0)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            return tensor.to(self.device)

        with torch.no_grad():
            return float(self.model(prepare(prediction), prepare(target)).item())


def init_pyiqa(metric_names: Iterable[str], device: str):
    import pyiqa
    output = {}
    for name in metric_names:
        output[name] = pyiqa.create_metric(name, device=device)
        print(f"IQA metric {name}: OK")
    return output


def pyiqa_score(metric, image: np.ndarray, device: str) -> float:
    import torch

    tensor = (
        torch.from_numpy(image / 255.0)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        return float(metric(tensor).detach().mean().item())


def evaluate_severity(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from datasets.severity_pairs import SeverityImageDataset
    from models.severity_siamese import build_severity_encoder_from_config
    from utils.common import load_checkpoint, load_yaml, resolve_device
    from utils.severity_metrics import evaluate_severity_predictions, predict_image_dataset, save_severity_report

    cfg = load_yaml(args.config)
    data_cfg = cfg["severity_data"]
    train_cfg = cfg["severity_train"]
    loss_cfg = cfg["severity_loss"]
    dataset = SeverityImageDataset(
        data_cfg[f"{args.split}_turb_dir"],
        data_cfg[f"{args.split}_pair_csv"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("eval_batch_size", 64)),
        shuffle=False,
        num_workers=int(train_cfg.get("eval_num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    encoder = build_severity_encoder_from_config(cfg)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if "global_severity" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain global_severity.")
    encoder.load_state_dict(checkpoint["global_severity"], strict=True)
    device = resolve_device(args.device)
    encoder.to(device)
    names, severity, probabilities, class_logits = predict_image_dataset(
        encoder,
        loader,
        device=device,
        precision=str(train_cfg.get("precision", "bf16")),
    )
    summary, pair_df, sample_df = evaluate_severity_predictions(
        names=names,
        severity=severity,
        probabilities=probabilities,
        class_logits=class_logits,
        records=dataset.records,
        anchor_targets={
            key: float(value)
            for key, value in loss_cfg["anchor_targets"].items()
        },
        margin_base=float(loss_cfg.get("base_margin", 0.08)),
    )
    save_severity_report(Path(args.output_dir), summary, pair_df, sample_df)
    print(summary)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['paired', 'noref', 'severity'], default='paired')
    parser.add_argument('--pred_dir', default='')
    parser.add_argument('--gt_dir', default='')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--metrics', nargs='+', default=None)
    parser.add_argument('--no_ref', action='store_true', help='Alias for --mode noref.')
    parser.add_argument('--crop_border', type=int, default=0)
    parser.add_argument('--test_y_channel', action='store_true')
    parser.add_argument('--lpips_net', default='alex', choices=['alex', 'vgg', 'squeeze'])
    parser.add_argument('--strip_suffix', nargs='*', default=[])
    parser.add_argument('--key_mode', choices=['relative', 'stem'], default='relative')
    parser.add_argument('--average', choices=['image', 'sequence'], default='image')
    parser.add_argument('--sequence_regex', default='', help='Capture group identifying a sequence in flat filenames.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--config', default='configs/train_severity.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/severity_selected.pth')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    args = parser.parse_args()

    if args.no_ref:
        if args.mode == 'severity':
            parser.error('--no_ref cannot be combined with --mode severity.')
        args.mode = 'noref'
    if args.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; choose --device cpu explicitly.')
    if args.mode == 'severity':
        evaluate_severity(args)
        return
    if not args.pred_dir or (args.mode == 'paired' and not args.gt_dir):
        parser.error('Provide --pred_dir and, for paired evaluation, --gt_dir.')
    if args.crop_border < 0:
        parser.error('--crop_border must be nonnegative.')

    allowed = {'psnr', 'ssim', 'lpips'} if args.mode == 'paired' else {'niqe', 'brisque', 'musiq'}
    defaults = ['psnr', 'ssim'] if args.mode == 'paired' else ['niqe', 'brisque', 'musiq']
    names = list(dict.fromkeys(name.lower() for name in (args.metrics or defaults)))
    if set(names) - allowed:
        parser.error(f'{args.mode} metrics: {", ".join(sorted(allowed))}')
    source, output = Path(args.pred_dir).resolve(), Path(args.output_dir).resolve()
    for root in [source, *([Path(args.gt_dir).resolve()] if args.gt_dir else [])]:
        if output == root or output in root.parents or root in output.parents:
            raise ValueError('Evaluation output must be separate from image/GT directories.')
    rows = []
    if args.mode == 'paired':
        pred = build_index(source, DEFAULT_STRIP + args.strip_suffix, args.key_mode)
        gt = build_index(Path(args.gt_dir), [], args.key_mode)
        if pred.keys() != gt.keys():
            raise ValueError(f'Prediction/GT mismatch: {len(gt.keys()-pred.keys())} missing, '
                             f'{len(pred.keys()-gt.keys())} extra predictions.')
        lpips_fn = LPIPSWrapper(args.lpips_net, args.device) if 'lpips' in names else None
        for key in sorted(pred):
            image, target = read_rgb(pred[key]), read_rgb(gt[key])
            if image.shape != target.shape:
                raise ValueError(f'Image/GT size mismatch: {key}: {image.shape} / {target.shape}')
            row = {'image': key}
            for name in names:
                if name == 'psnr':
                    row[name] = calc_psnr(image, target, args.crop_border, args.test_y_channel)
                elif name == 'ssim':
                    row[name] = calc_ssim(image, target, args.crop_border, args.test_y_channel)
                else:
                    row[name] = lpips_fn(crop_image(image, args.crop_border), crop_image(target, args.crop_border))
            rows.append(row)
    else:
        metrics = init_pyiqa(names, args.device)
        for path in list_images(source):
            image = read_rgb(path)
            rows.append({'image': path.relative_to(source).as_posix(),
                         **{name: pyiqa_score(metrics[name], image, args.device) for name in names}})

    for row in rows:
        for name in names:
            value = row[name]
            if not np.isfinite(value) and not (name == 'psnr' and value == float('inf')):
                raise RuntimeError(f'Invalid {name} for {row["image"]}: {value}')
    averages = rows
    group_rows = []
    if args.average == 'sequence':
        groups = {}
        for row in rows:
            key = row['image']
            if args.sequence_regex:
                match = re.search(args.sequence_regex, key)
                if match is None or match.lastindex is None:
                    raise ValueError(f'No sequence capture in {key}')
                group = match.group(1)
            else:
                if '/' not in key:
                    raise ValueError('Sequence averaging requires subfolders or --sequence_regex.')
                group = key.split('/', 1)[0]
            groups.setdefault(group, []).append(row)
        for group, items in sorted(groups.items()):
            group_rows.append({'sequence': group, 'num_images': len(items),
                               **{name: float(np.mean([r[name] for r in items])) for name in names}})
        averages = group_rows
    summary = {'num_images': len(rows), 'mode': args.mode, 'average': args.average,
               'metrics': names, 'crop_border': args.crop_border,
               'test_y_channel': args.test_y_channel, 'lpips_net': args.lpips_net}
    if group_rows:
        summary['num_sequences'] = len(group_rows)
    for name in names:
        value = float(np.mean([row[name] for row in averages]))
        summary[name] = value if np.isfinite(value) else 'Infinity'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'per_image.csv', rows)
    if group_rows:
        write_csv(output / 'per_sequence.csv', group_rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
