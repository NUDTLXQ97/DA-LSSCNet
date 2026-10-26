"""Restore a single image or an image folder with DA-LSSCNet."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

from models.da_lsscnet import build_model_from_config
from utils.common import extract_model_state, load_checkpoint, load_yaml, resolve_device, resolve_precision
from utils.image_utils import crop_to_size, pad_to_multiple, save_prior_map, save_tensor_image

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Unsupported image format: {source}')
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    paths = source.rglob('*') if recursive else source.iterdir()
    images = sorted(p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f'No supported images in {source}')
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', '--input_dir', dest='input', required=True)
    parser.add_argument('--output', '--output_dir', dest='output', required=True)
    parser.add_argument('--checkpoint', default='checkpoints/final_candidate.pth')
    parser.add_argument('--config', default='configs/model.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    parser.add_argument('--flat', action='store_true', help='Read only the top-level input folder.')
    parser.add_argument('--save_prior', action='store_true')
    parser.add_argument('--save_gate', action='store_true')
    parser.add_argument('--save_severity', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    source, output = Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve()
    images = collect_images(source, recursive=not args.flat)
    single = source.is_file()
    protected = {source, Path(args.config).resolve(), Path(args.checkpoint).resolve()}
    if single:
        if output.suffix.lower() != '.png':
            raise ValueError('For a single image, --output must be a .png file.')
        restored = [output]
        priors = [output.with_name(output.stem + '_prior.png')]
        gates = [output.with_name(output.stem + '_gate.png')]
        scores_path = output.with_name(output.stem + '_severity.csv')
    else:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('Input and output folders must be separate, non-nested directories.')
        if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not args.overwrite)):
            raise FileExistsError(f'Choose an empty output folder or use --overwrite: {output}')
        relative = [p.relative_to(source).with_suffix('.png') for p in images]
        restored = [output / 'restored' / p for p in relative]
        priors = [output / 'prior' / p for p in relative]
        gates = [output / 'output_gate' / p for p in relative]
        scores_path = output / 'severity_scores.csv'

    targets = restored + (priors if args.save_prior else []) + (gates if args.save_gate else [])
    if args.save_severity:
        targets.append(scores_path)
    if len({str(p).casefold() for p in targets}) != len(targets):
        raise ValueError('Multiple inputs map to the same PNG output; rename the duplicate files.')
    protected.update(images)
    for path in targets:
        if path in protected:
            raise ValueError(f'Output would replace an input, config, or checkpoint: {path}')
        if path.exists() and (not path.is_file() or not args.overwrite):
            raise FileExistsError(f'Output already exists; use --overwrite: {path}')

    config = load_yaml(args.config)
    device = resolve_device(args.device)
    model = build_model_from_config(config).to(device)
    model.load_state_dict(extract_model_state(load_checkpoint(args.checkpoint, map_location=device)), strict=True)
    model.eval()
    use_amp, dtype = resolve_precision(device, args.precision)
    multiple = int(config.get('model', {}).get('window_size', 16))
    multiple = multiple if multiple > 0 else 16
    scores = []
    with torch.inference_mode():
        for i, path in enumerate(tqdm(images, desc='Restoring')):
            with Image.open(path) as im:
                image = TF.to_tensor(im.convert('RGB')).unsqueeze(0).to(device)
            # Reflection padding requires both dimensions to exceed the padding size.
            if min(image.shape[-2:]) < 16:
                raise ValueError(f'Input images must be at least 16x16: {path}')
            padded, size = pad_to_multiple(image, multiple)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                prediction, prior, gate, severity = model(
                    padded, return_prior=True, return_gate=True, return_severity=True
                )
            save_tensor_image(crop_to_size(prediction, size), restored[i])
            if args.save_prior:
                save_prior_map(crop_to_size(prior, size), priors[i])
            if args.save_gate:
                save_prior_map(crop_to_size(gate, size), gates[i])
            if args.save_severity:
                name = path.name if single else path.relative_to(source).as_posix()
                scores.append({'image': name, 'severity': float(severity.float().mean())})
    if scores:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image', 'severity'])
            writer.writeheader()
            writer.writerows(scores)
    print(f'Saved {len(images)} restored image(s) to {output}')


if __name__ == '__main__':
    main()
