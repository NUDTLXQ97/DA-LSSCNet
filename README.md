# DA-LSSCNet

Degradation-Aware Local Spectral–Spatial Coupling Network for Single-Image Atmospheric Turbulence Mitigation.

Run commands from this directory with Python 3.10 or newer. Keep your CUDA-enabled PyTorch and torchvision installation, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## 1. Training

The three stages use `main_train.py`. Edit data paths in `configs/train_*.yaml` before starting.

```text
data_p2s_mixed_v2/
├── train/
│   ├── turb/
│   └── gt/
└── val/
    ├── turb/
    └── gt/
metadata/
├── train_pair_groups.csv
└── val_pair_groups.csv
checkpoints/
├── restoration_anchor.pth
├── severity_selected.pth
├── connection_selected.pth
└── final_candidate.pth
```

Turbulent and GT images must have matching names and dimensions. Each severity CSV has columns `group_id,identity_files,mild_files,medium_files,strong_files`; multiple filenames in a cell are separated by semicolons. Image names start with `identity_`, `mild_`, `medium_`, or `strong_`.

Trained weights are supplied separately. The training workflow starts from `restoration_anchor.pth`, produced by the original anchor-pretraining run. This package contains severity training, connection training, and final adaptation. Inference requires the full `final_candidate.pth` checkpoint.

Configure `extra_val_sets` for your available validation folders. On Windows, set `num_workers` and `eval_num_workers` to 0 if multiprocessing causes problems.

### Severity encoder

```bash
python main_train.py --stage severity --device cuda
```

The default schedule is 30 epochs with batch size 16. Save the selected checkpoint as `checkpoints/severity_selected.pth`.

### Severity connection

```bash
python main_train.py --stage connection --anchor_checkpoint checkpoints/restoration_anchor.pth --severity_checkpoint checkpoints/severity_selected.pth --device cuda
```

This stage trains the controller for 3 epochs, with batch size 4 and two-step accumulation. Save the selected checkpoint as `checkpoints/connection_selected.pth`.

### Final adaptation

```bash
python main_train.py --stage final --init_checkpoint checkpoints/connection_selected.pth --teacher_checkpoint checkpoints/connection_selected.pth --device cuda
```

This stage trains the last two LSSC blocks and reconstruction head for 8 epochs with batch size 4. Save the selected checkpoint as `checkpoints/final_candidate.pth`.

Training outputs are saved under `experiments/`. Add `--resume PATH` to resume a run, or `--config PATH` to use another configuration. Use separate output directories for new experiments. The original single-GPU and distributed training implementations are in `training/`.

## 2. Inference

Folder inference preserves subdirectories and saves restored images in `results/restored/`:

```bash
python inference.py --input data/input --output results --checkpoint checkpoints/final_candidate.pth --device cuda --precision bf16
```

The same entry point handles one image:

```bash
python inference.py --input data/example.png --output results/example.png --checkpoint checkpoints/final_candidate.pth --device cuda --precision bf16
```

Add `--save_prior`, `--save_gate`, or `--save_severity` to export guidance maps or severity scores. Guidance PNGs are independently normalized for display. For a single input, outputs use the suffixes `_prior.png`, `_gate.png`, and `_severity.csv`; for folders, they use `prior/`, `output_gate/`, and `severity_scores.csv`.

Use distinct input and output locations. Existing outputs require `--overwrite`. Images must be at least 16×16; restored images retain their dimensions. CPU inference uses `--device cpu --precision fp32`.

## 3. Testing

PSNR and SSIM are computed from saved RGB images with matching names and dimensions:

```bash
python main_test.py --pred_dir results/restored --gt_dir data/gt --output_dir eval/paired --metrics psnr ssim
```

For LPIPS or no-reference metrics, install the optional packages:

```bash
python -m pip install lpips pyiqa
python main_test.py --pred_dir results/restored --gt_dir data/gt --output_dir eval/paired_lpips --metrics psnr ssim lpips --device cuda
python main_test.py --mode noref --pred_dir results/OTIS --output_dir eval/OTIS --metrics niqe brisque musiq --average sequence --device cuda
```

`--average sequence` averages each first-level scene directory and then gives scenes equal weight. A flat image folder can use `--sequence_regex` with one capture group. Default paired evaluation uses zero border crop, an 11×11 Gaussian SSIM window with sigma 1.5, and AlexNet LPIPS. Metric packages may download their pretrained resources on first use.

Results are saved as `per_image.csv` and `summary.json`, plus `per_sequence.csv` for sequence averaging. Images are evaluated at their saved resolution.

Severity verification uses the same test entry point:

```bash
python main_test.py --mode severity --config configs/train_severity.yaml --checkpoint checkpoints/severity_selected.pth --split val --output_dir eval/severity --device cuda
```

## 4. Network structure

```bash
python check_model.py --device cuda --size 256 --print-model
```

The default network has 303,152 parameters. This command reports output shapes, parameter count, latency, and CUDA peak memory using a random input. For a CPU check, use `--device cpu --size 32`. The architecture and trained-state keys are defined in `models/`.
