#!/usr/bin/env python3
"""Train severity connection or overlap-aware restoration adaptation.

The script supports single-GPU execution and single-node DDP via torchrun.
In DDP mode, ``train.batch_size`` is per GPU and
``train.samples_per_epoch`` is global across all ranks.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets.paired_image_dataset import PairedImageDataset
from models.da_lsscnet import build_model_from_config
from utils.common import (
    extract_model_state,
    load_checkpoint,
    load_restoration_anchor,
    load_severity_head,
    load_yaml,
    resolve_precision,
    set_trainable_prefixes,
    trainable_parameter_summary,
)
from utils.distributed import (
    DistributedEvalSampler,
    all_reduce_sum,
    barrier,
    cleanup_distributed,
    distributed_sampler_from_weighted,
    init_distributed,
    rank_print,
    reduce_scalar_dict,
    seed_worker,
    set_rank_seed,
    unwrap_model,
    wrap_ddp,
)
from utils.image_utils import save_prior_map, save_tensor_image
from utils.losses import (
    CharbonnierLoss,
    gradient_loss,
    local_log_amplitude_frequency_loss,
)
from utils.metrics import psnr, ssim


DEGRADATION_ORDER = ("identity", "mild", "medium", "strong")


def weighted_sampler(
    dataset: PairedImageDataset,
    target_ratio: Mapping[str, float],
    samples_per_epoch: int,
) -> WeightedRandomSampler:
    counts = Counter(dataset.degradation_types)
    weights = []
    for sample_type in dataset.degradation_types:
        desired = float(target_ratio.get(sample_type, 0.0))
        count = counts.get(sample_type, 0)
        weights.append(desired / max(count, 1))
    if not any(weight > 0 for weight in weights):
        raise RuntimeError("All sampler weights are zero.")
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(samples_per_epoch),
        replacement=True,
    )


def range_loss(pred: torch.Tensor) -> torch.Tensor:
    return (F.relu(-pred).square() + F.relu(pred - 1.0).square()).mean()


def identity_mask(values: Iterable[object], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [bool(value) for value in values], device=device, dtype=torch.bool
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_psnr: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_psnr": float(best_psnr),
            "config": config,
        },
        path,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    context,
    save_dir: Path | None = None,
    max_save: int = 0,
) -> Dict[str, float]:
    """Validate on a rank-local shard and all-reduce exact global sums."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_gate = 0.0
    total_severity = 0.0
    total_drive = 0.0
    count = 0
    saved = 0
    for batch in loader:
        turb = batch["turb"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            outputs = model(
                turb,
                return_prior=True,
                return_gate=True,
                return_severity=True,
                return_ordinal=True,
            )
            pred, prior, gate, severity, _, _, drive = outputs
        pred_clamped = pred.clamp(0, 1)
        batch_size = turb.shape[0]
        for index in range(batch_size):
            total_psnr += psnr(
                pred_clamped[index : index + 1], gt[index : index + 1]
            )
            total_ssim += ssim(
                pred_clamped[index : index + 1], gt[index : index + 1]
            )
            total_gate += float(gate[index].float().mean())
            total_severity += float(severity[index].float().mean())
            total_drive += float(drive[index].float().mean())
            if (
                context.is_main
                and save_dir is not None
                and saved < max_save
            ):
                name = str(batch["name"][index])
                save_tensor_image(pred[index], save_dir / f"{name}_restored.png")
                save_prior_map(
                    prior[index : index + 1], save_dir / f"{name}_prior.png"
                )
                save_prior_map(
                    gate[index : index + 1], save_dir / f"{name}_gate.png"
                )
                saved += 1
            count += 1
    stats = torch.tensor(
        [
            total_psnr,
            total_ssim,
            total_gate,
            total_severity,
            total_drive,
            float(count),
        ],
        dtype=torch.float64,
        device=device,
    )
    all_reduce_sum(stats, context)
    global_count = int(stats[5].item())
    if global_count == 0:
        raise RuntimeError("Validation loader is empty.")
    return {
        "num_images": global_count,
        "psnr": float(stats[0].item() / global_count),
        "ssim": float(stats[1].item() / global_count),
        "gate_mean": float(stats[2].item() / global_count),
        "severity_mean": float(stats[3].item() / global_count),
        "spectral_drive_mean": float(stats[4].item() / global_count),
    }


def build_loader(
    dataset: PairedImageDataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = False,
    sampler=None,
    drop_last: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor_checkpoint", default="")
    parser.add_argument("--severity_checkpoint", default="")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    args = parser.parse_args()

    chosen = sum(bool(value) for value in [args.resume, args.init_checkpoint])
    if chosen > 1:
        raise ValueError("Use only one of --resume or --init_checkpoint.")
    if not args.resume and not args.init_checkpoint:
        if not args.anchor_checkpoint or not args.severity_checkpoint:
            raise ValueError(
                "Fresh connection training requires --anchor_checkpoint and "
                "--severity_checkpoint."
            )

    context = init_distributed(args.device)
    try:
        cfg = load_yaml(args.config)
        set_rank_seed(args.seed, context)
        device = context.device
        train_cfg = cfg["train"]
        data_cfg = cfg["data"]
        loss_cfg = cfg["loss"]
        output_dir = Path(cfg["output_dir"])
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoints").mkdir(exist_ok=True)
        barrier()

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
            torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
            torch.backends.cudnn.allow_tf32 = bool(train_cfg.get("allow_tf32", True))

        raw_model = build_model_from_config(cfg).to(device)
        start_epoch = 1
        best_primary_psnr = -1.0
        resume_payload = None
        if args.resume:
            resume_payload = load_checkpoint(args.resume, map_location=device)
            raw_model.load_state_dict(
                extract_model_state(resume_payload), strict=True
            )
            start_epoch = int(resume_payload["epoch"]) + 1
            best_primary_psnr = float(
                resume_payload.get("best_psnr", -1.0)
            )
            rank_print(context, f"Resumed model from {args.resume}")
        elif args.init_checkpoint:
            payload = load_checkpoint(args.init_checkpoint, map_location=device)
            raw_model.load_state_dict(extract_model_state(payload), strict=True)
            rank_print(
                context, f"Initialized full DA-LSSCNet model from {args.init_checkpoint}"
            )
        else:
            load_restoration_anchor(raw_model, args.anchor_checkpoint, device)
            load_severity_head(raw_model, args.severity_checkpoint, device)

        teacher_output_weight = float(loss_cfg.get("lambda_teacher_output", 0.0))
        teacher_gate_weight = float(loss_cfg.get("lambda_teacher_gate", 0.0))
        teacher_types = {
            str(value) for value in loss_cfg.get("teacher_types", ["identity", "mild"])
        }
        teacher_model = None
        if teacher_output_weight > 0.0 or teacher_gate_weight > 0.0:
            teacher_path = args.teacher_checkpoint or args.init_checkpoint
            if not teacher_path:
                raise ValueError(
                    "Teacher-preserving adaptation requires --teacher_checkpoint "
                    "or --init_checkpoint."
                )
            teacher_model = build_model_from_config(cfg).to(device)
            teacher_payload = load_checkpoint(teacher_path, map_location=device)
            teacher_model.load_state_dict(
                extract_model_state(teacher_payload), strict=True
            )
            teacher_model.eval()
            for parameter in teacher_model.parameters():
                parameter.requires_grad = False
            rank_print(
                context,
                f"Frozen teacher loaded from {teacher_path}; "
                f"teacher_types={sorted(teacher_types)}; "
                f"lambda_output={teacher_output_weight}; "
                f"lambda_gate={teacher_gate_weight}",
            )

        trainable_prefixes = [
            str(value) for value in train_cfg["trainable_modules"]
        ]
        set_trainable_prefixes(raw_model, trainable_prefixes)
        summary = trainable_parameter_summary(raw_model)
        rank_print(context, "Trainable prefixes:", trainable_prefixes)
        rank_print(context, "Parameter summary:", summary)
        if summary["trainable"] <= 0:
            raise RuntimeError("No trainable parameters.")

        train_dataset = PairedImageDataset(
            data_cfg["train_turb_dir"],
            data_cfg["train_gt_dir"],
            patch_size=int(data_cfg.get("patch_size", 256)),
            augment=bool(data_cfg.get("augment", True)),
        )
        target_ratio = {
            key: float(value)
            for key, value in train_cfg["type_sampling_ratio"].items()
        }
        global_samples_per_epoch = int(
            train_cfg.get("samples_per_epoch", len(train_dataset))
        )
        base_sampler = weighted_sampler(
            train_dataset, target_ratio, global_samples_per_epoch
        )
        train_sampler = distributed_sampler_from_weighted(
            base_sampler,
            global_num_samples=global_samples_per_epoch,
            context=context,
            seed=args.seed,
        )
        workers = int(train_cfg.get("num_workers", 8))
        per_gpu_batch = int(train_cfg.get("batch_size", 4))
        train_loader = build_loader(
            train_dataset,
            batch_size=per_gpu_batch,
            num_workers=workers,
            sampler=train_sampler,
            drop_last=True,
            prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        )

        # Full deterministic validation on rank 0 only.
        primary_dataset = PairedImageDataset(
            data_cfg["val_turb_dir"],
            data_cfg["val_gt_dir"],
            patch_size=0,
            augment=False,
        )
        primary_eval_sampler = DistributedEvalSampler(
            len(primary_dataset), context.rank, context.world_size
        )
        primary_loader = build_loader(
            primary_dataset, 1, 0, sampler=primary_eval_sampler
        )
        extra_loaders: Dict[str, DataLoader] = {}
        for name, item in data_cfg.get("extra_val_sets", {}).items():
            dataset = PairedImageDataset(
                item["turb_dir"], item["gt_dir"], patch_size=0, augment=False
            )
            eval_sampler = DistributedEvalSampler(
                len(dataset), context.rank, context.world_size
            )
            extra_loaders[name] = build_loader(
                dataset, 1, 0, sampler=eval_sampler
            )

        # Wrap only after checkpoint loading and trainable-prefix selection.
        model = wrap_ddp(raw_model, context, find_unused_parameters=False)
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(train_cfg.get("lr", 1e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        epochs = int(train_cfg.get("epochs", 3))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )
        if resume_payload is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])

        use_amp, amp_dtype = resolve_precision(
            device, str(train_cfg.get("precision", "bf16"))
        )
        accumulation = int(train_cfg.get("accum_iter", 1))
        grad_clip = float(train_cfg.get("grad_clip", 1.0))
        max_train_batches = int(train_cfg.get("max_train_batches", 0))
        rec_loss_fn = CharbonnierLoss()

        global_effective_batch = (
            per_gpu_batch * context.world_size * accumulation
        )
        expected_steps = int(
            global_samples_per_epoch / max(global_effective_batch, 1)
        )
        rank_print(context, "=" * 72)
        rank_print(
            context,
            f"Restoration training: world_size={context.world_size}, backend={context.backend}, "
            f"per_gpu_batch={per_gpu_batch}, accum={accumulation}, "
            f"global_effective_batch={global_effective_batch}, "
            f"global_samples_per_epoch={global_samples_per_epoch}, "
            f"expected_optimizer_steps≈{expected_steps}",
        )
        rank_print(context, "=" * 72)

        handle = None
        writer = None
        fields = [
            "epoch",
            "train_loss",
            "rec",
            "grad",
            "freq",
            "range",
            "identity",
            "gate_identity",
            "teacher_output",
            "teacher_gate",
            "lr",
            "primary_psnr",
            "primary_ssim",
            "primary_gate",
            "primary_severity",
            "primary_drive",
        ]
        for name in extra_loaders:
            fields.extend(
                [
                    f"{name}_psnr",
                    f"{name}_ssim",
                    f"{name}_gate",
                    f"{name}_severity",
                    f"{name}_drive",
                ]
            )
        if context.is_main:
            history_path = output_dir / "history.csv"
            write_header = not history_path.exists() or start_epoch == 1
            handle = open(
                history_path, "a", newline="", encoding="utf-8-sig"
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()

        for epoch in range(start_epoch, epochs + 1):
            train_sampler.set_epoch(epoch)
            model.train()
            raw = unwrap_model(model)
            # The image-space severity encoder is calibrated separately and
            # remains frozen in connection/adaptation stages.
            if not any(
                parameter.requires_grad
                for parameter in raw.global_severity.parameters()
            ):
                raw.global_severity.eval()

            optimizer.zero_grad(set_to_none=True)
            running = {
                key: 0.0
                for key in [
                    "loss",
                    "rec",
                    "grad",
                    "freq",
                    "range",
                    "identity",
                    "gate_identity",
                    "teacher_output",
                    "teacher_gate",
                ]
            }
            batches = 0
            optim_steps = 0
            type_counts = Counter()
            start_time = time.time()
            progress = tqdm(
                train_loader,
                desc=f"Restoration epoch {epoch}/{epochs}",
                disable=not context.is_main,
            )
            loader_length = len(train_loader)
            for batch_index, batch in enumerate(progress, start=1):
                if max_train_batches > 0 and batch_index > max_train_batches:
                    break
                turb = batch["turb"].to(device, non_blocking=True)
                gt = batch["gt"].to(device, non_blocking=True)
                types = [str(value) for value in batch["degradation_type"]]
                type_counts.update(types)
                mask = identity_mask(batch["is_identity"], device)
                is_last = batch_index == loader_length or (
                    max_train_batches > 0 and batch_index == max_train_batches
                )
                should_step = batch_index % accumulation == 0 or is_last
                sync_context = (
                    nullcontext()
                    if should_step or not context.distributed
                    else model.no_sync()  # type: ignore[attr-defined]
                )
                with sync_context:
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        outputs = model(
                            turb,
                            return_gate=True,
                            return_severity=True,
                            return_ordinal=True,
                        )
                        pred, gate, severity, _, _, drive = outputs
                    pred32 = pred.float()
                    gt32 = gt.float()
                    turb32 = turb.float()
                    rec = rec_loss_fn(pred32, gt32)
                    grad = gradient_loss(pred32, gt32)
                    freq = local_log_amplitude_frequency_loss(
                        pred32,
                        gt32,
                        window_size=int(loss_cfg.get("frequency_window", 16)),
                    )
                    rng = range_loss(pred32)
                    if mask.any():
                        ident = F.l1_loss(pred32[mask], turb32[mask])
                        gate_ident = gate.float()[mask].mean()
                    else:
                        ident = pred32.new_zeros(())
                        gate_ident = pred32.new_zeros(())

                    preserve_mask = torch.tensor(
                        [sample_type in teacher_types for sample_type in types],
                        device=device,
                        dtype=torch.bool,
                    )
                    if teacher_model is not None and preserve_mask.any():
                        with torch.no_grad():
                            with torch.amp.autocast(
                                device_type=device.type,
                                dtype=amp_dtype,
                                enabled=use_amp,
                            ):
                                teacher_outputs = teacher_model(
                                    turb,
                                    return_gate=True,
                                    return_severity=True,
                                    return_ordinal=True,
                                )
                                teacher_pred, teacher_gate, *_ = teacher_outputs
                        teacher_out_loss = F.l1_loss(
                            pred32[preserve_mask],
                            teacher_pred.float()[preserve_mask],
                        )
                        teacher_gate_loss = F.l1_loss(
                            gate.float()[preserve_mask],
                            teacher_gate.float()[preserve_mask],
                        )
                    else:
                        teacher_out_loss = pred32.new_zeros(())
                        teacher_gate_loss = pred32.new_zeros(())

                    loss = (
                        float(loss_cfg.get("lambda_rec", 1.0)) * rec
                        + float(loss_cfg.get("lambda_grad", 0.02)) * grad
                        + float(loss_cfg.get("lambda_freq", 0.005)) * freq
                        + float(loss_cfg.get("lambda_range", 0.001)) * rng
                        + float(loss_cfg.get("lambda_identity", 0.10)) * ident
                        + float(loss_cfg.get("lambda_gate_identity", 0.005))
                        * gate_ident
                        + teacher_output_weight * teacher_out_loss
                        + teacher_gate_weight * teacher_gate_loss
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite loss at epoch={epoch}, batch={batch_index}: {loss.detach()}"
                        )
                    (loss / accumulation).backward()

                if should_step:
                    bad_gradient = None
                    for parameter_name, parameter in model.named_parameters():
                        if parameter.requires_grad and parameter.grad is not None:
                            if not torch.isfinite(parameter.grad).all():
                                bad_gradient = parameter_name
                                break
                    if bad_gradient is not None:
                        raise RuntimeError(
                            "Non-finite gradient in "
                            + bad_gradient
                            + f" at epoch={epoch}, batch={batch_index}."
                        )
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optim_steps += 1

                batches += 1
                for key, value in [
                    ("loss", loss),
                    ("rec", rec),
                    ("grad", grad),
                    ("freq", freq),
                    ("range", rng),
                    ("identity", ident),
                    ("gate_identity", gate_ident),
                    ("teacher_output", teacher_out_loss),
                    ("teacher_gate", teacher_gate_loss),
                ]:
                    running[key] += float(value.detach())
                if context.is_main and batch_index % args.log_interval == 0:
                    progress.set_postfix(
                        loss=f"{running['loss'] / batches:.4f}",
                        gate=f"{float(gate.detach().mean()):.3f}",
                        severity=f"{float(severity.detach().mean()):.3f}",
                        drive=f"{float(drive.detach().mean()):.3f}",
                    )

            scheduler.step()
            global_running = reduce_scalar_dict(
                {
                    **running,
                    "batches": float(batches),
                    "optim_steps": float(optim_steps),
                    **{
                        f"type_{name}": float(type_counts.get(name, 0))
                        for name in DEGRADATION_ORDER
                    },
                },
                context,
            )
            global_batches = max(global_running.pop("batches"), 1.0)
            global_optim_steps = int(round(global_running.pop("optim_steps") / context.world_size))
            global_type_counts = {
                name: int(global_running.pop(f"type_{name}"))
                for name in DEGRADATION_ORDER
            }
            averages = {
                key: value / global_batches for key, value in global_running.items()
            }

            barrier()
            raw = unwrap_model(model)
            primary = validate(
                raw,
                primary_loader,
                device,
                use_amp,
                amp_dtype,
                context,
                save_dir=output_dir / "val_images" / f"epoch_{epoch:03d}",
                max_save=int(train_cfg.get("save_val_images", 4)),
            )
            extras = {
                name: validate(raw, loader, device, use_amp, amp_dtype, context)
                for name, loader in extra_loaders.items()
            }
            previous_best = best_primary_psnr
            best_primary_psnr = max(best_primary_psnr, primary["psnr"])

            if context.is_main:
                print(
                    f"[Train] epoch={epoch} loss={averages['loss']:.6f} "
                    f"optim_steps={global_optim_steps} types={global_type_counts} "
                    f"time={(time.time() - start_time) / 60:.1f}min"
                )
                print("[Val:primary]", primary)
                for name, values in extras.items():
                    print(f"[Val:{name}]", values)

                row = {
                    "epoch": epoch,
                    "train_loss": averages["loss"],
                    "rec": averages["rec"],
                    "grad": averages["grad"],
                    "freq": averages["freq"],
                    "range": averages["range"],
                    "identity": averages["identity"],
                    "gate_identity": averages["gate_identity"],
                    "teacher_output": averages["teacher_output"],
                    "teacher_gate": averages["teacher_gate"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "primary_psnr": primary["psnr"],
                    "primary_ssim": primary["ssim"],
                    "primary_gate": primary["gate_mean"],
                    "primary_severity": primary["severity_mean"],
                    "primary_drive": primary["spectral_drive_mean"],
                }
                for name, values in extras.items():
                    row.update(
                        {
                            f"{name}_psnr": values["psnr"],
                            f"{name}_ssim": values["ssim"],
                            f"{name}_gate": values["gate_mean"],
                            f"{name}_severity": values["severity_mean"],
                            f"{name}_drive": values["spectral_drive_mean"],
                        }
                    )
                assert writer is not None and handle is not None
                writer.writerow(row)
                handle.flush()

                save_checkpoint(
                    output_dir / "checkpoints" / f"epoch_{epoch:03d}.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_primary_psnr,
                    cfg,
                )
                save_checkpoint(
                    output_dir / "latest.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_primary_psnr,
                    cfg,
                )
                if primary["psnr"] > previous_best:
                    save_checkpoint(
                        output_dir / "best_primary_psnr.pth",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_primary_psnr,
                        cfg,
                    )
            barrier()

        if context.is_main:
            assert handle is not None
            handle.close()
            print("Training finished.")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
