#!/usr/bin/env python3
"""Train the image-space Siamese ordinal severity encoder.

Supports ordinary single-GPU execution and single-node multi-GPU DDP launched
with ``torchrun``. In DDP mode, ``severity_train.samples_per_epoch`` is a
global sample budget and ``batch_size`` is per GPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.severity_pairs import SeverityImageDataset, SeverityPairImageDataset
from models.severity_siamese import build_severity_encoder_from_config
from utils.common import load_checkpoint, load_yaml, resolve_precision, save_json
from utils.distributed import (
    barrier,
    broadcast_object,
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
from utils.severity_metrics import (
    evaluate_severity_predictions,
    ordinal_targets,
    predict_image_dataset,
    save_severity_report,
    severity_validation_score,
)


def save_checkpoint(
    path: Path,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_score: Tuple[float, ...],
    cfg: dict,
    summary: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_severity": unwrap_model(encoder).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_score": tuple(float(value) for value in best_score),
            "config": cfg,
            "summary": dict(summary),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    # Accepted for compatibility with current and older torchrun launchers.
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    args = parser.parse_args()

    context = init_distributed(args.device)
    try:
        cfg = load_yaml(args.config)
        set_rank_seed(args.seed, context)
        device = context.device
        data_cfg = cfg["severity_data"]
        train_cfg = cfg["severity_train"]
        loss_cfg = cfg["severity_loss"]
        output_dir = Path(cfg["output_dir"])
        if context.is_main:
            (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (output_dir / "validation").mkdir(parents=True, exist_ok=True)
        barrier()

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
            torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
            torch.backends.cudnn.allow_tf32 = bool(train_cfg.get("allow_tf32", True))

        anchor_targets = {
            key: float(value) for key, value in loss_cfg["anchor_targets"].items()
        }
        pair_loss_weights = {
            key: float(value)
            for key, value in loss_cfg.get("pair_loss_weights", {}).items()
        }
        sampling_multipliers = {
            key: float(value)
            for key, value in train_cfg.get("pair_sampling_multipliers", {}).items()
        }

        train_dataset = SeverityPairImageDataset(
            turb_dir=data_cfg["train_turb_dir"],
            pair_csv=data_cfg["train_pair_csv"],
            anchor_targets=anchor_targets,
            base_margin=float(loss_cfg.get("base_margin", 0.08)),
            augment=bool(data_cfg.get("augment", True)),
            shared_photometric_prob=float(
                data_cfg.get("shared_photometric_prob", 0.5)
            ),
            pair_loss_weights=pair_loss_weights,
        )
        global_samples_per_epoch = int(
            train_cfg.get("samples_per_epoch", len(train_dataset))
        )
        base_sampler = train_dataset.balanced_sampler(
            samples_per_epoch=global_samples_per_epoch,
            sampling_multipliers=sampling_multipliers,
        )
        train_sampler = distributed_sampler_from_weighted(
            base_sampler,
            global_num_samples=global_samples_per_epoch,
            context=context,
            seed=args.seed,
        )
        workers = int(train_cfg.get("num_workers", 8))
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg.get("batch_size", 16)),  # per GPU
            sampler=train_sampler,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            persistent_workers=workers > 0,
            prefetch_factor=int(train_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
            worker_init_fn=seed_worker,
        )

        # Validation is deliberately performed only on rank 0. The set is
        # small, and this keeps pair aggregation deterministic and simple.
        val_dataset = SeverityImageDataset(
            data_cfg["val_turb_dir"], data_cfg["val_pair_csv"]
        )
        val_workers = int(train_cfg.get("eval_num_workers", 4))
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("eval_batch_size", 128)),
            shuffle=False,
            num_workers=val_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=val_workers > 0,
            prefetch_factor=int(train_cfg.get("eval_prefetch_factor", 2))
            if val_workers > 0
            else None,
            worker_init_fn=seed_worker,
        )

        raw_encoder = build_severity_encoder_from_config(cfg).to(device)
        start_epoch = 1
        best_score: Tuple[float, ...] = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            -999.0,
            0.0,
        )
        resume_payload = None
        if args.resume:
            resume_payload = load_checkpoint(args.resume, map_location=device)
            raw_encoder.load_state_dict(
                resume_payload["global_severity"], strict=True
            )
            start_epoch = int(resume_payload["epoch"]) + 1
            best_score = tuple(
                float(value)
                for value in resume_payload.get("best_score", best_score)
            )
            rank_print(
                context,
                f"Resumed model from {args.resume}; next epoch={start_epoch}",
            )

        encoder = wrap_ddp(raw_encoder, context, find_unused_parameters=True)
        optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=float(train_cfg.get("lr", 2e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        epochs = int(train_cfg.get("epochs", 30))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(train_cfg.get("min_lr", 1e-5)),
        )
        if resume_payload is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])

        use_amp, amp_dtype = resolve_precision(
            device, str(train_cfg.get("precision", "bf16"))
        )
        ordinal_weight = float(loss_cfg.get("ordinal_weight", 0.5))
        class_weight = float(loss_cfg.get("class_weight", 0.5))
        rank_weight = float(loss_cfg.get("rank_weight", 4.0))
        anchor_weight = float(loss_cfg.get("anchor_weight", 0.5))
        gap_weight = float(loss_cfg.get("gap_weight", 1.0))
        grad_clip = float(train_cfg.get("grad_clip", 1.0))
        patience = int(train_cfg.get("early_stop_patience", 10))
        epochs_without_improvement = 0

        per_gpu_batch = int(train_cfg.get("batch_size", 16))
        global_batch = per_gpu_batch * context.world_size
        rank_print(context, "=" * 72)
        rank_print(
            context,
            f"Severity training: world_size={context.world_size}, backend={context.backend}, "
            f"per_gpu_batch={per_gpu_batch} pairs, global_batch={global_batch} pairs, "
            f"global_samples_per_epoch={global_samples_per_epoch}, precision={train_cfg.get('precision', 'bf16')}",
        )
        rank_print(context, "=" * 72)

        history_handle = None
        writer = None
        if context.is_main:
            history_path = output_dir / "history.csv"
            write_header = not history_path.exists() or start_epoch == 1
            history_handle = open(
                history_path, "a", newline="", encoding="utf-8-sig"
            )
            history_fields = [
                "epoch",
                "loss",
                "ordinal_loss",
                "class_loss",
                "rank_loss",
                "anchor_loss",
                "gap_loss",
                "lr",
                "overall_pair_accuracy",
                "degraded_pair_accuracy",
                "mild_medium_accuracy",
                "medium_strong_accuracy",
                "spearman_degraded",
                "severity_mae",
                "ordinal_level_accuracy",
                "class_level_accuracy",
                "strict_type_mean_order",
                "passes_validation",
                "identity_mean",
                "mild_mean",
                "medium_mean",
                "strong_mean",
            ]
            writer = csv.DictWriter(history_handle, fieldnames=history_fields)
            if write_header:
                writer.writeheader()

        for epoch in range(start_epoch, epochs + 1):
            train_sampler.set_epoch(epoch)
            encoder.train()
            running: Dict[str, float] = {
                "loss": 0.0,
                "ordinal": 0.0,
                "class": 0.0,
                "rank": 0.0,
                "anchor": 0.0,
                "gap": 0.0,
            }
            batches = 0
            progress = tqdm(
                train_loader,
                desc=f"Siamese severity {epoch}/{epochs}",
                disable=not context.is_main,
            )
            for batch in progress:
                low_image = batch["low_image"].to(device, non_blocking=True)
                high_image = batch["high_image"].to(device, non_blocking=True)
                low_level = batch["low_level"].long().to(device, non_blocking=True)
                high_level = batch["high_level"].long().to(device, non_blocking=True)
                low_target = batch["low_target"].float().to(device, non_blocking=True)
                high_target = batch["high_target"].float().to(device, non_blocking=True)
                margin = batch["margin"].float().to(device, non_blocking=True)
                pair_scale = batch["loss_weight"].float().to(
                    device, non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)
                # One DDP forward per optimizer step is both faster and safer
                # than invoking the wrapped module separately for low/high.
                images = torch.cat([low_image, high_image], dim=0)
                pair_batch = low_image.shape[0]
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    logits, severity, _, class_logits, _ = encoder(
                        images, return_aux=True
                    )
                    low_logits, high_logits = logits.split(pair_batch, dim=0)
                    low_severity, high_severity = severity.split(pair_batch, dim=0)
                    low_class_logits, high_class_logits = class_logits.split(
                        pair_batch, dim=0
                    )

                    ordinal_loss = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            low_logits, ordinal_targets(low_level)
                        )
                        + F.binary_cross_entropy_with_logits(
                            high_logits, ordinal_targets(high_level)
                        )
                    )
                    class_loss = 0.5 * (
                        F.cross_entropy(low_class_logits, low_level)
                        + F.cross_entropy(high_class_logits, high_level)
                    )
                    low_s = low_severity[:, 0]
                    high_s = high_severity[:, 0]
                    rank_per_item = F.relu(margin - high_s + low_s)
                    rank_loss = (
                        (rank_per_item * pair_scale).sum()
                        / pair_scale.sum().clamp_min(1e-6)
                    )
                    anchor_loss = 0.5 * (
                        F.smooth_l1_loss(low_s, low_target)
                        + F.smooth_l1_loss(high_s, high_target)
                    )
                    target_gap = high_target - low_target
                    predicted_gap = high_s - low_s
                    gap_per_item = F.smooth_l1_loss(
                        predicted_gap, target_gap, reduction="none"
                    )
                    gap_loss = (
                        (gap_per_item * pair_scale).sum()
                        / pair_scale.sum().clamp_min(1e-6)
                    )
                    loss = (
                        ordinal_weight * ordinal_loss
                        + class_weight * class_loss
                        + rank_weight * rank_loss
                        + anchor_weight * anchor_loss
                        + gap_weight * gap_loss
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite severity loss: {loss.detach()}")
                loss.backward()
                for name, parameter in encoder.named_parameters():
                    if (
                        parameter.grad is not None
                        and not torch.isfinite(parameter.grad).all()
                    ):
                        raise RuntimeError(
                            f"Non-finite severity gradient in {name}"
                        )
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                optimizer.step()

                batches += 1
                values = {
                    "loss": loss,
                    "ordinal": ordinal_loss,
                    "class": class_loss,
                    "rank": rank_loss,
                    "anchor": anchor_loss,
                    "gap": gap_loss,
                }
                for key, value in values.items():
                    running[key] += float(value.detach())
                if context.is_main:
                    progress.set_postfix(
                        loss=f"{running['loss'] / batches:.4f}",
                        rank=f"{running['rank'] / batches:.4f}",
                    )

            scheduler.step()
            global_running = reduce_scalar_dict(
                {**running, "batches": float(batches)}, context
            )
            global_batches = max(global_running.pop("batches"), 1.0)
            averages = {
                key: value / global_batches for key, value in global_running.items()
            }

            barrier()
            payload = None
            if context.is_main:
                raw = unwrap_model(encoder)
                names, severity_np, probabilities, class_logits_np = (
                    predict_image_dataset(
                        raw,
                        val_loader,
                        device=device,
                        precision=str(train_cfg.get("precision", "bf16")),
                    )
                )
                summary, pair_df, sample_df = evaluate_severity_predictions(
                    names=names,
                    severity=severity_np,
                    probabilities=probabilities,
                    class_logits=class_logits_np,
                    records=val_dataset.records,
                    anchor_targets=anchor_targets,
                    margin_base=float(loss_cfg.get("base_margin", 0.08)),
                )
                report_dir = output_dir / "validation" / f"epoch_{epoch:03d}"
                save_severity_report(report_dir, summary, pair_df, sample_df)
                score = severity_validation_score(summary)
                per_pair = summary.get("per_pair", {})
                mild_medium = (per_pair.get("mild<medium", {}) or {}).get(
                    "accuracy"
                )
                medium_strong = (per_pair.get("medium<strong", {}) or {}).get(
                    "accuracy"
                )
                means = summary["type_means"]
                row = {
                    "epoch": epoch,
                    "loss": averages["loss"],
                    "ordinal_loss": averages["ordinal"],
                    "class_loss": averages["class"],
                    "rank_loss": averages["rank"],
                    "anchor_loss": averages["anchor"],
                    "gap_loss": averages["gap"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "overall_pair_accuracy": summary["overall_pair_accuracy"],
                    "degraded_pair_accuracy": summary[
                        "degraded_pair_accuracy"
                    ],
                    "mild_medium_accuracy": mild_medium,
                    "medium_strong_accuracy": medium_strong,
                    "spearman_degraded": summary["spearman_degraded"],
                    "severity_mae": summary["severity_mae"],
                    "ordinal_level_accuracy": summary[
                        "ordinal_level_accuracy"
                    ],
                    "class_level_accuracy": summary["class_level_accuracy"],
                    "strict_type_mean_order": summary[
                        "strict_type_mean_order"
                    ],
                    "passes_validation": summary["passes_validation"],
                    "identity_mean": means["identity"],
                    "mild_mean": means["mild"],
                    "medium_mean": means["medium"],
                    "strong_mean": means["strong"],
                }
                assert writer is not None and history_handle is not None
                writer.writerow(row)
                history_handle.flush()
                print("Validation summary:", summary)

                candidate_best = max(best_score, score)
                save_checkpoint(
                    output_dir / "checkpoints" / f"epoch_{epoch:03d}.pth",
                    encoder,
                    optimizer,
                    scheduler,
                    epoch,
                    candidate_best,
                    cfg,
                    summary,
                )
                save_checkpoint(
                    output_dir / "latest.pth",
                    encoder,
                    optimizer,
                    scheduler,
                    epoch,
                    candidate_best,
                    cfg,
                    summary,
                )
                if score > best_score:
                    best_score = score
                    epochs_without_improvement = 0
                    save_checkpoint(
                        output_dir / "best_severity.pth",
                        encoder,
                        optimizer,
                        scheduler,
                        epoch,
                        best_score,
                        cfg,
                        summary,
                    )
                    print(
                        f"Saved best_severity.pth at epoch {epoch}; score={score}"
                    )
                else:
                    epochs_without_improvement += 1

                should_stop = bool(
                    bool(summary.get("passes_validation", False))
                    and bool(train_cfg.get("stop_when_passed", False))
                ) or bool(
                    patience > 0 and epochs_without_improvement >= patience
                )
                payload = {
                    "best_score": best_score,
                    "epochs_without_improvement": epochs_without_improvement,
                    "should_stop": should_stop,
                }

            payload = broadcast_object(payload, context)
            best_score = tuple(float(value) for value in payload["best_score"])
            epochs_without_improvement = int(
                payload["epochs_without_improvement"]
            )
            if bool(payload["should_stop"]):
                rank_print(context, "Distributed early stop requested.")
                break
            barrier()

        if context.is_main:
            assert history_handle is not None
            history_handle.close()
            save_json(
                {"best_score": best_score}, output_dir / "training_summary.json"
            )
            print(f"Training finished. Best score={best_score}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
